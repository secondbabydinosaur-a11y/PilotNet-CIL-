"""보호한 시험·검증 회랑 밖에서 정상 차로의 긴 연결 수집 경로를 만든다.

원본 경로·모델·센서 설정은 수정하지 않는다. 가능한 새 링크를 먼저 사용하되
회랑 재사용 길이도 기록한다. 수집 시간 자체를 고유 도로 길이로 해석하지 않는다.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, pose_heading, route_from_xyz, sha256, write_route
from build_east_outer_data_experiment import radius_screen


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), prefix="." + path.name, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--snippets", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    source_plan = BASE / "routes/experiments/east_outer_data_v1/plan.json"
    frozen = json.loads(source_plan.read_text())
    plan = copy.deepcopy(frozen)
    map_path = Path(plan["source"]) / "link_set.json"
    if sha256(map_path) != plan["source_hashes"]["link_set.json"]:
        raise ValueError("고정한 MGeo 지도와 다름")
    links = {link["idx"]: link for link in json.loads(map_path.read_text()) if not link["lazy_init"]}
    native = {key: route_from_xyz(link["points"]) for key, link in links.items()}
    destination = BASE / "routes/experiments/east_outer_90min"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE / "routes/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protected = {key: np.asarray(item["guard_reference_xyz"]) for key, item in plan["test_reservations"].items()}
    for key, info in manifest["routes"].items():
        if info.get("split") == "test":
            protected[key] = np.loadtxt(BASE / info["route_csv"], delimiter=",", skiprows=1)[:, 1:4]
    for key in ("U1", "P2Q", "V1", "V2"):
        if key in plan["test_reservations"] and key not in plan["routes"]: continue
        if key in plan["routes"]:
            protected[key] = np.loadtxt(BASE / plan["routes"][key]["route_csv"], delimiter=",", skiprows=1)[:, 1:4]
    for key, xyz in protected.items():
        plan["test_reservations"][key] = {"guard_reference_xyz": xyz.tolist(), "buffer_m": 20.0}
    trees = {key: cKDTree(xyz[:, :2]) for key, xyz in protected.items()}
    actual_val = []
    completion_path = BASE / "data/collection_completion.json"
    for item in json.loads(completion_path.read_text())["accepted"]:
        if item["split"] != "validation": continue
        with (BASE / item["session"] / "vehicle.csv").open() as stream:
            actual_val.extend([[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)])
    trees["accepted_validation_positions"] = cKDTree(np.asarray(actual_val))
    combined_tree = cKDTree(np.vstack([np.asarray(tree.data) for tree in trees.values()]))

    allowed, reasons = {}, Counter()
    for key, route in native.items():
        link = links[key]
        if route[-1, 0] < .5: continue
        if not link["lane_mark_left"] and not link["lane_mark_right"]:
            reasons["map_has_no_lane_boundary"] += 1
            continue
        if float(combined_tree.query(route[:, 1:3])[0].min()) <= 20.8:
            reasons["protected_corridor"] += 1
            continue
        if radius_screen(route) < 5.87:
            reasons["minimum_radius"] += 1
            continue
        allowed[key] = route
    outgoing = defaultdict(list)
    for key in allowed: outgoing[links[key]["from_node_idx"]].append(key)
    compatible = {}
    for key, route in allowed.items():
        options = []
        for nxt in outgoing[links[key]["to_node_idx"]]:
            q = allowed[nxt]
            if np.linalg.norm(route[-1, 1:4] - q[0, 1:4]) > .01: continue
            tail = interval(route, max(0.0, route[-1, 0] - 8.0), route[-1, 0])
            head = interval(q, 0.0, min(8.0, q[-1, 0]))
            joined = route_from_xyz(np.vstack((tail[:, 1:4], head[:, 1:4])))
            if radius_screen(joined) >= 5.87: options.append(nxt)
        compatible[key] = options
    starts = [key for key in allowed if str(links[key]["link_type"]) != "1" and allowed[key][-1, 0] >= 25.0]
    used, signatures, records, all_selected = Counter(), set(), [], set()
    existing_path = destination / "plan.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing["experiment"]["source_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("시험 보호 목록이 바뀌었으므로 기존 수집 경로를 먼저 재검토해야 함")
        for record in existing["experiment"]["reports"]:
            name = record["route"]
            info = existing["routes"][name]
            if sha256(BASE / info["route_csv"]) != info["route_sha256"]:
                raise ValueError("기존 경로 파일이 변경됨: " + name)
            plan["routes"][name] = info
            records.append(record)
            used.update(info["source_links"])
            signatures.add(tuple(info["source_links"]))
            all_selected.update(info["source_links"])
    config = json.loads((BASE / "config.json").read_text())
    config["paths"]["route_plan"] = str((destination / "plan.json").relative_to(BASE))
    config["paths"]["raw_data"] = "runs/east_outer_90min/raw"

    def publish():
        plan["status"] = "additional_long_collection_routes_prepared_not_collected"
        plan["experiment"] = {"id": "east_outer_90min", "collection_route_ids": [r["route"] for r in records],
                              "frozen_previous_plan_sha256": sha256(source_plan), "source_manifest_sha256": sha256(manifest_path),
                              "reports": records, "allowed_native_links": len(allowed),
                              "allowed_lane_length_m": float(sum(route[-1, 0] for route in allowed.values())),
                              "selected_unique_link_length_m": float(sum(allowed[key][-1, 0] for key in all_selected)),
                              "selected_total_route_length_m": float(sum(r["length_m"] for r in records)),
                              "old_routes_preserved_for_provenance_only": True, "collect_only_new_route_ids": True,
                              "screening_exclusions": dict(reasons),
                              "notes": ["과거 학습 도로가 포함된 COMPETITION/REVERSE 시연은 시험 보호 목록으로 새로 추가하지 않는다.",
                                        "현재 split=test인 모든 EAST 확장·중간교차로 시험과 원래U1/U2/J/P2Q,검증V1/V2를20m이상보호한다.",
                                        "길이는 차로별 길이이며 새 도로의 고유 길이나 수집 시간과 동일하지 않다.",
                                        "각 경로 안에서는 링크를 반복하지 않는다. 다른 경로와 공유하는 연결로 길이는 별도로 기록한다.",
                                        "목표90분의 부족분은 정상/복구 조건을 구분해 수집해야 하며 같은 길을 전부 새 장소로 표시하지 않는다.",
                                        "지도 경계·최소반경 검사 뒤 실제 영상·센서·추종 품질도 수집 후 별도로 검사해야 한다."]}
        atomic_json(destination / "plan.json", plan)
        atomic_json(destination / "config.json", config)
        atomic_json(destination / "route_summary.json", records)

    existing_long = sum(record["route"].startswith("LONG_") for record in records)
    for number in range(existing_long + 1, args.routes + 1):
        candidates = []
        for trial in range(800):
            first = rng.choice(starts)
            path, length = [first], float(allowed[first][-1, 0])
            for _ in range(160):
                choices = [key for key in compatible[path[-1]] if key not in path]
                if not choices or length >= 1500.0: break
                weights = [math.sqrt(allowed[key][-1, 0] + 1.0) / (1.0 + used[key] * 2.0) for key in choices]
                nxt = rng.choices(choices, weights=weights, k=1)[0]
                path.append(nxt)
                length += float(allowed[nxt][-1, 0])
            while path and (str(links[path[-1]]["link_type"]) == "1" or allowed[path[-1]][-1, 0] < 20):
                path.pop()
            if not path or tuple(path) in signatures: continue
            length = sum(float(allowed[key][-1, 0]) for key in path)
            if length < 300.0: continue
            turns = sum(str(links[key]["link_type"]) == "1" for key in path)
            score = sum(float(allowed[key][-1, 0]) / (1.0 + used[key] * 5.0) for key in path) + 10.0 * turns
            candidates.append((score, path))
        if not candidates: break
        generated = False
        for score, path in sorted(candidates, key=lambda item: -item[0])[:60]:
            route = route_from_xyz(np.vstack([allowed[key][:, 1:4] for key in path]))
            radius = radius_screen(route)
            if radius < 5.87: continue
            sample_s = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
            xy = np.column_stack([np.interp(sample_s, route[:, 0], route[:, k]) for k in (1, 2)])
            checks = {key: {"nearest_sample_m": float(tree.query(xy)[0].min()), "required_m": 20.0} for key, tree in trees.items()}
            if min(item["nearest_sample_m"] for item in checks.values()) <= 20.5: continue
            nav = []
            for key in path:
                if str(links[key]["link_type"]) != "1": continue
                item = navigation(route, links[key], "")
                turn = item["measured_turn_deg"]
                if turn is None: continue
                item["command"] = "left" if turn > 45 else "right" if turn < -45 else "straight"
                nav.append(item)
            if not nav: continue
            name = "LONG_%02d" % number
            output = destination / (name + ".csv")
            write_route(output, route)
            new_length = sum(float(allowed[key][-1, 0]) for key in path if not used[key])
            plan["routes"][name] = {"name_ko": "장거리 추가 수집 " + name, "split": "train", "route_csv": str(output.relative_to(BASE)),
                                   "route_sha256": sha256(output), "length_m": float(route[-1, 0]),
                                   "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
                                   "initial_yaw_deg_map": pose_heading(route), "source_links": path, "navigation": nav,
                                   "minimum_radius_from_3m_chords_m": radius, "spatial_separation_checks": checks,
                                   "experiment_id": "east_outer_90min", "collection_note": "정상 지도 링크만 연결했다. 시험 보호·센서·실제 영상·추종 검사를 통과한 세션만 학습에 사용한다."}
            counts = Counter(item["command"] for item in nav)
            record = {"route": name, "length_m": float(route[-1, 0]), "junction_commands": dict(counts),
                      "new_native_link_length_vs_preceding_routes_m": new_length,
                      "shared_native_link_length_m": max(0.0, float(route[-1, 0]) - new_length),
                      "minimum_radius_m": radius, "minimum_protected_distance_m": min(item["nearest_sample_m"] for item in checks.values())}
            records.append(record)
            signatures.add(tuple(path))
            used.update(path)
            all_selected.update(path)
            publish()
            print(name, round(route[-1, 0]), dict(counts), "새연결길이", round(new_length), flush=True)
            generated = True
            break
        if not generated: break
    # 긴 경로에 아직 포함되지 않은 교차로 기하를 짧은 별도 수집으로 보완한다.
    incoming = defaultdict(list)
    for key in allowed: incoming[links[key]["to_node_idx"]].append(key)

    def flank(key, backwards):
        path = [key]
        result = allowed[key]
        for _ in range(4):
            if result[-1, 0] >= 30.0: break
            options = incoming[links[path[0]]["from_node_idx"]] if backwards else outgoing[links[path[-1]]["to_node_idx"]]
            options = [n for n in options if n not in path and str(links[n]["link_type"]) != "1"]
            if not options: break
            chosen = max(options, key=lambda n: allowed[n][-1, 0])
            path = [chosen] + path if backwards else path + [chosen]
            result = route_from_xyz(np.vstack([allowed[n][:, 1:4] for n in path]))
        limit = min(25.0 if backwards else 30.0, result[-1, 0])
        start, end = (result[-1, 0] - limit, result[-1, 0]) if backwards else (0.0, limit)
        return interval(result, start, end), path

    captured_junctions = {key for info in plan["routes"].values() if info.get("split") == "train" for key in info.get("source_links", []) if key in links and str(links[key]["link_type"]) == "1"}
    local_candidates = []
    for key in allowed:
        if str(links[key]["link_type"]) != "1" or key in captured_junctions: continue
        for pre in incoming[links[key]["from_node_idx"]]:
            if str(links[pre]["link_type"]) == "1": continue
            lead, lead_path = flank(pre, True)
            if lead[-1, 0] < 20.0: continue
            for post in outgoing[links[key]["to_node_idx"]]:
                if str(links[post]["link_type"]) == "1": continue
                tail, tail_path = flank(post, False)
                if tail[-1, 0] < 25.0: continue
                route = route_from_xyz(np.vstack((lead[:, 1:4], allowed[key][:, 1:4], tail[:, 1:4])))
                radius = radius_screen(route)
                if radius < 5.87: continue
                sample_s = np.r_[np.arange(0, route[-1, 0], .25), route[-1, 0]]
                xy = np.column_stack([np.interp(sample_s, route[:, 0], route[:, k]) for k in (1, 2)])
                checks = {name: {"nearest_sample_m": float(tree.query(xy)[0].min()), "required_m": 20.0} for name, tree in trees.items()}
                if min(item["nearest_sample_m"] for item in checks.values()) <= 20.5: continue
                item = navigation(route, links[key], "")
                if item["measured_turn_deg"] is None: continue
                angle = item["measured_turn_deg"]
                item["command"] = "left" if angle > 45 else "right" if angle < -45 else "straight"
                local_candidates.append((key, lead_path + [key] + tail_path, route, item, radius, checks))
    local_candidates.sort(key=lambda item: (item[3]["command"] == "straight", -min(v["nearest_sample_m"] for v in item[5].values())))
    existing_short = sum(record["route"].startswith("SHORT_") for record in records)
    added, chosen_junctions = 0, set()
    for key, path, route, nav, radius, checks in local_candidates:
        if added >= args.snippets or key in chosen_junctions: continue
        if key in captured_junctions: continue
        name = "SHORT_%02d" % (existing_short + added + 1)
        output = destination / (name + ".csv")
        if output.exists(): raise FileExistsError(output)
        write_route(output, route)
        plan["routes"][name] = {"name_ko": "추가 교차로 수집 " + name, "split": "train", "route_csv": str(output.relative_to(BASE)),
                               "route_sha256": sha256(output), "length_m": float(route[-1, 0]),
                               "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(), "initial_yaw_deg_map": pose_heading(route),
                               "source_links": path, "navigation": [nav], "minimum_radius_from_3m_chords_m": radius,
                               "spatial_separation_checks": checks, "experiment_id": "east_outer_90min",
                               "junction_not_in_previous_training_or_long_route_plan": True,
                               "collection_note": "기존 학습·보완·장거리 계획에서 아직 사용하지 않은 교차로의 기하를 별도로 관측한다. 지도 주변 영역을 별개의 새 장소로 과대계산하지 않는다."}
        record = {"route": name, "length_m": float(route[-1, 0]), "junction_commands": {nav["command"]: 1},
                  "new_junction_link_id": key, "new_native_link_length_vs_preceding_routes_m": sum(float(allowed[n][-1, 0]) for n in path if not used[n]),
                  "minimum_radius_m": radius, "minimum_protected_distance_m": min(v["nearest_sample_m"] for v in checks.values())}
        records.append(record)
        used.update(path)
        all_selected.update(path)
        chosen_junctions.add(key)
        added += 1
        publish()
        print(name, key, nav["command"], round(route[-1, 0], 1), flush=True)
    # 이미 확인한 교차로를 연속으로 연결해 세션 초기화·일반 직선의 비중을 낮춘다.
    # 새 장소가 늘어난 것은 아니므로 기존 경로와 독립된 기하로 계산하지 않는다.
    compact_choices = {
        "COMPACT_01": (["000315", "000318", "000308", "000083", "000304"], 25.0, 30.0),
        "COMPACT_02": (["000315", "000318", "000308", "000079", "000160"], 25.0, 30.0),
        "COMPACT_03": (["000213", "000111", "000225", "000116", "000365"], 20.0, 25.0),
        "COMPACT_04": (["000067", "000715", "000066", "000207", "000205", "000315", "000318", "000308", "000083", "000304"], 25.0, 30.0),
        "COMPACT_05": (["001083", "000598", "000323", "000308", "000083", "000304"], 20.0, 25.0),
        "COMPACT_06": (["000321", "000313", "000204", "000087", "000197", "000196", "000142", "000076", "000264", "000287"], 20.0, 25.0),
        "COMPACT_07": (["000067", "000715", "001124", "000197", "000196", "000142", "000076", "000264", "000287"], 20.0, 25.0),
    }
    if args.compact:
        sys.path.insert(0, str(BASE))
        from collect import Route
        for name, (suffixes, approach_m, exit_m) in compact_choices.items():
            if name in plan["routes"]:
                continue
            path = ["A2256W" + suffix for suffix in suffixes]
            for first, second in zip(path, path[1:]):
                if first not in allowed or second not in compatible[first]:
                    raise ValueError((name, "기존 보호·정상 연결 검사를 통과하지 못함", first, second))
            junction_indices = [i for i, key in enumerate(path) if str(links[key]["link_type"]) == "1"]
            first_junction, last_junction = junction_indices[0], junction_indices[-1]
            lead = route_from_xyz(np.vstack([allowed[key][:, 1:4] for key in path[:first_junction]]))
            tail = route_from_xyz(np.vstack([allowed[key][:, 1:4] for key in path[last_junction + 1:]]))
            if lead[-1, 0] < approach_m or tail[-1, 0] < exit_m:
                raise ValueError((name, "진입 또는 출구 길이가 부족함"))
            lead = interval(lead, lead[-1, 0] - approach_m, lead[-1, 0])
            tail = interval(tail, 0.0, exit_m)
            route = route_from_xyz(np.vstack([lead[:, 1:4]] + [allowed[key][:, 1:4] for key in path[first_junction:last_junction + 1]] + [tail[:, 1:4]]))
            radius = radius_screen(route)
            if radius < 5.87:
                raise ValueError((name, "차량 최소반경보다 급한 경로", radius))
            sample_s = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
            xy = np.column_stack([np.interp(sample_s, route[:, 0], route[:, k]) for k in (1, 2)])
            checks = {key: {"nearest_sample_m": float(tree.query(xy)[0].min()), "required_m": 20.0} for key, tree in trees.items()}
            if min(item["nearest_sample_m"] for item in checks.values()) <= 20.5:
                raise ValueError((name, "시험·검증 회랑과 너무 가까움"))
            nav = []
            for index in junction_indices:
                item = navigation(route, links[path[index]], "")
                angle = item["measured_turn_deg"]
                if angle is None:
                    raise ValueError((name, "교차로의 실제 회전각을 구할 수 없음"))
                item["command"] = "left" if angle > 45 else "right" if angle < -45 else "straight"
                nav.append(item)
            output = destination / (name + ".csv")
            if output.exists():
                raise FileExistsError(output)
            write_route(output, route)
            estimate_route = Route(output, nav)
            estimate_route.make_speed_profile(19.0)
            stations = np.arange(0.0, max(0.0, route[-1, 0] - 5.0), .5)
            seconds = .5 / np.maximum(1.5, np.interp(stations, estimate_route.s, estimate_route.speed_profile) / 3.6)
            command_seconds = Counter()
            for station, duration in zip(stations, seconds):
                command_seconds[estimate_route.command(station)] += float(duration)
            rare_seconds = command_seconds["left"] + command_seconds["right"]
            estimated_total = sum(command_seconds.values())
            density = {"cruise_kph": 19.0, "stop_before_end_m": 5.0, "reset_overhead_s": 9.0,
                       "command_seconds": dict(command_seconds), "left_right_fraction_driving": rare_seconds / estimated_total,
                       "left_right_fraction_with_reset": rare_seconds / (estimated_total + 9.0),
                       "left_straight_fraction_with_reset": (command_seconds["left"] + command_seconds["straight"]) / (estimated_total + 9.0),
                       "navigation_fraction_driving": (estimated_total - command_seconds["follow_road"]) / estimated_total,
                       "navigation_fraction_with_reset": (estimated_total - command_seconds["follow_road"]) / (estimated_total + 9.0),
                       "note_ko": "수집기의 곡률 감속·범주 명령 규칙으로 계산한 예상치이며 실제 유효 시간은 수집 후 확인한다."}
            plan["routes"][name] = {"name_ko": "연속 교차로 효율 수집 " + name, "split": "train", "route_csv": str(output.relative_to(BASE)),
                                   "route_sha256": sha256(output), "length_m": float(route[-1, 0]),
                                   "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
                                   "initial_yaw_deg_map": pose_heading(route), "source_links": path, "navigation": nav,
                                   "minimum_radius_from_3m_chords_m": radius, "spatial_separation_checks": checks,
                                   "approach_length_m": approach_m, "exit_length_m": exit_m,
                                   "estimated_collection_density": density, "experiment_id": "east_outer_90min",
                                   "collection_note": "기존 장소의 교차로를 정상 차로로 연속 연결한 효율 수집이다. 새로운 독립 장소나 추가 시험 코스가 아니다."}
            record = {"route": name, "length_m": float(route[-1, 0]), "junction_commands": dict(Counter(item["command"] for item in nav)),
                      "minimum_radius_m": radius, "minimum_protected_distance_m": min(item["nearest_sample_m"] for item in checks.values()),
                      "estimated_collection_density": density, "new_independent_geometry": False}
            records.append(record)
            used.update(path)
            all_selected.update(path)
            publish()
            print(name, round(route[-1, 0], 2), [item["command"] for item in nav], "예상좌우명령비율", round(density["left_right_fraction_with_reset"], 3), flush=True)
    if not records: raise ValueError("조건을 만족하는 장거리 경로를 찾지 못함")
    print("완료", destination, "경로수", len(records), flush=True)


if __name__ == "__main__":
    main()
