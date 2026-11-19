"""시험 도로를 보존하고 서로 다른 우회전·좌회전·직진 수집 경로를 준비한다.

원본 설정·분할·모델을 수정하지 않는다. 새 설정으로 실행할 경로와 검사 근거만
routes/experiments/east_outer_data_v1에 보관한다. 시뮬레이터를 제어하지 않는다.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, pose_heading, route_from_xyz, sha256, write_route


CHOICES = {
    "EX_R1": ("right", ["000225", "000116", "000365"]),
    "EX_R2": ("right", ["000544", "000465", "001076"]),
    "EX_R3": ("right", ["000428", "000089", "000413"]),
    "EX_R4": ("right", ["000213", "000111", "000225"]),
    "EX_L1": ("left", ["000544", "001078", "001077"]),
    "EX_L2": ("left", ["000224", "000118", "000360"]),
    "EX_S1": ("straight", ["001067", "001079", "001077"]),
    "EX_S2": ("straight", ["000407", "000405", "000413"]),
}


def radius_screen(route):
    s = route[:, 0]
    back, front = np.maximum(s[0], s - 1.5), np.minimum(s[-1], s + 1.5)
    chord = np.column_stack([np.interp(front, s, route[:, k]) - np.interp(back, s, route[:, k]) for k in (1, 2)])
    yaw = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    return float(1.0 / max(np.abs(np.gradient(yaw, s)).max(), 1e-9))


def main():
    source_plan = BASE / "routes/plan.json"
    frozen = json.loads(source_plan.read_text())
    plan = copy.deepcopy(frozen)
    map_path = Path(frozen["source"]) / "link_set.json"
    if sha256(map_path) != frozen["source_hashes"]["link_set.json"]:
        raise ValueError("MGeo 지도가 고정한 원본과 다름")
    links = {item["idx"]: item for item in json.loads(map_path.read_text()) if not item["lazy_init"]}
    destination = BASE / "routes/experiments/east_outer_data_v1"
    destination.mkdir(parents=True, exist_ok=True)

    protected = {name: np.asarray(item["guard_reference_xyz"]) for name, item in frozen["test_reservations"].items()}
    manifest = json.loads((BASE / "routes/test/manifest.json").read_text())
    for name in ("EAST_OUTER", "U2", "J_left", "J_straight", "J_right"):
        protected[name] = np.loadtxt(BASE / manifest["routes"][name]["route_csv"], delimiter=",", skiprows=1)[:, 1:4]
    for name in ("P2Q", "V1", "V2"):
        protected[name] = np.loadtxt(BASE / frozen["routes"][name]["route_csv"], delimiter=",", skiprows=1)[:, 1:4]
    for name, xyz in protected.items():
        if name not in plan["test_reservations"]:
            plan["test_reservations"][name] = {"guard_reference_xyz": xyz.tolist(), "buffer_m": 20.0}
    trees = {name: cKDTree(xyz[:, :2]) for name, xyz in protected.items()}
    completion = json.loads((BASE / "data/collection_completion.json").read_text())
    actual_val = []
    for item in completion["accepted"]:
        if item["split"] != "validation": continue
        with (BASE / item["session"] / "vehicle.csv").open() as stream:
            actual_val.extend([[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)])
    trees["accepted_validation_positions"] = cKDTree(np.asarray(actual_val))
    reports = []
    for name, (expected_command, short_keys) in CHOICES.items():
        keys = ["A2256W" + key for key in short_keys]
        for a, b in zip(keys, keys[1:]):
            if links[a]["to_node_idx"] != links[b]["from_node_idx"]:
                raise ValueError((name, "정상 진행 노드가 연결되지 않음"))
        native = [route_from_xyz(links[key]["points"]) for key in keys]
        approach = interval(native[0], max(0.0, native[0][-1, 0] - 30.0), native[0][-1, 0])
        outgoing = route_from_xyz(np.vstack([route[:, 1:4] for route in native[2:]]))
        if outgoing[-1, 0] < 25.0: raise ValueError((name, "출구 도로가 너무 짧음"))
        outgoing = interval(outgoing, 0.0, min(35.0, outgoing[-1, 0]))
        route = route_from_xyz(np.vstack((approach[:, 1:4], native[1][:, 1:4], outgoing[:, 1:4])))
        nav = navigation(route, links[keys[1]], expected_command)
        measured_command = "left" if nav["measured_turn_deg"] > 45 else "right" if nav["measured_turn_deg"] < -45 else "straight"
        if measured_command != expected_command: raise ValueError((name, "명령과 실제 회전 기하가 다름", nav))
        radius = radius_screen(route)
        if radius < 5.87: raise ValueError((name, "차량 최소 반경보다 급한 지도 기하", radius))
        sample_s = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
        xy = np.column_stack([np.interp(sample_s, route[:, 0], route[:, k]) for k in (1, 2)])
        separation = {}
        for label, tree in trees.items():
            measured = float(tree.query(xy)[0].min())
            lower = max(0.0, measured - .25)
            separation[label] = {"nearest_sample_m": measured, "conservative_lower_bound_m": lower, "required_m": 20.0}
            if lower <= 20.5: raise ValueError((name, label, "시험·검증 위치와의 거리가 부족함", lower))
        path = destination / (name + ".csv")
        write_route(path, route)
        info = {"name_ko": "추가 수집 " + {"right": "우회전", "left": "좌회전", "straight": "직진"}[expected_command] + " " + name,
                "split": "train", "route_csv": str(path.relative_to(BASE)), "route_sha256": sha256(path),
                "length_m": float(route[-1, 0]), "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
                "initial_yaw_deg_map": pose_heading(route), "source_links": keys, "navigation": [nav],
                "approach_length_m": float(approach[-1, 0]), "exit_length_m": float(outgoing[-1, 0]),
                "minimum_radius_from_3m_chords_m": radius, "junction_command": expected_command,
                "experiment_id": "east_outer_data_v1", "same_approach_group": keys[0],
                "spatial_separation_checks": separation,
                "map_widths_m": {key: [links[key]["width_start"], links[key]["width_end"]] for key in keys},
                "map_lane_boundaries": {key: {"left": links[key]["lane_mark_left"], "right": links[key]["lane_mark_right"]} for key in keys},
                "collection_note": "지도에 양측 차선 경계가 있는 접근·출구를 선택했다. 실제 영상·추종·센서 품질 검사는 수집 후 별도로 통과해야 한다."}
        plan["routes"][name] = info
        reports.append({"route": name, "junction_link": keys[1], "command": expected_command,
                        "length_m": info["length_m"], "minimum_radius_m": radius,
                        "minimum_guard_distance_lower_bound_m": min(value["conservative_lower_bound_m"] for value in separation.values())})
        print(name, keys, expected_command, round(radius, 3), round(reports[-1]["minimum_guard_distance_lower_bound_m"], 3), flush=True)
    plan["status"] = "additional_data_experiment_routes_prepared_not_collected"
    plan["experiment"] = {"id": "east_outer_data_v1", "collection_route_ids": list(CHOICES),
                          "frozen_source_plan_sha256": sha256(source_plan), "reports": reports,
                          "old_routes_preserved_for_provenance_only": True,
                          "collect_only_new_route_ids": True,
                          "notes": ["신규 우회전 네 가지와 좌회전 두 가지·직진 두 가지를 함께 수집한다.",
                                    "EX_R2/EX_L1은 같은 접근로에서 서로 다른 출구를 선택한다. 나머지도 다른 방향과 도로 추종을 포함해 우회전만 추가하지 않는다.",
                                    "EX_S1은 남서쪽의 반대 접근 직진, EX_S2는 기존 T6 직진 대신 북동쪽의 새 직진 접근로를 선택했다.",
                                    "시험 회랑과 검증 위치는 수집 금지 구역이며 EAST_OUTER를 test_reservations에 추가했다.",
                                    "022·637 후보는 최소 반경이 각각 약2.83·3.01m여서 제외했다."]}
    plan_path = destination / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    config = json.loads((BASE / "config.json").read_text())
    config["paths"]["route_plan"] = str(plan_path.relative_to(BASE))
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (destination / "route_summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print("계획:", plan_path)


if __name__ == "__main__":
    main()
