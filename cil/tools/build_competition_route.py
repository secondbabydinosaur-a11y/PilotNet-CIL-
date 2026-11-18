"""대회 출발점부터 회전교차로 전 마지막 신호 좌회전의 출구까지 자른다.

지도는 초기 배치·진행 명령·평가에만 사용한다. 학습한 도로가 포함된 연속
명령 주행 시연이며, 신호색·장애물 대응이나 미학습 도로 일반화 시험은 아니다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import (
    BASE, interval, navigation, original_heading, route_from_xyz, sha256, write_route,
)


def matching_links(reference, links, end_s):
    """실제 연결 관계와 원본 좌표를 동시에 만족하는 링크 순서를 찾는다."""
    key, station, result = "A2256W000748", 0.0, []
    for _ in range(100):
        link = links[key]
        native = route_from_xyz(link["points"])
        length = native[-1, 0]
        result.append((link, station, station + length))
        station += length
        if station >= end_s:
            return result
        candidates = []
        for other in links.values():
            if other["from_node_idx"] != link["to_node_idx"]:
                continue
            route = route_from_xyz(other["points"])
            local_s = np.linspace(0.0, min(route[-1, 0], end_s - station), 100)
            actual = np.column_stack([np.interp(local_s, route[:, 0], route[:, k]) for k in (1, 2)])
            expected = np.column_stack([np.interp(station + local_s, reference[:, 0], reference[:, k]) for k in (1, 2)])
            candidates.append((float(np.linalg.norm(actual - expected, axis=1).max()), other["idx"]))
        if not candidates or min(candidates)[0] > .01:
            raise ValueError("official route does not match connected MGeo links")
        _, key = min(candidates)
    raise ValueError("too many route links")


def overlap_checks(route):
    """실제 학습·검증 위치와 가까운 구간의 비율을 공개한다."""
    stations = np.r_[np.arange(0.0, route[-1, 0], .5), route[-1, 0]]
    samples = np.column_stack([np.interp(stations, route[:, 0], route[:, k]) for k in (1, 2)])
    completion = json.loads((BASE / "data/collection_completion.json").read_text())
    positions = {"raw_train": [], "raw_validation": []}
    for item in completion["accepted"]:
        name = "raw_train" if item["split"] == "train" else "raw_validation"
        with (BASE / item["session"] / "vehicle.csv").open() as stream:
            positions[name].extend([[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)])
    for split in ("train", "val"):
        with (BASE / "data/prepared" / (split + ".csv")).open() as stream:
            positions["prepared_" + split] = [[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)]
    result = {}
    for name, xy in positions.items():
        distances = cKDTree(np.asarray(xy)).query(samples)[0]
        mask = distances <= 3.5
        intervals, opened = [], None
        for i, inside in enumerate(mask):
            if inside and opened is None: opened = float(stations[i])
            if opened is not None and (not inside or i == len(mask) - 1):
                intervals.append([opened, float(stations[i if inside else i - 1])])
                opened = None
        result[name] = {"minimum_sample_distance_m": float(distances.min()),
                        "reference_samples": len(stations), "reference_sample_step_m": .5,
                        "fraction_within_distance": {str(d): float(np.mean(distances <= d)) for d in (.5, 1.0, 2.0, 3.5, 10.0, 20.0)},
                        "reference_intervals_within_3_5m": intervals}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, default=Path("/home/kuuve/catkin_ws_구동현_2026카사/src/kcity2026_stack/kcity2026/data/kcity_official"))
    args = parser.parse_args()
    plan_path = BASE / "routes/plan.json"
    plan = json.loads(plan_path.read_text())
    reference_path = (BASE / plan["reference_route"]).resolve()
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1)
    official_path, tags_path = args.official_dir / "route.csv", args.official_dir / "mission_tags.json"
    official = np.loadtxt(official_path, delimiter=",", skiprows=1)
    if official.shape != reference[:, 1:4].shape or not np.allclose(official, reference[:, 1:4], rtol=0, atol=1e-8):
        raise ValueError("PilotNet evaluation reference differs from the official route")
    tags = json.loads(tags_path.read_text())
    rb_entry = min(float(item["entry_s"]) for item in tags["roundabouts"])
    stops = [item for item in tags["stoplines"] if item.get("signalized") and item.get("go") == "left" and item["s"] < rb_entry]
    last = max(stops, key=lambda item: item["s"])
    map_dir = Path(plan["source"])
    links_path, lights_path = map_dir / "link_set.json", map_dir / "traffic_light_set.json"
    if sha256(links_path) != plan["source_hashes"]["link_set.json"]:
        raise ValueError("MGeo links changed since collection planning")
    links = {item["idx"]: item for item in json.loads(links_path.read_text()) if not item["lazy_init"]}
    light = next(item for item in json.loads(lights_path.read_text()) if item["idx"] == last["tl_id"])
    chain = matching_links(reference, links, last["s"] + 60.0)
    candidates = [(link, start, end) for link, start, end in chain
                  if link["idx"] in light["link_id_list"] and str(link.get("link_type")) == "1"
                  and abs(start - last["s"]) < 5.0]
    if len(candidates) != 1:
        raise ValueError("last signal cannot identify a unique route junction")
    final_link, final_entry, final_exit = candidates[0]
    end_s = final_exit + 10.0  # 좌회전 출구 뒤 10m를 더 주행한다.
    if end_s >= rb_entry - 20.0:
        raise ValueError("chosen endpoint entered the roundabout approach")
    route = interval(reference, 0.0, end_s)
    commands = []
    for link, start, end in chain:
        if start >= end_s or str(link.get("link_type")) != "1": continue
        item = navigation(route, link, "")
        turn = item["measured_turn_deg"]
        if turn is None: raise ValueError("junction turn geometry unavailable")
        item["command"] = "left" if turn > 45 else "right" if turn < -45 else "straight"
        commands.append(item)
    final_navigation = next(item for item in commands if item["link_id"] == final_link["idx"])
    if final_navigation["command"] != "left": raise ValueError("last signal exit is not geometrically a left turn")
    path = BASE / "routes/test/COMPETITION.csv"
    write_route(path, route)
    overlap = overlap_checks(route)
    info = {"name_ko": "대회 출발부터 마지막 신호 좌회전까지 연속 주행", "split": "demonstration",
            "evaluation_scope": "continuous_command_driving_demonstration_including_training_roads",
            "not_an_unseen_only_test": True, "traffic_light_or_obstacle_compliance_evaluated": False,
            "route_csv": str(path.relative_to(BASE)), "route_sha256": sha256(path),
            "reference_interval_s_m": [0.0, end_s], "length_m": float(route[-1, 0]),
            "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
            "initial_yaw_deg_map": original_heading(reference, 0.0),
            "source_links": [link["idx"] for link, start, end in chain if start < end_s],
            "navigation": commands, "map_use": "reset, command selection and evaluation only; never steering",
            "endpoint_basis": {"last_left_signal_id": last["tl_id"], "signal_position_xyz_m": light["point"],
                               "stopline_official_s_m": last["s"], "junction_link_id": final_link["idx"],
                               "junction_entry_official_s_m": final_entry, "junction_exit_official_s_m": final_exit,
                               "exit_road_distance_m": 10.0, "roundabout_entry_official_s_m": rb_entry,
                               "remaining_to_roundabout_entry_m": rb_entry - end_s},
            "collection_overlap_checks": overlap,
            "reference_matches_official_xyz_max_error_m": float(np.max(np.abs(official - reference[:, 1:4]))),
            "source_sha256": {"official_route.csv": sha256(official_path), "official_mission_tags.json": sha256(tags_path),
                              "pilotnet/assets/evaluation_route.csv": sha256(reference_path),
                              "MGeo/link_set.json": sha256(links_path), "MGeo/traffic_light_set.json": sha256(lights_path)},
            "notes": ["Original route geometry is unchanged; only the final endpoint is interpolated.",
                      "Includes roads observed during CIL collection; do not report as held-out generalization.",
                      "Navigation uses actual junction geometry,18m approach and6m exit hold, matching collection policy.",
                      "Signal color, stop-line response and obstacle handling were not learned; this demonstrates command-conditioned route following, not competition-rule compliance."]}
    manifest_path = BASE / "routes/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())  # 직전 최신본을 읽어 다른 시험 경로의 변경을 유지한다.
    manifest["routes"]["COMPETITION"] = info
    manifest["scope"] = "Each route declares its scope; COMPETITION is a continuous demonstration including training roads."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"endpoint": info["endpoint_basis"], "end_xyz_m": info["end_xyz_m"],
                      "commands": [(n["link_id"], n["command"], n["entry_s_m"], n["exit_s_m"]) for n in commands],
                      "training_nearby_fraction_3_5m": overlap["raw_train"]["fraction_within_distance"]["3.5"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
