"""고정된 PilotNet 시험 구간과 MGeo 출구로 CIL 시험 경로만 생성한다.

지도는 초기 배치·명령 선택·진행률 평가에만 사용한다. 신경망의 조향이나
학습 데이터는 변경하지 않는다. P2Q는 과거 수집 예약 자료로만 보존한다.
대회 연속 경로는 build_competition_route.py로 추가하며 재생성할 때도 보존한다.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(route, start, end):
    """원래 선분을 바꾸지 않고 지정한 거리의 두 끝점만 보간한다."""
    if not route[0, 0] <= start < end <= route[-1, 0]:
        raise ValueError("interval outside source route")
    station = np.r_[start, route[(route[:, 0] > start) & (route[:, 0] < end), 0], end]
    return np.column_stack((station - start,
                            *[np.interp(station, route[:, 0], route[:, k]) for k in (1, 2, 3)]))


def route_from_xyz(xyz):
    xyz = np.asarray(xyz, dtype=float)
    xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) > 1e-5]]
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
    return np.column_stack((station, xyz))


def pose_heading(route):
    delta = route[1, 1:3] - route[0, 1:3]
    return math.degrees(math.atan2(delta[1], delta[0]))


def original_heading(reference, station):
    """PilotNet Reference.pose와 같은 원본 선분 접선을 사용한다."""
    j = max(0, min(len(reference) - 2, int(np.searchsorted(reference[:, 0], station) - 1)))
    return pose_heading(reference[j:j + 2])


def projected_station(route, point):
    edges = np.diff(route[:, 1:3], axis=0)
    length2 = np.sum(edges ** 2, axis=1)
    valid = length2 > 1e-10
    indices = np.flatnonzero(valid)
    fraction = np.clip(np.sum((point[:2] - route[:-1, 1:3][valid]) * edges[valid], axis=1) / length2[valid], 0, 1)
    projected = route[:-1, 1:3][valid] + fraction[:, None] * edges[valid]
    i = int(np.argmin(np.linalg.norm(projected - point[:2], axis=1)))
    j = indices[i]
    return float(route[j, 0] + fraction[i] * (route[j + 1, 0] - route[j, 0]))


def point(route, station):
    return np.array([np.interp(station, route[:, 0], route[:, k]) for k in (1, 2)])


def navigation(route, link, command, entry=None):
    points = np.asarray(link["points"])
    start = projected_station(route, points[0]) if entry is None else entry
    end = projected_station(route, points[-1])
    before = point(route, start) - point(route, max(0.0, start - 3.0))
    after = point(route, min(route[-1, 0], end + 3.0)) - point(route, end)
    turn = None if np.linalg.norm(before) < .1 else (math.degrees(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0])) + 180) % 360 - 180
    return {"link_id": link["idx"], "entry_s_m": start, "exit_s_m": end,
            "approach_m": 18.0, "exit_hold_m": 6.0, "command": command,
            "map_related_signal": link.get("related_signal"),
            "measured_turn_deg": None if turn is None else round(turn, 3)}


def write_route(path, route):
    if not np.isfinite(route).all() or np.any(np.diff(route[:, 0]) <= 0):
        raise ValueError("invalid evaluation route")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["s_m", "x_m", "y_m", "z_m"])
        writer.writerows([[f"{value:.9f}" for value in row] for row in route])


def main():
    plan_path = BASE / "routes/plan.json"
    plan = json.loads(plan_path.read_text())
    map_path = Path(plan["source"]) / "link_set.json"
    if sha256(map_path) != plan["source_hashes"]["link_set.json"]:
        raise ValueError("MGeo source changed since collection planning")
    links = {row["idx"]: row for row in json.loads(map_path.read_text()) if not row["lazy_init"]}
    reference_path = (BASE / plan["reference_route"]).resolve()
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1)
    protocol_path = BASE.parent / "pilotnet/docs/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["held_out_sections"]["U2"] != [978.337, 1210.0]:
        raise ValueError("frozen PilotNet U2 protocol changed")
    output = BASE / "routes/test"
    previous_manifest = output / "manifest.json"
    previous_routes = json.loads(previous_manifest.read_text()).get("routes", {}) if previous_manifest.exists() else {}
    competition = previous_routes.get("COMPETITION")  # 연속 시연은 미학습 구간 분리 검사의 대상이 아니다.
    routes, data = {}, {}

    data["U2"] = interval(reference, 978.337, 1210.0)
    routes["U2"] = {"name_ko": "PilotNet 실패 곡선 재시험", "split": "test",
                    "reference_interval_s_m": [978.337, 1210.0],
                    "source_links": ["A2256W000054", "A2256W000846", "A2256W000411"],
                    "initial_yaw_deg_map": original_heading(reference, 978.337),
                    "navigation": [navigation(data["U2"], links["A2256W000054"], "straight", entry=0.0)]}

    approach = links["A2256W000748"]
    approach_length = route_from_xyz(approach["points"])[-1, 0]
    shared = interval(reference, 40.0, approach_length)
    left_link = links["A2256W000182"]
    exit_distance = 108.0 - approach_length - route_from_xyz(left_link["points"])[-1, 0]
    data["J_left"] = interval(reference, 40.0, 108.0)
    variants = {"J_left": ("A2256W000182", "A2256W000329", "left"),
                "J_straight": ("A2256W000338", "A2256W000335", "straight"),
                "J_right": ("A2256W000078", "A2256W000324", "right")}
    for name, (junction_key, exit_key, command) in variants.items():
        junction, outgoing = links[junction_key], links[exit_key]
        if approach["to_node_idx"] != junction["from_node_idx"] or junction["to_node_idx"] != outgoing["from_node_idx"]:
            raise ValueError("disconnected MGeo junction topology")
        if name != "J_left":
            tail = interval(route_from_xyz(outgoing["points"]), 0.0, exit_distance)
            data[name] = route_from_xyz(np.vstack((shared[:, 1:4], junction["points"], tail[:, 1:4])))
        routes[name] = {"name_ko": "PilotNet 교차로 J " + {"left": "좌회전", "straight": "직진", "right": "우회전"}[command],
                        "split": "test", "source_links": [approach["idx"], junction_key, exit_key],
                        "same_approach_group": "pilotnet_J_node_736", "junction_command": command,
                        "shared_reference_approach_s_m": [40.0, float(approach_length)],
                        "exit_road_length_after_junction_m": float(exit_distance),
                        "initial_yaw_deg_map": original_heading(reference, 40.0),
                        "navigation": [navigation(data[name], junction, command)]}
        if name == "J_left": routes[name]["reference_interval_s_m"] = [40.0, 108.0]

    raw_xy = {"train": [], "validation": []}
    completion_path = BASE / "data/collection_completion.json"
    completion = json.loads(completion_path.read_text())
    for item in completion["accepted"]:
        session = BASE / item["session"]
        with (session / "vehicle.csv").open() as stream:
            xy = [[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)]
        raw_xy[item["split"]].extend(xy)
    trees = {split: cKDTree(np.asarray(xy)) for split, xy in raw_xy.items()}
    prepared_hashes = {}
    for split in ("train", "val"):
        path = BASE / "data/prepared" / (split + ".csv")
        prepared_hashes[str(path.relative_to(BASE))] = sha256(path)
        with path.open() as stream:
            xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)])
        trees["prepared_" + split] = cKDTree(xy)

    for name, route in data.items():
        path = output / (name + ".csv")
        write_route(path, route)
        info = routes[name]
        info.update(route_csv=str(path.relative_to(BASE)), length_m=float(route[-1, 0]),
                    start_xyz_m=route[0, 1:4].tolist(), end_xyz_m=route[-1, 1:4].tolist(),
                    route_sha256=sha256(path), max_segment_m=float(np.linalg.norm(np.diff(route[:, 1:3], axis=0), axis=1).max()),
                    map_use="reset, navigation command and evaluation only; never steering")
        stations = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
        samples = np.column_stack([np.interp(stations, route[:, 0], route[:, k]) for k in (1, 2)])
        separation = {}
        for split, tree in trees.items():
            measured = float(tree.query(samples)[0].min())
            lower_bound = max(0.0, measured - .25)
            separation[split] = {"nearest_sample_m": measured, "conservative_test_polyline_distance_lower_bound_m": lower_bound,
                                 "within_20m_guard": lower_bound < 20.0}
            if lower_bound < 20.0: raise ValueError((name, split, measured))
        info["collection_overlap_checks"] = separation
        print(name, round(info["length_m"], 3), info["start_xyz_m"], info["initial_yaw_deg_map"],
              [(n["command"], round(n["entry_s_m"], 3), round(n["exit_s_m"], 3)) for n in info.get("navigation", [])])
    if competition:
        if sha256(BASE / competition["route_csv"]) != competition["route_sha256"]:
            raise ValueError("existing competition demonstration route changed")
        routes["COMPETITION"] = competition
    manifest = {"schema_version": 1, "status": "routes_prepared",
                "coordinate_frame": plan["coordinate_frame"], "vehicle_pose_frame": "rear axle; map yaw atan2(dy,dx)",
                "scope": "held-out road trials; optional competition demonstration can overlap training roads; no new town claim",
                "map_use": "reset, navigation command, route progress and CTE evaluation only; CNN determines steering",
                "no_training_or_model_updates": True, "routes": routes,
                "command_policy": "current junction first, upcoming within18m, exit hold6m, otherwise follow_road",
                "command_numbers": {"follow_road": 0, "left": 1, "straight": 2, "right": 3},
                "source_sha256": {str(reference_path.relative_to(BASE.parent)): sha256(reference_path),
                                  str(protocol_path.relative_to(BASE.parent)): sha256(protocol_path),
                                  "routes/plan.json": sha256(plan_path), "MGeo/link_set.json": sha256(map_path),
                                  "data/collection_completion.json": sha256(completion_path), **prepared_hashes},
                "notes": ["U2 and J_left preserve the exact frozen PilotNet reference intervals.",
                          "J_straight/right share the same approach and map-junction topology, then stop13.046m after the chosen junction exit.",
                          "J has map related_signal='straight' on all exits; commands use physical exit geometry.",
                          "Overlap checks use all accepted raw Ego positions and prepared training/validation positions; test polyline sampled every0.25m with0.25m conservative subtraction.",
                          "Spatial separation does not guarantee no common visual backgrounds.",
                          "P2Q is not an active evaluation route; its CSV remains a historical collection reservation."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
