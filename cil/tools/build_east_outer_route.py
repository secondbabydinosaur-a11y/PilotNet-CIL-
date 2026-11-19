"""대회 출발점에서 동쪽 외곽도로를 따라 다음 좌회전 10m 전까지 준비한다.

MGeo 정상 진행 링크를 그대로 연결하며 지도는 초기 배치·명령·평가에만 쓴다.
실제 CIL 학습·검증 위치와 20m 이상 분리된 구간인지 확인한 뒤 목록에 추가한다.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, original_heading, route_from_xyz, sha256, write_route


def main():
    plan_path = BASE / "routes/plan.json"
    plan = json.loads(plan_path.read_text())
    map_path = Path("/home/kuuve/catkin_ws_구동현_2026카사/src/kcity2026_stack/kcity2026/data/mgeo_k-city_2025/link_set.json")
    if sha256(map_path) != plan["source_hashes"]["link_set.json"]:
        raise ValueError("대회 작업공간 지도와 고정한 수집 지도의 해시가 다름")
    links = {link["idx"]: link for link in json.loads(map_path.read_text()) if not link["lazy_init"]}
    keys = ["A2256W" + number for number in ("000748", "000078", "000324", "000597", "000241", "000243", "000231", "000233", "000253")]
    segments, gaps = [], []
    for i, key in enumerate(keys):
        xyz = np.asarray(links[key]["points"], dtype=float)
        if i:
            previous = links[keys[i - 1]]
            if previous["to_node_idx"] != links[key]["from_node_idx"]:
                raise ValueError("선택 링크의 정상 진행 노드가 연결되지 않음")
            gaps.append(float(np.linalg.norm(segments[-1][-1] - xyz[0])))
            if gaps[-1] > .01: raise ValueError("연결 링크의 좌표에 틈이 있음")
        segments.append(xyz)
    complete = route_from_xyz(np.vstack(segments))
    route = interval(complete, 0.0, float(complete[-1, 0]) - 10.0)
    reference_path = (BASE / plan["reference_route"]).resolve()
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1)
    if np.linalg.norm(route[0, 1:4] - reference[0, 1:4]) > 1e-7:
        raise ValueError("출발점이 고정한 대회 s0와 다름")

    nav = []
    for key in keys:
        if str(links[key].get("link_type")) != "1": continue
        item = navigation(route, links[key], "")
        turn = item["measured_turn_deg"]
        if turn is None: raise ValueError("교차로 회전 방향을 측정하지 못함")
        item["command"] = "left" if turn > 45 else "right" if turn < -45 else "straight"
        nav.append(item)
    if [item["command"] for item in nav] != ["right", "left", "straight", "straight"]:
        raise ValueError("검토한 교차로 명령과 다른 결과임")

    # 원본 주행 위치와 준비된 학습·검증 이미지 위치를 각각 검사한다.
    completion_path = BASE / "data/collection_completion.json"
    completion = json.loads(completion_path.read_text())
    positions = {"raw_train": [], "raw_validation": []}
    sources = {"routes/plan.json": sha256(plan_path), "data/collection_completion.json": sha256(completion_path)}
    for item in completion["accepted"]:
        path = BASE / item["session"] / "vehicle.csv"
        sources[str(path.relative_to(BASE))] = sha256(path)
        group = "raw_train" if item["split"] == "train" else "raw_validation"
        with path.open() as stream:
            positions[group].extend([[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)])
    for split in ("train", "val"):
        path = BASE / "data/prepared" / (split + ".csv")
        sources[str(path.relative_to(BASE))] = sha256(path)
        with path.open() as stream:
            positions["prepared_" + split] = [[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)]
    sample_s = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
    samples = np.column_stack([np.interp(sample_s, route[:, 0], route[:, k]) for k in (1, 2)])
    checks = {}
    for name, xy in positions.items():
        measured = float(cKDTree(np.asarray(xy)).query(samples)[0].min())
        lower_bound = max(0.0, measured - .25)  # 경로 표본 사이의 오차를 보수적으로 뺀다.
        checks[name] = {"recorded_positions": len(xy), "nearest_sample_distance_m": measured,
                        "test_sample_step_m": .25, "distance_lower_bound_to_recorded_positions_m": lower_bound,
                        "required_separation_m": 20.0, "separation_passed": lower_bound >= 20.0}
        if lower_bound < 20.0: raise ValueError(("CIL 수집 위치와 20m 이상 분리되지 않음", name, measured))
    output = BASE / "routes/test/EAST_OUTER.csv"
    write_route(output, route)
    sources.update({"competition_workspace/MGeo/link_set.json": sha256(map_path),
                    "pilotnet/assets/evaluation_route.csv": sha256(reference_path),
                    "tools/build_east_outer_route.py": sha256(Path(__file__))})
    info = {"name_ko": "대회 출발점에서 동쪽 외곽도로까지", "split": "test",
            "evaluation_scope": "held_out_from_cil_training_and_validation_same_kcity_map",
            "route_csv": str(output.relative_to(BASE)), "route_sha256": sha256(output),
            "length_m": float(route[-1, 0]), "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
            "initial_yaw_deg_map": original_heading(reference, 0.0), "source_links": keys,
            "source_mgeo_path": str(map_path.parent), "navigation": nav,
            "endpoint_basis": {"start_original_competition_s_m": 0.0, "last_link_id": keys[-1],
                               "next_junction_link_id": "A2256W000006", "end_before_next_junction_m": 10.0,
                               "full_selected_links_length_m": float(complete[-1, 0])},
            "geometry_validation": {"native_link_direction_only": True, "all_nodes_connected": True,
                                    "maximum_link_endpoint_gap_m": max(gaps),
                                    "maximum_output_segment_m": float(np.linalg.norm(np.diff(route[:, 1:3], axis=0), axis=1).max()),
                                    "planar_geometry_modified": False},
            "collection_overlap_checks": checks, "source_sha256": sources,
            "map_use": "초기 배치·진행 명령·평가에만 사용하며 신경망 조향 입력에 사용하지 않음",
            "notes": ["손으로 표시한 후보의 동쪽 외곽도로 부분만 사용하고, 학습 위치에 가까운 다음 좌회전 이후는 포함하지 않는다.",
                      "모든 링크를 원래 진행 방향으로 연결하고 마지막 링크의 끝 10m만 제외했다.",
                      "지도 related_signal이 straight여도 실제 출구 기하가 꺾이는078은우회전,597은좌회전이다.",
                      "교차로 접근18m·출구유지6m의 수집 정책을 동일하게 사용한다.",
                      "20m 분리는 CIL 학습·검증 위치 기준이다. 출발부 원래s0~65m는 과거 PilotNet 학습에 포함됐으므로 PilotNet에도 전부 미학습인 도로라 주장하지 않는다.",
                      "같은 K-City 맵의 별도 구간 시험이며 새로운 도시나 날씨에 대한 일반화 시험은 아니다."]}
    manifest_path = BASE / "routes/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["routes"]["EAST_OUTER"] = info
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: info[key] for key in ("length_m", "start_xyz_m", "end_xyz_m", "initial_yaw_deg_map", "navigation", "collection_overlap_checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
