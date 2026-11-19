"""마지막 좌회전 직전의 반대 차로에서 출발해 대회 시작점 근처로 돌아간다.

원본 좌표를 뒤집지 않는다. MGeo의 정상 진행 링크와 연결 노드만 사용하며,
대회 도로 회랑에서 벗어나는 지름길을 제외한다. 지도는 조향 입력이 아니다.
"""
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, route_from_xyz, sha256, write_route
from build_competition_route import overlap_checks


def projection(route, xy):
    edges = np.diff(route[:, 1:3], axis=0)
    length2 = np.sum(edges ** 2, axis=1)
    valid = length2 > 1e-10
    indices = np.flatnonzero(valid)
    fraction = np.clip(np.sum((xy - route[:-1, 1:3][valid]) * edges[valid], axis=1) / length2[valid], 0.0, 1.0)
    points = route[:-1, 1:3][valid] + fraction[:, None] * edges[valid]
    i = int(np.argmin(np.linalg.norm(points - xy, axis=1)))
    j = indices[i]
    station = float(route[j, 0] + fraction[i] * (route[j + 1, 0] - route[j, 0]))
    return station, points[i], math.degrees(math.atan2(edges[j, 1], edges[j, 0]))


def point(route, station):
    return np.array([np.interp(station, route[:, 0], route[:, k]) for k in (1, 2)])


def main():
    plan_path = BASE / "routes/plan.json"
    plan = json.loads(plan_path.read_text())
    reference_path = (BASE / plan["reference_route"]).resolve()
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1)
    map_path = Path(plan["source"]) / "link_set.json"
    if sha256(map_path) != plan["source_hashes"]["link_set.json"]:
        raise ValueError("수집 계획 이후 MGeo 링크가 변경됨")
    links = {link["idx"]: link for link in json.loads(map_path.read_text()) if not link["lazy_init"]}
    native = {key: route_from_xyz(link["points"]) for key, link in links.items()}
    corridor = interval(reference, 0.0, 609.601)
    corridor_tree = cKDTree(corridor[:, 1:3])
    start_key, end_key = "A2256W000292", "A2256W000336"
    desired_start_s, desired_end_s = 605.0, 0.0
    start_s, _, start_yaw = projection(native[start_key], point(reference, desired_start_s))
    end_s, _, _ = projection(native[end_key], point(reference, desired_end_s))

    # 원본 회랑에 가까우며 원래 station이 감소하는 정상 진행 링크만 허용한다.
    allowed, adjacency = {}, {}
    for key, route in native.items():
        distances, nearest = corridor_tree.query(route[:, 1:3])
        station = corridor[nearest, 0]
        if distances.max() > 18.0 or station[-1] > station[0] + 2.0:
            continue
        length = float(route[-1, 0])
        cost = length * (1.0 + .15 * float(distances.mean())) + 20.0 * float(np.maximum(np.diff(station), 0.0).sum())
        link = links[key]
        allowed[key] = dict(cost=cost, maximum_corridor_sample_distance_m=float(distances.max()))
        adjacency.setdefault(link["from_node_idx"], []).append((link["to_node_idx"], key, cost))
    begin, goal = links[start_key]["to_node_idx"], links[end_key]["from_node_idx"]
    queue, visited, middle = [(0.0, begin, [])], set(), None
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited: continue
        visited.add(node)
        if node == goal:
            middle = path
            break
        for next_node, key, weight in adjacency.get(node, []):
            if key in (start_key, end_key, "A2256W000083"): continue
            heapq.heappush(queue, (cost + weight, next_node, path + [key]))
    if middle is None: raise ValueError("정상 반대 방향 차로의 연결 경로를 찾지 못함")
    keys = [start_key] + middle + [end_key]
    if "A2256W000083" in keys: raise ValueError("기존 마지막 좌회전 링크에 진입함")

    segments, ranges, gaps, boundaries = [], [], [], []
    accumulated = 0.0
    for i, key in enumerate(keys):
        original = native[key]
        a = start_s if i == 0 else 0.0
        z = end_s if i == len(keys) - 1 else float(original[-1, 0])
        segment = interval(original, a, z)
        if i:
            previous = links[keys[i - 1]]
            if previous["to_node_idx"] != links[key]["from_node_idx"]:
                raise ValueError("MGeo 노드가 연결되지 않음")
            gaps.append(float(np.linalg.norm(segments[-1][-1, 1:4] - segment[0, 1:4])))
            if gaps[-1] > .05: raise ValueError("연결 링크의 끝점 좌표가 맞지 않음")
        ranges.append({"link_id": key, "native_start_s_m": a, "native_end_s_m": z,
                       "reverse_route_start_s_m": accumulated, "reverse_route_end_s_m": accumulated + segment[-1, 0]})
        boundaries.append(accumulated)
        accumulated += float(segment[-1, 0])
        segments.append(segment)
    route = route_from_xyz(np.vstack([segment[:, 1:4] for segment in segments]))
    nav = []
    for key in keys:
        link = links[key]
        if str(link.get("link_type")) != "1": continue
        item = navigation(route, link, "")
        turn = item["measured_turn_deg"]
        if turn is None: raise ValueError("교차로 회전 방향을 확인하지 못함")
        item["command"] = "left" if turn > 45 else "right" if turn < -45 else "straight"
        nav.append(item)

    dense_s = np.r_[np.arange(0.0, route[-1, 0], .25), route[-1, 0]]
    samples = np.column_stack([np.interp(dense_s, route[:, 0], route[:, k]) for k in (1, 2)])
    distance, index = corridor_tree.query(samples)
    original_s = corridor[index, 0]
    if distance.max() > 18.0 or np.max(np.diff(original_s)) > 2.0:
        raise ValueError("도로 회랑을 벗어나거나 지름길로 원래 진행 순서를 건너뜀")
    correspondence = [{"reverse_route_s_m": float(s), "original_reference_s_m": float(corridor[j, 0]),
                       "opposite_lane_distance_m": float(d)}
                      for s, d, j in zip(dense_s, distance, index)]
    correspondence = correspondence[::40] + [correspondence[-1]]
    first_original_s, first_xy, _ = projection(corridor, route[0, 1:3])
    last_original_s, last_xy, _ = projection(corridor, route[-1, 1:3])
    output = BASE / "routes/test/COMPETITION_REVERSE.csv"
    write_route(output, route)
    overlap = overlap_checks(route)
    info = {"name_ko": "마지막 좌회전 직전에서 대회 출발점 근처까지 반대 차로 주행", "split": "demonstration",
            "evaluation_scope": "continuous_reverse_corridor_normal_direction_lane_demonstration",
            "not_an_unseen_only_test": True, "traffic_light_or_obstacle_compliance_evaluated": False,
            "route_csv": str(output.relative_to(BASE)), "route_sha256": sha256(output),
            "length_m": float(route[-1, 0]), "start_xyz_m": route[0, 1:4].tolist(), "end_xyz_m": route[-1, 1:4].tolist(),
            "initial_yaw_deg_map": start_yaw, "source_links": keys, "source_link_intervals": ranges,
            "navigation": nav, "map_use": "초기 배치·진행 명령·평가에만 사용하며 조향 입력에 사용하지 않음",
            "endpoint_basis": {"desired_original_start_s_m": desired_start_s, "desired_original_end_s_m": desired_end_s,
                               "start_nearest_original_s_m": first_original_s, "end_nearest_original_s_m": last_original_s,
                               "start_opposite_lane_offset_m": float(np.linalg.norm(route[0, 1:3] - first_xy)),
                               "end_opposite_lane_offset_m": float(np.linalg.norm(route[-1, 1:3] - last_xy)),
                               "original_last_left_stopline_s_m": 609.601,
                               "start_before_original_left_stopline_m": 609.601 - desired_start_s},
            "geometry_validation": {"uses_only_native_link_direction": True, "source_points_reversed": False,
                                    "all_link_nodes_connected": True, "max_link_endpoint_gap_m": max(gaps),
                                    "forbidden_original_last_left_link_absent": True,
                                    "max_corridor_sample_distance_m": float(distance.max()),
                                    "maximum_forward_reference_step_m": float(np.max(np.diff(original_s))),
                                    "reference_station_correspondence": correspondence,
                                    "dijkstra_corridor_limit_m": 18.0,
                                    "cost_definition": "링크 길이에 회랑 거리와 원래 station의 역방향 이탈 비용을 더함"},
            "collection_overlap_checks": overlap,
            "source_sha256": {"MGeo/link_set.json": sha256(map_path),
                              "pilotnet/assets/evaluation_route.csv": sha256(reference_path),
                              "tools/build_competition_reverse_route.py": sha256(Path(__file__))},
            "notes": ["원래 대회 좌표를 역순으로 사용하는 역주행 경로가 아니라 정상 반대 방향 차로의 연결 경로다.",
                      "원래 마지막 좌회전083에는 진입하지 않고, 그 직전 위치에 대응하는 반대 차로에서 서쪽으로 출발한다.",
                      "같은 도로 배경을 반대 시점에서 관찰하므로 새로운 시점이라는 이유로 미학습 도로 일반화 시험이라 부르지 않는다.",
                      "교차로 명령은 실제 출구 기하로 판정하며 수집과 같은 접근18m·출구유지6m 정책을 사용한다.",
                      "신호색·장애물 대응이나 교통 규정 준수는 이 명령 주행 시연의 평가 범위가 아니다."]}
    manifest_path = BASE / "routes/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["routes"]["COMPETITION_REVERSE"] = info
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: info[key] for key in ("length_m", "start_xyz_m", "end_xyz_m", "initial_yaw_deg_map", "endpoint_basis", "navigation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
