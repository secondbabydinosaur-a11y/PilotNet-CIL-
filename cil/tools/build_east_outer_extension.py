"""Extend EAST_OUTER through the next left turn and the first right turn."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, route_from_xyz, sha256, write_route


def coordinates(path):
    with path.open(newline="", encoding="utf-8") as stream:
        xy = np.asarray([(float(row["x_m"]), float(row["y_m"])) for row in csv.DictReader(stream)])
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"invalid sample positions: {path}")
    return cKDTree(xy)


def distances(route, start, end, tree):
    stations = np.linspace(start, end, max(2, int(np.ceil((end - start) / .25)) + 1))
    points = np.column_stack([np.interp(stations, route[:, 0], route[:, k]) for k in (1, 2)])
    nearest, _ = tree.query(points)
    return {"min_m": float(nearest.min()), "fraction_within_3_5m": float(np.mean(nearest <= 3.5)),
            "fraction_within_20m": float(np.mean(nearest <= 20.))}


def middle_turns(manifest, original, prefix, links, data_paths):
    names = ("000231", "000085", "000234", "000086", "000343", "000346")
    approach, left, connector, right, exit_lane, continuation = (links["A2256W" + name] for name in names)
    chain = (approach, left, connector, right, exit_lane, continuation)
    for before, after in zip(chain, chain[1:]):
        if before["to_node_idx"] != after["from_node_idx"] or not np.allclose(
            before["points"][-1], after["points"][0], rtol=0, atol=.01
        ):
            raise ValueError("middle turns contain disconnected map lanes")

    former_straight = next(item for item in original["navigation"] if item["link_id"] == "A2256W000233")
    branch_s = former_straight["entry_s_m"]
    approach_route = interval(prefix, 0., branch_s)
    if not np.allclose(approach_route[-1, 1:], left["points"][0], rtol=0, atol=1e-6):
        raise ValueError("middle left does not join EAST_OUTER")
    after_right = route_from_xyz(np.vstack((exit_lane["points"], continuation["points"])))
    tail = route_from_xyz(np.vstack((left["points"], connector["points"], right["points"],
                                    interval(after_right, 0., 20.)[:, 1:])))
    tail[:, 0] += branch_s
    route = np.vstack((approach_route, tail[1:]))
    turns = [navigation(route, left, "left", entry=branch_s), navigation(route, right, "right")]
    if turns[0]["measured_turn_deg"] < 60. or turns[1]["measured_turn_deg"] > -60.:
        raise ValueError("middle commands disagree with the map geometry")
    overlap = {split: distances(route, branch_s, route[-1, 0], coordinates(path))
               for split, path in data_paths.items()}

    course = "EAST_OUTER_MIDDLE_LEFT_RIGHT"
    output = BASE / "routes/test" / (course + ".csv")
    write_route(output, route)
    manifest["routes"][course] = {
        "name_ko": "동쪽 바깥길 영상 45초 교차로에서 좌회전 후 우회전", "split": "test",
        "evaluation_scope": "middle_junction_diagnostic_with_training_proximity_reported",
        "route_csv": str(output.relative_to(BASE)), "route_sha256": sha256(output),
        "length_m": float(route[-1, 0]), "start_xyz_m": original["start_xyz_m"],
        "end_xyz_m": route[-1, 1:].tolist(), "initial_yaw_deg_map": original["initial_yaw_deg_map"],
        "navigation": [item for item in original["navigation"] if item["exit_s_m"] < branch_s] + turns,
        "prefix_course": "EAST_OUTER", "prefix_sha256": original["route_sha256"],
        "branch_s_m": branch_s, "replaces_straight_link": former_straight["link_id"],
        "source_links": original["source_links"][:7] + [link["idx"] for link in chain[1:]],
        "extension_proximity": overlap, "data_sha256": {split: sha256(path) for split, path in data_paths.items()},
    }
    return course


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--middle-turns", action="store_true", help="take the left turn visible at 0:45 and the next right")
    args = parser.parse_args()

    manifest_path = BASE / "routes/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = manifest["routes"]["EAST_OUTER"]
    prefix_path = BASE / original["route_csv"]
    if sha256(prefix_path) != original["route_sha256"]:
        raise ValueError("EAST_OUTER source changed")
    prefix = np.loadtxt(prefix_path, delimiter=",", skiprows=1)
    map_dir = Path(original["source_mgeo_path"])
    map_path = map_dir / "link_set.json"
    if sha256(map_path) != original["source_sha256"]["competition_workspace/MGeo/link_set.json"]:
        raise ValueError("EAST_OUTER map source changed")
    links = {row["idx"]: row for row in json.loads(map_path.read_text())}
    data_paths = {"train": (BASE / args.train_csv).resolve(), "val": (BASE / args.val_csv).resolve()}
    if args.middle_turns:
        course = middle_turns(manifest, original, prefix, links, data_paths)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest["routes"][course], ensure_ascii=False, indent=2))
        return
    last, left, inner, outer, right, exit_lane = [links["A2256W" + key] for key in (
        "000253", "000006", "000259", "000258", "000359", "000363")]
    pairs = ((last, left), (left, inner), (outer, right), (right, exit_lane))
    for before, after in pairs:
        if before["to_node_idx"] != after["from_node_idx"] or not np.allclose(
            before["points"][-1], after["points"][0], rtol=0, atol=.01
        ):
            raise ValueError("extension contains disconnected map lanes")

    # The left turn feeds the inner lane. Move to the right lane on the dashed line before the turn.
    if not inner["can_move_right_lane"] or inner["right_lane_change_dst_link_idx"] != outer["idx"]:
        raise ValueError("map does not allow entry to the adjacent right-turn lane")
    boundaries = {row["idx"]: row for row in json.loads((map_dir / "lane_boundary_set.json").read_text())}
    marking_id = "B2256W000498"
    marking = boundaries[marking_id]
    if marking_id not in inner["lane_mark_right"] or "broken" not in marking["lane_shape"]:
        raise ValueError("missing expected dashed divider")
    dashed_m = route_from_xyz(marking["points"])[-1, 0]
    inner_route, outer_route = [route_from_xyz(link["points"]) for link in (inner, outer)]
    station = np.linspace(0., inner_route[-1, 0], 90)
    blend_end_m = float(dashed_m) - .6
    fraction = np.clip(station / blend_end_m, 0., 1.)
    weight = fraction**3 * (10. + fraction * (-15. + 6. * fraction))
    inner_xyz = np.column_stack([np.interp(station, inner_route[:, 0], inner_route[:, k]) for k in (1, 2, 3)])
    outer_s = station * outer_route[-1, 0] / inner_route[-1, 0]
    outer_xyz = np.column_stack([np.interp(outer_s, outer_route[:, 0], outer_route[:, k]) for k in (1, 2, 3)])
    lane_change = inner_xyz + weight[:, None] * (outer_xyz - inner_xyz)

    remaining = route_from_xyz(last["points"])
    remaining = interval(remaining, remaining[-1, 0] - original["endpoint_basis"]["end_before_next_junction_m"], remaining[-1, 0])
    if not np.allclose(remaining[0, 1:], prefix[-1, 1:], rtol=0, atol=1e-6):
        raise ValueError("extended geometry does not meet the original endpoint")
    exit_route = route_from_xyz(exit_lane["points"])
    tail = route_from_xyz(np.vstack((remaining[:, 1:], left["points"], lane_change, right["points"],
                                    interval(exit_route, 0., 25.)[:, 1:])))
    tail[:, 0] += prefix[-1, 0]
    route = np.vstack((prefix, tail[1:]))
    added_commands = [navigation(route, left, "left"), navigation(route, right, "right")]
    if added_commands[0]["measured_turn_deg"] < 60. or added_commands[1]["measured_turn_deg"] > -60.:
        raise ValueError("extension commands disagree with the map geometry")
    # The new LEFT input begins before the old endpoint, at the same approach distance as other turns.
    assessment_start = added_commands[0]["entry_s_m"] - added_commands[0]["approach_m"]
    overlap = {split: distances(route, assessment_start, route[-1, 0], coordinates(path))
               for split, path in data_paths.items()}

    output = BASE / "routes/test/EAST_OUTER_EXTENDED.csv"
    write_route(output, route)
    manifest["routes"]["EAST_OUTER_EXTENDED"] = {
        "name_ko": "동쪽 바깥길에서 추가 좌회전·우회전", "split": "test",
        "evaluation_scope": "extended_route_diagnostic_with_training_proximity_reported",
        "route_csv": str(output.relative_to(BASE)), "route_sha256": sha256(output),
        "length_m": float(route[-1, 0]), "start_xyz_m": original["start_xyz_m"],
        "end_xyz_m": route[-1, 1:].tolist(), "initial_yaw_deg_map": original["initial_yaw_deg_map"],
        "navigation": original["navigation"] + added_commands,
        "prefix_course": "EAST_OUTER", "prefix_sha256": original["route_sha256"],
        "original_endpoint_s_m": float(prefix[-1, 0]),
        "source_links": original["source_links"] + [link["idx"] for link in (left, inner, outer, right, exit_lane)],
        "lane_change": {"from": inner["idx"], "to": outer["idx"], "dashed_divider": marking_id,
                        "blend_end_m": blend_end_m},
        "assessment_start_s_m": assessment_start, "extension_proximity": overlap,
        "data_sha256": {split: sha256(path) for split, path in data_paths.items()},
    }
    # If autonomous driving stops at the new left, the right-turn approach can be tested from rest on its own.
    right_start = added_commands[1]["entry_s_m"] - added_commands[1]["approach_m"]
    right_only = interval(route, right_start, route[-1, 0])
    right_heading = np.degrees(np.arctan2(*(right_only[1, 1:3] - right_only[0, 1:3])[::-1]))
    right_command = navigation(right_only, right, "right")
    right_output = BASE / "routes/test/EAST_OUTER_EXTENDED_RIGHT.csv"
    write_route(right_output, right_only)
    manifest["routes"]["EAST_OUTER_EXTENDED_RIGHT"] = {
        "name_ko": "연장 코스의 추가 우회전 단독 시험", "split": "test",
        "evaluation_scope": "right_turn_from_rest_on_extended_route_with_training_proximity",
        "route_csv": str(right_output.relative_to(BASE)), "route_sha256": sha256(right_output),
        "length_m": float(right_only[-1, 0]), "start_xyz_m": right_only[0, 1:].tolist(),
        "end_xyz_m": right_only[-1, 1:].tolist(), "initial_yaw_deg_map": float(right_heading),
        "navigation": [right_command], "source_course": "EAST_OUTER_EXTENDED",
        "source_course_start_s_m": right_start,
        "extension_proximity": {split: distances(right_only, 0., right_only[-1, 0], coordinates(path))
                                for split, path in data_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["routes"]["EAST_OUTER_EXTENDED"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
