"""Choose Sangam routes from the directed MGeo graph before training."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_evaluation_routes import BASE, interval, navigation, pose_heading, route_from_xyz, sha256, write_route


GUARD_M = 30.0


def heading_delta(points):
    approach = points[1, 1:3] - points[0, 1:3]
    exit_direction = points[-1, 1:3] - points[-2, 1:3]
    a, b = (math.atan2(y, x) for x, y in (approach, exit_direction))
    return (math.degrees(b - a) + 180) % 360 - 180


def bend_strength(route):
    """Return heading change over a 5 m road segment, smoothed at its endpoints."""
    centres = np.arange(1.5, route[-1, 0] - 1.5, 0.5)
    if len(centres) < 11:
        return 0.0
    tangents = [np.interp(centres + 1.5, route[:, 0], route[:, i])
                - np.interp(centres - 1.5, route[:, 0], route[:, i]) for i in (1, 2)]
    heading = np.unwrap(np.arctan2(tangents[1], tangents[0]))
    return float(np.degrees(np.max(np.abs(heading[10:] - heading[:-10]))))


def end_directions(route):
    length = route[-1, 0]
    xy = lambda station: np.array([np.interp(station, route[:, 0], route[:, k]) for k in (1, 2)])
    directions = (xy(3.) - xy(0.), xy(length) - xy(length - 3.))
    return tuple(direction / np.linalg.norm(direction) for direction in directions)


def read_graph(path):
    links, outgoing = {}, defaultdict(list)
    for link in json.loads(path.read_text()):
        if (link["lazy_init"] or str(link["link_type"]) not in ("1", "6")
                or str(link["road_type"]) not in ("1", "None")
                or "uturn" in (link.get("related_signal") or "")):
            continue
        route = route_from_xyz(link["points"])
        if len(route) < 2 or route[-1, 0] < 3:
            continue
        bend = bend_strength(route)
        if bend > 52:
            continue
        junction = str(link["link_type"]) == "1"
        turn = heading_delta(route)
        if junction and abs(turn) > 130:
            continue
        command = "follow_road"
        if junction:
            signal = link.get("related_signal") or ""
            if signal.startswith("left") and turn > 28.:
                command = "left"
            elif signal.startswith("right") and turn < -28.:
                command = "right"
            elif signal == "straight" and abs(turn) < 32. and bend < 18.:
                command = "straight"
            else:
                # The map labels unmarked circular junction arcs as straight.
                continue
        ident = link["idx"]
        begin, end = end_directions(route)
        links[ident] = {"link": link, "route": route, "bend": bend, "command": command,
                        "begin_direction": begin, "end_direction": end}
        outgoing[link["from_node_idx"]].append(ident)
    return links, outgoing


def sample_candidate(links, outgoing, rng, possible_starts, *, goal_range=(205., 385.), maximum_length=525.):
    source = links[rng.choice(possible_starts)]
    keys = [source["link"]["idx"]]
    first = source["route"]
    start = max(1.0, first[-1, 0] - rng.uniform(48., 82.))
    segments = [interval(first, start, first[-1, 0])]
    length = segments[0][-1, 0]
    goal = rng.uniform(*goal_range)
    used = {source["link"]["from_node_idx"], source["link"]["to_node_idx"]}
    exit_node = source["link"]["to_node_idx"]
    turns = []
    for _ in range(17):
        previous_end = segments[-1][-1, 1:3]
        choices = [ident for ident in outgoing[exit_node]
                   if links[ident]["link"]["to_node_idx"] not in used
                   and links[keys[-1]]["end_direction"] @ links[ident]["begin_direction"] > math.cos(math.radians(27.))
                   and np.linalg.norm(links[ident]["route"][0, 1:3] - previous_end) < 0.55]
        if not choices:
            return None
        weights = [3 if links[ident]["command"] in ("left", "right") else 1 for ident in choices]
        ident = rng.choices(choices, weights, k=1)[0]
        entry = links[ident]
        segment, command = entry["route"], entry["command"]
        if command == "follow_road" and turns and length + segment[-1, 0] >= goal:
            end = max(24.0, goal - length)
            if segment[-1, 0] >= end:
                segments.append(interval(segment, 0.0, end))
                keys.append(ident)
                break
        if length + segment[-1, 0] > maximum_length:
            return None
        if command in ("left", "right"):
            turns.append(command)
        segments.append(segment)
        keys.append(ident)
        length += segment[-1, 0]
        exit_node = entry["link"]["to_node_idx"]
        used.add(exit_node)
    else:
        return None

    points = route_from_xyz(np.concatenate([run[:, 1:] for run in segments]))
    if bend_strength(points) > 55.:
        return None
    commands = []
    for ident in keys:
        entry = links[ident]
        if entry["command"] != "follow_road":
            item = navigation(points, entry["link"], entry["command"])
            item["max_heading_change_5m_deg"] = round(entry["bend"], 2)
            commands.append(item)
    if any(item["entry_s_m"] < 22. or item["exit_s_m"] - item["entry_s_m"] < 2. for item in commands):
        return None
    counts = Counter(item["command"] for item in commands)
    return dict(points=points, keys=keys, commands=commands, counts=counts,
                right_bends=[item["max_heading_change_5m_deg"] for item in commands if item["command"] == "right"])


def separated(candidate, others, minimum):
    xy = candidate["points"][:, 1:3]
    return all(float(tree.query(xy)[0].min()) >= minimum for tree in others)


def pick_routes(candidates):
    routes, trees = {}, {}

    def choose(name, split, condition, key):
        eligible = [candidate for candidate in candidates if condition(candidate)
                    and separated(candidate, [tree for prior, tree in trees.items()
                                              if routes[prior][0] != split], GUARD_M)
                    and separated(candidate, [tree for prior, tree in trees.items()
                                              if routes[prior][0] == split], 18. if split != "train" else 6.)]
        if not eligible:
            raise RuntimeError(f"No geographically separated route for {name}")
        best = max(eligible, key=key)
        routes[name] = (split, best)
        trees[name] = cKDTree(best["points"][:, 1:3])

    def compact(candidate):
        return -abs(candidate["points"][-1, 0] - 280.)

    choose("test_two_rights", "test", lambda c: len(c["right_bends"]) >= 2,
           lambda c: max(c["right_bends"]) - min(c["right_bends"]) + compact(c) / 20)
    choose("test_left_right", "test", lambda c: all(c["counts"][i] for i in ("left", "right", "straight")), compact)
    choose("test_sharp_right", "test", lambda c: c["counts"]["right"] and c["counts"]["straight"],
           lambda c: max(c["right_bends"]) + compact(c) / 12)
    choose("val_mixed", "validation", lambda c: all(c["counts"][i] for i in ("left", "right", "straight")), compact)
    choose("val_rights", "validation", lambda c: c["counts"]["right"] >= 2, compact)
    choose("val_left", "validation", lambda c: c["counts"]["left"] >= 2, compact)
    for index in range(12):
        primary = ("right", "left", "straight")[index % 3]
        choose(f"train_{index + 1:02}", "train", lambda c: c["counts"][primary] and sum(c["counts"].values()) >= 2,
               lambda c: min(3, c["counts"][primary]) * 4 + len(c["counts"]) * 3 + compact(c) / 25)
    return routes, trees


def main():
    import random

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="Sangam link_set.json")
    parser.add_argument("--output", type=Path, required=True, help="new route directory under cil/")
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    output.relative_to(BASE)
    rng = random.Random(7)
    links, outgoing = read_graph(args.map)
    starts = [key for key, entry in links.items() if entry["command"] == "follow_road" and entry["route"][-1, 0] >= 50.]
    candidates = [candidate for _ in range(8500) for candidate in [sample_candidate(links, outgoing, rng, starts)]
                  if candidate]
    routes, trees = pick_routes(candidates)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"map": "R_KR_PR_Sangam_NoBuildings", "link_set_sha256": sha256(args.map),
                "route_selection_seed": 7, "reserved_before_training": True, "route_guard_m": GUARD_M,
                "junction_filter": "signed turn agrees with the marked movement; sharp unmarked arcs excluded",
                "test_reservations": {}, "routes": {}}
    for name, (split, candidate) in routes.items():
        path = output / (name + ".csv")
        points = candidate["points"]
        write_route(path, points)
        checks = {prior: float(tree.query(points[:, 1:3])[0].min()) for prior, tree in trees.items()
                  if routes[prior][0] != split}
        manifest["routes"][name] = {
            "split": split, "route_csv": str(path.relative_to(BASE)), "route_sha256": sha256(path),
            "start_xyz_m": points[0, 1:].tolist(), "end_xyz_m": points[-1, 1:].tolist(),
            "initial_yaw_deg_map": pose_heading(points), "length_m": float(points[-1, 0]),
            "navigation": candidate["commands"], "source_links": candidate["keys"],
            "other_split_min_distances_m": checks,
            "evaluation_scope": "geographically held-out Sangam roads" if split == "test" else split,
        }
        print(name, split, round(points[-1, 0]), dict(candidate["counts"]), candidate["right_bends"], flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
