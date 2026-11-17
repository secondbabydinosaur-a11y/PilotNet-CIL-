#!/usr/bin/env python3
"""Prepare reviewed raw runs without defining or training a neural network.

Primary target: curvature_paper_1pm (left negative/right positive).
Center labels use measured front-wheel steering and a kinematic bicycle model.
Optional side labels use our explicit PP-delta approximation, NOT an unpublished
NVIDIA formula. Raw images/labels remain untouched; validation/test are center only.
"""
import argparse
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from data_utils import BASE, EgoTimeline, HumanCommands, ExpertCommands, EXPERT_KIND, preprocess, read_csv, read_json, write_json
from ..collection.inspect_session import inspect
from ..collection.manual_control import CONTROL_KIND

INDEX_FIELDS = ["session_id", "group", "course", "camera", "stamp_ns", "source_path",
                "array_path", "array_index", "speed_kph", "steer_raw_deg", "steer_left_rad",
                "curvature_paper_1pm", "label_method", "route_s_m"]


def validate_splits(splits):
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("split JSON must contain only train/val/test group lists")
    used = set()
    for name, groups in splits.items():
        if not isinstance(groups, list) or not groups or not all(isinstance(g, str) for g in groups):
            raise ValueError(name + ": nonempty group list required")
        if used.intersection(groups) or len(groups) != len(set(groups)):
            raise ValueError("a group appears more than once: train/val/test leakage")
        used.update(groups)
    return {group: name for name, groups in splits.items() for group in groups}


def pp_curvature(status, future, shift_m=0):
    yaw = math.radians(status["yaw_deg"])
    # Shift the hypothetical rear axle in +local Y, matching the translated camera.
    dx = future["x_m"] - (status["x_m"] - math.sin(yaw) * shift_m)
    dy = future["y_m"] - (status["y_m"] + math.cos(yaw) * shift_m)
    x = math.cos(yaw) * dx + math.sin(yaw) * dy
    y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if x <= 1.0 or x * x + y * y < 4.0:
        return None
    return 2 * y / (x * x + y * y)


def side_curvature_paper(status, future, shift_m, measured_left_curvature):
    real = pp_curvature(status, future)
    virtual = pp_curvature(status, future, shift_m)
    if real is None or virtual is None:
        return None
    # Delta preserves the center driver's command when camera shift is zero.
    return -(measured_left_curvature + virtual - real)


def course_match(status, reference):
    distance2 = (reference[:, 1] - status["x_m"]) ** 2 + (reference[:, 2] - status["y_m"]) ** 2
    i = int(np.argmin(distance2))
    heading_error = abs((status["yaw_deg"] - reference[i, 3] + 180) % 360 - 180)
    if distance2[i] > 2.0 ** 2 or heading_error > 45:
        return None
    return float(reference[i, 0])


def discover(root):
    return [(path.parent, read_json(path)) for path in sorted(Path(root).glob("*/session.json"))]


def export(root, split_file, output, steer_left_sign, side_labels=False):
    splits = read_json(split_file)
    mapping = validate_splits(splits)
    all_sessions = discover(root)
    sessions = [(path, meta) for path, meta in all_sessions
                if meta["course"] != "calibration" and meta["group"] in mapping]
    for group in mapping:
        if not any(meta["group"] == group for _, meta in sessions):
            raise ValueError("no non-calibration runs for group " + group)
    # Check every source before creating any output. No partial success on bad input.
    reports = []
    for path, _ in sessions:
        report = inspect(path, decode_all=True)
        if not report["data_checks_passed"]:
            raise ValueError("{}: {}".format(path.name, "; ".join(report["errors"])))
        reports.append(report)
    configs = [meta["config"] for _, meta in sessions]
    reference_cfg = configs[0]
    for cfg in configs[1:]:
        for key in ("camera_common", "cameras", "vehicle", "preprocessing", "experiment", "map_name"):
            if cfg[key] != reference_cfg[key]:
                raise ValueError("mixed sensor/experiment settings: " + key)
    ref_rows = read_csv(BASE / "assets/course_reference.csv")
    references = {course: np.array([[float(r[k]) for k in ("route_s_m", "x_m", "y_m", "yaw_deg")]
                                   for r in ref_rows if r["course"] == course]) for course in ("A", "D")}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "arrays").mkdir()
    source_ids = []
    stats, discarded = Counter(), Counter()
    manifest = {
        "schema_version": 1, "status": "building", "splits": splits,
        "steer_left_sign": steer_left_sign, "front_wheel_degrees_operator_confirmed": True,
        "driving_and_crop_operator_reviewed": True, "config": reference_cfg,
        "preprocessing": "BGR JPEG -> configured crop/resize -> OpenCV YUV uint8 HWC; no normalization",
        "target": "curvature_paper_1pm=-tan(left_positive_front_wheel_radians)/wheelbase_m",
        "side_labels": "our PP delta at expert pose t+2s" if side_labels else "disabled (center baseline)",
        "side_labels_are_original_paper_formula": False,
        "validation_and_test": "center camera only; no balancing/augmentation",
        "filters": {"gear": 4, "ctrl_mode": "1=native manual; 2=explicitly identified logged human controller or classical expert", "course_distance_max_m": 2.0,
                    "course_heading_error_max_deg": 45.0, "speed_kph": [reference_cfg["experiment"]["min_speed_kph"], reference_cfg["experiment"]["max_speed_kph"]]},
        "source_reports": reports,
    }
    write_json(output / "dataset.json", manifest)
    streams = {}
    try:
        writers = {}
        for split in ("train", "val", "test"):
            streams[split] = (output / (split + ".csv")).open("x", newline="", encoding="utf-8")
            writers[split] = csv.DictWriter(streams[split], fieldnames=INDEX_FIELDS)
            writers[split].writeheader()
        for path, meta in sessions:
            cfg = meta["config"]
            course = meta["course"]
            if course not in references:
                raise ValueError("no reference for course " + course)
            split = mapping[meta["group"]]
            timeline = EgoTimeline(read_csv(path / "vehicle.csv"))
            assisted = meta.get("control", {}).get("kind") == CONTROL_KIND
            expert = meta.get("control", {}).get("kind") == EXPERT_KIND
            commands = (ExpertCommands if expert else HumanCommands)(read_csv(path / "commands.csv")) if assisted or expert else None
            frames = read_csv(path / "frames.csv")
            max_delta = int(cfg["quality"]["max_pair_delta_ms"] * 1e6)
            max_bracket = int(cfg["quality"]["max_interpolation_gap_ms"] * 1e6) if "max_interpolation_gap_ms" in cfg["quality"] else None
            period_ns = round(1e9 / cfg["preprocessing"]["sample_hz"])
            origin = min(int(row["stamp_ns"]) for row in frames)
            selected = []
            for camera in ("center", "left", "right"):
                if camera != "center" and (not side_labels or split != "train"):
                    continue
                last_bin = None
                for row in sorted((r for r in frames if r["camera"] == camera), key=lambda r: int(r["stamp_ns"])):
                    stamp = int(row["stamp_ns"])
                    time_bin = (stamp - origin) // period_ns
                    if time_bin == last_bin:
                        continue
                    last_bin = time_bin
                    status = timeline.at(stamp, max_delta, max_bracket)
                    if status is None:
                        discarded["no_ego_match"] += 1
                        continue
                    authorized_mode = (status["ctrl_mode"] == 1 or (status["ctrl_mode"] == 2 and (assisted or expert)
                                  and commands.valid_at(int(row["recv_ns"]))))
                    if status["gear"] != 4 or not authorized_mode:
                        discarded["not_documented_forward_drive"] += 1
                        continue
                    exp = cfg["experiment"]
                    if not exp["min_speed_kph"] <= status["speed_kph"] <= exp["max_speed_kph"]:
                        discarded["speed_outside_range"] += 1
                        continue
                    station = course_match(status, references[course])
                    if station is None:
                        discarded["outside_course_or_heading"] += 1
                        continue
                    delta_left = math.radians(steer_left_sign * status["steer_raw_deg"])
                    curvature_left = math.tan(delta_left) / cfg["vehicle"]["wheelbase_m"]
                    target = -curvature_left
                    method = "measured_front_wheel_bicycle"
                    if camera != "center":
                        future = timeline.at(stamp + 2_000_000_000, max_delta, max_bracket)
                        if future is None or future["gear"] != 4 or future["speed_kph"] < exp["min_speed_kph"]:
                            discarded["side_missing_2s_context"] += 1
                            continue
                        shift = cfg["cameras"][camera]["xyz_m"][1] - cfg["cameras"]["center"]["xyz_m"][1]
                        target = side_curvature_paper(status, future, shift, curvature_left)
                        if target is None:
                            discarded["side_invalid_future_geometry"] += 1
                            continue
                        method = "our_pp_delta_2s"
                    selected.append((row, status, station, delta_left, target, method))
            if not selected:
                raise ValueError("{}: no eligible samples; check pose, gear, mode, speed and course".format(path.name))
            relative_array = "arrays/{}.npy".format(meta["session_id"])
            settings = cfg["preprocessing"]
            array = np.lib.format.open_memmap(output / relative_array, mode="w+", dtype=np.uint8,
                        shape=(len(selected), settings["height"], settings["width"], 3))
            for i, (row, status, station, delta_left, target, method) in enumerate(selected):
                image = cv2.imread(str(path / row["path"]))
                if image is None:
                    raise ValueError("unreadable image " + row["path"])
                array[i] = preprocess(image, settings)
                values = [meta["session_id"], meta["group"], course, row["camera"], row["stamp_ns"],
                          str(path / row["path"]), relative_array, i, status["speed_kph"], status["steer_raw_deg"],
                          delta_left, target, method, station]
                writers[split].writerow(dict(zip(INDEX_FIELDS, values)))
                stats[split + "/" + row["camera"]] += 1
            array.flush()
            del array
            source_ids.append({"session_id": meta["session_id"], "path": str(path),
                               "control": meta.get("control"),
                               "session_sha256": hashlib.sha256((path / "session.json").read_bytes()).hexdigest(),
                               "frames_sha256": hashlib.sha256((path / "frames.csv").read_bytes()).hexdigest(),
                               "vehicle_sha256": hashlib.sha256((path / "vehicle.csv").read_bytes()).hexdigest(),
                               "commands_sha256": hashlib.sha256((path / "commands.csv").read_bytes()).hexdigest() if (path / "commands.csv").exists() else None})
            print("{} -> {}: {} samples".format(path.name, split, len(selected)), flush=True)
        for split in splits:
            if sum(count for key, count in stats.items() if key.startswith(split + "/")) == 0:
                raise ValueError("empty split: " + split)
        manifest.update(status="prepared", counts=dict(stats), discarded=dict(discarded), sources=source_ids)
    except Exception as exc:
        manifest.update(status="build_error", error=str(exc))
        raise
    finally:
        for stream in streams.values():
            stream.close()
        write_json(output / "dataset.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=BASE / "dataset/raw")
    parser.add_argument("--write-split-template", type=Path, help="write a group split for review; do not export")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steer-left-sign", type=int, choices=[-1, 1], help="+1 if raw steer positive on a left turn; else -1")
    parser.add_argument("--confirm-front-wheel-degrees", action="store_true")
    parser.add_argument("--confirm-reviewed-driving-and-crop", action="store_true")
    parser.add_argument("--side-labels", choices=["none", "pp-delta-2s"], default="none")
    args = parser.parse_args()
    if args.write_split_template:
        groups = sorted({meta["group"] for _, meta in discover(args.raw_root)
                         if meta["course"] != "calibration" and not meta["group"].startswith("trial")})
        if len(groups) < 3:
            parser.error("need >=3 independent non-trial groups; collect g01..g06 first")
        if args.write_split_template.exists():
            parser.error("split template already exists; review it instead of overwriting")
        splits = {"train": groups[:-2], "val": groups[-2:-1], "test": groups[-1:]}
        write_json(args.write_split_template, splits)
        print("Review group independence and A/D coverage: " + str(args.write_split_template))
        return 0
    if not all((args.split_file, args.output, args.steer_left_sign,
                args.confirm_front_wheel_degrees, args.confirm_reviewed_driving_and_crop)):
        parser.error("export requires --split-file, --output, --steer-left-sign and both confirmation flags; see README")
    result = export(args.raw_root, args.split_file, args.output, args.steer_left_sign,
                    args.side_labels == "pp-delta-2s")
    print("Prepared: " + str(result["counts"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
