#!/usr/bin/env python3
"""Pair CENTER JPEG paths with steering-derived curvature. Never process images.

Reads the fixed dataset/splits lists. No camera resampling, image decoding,
crop, resize, color conversion, augmentation, or model/training work occurs.
Side-camera labels are deferred until their recovery correction is chosen.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from data_utils import (BASE, EXPERT_KIND, EgoTimeline, ExpertCommands, HumanCommands,
                        read_csv, read_json, steering_to_curvature, write_json)
from ..collection.manual_control import CONTROL_KIND

FIELDS = ["image_path", "y_curvature_1pm", "steer_raw_deg", "steer_raw_rad",
          "steer_equivalent_left_rad", "wheelbase_m", "speed_kph", "speed_eligible",
          "session_id", "group", "course", "camera", "frame_id", "stamp_ns", "vehicle_csv",
          "match_method", "before_stamp_ns", "after_stamp_ns", "before_steer_deg",
          "after_steer_deg", "interpolation_weight", "nearest_delta_ms", "bracket_gap_ms"]
REJECT_FIELDS = ["split", "session_id", "camera", "frame_id", "stamp_ns", "image_path", "reason"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(root, value):
    path = (root / value).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("missing source or path outside project: " + str(path))
    return path


def matched_status(timeline, stamp, quality):
    """Return interpolated state plus auditable native-clock bracket details."""
    index = bisect_left(timeline.stamps, stamp)
    exact = index < len(timeline.stamps) and timeline.stamps[index] == stamp
    if not exact and (index == 0 or index == len(timeline.stamps)):
        return None, None, "outside_ego_time_range"
    max_delta = round(quality["max_pair_delta_ms"] * 1e6)
    bracket = round(quality["max_interpolation_gap_ms"] * 1e6)
    status = timeline.at(stamp, max_delta, bracket)
    if status is None:
        return None, None, "invalid_time_gap_or_mode_transition"
    a = timeline.numeric(timeline.rows[index if exact else index - 1])
    b = timeline.numeric(timeline.rows[index])
    before, after = a["stamp_ns"], b["stamp_ns"]
    weight = 0. if exact else (stamp - before) / (after - before)
    details = {"match_method": "exact" if exact else "linear_interpolation",
        "before_stamp_ns": before, "after_stamp_ns": after,
        "before_steer_deg": a["steer_raw_deg"], "after_steer_deg": b["steer_raw_deg"],
        "interpolation_weight": weight,
        "nearest_delta_ms": min(stamp - before, after - stamp) / 1e6,
        "bracket_gap_ms": (after - before) / 1e6}
    return status, details, None


def build(split_dir, output, project_root=BASE, steer_left_sign=1, road_wheel_gain=1.):
    root = Path(project_root).resolve()
    split_dir, output = Path(split_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; choose a new directory: " + str(output))
    # Validate the model even if an input happens to contain no eligible images.
    steering_to_curvature(0., 3., steer_left_sign, road_wheel_gain)
    groups = read_json(split_dir / "groups.json")
    if set(groups) != {"train", "val", "test"}:
        raise ValueError("groups.json must contain train, val, test")
    membership = {}
    for split, names in groups.items():
        if not isinstance(names, list) or not names:
            raise ValueError("nonempty group list required: " + split)
        for group in names:
            if group in membership:
                raise ValueError("group occurs in more than one assignment: " + group)
            membership[group] = split
    indexes = {s: read_csv(split_dir / (s + ".csv")) for s in ("train", "val", "test")}
    sessions, image_splits, session_splits = {}, {}, {}
    snapshots = {}
    for name in ("groups.json", "train.csv", "val.csv", "test.csv"):
        p = split_dir / name
        snapshots[p] = sha256(p)
    # Preflight all listed paths and source metadata before creating outputs.
    for split, rows in indexes.items():
        if not rows:
            raise ValueError("empty raw split: " + split)
        for row in rows:
            sid = row["session_id"]
            if membership.get(row["group"]) != split:
                raise ValueError("group assignment differs from split index")
            image = project_path(root, row["image_path"])
            if image in image_splits:
                raise ValueError("repeated image file in split lists: " + str(image))
            image_splits[image] = split
            if session_splits.setdefault(sid, split) != split:
                raise ValueError("session occurs in multiple splits: " + sid)
            if row["camera"] not in ("left", "center", "right"):
                raise ValueError("unknown camera role")
            vehicle = project_path(root, row["vehicle_csv"])
            session_dir = vehicle.parent
            if image.parent.parent.parent != session_dir or session_dir.name != sid:
                raise ValueError("image and vehicle CSV belong to different sessions")
            if sid in sessions:
                meta = sessions[sid]["meta"]
                if vehicle != sessions[sid]["vehicle"]:
                    raise ValueError("mixed vehicle sources for a session")
            else:
                meta = read_json(session_dir / "session.json")
                report = read_json(session_dir / "quality_report.json")
                if meta["status"] != "capture_finished" or not report["data_checks_passed"]:
                    raise ValueError("source capture did not pass inspection: " + sid)
                if meta["course"] == "calibration":
                    raise ValueError("calibration cannot be used for labels")
                kind = meta.get("control", {}).get("kind")
                if kind not in (EXPERT_KIND, CONTROL_KIND, "native_manual"):
                    raise ValueError("unsupported control provenance: " + str(kind))
                if kind == EXPERT_KIND and meta["control"].get("human_steering") is not False:
                    raise ValueError("automatic teacher must not claim human driving")
                commands_path = session_dir / "commands.csv"
                commands = None
                if kind in (EXPERT_KIND, CONTROL_KIND):
                    commands = (ExpertCommands if kind == EXPERT_KIND else HumanCommands)(read_csv(commands_path))
                frame_rows = read_csv(session_dir / "frames.csv")
                original_frames = {}
                for original in frame_rows:
                    key = (original["camera"], original["frame_id"])
                    if key in original_frames:
                        raise ValueError("duplicate frame identity in raw capture")
                    original_frames[key] = original
                sessions[sid] = {"meta": meta, "vehicle": vehicle,
                    "timeline": EgoTimeline(read_csv(vehicle)), "commands": commands,
                    "original_frames": original_frames}
                for name in ("session.json", "frames.csv", "vehicle.csv", "commands.csv", "quality_report.json"):
                    p = session_dir / name
                    if p.exists():
                        snapshots[p] = sha256(p)
            if meta["group"] != row["group"] or meta["course"] != row["course"]:
                raise ValueError("split row differs from source metadata")
            original = sessions[sid]["original_frames"].get((row["camera"], row["frame_id"]))
            if (original is None or original["stamp_ns"] != row["stamp_ns"]
                    or original["recv_ns"] != row["recv_ns"]
                    or (session_dir / original["path"]).resolve() != image):
                raise ValueError("split row differs from original image/time metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".labels-staging-", dir=str(output.parent)))
    stats = {s: {"input_images_all_cameras": len(rows), "center_images": 0,
                  "paired": 0, "speed_eligible": 0, "match_methods": Counter(),
                  "rejected_center": Counter(), "side_images_deferred": 0}
             for s, rows in indexes.items()}
    ratios, curvature_errors = [], []
    try:
        with (stage / "rejected_center.csv").open("x", newline="") as rejected_file:
            rejected = csv.DictWriter(rejected_file, fieldnames=REJECT_FIELDS)
            rejected.writeheader()
            for split, rows in indexes.items():
                with (stage / (split + ".csv")).open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    for row in rows:
                        if row["camera"] != "center":
                            stats[split]["side_images_deferred"] += 1
                            continue
                        stats[split]["center_images"] += 1
                        source = sessions[row["session_id"]]
                        meta = source["meta"]
                        status, match, reason = matched_status(source["timeline"], int(row["stamp_ns"]), meta["config"]["quality"])
                        if status is not None:
                            kind = meta["control"]["kind"]
                            mode_ok = (status["ctrl_mode"] == 1 and kind == "native_manual") or (
                                status["ctrl_mode"] == 2 and source["commands"] is not None
                                and source["commands"].valid_at(int(row["recv_ns"])))
                            if status["gear"] != 4 or not mode_ok:
                                reason = "not_documented_forward_drive"
                        if reason:
                            stats[split]["rejected_center"][reason] += 1
                            rejected.writerow({**{k: row[k] for k in REJECT_FIELDS if k not in ("split", "reason")},
                                               "split": split, "reason": reason})
                            continue
                        wheelbase = float(meta["config"]["vehicle"]["wheelbase_m"])
                        angle = status["steer_raw_deg"]
                        target = steering_to_curvature(angle, wheelbase, steer_left_sign, road_wheel_gain)
                        exp = meta["config"]["experiment"]
                        speed_ok = exp["min_speed_kph"] <= status["speed_kph"] <= exp["max_speed_kph"]
                        result = {k: row[k] for k in ("image_path", "session_id", "group", "course", "camera", "frame_id", "stamp_ns", "vehicle_csv")}
                        result.update(match)
                        result.update(y_curvature_1pm=target, steer_raw_deg=angle,
                            steer_raw_rad=math.radians(angle),
                            steer_equivalent_left_rad=math.radians(angle * steer_left_sign * road_wheel_gain),
                            wheelbase_m=wheelbase, speed_kph=status["speed_kph"], speed_eligible=int(speed_ok))
                        writer.writerow(result)
                        stats[split]["paired"] += 1
                        stats[split]["speed_eligible"] += int(speed_ok)
                        stats[split]["match_methods"][match["match_method"]] += 1
                        # Diagnostic only, using TRAIN sources; never fits label parameters.
                        speed = status["speed_kph"] / 3.6
                        if split == "train" and speed > 4 and abs(angle) > 1.5 and abs(status["yaw_rate_deg_s"]) > 1:
                            motion = -math.radians(status["yaw_rate_deg_s"]) / speed
                            ratios.append(motion / target)
                            curvature_errors.append(motion - target)
                if not stats[split]["paired"]:
                    raise ValueError("no valid center pairs in split: " + split)
        for path, digest in snapshots.items():
            if sha256(path) != digest:
                raise ValueError("source changed while labeling: " + str(path))
        report = {"schema_version": 1, "status": "paired_center_labels_created",
            "path_base": str(root), "input_split_directory": str(split_dir), "groups": groups,
            "input": "one original center JPEG, referenced by image_path",
            "target_column": "y_curvature_1pm", "target_unit": "1/m",
            "target_sign": "left negative, right positive",
            "model": {"kind": "kinematic_bicycle_reported_steering",
                "formula": "-tan(radians(steer_raw_deg * steer_left_sign * road_wheel_gain)) / wheelbase_m",
                "steer_left_sign": steer_left_sign, "road_wheel_gain": road_wheel_gain,
                "wheelbase_source": "each source session config.vehicle.wheelbase_m",
                "assumption": "reported steering * road_wheel_gain is treated as equivalent front-wheel angle",
                "physical_trajectory_curvature_validated": False},
            "clock": "native integer stamp_ns; recv_ns used only to check causal command provenance",
            "matching": "exact or linear interpolation; source-session 25ms nearest / 50ms bracket limits; no extrapolation/mode-transition interpolation",
            "image_processing": "NONE: no decoding, crop, resize, YUV, normalization, or image copies",
            "frame_selection": "all center frames, no 10Hz resampling; invalid time/provenance pairs listed separately",
            "speed_policy": "start/stop frames retained; speed_eligible marks the source config min/max range for later selection",
            "side_cameras": "deferred; original split lists and images retained, no uncorrected side labels emitted",
            "source_control": "classical PP automatic teacher, not human demonstrations",
            "splits": stats,
            "train_motion_diagnostic": {"samples": len(ratios),
                "median_motion_to_label_curvature_ratio": float(np.median(ratios)) if ratios else None,
                "rmse_1pm": float(np.sqrt(np.mean(np.square(curvature_errors)))) if ratios else None,
                "note": "yaw-rate/speed comparison only; label sign/gain not fitted to these values. Dynamic response and steering model can differ."},
            "source_sha256": {str(path): digest for path, digest in snapshots.items()}}
        write_json(stage / "summary.json", report)
        stage.rename(output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=BASE / "dataset/splits")
    parser.add_argument("--output", type=Path, default=BASE / "dataset/paired")
    parser.add_argument("--steer-left-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--road-wheel-gain", type=float, default=1.)
    args = parser.parse_args()
    result = build(args.split_dir, args.output, steer_left_sign=args.steer_left_sign,
                   road_wheel_gain=args.road_wheel_gain)
    for split, values in result["splits"].items():
        print("{}: {} pairs, {} unmatched/invalid, {} speed-eligible".format(split,
              values["paired"], sum(values["rejected_center"].values()), values["speed_eligible"]))
    print("Created original-image paths + curvature labels: " + str(args.output))


if __name__ == "__main__":
    main()
