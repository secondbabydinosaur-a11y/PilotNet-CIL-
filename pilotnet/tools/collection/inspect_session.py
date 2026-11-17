#!/usr/bin/env python3
"""Inspect clocks, synchronization, missing files, JPEGs and steering/pose consistency."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from data_utils import BASE, EgoTimeline, HumanCommands, ExpertCommands, EXPERT_KIND, read_csv, read_json, write_json
from .manual_control import CONTROL_KIND


def stream_stats(rows):
    stamps = np.asarray([int(r["stamp_ns"]) for r in rows], dtype=np.int64)
    delta = np.diff(stamps) / 1e9
    ordered = np.unique(stamps)
    span = (int(ordered[-1]) - int(ordered[0])) / 1e9 if len(ordered) > 1 else 0
    return {
        "count": len(rows), "unique_stamps": len(ordered),
        "first_stamp_ns": int(ordered[0]) if len(ordered) else None,
        "last_stamp_ns": int(ordered[-1]) if len(ordered) else None,
        "span_s": span, "hz": (len(ordered) - 1) / span if span > 0 else 0,
        "max_gap_s": float(np.max(np.diff(ordered)) / 1e9) if len(ordered) > 1 else 0,
        "reordered_or_duplicate": int(np.sum(delta <= 0)),
        "clock_reset": bool(np.any(delta < -0.25)),
    }


def inspect(session, decode_all=False):
    session = Path(session).resolve()
    meta = read_json(session / "session.json")
    cfg = meta["config"]
    frames, ego = read_csv(session / "frames.csv"), read_csv(session / "vehicle.csv")
    timeline = EgoTimeline(ego)
    quality = cfg["quality"]
    errors, warnings, streams = [], [], {}
    if meta["status"] != "capture_finished":
        errors.append("session did not finish cleanly: " + meta["status"])
    if meta.get("fatal_errors"):
        errors.extend(meta["fatal_errors"])
    expert_report = None
    if meta.get("control", {}).get("kind") == EXPERT_KIND:
        if meta["control"].get("human_steering") is not False:
            errors.append("automatic expert must not claim human steering")
        telemetry_path = session / "expert.csv"
        command_path = session / "commands.csv"
        if not telemetry_path.is_file() or not command_path.is_file():
            errors.append("expert run is missing telemetry or commands")
        else:
            telemetry = read_csv(telemetry_path)
            commands = ExpertCommands(read_csv(command_path))
            if not telemetry or not any(commands.valid_at(t) for t in commands.stamps):
                errors.append("expert run has no fresh logged teacher commands")
            else:
                cte = np.abs([float(r["cte_m"]) for r in telemetry])
                speed = np.array([float(r["speed_kph"]) for r in telemetry])
                heading = np.abs([float(r["heading_error_deg"]) for r in telemetry])
                expert_report = {"samples": len(telemetry), "max_abs_cte_m": float(max(cte)),
                    "p95_abs_cte_m": float(np.percentile(cte, 95)),
                    "max_speed_kph": float(max(speed)), "max_heading_error_deg": float(max(heading))}
                if max(cte) > .8 or max(heading) > 25 or max(speed) > 20.3:
                    errors.append("expert tracking/speed outside collection bounds")
                fraction = sum(commands.valid_at(int(r["recv_ns"])) for r in frames) / max(1, len(frames))
                expert_report["command_coverage_fraction"] = fraction
                if fraction < .95:
                    errors.append("insufficient causal expert command coverage")
    if meta.get("control", {}).get("kind") == CONTROL_KIND:
        command_path = session / "commands.csv"
        if not command_path.is_file():
            errors.append("assisted human run is missing commands.csv")
        else:
            commands = HumanCommands(read_csv(command_path))
            if not any(commands.valid_at(t) for t in commands.stamps):
                errors.append("no focused, fresh human command samples; click Simulator and release W once")
    names = {r["camera"] for r in frames}
    if names - set(cfg["cameras"]):
        errors.append("unexpected camera roles in frames.csv")
    all_rows = {name: [r for r in frames if r["camera"] == name] for name in cfg["cameras"]}
    all_rows["ego"] = ego
    max_delta = int(quality["max_pair_delta_ms"] * 1e6)
    max_bracket = int(quality["max_interpolation_gap_ms"] * 1e6) if "max_interpolation_gap_ms" in quality else None
    for name, rows in all_rows.items():
        stats = stream_stats(rows)
        streams[name] = stats
        expected_hz = cfg["ego"]["hz"] if name == "ego" else cfg["camera_common"]["fps"]
        if stats["count"] < 2:
            errors.append(name + ": fewer than two samples")
        if stats["hz"] < expected_hz * quality["min_rate_fraction"]:
            errors.append(name + ": low receive rate")
        if stats["max_gap_s"] > quality["max_stream_gap_s"]:
            errors.append(name + ": long stream gap")
        if stats["clock_reset"]:
            errors.append(name + ": native clock moved backwards; start a new run after reset")
        elif stats["reordered_or_duplicate"]:
            warnings.append(name + ": duplicate/reordered timestamps")
        if meta.get("saved", {}).get(name) != len(rows):
            errors.append(name + ": saved count differs from CSV")
        if meta.get("received", {}).get(name) != meta.get("saved", {}).get(name):
            errors.append(name + ": received count differs from saved count")
        if name == "ego":
            continue
        matched = sum(timeline.at(int(row["stamp_ns"]), max_delta, max_bracket) is not None for row in rows)
        stats["matched"] = matched
        stats["match_fraction"] = matched / len(rows) if rows else 0
        if stats["match_fraction"] < quality["min_match_fraction"]:
            errors.append(name + ": insufficient camera/Ego timestamp matches")
        missing = [row for row in rows if not (session / row["path"]).is_file()]
        stats["missing_files"] = len(missing)
        if missing:
            errors.append(name + ": missing JPEG files")
        indexes = range(len(rows)) if decode_all else np.unique(np.linspace(0, max(0, len(rows) - 1), min(20, len(rows))).astype(int))
        bad, checked = 0, 0
        for index in indexes:
            path = session / rows[index]["path"]
            checked += 1
            frame = cv2.imread(str(path)) if path.is_file() else None
            if frame is None or frame.shape[:2] != (cfg["camera_common"]["height"], cfg["camera_common"]["width"]):
                bad += 1
        stats.update(decoded_checked=checked, bad_images=bad)
        if bad:
            errors.append(name + ": corrupt JPEG or unexpected resolution")
    if ego:
        ego_start, ego_end = min(timeline.stamps), max(timeline.stamps)
        for name in cfg["cameras"]:
            stats = streams[name]
            overlap = max(0, min(stats["last_stamp_ns"] or 0, ego_end) - max(stats["first_stamp_ns"] or 0, ego_start))
            fraction = overlap / max(1, ego_end - ego_start)
            stats["ego_span_coverage"] = fraction
            if fraction < .90:
                errors.append(name + ": camera covers less than 90% of Ego time span")
        ordered_ego = [timeline.numeric(row) for row in timeline.rows]
        speeds = [float(row["speed_kph"]) for row in ego]
        streams["ego"]["max_speed_kph"] = max(speeds)
        manual = cfg.get("manual_control", {})
        if manual.get("enabled"):
            limit = manual["speed_limit_kph"]
            streams["ego"]["above_speed_limit_samples"] = sum(v > limit for v in speeds)
            if max(speeds) > limit:
                warnings.append("measured max speed {:.2f}kph exceeded {}kph governor ceiling; retain actual values and inspect response".format(max(speeds), limit))
        for a, b in zip(ordered_ego, ordered_ego[1:]):
            dt = (b["stamp_ns"] - a["stamp_ns"]) / 1e9
            distance = np.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
            # Detect teleport as well as a decoder that mistakes fields for pose.
            max_distance = max(abs(a["speed_kph"]), abs(b["speed_kph"])) / 3.6 * dt + 2.0
            if distance > max_distance:
                errors.append("pose discontinuity/teleport: do not reposition within a run")
                break
        valid = [r for r in ordered_ego if abs(r["speed_kph"]) > 5 and abs(r["steer_raw_deg"]) > 1
                 and abs(r["yaw_rate_deg_s"]) > 1]
        if len(valid) >= 10:
            gains = [r["yaw_rate_deg_s"] / np.degrees(
                r["speed_kph"] / 3.6 / cfg["vehicle"]["wheelbase_m"] * np.tan(np.radians(r["steer_raw_deg"]))) for r in valid]
            steering_check = {"samples": len(valid), "median_yaw_to_bicycle_gain": float(np.median(gains)),
                              "note": "diagnostic only; HUD front-wheel degrees and sign must be confirmed manually"}
        else:
            steering_check = {"samples": len(valid), "note": "need moving left AND right turns for calibration"}
    else:
        steering_check = {"samples": 0, "note": "no Ego samples"}
    for name, count in meta.get("parse_errors", {}).items():
        if count:
            warnings.append("{}: {} packet parse errors; ordinary Ego raw packets retained".format(name, count))
    for name, count in meta.get("incomplete_evicted", {}).items():
        if count:
            warnings.append("{}: {} incomplete frames evicted".format(name, count))
    longest_span = max((s["span_s"] for s in streams.values()), default=0)
    if longest_span < .75 * meta.get("elapsed_wall_s", 0):
        warnings.append("streams cover less than 75% of wall recording time; check startup/stall/pause")
    return {"session": str(session), "session_id": meta["session_id"], "course": meta["course"],
            "group": meta["group"], "data_checks_passed": not errors, "decode_all": decode_all,
            "streams": streams, "errors": errors, "warnings": warnings,
            "steering_motion_check": steering_check, "expert_tracking": expert_report,
            "operator_review_still_required": True}


def play(session, camera):
    session = Path(session)
    cfg = read_json(session / "session.json")["config"]
    rows = sorted((r for r in read_csv(session / "frames.csv") if r["camera"] == camera), key=lambda r: int(r["stamp_ns"]))
    timeline = EgoTimeline(read_csv(session / "vehicle.csv"))
    limit = int(cfg["quality"]["max_pair_delta_ms"] * 1e6)
    bracket = int(cfg["quality"]["max_interpolation_gap_ms"] * 1e6) if "max_interpolation_gap_ms" in cfg["quality"] else None
    for row in rows:
        frame = cv2.imread(str(session / row["path"]))
        if frame is None:
            continue
        x0, y0, x1, y1 = cfg["preprocessing"]["crop_xyxy"]
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        status = timeline.at(int(row["stamp_ns"]), limit, bracket)
        label = "NO MATCH" if status is None else "v={:.1f}kph steer={:.2f}deg".format(status["speed_kph"], status["steer_raw_deg"])
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Inspection | green: proposed crop | space: pause, q: exit", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(round(1000 / cfg["camera_common"]["fps"])) & 0xff
        if key == ord(" "):
            key = cv2.waitKey(0) & 0xff
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def resolve_session(value):
    if value != "latest":
        return Path(value).resolve()
    choices = list((BASE / "dataset/raw").glob("*/session.json"))
    if not choices:
        raise ValueError("no sessions in dataset/raw")
    return max(choices, key=lambda p: p.stat().st_mtime).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session folder, or latest")
    parser.add_argument("--decode-all", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--camera", choices=["left", "center", "right"], default="center")
    args = parser.parse_args()
    session = resolve_session(args.session)
    report = inspect(session, args.decode_all)
    write_json(session / "quality_report.json", report)
    print("{} | course={} group={}".format(session, report["course"], report["group"]))
    for name, stats in report["streams"].items():
        match = " match={:.1%}".format(stats["match_fraction"]) if "match_fraction" in stats else ""
        print("{}: {} samples {:.1f}Hz max_gap={:.3f}s{}".format(name, stats["count"], stats["hz"], stats["max_gap_s"], match))
    for message in report["warnings"]:
        print("WARN: " + message)
    for message in report["errors"]:
        print("FAIL: " + message)
    print("DATA CHECKS " + ("PASS" if report["data_checks_passed"] else "FAIL"))
    print("Steering motion check: " + str(report["steering_motion_check"]))
    print("Review driving, correct camera roles, crop and HUD labels before training.")
    if args.play:
        play(session, args.camera)
    return 0 if report["data_checks_passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(130)
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
