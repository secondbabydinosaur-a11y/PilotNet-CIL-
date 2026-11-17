#!/usr/bin/env python3
"""Summarize existing capture checks. Does not preprocess or split any dataset."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from data_utils import BASE, read_json, write_json


def summarize(root, prefix="expert_g"):
    runs, pending = [], []
    for path in sorted(Path(root).glob("*/session.json")):
        meta = read_json(path)
        if not meta["group"].startswith(prefix):
            continue
        report_path = path.parent / "quality_report.json"
        if meta["status"] == "recording" or not report_path.exists():
            pending.append(meta["session_id"])
            continue
        report = read_json(report_path)
        tracking = report.get("expert_tracking") or {}
        cameras = [report["streams"][name] for name in ("left", "center", "right")]
        runs.append({"session_id": meta["session_id"], "group": meta["group"], "course": meta["course"],
            "path": str(path.parent.resolve()), "passed": report["data_checks_passed"],
            "control_kind": meta["control"]["kind"], "human_steering": meta["control"]["human_steering"],
            "variant": meta["control"]["variant"],
            "recorded_seconds": report["streams"]["ego"]["span_s"],
            "left_images": meta["saved"]["left"], "center_images": meta["saved"]["center"],
            "right_images": meta["saved"]["right"], "ego_samples": meta["saved"]["ego"],
            "minimum_camera_hz": min(c["hz"] for c in cameras),
            "ego_hz": report["streams"]["ego"]["hz"],
            "minimum_match_fraction": min(c["match_fraction"] for c in cameras),
            "max_speed_kph": tracking.get("max_speed_kph"),
            "max_abs_cte_m": tracking.get("max_abs_cte_m"),
            "parse_errors": sum(meta["parse_errors"].values()),
            "errors": report["errors"], "warnings": report["warnings"]})
    counts = Counter("{}/{}".format(r["group"], r["course"]) for r in runs if r["passed"])
    return {"scope": "raw capture and quality inspection only; no preprocessing/splits/model/training",
        "source": "MORAI simulator RGB JPEGs and ordinary Ego UDP; classical map Pure Pursuit teacher",
        "human_demonstrations": False,
        "generalization_note": "same two map sections and fixed clear/noon conditions; groups are separate recording batches, not unseen roads",
        "runs_completed": len(runs), "runs_passed": sum(r["passed"] for r in runs),
        "pending": pending, "group_course_counts": dict(counts),
        "total_recorded_seconds": sum(r["recorded_seconds"] for r in runs if r["passed"]),
        "total_images": sum(r[k] for r in runs if r["passed"] for k in ("left_images", "center_images", "right_images")),
        "total_ego_samples": sum(r["ego_samples"] for r in runs if r["passed"]),
        "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=BASE / "dataset/raw")
    parser.add_argument("--prefix", default="expert_g")
    parser.add_argument("--output", type=Path, default=BASE / "dataset/collection_summary.json")
    args = parser.parse_args()
    summary = summarize(args.raw_root, args.prefix)
    write_json(args.output, summary)
    if summary["runs"]:
        with args.output.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["runs"][0]))
            writer.writeheader()
            writer.writerows(summary["runs"])
    print("{} / {} runs passed, {:.1f} min, {:,} images, pending={}".format(
        summary["runs_passed"], summary["runs_completed"], summary["total_recorded_seconds"] / 60,
        summary["total_images"], len(summary["pending"])))


if __name__ == "__main__":
    main()
