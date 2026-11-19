"""급우회전 v1에서 가상 좌우 카메라 때문에 빠진 실제 center 정답을 복원한다.

가상 카메라의 불가능한 조향 정답을 만들지 않는다. 기존 side 영상은 독립 학습
표본으로 유지하고 실제 RIGHT center만 같은 phase/noise/recovery index에 넣는다.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools.prepare_east_outer_90min import (
    count_rows, distances, read_csv, read_json, rel, resolve, sha256, write_json, write_rows,
)
from tools.prepare_follow_recovery_variant import guard_trees
from tools.prepare_sharp_right_variant import turn_for


def build_variant(parent, capture, source_root, output):
    parent, capture, source_root, output = map(resolve, (parent, capture, source_root, output))
    output.relative_to(BASE / "runs")
    if output.exists(): raise FileExistsError(output)
    summary = read_json(parent / "prepared/summary.json")
    original_path = parent / "prepared/train.csv"
    if sha256(original_path) != summary["train_sha256"]:
        raise ValueError("sharp-right parent training CSV changed")
    if sha256(parent / "prepared/val.csv") != sha256(BASE / "data/prepared/val.csv"):
        raise ValueError("fixed validation changed")
    ledger = read_json(source_root / "prepared/source_ledger.json")
    if sha256(BASE / ledger["frozen_route"]) != ledger["frozen_route_sha256"]:
        raise ValueError("entire frozen evaluation corridor changed")
    plan = read_json(BASE / ledger["plan_snapshot"])
    routes = dict(plan["routes"])
    metadata = {}
    for report in [ledger["baseline"]] + [r for r in ledger["accepted_sessions"] if r["status"] == "prepared"]:
        metadata.update(report["metadata"])
    original = read_csv(original_path)
    original_paths = {row["image_path"] for row in original}
    cohort = summary["sharp_right_audit"]["source_sessions"]
    approved = set(summary["sharp_right_audit"]["approved_routes"])
    candidates, source_hashes = [], {}
    new_sessions = {item["session_id"] for item in cohort}
    for item in cohort:
        session = capture / "raw" / item["session_id"]
        meta, quality = read_json(session / "session.json"), read_json(session / "quality_report.json")
        if (meta.get("status") != "route_finished" or not meta.get("collection_accepted")
                or not quality.get("data_checks_passed") or meta["route_id"] not in approved):
            raise ValueError("parent sharp-right source is not an approved quality-passed session")
        cache = capture / "prepared_cache/cache" / item["session_id"]
        report = read_json(cache / "ledger.json")
        if sha256(cache / "train.csv") != report["train_sha256"]:
            raise ValueError("vetted source cache changed")
        for name in ("session.json", "quality_report.json", "vehicle.csv", "commands.csv", "images.csv"):
            source_hashes[rel(session / name)] = sha256(session / name)
        metadata[session.name] = meta
        routes[meta["route_id"]] = {"navigation": meta["navigation"]}
        candidates.extend(row for row in read_csv(cache / "train.csv")
                          if row["camera"] == "center" and row["command"] == "3" and row["image_path"] not in original_paths)
    if not candidates or len({row["image_path"] for row in candidates}) != len(candidates):
        raise ValueError("no unique omitted real right-center observations")

    def phase(row):
        return turn_for(row, routes[row["route_id"]]["navigation"])[1]
    def stratum(row):
        return (phase(row), bool(metadata[row["session_id"]].get("noise_enabled")), int(row["recovery_active"]))
    link_counts = Counter(turn_for(row, routes[row["route_id"]]["navigation"])[0]["link_id"]
                          for row in original if row["camera"] == "center" and row["command"] == "3")
    slots = []
    for index, row in enumerate(original):
        if (row["camera"] == "center" and row["command"] == "3" and row["session_id"] not in new_sessions
                and abs(float(row["steer"]) * 40.) < 20.):
            slots.append((index, row))  # 기존 강우조향과 새 실측 표본은 제거하지 않는다.
    selected = list(original)
    used, changes = set(), []
    # 희소한 복귀 층을 먼저 배정하고 같은 층·단계 안에서만 실제 관측을 교체한다.
    ordered = sorted(candidates, key=lambda row: (not bool(int(row["recovery_active"])),
                                                  phase(row), float(row["steer"]), row["image_path"]))
    for new in ordered:
        compatible = [(i, old) for i, old in slots if i not in used and stratum(old) == stratum(new)]
        if not compatible:
            raise ValueError("not enough weak RIGHT center slots matching phase/noise/recovery; no counts were relaxed")
        def preference(item):
            i, old = item
            link = turn_for(old, routes[old["route_id"]]["navigation"])[0]["link_id"]
            return (link != "A2256W000318", -link_counts[link], abs(float(old["steer"])), i)
        index, old = min(compatible, key=preference)
        selected[index] = new
        used.add(index)
        changes.append({"index": index, "old_image_path": old["image_path"], "new_image_path": new["image_path"],
                        "phase": phase(new), "noise_session": stratum(new)[1], "recovery_active": stratum(new)[2],
                        "removed_link": turn_for(old, routes[old["route_id"]]["navigation"])[0]["link_id"],
                        "added_link": turn_for(new, routes[new["route_id"]]["navigation"])[0]["link_id"],
                        "old_steer_deg": float(old["steer"]) * 40., "new_steer_deg": float(new["steer"]) * 40.})
    before, after = count_rows(original, metadata), count_rows(selected, metadata)
    for key in ("center_counts", "camera_counts", "train_rows", "noise_session_center_counts", "recovery_active_center_counts"):
        if before[key] != after[key]: raise ValueError("controlled counts changed: " + key)
    for i, (old, new) in enumerate(zip(original, selected)):
        if i not in used and old != new: raise ValueError("unselected row changed")
        if i in used and ((old["command"], old["camera"]) != ("3", "center") or stratum(old) != stratum(new)):
            raise ValueError("per-index command/camera/phase/recovery contract changed")
    paths = [resolve(row["image_path"]) for row in selected]
    if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("duplicate or missing image")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in selected], guard_trees(ledger))
    if min(minimum.values()) < 20.:
        raise ValueError("new center observations entered a protected buffer")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sharp_center_", dir=str(output.parent)))
    try:
        prepared = stage / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", selected)
        shutil.copy2(parent / "prepared/val.csv", prepared / "val.csv")
        strong = [row for row in candidates if float(row["steer"]) * 40. <= -20.]
        after.update(status="prepared", variant="sharp_right_v1_plus_every_omitted_real_center",
                     created_utc=datetime.now(timezone.utc).isoformat(), parent_prepared=rel(parent / "prepared"),
                     parent_train_sha256=summary["train_sha256"], train_sha256=sha256(prepared / "train.csv"),
                     val_sha256=sha256(prepared / "val.csv"), center_duration_s=5400.,
                     added_center_frames=len(candidates), changed_row_indices_count=len(used),
                     unchanged_sample_indices_count=len(original)-len(used),
                     restored_strong_right_centers=len(strong), strong_right_threshold_deg=-20.,
                     added_link_counts=dict(Counter(change["added_link"] for change in changes)),
                     removed_link_counts=dict(Counter(change["removed_link"] for change in changes)),
                     phase_counts_preserved_at_each_changed_index=True,
                     noise_and_recovery_preserved_at_each_changed_index=True,
                     all_side_rows_and_other_commands_unchanged=True, no_fabricated_side_labels=True,
                     camera_alignment_note="Each camera image is a separate CNN sample. Existing side rows stay valid standalone augmented samples; matched center/side membership is intentionally not required for restored centers.",
                     protected_min_distances_m=minimum,
                     source_raw_sha256=source_hashes,
                     note="Every valid omitted RIGHT center from the same parent-approved physical cohort is restored, including all strong centers lacking virtual LEFT labels. Real teacher targets and flags are unchanged; no training/simulator action performed.")
        write_json(prepared / "summary.json", after)
        write_json(stage / "changes.json", changes)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return after


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="runs/east_outer_sharp_right_variant_v1")
    parser.add_argument("--capture-root", default="runs/east_outer_sharp_right_data/capture")
    parser.add_argument("--source-root", default="runs/east_outer_90min")
    parser.add_argument("--output", default="runs/east_outer_sharp_right_center_v2_ordered")
    args = parser.parse_args()
    print(json.dumps(build_variant(args.parent, args.capture_root, args.source_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
