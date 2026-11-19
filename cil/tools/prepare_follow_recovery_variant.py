"""90분 원본 선택을 최소 변경해 누락된 FOLLOW ROAD 실제 복귀 표본을 복원한다.

LEFT/STRAIGHT/RIGHT 선택과 기존 Validation은 그대로 보존한다. 전체 회랑에서
분리 검사를 이미 통과한 캐시만 읽으며 추가 수집·학습·차량 제어는 하지 않는다.
"""

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools.prepare_east_outer_90min import (
    PERIOD_NS, count_rows, dense_xy, distances, read_csv, read_json, rel, resolve,
    sha256, spread_subset, write_json, write_rows,
)


def center_changes(original, pool, metadata, origins):
    """실제 복귀 행은 모두 포함하고 같은 noise-session 층의 평상시 행만 교체한다."""
    old_centers = [row for row in original if row["camera"] == "center"]
    old_paths = {row["image_path"] for row in old_centers}
    all_follow_recovery = [row for row in pool if row["camera"] == "center"
                           and row["command"] == "0" and int(row["recovery_active"]) == 1]
    additions = [row for row in all_follow_recovery if row["image_path"] not in old_paths]
    masks = {}
    for row in pool:
        key = (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS, row["command"])
        masks.setdefault(key, set()).add(row["camera"])

    def stratum(row):
        key = (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS, row["command"])
        return bool(metadata[row["session_id"]].get("noise_enabled")), tuple(sorted(masks[key]))

    removals = []
    strata = Counter(stratum(row) for row in additions)
    for group, required in sorted(strata.items()):
        candidates = [row for row in old_centers if row["command"] == "0" and not int(row["recovery_active"])
                      and stratum(row) == group]
        if len(candidates) < required:
            raise ValueError("not enough non-recovery FOLLOW rows matching noise-session and camera-presence strata: " + str(group))
        if required:
            removals.extend(spread_subset(candidates, required))
    removed = {row["image_path"] for row in removals}
    selected = [row for row in old_centers if row["image_path"] not in removed] + additions
    if len(selected) != len(old_centers) or len({row["image_path"] for row in selected}) != len(selected):
        raise ValueError("center count or uniqueness changed during recovery restoration")
    return selected, additions, removals, all_follow_recovery


def aligned_rows(pool, selected_centers, origins):
    """선택된 실제 center와 동일한 100ms 시간칸·명령에 있는 좌우만 연결한다."""
    bins = {(row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS): row["command"]
            for row in selected_centers}
    if len(bins) != len(selected_centers):
        raise ValueError("duplicate selected center time bin")
    center_paths = {row["image_path"] for row in selected_centers}
    result, used = [], set()
    for row in pool:
        key = (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS)
        if bins.get(key) != row["command"]:
            continue
        if row["camera"] == "center" and row["image_path"] not in center_paths:
            continue
        if row["image_path"] in used:
            raise ValueError("duplicate image path in eligible pool")
        used.add(row["image_path"])
        result.append(row)
    return sorted(result, key=lambda row: (row["session_id"], int(row["stamp_ns"]), row["camera"]))


def preserve_sample_indices(original, selected):
    """변경된 행만 같은 명령·카메라의 원래 위치에 넣어 다른 학습 index를 유지한다."""
    old_paths = {row["image_path"] for row in original}
    selected_paths = {row["image_path"] for row in selected}
    replacements = defaultdict(deque)
    for row in selected:
        if row["image_path"] not in old_paths:
            replacements[(row["command"], row["camera"])].append(row)
    result, changed = [], []
    for index, row in enumerate(original):
        if row["image_path"] in selected_paths:
            result.append(row)  # 미변경 샘플은 값과 위치를 모두 보존한다.
        else:
            replacement = replacements[(row["command"], row["camera"])].popleft()
            result.append(replacement)
            changed.append(index)
    if any(replacements.values()):
        raise ValueError("replacement rows remain after preserving sample indices")
    return result, changed


def guard_trees(source_ledger):
    plan = read_json(BASE / source_ledger["plan_snapshot"])
    trees = {name: cKDTree(dense_xy(value["guard_reference_xyz"]))
             for name, value in plan["test_reservations"].items()}
    for name in ("P2Q", "V1", "V2"):
        route = np.loadtxt(BASE / plan["routes"][name]["route_csv"], delimiter=",", skiprows=1, ndmin=2)
        trees[name] = cKDTree(dense_xy(route[:, 1:4]))
    validation_positions = []
    baseline = read_json(BASE / "data/collection_completion.json")
    for item in baseline["accepted"]:
        if item["split"] == "validation":
            validation_positions.extend([[float(row["x_m"]), float(row["y_m"])]
                                         for row in read_csv(BASE / item["session"] / "vehicle.csv")])
    trees["actual_validation_positions"] = cKDTree(validation_positions)
    return trees


def build_variant(source_root, output):
    source_root, output = resolve(source_root), resolve(output)
    output.relative_to(BASE / "runs")
    if output.exists():
        raise FileExistsError("choose a new isolated output: " + str(output))
    original_path = source_root / "prepared/train.csv"
    original_summary = read_json(source_root / "prepared/summary.json")
    source_ledger_path = source_root / "prepared/source_ledger.json"
    source_ledger = read_json(source_ledger_path)
    if sha256(original_path) != original_summary["train_sha256"]:
        raise ValueError("parent 90-minute training CSV changed")
    if sha256(source_ledger_path) != original_summary["source_ledger_sha256"]:
        raise ValueError("parent eligible-pool ledger changed")
    val = source_root / "prepared/val.csv"
    if sha256(val) != original_summary["val_sha256"] or sha256(val) != sha256(BASE / "data/prepared/val.csv"):
        raise ValueError("fixed validation differs")
    frozen = BASE / source_ledger["frozen_route"]
    if sha256(frozen) != source_ledger["frozen_route_sha256"]:
        raise ValueError("entire frozen extended evaluation route changed")

    pool, metadata, origins = [], {}, {}
    reports = [source_ledger["baseline"]] + [item for item in source_ledger["accepted_sessions"] if item["status"] == "prepared"]
    for report in reports:
        cache = BASE / report["cache"]
        if sha256(cache / "train.csv") != report["train_sha256"]:
            raise ValueError("eligible prepared cache changed: " + str(cache))
        pool.extend(read_csv(cache / "train.csv"))
        metadata.update(report["metadata"])
        origins.update({sid: int(value) for sid, value in report["origins_ns"].items()})
    if len({row["image_path"] for row in pool}) != len(pool):
        raise ValueError("source pool has repeated image paths")
    original = read_csv(original_path)
    original_counts = count_rows(original, metadata)
    if original_counts["center_counts"] != dict.fromkeys(("0", "1", "2", "3"), 13500):
        raise ValueError("parent dataset must be the fixed balanced 54000-center version")
    centers, added, removed, follow_recovery = center_changes(original, pool, metadata, origins)
    selected = aligned_rows(pool, centers, origins)
    selected, changed_indices = preserve_sample_indices(original, selected)
    summary = count_rows(selected, metadata)
    if summary["center_counts"] != original_counts["center_counts"]:
        raise ValueError("balanced command counts changed")
    if summary["noise_session_center_counts"] != original_counts["noise_session_center_counts"]:
        raise ValueError("noise-session command counts changed")
    if summary["recovery_active_center_counts"]["0"] != len(follow_recovery):
        raise ValueError("some available actual FOLLOW recovery frames were lost")
    old_other = [row for row in original if row["command"] != "0"]
    new_other = [row for row in selected if row["command"] != "0"]
    if old_other != new_other:  # 다른 세 명령은 좌우 카메라까지 원래 행을 그대로 유지한다.
        raise ValueError("a non-FOLLOW camera sample changed")
    command_camera_before = Counter(row["command"] + ":" + row["camera"] for row in original)
    command_camera_after = Counter(row["command"] + ":" + row["camera"] for row in selected)
    if command_camera_before != command_camera_after:
        raise ValueError("per-command camera-row counts changed")
    image_paths = [resolve(row["image_path"]) for row in selected]
    if len(set(image_paths)) != len(image_paths) or any(not path.is_file() for path in image_paths):
        raise ValueError("final image paths are duplicated or missing")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in selected], guard_trees(source_ledger))
    if any(value < source_ledger["buffer_m"] for value in minimum.values()):
        raise ValueError("recovery restoration entered a protected test/validation buffer")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".follow_recovery_", dir=str(output.parent)))
    try:
        prepared = staging / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", selected)
        shutil.copy2(val, prepared / "val.csv")
        shutil.copy2(source_ledger_path, prepared / "source_ledger.json")
        summary.update(
            status="prepared", variant="restore_all_available_follow_road_recovery",
            created_utc=datetime.now(timezone.utc).isoformat(), center_duration_s=5400.0,
            parent_prepared=rel(source_root / "prepared"), parent_train_sha256=original_summary["train_sha256"],
            parent_summary_sha256=sha256(source_root / "prepared/summary.json"),
            train_sha256=sha256(prepared / "train.csv"), val_sha256=sha256(prepared / "val.csv"),
            source_ledger="source_ledger.json", source_ledger_sha256=sha256(prepared / "source_ledger.json"),
            follow_recovery_before=original_counts["recovery_active_center_counts"]["0"],
            follow_recovery_available=len(follow_recovery), follow_recovery_selected=len(follow_recovery),
            added_center_frames=len(added), removed_center_frames=len(removed),
            unchanged_other_command_center_frames=sum(row["camera"] == "center" for row in new_other),
            other_command_all_camera_rows_byte_value_identical=True,
            command_camera_counts=dict(command_camera_after),
            command_camera_counts_unchanged=True,
            command_and_camera_at_every_index_unchanged=True,
            changed_row_indices_count=len(changed_indices),
            unchanged_sample_indices_count=len(original) - len(changed_indices),
            noise_session_counts_preserved=True,
            selection="minimal replacement: add every eligible real FOLLOW recovery center; remove equal non-recovery centers matching noise-session and side-camera-presence strata with route/session/time spreading; align side bins",
            raw_and_side_frames_not_counted_as_additional_driving_time=True,
            target_noise_fraction=original_summary["target_noise_fraction"],
            actual_noise_session_fraction=sum(summary["noise_session_center_counts"].values()) / 54000,
            protected_min_distances_m=minimum, comparability=original_summary["comparability"],
            matched_training_budget={"seed": 7, "batch_size": 120, "optimizer_updates": 2025,
                                     "sampling": "fixed_coverage", "evaluation_checkpoint": "last.pt",
                                     "note": "requested matched control/variant comparison; not executed by this data tool"},
            interpretation="Actual recovery selection is changed; command balance, selected duration, noise-session counts, other commands, validation and geometry remain fixed. It does not add new recovery observations beyond the eligible pool.")
        write_json(prepared / "summary.json", summary)
        write_json(staging / "changes.json", {
            "added_center_image_paths": [row["image_path"] for row in added],
            "removed_center_image_paths": [row["image_path"] for row in removed],
            "changed_row_indices": changed_indices,
            "sources": {"original_train": original_summary["train_sha256"],
                        "eligible_ledger": original_summary["source_ledger_sha256"]},
        })
        write_json(staging / "ledger.json", {"status": "finalized", "variant": summary["variant"],
                   "parent": rel(source_root), "selected": summary, "target_met": True,
                   "target_center_frames": 54000, "target_per_command": 13500})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="runs/east_outer_90min")
    parser.add_argument("--output", default="runs/east_outer_follow_recovery_v2_ordered")
    args = parser.parse_args()
    summary = build_variant(args.source_root, args.output)
    print(json.dumps({key: summary[key] for key in (
        "center_counts", "recovery_active_center_counts", "noise_session_center_counts",
        "added_center_frames", "removed_center_frames", "follow_recovery_available", "train_sha256")}, indent=2))


if __name__ == "__main__":
    main()
