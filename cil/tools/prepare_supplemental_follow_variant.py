"""완료·검사된 추가 실측 복귀 구간만 v2 데이터에 격리해서 삽입한다.

초기 위치 교란은 recovery_active가 0일 수 있으므로 원래 값을 바꾸지 않고 별도
증거 목록에 기록한다. 진행 중이거나 품질 검사에 실패한 원본은 읽어 준비하지 않는다.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from prepare import Timeline, TEACHER_COMPARE, linear
from tools.prepare_east_outer_90min import (
    PERIOD_NS, count_rows, digest_value, distances, prepare_session, read_csv, read_json,
    rel, resolve, sha256, spread_subset, write_json, write_rows,
)
from tools.prepare_follow_recovery_variant import aligned_rows, guard_trees, preserve_sample_indices


def time_key(row, origins):
    return (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS, row["command"])


def recovery_evidence(session, rows, meta):
    commands = read_csv(session / "commands.csv")
    timeline = Timeline(commands, "ego_stamp_ns", TEACHER_COMPARE)
    drive = [row for row in commands if row["control_phase"] == "drive" and int(row["label_valid"]) == 1]
    first = min(int(row["ego_stamp_ns"]) for row in drive)
    initial_end, stable_start = first + 10_000_000_000, None
    for row in drive:
        stamp = int(row["ego_stamp_ns"])
        if stamp > initial_end or float(row["route_s_m"]) > 20.0:
            initial_end = min(initial_end, stamp)
            break
        stable = abs(float(row["cte_m"])) <= .08 and abs(float(row["heading_error_deg"])) <= 1.0
        stable_start = (stable_start if stable_start is not None else stamp) if stable else None
        if stable_start is not None and stamp - stable_start >= 500_000_000:
            initial_end = stamp
            break
    evidence = {}
    for row in rows:
        if row["camera"] != "center" or row["command"] != "0": continue
        stamp = int(row["stamp_ns"])
        match = timeline.bracket(stamp)
        if match is None: raise ValueError("prepared supplemental center cannot be matched to its real command log")
        reasons = []
        if int(row["recovery_active"]): reasons.append("original_recovery_active")
        if abs(float(meta.get("initial_offset_m", 0))) >= .15 and first <= stamp <= initial_end and float(row["route_s_m"]) <= 20.0:
            reasons.append("measured_initial_offset_recovery_window")
        if reasons:
            evidence[row["image_path"]] = {
                "image_path": row["image_path"], "session_id": row["session_id"],
                "route_id": row["route_id"], "stamp_ns": stamp,
                "initial_offset_m": meta["initial_offset_m"], "reasons": reasons,
                "original_recovery_active": int(row["recovery_active"]),
                "measured_cte_m": linear(match, "cte_m"),
                "measured_heading_error_deg": linear(match, "heading_error_deg"),
                "teacher_steer": row["center_teacher_steer"],
            }
    return evidence


def prepare_available(source_root, supplemental):
    source = read_json(source_root / "prepared/source_ledger.json")
    val_sha = sha256(source_root / "prepared/val.csv")
    config = read_json(supplemental / "config.json")
    plan_path = resolve(config["paths"]["route_plan"])
    baseline = read_json(BASE / "data/collection_completion.json")
    val_entries = [item for item in baseline["accepted"] if item["split"] == "validation"]
    cache_root = supplemental / "prepared_cache"
    cache_root.mkdir(exist_ok=True)
    trees = guard_trees(source)
    contract = digest_value({"tool": sha256(Path(__file__)), "plan": sha256(plan_path),
                             "source_ledger": sha256(source_root / "prepared/source_ledger.json"),
                             "window": "initial<=10s/20m until .08m/1deg stable.5s, or original recovery_active"})
    prepared, pending = [], []
    for session in sorted((supplemental / "raw").iterdir()):
        if not session.is_dir(): continue
        if not (session / "quality_report.json").is_file():
            pending.append(session.name)
            continue  # 진행 중인 session.json이나 CSV는 읽지 않는다.
        quality = read_json(session / "quality_report.json")
        if not quality.get("data_checks_passed"): continue
        meta = read_json(session / "session.json")
        if meta.get("status") != "route_finished" or not meta.get("collection_accepted"): continue
        item = {"session": rel(session), "split": "train", "route_id": meta["route_id"]}
        report = prepare_session(item, plan_path, cache_root, val_entries, val_sha, trees, 20., contract)
        if report["status"] != "prepared":
            raise ValueError("supplemental run failed complete-corridor separation: " + session.name)
        rows = read_csv(BASE / report["cache"] / "train.csv")
        evidence = recovery_evidence(session, rows, meta)
        prepared.append({"session": session, "meta": meta, "report": report, "rows": rows, "evidence": evidence})
    return source, trees, prepared, pending


def build_variant(source_root, parent, supplemental, output, prepare_only=False):
    source_root, parent, supplemental, output = map(resolve, (source_root, parent, supplemental, output))
    output.relative_to(BASE / "runs")
    if output.exists(): raise FileExistsError(output)
    source, trees, sessions, pending = prepare_available(source_root, supplemental)
    reports = [source["baseline"]] + [r for r in source["accepted_sessions"] if r["status"] == "prepared"]
    metadata, origins = {}, {}
    for report in reports:
        metadata.update(report["metadata"])
        origins.update({sid: int(stamp) for sid, stamp in report["origins_ns"].items()})
    supplements, evidence = [], {}
    for item in sessions:
        supplements.extend(item["rows"])
        evidence.update(item["evidence"])
        metadata.update(item["report"]["metadata"])
        origins.update({sid: int(stamp) for sid, stamp in item["report"]["origins_ns"].items()})
    masks = defaultdict(set)
    for row in supplements: masks[time_key(row, origins)].add(row["camera"])
    complete = [row for row in supplements if row["camera"] == "center" and row["image_path"] in evidence
                and masks[time_key(row, origins)] == {"center", "left", "right"}]
    audit = {"completed_routes": sorted({item["meta"]["route_id"] for item in sessions}),
             "pending_sessions_not_read": pending, "candidate_center_frames": len(evidence),
             "complete_three_camera_center_frames": len(complete),
             "session_windows": [{"session_id": item["session"].name, "route_id": item["meta"]["route_id"],
                                  "initial_offset_m": item["meta"]["initial_offset_m"],
                                  "window_centers": len(item["evidence"]),
                                  "initial_offset_centers": sum("measured_initial_offset_recovery_window" in e["reasons"] for e in item["evidence"].values()),
                                  "original_active_centers": sum(e["original_recovery_active"] for e in item["evidence"].values())}
                                 for item in sessions]}
    if prepare_only: return audit
    if not {"LONG_11", "LONG_09", "LONG_03"}.issubset(audit["completed_routes"]):
        raise ValueError("wait for quality-passed LONG_11/LONG_09/LONG_03; pending=" + str(pending))
    original_summary = read_json(parent / "prepared/summary.json")
    if sha256(parent / "prepared/train.csv") != original_summary["train_sha256"]:
        raise ValueError("parent v2 training CSV changed")
    original = read_csv(parent / "prepared/train.csv")
    for row in original: masks[time_key(row, origins)].add(row["camera"])
    old_centers = [row for row in original if row["camera"] == "center"]
    def removable(row):
        sid = row["session_id"]
        early_offset = abs(float(metadata[sid].get("initial_offset_m") or 0)) >= .15 and int(row["stamp_ns"]) - origins[sid] <= 10_000_000_000
        return (row["command"] == "0" and not int(row["recovery_active"]) and not early_offset
                and masks[time_key(row, origins)] == {"center", "left", "right"})
    additions, removals = [], []
    for noisy in (False, True):
        pool = [row for row in complete if bool(metadata[row["session_id"]].get("noise_enabled")) == noisy]
        available = [row for row in old_centers if removable(row)
                     and bool(metadata[row["session_id"]].get("noise_enabled")) == noisy]
        # 먼저 새로 실측한 초기 위치 복귀를 보존하고 나머지를 실제 pulse 복귀로 채운다.
        limit = min(len(pool), len(available))
        initial = [row for row in pool if "measured_initial_offset_recovery_window" in evidence[row["image_path"]]["reasons"]]
        pulse_only = [row for row in pool if row not in initial]
        selected = spread_subset(initial, min(limit, len(initial))) if initial and limit else []
        if limit > len(selected): selected += spread_subset(pulse_only, limit - len(selected))
        additions.extend(selected)
        if selected: removals.extend(spread_subset(available, len(selected)))
    if not additions: raise ValueError("no matched supplemental FOLLOW recovery frames can be inserted")
    removed_paths = {row["image_path"] for row in removals}
    centers = [row for row in old_centers if row["image_path"] not in removed_paths] + additions
    selected = aligned_rows(original + supplements, centers, origins)
    selected, changed = preserve_sample_indices(original, selected)
    before, after = count_rows(original, metadata), count_rows(selected, metadata)
    for key in ("center_counts", "noise_session_center_counts", "camera_counts", "train_rows"):
        if before[key] != after[key]: raise ValueError("matched count changed: " + key)
    if any((a["command"], a["camera"]) != (b["command"], b["camera"]) or (a["command"] != "0" and a != b)
           for a, b in zip(original, selected)):
        raise ValueError("non-FOLLOW row or sampler group index changed")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in selected], trees)
    if min(minimum.values()) < 20: raise ValueError("supplemental variant touched protected evaluation/validation data")
    if len({row["image_path"] for row in selected}) != len(selected): raise ValueError("duplicate final image path")
    output.parent.mkdir(exist_ok=True, parents=True)
    staging = Path(tempfile.mkdtemp(prefix=".supplemental_follow_", dir=str(output.parent)))
    try:
        target = staging / "prepared"
        target.mkdir()
        write_rows(target / "train.csv", selected)
        shutil.copy2(parent / "prepared/val.csv", target / "val.csv")
        if sha256(target / "val.csv") != sha256(BASE / "data/prepared/val.csv"):
            raise ValueError("fixed validation changed")
        selected_evidence = [evidence[row["image_path"]] for row in additions]
        write_json(staging / "supplemental_evidence.json", selected_evidence)
        after.update(status="prepared", variant="v2_plus_physically_collected_follow_recovery", created_utc=datetime.now(timezone.utc).isoformat(),
                     parent_prepared=rel(parent / "prepared"), parent_train_sha256=original_summary["train_sha256"],
                     train_sha256=sha256(target / "train.csv"), val_sha256=sha256(target / "val.csv"),
                     center_duration_s=5400., supplemental_audit=audit, added_center_frames=len(additions),
                     removed_center_frames=len(removals), changed_row_indices_count=len(changed), unchanged_sample_indices_count=len(original)-len(changed),
                     original_tagged_recovery_active_before=before["recovery_active_center_counts"],
                     initial_offset_recovery_not_relabelled=True,
                     supplemental_initial_offset_centers=sum("measured_initial_offset_recovery_window" in e["reasons"] for e in selected_evidence),
                     supplemental_original_active_centers=sum(e["original_recovery_active"] for e in selected_evidence),
                     command_camera_index_and_other_commands_unchanged=True, fixed_val_unchanged=True,
                     protected_min_distances_m=minimum,
                     note="Original teacher actions and recovery_active bits are preserved; initially offset examples are identified by measured pose/window evidence, not fabricated flags. No training or driving launched.")
        write_json(target / "summary.json", after)
        write_json(staging / "changes.json", {"changed_row_indices": changed,
                   "added_center_image_paths": [row["image_path"] for row in additions],
                   "removed_center_image_paths": [row["image_path"] for row in removals]})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return after


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="runs/east_outer_90min")
    parser.add_argument("--parent", default="runs/east_outer_follow_recovery_v2_ordered")
    parser.add_argument("--supplemental", default="runs/east_outer_recovery_check/supplemental")
    parser.add_argument("--output", default="runs/east_outer_follow_supplemental_v1_ordered")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = build_variant(args.source_root, args.parent, args.supplemental, args.output, args.prepare_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
