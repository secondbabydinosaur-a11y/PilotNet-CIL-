"""V3의 RIGHT를 보존하며 실제 STRAIGHT 교차로 center만 격리 교체한다.

수집·학습·차량 제어는 실행하지 않는다. 완료된 QA 합격 세션을 기존 prepare로
준비하고, 정상 교차로 통과와 실제 복귀를 1:1로 구성한 focus 목록도 저장한다.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from prepare import Timeline, TEACHER_COMPARE, linear
from tools.prepare_curve_right_recovery_variant import ready_state, signed_summary
from tools.prepare_east_outer_90min import count_rows, digest_value, distances, prepare_session, read_csv, read_json, rel, resolve, sha256, spread_subset, write_json, write_rows
from tools.prepare_follow_recovery_variant import guard_trees

APPROVED_ROUTES = ("EX_S1", "SHORT_15", "SHORT_16", "PAIR_STRAIGHT_222")


def straight_for(row, navigation):
    """실제 STRAIGHT 명령 창에서 접근·교차로 내부·출구를 구분한다."""
    station = float(row["route_s_m"])
    turns = [n for n in navigation if n["command"] == "straight"]
    inside = [n for n in turns if n["entry_s_m"] <= station <= n["exit_s_m"]]
    if inside: return inside[0], "inside"
    upcoming = [n for n in turns if 0 < n["entry_s_m"] - station <= n["approach_m"] + .001]
    if upcoming: return min(upcoming, key=lambda n: n["entry_s_m"]), "approach"
    passed = [n for n in turns if 0 < station - n["exit_s_m"] <= n["exit_hold_m"] + .001]
    if passed: return passed[-1], "exit_hold"
    raise ValueError("STRAIGHT label lacks an actual matching navigation window")


def prepare_completed(capture, ledger, original_plan, trees):
    """확정된 세션별 입력 계약으로 준비 캐시를 재사용한다. 활성 원본은 읽지 않는다."""
    config = read_json(capture / "config.json")
    plan = read_json(resolve(config["paths"]["route_plan"]))
    map_path = Path(plan["source"]) / "link_set.json"
    if sha256(map_path) != plan["source_hashes"]["link_set.json"]: raise ValueError("fixed MGeo changed")
    baseline = read_json(BASE / "data/collection_completion.json")
    val_entries = [item for item in baseline["accepted"] if item["split"] == "validation"]
    val_sha = sha256(BASE / "data/prepared/val.csv")
    cache = capture / "prepared_cache_balanced"
    (cache / "plans").mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for session in sorted((capture / "raw").iterdir()):
        if not session.is_dir(): continue
        qa_path = session / "quality_report.json"
        if not qa_path.is_file():
            skipped.append({"session_id": session.name, "reason": "unfinished_or_not_quality_checked"})
            continue
        qa = read_json(qa_path)
        if not qa.get("data_checks_passed"):
            skipped.append({"session_id": session.name, "reason": "quality_failed"})
            continue
        meta = read_json(session / "session.json")
        if meta.get("status") != "route_finished" or not meta.get("collection_accepted") or meta.get("split") != "train":
            skipped.append({"session_id": session.name, "reason": "not_accepted_finished_training"})
            continue
        if meta["route_id"] not in APPROVED_ROUTES:
            skipped.append({"session_id": session.name, "reason": "route_not_in_approved_experiment"})
            continue
        info = copy.deepcopy(plan["routes"][meta["route_id"]])
        route_sha = sha256(resolve(info["route_csv"]))
        if route_sha != meta["route_sha256"] or route_sha != info["route_sha256"] or meta["navigation"] != info["navigation"]:
            raise ValueError("existing STRAIGHT geometry/navigation changed")
        crossings = [n for n in info["navigation"] if n["command"] == "straight"]
        if not crossings or any(n.get("map_related_signal") != "straight" for n in crossings):
            raise ValueError("new source is not a reviewed straight-through map approach")
        snapshot = copy.deepcopy(plan)
        snapshot["test_reservations"] = copy.deepcopy(original_plan["test_reservations"])
        snapshot_path = cache / "plans" / (digest_value(snapshot)[:16] + ".json")
        if not snapshot_path.exists(): write_json(snapshot_path, snapshot)
        contract = digest_value({"route_sha256": route_sha, "navigation": meta["navigation"],
                                 "guards": original_plan["test_reservations"], "val_sha256": val_sha,
                                 "prepare_sha256": sha256(BASE / "prepare.py"), "cache_schema": "straight_junction_session_v1",
                                 "input_contract": {k: meta["config"][k] for k in ("vehicle", "cameras", "camera_common", "commands", "quality")}})
        entry = {"session": rel(session), "route_id": meta["route_id"], "split": "train"}
        report = prepare_session(entry, snapshot_path, cache, val_entries, val_sha, trees, 20., contract)
        if report["status"] != "prepared": raise ValueError("new STRAIGHT source touched protected full test/validation corridor")
        rows = read_csv(BASE / report["cache"] / "train.csv")
        done.append({"session": session, "metadata": meta, "route": info, "report": report, "rows": rows})
    return done, skipped


def actual_centers(item):
    """정답·복귀 플래그를 바꾸지 않고 실제 센서 시각의 측정 상태를 기록한다."""
    timeline = Timeline(read_csv(item["session"] / "commands.csv"), "ego_stamp_ns", TEACHER_COMPARE)
    chosen, evidence = [], []
    for row in item["rows"]:
        if row["camera"] != "center" or row["command"] != "2": continue
        match = timeline.bracket(int(row["stamp_ns"]))
        if match is None: raise ValueError("new STRAIGHT center lacks timestamp-matched teacher state")
        crossing, phase = straight_for(row, item["route"]["navigation"])
        chosen.append(row)
        evidence.append({"image_path": row["image_path"], "session_id": row["session_id"], "route_id": row["route_id"],
                         "link_id": crossing["link_id"], "phase": phase,
                         "crossing_length_m": crossing["exit_s_m"] - crossing["entry_s_m"],
                         "route_s_m": float(row["route_s_m"]), "cte_m": linear(match, "cte_m"),
                         "heading_error_deg": linear(match, "heading_error_deg"), "speed_kph": float(row["speed_kph"]),
                         "teacher_steer": float(row["steer"]), "stamp_ns": int(row["stamp_ns"]),
                         "original_recovery_active": int(row["recovery_active"]),
                         "original_noise_steer": float(row["noise_steer"]), "noise_session": bool(item["metadata"]["noise_enabled"])})
    return chosen, evidence


def build(args):
    source_root, parent, capture, output = map(resolve, (args.source_root, args.parent, args.capture_root, args.output))
    output.relative_to(BASE / "runs")
    if output.exists() and args.finalize: raise FileExistsError(output)
    previous = read_json(parent / "prepared/summary.json")
    if sha256(parent / "prepared/train.csv") != previous["train_sha256"]: raise ValueError("V3 parent changed")
    fixed_val_sha = sha256(BASE / "data/prepared/val.csv")
    if sha256(parent / "prepared/val.csv") != fixed_val_sha: raise ValueError("fixed validation changed")
    original = read_csv(parent / "prepared/train.csv")
    metadata, routes = {}, {}
    for row in original:
        if row["session_id"] in metadata: continue
        meta = read_json(resolve(row["image_path"]).parents[2] / "session.json")
        if meta["route_id"] != row["route_id"]: raise ValueError("parent route/session mismatch")
        if meta["route_id"] in routes and routes[meta["route_id"]]["navigation"] != meta["navigation"]:
            raise ValueError("same parent route name has inconsistent navigation")
        metadata[row["session_id"]] = meta
        routes[meta["route_id"]] = {"navigation": meta["navigation"]}
    right_focus_path = resolve(args.right_focus)
    right_focus_lines = right_focus_path.read_text(encoding="utf-8").splitlines()
    right_focus = {resolve(line) for line in right_focus_lines if line}
    if not right_focus or len(right_focus) != len(right_focus_lines): raise ValueError("empty or repeated hard RIGHT focus")
    right_paths = {resolve(row["image_path"]) for row in original if row["command"] == "3" and row["camera"] == "center"}
    if not right_focus.issubset(right_paths): raise ValueError("hard RIGHT focus is not actual V3 RIGHT center data")
    ledger = read_json(source_root / "prepared/source_ledger.json")
    if sha256(BASE / ledger["frozen_route"]) != ledger["frozen_route_sha256"]: raise ValueError("frozen evaluation route changed")
    plan, trees = read_json(BASE / ledger["plan_snapshot"]), guard_trees(ledger)
    readiness = ready_state(capture, args.expected_sessions)
    done, skipped = prepare_completed(capture, ledger, plan, trees)
    new_rows, evidence = [], []
    for item in done:
        metadata.update(item["report"]["metadata"])
        routes[item["metadata"]["route_id"]] = item["route"]
        rows, states = actual_centers(item)
        new_rows.extend(rows)
        evidence.extend(states)
    old_paths = {row["image_path"] for row in original}
    if len({row["image_path"] for row in new_rows}) != len(new_rows) or any(row["image_path"] in old_paths for row in new_rows):
        raise ValueError("duplicate new physical center or overlap with V3")
    by_path = {row["image_path"]: row for row in new_rows}
    normal = [by_path[e["image_path"]] for e in evidence if not e["noise_session"] and not e["original_recovery_active"] and e["phase"] == "inside"]
    # 직진의 작은 실제 교란도 포함한다. 큰 오차 기준은 별도 진단에만 사용한다.
    recovery = [by_path[e["image_path"]] for e in evidence if e["original_recovery_active"]]
    measured_recovery = [e for e in evidence if e["original_recovery_active"] and
                         (abs(e["cte_m"]) >= args.min_cte_m or abs(e["heading_error_deg"]) >= args.min_heading_deg)]
    focus_per_kind = min(len(normal), len(recovery))
    straight_focus_rows = spread_subset(normal, focus_per_kind) + spread_subset(recovery, focus_per_kind) if focus_per_kind else []
    pairing = Counter(item["metadata"]["route_id"] + ":" + ("noise" if item["metadata"]["noise_enabled"] else "normal") for item in done)
    expected_pairs = {rid + ":" + mode for rid in APPROVED_ROUTES for mode in ("normal", "noise")}
    audit = {"capture_readiness": readiness, "approved_routes": list(APPROVED_ROUTES), "session_pair_counts": dict(pairing),
             "quality_passed_sessions": len(done), "skipped_sessions": skipped, "new_straight_center_frames": len(new_rows),
             "link_phase_counts": dict(Counter(e["link_id"] + ":" + e["phase"] for e in evidence)),
             "recovery_active_center_frames": sum(e["original_recovery_active"] for e in evidence),
             "eligible_actual_recovery_focus": len(recovery), "measured_error_recovery_centers": len(measured_recovery),
             "eligible_normal_inside_focus": len(normal),
             "focus_per_kind": focus_per_kind, "straight_focus_count": len(straight_focus_rows), "signed_state": signed_summary(evidence),
             "diagnostic_error_thresholds": {"min_abs_cte_m": args.min_cte_m, "min_abs_heading_deg": args.min_heading_deg, "combination": "OR; diagnostic only, does not exclude actual recovery"},
             "source_sessions": [{"session_id": item["session"].name, "route_id": item["metadata"]["route_id"],
                                  "noise_enabled": item["metadata"]["noise_enabled"], "prepared_cache": item["report"]["cache"],
                                  "quality_report_sha256": sha256(item["session"] / "quality_report.json"),
                                  "session_sha256": sha256(item["session"] / "session.json")} for item in done]}
    if not args.finalize: return audit
    if not readiness["ready"] or set(pairing) != expected_pairs or any(n != 1 for n in pairing.values()):
        raise ValueError("expected exactly four approved normal/noise pairs; captures are unfinished, missing, or duplicated")
    if not new_rows or not focus_per_kind: raise ValueError("no measured real recovery available for balanced STRAIGHT focus")

    def crossing(row): return straight_for(row, routes[row["route_id"]]["navigation"])
    frequencies = Counter(crossing(r)[0]["link_id"] for r in original if r["camera"] == "center" and r["command"] == "2")
    slots = [(i, r) for i, r in enumerate(original) if r["camera"] == "center" and r["command"] == "2" and not int(r["recovery_active"])]
    result, used, changes = list(original), set(), []
    for row in sorted(new_rows, key=lambda r: (crossing(r)[1], r["session_id"], int(r["stamp_ns"]))):
        phase = crossing(row)[1]
        candidates = [(i, old) for i, old in slots if i not in used and crossing(old)[1] == phase]
        if not candidates: raise ValueError("insufficient old non-recovery STRAIGHT centers in matching phase")
        i, old = min(candidates, key=lambda pair: (-frequencies[crossing(pair[1])[0]["link_id"]], abs(float(pair[1]["steer"])), pair[0]))
        result[i] = row
        used.add(i)
        changes.append({"index": i, "old_image_path": old["image_path"], "new_image_path": row["image_path"], "phase": phase})
    before, after = count_rows(original, metadata), count_rows(result, metadata)
    for key in ("center_counts", "camera_counts", "train_rows"):
        if before[key] != after[key]: raise ValueError("fixed dataset cardinality changed")
    if Counter((r["command"], r["camera"]) for r in result) != Counter((r["command"], r["camera"]) for r in original):
        raise ValueError("command/camera row count changed")
    for i, (old, new) in enumerate(zip(original, result)):
        if i not in used and old != new: raise ValueError("untouched index changed")
        if i in used and ((old["command"], old["camera"]) != ("2", "center") or crossing(old)[1] != crossing(new)[1]):
            raise ValueError("non-STRAIGHT-center or matching phase changed")
    paths = {resolve(r["image_path"]) for r in result}
    if len(paths) != len(result): raise ValueError("duplicate physical image path")
    minimum = distances([[float(r["x_m"]), float(r["y_m"])] for r in result], trees)
    if min(minimum.values()) < 20.: raise ValueError("new data overlaps protected test/validation buffer")
    straight_focus = {resolve(r["image_path"]) for r in straight_focus_rows}
    if not (right_focus | straight_focus).issubset(paths): raise ValueError("focus contains omitted training frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".straight_variant_", dir=str(output.parent)))
    try:
        prepared = stage / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", result)
        shutil.copy2(parent / "prepared/val.csv", prepared / "val.csv")
        if sha256(prepared / "val.csv") != fixed_val_sha: raise ValueError("fixed validation changed while copying")
        shutil.copy2(right_focus_path, stage / "focus_hard_right.txt")
        # 기존 RIGHT 목록의 값과 순서를 그대로 두고 새 STRAIGHT만 뒤에 붙인다.
        combined = right_focus_lines + [rel(p) for p in sorted(straight_focus)]
        (stage / "focus_right_straight.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")
        after.update(status="prepared", variant="v3_plus_real_straight_junction_centers", created_utc=datetime.now(timezone.utc).isoformat(),
                     parent_prepared=rel(parent / "prepared"), parent_train_sha256=previous["train_sha256"],
                     train_sha256=sha256(prepared / "train.csv"), val_sha256=fixed_val_sha, center_duration_s=5400.,
                     straight_capture_audit=audit, added_center_frames=len(new_rows), changed_row_indices_count=len(used),
                     unchanged_sample_indices_count=len(original)-len(used), tagged_recovery_before=before["recovery_active_center_counts"],
                     all_right_rows_all_sides_other_commands_and_index_unchanged=True, phase_preserved_at_each_changed_index=True,
                     original_teacher_targets_and_recovery_flags_unchanged=True, protected_min_distances_m=minimum,
                     retained_hard_right_focus_sha256=sha256(stage / "focus_hard_right.txt"),
                     focus_manifest="focus_right_straight.txt", focus_manifest_sha256=sha256(stage / "focus_right_straight.txt"),
                     focus_counts={"RIGHT": len(right_focus), "STRAIGHT_normal_inside": focus_per_kind, "STRAIGHT_actual_recovery": focus_per_kind, "total": len(combined)},
                     focus_quota_interpretation="Existing command-balanced sampler with focus-per-command=10 reserves RIGHT10 and STRAIGHT10 independently per 120-sample batch. STRAIGHT focus has a 1:1 normal/recovery population; individual batches do not enforce5:5. Other command populations remain fixed.",
                     interpretation="Only STRAIGHT physical center observations change. Existing V3 RIGHT data and hard-RIGHT focus are retained exactly. Full approach/traverse/exit windows and unchanged real teacher targets are used; no inference, side labels, flags, or validation data are modified.")
        write_json(prepared / "summary.json", after)
        write_json(stage / "new_straight_evidence.json", evidence)
        write_json(stage / "focus_straight_kinds.json", {"normal_inside": [r["image_path"] for r in straight_focus_rows[:focus_per_kind]], "actual_recovery": [r["image_path"] for r in straight_focus_rows[focus_per_kind:]]})
        write_json(stage / "changes.json", changes)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return after


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="runs/east_outer_90min")
    parser.add_argument("--parent", default="runs/east_outer_curve_right_recovery_v1")
    parser.add_argument("--right-focus", default="runs/east_outer_curve_right_recovery_v1/focus_hard_right.txt")
    parser.add_argument("--capture-root", default="runs/east_outer_sharp_right_data/straight_capture")
    parser.add_argument("--output", default="runs/east_outer_straight_junction_v1")
    parser.add_argument("--expected-sessions", type=int, default=8)
    parser.add_argument("--min-cte-m", type=float, default=.12)
    parser.add_argument("--min-heading-deg", type=float, default=3.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.expected_sessions < 1 or not all(np.isfinite(x) and x > 0 for x in (args.min_cte_m, args.min_heading_deg)):
        parser.error("positive expected-session count and measured-error thresholds required")
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
