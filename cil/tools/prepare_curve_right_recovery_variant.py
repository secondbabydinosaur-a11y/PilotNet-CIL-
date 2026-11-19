"""완료된 실제 우회전 교란에서 상태 오차가 있는 복귀 center만 v2에 추가한다.

기존 sharp focus 표본·모든 side·다른 명령·행 index·검증 데이터는 유지한다.
여러 capture-root를 지원하며 LEFT 수집은 별도 보관한다. --finalize 전에는
집계만 하고 최종 학습 목록은 만들지 않는다. 수집·학습·차량 제어는 하지 않는다.
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
from prepare import Timeline, TEACHER_COMPARE, linear
from tools.prepare_east_outer_90min import count_rows, distances, read_csv, read_json, rel, resolve, sha256, write_json, write_rows
from tools.prepare_follow_recovery_variant import guard_trees
from tools.prepare_sharp_right_variant import prepare_new, turn_for


def signed_summary(evidence):
    if not evidence: return {"samples": 0}
    cte = np.array([e["cte_m"] for e in evidence])
    heading = np.array([e["heading_error_deg"] for e in evidence])
    return {"samples": len(evidence), "cte_p05_p50_p95_m": np.percentile(cte, [5, 50, 95]).tolist(),
            "cte_min_max_m": [float(cte.min()), float(cte.max())],
            "heading_p05_p50_p95_deg": np.percentile(heading, [5, 50, 95]).tolist(),
            "heading_min_max_deg": [float(heading.min()), float(heading.max())],
            "cte_negative": int(sum(cte < -.12)), "cte_positive": int(sum(cte > .12)),
            "heading_negative": int(sum(heading < -3)), "heading_positive": int(sum(heading > 3))}


def recovery_candidates(item, cte_min, heading_min):
    """원래 플래그와 실제 센서 시각의 signed 상태를 모두 검사한다."""
    timeline = Timeline(read_csv(item["session"] / "commands.csv"), "ego_stamp_ns", TEACHER_COMPARE)
    chosen, evidence = [], []
    for row in item["rows"]:
        if row["camera"] != "center" or row["command"] != "3" or int(row["recovery_active"]) != 1:
            continue
        match = timeline.bracket(int(row["stamp_ns"]))
        if match is None: raise ValueError("prepared recovery center has no real matched command state")
        cte, heading = linear(match, "cte_m"), linear(match, "heading_error_deg")
        if abs(cte) < cte_min and abs(heading) < heading_min: continue
        turn, phase = turn_for(row, item["route"]["navigation"])
        chosen.append(row)
        evidence.append({"image_path": row["image_path"], "session_id": row["session_id"],
                         "route_id": row["route_id"], "link_id": turn["link_id"], "phase": phase,
                         "cte_m": cte, "heading_error_deg": heading, "speed_kph": float(row["speed_kph"]),
                         "teacher_steer": float(row["steer"]), "stamp_ns": int(row["stamp_ns"]),
                         "original_recovery_active": int(row["recovery_active"]),
                         "original_noise_steer": float(row["noise_steer"])})
    return chosen, evidence


def ready_state(capture, expected):
    progress_path = capture / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {"accepted": [], "errors": []}
    pending = [path.name for path in (capture / "raw").iterdir()
               if path.is_dir() and not (path / "quality_report.json").is_file()]
    accepted = len({item["session"] for item in progress.get("accepted", [])})
    return {"capture_root": rel(capture), "accepted_sessions": accepted, "expected_sessions": expected,
            "pending_sessions_not_read": pending, "errors": progress.get("errors", []),
            "ready": accepted >= expected and not pending and not progress.get("errors")}


def build(args):
    source_root, parent = resolve(args.source_root), resolve(args.parent)
    output = resolve(args.output)
    output.relative_to(BASE / "runs")
    if output.exists() and args.finalize: raise FileExistsError(output)
    parent_summary = read_json(parent / "prepared/summary.json")
    if sha256(parent / "prepared/train.csv") != parent_summary["train_sha256"]:
        raise ValueError("v2 parent CSV changed")
    if sha256(parent / "prepared/val.csv") != sha256(BASE / "data/prepared/val.csv"):
        raise ValueError("fixed validation changed")
    ledger = read_json(source_root / "prepared/source_ledger.json")
    if sha256(BASE / ledger["frozen_route"]) != ledger["frozen_route_sha256"]:
        raise ValueError("entire evaluation corridor changed")
    plan = read_json(BASE / ledger["plan_snapshot"])
    routes = dict(plan["routes"])
    metadata = {}
    for report in [ledger["baseline"]] + [r for r in ledger["accepted_sessions"] if r["status"] == "prepared"]:
        metadata.update(report["metadata"])
    # v2의 실제 급우회전 12개 수집 코호트도 phase 식별과 제거 금지에 사용한다.
    sharp = read_json(resolve(parent_summary["parent_prepared"]) / "summary.json")
    approved = set(sharp["sharp_right_audit"]["approved_routes"])
    original_capture = BASE / "runs/east_outer_sharp_right_data/capture/raw"
    sharp_session_ids = {item["session_id"] for item in sharp["sharp_right_audit"]["source_sessions"]}
    for item in sharp["sharp_right_audit"]["source_sessions"]:
        meta = read_json(original_capture / item["session_id"] / "session.json")
        metadata[item["session_id"]] = meta
        routes[meta["route_id"]] = {"navigation": meta["navigation"]}
    original = read_csv(parent / "prepared/train.csv")
    original_paths = {row["image_path"] for row in original}
    parent_focus_path = resolve(args.parent_focus)
    parent_focus = {resolve(line.strip()) for line in parent_focus_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    expected_focus = {resolve(row["image_path"]) for row in original if row["camera"] == "center" and row["command"] == "3"}
    if not parent_focus or not parent_focus.issubset(expected_focus):
        raise ValueError("parent focus contains images outside actual v2 RIGHT center data")
    actual_sharp_focus = {resolve(row["image_path"]) for row in original
                          if row["camera"] == "center" and row["command"] == "3" and row["session_id"] in sharp_session_ids}
    if parent_focus != actual_sharp_focus:
        raise ValueError("parent focus must preserve every actual v2 sharp-center observation")
    trees = guard_trees(ledger)
    capture_roots = args.capture_root or ["runs/east_outer_sharp_right_data/curve_capture"]
    readiness, done, skipped = [], [], []
    for value in capture_roots:
        capture = resolve(value)
        readiness.append(ready_state(capture, args.expected_sessions))
        sessions, ignored, _ = prepare_new(capture, ledger, plan, approved, trees)
        done.extend(sessions)
        skipped.extend(ignored)  # 승인 목록 밖의 LEFT 세션은 학습 목록에 넣지 않는다.
    new_rows, evidence = [], []
    for item in done:
        metadata.update(item["report"]["metadata"])
        routes[item["metadata"]["route_id"]] = item["route"]
        selected, actual = recovery_candidates(item, args.min_cte_m, args.min_heading_deg)
        new_rows.extend(selected)
        evidence.extend(actual)
    if len({row["image_path"] for row in new_rows}) != len(new_rows):
        raise ValueError("capture roots refer to duplicate physical recovery frames")
    if any(row["image_path"] in original_paths for row in new_rows):
        raise ValueError("recovery input is already present in the parent dataset")
    report = {"capture_readiness": readiness, "all_capture_roots_ready": all(r["ready"] for r in readiness),
              "quality_passed_right_sessions": len(done), "skipped_sessions_including_left": skipped,
              "meaningful_right_recovery_centers": len(new_rows),
              "thresholds": {"min_abs_cte_m": args.min_cte_m, "min_abs_heading_deg": args.min_heading_deg, "combination": "OR; original recovery_active must equal1"},
              "signed_state": signed_summary(evidence),
              "link_phase_counts": dict(Counter(e["link_id"] + ":" + e["phase"] for e in evidence)),
              "parent_focus_centers": len(parent_focus)}
    if not args.finalize: return report
    if not report["all_capture_roots_ready"]:
        raise ValueError("captures are still active or incomplete; use --prepare-only and wait: " + str(readiness))
    if not new_rows: raise ValueError("no meaningful real RIGHT recovery states; no data was fabricated")

    def phase(row):
        return turn_for(row, routes[row["route_id"]]["navigation"])[1]
    frequencies = Counter(turn_for(row, routes[row["route_id"]]["navigation"])[0]["link_id"]
                          for row in original if row["camera"] == "center" and row["command"] == "3")
    slots = [(i, row) for i, row in enumerate(original)
             if row["camera"] == "center" and row["command"] == "3" and not int(row["recovery_active"])
             and resolve(row["image_path"]) not in parent_focus and abs(float(row["steer"]) * 40.) < 20.]
    result, used, changes = list(original), set(), []
    for row in sorted(new_rows, key=lambda r: (phase(r), -abs(float(r["steer"])), r["image_path"])):
        candidates = [(i, old) for i, old in slots if i not in used and phase(old) == phase(row)]
        if not candidates: raise ValueError("insufficient weak non-recovery RIGHT slots in the same phase")
        def priority(item):
            i, old = item
            link = turn_for(old, routes[old["route_id"]]["navigation"])[0]["link_id"]
            return (link != "A2256W000318", -frequencies[link], abs(float(old["steer"])), i)
        i, old = min(candidates, key=priority)
        result[i] = row
        used.add(i)
        changes.append({"index": i, "old_image_path": old["image_path"], "new_image_path": row["image_path"], "phase": phase(row)})
    before, after = count_rows(original, metadata), count_rows(result, metadata)
    for key in ("center_counts", "camera_counts", "train_rows"):
        if before[key] != after[key]: raise ValueError("fixed dataset cardinality changed")
    for i, (old, new) in enumerate(zip(original, result)):
        if i not in used and old != new: raise ValueError("nonselected row changed")
        if i in used and ((old["command"], old["camera"]) != ("3", "center") or phase(old) != phase(new)):
            raise ValueError("non-RIGHT or phase index changed")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in result], trees)
    if min(minimum.values()) < 20: raise ValueError("recovery variant touches protected test/validation data")
    if len({resolve(row["image_path"]) for row in result}) != len(result): raise ValueError("duplicate physical image")
    focus = parent_focus | {resolve(row["image_path"]) for row in new_rows}
    if not focus.issubset({resolve(row["image_path"]) for row in result}): raise ValueError("focus includes omitted/nontraining frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".curve_recovery_", dir=str(output.parent)))
    try:
        prepared = stage / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", result)
        shutil.copy2(parent / "prepared/val.csv", prepared / "val.csv")
        if sha256(prepared / "val.csv") != sha256(BASE / "data/prepared/val.csv"):
            raise ValueError("fixed validation changed during variant creation")
        (stage / "focus_right_center.txt").write_text("".join(rel(path) + "\n" for path in sorted(focus)), encoding="utf-8")
        after.update(status="prepared", variant="v2_plus_real_curve_phase_right_recovery", created_utc=datetime.now(timezone.utc).isoformat(),
                     parent_prepared=rel(parent / "prepared"), parent_train_sha256=parent_summary["train_sha256"],
                     train_sha256=sha256(prepared / "train.csv"), val_sha256=sha256(prepared / "val.csv"), center_duration_s=5400.,
                     curve_capture_audit=report, added_recovery_center_frames=len(new_rows), changed_row_indices_count=len(used),
                     unchanged_sample_indices_count=len(original)-len(used),
                     tagged_recovery_before=before["recovery_active_center_counts"],
                     all_other_commands_side_images_and_index_unchanged=True,
                     phase_preserved_at_each_changed_index=True,
                     no_fake_side_or_recovery_flags=True, protected_min_distances_m=minimum,
                     focus_manifest="focus_right_center.txt", focus_manifest_sha256=sha256(stage / "focus_right_center.txt"),
                     focus_center_counts={"v2_original_sharp": len(parent_focus), "new_actual_recovery": len(new_rows), "total": len(focus)},
                     interpretation="Only actual QA-passed RIGHT recovery-active center states meeting measured-error thresholds are added. Recovery and noise-session counts intentionally rise; no claim of matching the old distribution or teaching LEFT from this collection.")
        write_json(prepared / "summary.json", after)
        write_json(stage / "new_recovery_evidence.json", evidence)
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
    parser.add_argument("--parent", default="runs/east_outer_sharp_right_center_v2_ordered")
    parser.add_argument("--parent-focus", default="runs/east_outer_sharp_right_data/focus_right_center.txt")
    parser.add_argument("--capture-root", action="append")
    parser.add_argument("--expected-sessions", type=int, default=8)
    parser.add_argument("--output", default="runs/east_outer_curve_right_recovery_v1")
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
