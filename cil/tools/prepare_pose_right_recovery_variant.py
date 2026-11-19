"""V3에 실제 초기 위치·헤딩 교란을 포함한 우회전 center만 추가한다.

완료된 QA 합격 세션만 읽는다. 초기 교란 표본을 recovery_active=1로
재기록하지 않으며 teacher 정답·side·다른 명령·검증 데이터는 그대로 둔다.
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
from tools.prepare_curve_right_recovery_variant import ready_state, signed_summary
from tools.prepare_east_outer_90min import count_rows, distances, read_csv, read_json, rel, resolve, sha256, write_json, write_rows
from tools.prepare_follow_recovery_variant import guard_trees
from tools.prepare_sharp_right_variant import prepare_new, turn_for

APPROVED_PARENTS = {"POSE_SHARP_R216": "SHARP_R216", "POSE_SHARP_R265": "SHARP_R265"}


def verify_trimmed_geometry(capture):
    """새 시험 도로가 아닌, 이미 승인한 학습 도로의 정확한 부분 경로인지 검사한다."""
    config = read_json(capture / "config.json")
    plan = read_json(resolve(config["paths"]["route_plan"]))
    original_config = read_json(BASE / "runs/east_outer_sharp_right_data/capture/config.json")
    original_plan = read_json(resolve(original_config["paths"]["route_plan"]))
    audits = []
    for route_id, parent_id in APPROVED_PARENTS.items():
        info, parent = plan["routes"][route_id], original_plan["routes"][parent_id]
        if info.get("parent_training_route") != parent_id:
            raise ValueError("pose route is not linked to the independently approved training approach")
        actual, original = resolve(info["route_csv"]), resolve(parent["route_csv"])
        if sha256(actual) != info["route_sha256"] or sha256(original) != parent["route_sha256"]:
            raise ValueError("approved route CSV changed")
        new = np.loadtxt(actual, delimiter=",", skiprows=1)
        old = np.loadtxt(original, delimiter=",", skiprows=1)
        if len(info["navigation"]) != 1 or len(parent["navigation"]) != 1:
            raise ValueError("pose experiment must contain one approved right turn")
        turn, previous = info["navigation"][0], parent["navigation"][0]
        if turn["command"] != "right" or turn["link_id"] != previous["link_id"]:
            raise ValueError("pose route changed the approved turn direction or link")
        shift = previous["exit_s_m"] - turn["exit_s_m"]
        if shift < 0 or new[-1, 0] + shift > old[-1, 0] + 1e-6:
            raise ValueError("pose route extends beyond the approved training route")
        expected = np.column_stack([np.interp(new[:, 0] + shift, old[:, 0], old[:, k]) for k in (1, 2, 3)])
        error = float(np.linalg.norm(expected - new[:, 1:4], axis=1).max())
        if error > 1e-5 or abs(turn["entry_s_m"] - max(0., previous["entry_s_m"] - shift)) > 1e-6:
            raise ValueError("pose route is not an exact station-shifted training geometry")
        audits.append({"route_id": route_id, "approved_parent": parent_id, "station_shift_m": shift,
                       "maximum_xyz_error_m": error, "route_sha256": sha256(actual), "link_id": turn["link_id"]})
    return audits


def pose_candidates(item, cte_min, heading_min):
    """원래 복귀 플래그 또는 첫 noise 이전의 실제 초기 교란 증거를 사용한다."""
    meta = item["metadata"]
    if abs(float(meta.get("initial_offset_m", 0.))) < .2 or abs(float(meta.get("initial_heading_offset_deg", 0.))) < 3.:
        raise ValueError("pose source does not declare the intended real initial perturbation")
    command_rows = read_csv(item["session"] / "commands.csv")
    timeline = Timeline(command_rows, "ego_stamp_ns", TEACHER_COMPARE)
    first_noise = min((int(r["ego_stamp_ns"]) for r in command_rows if abs(float(r["noise_steer"])) > 1e-7), default=None)
    if first_noise is None: raise ValueError("pose source contains no actual in-turn perturbation")
    selected, evidence = [], []
    for row in item["rows"]:
        if row["camera"] != "center" or row["command"] != "3": continue
        stamp, active = int(row["stamp_ns"]), bool(int(row["recovery_active"]))
        if not active and stamp >= first_noise: continue
        match = timeline.bracket(stamp)
        if match is None: raise ValueError("pose center lacks actual timestamp-matched teacher state")
        cte, heading = linear(match, "cte_m"), linear(match, "heading_error_deg")
        if abs(cte) < cte_min and abs(heading) < heading_min: continue
        turn, phase = turn_for(row, item["route"]["navigation"])
        selected.append(row)
        evidence.append({"image_path": row["image_path"], "session_id": row["session_id"], "route_id": row["route_id"],
                         "link_id": turn["link_id"], "phase": phase, "cte_m": cte, "heading_error_deg": heading,
                         "speed_kph": float(row["speed_kph"]), "teacher_steer": float(row["steer"]),
                         "stamp_ns": stamp, "original_recovery_active": int(active),
                         "original_noise_steer": float(row["noise_steer"]),
                         "evidence_kind": "tagged_in_turn_recovery" if active else "initial_pose_before_first_noise",
                         "first_noise_stamp_ns": first_noise, "declared_initial_offset_m": meta["initial_offset_m"],
                         "declared_initial_heading_deg": meta["initial_heading_offset_deg"]})
    return selected, evidence


def build(args):
    source_root, parent, capture, output = map(resolve, (args.source_root, args.parent, args.capture_root, args.output))
    output.relative_to(BASE / "runs")
    if output.exists() and args.finalize: raise FileExistsError(output)
    previous = read_json(parent / "prepared/summary.json")
    if sha256(parent / "prepared/train.csv") != previous["train_sha256"]:
        raise ValueError("V3 parent training CSV changed")
    val_sha = sha256(BASE / "data/prepared/val.csv")
    if sha256(parent / "prepared/val.csv") != val_sha: raise ValueError("fixed validation changed")
    focus_path = parent / previous["focus_manifest"]
    if sha256(focus_path) != previous["focus_manifest_sha256"]: raise ValueError("V3 focus manifest changed")
    original = read_csv(parent / "prepared/train.csv")
    # 실제 원본 이미지 경로에서 세션 metadata를 읽어 모든 코호트를 빠짐없이 계산한다.
    metadata, routes = {}, {}
    for row in original:
        if row["session_id"] in metadata: continue
        meta = read_json(resolve(row["image_path"]).parents[2] / "session.json")
        if meta["route_id"] != row["route_id"]: raise ValueError("parent session/route identity mismatch")
        if meta["route_id"] in routes and routes[meta["route_id"]]["navigation"] != meta["navigation"]:
            raise ValueError("same parent route name has inconsistent navigation")
        metadata[row["session_id"]] = meta
        routes[meta["route_id"]] = {"navigation": meta["navigation"]}
    old_paths = {row["image_path"] for row in original}
    parent_focus = {resolve(line) for line in focus_path.read_text(encoding="utf-8").splitlines() if line}
    right_centers = {resolve(row["image_path"]) for row in original if row["command"] == "3" and row["camera"] == "center"}
    if not parent_focus.issubset(right_centers): raise ValueError("parent focus is not actual selected RIGHT center data")
    ledger = read_json(source_root / "prepared/source_ledger.json")
    if sha256(BASE / ledger["frozen_route"]) != ledger["frozen_route_sha256"]:
        raise ValueError("frozen full test corridor changed")
    plan, trees = read_json(BASE / ledger["plan_snapshot"]), guard_trees(ledger)
    readiness = ready_state(capture, args.expected_sessions)
    geometry_audit = verify_trimmed_geometry(capture)
    done, skipped, _ = prepare_new(capture, ledger, plan, set(APPROVED_PARENTS), trees)
    new_rows, evidence = [], []
    for item in done:
        metadata.update(item["report"]["metadata"])
        routes[item["metadata"]["route_id"]] = item["route"]
        chosen, state = pose_candidates(item, args.min_cte_m, args.min_heading_deg)
        new_rows.extend(chosen)
        evidence.extend(state)
    if len({row["image_path"] for row in new_rows}) != len(new_rows) or any(row["image_path"] in old_paths for row in new_rows):
        raise ValueError("duplicate physical pose observation or parent overlap")
    audit = {"capture_readiness": readiness, "approved_trimmed_training_geometry": geometry_audit,
             "quality_passed_sessions": len(done), "skipped_sessions": skipped, "eligible_center_frames": len(new_rows),
             "evidence_kind_counts": dict(Counter(e["evidence_kind"] for e in evidence)),
             "original_recovery_flag_counts": dict(Counter(str(e["original_recovery_active"]) for e in evidence)),
             "link_phase_counts": dict(Counter(e["link_id"] + ":" + e["phase"] for e in evidence)),
             "signed_state": signed_summary(evidence),
             "thresholds": {"min_abs_cte_m": args.min_cte_m, "min_abs_heading_deg": args.min_heading_deg, "combination": "OR"},
             "source_sessions": [{"session_id": item["session"].name, "route_id": item["metadata"]["route_id"],
                                  "quality_report_sha256": sha256(item["session"] / "quality_report.json"),
                                  "raw_session_sha256": sha256(item["session"] / "session.json"),
                                  "prepared_cache": item["report"]["cache"]} for item in done]}
    if not args.finalize: return audit
    if not readiness["ready"]: raise ValueError("pose capture still active/incomplete; use --prepare-only: " + str(readiness))
    if not new_rows: raise ValueError("no QA-passed real pose recovery centers")

    def phase(row): return turn_for(row, routes[row["route_id"]]["navigation"])[1]
    def link(row): return turn_for(row, routes[row["route_id"]]["navigation"])[0]["link_id"]
    frequencies = Counter(link(r) for r in original if r["camera"] == "center" and r["command"] == "3")
    slots = [(i, row) for i, row in enumerate(original) if row["camera"] == "center" and row["command"] == "3"
             and not int(row["recovery_active"]) and resolve(row["image_path"]) not in parent_focus and abs(float(row["steer"]) * 40.) < 20.]
    result, used, changes = list(original), set(), []
    for row in sorted(new_rows, key=lambda r: (phase(r), -abs(float(r["steer"])), r["image_path"])):
        candidates = [(i, old) for i, old in slots if i not in used and phase(old) == phase(row)]
        if not candidates: raise ValueError("insufficient weak non-recovery RIGHT center slots in the same phase")
        i, old = min(candidates, key=lambda pair: (link(pair[1]) != "A2256W000318", -frequencies[link(pair[1])], abs(float(pair[1]["steer"])), pair[0]))
        result[i] = row
        used.add(i)
        changes.append({"index": i, "old_image_path": old["image_path"], "new_image_path": row["image_path"], "phase": phase(row)})
    before, after = count_rows(original, metadata), count_rows(result, metadata)
    for key in ("center_counts", "camera_counts", "train_rows"):
        if before[key] != after[key]: raise ValueError("fixed command/camera cardinality changed")
    for i, (old, new) in enumerate(zip(original, result)):
        if i not in used and old != new: raise ValueError("an untouched row changed")
        if i in used and ((old["command"], old["camera"]) != ("3", "center") or phase(old) != phase(new)):
            raise ValueError("non-RIGHT center or phase sample index changed")
    if len({resolve(row["image_path"]) for row in result}) != len(result): raise ValueError("duplicate physical training image")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in result], trees)
    if min(minimum.values()) < 20.: raise ValueError("pose observations touch protected test/validation data")
    focus = parent_focus | {resolve(row["image_path"]) for row in new_rows}
    if not focus.issubset({resolve(row["image_path"]) for row in result}): raise ValueError("focus references omitted observations")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pose_recovery_", dir=str(output.parent)))
    try:
        prepared = stage / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", result)
        shutil.copy2(parent / "prepared/val.csv", prepared / "val.csv")
        if sha256(prepared / "val.csv") != val_sha: raise ValueError("fixed validation changed while copying")
        (stage / "focus_right_center.txt").write_text("".join(rel(p) + "\n" for p in sorted(focus)), encoding="utf-8")
        after.update(status="prepared", variant="v3_plus_real_initial_pose_right_recovery", created_utc=datetime.now(timezone.utc).isoformat(),
                     parent_prepared=rel(parent / "prepared"), parent_train_sha256=previous["train_sha256"],
                     train_sha256=sha256(prepared / "train.csv"), val_sha256=val_sha, center_duration_s=5400.,
                     pose_capture_audit=audit, added_center_frames=len(new_rows), changed_row_indices_count=len(used),
                     unchanged_sample_indices_count=len(original)-len(used), tagged_recovery_before=before["recovery_active_center_counts"],
                     all_other_commands_side_images_and_index_unchanged=True, phase_preserved_at_each_changed_index=True,
                     original_teacher_targets_and_recovery_flags_unchanged=True, protected_min_distances_m=minimum,
                     focus_manifest="focus_right_center.txt", focus_manifest_sha256=sha256(stage / "focus_right_center.txt"),
                     focus_center_counts={"v3_retained": len(parent_focus), "new_pose_evidence": len(new_rows), "total": len(focus)},
                     interpretation="Experiment-only real pose recovery centers. Initial states retain recovery_active=0; only measured error before the first real injection admits those rows. Tagged in-turn states retain their original flags. No synthetic targets or side-camera labels are added.")
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
    parser.add_argument("--parent", default="runs/east_outer_curve_right_recovery_v1")
    parser.add_argument("--capture-root", default="runs/east_outer_sharp_right_data/pose_curve_capture")
    parser.add_argument("--output", default="runs/east_outer_pose_right_recovery_v1")
    parser.add_argument("--expected-sessions", type=int, default=4)
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
