"""검증된 급우회전 새 실측으로 v2의 완만한 318 우회전만 격리 교체한다.

명령·카메라·단계·noise/recovery 층을 맞추고 원래 sample index를 유지한다.
수집 중 파일은 읽지 않으며 수집·학습·시뮬레이터 제어는 실행하지 않는다.
"""

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools.build_evaluation_routes import route_from_xyz
from tools.prepare_east_outer_90min import (
    PERIOD_NS, count_rows, digest_value, distances, prepare_session, read_csv, read_json,
    rel, resolve, sha256, spread_subset, write_json, write_rows,
)
from tools.prepare_follow_recovery_variant import aligned_rows, guard_trees, preserve_sample_indices

RELAXED_LINK = "A2256W000318"
BLOCKED_LINKS = {"A2256W000079", "A2256W000089"}  # 현재 도로 화살표 검토에서 부적합 판정을 받은 접근이다.


def turn_for(row, navigation):
    """Teacher가 명령을 부여한 현재 우회전과 접근·내부·출구 유지 단계를 구한다."""
    station = float(row["route_s_m"])
    turns = [n for n in navigation if n["command"] == "right"]
    inside = [n for n in turns if n["entry_s_m"] <= station <= n["exit_s_m"]]
    if inside: return inside[0], "inside"
    upcoming = [n for n in turns if 0 < n["entry_s_m"] - station <= n["approach_m"] + .001]
    if upcoming: return min(upcoming, key=lambda n: n["entry_s_m"]), "approach"
    passed = [n for n in turns if 0 < station - n["exit_s_m"] <= n["exit_hold_m"] + .001]
    if passed: return passed[-1], "exit_hold"
    raise ValueError("RIGHT label has no matching right-turn command window")


def radius_proxy(link):
    route = route_from_xyz(link["points"])
    station = route[:, 0]
    back, front = np.maximum(station[0], station - 1.5), np.minimum(station[-1], station + 1.5)
    chord = np.column_stack([np.interp(front, station, route[:, k]) - np.interp(back, station, route[:, k]) for k in (1, 2)])
    yaw = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    return float(1.0 / max(np.abs(np.gradient(yaw, station)).max(), 1e-9))


def time_key(row, origins):
    return (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS, row["command"])


def phase_counts(rows, route_info):
    counts = Counter()
    for row in rows:
        if row["camera"] == "center" and row["command"] == "3":
            turn, phase = turn_for(row, route_info[row["route_id"]]["navigation"])
            counts[turn["link_id"] + ":" + phase] += 1
    return dict(sorted(counts.items()))


def load_parent(source_root, parent):
    pool_ledger_path = source_root / "prepared/source_ledger.json"
    source = read_json(pool_ledger_path)
    plan = read_json(BASE / source["plan_snapshot"])
    summary = read_json(parent / "prepared/summary.json")
    if sha256(parent / "prepared/train.csv") != summary["train_sha256"]:
        raise ValueError("parent v2 training CSV changed")
    if sha256(parent / "prepared/val.csv") != sha256(BASE / "data/prepared/val.csv"):
        raise ValueError("parent validation differs from the fixed validation")
    if sha256(BASE / source["frozen_route"]) != source["frozen_route_sha256"]:
        raise ValueError("entire fixed evaluation corridor changed")
    metadata, origins = {}, {}
    for report in [source["baseline"]] + [x for x in source["accepted_sessions"] if x["status"] == "prepared"]:
        metadata.update(report["metadata"])
        origins.update({sid: int(stamp) for sid, stamp in report["origins_ns"].items()})
    return source, plan, summary, metadata, origins, read_csv(parent / "prepared/train.csv")


def prepare_new(capture, source, original_plan, approved, trees):
    config = read_json(capture / "config.json")
    current_plan = read_json(resolve(config["paths"]["route_plan"]))
    map_path = Path(current_plan["source"]) / "link_set.json"
    if sha256(map_path) != current_plan["source_hashes"]["link_set.json"]:
        raise ValueError("fixed MGeo geometry changed")
    links = {row["idx"]: row for row in read_json(map_path) if not row["lazy_init"]}
    reference_radius = radius_proxy(links[RELAXED_LINK])
    baseline = read_json(BASE / "data/collection_completion.json")
    val_entries = [row for row in baseline["accepted"] if row["split"] == "validation"]
    val_sha = sha256(BASE / "data/prepared/val.csv")
    cache_root = capture / "prepared_cache"
    cache_root.mkdir(exist_ok=True)
    snapshots = cache_root / "plans"
    snapshots.mkdir(exist_ok=True)
    done, skipped = [], []
    for session in sorted((capture / "raw").iterdir()):
        if not session.is_dir(): continue
        if not (session / "quality_report.json").is_file():
            skipped.append({"session_id": session.name, "reason": "unfinished_or_not_yet_quality_checked"})
            continue
        quality = read_json(session / "quality_report.json")
        if not quality.get("data_checks_passed"):
            skipped.append({"session_id": session.name, "reason": "quality_checks_failed"})
            continue
        meta = read_json(session / "session.json")
        if meta.get("status") != "route_finished" or not meta.get("collection_accepted"):
            skipped.append({"session_id": session.name, "reason": "not_an_accepted_finished_session"})
            continue
        if meta["route_id"] not in approved:
            skipped.append({"session_id": session.name, "reason": "approach_marking_not_explicitly_approved"})
            continue
        info = copy.deepcopy(current_plan["routes"][meta["route_id"]])
        if sha256(resolve(info["route_csv"])) != meta["route_sha256"] or info["navigation"] != meta["navigation"]:
            raise ValueError("geometry/navigation of an existing named route changed: " + meta["route_id"])
        turns = [n for n in info["navigation"] if n["command"] == "right"]
        if not turns or any(n["link_id"] in BLOCKED_LINKS for n in turns):
            raise ValueError("route contains a right approach with a rejected marking or no right turn")
        radii = {n["link_id"]: radius_proxy(links[n["link_id"]]) for n in turns}
        if any(key == RELAXED_LINK or value >= reference_radius for key, value in radii.items()):
            raise ValueError("new route does not contain only distinct tighter right geometry")

        # 다른 route가 plan에 추가되어도 이 세션의 고정 geometry/nav 계약은 바뀌지 않는다.
        snapshot = copy.deepcopy(current_plan)
        snapshot["test_reservations"] = copy.deepcopy(original_plan["test_reservations"])
        snapshot["routes"][meta["route_id"]] = info
        snapshot_path = snapshots / (digest_value(snapshot)[:16] + ".json")
        if not snapshot_path.exists(): write_json(snapshot_path, snapshot)
        contract = digest_value({"route_sha256": meta["route_sha256"], "navigation": meta["navigation"],
                                 "guards": original_plan["test_reservations"], "val_sha256": val_sha,
                                 "prepare_sha256": sha256(BASE / "prepare.py"), "tool_sha256": sha256(Path(__file__)),
                                 "input_contract": {key: meta["config"][key] for key in ("vehicle", "cameras", "camera_common", "commands", "quality")}})
        item = {"session": rel(session), "route_id": meta["route_id"], "split": "train"}
        report = prepare_session(item, snapshot_path, cache_root, val_entries, val_sha, trees, 20., contract)
        if report["status"] != "prepared": raise ValueError("new raw session entered the complete protected corridor")
        rows = read_csv(BASE / report["cache"] / "train.csv")
        done.append({"session": session, "metadata": meta, "route": info, "report": report,
                     "rows": rows, "turn_radii_m": radii})
    return done, skipped, reference_radius


def build_variant(source_root, parent, capture, output, approved, prepare_only=False):
    source_root, parent, capture, output = map(resolve, (source_root, parent, capture, output))
    output.relative_to(BASE / "runs")
    if output.exists() and not prepare_only: raise FileExistsError(output)
    source, original_plan, old_summary, metadata, origins, original = load_parent(source_root, parent)
    trees = guard_trees(source)
    sessions, skipped, relaxed_radius = prepare_new(capture, source, original_plan, set(approved), trees)
    new_rows, route_info = [], dict(original_plan["routes"])
    for item in sessions:
        new_rows.extend(item["rows"])
        route_info[item["metadata"]["route_id"]] = item["route"]
        metadata.update(item["report"]["metadata"])
        origins.update({sid: int(stamp) for sid, stamp in item["report"]["origins_ns"].items()})
    pool = original + new_rows
    masks = defaultdict(set)
    for row in pool: masks[time_key(row, origins)].add(row["camera"])
    def stratum(row):
        turn, phase = turn_for(row, route_info[row["route_id"]]["navigation"])
        return (bool(metadata[row["session_id"]].get("noise_enabled")), bool(int(row["recovery_active"])),
                phase, tuple(sorted(masks[time_key(row, origins)])))
    existing, incoming = defaultdict(list), defaultdict(list)
    for row in original:
        if row["camera"] == "center" and row["command"] == "3":
            turn, _ = turn_for(row, route_info[row["route_id"]]["navigation"])
            if turn["link_id"] == RELAXED_LINK: existing[stratum(row)].append(row)
    for row in new_rows:
        if row["camera"] == "center" and row["command"] == "3": incoming[stratum(row)].append(row)
    additions, removals, strata = [], [], []
    for key in sorted(incoming):
        count = min(len(incoming[key]), len(existing[key]))
        if count:
            additions.extend(spread_subset(incoming[key], count))
            removals.extend(spread_subset(existing[key], count))
        strata.append({"noise_session": key[0], "recovery_active": key[1], "phase": key[2], "cameras": list(key[3]),
                       "available_new": len(incoming[key]), "replaceable_relaxed": len(existing[key]), "selected": count})
    audit = {"approved_routes": sorted(approved), "quality_passed_sessions": len(sessions), "skipped_sessions": skipped,
             "new_center_counts_by_link_phase": phase_counts(new_rows, route_info),
             "selected_new_center_counts_by_link_phase": phase_counts(additions, route_info),
             "selected_centers": len(additions), "eligible_new_right_centers": sum(map(len, incoming.values())),
             "stratum_matching": strata, "relaxed_link": RELAXED_LINK, "relaxed_radius_proxy_m": relaxed_radius,
             "source_sessions": [{"session_id": item["session"].name, "route_id": item["metadata"]["route_id"],
                                  "speed_kph": item["metadata"]["cruise_kph"], "noise_enabled": item["metadata"]["noise_enabled"],
                                  "turn_radii_m": item["turn_radii_m"]} for item in sessions]}
    if prepare_only: return audit
    if not additions: raise ValueError("no matched approved sharp-right rows yet; run --prepare-only and wait for valid sessions")
    removed = {row["image_path"] for row in removals}
    centers = [row for row in original if row["camera"] == "center" and row["image_path"] not in removed] + additions
    selected = aligned_rows(pool, centers, origins)
    selected, changed = preserve_sample_indices(original, selected)
    before, after = count_rows(original, metadata), count_rows(selected, metadata)
    for key in ("center_counts", "camera_counts", "train_rows", "noise_session_center_counts", "recovery_active_center_counts"):
        if before[key] != after[key]: raise ValueError("controlled stratum count changed: " + key)
    for i, (old, new) in enumerate(zip(original, selected)):
        if (old["command"], old["camera"]) != (new["command"], new["camera"]) or (old["command"] != "3" and old != new):
            raise ValueError("non-RIGHT row or command/camera sample index changed")
    phase_before = Counter(turn_for(row, route_info[row["route_id"]]["navigation"])[1] for row in original if row["camera"] == "center" and row["command"] == "3")
    phase_after = Counter(turn_for(row, route_info[row["route_id"]]["navigation"])[1] for row in selected if row["camera"] == "center" and row["command"] == "3")
    if phase_before != phase_after: raise ValueError("RIGHT approach/inside/exit phase distribution changed")
    minimum = distances([[float(row["x_m"]), float(row["y_m"])] for row in selected], trees)
    if min(minimum.values()) < 20: raise ValueError("sharp-right restoration touched a protected buffer")
    paths = [resolve(row["image_path"]) for row in selected]
    if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("duplicate or missing image path")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sharp_right_variant_", dir=str(output.parent)))
    try:
        prepared = stage / "prepared"
        prepared.mkdir()
        write_rows(prepared / "train.csv", selected)
        shutil.copy2(parent / "prepared/val.csv", prepared / "val.csv")
        if sha256(prepared / "val.csv") != sha256(BASE / "data/prepared/val.csv"):
            raise ValueError("validation changed")
        after.update(status="prepared", variant="v2_controlled_sharp_right_geometry", created_utc=datetime.now(timezone.utc).isoformat(),
                     parent_prepared=rel(parent / "prepared"), parent_train_sha256=old_summary["train_sha256"],
                     train_sha256=sha256(prepared / "train.csv"), val_sha256=sha256(prepared / "val.csv"),
                     center_duration_s=5400., changed_row_indices_count=len(changed), unchanged_sample_indices_count=len(original)-len(changed),
                     added_center_frames=len(additions), removed_center_frames=len(removals),
                     command_camera_index_and_other_commands_unchanged=True, fixed_val_unchanged=True,
                     right_phase_counts_unchanged=dict(phase_after), sharp_right_audit=audit,
                     right_link_phase_before=phase_counts(original, route_info), right_link_phase_after=phase_counts(selected, route_info),
                     protected_min_distances_m=minimum,
                     note="Only RIGHT geometry observations change. New real teacher labels are copied unchanged; recovery flags, command/camera/phase counts and all other sample indices remain fixed. Distinct from relaxed318 does not necessarily mean a previously unseen junction.")
        write_json(prepared / "summary.json", after)
        write_json(stage / "changes.json", {"changed_row_indices": changed,
                   "added_center_image_paths": [row["image_path"] for row in additions],
                   "removed_center_image_paths": [row["image_path"] for row in removals]})
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return after


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="runs/east_outer_90min")
    parser.add_argument("--parent", default="runs/east_outer_follow_recovery_v2_ordered")
    parser.add_argument("--capture-root", default="runs/east_outer_sharp_right_data/capture")
    parser.add_argument("--output", default="runs/east_outer_sharp_right_variant_v1")
    parser.add_argument("--approved-route", action="append", help="repeat for each independently reviewed legal approach")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = build_variant(args.source_root, args.parent, args.capture_root, args.output,
                           args.approved_route or ["EX_R4"], args.prepare_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
