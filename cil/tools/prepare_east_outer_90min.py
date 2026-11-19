"""90분 데이터 실험의 세션별 준비 캐시·명령별 원장·격리된 학습 CSV를 만든다.

수집기는 별도로 실행한다. 이 도구는 accepted 목록만 읽으며 원본·기존 train/val·
모델을 바꾸지 않는다. 반복 호출은 새 세션만 prepare.build로 준비한다.
"""

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from prepare import FIELDS, build, read_csv, sha256

BASELINE_HASHES = {
    "train.csv": "ae54b55fa27d162de1076873f48a4dc9354b49a96f9b53d598b03bd9ac02db41",
    "val.csv": "6a977f475e0ea5d356ca47e8dcdf3bd1630e1475ebaa20caaba3e2e86e562872",
    "model.py": "70d5490f49d2f7854bbf83aed1a6f6a6f8dbd69cc16da80d949cef3fd3457e34",
    "dataset.py": "e855f92a7fdbf7f9262b1b0e806bf73a73e8269e8dbbf60975c92112f66e2a87",
    "config.json": "11ee5834956cf6bbb258202c504beaaac2a54b6732f049cc8b1d9359de3a7138",
}
BASE_COMPLETIONS = (
    "data/collection_completion.json",
    "runs/east_outer_data_v1/verified_retry/additional_completion.json",
)
COMMANDS = ("0", "1", "2", "3")
SAMPLE_HZ = 10
PERIOD_NS = 100_000_000
MARGIN_M = .25  # 시험 선분을 0.25m 간격으로 표본화한 데 대한 보수 여유다.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE / path).resolve()


def rel(path):
    return str(Path(path).resolve().relative_to(BASE))


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dense_xy(xyz):
    """XY 선분을 촘촘히 만들어 전체 회랑을 검사하며 점 사이 구간도 보호한다."""
    xy = np.asarray(xyz, dtype=float)[:, :2]
    if len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("invalid protected route")
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-7]]
    station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    samples = np.r_[np.arange(0., station[-1], MARGIN_M), station[-1]]
    return np.column_stack([np.interp(samples, station, xy[:, k]) for k in (0, 1)])


def distances(xy, trees):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or len(xy) == 0 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("empty or invalid observed positions")
    return {name: max(0., float(tree.query(xy)[0].min()) - MARGIN_M)
            for name, tree in trees.items()}


def source_stats(paths):
    """완료 원본의 크기·mtime을 저장해 재호출 때 큰 CSV를 다시 읽지 않는다."""
    return {rel(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths}


def source_paths(session):
    return [session / name for name in ("session.json", "quality_report.json", "images.csv", "vehicle.csv", "commands.csv")]


def session_meta(session):
    meta, quality = read_json(session / "session.json"), read_json(session / "quality_report.json")
    if (meta.get("status") != "route_finished" or not meta.get("collection_accepted")
            or not quality.get("data_checks_passed") or meta.get("split") != "train"):
        raise ValueError("accepted entry is not a quality-passed training session: " + str(session))
    return meta


def count_rows(rows, metadata):
    counts, noise, recovery, cameras = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        cameras[row["camera"]] += 1
        if row["camera"] != "center":
            continue  # 좌우 영상을 시간을 세 배로 부풀리는 데 사용하지 않는다.
        command = str(row["command"])
        counts[command] += 1
        if metadata[row["session_id"]].get("noise_enabled"):
            noise[command] += 1
        if int(row.get("recovery_active", 0)):
            recovery[command] += 1
    return {"center_counts": {c: counts[c] for c in COMMANDS},
            "noise_session_center_counts": {c: noise[c] for c in COMMANDS},
            "recovery_active_center_counts": {c: recovery[c] for c in COMMANDS},
            "camera_counts": dict(cameras), "train_rows": sum(cameras.values())}


def compact_metadata(meta):
    names = ("route_id", "seed", "noise_enabled", "initial_offset_m", "cruise_kph", "noise_episodes")
    return {key: meta.get(key) for key in names}


def image_origin(session):
    with (session / "images.csv").open(newline="", encoding="utf-8-sig") as stream:
        return min(int(row["stamp_ns"]) for row in csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def combined_base_completion(paths):
    """실제 비교 모델이 사용한 기존·추가 수집 원본의 합집합을 복원한다."""
    items = {}
    for path in paths:
        completion = read_json(path)
        if completion.get("status") != "complete":
            raise ValueError("base source completion is not complete: " + str(path))
        for item in completion["accepted"]:
            source = str(resolve(item["session"]))
            if source in items and items[source]["split"] != item["split"]:
                raise ValueError("base session has conflicting splits")
            items[source] = item
    return {"status": "complete", "accepted": list(items.values())}


def baseline_cache(output, train_path, completion, frozen_tree, buffer_m, contract):
    cache = output / "cache/baseline"
    report_path = cache / "ledger.json"
    accepted = {Path(item["session"]).name: item for item in completion["accepted"] if item["split"] == "train"}
    paths = [train_path] + [path for item in accepted.values() for path in source_paths(BASE / item["session"])]
    stats = source_stats(paths)
    if report_path.exists():
        report = read_json(report_path)
        if report["contract"] != contract or report["source_stats"] != stats:
            raise ValueError("baseline cache inputs changed; choose another --output")
        return report
    rows = read_csv(train_path)
    metadata, origins, excluded, kept = {}, {}, [], set()
    for sid, item in accepted.items():
        session = BASE / item["session"]
        meta = session_meta(session)
        vehicle = read_csv(session / "vehicle.csv")
        near = distances([[float(row["x_m"]), float(row["y_m"])] for row in vehicle], {"frozen_route": frozen_tree})["frozen_route"]
        if near < buffer_m:
            removed = [row for row in rows if row["session_id"] == sid]
            excluded.append({"session_id": sid, "route_id": meta["route_id"],
                             "reason": "whole_raw_session_intersects_extended_test_buffer",
                             "conservative_min_distance_m": near,
                             "removed_rows": len(removed),
                             "removed_center_counts": dict(Counter(row["command"] for row in removed if row["camera"] == "center"))})
        else:
            kept.add(sid)
            metadata[sid], origins[sid] = compact_metadata(meta), image_origin(session)
    if any(row["session_id"] not in accepted for row in rows):
        raise ValueError("baseline prepared CSV contains an unknown source session")
    retained = [row for row in rows if row["session_id"] in kept]
    cache.mkdir(parents=True)
    write_rows(cache / "train.csv", retained)
    report = dict(status="prepared", cache=rel(cache), contract=contract, source_stats=stats,
                  train_sha256=sha256(cache / "train.csv"),
                  metadata=metadata, origins_ns=origins, excluded_sessions=excluded,
                  original_center_count=sum(row["camera"] == "center" for row in rows),
                  **count_rows(retained, metadata))
    write_json(report_path, report)
    return report


def prepare_session(item, plan_path, output, val_entries, val_sha, trees, buffer_m, contract):
    session = resolve(item["session"])
    session.relative_to(BASE)  # 원본과 캐시는 CIL 안의 격리된 실행 폴더를 사용한다.
    sid = session.name
    cache = output / "cache" / sid
    paths = source_paths(session)
    stats = source_stats(paths)
    report_path = cache / "ledger.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["contract"] != contract or report["source_stats"] != stats:
            raise ValueError("cached source session changed: " + sid)
        return report
    meta = session_meta(session)
    if item.get("split", "train") != "train" or item.get("route_id", meta["route_id"]) != meta["route_id"]:
        raise ValueError("completion entry and session metadata differ")
    raw = read_csv(session / "vehicle.csv")
    separation = distances([[float(row["x_m"]), float(row["y_m"])] for row in raw], trees)
    cache.mkdir(parents=True)
    common = dict(session_id=sid, session=rel(session), cache=rel(cache), contract=contract,
                  source_stats=stats, metadata={sid: compact_metadata(meta)},
                  conservative_min_distances_m=separation)
    if any(value < buffer_m for value in separation.values()):
        report = dict(common, status="excluded_guard", reason="raw_session_touched_protected_buffer")
        write_json(report_path, report)
        return report

    single = cache / "completion.json"
    write_json(single, {"status": "complete", "accepted": [item] + val_entries,
                        "note": "one completed training session plus the unchanged validation sessions only"})
    prepared = cache / "prepared"
    if prepared.exists():  # 이전 준비가 완료 기록을 남기기 전에 실패한 경우의 임시 결과만 정리한다.
        shutil.rmtree(prepared)
    build(output=prepared, sample_hz=SAMPLE_HZ, recovery_s=2.0, min_lookahead_m=6.0,
          plan=plan_path, completion=single)  # 원래 시각 매칭·정답 보정·품질 기준을 그대로 재사용한다.
    if sha256(prepared / "val.csv") != val_sha:
        raise ValueError("single-session preparation changed the fixed validation CSV")
    (prepared / "val.csv").unlink()  # 큰 데이터 전체 대신 작은 고정 Val만 재검사하며 중복 사본은 남기지 않는다.
    rows = read_csv(prepared / "train.csv")
    observed = distances([[float(row["x_m"]), float(row["y_m"])] for row in rows], trees)
    if any(value < buffer_m for value in observed.values()):
        raise ValueError("prepared rows entered a protected buffer")
    (prepared / "train.csv").replace(cache / "train.csv")
    report = dict(common, status="prepared", origins_ns={sid: image_origin(session)},
                  train_sha256=sha256(cache / "train.csv"),
                  prepared_min_distances_m=observed, **count_rows(rows, common["metadata"]))
    write_json(report_path, report)
    return report


def spread_subset(rows, target):
    """도로→세션에 기회를 고르게 나누고 각 세션의 시간 전체에서 실제 행을 선택한다."""
    by_route = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_route[row["route_id"]][row["session_id"]].append(row)

    def allocate(capacities, amount):
        quotas = {key: 0 for key in sorted(capacities)}
        while amount:
            available = [key for key in quotas if quotas[key] < capacities[key]]
            if not available:
                raise ValueError("not enough unique real samples")
            step = max(1, amount // len(available))
            for key in available:
                added = min(step, capacities[key] - quotas[key], amount)
                quotas[key] += added
                amount -= added
                if not amount: break
        return quotas

    route_quota = allocate({route: sum(map(len, sessions.values())) for route, sessions in by_route.items()}, target)
    selected = []
    for route, total in route_quota.items():
        session_quota = allocate({sid: len(values) for sid, values in by_route[route].items()}, total)
        for sid, quota in session_quota.items():
            values = sorted(by_route[route][sid], key=lambda row: int(row["stamp_ns"]))
            if quota:
                indices = [len(values) // 2] if quota == 1 else np.linspace(0, len(values) - 1, quota).round().astype(int)
                selected.extend(values[index] for index in indices)
    return selected


def select_rows(rows, metadata, origins, per_command, noise_fraction):
    centers = [row for row in rows if row["camera"] == "center"]
    chosen = []
    for command in COMMANDS:
        group = [row for row in centers if row["command"] == command]
        noisy = [row for row in group if metadata[row["session_id"]].get("noise_enabled")]
        normal = [row for row in group if not metadata[row["session_id"]].get("noise_enabled")]
        noise_n = min(len(noisy), max(round(per_command * noise_fraction), per_command - len(normal)))
        if noise_n: chosen.extend(spread_subset(noisy, noise_n))
        if per_command - noise_n: chosen.extend(spread_subset(normal, per_command - noise_n))
    selected = {(row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS): row["command"]
                for row in chosen}
    if len(selected) != per_command * 4:
        raise ValueError("duplicate center time bins were found; no sample is counted twice")
    result, images = [], set()
    center_paths = {row["image_path"] for row in chosen}
    for row in rows:
        key = (row["session_id"], (int(row["stamp_ns"]) - origins[row["session_id"]]) // PERIOD_NS)
        if key not in selected or selected[key] != row["command"]:
            continue
        if row["camera"] == "center" and row["image_path"] not in center_paths:
            continue
        if row["image_path"] in images:
            raise ValueError("duplicate image path in final selection")
        images.add(row["image_path"])
        result.append(row)
    return sorted(result, key=lambda row: (row["session_id"], int(row["stamp_ns"]), row["camera"]))


def run(args):
    plan_path, completion_path, output = resolve(args.plan), resolve(args.completion), resolve(args.output)
    output.relative_to(BASE / "runs")  # 새 자산은 runs 아래에만 작성한다.
    frozen_path = resolve(args.frozen_route)
    train_path, val_path = resolve(args.base_train), resolve(args.val_csv)
    for path, key in ((train_path, "train.csv"), (val_path, "val.csv")):
        if sha256(path) != BASELINE_HASHES[key]:
            raise ValueError("frozen baseline CSV changed: " + str(path))
    for name in ("model.py", "dataset.py", "config.json"):
        if sha256(BASE / name) != BASELINE_HASHES[name]:
            raise ValueError("baseline source changed: " + name)
    output.mkdir(parents=True, exist_ok=True)
    base_completion_paths = [resolve(path) for path in (args.base_completion or BASE_COMPLETIONS)]
    baseline_completion = combined_base_completion(base_completion_paths)
    val_entries = [item for item in baseline_completion["accepted"] if item["split"] == "validation"]
    plan = copy.deepcopy(read_json(plan_path))
    frozen = np.loadtxt(frozen_path, delimiter=",", skiprows=1, ndmin=2)
    frozen_xy = dense_xy(frozen[:, 1:4])
    guard_xyz = np.column_stack((frozen_xy, np.zeros(len(frozen_xy))))
    plan["test_reservations"]["EAST_OUTER_EXTENDED_90MIN_FROZEN"] = {
        "guard_reference_xyz": guard_xyz.tolist(), "buffer_m": args.buffer_m,
        "route_sha256": sha256(frozen_path), "scope": "entire fixed extended route; no crop"}
    trees = {name: cKDTree(dense_xy(value["guard_reference_xyz"])) for name, value in plan["test_reservations"].items()}
    for name in ("P2Q", "V1", "V2"):
        route = np.loadtxt(BASE / plan["routes"][name]["route_csv"], delimiter=",", skiprows=1, ndmin=2)
        trees[name] = cKDTree(dense_xy(route[:, 1:4]))
    actual_val = []
    for item in val_entries:
        actual_val.extend([[float(row["x_m"]), float(row["y_m"])] for row in read_csv(BASE / item["session"] / "vehicle.csv")])
    trees["actual_validation_positions"] = cKDTree(actual_val)
    plan_digest = digest_value(plan)
    snapshot = output / ("plan_" + plan_digest[:12] + ".json")
    if not snapshot.exists(): write_json(snapshot, plan)
    contract = digest_value({"frozen_route_sha": sha256(frozen_path), "buffer_m": args.buffer_m,
                             "prepare_sha": sha256(BASE / "prepare.py"), "tool_sha": sha256(Path(__file__)),
                             "baseline_hashes": BASELINE_HASHES, "sample_hz": SAMPLE_HZ,
                             "guards": plan["test_reservations"],
                             "base_completions_sha256": {rel(path): sha256(path) for path in base_completion_paths},
                             "validation_raw_sha256": {item["session"]: sha256(BASE / item["session"] / "vehicle.csv") for item in val_entries}})
    previous_path = output / "ledger.json"
    if previous_path.exists() and read_json(previous_path)["contract"] != contract:
        raise ValueError("experiment inputs changed; choose another output directory")

    baseline = baseline_cache(output, train_path, baseline_completion,
                              cKDTree(frozen_xy), args.buffer_m, contract)
    reports, seen = [], set()
    baseline_sessions = {Path(item["session"]).name for item in baseline_completion["accepted"]}
    current = read_json(completion_path)
    for item in current.get("accepted", []):
        if item.get("split", "train") != "train": continue
        sid = Path(item["session"]).name
        if sid in baseline_sessions: continue  # 기존 원본은 baseline cache에서 한 번만 계산한다.
        if sid in seen: raise ValueError("duplicate accepted session: " + sid)
        seen.add(sid)
        reports.append(prepare_session(item, snapshot, output, val_entries, BASELINE_HASHES["val.csv"],
                                       trees, args.buffer_m, contract))
    prepared = [baseline] + [report for report in reports if report["status"] == "prepared"]
    totals = {key: {c: sum(report[key][c] for report in prepared) for c in COMMANDS}
              for key in ("center_counts", "noise_session_center_counts", "recovery_active_center_counts")}
    deficits = {c: max(0, args.per_command - totals["center_counts"][c]) for c in COMMANDS}
    ledger = dict(
        status="ready_to_finalize" if not any(deficits.values()) else "collecting",
        updated_utc=datetime.now(timezone.utc).isoformat(), contract=contract,
        plan_snapshot=rel(snapshot), completion_input=rel(completion_path),
        frozen_route=rel(frozen_path), frozen_route_sha256=sha256(frozen_path), buffer_m=args.buffer_m,
        nominal_sample_hz=SAMPLE_HZ, target_center_frames=args.per_command * 4,
        target_per_command=args.per_command, target_noise_fraction=args.noise_fraction,
        target_met=not any(deficits.values()), deficits_per_command=deficits,
        available_center_frames=sum(totals["center_counts"].values()),
        valid_center_seconds=sum(totals["center_counts"].values()) / SAMPLE_HZ,
        **totals, baseline=baseline, accepted_sessions=reports,
        cached_training_sessions=len(prepared) - 1,
        excluded_additional_sessions=[report["session_id"] for report in reports if report["status"] != "prepared"],
        comparability=dict(
            prior_video_model_train_csv=rel(train_path),
            prior_video_model_train_sha256=BASELINE_HASHES["train.csv"], fixed_val_sha256=BASELINE_HASHES["val.csv"],
            prior_training_source="actual previous video-model candidate, including original and earlier supplemental demonstrations",
            base_completion_inputs={rel(path): sha256(path) for path in base_completion_paths},
            baseline_filter="exclude whole raw training sessions touching any part of the extended test corridor",
            prior_video_model_had_extended_test_corridor_overlap=bool(baseline["excluded_sessions"]),
            not_a_pure_additional_data_ablation=True,
            note="Candidate changes include test-corridor removal, added demonstrations and balanced selection. The previous video model used the declared candidate CSV before this stricter extended-route exclusion. Report this asymmetry."),
    )
    write_json(previous_path, ledger)
    if args.finalize:
        if not ledger["target_met"]:
            raise ValueError("not enough real center frames; deficits=" + json.dumps(deficits))
        destination = output / "prepared"
        if destination.exists(): raise FileExistsError("final selection already exists: " + str(destination))
        rows, metadata, origins = [], {}, {}
        for report in prepared:
            cache = BASE / report["cache"]
            if sha256(cache / "train.csv") != report["train_sha256"]:
                raise ValueError("prepared cache changed: " + str(cache))
            rows.extend(read_csv(cache / "train.csv"))
            metadata.update(report["metadata"])
            origins.update({sid: int(stamp) for sid, stamp in report["origins_ns"].items()})
        selected = select_rows(rows, metadata, origins, args.per_command, args.noise_fraction)
        summary = count_rows(selected, metadata)
        if any(summary["center_counts"][c] != args.per_command for c in COMMANDS):
            raise ValueError("final center counts do not match the declared balanced target")
        selected_distances = distances([[float(row["x_m"]), float(row["y_m"])] for row in selected], trees)
        if any(value < args.buffer_m for value in selected_distances.values()):
            raise ValueError("final selected training rows entered a protected buffer")
        stage = Path(tempfile.mkdtemp(prefix=".final_prepared_", dir=str(output)))
        try:
            write_rows(stage / "train.csv", selected)
            shutil.copy2(val_path, stage / "val.csv")
            shutil.copy2(previous_path, stage / "source_ledger.json")
            if sha256(stage / "val.csv") != BASELINE_HASHES["val.csv"]:
                raise ValueError("validation changed while preparing final output")
            summary.update(status="prepared", center_duration_s=args.per_command * 4 / SAMPLE_HZ,
                           raw_and_side_frames_not_counted_as_additional_driving_time=True,
                           selection="unique center bins; command/noise strata, route/session fair allocation and evenly spread time indices; retain corresponding side bins only",
                           target_noise_fraction=args.noise_fraction,
                           actual_noise_session_fraction=sum(summary["noise_session_center_counts"].values()) / (args.per_command * 4),
                           protected_min_distances_m=selected_distances, comparability=ledger["comparability"],
                           train_sha256=sha256(stage / "train.csv"), val_sha256=sha256(stage / "val.csv"),
                           source_ledger="source_ledger.json", source_ledger_sha256=sha256(stage / "source_ledger.json"),
                           note="54000 center frames at nominal10Hz mean90 minutes of selected observations, not 90 minutes of independent road geometry. No duplicate image paths or artificial frames are added.")
            write_json(stage / "summary.json", summary)
            stage.rename(destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        ledger.update(status="finalized", prepared=rel(destination), selected=summary)
        write_json(previous_path, ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-route", default="routes/test/EAST_OUTER_EXTENDED.csv")
    parser.add_argument("--base-train", default="runs/east_outer_data_v1/verified_retry/train.csv")
    parser.add_argument("--base-completion", action="append", help="repeat for original and previous supplemental completion manifests")
    parser.add_argument("--val-csv", default="data/prepared/val.csv")
    parser.add_argument("--buffer-m", type=float, default=20.0)
    parser.add_argument("--per-command", type=int, default=13500)
    parser.add_argument("--noise-fraction", type=float, default=.10)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.buffer_m) or args.buffer_m < 20 or args.per_command < 1 or not 0 <= args.noise_fraction <= 1:
        parser.error("buffer>=20m, per-command>0 and noise-fraction in[0,1] required")
    output = resolve(args.output)
    output.relative_to(BASE / "runs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 동시 호출로 같은 캐시를 쓰지 않게 한다.
        ledger = run(args)
    print(json.dumps({key: ledger[key] for key in (
        "status", "center_counts", "noise_session_center_counts", "recovery_active_center_counts",
        "deficits_per_command", "valid_center_seconds", "target_met")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
