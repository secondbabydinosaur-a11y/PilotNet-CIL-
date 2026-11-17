"""3-camera train 라벨. 좌우 영상에는 명시적인 PP-delta 2초 복귀 근사를 적용한다.

원본/가운데 라벨과 val/test는 보존한다. NVIDIA의 비공개 보정식을 재현한 것이
아니며 위치 복귀용 근사다. 회전된 시점/목표 yaw 복귀 증강은 여기서 만들지 않는다.
"""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import shutil
import tempfile

from data_utils import BASE, EgoTimeline, ExpertCommands, read_csv, read_json, write_json
from .build_labels import matched_status, project_path, sha256
from .prepare_dataset import side_curvature_paper


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists: " + str(output))
    center_path = BASE / "dataset/paired/train.csv"
    center = read_csv(center_path)
    sources = read_csv(BASE / "dataset/splits/train.csv")
    groups = read_json(BASE / "dataset/splits/groups.json")["train"]
    fields = list(center[0]) + ["label_method", "recovery_horizon_s", "center_curvature_1pm"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".side-labels-", dir=str(output.parent)))
    counts, rejected, snapshots, sessions = Counter(), Counter(), {}, {}
    try:
        with (stage / "train.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in center:
                writer.writerow(dict(row, label_method="center_bicycle", recovery_horizon_s=0.,
                                     center_curvature_1pm=row["y_curvature_1pm"]))
                counts["center"] += 1
            for row in sources:
                if row["camera"] == "center":
                    continue
                if row["group"] not in groups:
                    raise ValueError("non-training group in side sources")
                vehicle = project_path(BASE, row["vehicle_csv"])
                image = project_path(BASE, row["image_path"])
                directory = vehicle.parent
                sid = row["session_id"]
                if sid not in sessions:
                    meta = read_json(directory / "session.json")
                    report = read_json(directory / "quality_report.json")
                    if not report["data_checks_passed"] or meta["group"] != row["group"]:
                        raise ValueError("invalid side source capture")
                    timeline = EgoTimeline(read_csv(vehicle))
                    commands = ExpertCommands(read_csv(directory / "commands.csv"))
                    frames = {(r["camera"], r["frame_id"]): r for r in read_csv(directory / "frames.csv")}
                    sessions[sid] = meta, timeline, commands, frames
                    for name in ("session.json", "frames.csv", "vehicle.csv", "commands.csv"):
                        p = directory / name
                        snapshots[str(p.relative_to(BASE))] = sha256(p)
                meta, timeline, commands, frames = sessions[sid]
                original = frames[(row["camera"], row["frame_id"])]
                if (original["stamp_ns"] != row["stamp_ns"] or original["recv_ns"] != row["recv_ns"]
                        or (directory / original["path"]).resolve() != image):
                    raise ValueError("side source index differs from original frame")
                cfg = meta["config"]
                stamp = int(row["stamp_ns"])
                state, audit, reason = matched_status(timeline, stamp, cfg["quality"])
                if reason or state["gear"] != 4 or state["ctrl_mode"] != 2 or not commands.valid_at(int(row["recv_ns"])):
                    rejected[reason or "not_forward_expert"] += 1
                    continue
                future = timeline.at(stamp + 2_000_000_000,
                    round(cfg["quality"]["max_pair_delta_ms"] * 1e6),
                    round(cfg["quality"]["max_interpolation_gap_ms"] * 1e6))
                if (future is None or future["gear"] != 4 or future["ctrl_mode"] != 2
                        or future["speed_kph"] < cfg["experiment"]["min_speed_kph"]):
                    rejected["missing_2s_context"] += 1
                    continue
                delta = math.radians(state["steer_raw_deg"])
                center_left_curvature = math.tan(delta) / cfg["vehicle"]["wheelbase_m"]
                shift = cfg["cameras"][row["camera"]]["xyz_m"][1] - cfg["cameras"]["center"]["xyz_m"][1]
                target = side_curvature_paper(state, future, shift, center_left_curvature)
                if target is None:
                    rejected["invalid_future_geometry"] += 1
                    continue
                exp = cfg["experiment"]
                result = {k: row[k] for k in ("image_path", "session_id", "group", "course", "camera", "frame_id", "stamp_ns", "vehicle_csv")}
                result.update(audit)
                result.update(y_curvature_1pm=target, steer_raw_deg=state["steer_raw_deg"],
                    steer_raw_rad=delta, steer_equivalent_left_rad=delta,
                    wheelbase_m=cfg["vehicle"]["wheelbase_m"], speed_kph=state["speed_kph"],
                    speed_eligible=int(exp["min_speed_kph"] <= state["speed_kph"] <= exp["max_speed_kph"]),
                    label_method="our_pp_delta_2s", recovery_horizon_s=2.,
                    center_curvature_1pm=-center_left_curvature)
                writer.writerow(result)
                counts[row["camera"]] += 1
        for name in ("val.csv", "test.csv"):
            shutil.copyfile(BASE / "dataset/paired" / name, stage / name)
        if any(sha256(BASE / name) != value for name, value in snapshots.items()):
            raise ValueError("source changed while making side labels")
        write_json(stage / "summary.json", dict(
            status="three_camera_training_labels", counts=dict(counts), rejected=dict(rejected),
            training_groups=groups, target_unit="1/m; left negative, right positive",
            side_labels="our PP-delta at expert pose t+2s, not the original unpublished formula",
            recovery_horizon_s=2., shifted_yaw_augmentation=False,
            center_train_sha256=sha256(center_path), source_sha256=snapshots,
            validation_and_test="byte-for-byte copies of original center-only CSVs"))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print("{}: {}".format(output, dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "dataset/paired_three")
    build(parser.parse_args().output)
