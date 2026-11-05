"""시험 전용 도로 이미지-곡률 연결과 고정된 두 모델의 open-loop 평가."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import BASE, EgoTimeline, ExpertCommands, read_csv, read_json, steering_to_curvature, write_json
from dataset import PilotNetDataset
from evaluate import make_plots
from learning.common import choose_device, file_hash, load_checkpoint, regression_metrics
from learning.live import Reference
from tools.collection.collect_unseen import verify_frozen
from tools.preparation.build_labels import FIELDS, matched_status


def build_manifest():
    verify_frozen()
    output = BASE / "dataset/unseen_paired"
    sessions = read_json(BASE / "dataset/unseen_sessions.json")
    rows, sources, rejected = [], {}, Counter()
    # Gain/sign are taken from the existing label contract, never fitted on test.
    model = read_json(BASE / "dataset/paired/summary.json")["model"]
    for course, path in sessions.items():
        session = BASE / path
        meta = read_json(session / "session.json")
        if not meta.get("evaluation_only") or not read_json(session / "quality_report.json")["data_checks_passed"]:
            raise RuntimeError("only inspected test-only sessions are accepted")
        timeline = EgoTimeline(read_csv(session / "vehicle.csv"))
        commands = ExpertCommands(read_csv(session / "commands.csv"))
        ref = Reference(course)
        for frame in read_csv(session / "frames.csv"):
            if frame["camera"] != "center":
                continue
            status, details, reason = matched_status(timeline, int(frame["stamp_ns"]), meta["config"]["quality"])
            if reason:
                rejected[reason] += 1
                continue
            station, cte, _ = ref.project(status)
            if not ref.start <= station <= ref.end:
                rejected["outside_evaluation_stations"] += 1
                continue
            if (status["ctrl_mode"] != 2 or status["gear"] != 4
                    or not commands.valid_at(int(frame["recv_ns"]))):
                rejected["invalid_teacher_command"] += 1
                continue
            angle = status["steer_raw_deg"]
            gain, sign = model["road_wheel_gain"], model["steer_left_sign"]
            row = dict(image_path=str((session / frame["path"]).relative_to(BASE)),
                y_curvature_1pm=steering_to_curvature(angle, 3., sign, gain),
                steer_raw_deg=angle, steer_raw_rad=float(np.radians(angle)),
                steer_equivalent_left_rad=float(np.radians(angle * sign * gain)),
                wheelbase_m=3., speed_kph=status["speed_kph"],
                speed_eligible=int(10 <= status["speed_kph"] <= 30),
                session_id=session.name, group=meta["group"], course=course,
                camera="center", frame_id=frame["frame_id"], stamp_ns=frame["stamp_ns"],
                vehicle_csv=str((session / "vehicle.csv").relative_to(BASE)), **details)
            rows.append(row)
        for name in ("session.json", "quality_report.json", "frames.csv", "vehicle.csv", "commands.csv", "expert.csv"):
            sources[str((session / name).relative_to(BASE))] = file_hash(session / name)
    if {r["course"] for r in rows} != {"U1", "U2"}:
        raise RuntimeError("both unseen courses need valid labels")
    # Test images must never occur in either original training manifest.
    original_paths = {r["image_path"] for folder in ("paired", "paired_three")
                      for split in ("train", "val", "test")
                      for r in read_csv(BASE / "dataset" / folder / (split + ".csv"))}
    if any(r["image_path"] in original_paths for r in rows):
        raise RuntimeError("unseen images leaked into an existing split")
    output.mkdir(exist_ok=True)
    with (output / "test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", dict(evaluation_only=True, sources_sha256=sources,
        rejected=dict(rejected), labels_by_course=dict(Counter(r["course"] for r in rows)),
        method="same native timestamp interpolation, bicycle label and preprocessing as original test"))
    return output


def main():
    torch.set_num_threads(4)
    paired = build_manifest()
    data = PilotNetDataset("test", paired_dir=paired)
    device = choose_device()
    for mode in ("center", "three"):
        checkpoint = BASE / ("runs/pilotnet_" + mode + "_v1/best.pt")
        output = checkpoint.parent / "open_loop_unseen"
        if output.exists():
            previous = read_json(output / "metrics.json")
            if (previous["checkpoint_sha256"] != file_hash(checkpoint)
                    or previous["test_manifest_sha256"] != file_hash(paired / "test.csv")):
                raise RuntimeError("existing evaluation has different model/data: " + str(output))
            print("EXISTING UNSEEN OPEN LOOP", mode, previous["metrics"], flush=True)
            continue
        model, payload = load_checkpoint(checkpoint, device, check_data=True)
        targets, predictions = [], []
        with torch.inference_mode():
            for images, target in DataLoader(data, batch_size=128, shuffle=False, num_workers=2):
                predictions.append(model(images.to(device)).cpu().numpy().reshape(-1))
                targets.append(target.numpy().reshape(-1))
        target, prediction = np.concatenate(targets), np.concatenate(predictions)
        report = dict(evaluation="open_loop_unseen_road_center_images", split="test_only_unseen",
            checkpoint_sha256=file_hash(checkpoint), selected_epoch=payload["epoch"],
            metrics=regression_metrics(target, prediction),
            zero_curvature_baseline=regression_metrics(target, np.zeros_like(target)),
            by_course={}, by_session={}, zero_curvature_baseline_by_course={},
            scope="new roads, same K-City map and clear/noon conditions; one teacher run per road",
            test_manifest_sha256=file_hash(paired / "test.csv"), model_updated=False)
        for column, dest in (("course", "by_course"), ("session_id", "by_session")):
            for value in sorted({r[column] for r in data.rows}):
                mask = np.array([r[column] == value for r in data.rows])
                report[dest][value] = regression_metrics(target[mask], prediction[mask])
                if column == "course":
                    report["zero_curvature_baseline_by_course"][value] = regression_metrics(target[mask], np.zeros(int(mask.sum())))
        output.mkdir()
        fields = ("image_path", "session_id", "course", "stamp_ns", "target_1pm", "prediction_1pm", "error_1pm")
        with (output / "predictions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, y, p in zip(data.rows, target, prediction):
                writer.writerow(dict(**{k: row[k] for k in fields[:4]}, target_1pm=float(y),
                                     prediction_1pm=float(p), error_1pm=float(p-y)))
        write_json(output / "metrics.json", report)
        make_plots(target, prediction, data.rows, report, output)
        print("UNSEEN OPEN LOOP", mode, report["metrics"], flush=True)
    verify_frozen()


if __name__ == "__main__":
    main()
