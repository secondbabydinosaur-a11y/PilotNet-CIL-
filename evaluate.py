"""고정된 기록 이미지로 open-loop 평가. 예측 CSV·지표 JSON·그림을 저장한다."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import write_json
from dataset import PilotNetDataset
from learning.common import (choose_device, file_hash, json_print, load_checkpoint,
                             regression_metrics, seed_worker)


def make_plots(target, prediction, rows, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    low, high = min(target.min(), prediction.min()), max(target.max(), prediction.max())
    axes[0, 0].scatter(target, prediction, s=5, alpha=.4)
    axes[0, 0].plot([low, high], [low, high], "k--")
    axes[0, 0].set(xlabel="Target curvature (1/m)", ylabel="Predicted curvature (1/m)")
    axes[0, 1].hist(prediction - target, bins=50)
    axes[0, 1].set(xlabel="Prediction error (1/m)", ylabel="Images")
    # 세션 사이를 한 주행처럼 연결하지 않고 대표 곡선 주행 하나를 보여준다.
    session = next((r["session_id"] for r in rows if r["course"] == "D"), rows[0]["session_id"])
    indices = np.array([i for i, r in enumerate(rows) if r["session_id"] == session])
    stamps = np.array([int(rows[i]["stamp_ns"]) for i in indices], dtype=np.int64)
    seconds = (stamps - stamps[0]) / 1e9
    axes[1, 0].plot(seconds, target[indices], label="target")
    axes[1, 0].plot(seconds, prediction[indices], label="prediction")
    axes[1, 0].set(xlabel="Session time (s)", ylabel="Curvature (1/m)")
    axes[1, 0].legend()
    names = list(report["by_course"])
    axes[1, 1].bar(names, [report["by_course"][c]["rmse_1pm"] for c in names])
    axes[1, 1].set(xlabel="Course", ylabel="Curvature RMSE (1/m)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "open_loop.png", dpi=170)
    fig.savefig(output / "open_loop.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("invalid batch size/workers")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device, check_data=True)
    dataset = PilotNetDataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, worker_init_fn=seed_worker,
                        pin_memory=device.type == "cuda")
    targets, predictions = [], []
    with torch.inference_mode():
        for images, target in loader:
            prediction = model(images.to(device)).cpu()
            if not torch.isfinite(prediction).all().item():
                raise RuntimeError("non-finite model prediction")
            targets.append(target.numpy().reshape(-1))
            predictions.append(prediction.numpy().reshape(-1))
    target, prediction = np.concatenate(targets), np.concatenate(predictions)
    wheelbase = checkpoint["data_contract"]["wheelbase_m"]
    output = (args.output or args.checkpoint.parent / ("open_loop_" + args.split)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"evaluation": "open_loop_recorded_center_images", "split": args.split,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
              "selected_epoch": checkpoint["epoch"], "metrics": regression_metrics(target, prediction, wheelbase),
              "zero_curvature_baseline": regression_metrics(target, np.zeros_like(target), wheelbase),
              "by_course": {}, "by_session": {}, "by_turn": {},
              "scope": "held-out runs on the same A/D roads; not unseen-road or closed-loop performance"}
    for column, destination in (("course", "by_course"), ("session_id", "by_session")):
        for value in sorted({r[column] for r in dataset.rows}):
            mask = np.array([r[column] == value for r in dataset.rows])
            report[destination][value] = regression_metrics(target[mask], prediction[mask], wheelbase)
    for name, mask in (("left", target < -.003), ("straight", np.abs(target) <= .003),
                       ("right", target > .003)):
        if mask.any():
            report["by_turn"][name] = regression_metrics(target[mask], prediction[mask], wheelbase)
    fields = ("image_path", "session_id", "course", "stamp_ns", "target_1pm", "prediction_1pm", "error_1pm")
    with (output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, y, pred in zip(dataset.rows, target, prediction):
            writer.writerow({**{k: row[k] for k in fields[:4]}, "target_1pm": float(y),
                             "prediction_1pm": float(pred), "error_1pm": float(pred - y)})
    write_json(output / "metrics.json", report)
    make_plots(target, prediction, dataset.rows, report, output)
    json_print({"output": str(output), "metrics": report["metrics"],
                "zero_curvature_baseline": report["zero_curvature_baseline"]})


if __name__ == "__main__":
    main()
