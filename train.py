"""PilotNet 학습. train으로 업데이트하고 val MSE가 가장 낮은 모델을 저장한다."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import BASE, write_json
from dataset import PilotNetDataset
from learning.common import (choose_device, data_contract, environment_info,
                             file_hash, save_checkpoint, seed_everything, seed_worker)
from learning.data import CachedDataset, curve_sampler
from model import PilotNet


def run_epoch(model, loader, device, optimizer=None, loss_scale=100.):
    training = optimizer is not None
    model.train(training)
    squared_error, absolute_error, count = 0., 0., 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            if prediction.shape != targets.shape or not torch.isfinite(prediction).all().item():
                raise RuntimeError("invalid model output")
            error = prediction - targets
            # 곡률 값이 작으므로 최적화 loss만 상수배한다. 로그는 원래 1/m 단위다.
            loss = ((error * loss_scale) ** 2).mean()
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                if not torch.isfinite(norm).item():
                    raise RuntimeError("non-finite gradient")
                optimizer.step()
            squared_error += error.detach().square().sum().item()
            absolute_error += error.detach().abs().sum().item()
            count += targets.numel()
    return {"mse": squared_error / count, "mae": absolute_error / count}


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("train_mse", "val_mse"):
        ax.plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    ax.set(xlabel="Epoch", ylabel="Curvature MSE (1/m²)", yscale="log")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache", action="store_true", help="cache processed uint8 images in RAM")
    parser.add_argument("--camera-mode", choices=("center", "three"), default="center")
    parser.add_argument("--curve-weight", type=float, default=2.)
    parser.add_argument("--curve-basis", choices=("target", "center"), default="target",
                        help="target reproduces v1; center excludes side-camera recovery correction from curve selection")
    parser.add_argument("--loss-scale", type=float, default=100.)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.workers < 0:
        parser.error("counts must be positive; workers may be zero")
    if args.lr <= 0 or args.loss_scale <= 0 or args.curve_weight < 1:
        parser.error("invalid learning rate, loss scale, or curve weight")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    paired_dir = BASE / "dataset" / ("paired_three" if args.camera_mode == "three" else "paired")
    train_set = PilotNetDataset("train", paired_dir=paired_dir, allow_side=args.camera_mode == "three")
    val_set = PilotNetDataset("val")  # 두 모델 모두 같은 가운데 validation으로 선택한다.
    training_data = {"camera_mode": args.camera_mode, "train_csv": str(paired_dir / "train.csv"),
                     "train_csv_sha256": file_hash(paired_dir / "train.csv")}
    assert not {r["session_id"] for r in train_set.rows}.intersection(
        r["session_id"] for r in val_set.rows)
    contract = data_contract()
    if args.cache:
        train_set = CachedDataset(train_set, args.workers)
        val_set = CachedDataset(val_set, args.workers)
    sampler = curve_sampler(train_set, args.curve_weight, args.seed, args.curve_basis)
    workers = 0 if args.cache else args.workers
    options = dict(batch_size=args.batch_size, num_workers=workers,
                   pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                   persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler,
                              generator=torch.Generator().manual_seed(args.seed), **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    run = args.run_dir or BASE / "runs/train" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(run_dir=str(run), device_used=str(device))
    metadata = dict(settings=settings, data_contract=contract, training_data=training_data, environment=environment_info(),
                    counts={"train": len(train_set), "val": len(val_set)},
                    selection="lowest validation MSE; test unused during training")
    write_json(run / "settings.json", metadata)
    # 저장한 샘플 목록으로 어떤 원본을 사용했는지 재현할 수 있다.
    for split, dataset in (("train", train_set), ("val", val_set)):
        with (run / (split + "_samples.csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_path", "y_curvature_1pm", "session_id"))
            writer.writeheader()
            writer.writerows({k: row[k] for k in writer.fieldnames} for row in dataset.rows)
    model = PilotNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
    best, best_epoch, bad_epochs, history = float("inf"), 0, 0, []
    started = time.monotonic()
    print("RUN {} | {} | train={} val={}".format(run, device, len(train_set), len(val_set)), flush=True)
    try:
        for epoch in range(1, args.epochs + 1):
            before = time.monotonic()
            train = run_epoch(model, train_loader, device, optimizer, args.loss_scale)
            val = run_epoch(model, val_loader, device)
            improved = val["mse"] < best
            if improved:
                best, best_epoch, bad_epochs = val["mse"], epoch, 0
            else:
                bad_epochs += 1
            scheduler.step(val["mse"])
            row = dict(epoch=epoch, train_mse=train["mse"], val_mse=val["mse"],
                       train_mae=train["mae"], val_mae=val["mae"],
                       lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - before)
            history.append(row)
            with (run / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            checkpoint = dict(epoch=epoch, best_epoch=best_epoch, best_val_mse=best,
                              data_contract=contract, settings=settings, training_data=training_data,
                              environment=metadata["environment"])
            save_checkpoint(run / "last.pt", model, **checkpoint)
            if improved:
                save_checkpoint(run / "best.pt", model, **checkpoint)
            print("epoch {:02d} train_rmse={:.6f} val_rmse={:.6f} lr={:.2g} {:.1f}s{}".format(
                epoch, np.sqrt(train["mse"]), np.sqrt(val["mse"]), row["lr"], row["seconds"],
                " *best" if improved else ""), flush=True)
            if bad_epochs >= args.patience:
                print("Early stopping", flush=True)
                break
        summary = dict(status="trained", best_epoch=best_epoch, best_val_mse=best,
                       epochs_completed=len(history), elapsed_s=time.monotonic() - started,
                       checkpoint=str(run / "best.pt"))
        write_json(run / "summary.json", summary)
        plot_history(history, run / "learning_curve.png")
        print("BEST {} (epoch {})".format(run / "best.pt", best_epoch), flush=True)
    except BaseException as exc:
        write_json(run / "summary.json", dict(status="interrupted_or_failed", error=str(exc),
                    epochs_completed=len(history), best_epoch=best_epoch))
        raise


if __name__ == "__main__":
    main()
