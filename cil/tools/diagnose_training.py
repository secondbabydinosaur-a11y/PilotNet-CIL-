"""같은 평가 조건에서 학습 가운데 영상과 Validation의 오차를 비교한다."""

import argparse  # 진단할 학습 실행을 고른다.
import hashlib  # 학습에 사용한 데이터 목록이 그대로인지 확인한다.
import json  # 진단 결과를 저장한다.
from pathlib import Path
import sys

import cv2
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from dataset import CILDataset
from model import CILModel
from train import configure_dropout, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.run or any(character not in allowed for character in args.run):
        parser.error("invalid run name")

    run_dir = BASE / "runs" / args.run
    settings = json.loads((run_dir / "settings.json").read_text())
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {}
    for split in ("train", "val"):
        csv_path = BASE / "data/prepared" / f"{split}.csv"
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == settings["hashes"][csv_path.name]
        datasets[split] = CILDataset(csv_path, split=split, cameras=("center",), augment=False)

    results = {}
    for checkpoint_name in ("best", "last"):
        path = BASE / "checkpoints" / args.run / f"{checkpoint_name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = CILModel()
        configure_dropout(model, settings.get("conv_dropout", .2), settings.get("fc_dropout", .5))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        results[checkpoint_name] = {"epoch": checkpoint["epoch"]}
        for split, dataset in datasets.items():
            loader = DataLoader(
                dataset, batch_size=settings["batch_size"], shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            metrics = run_epoch(model, loader, device, settings["aux_weight"])
            results[checkpoint_name][split] = {"samples": len(dataset), **metrics}
            print(f"{checkpoint_name} epoch={checkpoint['epoch']} {split}_center "
                  f"macro_mse={metrics['mean_command_mse']:.8f}", flush=True)
        del model

    report = {
        "run_name": args.run, "results": results,
        "shared_evaluation_conditions": {
            "camera": "center", "augmentation": False, "model_mode": "eval",
            "sampling": "all samples once in CSV order; no oversampling",
            "batch_size": settings["batch_size"],
        },
        "changes_last_minus_best": {
            split + "_macro_mse": results["last"][split]["mean_command_mse"]
            - results["best"][split]["mean_command_mse"] for split in ("train", "val")
        },
        "interpretation_limit": "Separate roads and recovery-event distributions still differ; a gap alone does not prove overfitting.",
        "scope": "fixed train and validation only; no held-out test or closed-loop evaluation",
    }
    path = run_dir / "generalization_diagnostic.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
