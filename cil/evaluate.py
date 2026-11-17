"""저장된 이미지·속도·명령으로 CIL의 제어값을 예측해 Teacher와 비교한다.

학습이나 MORAI 제어는 하지 않는다. 기본 입력은 모델 선택에 사용한 Validation이다.
"""

import argparse  # 체크포인트와 평가 CSV를 실행할 때 선택한다.
import csv  # 이미지별 정답과 예측을 표로 저장한다.
import hashlib  # 평가한 가중치와 데이터 목록을 정확하게 기록한다.
import json  # 전체·명령별 지표를 저장한다.
from datetime import datetime, timezone  # 평가 결과마다 새 폴더를 만든다.
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # 화면 창을 열지 않고 그래프를 파일로 저장한다.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CILDataset, CROP, IMAGE_SIZE, SPEED_SCALE_KPH
from model import CILModel

BASE = Path(__file__).resolve().parent  # 상대경로는 cil 폴더를 기준으로 한다.
CONTROLS = ("steer", "throttle", "brake")  # 모델 출력과 정답의 공통 순서다.


def resolve_path(value):
    path = Path(value)  # 사용자 입력을 운영체제 경로로 바꾼다.
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()  # 같은 파일인지 비교할 식별값이다.


def metrics_for(target, prediction, speed, pred_speed):
    error = prediction - target  # 모든 제어 오차는 정규화 단위로 계산한다.
    return {
        "samples": len(target),
        "control_mse": float(np.mean(error ** 2)),  # 조향·가속·브레이크 MSE의 평균이다.
        **{name + "_mae": float(np.mean(np.abs(error[:, index])))
           for index, name in enumerate(CONTROLS)},
        "speed_mae_kph": float(np.mean(np.abs(pred_speed - speed))),
    }


def evaluate(model, loader, device):
    targets, predictions, speeds, predicted_speeds, commands, paths = [], [], [], [], [], []
    model.eval()  # Dropout을 끄고 학습된 BatchNorm 통계로 평가한다.
    with torch.inference_mode():  # 가중치 갱신과 gradient 계산을 모두 끈다.
        for batch in loader:
            output = model(
                batch["image"].to(device),
                batch["speed"].to(device),
                batch["command"].to(device),
            )  # 각 이미지에 기록된 명령의 branch 출력을 선택한다.
            targets.append(batch["target"].numpy())
            predictions.append(output["control"].cpu().numpy())  # 손실 계산 전 출력을 자르지 않는다.
            speeds.append(batch["speed"].numpy().reshape(-1) * SPEED_SCALE_KPH)
            predicted_speeds.append(output["pred_speed"].cpu().numpy().reshape(-1) * SPEED_SCALE_KPH)
            commands.append(batch["command"].numpy())
            paths.extend(batch["image_path"])  # 실패한 예측에서 원본 이미지를 찾을 수 있다.

    arrays = [np.concatenate(values).astype(np.float64)
              for values in (targets, predictions, speeds, predicted_speeds)]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("non-finite target or prediction")  # NaN을 정상 지표로 저장하지 않는다.
    return (*arrays, np.concatenate(commands), paths)


def save_predictions(output_dir, target, prediction, speed, pred_speed, command, paths):
    fields = ["image_path", "command", "command_name", "speed_kph", "pred_speed_kph"]
    fields += [prefix + name for name in CONTROLS for prefix in ("target_", "pred_", "abs_error_")]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()  # 한 행이 이미지 한 장에 대응한다.
        for index, image_path in enumerate(paths):
            path = Path(image_path)
            stored_path = str(path.relative_to(BASE)) if BASE in path.parents else str(path)
            row = {"image_path": stored_path, "command": int(command[index]),
                   "command_name": CILModel.command_names[command[index]],
                   "speed_kph": speed[index], "pred_speed_kph": pred_speed[index]}
            for column, name in enumerate(CONTROLS):
                row["target_" + name] = target[index, column]
                row["pred_" + name] = prediction[index, column]
                row["abs_error_" + name] = abs(prediction[index, column] - target[index, column])
            writer.writerow(row)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for column, (axis, name) in enumerate(zip(axes, CONTROLS)):
        axis.plot(target[:, column], label="Teacher", linewidth=1)
        axis.plot(prediction[:, column], label="CIL prediction", linewidth=.8, alpha=.8)
        axis.set_ylabel(name + " (normalized)")
        axis.grid(alpha=.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Sample index (CSV order; separate sessions concatenated)")
    fig.suptitle("CIL open-loop: center camera, no augmentation")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")  # 생략하면 selected.json이 지정한 최종 가중치를 사용한다.
    parser.add_argument("--csv", default="data/prepared/val.csv")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")  # 기존 폴더를 덮어쓰지 않도록 새 경로만 허용한다.
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")

    selected = None
    if args.checkpoint is None:
        selected = json.loads((BASE / "checkpoints/selected.json").read_text())
    checkpoint_path = resolve_path(args.checkpoint or selected["checkpoint"])
    checkpoint_hash = sha256(checkpoint_path)
    if selected is not None and checkpoint_hash != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]  # 가중치가 기대하는 입출력 조건을 확인한다.
    expected = {"crop": list(CROP), "image_size_wh": list(IMAGE_SIZE),
                "color": "RGB", "pixel_range": [0, 1], "speed_scale_kph": SPEED_SCALE_KPH}
    if (settings["architecture"] != CILModel.architecture
            or settings["command_names"] != list(CILModel.command_names)
            or settings["control_order"] != list(CONTROLS)
            or json.loads(json.dumps(settings["preprocess"])) != expected):  # tuple·list 표기 차이는 허용한다.
        raise ValueError("checkpoint architecture or preprocessing mismatch")
    for name in ("model.py", "dataset.py"):
        if sha256(BASE / name) != settings["hashes"][name]:
            raise ValueError(name + " changed since training; review before evaluating")

    torch.set_num_threads(4)  # 이미지 디코딩과 CPU 연산의 과도한 스레드 경쟁을 줄인다.
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    csv_path = resolve_path(args.csv)
    csv_hash = sha256(csv_path)
    scope = "user_provided_csv_independence_not_verified"  # 새 CSV라고 자동으로 미학습 시험이 되지는 않는다.
    if csv_hash == settings["hashes"]["val.csv"]:
        scope = "validation_model_selection"
    elif csv_hash == settings["hashes"]["train.csv"]:
        scope = "training_data"

    dataset = CILDataset(csv_path, split="val", cameras=("center",), augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = CILModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target, prediction, speed, pred_speed, command, paths = evaluate(model, loader, device)
    per_command = {}
    for number, name in enumerate(CILModel.command_names):
        mask = command == number
        per_command[name] = (metrics_for(target[mask], prediction[mask], speed[mask], pred_speed[mask])
                             if mask.any() else None)  # 없는 명령은 0점으로 계산하지 않는다.
    metrics = metrics_for(target, prediction, speed, pred_speed)
    metrics["mean_command_mse"] = float(np.mean(
        [value["control_mse"] for value in per_command.values() if value is not None]
    ))  # 명령별 표본 수 차이가 모델 선택 점수를 지배하지 않도록 같은 비중으로 평균한다.
    metrics["per_command"] = per_command

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = resolve_path(args.output or "runs/evaluation/" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_predictions(output_dir, target, prediction, speed, pred_speed, command, paths)
    report = {"evaluation_scope": scope, "checkpoint": str(checkpoint_path.relative_to(BASE))
              if BASE in checkpoint_path.parents else str(checkpoint_path),
              "checkpoint_sha256": checkpoint_hash, "epoch": checkpoint["epoch"],
              "csv": str(csv_path.relative_to(BASE)) if BASE in csv_path.parents else str(csv_path),
              "csv_sha256": csv_hash, "preprocess": expected, "camera": "center",
              "augmentation": False, "device": str(device), "metrics": metrics,
              "notes": ["Raw linear outputs are evaluated without clamping.",
                        "Steering errors use normalized control units, not degrees.",
                        "Open-loop agreement does not measure closed-loop route completion."]}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{scope}: {len(dataset)} images, epoch={checkpoint['epoch']}")
    print(f"mean_command_mse={metrics['mean_command_mse']:.8f}, steer_mae={metrics['steer_mae']:.8f}")
    print(output_dir)


if __name__ == "__main__":
    main()  # import할 때는 평가를 자동 실행하지 않는다.
