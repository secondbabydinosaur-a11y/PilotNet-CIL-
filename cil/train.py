"""Dataset과 CILModel을 연결해 학습하고 validation으로 가중치를 선택한다."""

import argparse  # 실행할 때 학습 조건을 지정한다.
import csv  # epoch별 학습 기록을 저장한다.
import hashlib  # 어떤 데이터와 코드로 학습했는지 기록한다.
import json  # 학습 설정을 읽기 쉬운 파일로 저장한다.
import math  # 한 epoch의 배치 수를 계산한다.
import random  # Python의 난수 시작점을 고정한다.
import shutil  # 실행에 사용한 코드를 학습 기록과 함께 보관한다.
import time  # 배치 진행 상황과 학습 시간을 기록한다.
from datetime import datetime, timezone  # 실행별 저장 폴더 이름을 만든다.
from pathlib import Path  # 파일 경로를 다룬다.

import cv2  # 이미지 로딩에 쓰는 OpenCV의 스레드 수를 제한한다.
import numpy as np  # 명령별 샘플 선택과 영상 증강의 난수를 관리한다.
import torch  # GPU 계산과 자동 미분을 사용한다.
from torch.utils.data import DataLoader, Sampler  # 배치 로딩과 샘플 선택을 담당한다.

from dataset import BASE, CAMERAS, CROP, IMAGE_SIZE, SPEED_SCALE_KPH, CILDataset
from model import CILModel  # 네가 작성한 신경망 구조를 가져온다.


class CommandBalancedBatchSampler(Sampler):
    """각 배치에서 네 명령을 같은 수만큼 뽑는다. 부족한 명령은 재사용한다."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size <= 0 or batch_size % 4 != 0:  # 명령이 네 종류다.
            raise ValueError("batch_size must be a positive multiple of 4")

        self.groups = [
            [i for i, sample in enumerate(dataset.samples) if sample[3] == command]
            for command in range(4)
        ]  # CSV의 각 명령에 해당하는 Dataset 인덱스를 모은다.

        if any(len(group) == 0 for group in self.groups):  # 빈 branch가 있으면
            raise ValueError("training data must contain all four commands")

        self.per_command = batch_size // 4  # 배치 120이면 명령마다 30개다.
        self.steps = math.ceil(len(dataset) / batch_size)  # epoch당 배치 수다.
        self.seed = seed  # 샘플 선택을 재현하기 위한 시작값이다.
        self.epoch = 0  # epoch마다 다른 순서를 사용한다.

    def __len__(self):
        return self.steps  # DataLoader에 배치 수를 알려준다.

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)  # 이번 epoch의 난수다.

        for _ in range(self.steps):
            batch = np.concatenate([
                rng.choice(group, self.per_command, replace=True)
                for group in self.groups
            ])  # 명령마다 복원 추출하므로 희소한 명령도 충분히 학습한다.

            rng.shuffle(batch)  # 배치 내부에서도 명령 순서를 섞는다.
            yield batch.tolist()  # 실제 샘플 로딩은 Dataset이 담당한다.


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)  # worker마다 다른 seed다.
    np.random.seed(worker_seed)  # Dataset의 영상 증강 난수를 설정한다.
    random.seed(worker_seed)  # Python 난수도 함께 설정한다.
    cv2.setNumThreads(1)  # 여러 worker가 CPU 스레드를 과도하게 만들지 않게 한다.


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()  # 입력 버전을 기록한다.


def configure_dropout(model, conv_probability, fc_probability):
    """기존 Dropout의 확률만 바꾸며 층·채널·branch 구조는 그대로 둔다."""
    probabilities = (conv_probability, fc_probability)
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in probabilities):
        raise ValueError("dropout probabilities must be finite and in [0, 1)")

    counts = {"conv": 0, "fc": 0}  # 실제 변경한 Dropout 수를 기록한다.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            kind = "conv" if name.startswith("image_conv.") else "fc"
            module.p = conv_probability if kind == "conv" else fc_probability
            counts[kind] += 1
    return counts


def run_epoch(model, loader, device, aux_weight, optimizer=None):
    training = optimizer is not None  # optimizer를 받았을 때만 가중치를 갱신한다.
    model.train(training)  # 학습/검증에 맞춰 Dropout과 BatchNorm 동작을 바꾼다.

    total = 0  # 이번 epoch에서 실제 처리한 샘플 수다.
    total_loss = 0.0  # 전체 loss를 샘플 수로 가중합한다.
    absolute_error = torch.zeros(3, device=device)  # 제어값별 절댓값 오차 합이다.
    speed_error = torch.zeros((), device=device)  # 보조 속도 예측 오차 합이다.
    command_squared = torch.zeros(4, device=device)  # 명령별 제어 MSE 합이다.
    command_count = torch.zeros(4, device=device)  # 명령별 처리 샘플 수다.
    started = time.monotonic()  # 이번 학습 또는 검증 단계의 시작 시각이다.

    with torch.set_grad_enabled(training):  # validation에서는 gradient를 만들지 않는다.
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)  # [B, 3, 88, 200]
            speed = batch["speed"].to(device, non_blocking=True)  # [B, 1], 속도/20
            command = batch["command"].to(device, non_blocking=True)  # [B], 명령 번호
            target = batch["target"].to(device, non_blocking=True)  # [B, 3], Teacher 제어

            if training:
                optimizer.zero_grad(set_to_none=True)  # 이전 배치의 gradient를 지운다.

            output = model(image, speed, command)  # 각 샘플의 명령 branch를 선택한다.
            error = output["control"] - target  # 조향·가속·브레이크 예측 오차다.
            sample_mse = error.square().mean(dim=1)  # 샘플별 세 제어값의 평균 제곱 오차다.
            speed_mse = (output["pred_speed"] - speed).square().mean()  # 보조 출력 loss다.
            loss = sample_mse.mean() + aux_weight * speed_mse  # 최종 학습 목적함수다.

            if not torch.isfinite(loss):  # NaN/Inf 가중치가 저장되는 것을 막는다.
                raise RuntimeError("non-finite loss")

            if training:
                loss.backward()  # loss가 각 가중치에 미치는 영향을 계산한다.
                optimizer.step()  # Adam으로 가중치를 한 번 갱신한다.

            size = image.shape[0]  # 마지막 validation 배치는 더 작을 수 있다.
            total += size  # 실제 처리한 샘플 수를 누적한다.
            total_loss += loss.detach().item() * size  # 배치 평균을 샘플 합으로 바꾼다.
            absolute_error += error.detach().abs().sum(dim=0)  # 제어별 MAE 계산용이다.
            speed_error += (output["pred_speed"].detach() - speed).abs().sum()
            command_squared.index_add_(0, command, sample_mse.detach())  # 명령별 MSE 합이다.
            command_count.index_add_(0, command, torch.ones(size, device=device))

            if training and step % 50 == 0:  # 긴 epoch에서도 진행 상황을 확인한다.
                elapsed = time.monotonic() - started
                print(
                    f"  train batch={step}/{len(loader)} "
                    f"loss={total_loss / total:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if total == 0 or torch.any(command_count == 0):  # 명령별 비교가 가능해야 한다.
        raise ValueError("each epoch must contain samples for all four commands")

    control_mae = (absolute_error / total).cpu().tolist()  # 정규화 제어값의 MAE다.
    command_mse = (command_squared / command_count).cpu().tolist()  # 명령별 평균이다.

    return {
        "loss": total_loss / total,  # 보조 속도 loss까지 포함한 평균이다.
        "control_mse": command_squared.sum().item() / total,  # 처리한 전체 샘플의 제어 MSE다.
        "mean_command_mse": sum(command_mse) / 4,  # 네 명령의 MSE를 같은 비중으로 평균한다.
        "steer_mae": control_mae[0],  # 단위는 도/라디안이 아닌 정규화 조향이다.
        "throttle_mae": control_mae[1],  # 가속 페달 0~1 기준이다.
        "brake_mae": control_mae[2],  # 브레이크 0~1 기준이다.
        "speed_mae_kph": speed_error.item() / total * SPEED_SCALE_KPH,
        "seconds": time.monotonic() - started,  # 이번 단계의 소요 시간이다.
        **{f"command_{i}_mse": value for i, value in enumerate(command_mse)},
    }  # 학습 그래프를 붙잡지 않는 Python 숫자만 반환한다.


def main():
    parser = argparse.ArgumentParser()  # 실행 옵션을 정의한다.
    parser.add_argument("--epochs", type=int, default=30)  # 최대 학습 횟수다.
    parser.add_argument("--batch-size", type=int, default=120)  # 네 명령에 균등 배분한다.
    parser.add_argument("--lr", type=float, default=0.0002)  # Adam 학습률이다.
    parser.add_argument("--weight-decay", type=float, default=0.0)  # Adam의 L2 규제다.
    parser.add_argument("--conv-dropout", type=float, default=0.20)  # 기존 Conv Dropout 확률이다.
    parser.add_argument("--fc-dropout", type=float, default=0.50)  # 기존 FC Dropout 확률이다.
    parser.add_argument("--aux-weight", type=float, default=0.05)  # 이번 구현의 보조 loss 비중이다.
    parser.add_argument("--patience", type=int, default=8)  # 검증 성능이 개선되지 않으면 멈춘다.
    parser.add_argument("--workers", type=int, default=4)  # 이미지를 읽을 별도 프로세스 수다.
    parser.add_argument("--seed", type=int, default=7)  # 난수 시작점이다.
    parser.add_argument("--name")  # 결과를 저장할 실행 이름이다.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()  # 명령줄 옵션을 읽는다.

    if (args.epochs < 1 or args.patience < 1 or args.workers < 0
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or any(not math.isfinite(value) or not 0 <= value < 1
                   for value in (args.conv_dropout, args.fc_dropout))
            or not math.isfinite(args.aux_weight) or args.aux_weight < 0):
        parser.error("invalid training settings")  # 잘못된 조건은 실행 전에 알려준다.

    random.seed(args.seed)  # Python 난수를 고정한다.
    np.random.seed(args.seed)  # workers=0일 때도 영상 증강 seed를 설정한다.
    torch.manual_seed(args.seed)  # 모델 초기값과 DataLoader seed를 설정한다.
    cv2.setNumThreads(1)  # 부모 프로세스에서도 OpenCV 스레드를 제한한다.
    torch.backends.cudnn.benchmark = False  # 입력마다 다른 알고리즘을 탐색하지 않는다.
    torch.backends.cudnn.deterministic = True  # 가능한 범위에서 재현성을 높인다.

    device_name = "cuda" if torch.cuda.is_available() else "cpu"  # 자동 선택 기준이다.
    device = torch.device(device_name if args.device == "auto" else args.device)
    train_csv = BASE / "data/prepared/train.csv"  # 이미 고정한 학습 목록이다.
    val_csv = BASE / "data/prepared/val.csv"  # 별도 도로의 검증 목록이다.

    train_data = CILDataset(train_csv, split="train", cameras=CAMERAS, augment=True)
    val_data = CILDataset(val_csv, split="val", cameras=("center",), augment=False)
    sampler = CommandBalancedBatchSampler(train_data, args.batch_size, args.seed)

    loader_options = dict(
        num_workers=args.workers,  # 이미지 로딩 프로세스 수다.
        pin_memory=device.type == "cuda",  # CPU→GPU 전송을 돕는다.
        worker_init_fn=seed_worker,  # worker별 영상 증강 난수를 설정한다.
    )
    train_loader = DataLoader(train_data, batch_sampler=sampler, **loader_options)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )  # validation은 중복 추출이나 증강 없이 전체를 한 번씩 평가한다.

    model = CILModel().to(device)  # 신경망을 생성하고 GPU 또는 CPU에 올린다.
    dropout_counts = configure_dropout(model, args.conv_dropout, args.fc_dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )  # 구조를 바꾸지 않고 학습률·L2 규제만 조절한다.
    name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if any(character not in allowed for character in name):  # 실행 이름에 경로 문자를 막는다.
        parser.error("name must use letters, numbers, underscores or hyphens")

    run_dir = BASE / "runs" / name  # loss와 학습 설정을 보관한다.
    checkpoint_dir = BASE / "checkpoints" / name  # 학습된 가중치를 보관한다.
    if run_dir.exists() or checkpoint_dir.exists():  # 기존 실험은 덮어쓰지 않는다.
        raise FileExistsError(f"run name already exists: {name}")
    run_dir.mkdir(parents=True)  # 이번 실행의 기록 폴더를 만든다.
    checkpoint_dir.mkdir(parents=True)  # 이번 실행의 가중치 폴더를 만든다.

    software_dir = run_dir / "software"  # 학습 이후 코드가 바뀌어도 당시 코드를 보존한다.
    software_dir.mkdir()
    for source in (Path(__file__), BASE / "dataset.py", BASE / "model.py", BASE / "config.json"):
        shutil.copy2(source, software_dir / source.name)

    settings = {
        **vars(args), "name": name, "device": str(device),
        "architecture": model.architecture, "command_names": list(model.command_names),
        "control_order": ["steer", "throttle", "brake"], "steering_unit": "normalized_-1_1",
        "dropout_module_counts": dropout_counts,
        "state_dict_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": len(train_data), "val_samples": len(val_data),
        "train_commands": [len(group) for group in sampler.groups],
        "sampling": "equal commands per batch, with replacement",
        "batches_per_epoch": len(sampler), "samples_drawn_per_epoch": len(sampler) * args.batch_size,
        "preprocess": {"crop": CROP, "image_size_wh": IMAGE_SIZE,
                       "color": "RGB", "pixel_range": [0, 1], "speed_scale_kph": SPEED_SCALE_KPH},
        "hashes": {path.name: file_hash(path) for path in (
            train_csv, val_csv, BASE / "dataset.py", BASE / "model.py", BASE / "config.json", Path(__file__),
        )},  # 같은 이름의 파일을 수정해도 다른 버전임을 확인할 수 있다.
        "best_criterion": "minimum validation mean_command_mse",
    }  # 이후 평가와 실제 주행에도 사용할 설정을 남긴다.

    with (run_dir / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)  # 사람이 읽을 수 있게 저장한다.

    print(f"device={device} train={len(train_data)} val={len(val_data)}", flush=True)
    print(f"commands={settings['train_commands']} steps/epoch={len(sampler)}", flush=True)
    print(f"run={name}", flush=True)  # 이후 평가에서 사용할 실행 이름이다.
    print(
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"dropout_conv={args.conv_dropout} dropout_fc={args.fc_dropout}", flush=True,
    )
    best = float("inf")  # 작을수록 좋은 validation 점수다.
    stale_epochs = 0  # 마지막 개선 이후의 epoch 수다.
    best_epoch = 0  # 검증 기준으로 선택한 epoch 번호다.
    started = time.monotonic()  # 본 학습 전체 소요 시간을 측정한다.

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None  # 첫 epoch의 지표 이름으로 CSV 열을 만든다.

        for epoch in range(1, args.epochs + 1):
            print(f"epoch={epoch:02d}/{args.epochs} start", flush=True)
            sampler.epoch = epoch  # 이번 epoch에서는 다른 균형 배치를 뽑는다.
            train_metrics = run_epoch(model, train_loader, device, args.aux_weight, optimizer)
            val_metrics = run_epoch(model, val_loader, device, args.aux_weight)
            score = val_metrics["mean_command_mse"]  # 명령별 성능을 같은 비중으로 본다.
            improved = score < best  # 이번 모델이 기존 최선보다 나은지 확인한다.
            stale_epochs = 0 if improved else stale_epochs + 1
            best = min(best, score)  # 지금까지 가장 낮은 점수를 기억한다.
            if improved:
                best_epoch = epoch  # 가장 좋은 모델의 epoch를 기억한다.

            checkpoint = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "settings": settings,
                "val_metrics": val_metrics, "best_score": best,
            }  # 구조 이름·전처리·가중치를 함께 보관한다.
            torch.save(checkpoint, checkpoint_dir / "last.pt")  # 마지막 학습 상태다.
            if improved:
                torch.save(checkpoint, checkpoint_dir / "best.pt")  # 검증 기준 최선의 모델이다.

            row = {"epoch": epoch,
                   **{"train_" + key: value for key, value in train_metrics.items()},
                   **{"val_" + key: value for key, value in val_metrics.items()}}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))  # 열 순서를 고정한다.
                writer.writeheader()  # CSV 제목 행을 작성한다.
            writer.writerow(row)  # 이번 epoch의 모든 지표를 기록한다.
            stream.flush()  # 실행 중에도 기록을 확인할 수 있게 한다.

            print(
                f"epoch={epoch:02d} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_command_mse={score:.6f} "
                f"best={best:.6f} saved_best={improved} "
                f"seconds={train_metrics['seconds'] + val_metrics['seconds']:.1f}", flush=True,
            )  # 학습 진행과 validation 개선 여부를 출력한다.

            if stale_epochs >= args.patience:
                print("Early stopping: validation did not improve.", flush=True)
                break  # 학습 loss가 내려가더라도 검증 성능이 정체되면 멈춘다.

    summary = {
        "status": "completed", "run_name": name, "completed_epochs": epoch,
        "best_epoch": best_epoch, "best_validation_mean_command_mse": best,
        "stop_reason": "early_stopping" if stale_epochs >= args.patience else "max_epochs",
        "elapsed_seconds": time.monotonic() - started,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "best_checkpoint_sha256": file_hash(checkpoint_dir / "best.pt"),
        "training_samples": len(train_data), "validation_samples": len(val_data),
        "evaluation_scope": "validation only; closed-loop performance not evaluated",
    }  # 이번 학습의 완료 상태와 선택한 모델 정보를 남긴다.
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":  # import할 때 학습이 시작되지 않게 한다.
    main()  # 직접 실행했을 때만 학습한다.
