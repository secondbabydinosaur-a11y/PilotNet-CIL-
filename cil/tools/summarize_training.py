"""완료된 학습 기록으로 그래프·Validation 표를 만든다. 재학습·재평가는 하지 않는다."""

import argparse  # 요약할 실행 이름을 받는다.
import csv  # 학습 이력과 Validation 목록을 읽고 지표 표를 저장한다.
import hashlib  # 학습 당시 파일과 현재 파일이 일치하는지 확인한다.
import json  # 설정·학습 완료 기록·보고서를 처리한다.
import math  # NaN/Inf와 기록 사이의 수치 차이를 확인한다.
from collections import Counter  # Validation 명령별 샘플 수를 센다.
from datetime import datetime, timezone  # 보고서 생성 시각을 UTC로 기록한다.
from pathlib import Path  # CIL 폴더를 기준으로 파일을 찾는다.

BASE = Path(__file__).resolve().parents[1]  # 이 도구가 속한 cil 폴더다.


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))  # 저장된 JSON 객체를 읽는다.


def file_hash(path):
    digest = hashlib.sha256()  # 체크포인트 전체를 한 번에 메모리에 올리지 않는다.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, name):
    result = float(value)  # CSV 문자열도 숫자로 바꾼다.
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {name}")
    return result


def require_close(left, right, name, rel_tol=1e-6):
    if not math.isclose(number(left, name), number(right, name), rel_tol=rel_tol, abs_tol=1e-9):
        raise ValueError(f"inconsistent {name}: {left} != {right}")


def summarize(run_name, base=BASE):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_name or any(character not in allowed for character in run_name):
        raise ValueError("run name must use letters, numbers, underscores or hyphens")

    run_dir = base / "runs" / run_name  # 학습 기록만 읽는다.
    checkpoint_path = base / "checkpoints" / run_name / "best.pt"
    settings_path = run_dir / "settings.json"
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():  # 진행 중인 학습을 완료 결과로 발표하지 않는다.
        raise ValueError("summary.json is missing; run this tool after training completes")

    settings = read_json(settings_path)
    summary = read_json(summary_path)
    if summary.get("status") not in ("completed", "stopped_by_user") or summary.get("run_name") != run_name:
        raise ValueError("training summary must identify a completed or user-stopped run")
    if settings.get("name") != run_name:
        raise ValueError("settings run name differs from requested run")
    if settings.get("command_names") != ["follow_road", "left", "straight", "right"]:
        raise ValueError("unexpected command index mapping")
    if settings.get("control_order") != ["steer", "throttle", "brake"]:
        raise ValueError("unexpected control output order")
    if settings.get("best_criterion") != "minimum validation mean_command_mse":
        raise ValueError("unsupported best-checkpoint selection criterion")

    with history_path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))  # epoch별 집계를 읽는다. 영상은 읽지 않는다.
    if not history:
        raise ValueError("history.csv has no completed epochs")
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError("history must contain contiguous epochs starting at one")
    for row in history:
        for key, value in row.items():
            number(value, key)  # 그래프에 들어가는 모든 수치의 NaN/Inf를 검사한다.

    # 학습 코드와 동일하게 가장 낮은 명령 평균 MSE의 첫 epoch를 선택한다.
    best = min(history, key=lambda row: float(row["val_mean_command_mse"]))
    best_epoch = int(best["epoch"])
    if int(summary["completed_epochs"]) != epochs[-1] or int(summary["best_epoch"]) != best_epoch:
        raise ValueError("summary epoch differs from history")
    require_close(summary["best_validation_mean_command_mse"], best["val_mean_command_mse"], "summary best score")

    checkpoint_sha = file_hash(checkpoint_path)
    if checkpoint_sha != summary["best_checkpoint_sha256"]:
        raise ValueError("best checkpoint hash differs from training summary")

    import torch  # --help는 PyTorch나 GPU를 초기화하지 않는다.

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != best_epoch:
        raise ValueError("best checkpoint epoch differs from history")
    # JSON을 거치면 tuple과 list 표현 차이만 제거하고 설정값 자체를 비교할 수 있다.
    if json.loads(json.dumps(checkpoint["settings"])) != settings:
        raise ValueError("checkpoint settings differ from settings.json")
    metrics = checkpoint["val_metrics"]
    for key, value in metrics.items():
        require_close(value, best["val_" + key], "checkpoint " + key)
    require_close(checkpoint["best_score"], best["val_mean_command_mse"], "checkpoint best score")

    val_path = base / "data" / "prepared" / "val.csv"
    if file_hash(val_path) != settings["hashes"]["val.csv"]:
        raise ValueError("validation CSV has changed since training")
    with val_path.open(newline="", encoding="utf-8") as stream:
        counts = Counter(int(row["command"]) for row in csv.DictReader(stream) if row["camera"] == "center")
    if set(counts) != set(range(4)) or sum(counts.values()) != int(settings["val_samples"]):
        raise ValueError("validation command counts differ from settings")
    if int(summary["validation_samples"]) != sum(counts.values()):
        raise ValueError("validation sample count differs from summary")
    if int(summary["training_samples"]) != int(settings["train_samples"]):
        raise ValueError("training sample count differs from summary")

    command_mse = [number(metrics[f"command_{command}_mse"], "command MSE") for command in range(4)]
    require_close(sum(command_mse) / 4, metrics["mean_command_mse"], "macro command MSE")
    natural_mse = sum(counts[i] * command_mse[i] for i in range(4)) / sum(counts.values())
    require_close(natural_mse, metrics["control_mse"], "natural-distribution control MSE", rel_tol=1e-5)

    plot_path = run_dir / "learning_curve.png"
    table_path = run_dir / "validation_metrics.csv"
    report_path = run_dir / "training_report.json"
    rows = [{
        "scope": "command", "command_id": i, "command_name": name,
        "samples": counts[i], "control_mse": command_mse[i],
    } for i, name in enumerate(settings["command_names"])]
    rows += [{
        "scope": "overall_natural_distribution", "samples": sum(counts.values()),
        "control_mse": number(metrics["control_mse"], "overall control MSE"),
        "steer_mae_normalized": number(metrics["steer_mae"], "steer MAE"),
        "throttle_mae": number(metrics["throttle_mae"], "throttle MAE"),
        "brake_mae": number(metrics["brake_mae"], "brake MAE"),
        "aux_speed_mae_kph": number(metrics["speed_mae_kph"], "aux speed MAE"),
    }, {
        "scope": "equal_weight_command_average", "control_mse": sum(command_mse) / 4,
    }]  # 명령별 MAE는 원래 이력에 없으므로 추정해서 채우지 않는다.

    import matplotlib  # 저장용 그래프만 만들며 창은 띄우지 않는다.

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in (("train_loss", "Train: balanced, augmented"), ("val_loss", "Validation: natural, no augmentation")):
        axes[0].plot(epochs, [float(row[key]) for row in history], label=label)
    axes[0].set(title="Control + auxiliary speed loss", xlabel="Epoch", ylabel="Loss")
    for key, label in (("val_mean_command_mse", "Equal-weight command MSE (selection)"), ("val_control_mse", "Natural-distribution control MSE")):
        axes[1].plot(epochs, [float(row[key]) for row in history], label=label)
    axes[1].set(title="Validation control error", xlabel="Epoch", ylabel="Normalized control MSE")
    for axis in axes:
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.5, label=f"Best epoch: {best_epoch}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(run_name)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    columns = ["scope", "command_id", "command_name", "samples", "control_mse", "steer_mae_normalized", "throttle_mae", "brake_mae", "aux_speed_mae_kph"]
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": summary["status"], "run_status": summary["status"],
        "stop_reason": summary["stop_reason"], "run_name": run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "recorded validation metrics only; no held-out or closed-loop evaluation",
        "completed_epochs": epochs[-1], "selected_epoch": best_epoch,
        "selection_criterion": settings["best_criterion"],
        "selected_validation_metrics": {key: number(value, key) for key, value in metrics.items()},
        "validation_commands": rows[:4], "validation_samples": sum(counts.values()),
        "training_summary": summary, "training_settings": settings,
        "checks": {
            "history_summary_checkpoint_epochs_match": True,
            "history_checkpoint_metrics_match": True, "checkpoint_settings_match": True,
            "checkpoint_sha256_matches_summary": True, "validation_csv_sha256_matches_training": True,
            "macro_and_natural_control_mse_consistent": True,
        },
        "source_sha256": {"history.csv": file_hash(history_path), "settings.json": file_hash(settings_path),
                          "summary.json": file_hash(summary_path), "best.pt": checkpoint_sha,
                          "val.csv": file_hash(val_path)},
        "outputs": [path.name for path in (plot_path, table_path, report_path)],
        "notes": [
            "Control MSE averages normalized steering, throttle and brake errors; steering MAE is not degrees or radians.",
            "These offline errors do not measure route completion, lane departure or collision performance.",
            "The auxiliary head predicts actual speed / speed_scale_kph from the image; speed MAE is reported in km/h.",
            "Training uses balanced commands and augmentation; validation retains its natural command distribution.",
            "No images were re-evaluated and no model weights were changed by this tool.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="completed CIL training run name")
    args = parser.parse_args()
    report = summarize(args.run)
    print(json.dumps({key: report[key] for key in ("run_name", "completed_epochs", "selected_epoch", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
