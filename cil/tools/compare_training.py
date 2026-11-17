"""동일한 고정 Validation으로 완료된 학습 실행을 비교한다. 독립 시험·재학습은 하지 않는다."""

import argparse  # 비교할 실행 목록과 결과 폴더 이름을 받는다.
import csv  # epoch 기록을 읽고 비교 표를 저장한다.
import json  # 학습 설정과 선택 근거를 저장한다.
from datetime import datetime, timezone  # 결과를 만든 시각을 남긴다.
from pathlib import Path  # CIL 폴더를 기준으로 경로를 처리한다.

try:  # 직접 실행과 python -m tools.compare_training을 모두 지원한다.
    from .summarize_training import BASE, file_hash, number, read_json, require_close
except ImportError:
    from summarize_training import BASE, file_hash, number, read_json, require_close


def check_name(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(character not in allowed for character in name):
        raise ValueError("run names must use letters, numbers, underscores or hyphens")


def load_run(name, base):
    check_name(name)
    folder = base / "runs" / name  # 원본 실행 폴더에는 새 파일을 쓰지 않는다.
    checkpoint_path = base / "checkpoints" / name / "best.pt"
    settings = read_json(folder / "settings.json")
    summary = read_json(folder / "summary.json")  # 완료되지 않은 실행은 비교하지 않는다.
    if summary.get("status") not in ("completed", "stopped_by_user") or summary.get("run_name") != name or settings.get("name") != name:
        raise ValueError(f"{name}: expected completed or user-stopped records for the requested run")
    if settings.get("best_criterion") != "minimum validation mean_command_mse":
        raise ValueError(f"{name}: unsupported best-checkpoint selection criterion")

    with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history or [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"{name}: history must contain contiguous epochs starting at one")
    for row in history:
        for key, value in row.items():
            number(value, key)  # 손상된 기록이나 NaN/Inf 지표를 거부한다.
    best_row = min(history, key=lambda row: float(row["val_mean_command_mse"]))
    best_epoch = int(best_row["epoch"])
    if int(summary["completed_epochs"]) != len(history) or int(summary["best_epoch"]) != best_epoch:
        raise ValueError(f"{name}: summary epoch differs from history")
    require_close(summary["best_validation_mean_command_mse"], best_row["val_mean_command_mse"], name + " summary score")

    checkpoint_sha = file_hash(checkpoint_path)
    if checkpoint_sha != summary["best_checkpoint_sha256"]:
        raise ValueError(f"{name}: checkpoint hash differs from summary")

    import torch  # --help는 PyTorch·GPU를 초기화하지 않는다.

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != best_epoch:
        raise ValueError(f"{name}: checkpoint epoch differs from history")
    if json.loads(json.dumps(checkpoint["settings"])) != settings:
        raise ValueError(f"{name}: checkpoint settings differ from settings.json")
    metrics = checkpoint["val_metrics"]
    for key, value in metrics.items():
        require_close(value, best_row["val_" + key], name + " checkpoint " + key)
    require_close(checkpoint["best_score"], metrics["mean_command_mse"], name + " checkpoint score")
    command_mse = [number(metrics[f"command_{index}_mse"], "command MSE") for index in range(4)]
    require_close(sum(command_mse) / 4, metrics["mean_command_mse"], name + " macro MSE")

    if int(summary["training_samples"]) != int(settings["train_samples"]) or int(summary["validation_samples"]) != int(settings["val_samples"]):
        raise ValueError(f"{name}: summary sample counts differ from settings")
    layout = {key: list(value.shape) for key, value in checkpoint["model_state_dict"].items()}
    if not layout:
        raise ValueError(f"{name}: checkpoint has no model state")
    # Dropout 확률 변경에는 학습 가중치의 층 이름·크기 변경이 필요하지 않는다.
    row = {
        "run_name": name, "run_status": summary["status"],
        "stop_reason": summary["stop_reason"], "lr": number(settings["lr"], "lr"),
        "weight_decay": number(settings.get("weight_decay", 0.0), "weight_decay"),
        "conv_dropout": number(settings.get("conv_dropout", 0.20), "conv_dropout"),
        "fc_dropout": number(settings.get("fc_dropout", 0.50), "fc_dropout"),
        "completed_epochs": len(history), "best_epoch": best_epoch,
        "val_macro_mse": number(metrics["mean_command_mse"], "macro MSE"),
        "overall_control_mse": number(metrics["control_mse"], "overall control MSE"),
        "steer_mae": number(metrics["steer_mae"], "steer MAE"),
        "throttle_mae": number(metrics["throttle_mae"], "throttle MAE"),
        "brake_mae": number(metrics["brake_mae"], "brake MAE"),
        "aux_speed_mae_kph": number(metrics["speed_mae_kph"], "speed MAE"),
    }  # baseline이 기록하지 않았던 하이퍼파라미터만 기존 기본값으로 채운다.
    if row["lr"] <= 0 or row["weight_decay"] < 0 or any(not 0 <= row[key] < 1 for key in ("conv_dropout", "fc_dropout")):
        raise ValueError(f"{name}: invalid hyperparameters")
    return {"row": row, "settings": settings, "summary": summary, "history": history,
            "checkpoint": checkpoint_path, "checkpoint_sha256": checkpoint_sha, "layout": layout,
            "source_sha256": {filename: file_hash(folder / filename) for filename in ("history.csv", "settings.json", "summary.json")}}


def compare_runs(run_names, output_name="cil_hparam_tuning", base=BASE):
    check_name(output_name)
    if len(run_names) < 2 or len(set(run_names)) != len(run_names):
        raise ValueError("provide at least two distinct runs; the first run is the baseline")
    if output_name in run_names:
        raise ValueError("comparison output must differ from every training run")
    records = [load_run(name, base) for name in run_names]
    baseline = records[0]
    settings = baseline["settings"]
    if settings.get("command_names") != ["follow_road", "left", "straight", "right"]:
        raise ValueError("unexpected command index mapping")
    if settings.get("control_order") != ["steer", "throttle", "brake"] or settings.get("steering_unit") != "normalized_-1_1":
        raise ValueError("unexpected action order or units")

    fixed_fields = ("architecture", "command_names", "control_order", "steering_unit", "preprocess",
                    "train_samples", "val_samples", "train_commands", "batch_size", "aux_weight", "sampling",
                    "seed", "workers")
    fixed_hashes = ("train.csv", "val.csv", "model.py", "dataset.py")
    for record in records:
        current = record["settings"]
        for key in fixed_fields:
            if current[key] != settings[key]:
                raise ValueError(f"{record['row']['run_name']}: changed fixed setting {key}")
        for key in fixed_hashes:
            if current["hashes"][key] != settings["hashes"][key]:
                raise ValueError(f"{record['row']['run_name']}: changed fixed source {key}")
        if record["layout"] != baseline["layout"]:
            raise ValueError(f"{record['row']['run_name']}: model parameter layout differs from baseline")
    # train.py는 튜닝 인자를 추가했으므로 해시가 달라도 된다. 데이터·모델 코드는 고정한다.
    for key in fixed_hashes:
        path = base / "data" / "prepared" / key if key.endswith(".csv") else base / key
        if file_hash(path) != settings["hashes"][key]:
            raise ValueError(f"current fixed source differs from recorded training source: {key}")

    counts = [0] * 4
    with (base / "data" / "prepared" / "val.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["camera"] == "center":  # Validation에 사용한 가운데 영상만 센다.
                command = int(row["command"])
                if command not in range(4):
                    raise ValueError("invalid validation command")
                counts[command] += 1
    if not all(counts) or sum(counts) != int(settings["val_samples"]):
        raise ValueError("validation sample counts differ from the recorded run")
    for record in records:
        best_row = record["history"][record["row"]["best_epoch"] - 1]
        natural_mse = sum(counts[i] * float(best_row[f"val_command_{i}_mse"]) for i in range(4)) / sum(counts)
        require_close(natural_mse, record["row"]["overall_control_mse"], record["row"]["run_name"] + " natural MSE", rel_tol=1e-5)

    # 동점이면 입력 목록에서 먼저 나온 실행을 유지한다. baseline도 선택 대상이다.
    selected = min(records, key=lambda record: record["row"]["val_macro_mse"])
    baseline_score = baseline["row"]["val_macro_mse"]
    rows = []
    for record in records:
        row = dict(record["row"])
        row["selected"] = record is selected
        row["macro_mse_reduction_vs_baseline_percent"] = 100 * (baseline_score - row["val_macro_mse"]) / baseline_score if baseline_score > 0 else None
        rows.append(row)

    output = base / "runs" / output_name
    if any((output / name).exists() for name in ("settings.json", "history.csv", "summary.json", "software")):
        raise FileExistsError("output folder contains training records; choose a comparison folder name")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib  # 그래프는 저장만 하고 GUI를 열지 않는다.

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, record in enumerate(records):
        row = record["row"]
        title = "Baseline" if index == 0 else f"Trial {index}"
        label = (f"{title}: lr={row['lr']:g}, wd={row['weight_decay']:g}\n"
                 f"conv={row['conv_dropout']:g}, fc={row['fc_dropout']:g}")
        epochs = [int(item["epoch"]) for item in record["history"]]
        axes[0].plot(epochs, [float(item["val_mean_command_mse"]) for item in record["history"]], label=label)
        color = axes[0].lines[-1].get_color()
        axes[0].scatter([row["best_epoch"]], [row["val_macro_mse"]], marker="*", s=80, color=color)
        axes[1].plot(epochs, [float(item["train_control_mse"]) for item in record["history"]], color=color, label=label)
    axes[0].set(title="Fixed validation: equal-weight command MSE", xlabel="Epoch", ylabel="Normalized control MSE")
    axes[1].set(title="Training: control MSE", xlabel="Epoch", ylabel="Normalized control MSE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("CIL hyperparameter selection on fixed validation")
    figure.text(0.5, 0.02, "Train: active dropout, augmentation and balanced commands. Validation: eval mode and natural distribution. Not an independent test.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output / "learning_comparison.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "hyperparameter selection on the same fixed validation; not an independent test or closed-loop evaluation",
        "baseline_run": run_names[0], "runs": rows,
        "selected_run": selected["row"]["run_name"],
        "selected_checkpoint": str(selected["checkpoint"]),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_criterion": "minimum validation mean_command_mse; ties keep first input run",
        "baseline_defaults_for_missing_settings": {"weight_decay": 0.0, "conv_dropout": 0.20, "fc_dropout": 0.50},
        "validation_command_counts": dict(zip(settings["command_names"], counts)),
        "verified_fixed_settings": {key: settings[key] for key in fixed_fields},
        "verified_fixed_sha256": {key: settings["hashes"][key] for key in fixed_hashes},
        "train_source_hash_may_differ": True,
        "sources": {record["row"]["run_name"]: {**record["source_sha256"], "best.pt": record["checkpoint_sha256"]} for record in records},
        "outputs": ["comparison.csv", "comparison.json", "learning_comparison.png"],
        "notes": [
            "Steering MAE is in normalized control units, not degrees, radians or road-following error.",
            "Auxiliary speed MAE is for actual speed predicted from images, expressed in km/h.",
            "Training errors use different dropout settings and should not be interpreted as directly comparable validation errors.",
            "No model or image was re-evaluated. Public selected.json and original run records were not changed.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="baseline run first, then completed candidate runs")
    parser.add_argument("--output-name", default="cil_hparam_tuning", help="comparison folder name under cil/runs/")
    args = parser.parse_args()
    report = compare_runs(args.runs, args.output_name)
    print(json.dumps({key: report[key] for key in ("baseline_run", "selected_run", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
