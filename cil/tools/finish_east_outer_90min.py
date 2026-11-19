"""확정한 90분 데이터로 학습→고정 Val→실주행 1회→실제 영상 비교를 진행한다.

완료한 단계는 검증 후 재사용한다. 유효한 주행 실패도 결과이며 다시 운전하지 않는다.
시뮬레이터 I/O 실패를 다시 시도하려면 원인을 해결하고 --retry-io를 명시한다.
GitHub 업로드·기존 모델 선택 변경·추가 시험 경로 실행은 하지 않는다.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools.compare_morai_videos import probe
from tools.prepare_east_outer_90min import dense_xy

VALID_DRIVING_RESULTS = {"completed", "intervention_cte", "intervention_stalled", "intervention_reverse", "timeout"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else BASE / path).resolve()


def relative(path):
    return str(Path(path).resolve().relative_to(BASE))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)


def check_assist(actual, expected):
    require(isinstance(actual, dict), "missing declared first-left steering intervention")
    for key, value in expected.items():
        observed = actual.get(key)
        require(close(observed, value) if isinstance(value, (float, int)) and observed is not None else observed == value,
                "first-left steering intervention differs from protocol: " + key)


def read_rows(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def preflight(root):
    """훈련이나 통신을 시작하기 전에 실제 고정 입력과 최종 데이터 수를 검증한다."""
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path)
    require(protocol["evaluation_course"] == "EAST_OUTER_EXTENDED", "only the fixed primary extended course is supported")
    require(protocol["training_center_frame_target"] == 54000 and protocol["per_command_frame_target"] == 13500,
            "this runner requires the fixed 54000-center / 13500-per-command experiment")
    for name, expected in protocol["immutable_inputs_sha256"].items():
        require(file_hash(BASE / name) == expected, "immutable model/data input changed: " + name)
    route_path = resolve(protocol["route_csv"])
    require(file_hash(route_path) == protocol["route_sha256"], "fixed 452m evaluation route changed")
    manifest = read_json(BASE / "routes/test/manifest.json")["routes"][protocol["evaluation_course"]]
    require(manifest["route_sha256"] == protocol["route_sha256"], "driver route manifest differs from protocol")
    route = np.loadtxt(route_path, delimiter=",", skiprows=1, ndmin=2)
    require(close(route[-1, 0], protocol["closed_loop"]["route_length_m"]), "fixed route length changed")
    old_run = resolve(protocol["older_run"])
    old_settings, old_summary = read_json(old_run / "settings.json"), read_json(old_run / "summary.json")
    expected_old = protocol["older_checkpoint_sha256"]
    require(old_settings["checkpoint_sha256"] == old_summary["checkpoint_sha256"] == expected_old,
            "older video does not use the declared comparison model")
    require(file_hash(resolve(old_settings["checkpoint"])) == expected_old, "older frozen checkpoint changed")
    require(old_settings["route_sha256"] == protocol["route_sha256"], "older video is from another route")
    check_assist(old_settings.get("steering_assist"), protocol["closed_loop"]["same_earlier_steering_assist"])
    require(old_settings["route"]["navigation"] == manifest["navigation"], "categorical command geometry changed")
    old_csv = BASE / "runs/east_outer_data_v1/verified_retry/train.csv"
    require(file_hash(old_csv) == protocol["older_training_csv_sha256"], "prior video-model training list changed")
    old_centers = sum(row["camera"] == "center" for row in read_rows(old_csv))
    require(old_centers == protocol["older_valid_center_frames"], "prior training duration count differs")
    require(close(old_centers / 10., protocol["older_training_seconds_at_10hz"]), "prior training seconds differ")
    old_video = old_run / old_summary["screen_video"]
    probe(old_video)  # 전방 카메라 영상으로 실제 MORAI 화면을 대체하지 않는다.

    prepared = root / "prepared"
    require((prepared / "summary.json").is_file(),
            "90-minute prepared data is not finalized. Finish collection and --finalize first: " + str(prepared))
    ledger, summary = read_json(root / "ledger.json"), read_json(prepared / "summary.json")
    require(ledger.get("status") == "finalized", "collection ledger is not finalized")
    train, val = prepared / "train.csv", prepared / "val.csv"
    require(file_hash(train) == summary["train_sha256"], "final training CSV checksum changed")
    val_hash = protocol["immutable_inputs_sha256"]["data/prepared/val.csv"]
    require(file_hash(val) == summary["val_sha256"] == val_hash, "fixed validation CSV changed")
    require(file_hash(prepared / "source_ledger.json") == summary["source_ledger_sha256"], "selection source ledger changed")
    rows = read_rows(train)
    paths = [row["image_path"] for row in rows]
    require(len(paths) == len(set(paths)), "training CSV contains duplicated image paths")
    require(all(resolve(path).is_file() for path in paths), "one or more training images are missing")
    require(all(row.get("split") == "train" and "/closed_loop/" not in ("/" + row["image_path"].replace("\\", "/")) for row in rows),
            "evaluation rollout or non-training data was added to the training CSV")
    centers = [row for row in rows if row["camera"] == "center"]
    counts = Counter(str(row["command"]) for row in centers)
    expected = {str(i): int(protocol["per_command_frame_target"]) for i in range(4)}
    require(dict(counts) == expected and len(centers) == int(protocol["training_center_frame_target"]),
            "training requires exactly 54000 real center frames and 13500 per command")
    require(summary["center_counts"] == expected, "declared center counts differ from actual CSV")
    tree = cKDTree(dense_xy(route[:, 1:4]))
    xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    require(np.isfinite(xy).all(), "non-finite training positions")
    separation = max(0., float(tree.query(xy)[0].min()) - .25)
    require(separation >= float(protocol["reject_train_within_m"]), "training touches the entire frozen test corridor")
    audit = {"protocol_sha256": file_hash(protocol_path), "train_sha256": file_hash(train),
             "val_sha256": val_hash, "route_sha256": protocol["route_sha256"],
             "center_counts": dict(counts), "center_frames": len(centers),
             "training_observation_seconds": len(centers) / 10.,
             "train_rows": len(rows), "minimum_route_distance_lower_bound_m": separation,
             "old_training_observation_seconds": old_centers / 10.,
             "old_video_sha256": file_hash(old_video), "selection_summary_sha256": file_hash(prepared / "summary.json"),
             "interpretation": "new training includes whole-session corridor exclusions and balanced selection; not a pure added-data ablation"}
    return protocol, audit, old_run, old_video


def run_command(command, log_path):
    """자식의 진행·실패 기록을 보존하고, 사용자 중단 시 자식의 정리 코드를 실행한다."""
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
            raise


def training_command(protocol, root, name, workers, device):
    p = protocol["candidate_training"]
    command = [sys.executable, str(BASE / "train.py"), "--name", name,
               "--train-csv", relative(root / "prepared/train.csv"),
               "--val-csv", relative(root / "prepared/val.csv"), "--workers", str(workers), "--device", device]
    mapping = {"epochs": "epochs", "patience": "patience", "batch_size": "batch-size", "seed": "seed",
               "steps_per_epoch": "steps-per-epoch", "lr": "lr", "weight_decay": "weight-decay",
               "conv_dropout": "conv-dropout", "fc_dropout": "fc-dropout", "aux_weight": "aux-weight"}
    for key, flag in mapping.items():
        command += ["--" + flag, str(p[key])]
    return command


def audit_training(name, protocol, audit):
    run = BASE / "runs" / name
    settings, summary = read_json(run / "settings.json"), read_json(run / "summary.json")
    require(summary.get("status") == "completed", "training has no completed summary; inspect its log")
    for key, value in protocol["candidate_training"].items():
        if key in ("command_per_batch", "checkpoint_selection"): continue
        require(settings.get(key) == value, "trained setting differs from fixed protocol: " + key)
    require(settings["hashes"]["train.csv"] == audit["train_sha256"] and settings["hashes"]["val.csv"] == audit["val_sha256"],
            "checkpoint training/validation data differ from finalized selection")
    require(settings["batches_per_epoch"] == protocol["candidate_training"]["steps_per_epoch"]
            and settings["best_criterion"] == "minimum validation mean_command_mse",
            "optimizer update budget or checkpoint selection criterion differs")
    history = read_rows(run / "history.csv")
    require(len(history) == summary["completed_epochs"]
            and [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1)),
            "training history is missing or not contiguous")
    best = min(history, key=lambda row: float(row["val_mean_command_mse"]))
    require(int(best["epoch"]) == summary["best_epoch"]
            and close(best["val_mean_command_mse"], summary["best_validation_mean_command_mse"]),
            "best checkpoint was not selected by the declared fixed validation metric")
    for name_, expected in protocol["immutable_inputs_sha256"].items():
        if name_ in ("model.py", "dataset.py", "config.json"):
            require(settings["hashes"][name_] == expected, "trained input contract changed: " + name_)
    checkpoint = BASE / "checkpoints" / name / "best.pt"
    require(file_hash(checkpoint) == summary["best_checkpoint_sha256"], "best checkpoint changed")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)  # 확인만 하며 모델을 생성하지 않는다.
    require(saved["epoch"] == summary["best_epoch"]
            and json.loads(json.dumps(saved["settings"])) == settings,
            "best checkpoint and training summary differ")
    return checkpoint, summary


def validate_open_loop(folder, checkpoint, audit):
    report = read_json(folder / "metrics.json")
    require(report["checkpoint_sha256"] == file_hash(checkpoint), "validation used another checkpoint")
    require(report["csv_sha256"] == audit["val_sha256"] and report["evaluation_scope"] == "validation_model_selection",
            "open-loop report is not the fixed model-selection validation")
    require(report["camera"] == "center" and report["augmentation"] is False, "validation preprocessing differs")
    return report


def command_for(station, navigation):
    inside = [n for n in navigation if n["entry_s_m"] <= station <= n["exit_s_m"]]
    if inside: return inside[0]["command"]
    ahead = [n for n in navigation if 0 < n["entry_s_m"] - station <= n["approach_m"]]
    if ahead: return min(ahead, key=lambda n: n["entry_s_m"])["command"]
    behind = [n for n in navigation if 0 < station - n["exit_s_m"] <= n["exit_hold_m"]]
    return behind[-1]["command"] if behind else "follow_road"


def audit_drive(folder, protocol, expected_checkpoint, older_settings):
    settings, summary = read_json(folder / "settings.json"), read_json(folder / "summary.json")
    require(settings["course"] == summary["course"] == protocol["evaluation_course"], "drive used another course")
    require(settings["route_sha256"] == protocol["route_sha256"], "drive route differs")
    require(settings["checkpoint_sha256"] == summary["checkpoint_sha256"] == expected_checkpoint, "drive checkpoint differs")
    check_assist(settings.get("steering_assist"), protocol["closed_loop"]["same_earlier_steering_assist"])
    for key in ("max_cte_m", "seconds", "control_hz", "camera", "model_controls", "completion_definition", "stall_timeout_s"):
        require(settings[key] == older_settings[key], "old/new driving condition differs: " + key)
    require(settings["route"]["navigation"] == older_settings["route"]["navigation"], "old/new navigation differs")
    for key in ("cameras", "camera_common", "commands", "collection"):
        require(settings["sensor_config"][key] == older_settings["sensor_config"][key], "old/new sensor/control condition differs")
    require(not summary.get("cleanup_errors"), "drive cleanup/recording failed: " + str(summary.get("cleanup_errors")))
    require(summary["status"] in VALID_DRIVING_RESULTS, "not a valid driving outcome: " + summary["status"])
    rows = read_rows(folder / "telemetry.csv")
    require(len(rows) > 1 and len(rows) == summary["telemetry_rows"], "missing or truncated driving telemetry")
    s = np.array([float(row["route_s_m"]) for row in rows])
    cte = np.array([float(row["cte_m"]) for row in rows])
    elapsed = np.array([float(row["elapsed_s"]) for row in rows])
    require(np.isfinite(np.r_[s, cte, elapsed]).all() and np.all(np.diff(elapsed) >= 0), "invalid driving values")
    for observed, expected, name in ((s[-1], summary["terminal_station_m"], "station"),
                                     (abs(cte).max(), summary["max_abs_cte_m"], "max CTE"),
                                     (np.sqrt((cte ** 2).mean()), summary["cte_rmse_m"], "CTE RMSE")):
        require(close(observed, expected), "driving summary inconsistent with telemetry: " + name)
    progress = np.maximum.accumulate(s)
    assist = settings["steering_assist"]
    navigation = settings["route"]["navigation"]
    for i, row in enumerate(rows):
        require(row["command_name"] == command_for(s[i], navigation), "incorrect recorded navigation command")
        require(int(row["command"]) == settings["sensor_config"]["commands"][row["command_name"]], "command name/id mismatch")
        source = row.get("steering_source")
        require(source in ("CIL", "map_assist", "safety_stop", "safety_hold"), "unknown/unrecorded steering source")
        if source == "map_assist":
            require(assist["start_s_m"] - 1e-6 <= progress[i] < assist["end_s_m"] + 1e-6,
                    "steering intervention leaked outside the declared first-left interval")
        if source == "CIL":
            require(close(np.clip(float(row["raw_steer"]), -1, 1), row["steer"]), "non-CIL steering applied outside assist")
        if source in ("CIL", "map_assist"):
            throttle, braking = np.clip([float(row["raw_throttle"]), float(row["raw_brake"])], 0., 1.)
            if braking >= throttle: throttle = 0.
            else: braking = 0.
            if float(row["speed_kph"]) >= settings["sensor_config"]["collection"]["speed_cap_kph"]:
                throttle, braking = 0., max(float(braking), .15)
            require(close(throttle, row["throttle"]) and close(braking, row["brake"]),
                    "longitudinal control differs from CIL outputs and the declared speed cap")
    video = folder / summary["screen_video"]
    video_info = probe(video)
    require(summary.get("used_for_training") is False, "evaluation rollout marked as training data")
    turns = [n for n in navigation if n["entry_s_m"] > assist["end_s_m"] and n["command"] in ("left", "right")][-2:]
    require([n["command"] for n in turns] == ["left", "right"], "final two maneuvers are not the fixed left then right")
    gates = []
    for turn in turns:
        gate = turn["exit_s_m"] + turn["exit_hold_m"]
        hits = np.flatnonzero(s >= gate)
        stop = int(hits[0]) if len(hits) else len(rows) - 1
        window = (s >= turn["entry_s_m"] - turn["approach_m"]) & (np.arange(len(rows)) <= stop)
        sources = {row["steering_source"] for row, selected in zip(rows, window) if selected}
        reached = bool(len(hits) and np.max(np.abs(cte[:stop + 1])) < settings["max_cte_m"])
        gates.append(dict(command=turn["command"], link_id=turn["link_id"], exit_hold_gate_s_m=gate,
                          reached_before_cte_intervention=reached,
                          cil_steering_only_in_observed_maneuver=bool(window.any() and sources <= {"CIL"}),
                          no_additional_map_assistance_in_observed_maneuver=bool(window.any() and "map_assist" not in sources),
                          observed_steering_sources=sorted(sources),
                          maneuver_max_abs_cte_m=float(abs(cte[window]).max()) if window.any() else None,
                          interpretation="exit-hold gate observation under 7m CTE limit; not lane-compliance or collision-free proof"))
    return dict(status=summary["status"], terminal_station_m=summary["terminal_station_m"],
                completion_fraction=summary["completion_fraction"], max_abs_cte_m=summary["max_abs_cte_m"],
                cte_rmse_m=summary["cte_rmse_m"], elapsed_s=summary["elapsed_s"], gates=gates,
                assisted_diagnostic=True, steering_assist=summary.get("steering_assist"),
                video=relative(video), video_info=video_info, video_sha256=file_hash(video),
                sources_sha256={name: file_hash(folder / name) for name in ("settings.json", "summary.json", "telemetry.csv")})


def io_failure(folder):
    if not (folder / "summary.json").exists():
        return False  # 원인을 알 수 없는 중단을 자동 재시도로 덮어쓰지 않는다.
    summary = read_json(folder / "summary.json")
    if summary["status"] == "intervention_sensor": return True
    message = str(summary.get("error", "")).lower()
    return summary["status"] == "failed" and any(word in message for word in (
        "sensor", "packet", "grpc", "connection", "unavailable", "window", "display", "recorder", "ffmpeg"))


def simulator_ready(work):
    require(os.environ.get("DISPLAY"), "MORAI GUI recording needs DISPLAY. Start the desktop Simulator and set DISPLAY before retrying.")
    for program in ("ffmpeg", "ffprobe", "xdotool"):
        require(shutil.which(program), program + " is required for actual MORAI screen recording")
    try:
        from tools.simulator import Simulator
        simulator = Simulator()
        try:
            simulator.state()  # 준비 확인만 하며 재배치·제어 명령은 보내지 않는다.
        finally:
            simulator.channel.close()
        subprocess.run(["xdotool", "search", "--onlyvisible", "--name", "^Simulator$"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=True)
    except Exception as error:
        raise RuntimeError("Simulator/gRPC/window unavailable. Open MORAI with the saved sensor rig; then rerun this finisher. " + str(error)) from error
    log = work / ("sensor_check_" + uuid.uuid4().hex[:6] + ".log")
    code = run_command([sys.executable, str(BASE / "drive.py"), "--check"], log)
    require(code == 0, "fresh center-camera/Ego packets unavailable; fix simulator network ports, then rerun. See " + str(log))


def result_label(outcome):
    status = outcome["status"]
    label = {"completed": "종점 도달", "intervention_cte": "경로 이탈로 중단",
             "intervention_stalled": "주행 정체로 중단", "intervention_reverse": "후진 감지로 중단",
             "timeout": "시간 초과"}.get(status, status)
    return f"{label} · {outcome['terminal_station_m']:.1f}m"


def finish(args):
    root = resolve(args.root)
    protocol, audit, old_run, old_video = preflight(root)
    work = root / "finish" / args.run_name
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = read_json(state_path) if state_path.exists() else {"run_name": args.run_name, "inputs": audit, "stages": {}}
    require(state["inputs"] == audit and state["run_name"] == args.run_name, "finisher inputs changed; preserve this run and choose a new --run-name")
    write_json(work / "input_audit.json", audit)
    train_command = training_command(protocol, root, args.run_name, args.workers, args.device)
    if args.audit_only:
        print(json.dumps({"inputs": audit, "training_command": train_command,
                          "work_directory": relative(work), "no_training_or_simulator_actions": True}, indent=2))
        return

    def execute(stage, command):
        log = work / (stage + "_" + uuid.uuid4().hex[:6] + ".log")
        state["stages"][stage] = {"status": "running", "command": command, "log": relative(log),
                                   "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(state_path, state)
        code = run_command(command, log)
        state["stages"][stage].update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        return code

    training = BASE / "runs" / args.run_name
    checkpoint_dir = BASE / "checkpoints" / args.run_name
    if not (training / "summary.json").is_file():
        require(not training.exists() and not checkpoint_dir.exists(),
                "partial training exists and train.py has no resume support. Preserve it, inspect logs, then use a NEW --run-name to train fresh.")
        require(execute("training", train_command) == 0, "training failed; inspect the saved log. No evaluation was launched.")
    checkpoint, training_summary = audit_training(args.run_name, protocol, audit)
    checkpoint_sha = file_hash(checkpoint)
    state["stages"]["training"] = {"status": "completed", "checkpoint": relative(checkpoint),
                                   "checkpoint_sha256": checkpoint_sha, "summary": training_summary}
    write_json(root / "selected_90min_checkpoint.json", {"checkpoint": relative(checkpoint),
               "checkpoint_sha256": checkpoint_sha, "epoch": training_summary["best_epoch"],
               "selection": "fixed validation mean_command_mse only; existing selected.json unchanged"})
    write_json(state_path, state)

    validation = work / "validation"
    if not (validation / "metrics.json").is_file():
        require(not validation.exists(), "partial validation output exists; preserve/inspect it before explicitly clearing that incomplete output")
        command = [sys.executable, str(BASE / "evaluate.py"), "--checkpoint", relative(checkpoint),
                   "--csv", relative(root / "prepared/val.csv"), "--output", relative(validation),
                   "--workers", str(args.workers), "--device", args.device]
        require(execute("validation", command) == 0, "fixed validation failed; see stage log. No driving trial was launched.")
    validation_report = validate_open_loop(validation, checkpoint, audit)
    require(math.isclose(validation_report["metrics"]["mean_command_mse"],
                         training_summary["best_validation_mean_command_mse"], rel_tol=1e-4, abs_tol=1e-7),
            "fixed validation re-evaluation differs from saved checkpoint selection metrics")
    state["stages"]["validation"] = {"status": "completed", "report": relative(validation / "metrics.json"),
                                     "mean_command_mse": validation_report["metrics"]["mean_command_mse"]}
    write_json(state_path, state)

    driving = work / "closed_loop"
    if driving.exists() and io_failure(driving):
        require(args.retry_io, "previous Simulator/sensor I/O attempt failed. Fix MORAI, then use --retry-io to archive that attempt and run once; no valid failure is retried.")
        archive = work / "failed_io" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
        archive.parent.mkdir(exist_ok=True)
        driving.rename(archive)
        state.setdefault("archived_io_attempts", []).append(relative(archive))
        write_json(state_path, state)
    if not driving.exists():
        require(read_json(root / "capture_progress.json").get("status") == "complete", "collection is still active; do not drive while the collector owns the sensors")
        with (root / "collector.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            simulator_ready(work)
            old_settings = read_json(old_run / "settings.json")
            command = [sys.executable, str(BASE / "drive.py"), "--course", protocol["evaluation_course"],
                       "--checkpoint", relative(checkpoint), "--control", "--reset", "--record-screen",
                       "--assist-east-outer-left", "--max-cte", str(protocol["closed_loop"]["max_cte_m"]),
                       "--seconds", str(old_settings["seconds"]), "--device", args.device, "--output", relative(driving)]
            if args.show_command: command.append("--show-command")
            execute("driving", command)  # exit2일 수 있는 유효한 주행 실패도 아래 실제 기록으로 검증한다.
    require((driving / "summary.json").is_file(), "incomplete driving attempt preserved. Inspect its log; no automatic second driving evaluation is allowed.")
    if io_failure(driving):
        failure = read_json(driving / "summary.json")
        raise RuntimeError("Simulator/sensor I/O attempt stopped and was preserved: {}. Fix MORAI, then rerun with --retry-io. No automatic second evaluation was launched.".format(
            failure.get("error") or failure["status"]))
    old_settings = read_json(old_run / "settings.json")
    old_outcome = audit_drive(old_run, protocol, protocol["older_checkpoint_sha256"], old_settings)
    new_outcome = audit_drive(driving, protocol, checkpoint_sha, old_settings)
    state["stages"]["driving"] = {"status": "completed", "outcome": new_outcome}
    write_json(state_path, state)

    comparison = work / "comparison/previous_vs_90min.mp4"
    comparison_manifest = comparison.with_suffix(".comparison.json")
    if not comparison_manifest.exists():
        require(not comparison.exists(), "comparison MP4 exists without a manifest; inspect it instead of overwriting")
        command = [sys.executable, str(BASE / "tools/compare_morai_videos.py"), "--left", str(old_video),
                   "--right", str(driving / "morai_screen.mp4"), "--output", str(comparison),
                   "--left-label", "기존 CIL", "--right-label", "90분 데이터 CIL",
                   "--left-training-seconds", str(audit["old_training_observation_seconds"]),
                   "--right-training-seconds", str(audit["training_observation_seconds"]),
                   "--left-result", result_label(old_outcome), "--right-result", result_label(new_outcome)]
        require(execute("comparison", command) == 0, "comparison encoding failed; source videos and all driving results are preserved")
    comparison_info = read_json(comparison_manifest)
    require(comparison_info["left"]["sha256"] == file_hash(old_video)
            and comparison_info["right"]["sha256"] == new_outcome["video_sha256"], "comparison source videos changed")
    require(comparison_info["output_sha256"] == file_hash(comparison), "comparison output changed")
    require(close(comparison_info["left"]["training_data_seconds"], audit["old_training_observation_seconds"])
            and close(comparison_info["right"]["training_data_seconds"], audit["training_observation_seconds"]),
            "comparison training-duration labels differ from actual center-frame counts")
    state["stages"]["comparison"] = {"status": "completed", "video": relative(comparison), "manifest": relative(comparison_manifest)}
    state["status"] = "completed"
    report = {"status": "completed", "protocol_sha256": audit["protocol_sha256"], "data": audit,
              "checkpoint": relative(checkpoint), "checkpoint_sha256": checkpoint_sha,
              "selected_epoch": training_summary["best_epoch"],
              "fixed_validation_mean_command_mse": validation_report["metrics"]["mean_command_mse"],
              "old": old_outcome, "new": new_outcome,
              "comparison_video": relative(comparison), "comparison_manifest": relative(comparison_manifest),
              "interpretation": ["One matched primary452m route observation per model; no repeated-trial success-rate claim.",
                                 "Both runs have the same earlier first-left map-assist window; the recorded frame counts show whether each reached that window.",
                                 "The final added left/right gate results report only observed CIL behavior and preserve failures.",
                                 "Training-duration labels are unique center observations at10Hz, not GPU training time or independent scene coverage.",
                                 "Full-route training exclusion and balanced selection also changed; not a pure data-quantity ablation.",
                                 "Playback begins at each recording start; no automatic driving-event synchronization. No collision or lane-compliance claim."]}
    write_json(work / "report.json", report)
    write_json(state_path, state)
    print(json.dumps({"status": "completed", "report": relative(work / "report.json"),
                      "new_status": new_outcome["status"], "comparison": relative(comparison)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/east_outer_90min")
    parser.add_argument("--run-name", default="cil_east_outer_90min_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show-command", action="store_true")
    parser.add_argument("--retry-io", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.run_name or any(c not in allowed for c in args.run_name) or args.workers < 0:
        parser.error("run-name must use letters/digits/_/- and workers must be nonnegative")
    root = resolve(args.root)
    root.relative_to(BASE / "runs")
    require(root.is_dir(), "experiment root does not exist: " + str(root))
    with (root / ".finish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            finish(args)
        except Exception as error:
            print("FINISH STOPPED:", str(error), file=sys.stderr, flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
