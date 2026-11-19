"""평가 코스를 제외하고 교차로 데이터를 추가 수집해 별도 학습 목록을 만든다."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from prepare import build
from tools.inspect_collection import inspect


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture(config_path, raw, output, route, recovery, seed, *, label=None, seconds=125, speed=None, noise=False,
            junction_recovery=False, curve_recovery=False, curve_noise_scale=1., initial_offset=0.,
            initial_heading_offset=0., hour=12):
    label = label or route + ("_recovery" if recovery else "_normal")
    before = set(raw.iterdir())
    args = [sys.executable, "-u", str(BASE / "collect.py"), "--config", str(config_path),
            "--route", route, "--auto-reset", "--seed", str(seed), "--seconds", str(seconds)]
    if recovery:
        mode = "--curve-recovery" if curve_recovery else "--junction-recovery" if junction_recovery else "--turn-recovery"
        args.append(mode)
        speed = 17.5 if speed is None else speed
    elif noise:
        args.append("--noise")
    if curve_noise_scale != 1.:
        args += ["--curve-noise-scale", str(curve_noise_scale)]
    if initial_offset != 0.:
        args += ["--initial-offset", str(initial_offset)]
    if initial_heading_offset != 0.:
        args += ["--initial-heading-offset", str(initial_heading_offset)]
    if speed is not None:
        args += ["--speed", str(speed)]
    if hour != 12:
        args += ["--hour", str(hour)]
    print("COLLECT", label, flush=True)
    with (output / (label + ".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)  # 수집기의 finally 제동을 실행한다.
                process.wait(timeout=12)
            raise
    added = sorted(set(raw.iterdir()) - before)
    if code or len(added) != 1:
        raise ValueError(f"{label}: collection failed; inspect {label}.log")
    session = added[0]
    report = inspect(session, decode_all=True)
    if not report["data_checks_passed"]:
        raise ValueError(f"{label}: {report['errors']}")
    print(f"ACCEPT {label} {report['valid_driving_s']:.1f}s, CTE {report['max_abs_cte_m']:.2f}m, "
          f"commands {report['command_seconds']}", flush=True)
    return {"route_id": route, "split": "train", "session": str(session.relative_to(BASE)),
            "recovery": recovery, "seed": seed, "valid_driving_s": report["valid_driving_s"],
            "command_seconds": report["command_seconds"], "protected_min_distances_m": report["protected_min_distances_m"],
            "noise_enabled": report["noise_enabled"], "daylight_hour": hour}


def prepare_combined(plan_path, completion, output):
    extra = output / "additional_prepared"
    build(output=extra, plan=plan_path, completion=completion)
    base_train = (BASE / "data/prepared/train.csv").read_bytes()
    base_val = (BASE / "data/prepared/val.csv").read_bytes()
    extra_train = (extra / "train.csv").read_bytes()
    assert (extra / "val.csv").read_bytes() == base_val, "validation must match the baseline byte-for-byte"
    assert extra_train.partition(b"\n")[0] == base_train.partition(b"\n")[0] and base_train.endswith(b"\n")
    merged = output / "train.csv"
    merged.write_bytes(base_train + extra_train.partition(b"\n")[2])
    with (extra / "train.csv").open(newline="", encoding="utf-8") as stream:
        extras = list(csv.DictReader(stream))
    by_command = {str(command): sum(row["command"] == str(command) for row in extras) for command in range(4)}
    assert all(by_command.values()), "additional data must represent all four commands"
    stats = {"additional_samples": len(extras), "additional_commands": by_command,
             "additional_routes": sorted({row["route_id"] for row in extras}),
             "baseline_train_sha256": hashlib.sha256(base_train).hexdigest(),
             "validation_sha256": hashlib.sha256(base_val).hexdigest(),
             "experiment_train_sha256": hashlib.sha256(merged.read_bytes()).hexdigest()}
    write_json(output / "prepared_summary.json", stats)
    print("PREPARED", stats, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_path, output = (BASE / args.plan).resolve(), (BASE / args.output).resolve()
    config = read_json(BASE / "config.json")
    baseline = read_json(BASE / "data/collection_completion.json")
    original_ids = set(read_json(BASE / config["paths"]["route_plan"])["routes"])
    plan = read_json(plan_path)
    routes = {key: info for key, info in plan["routes"].items() if key not in original_ids and info["split"] == "train"}
    if not routes or "EAST_OUTER" not in plan["test_reservations"]:
        raise ValueError("experiment plan must add training routes and reserve EAST_OUTER")
    prioritized = {name: [] for name in ("right", "left", "straight")}
    for key, route in routes.items():
        main_command = route.get("focus_command") or route["navigation"][0]["command"]
        prioritized[main_command].append(key)
    if not all(prioritized.values()):
        raise ValueError("experiment plan needs right, left and straight demonstrations")

    # 먼저 각 명령을 고루 수집하고 회전 복귀 장면을 별도로 추가한다.
    normal = [key for group in prioritized.values() for key in group]
    recovery = prioritized["right"][:2] + prioritized["left"][:2]
    tasks = [(key, False, 7) for key in normal] + [(key, True, 51 + i) for i, key in enumerate(recovery)]
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    config["paths"]["route_plan"] = str(plan_path.relative_to(BASE))
    config["paths"]["raw_data"] = str(raw.relative_to(BASE))
    config_path = output / "capture_config.json"
    write_json(config_path, config)
    write_json(output / "capture_tasks.json", tasks)
    original_val = [item for item in baseline["accepted"] if item["split"] == "validation"]
    accepted, failures = [], []
    for route, recovery_mode, seed in tasks:
        try:
            accepted.append(capture(config_path, raw, output, route, recovery_mode, seed))
        except ValueError as error:
            failures.append(str(error))
            print("EXCLUDED", error, flush=True)
        write_json(output / "capture_progress.json", {"accepted": accepted, "failures": failures})
    covered = {item["route_id"] for item in accepted}
    if any(not (set(group) & covered) for group in prioritized.values()):
        raise RuntimeError("collection did not yield verified right, left and straight routes")
    completion = output / "additional_completion.json"
    write_json(completion, {"status": "complete", "accepted": accepted + original_val, "failures": failures})
    prepare_combined(plan_path, completion, output)


if __name__ == "__main__":
    main()
