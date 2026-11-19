"""고정된 데이터 실험의 저장 주행 기록만 비교한다. 학습·추론·차량 제어는 하지 않는다."""

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .summarize_training import BASE, file_hash, read_json, require_close
except ImportError:
    from summarize_training import BASE, file_hash, read_json, require_close

BASELINE_RUNS = {
    "J_left": "runs/closed_loop/20261117T233541Z_J_left_09c5c1",
    "J_straight": "runs/closed_loop/20261117T233511Z_J_straight_2f2083",
    "U2": "runs/closed_loop/20261117T233158Z_U2_fc301f",
}


def resolve(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def relative(path, base):
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def recorded_csv(settings, split, base):
    """기록된 경로 또는 동일 SHA의 준비 CSV를 찾는다. 파일명만으로 추정하지 않는다."""
    digest = settings["hashes"][split + ".csv"]
    options = []
    if settings.get(split + "_csv"):
        options.append(resolve(settings[split + "_csv"], base))
    for key in ("prepared_dir", "data_dir", "data_directory"):
        if settings.get(key):
            options.append(resolve(settings[key], base) / (split + ".csv"))
    options += [base / "data/prepared" / (split + ".csv")]
    options += sorted((base / "data").glob("prepared*/**/" + split + ".csv"))
    for path in dict.fromkeys(options):
        if path.is_file() and file_hash(path) == digest:
            return path
    raise ValueError(f"recorded {split}.csv hash cannot be found: {digest}")


def training_audit(checkpoint_path, protocol, base):
    import torch  # CPU에서 저장 설정만 읽으며 모델을 생성하거나 실행하지 않는다.

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    for name in ("model.py", "dataset.py", "config.json"):
        if settings["hashes"][name] != protocol["immutable_inputs_sha256"][name]:
            raise ValueError("checkpoint changed a fixed input contract: " + name)
    if settings["hashes"]["val.csv"] != protocol["immutable_inputs_sha256"]["data/prepared/val.csv"]:
        raise ValueError("checkpoint used a different validation split")
    if settings["command_names"] != ["follow_road", "left", "straight", "right"]:
        raise ValueError("checkpoint command mapping changed")
    if settings["control_order"] != ["steer", "throttle", "brake"]:
        raise ValueError("checkpoint action order changed")
    train_csv, val_csv = [recorded_csv(settings, split, base) for split in ("train", "val")]
    rows = read_rows(train_csv)
    if not rows:
        raise ValueError("checkpoint training CSV is empty")
    protected_names = set(protocol["reported_courses"]) | {"J_right"}
    for row in rows:
        if row.get("split", "train") != "train" or row.get("route_id") in protected_names:
            raise ValueError("training CSV contains a protected test or non-training row")
        if "runs/closed_loop" in row["image_path"].replace("\\", "/"):
            raise ValueError("closed-loop rollout image was used as a training label")
    xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    if not np.isfinite(xy).all():
        raise ValueError("non-finite training coordinates")
    tree = cKDTree(xy)
    separation = {}
    for course, info in protocol["reported_courses"].items():
        route = np.loadtxt(resolve(info["path"], base), delimiter=",", skiprows=1, ndmin=2)
        stations = np.r_[np.arange(route[0, 0], route[-1, 0], .25), route[-1, 0]]
        sample = np.column_stack([np.interp(stations, route[:, 0], route[:, k]) for k in (1, 2)])
        lower_bound = max(0., float(tree.query(sample)[0].min()) - .25)
        if lower_bound < protocol["protected_training_separation_m"]:
            raise ValueError(f"training coordinates overlap protected {course}: {lower_bound:.3f}m")
        separation[course] = lower_bound
    return {"checkpoint_epoch": checkpoint["epoch"],
            "train_csv": relative(train_csv, base), "train_csv_sha256": file_hash(train_csv),
            "val_csv": relative(val_csv, base), "val_csv_sha256": file_hash(val_csv),
            "training_rows": len(rows), "protected_distance_lower_bound_m": separation,
            "test_training_rows_detected": False}


def command_for(station, navigation):
    inside = [item for item in navigation if item["entry_s_m"] <= station <= item["exit_s_m"]]
    if inside:
        return inside[0]["command"]
    upcoming = [item for item in navigation if 0 < item["entry_s_m"] - station <= item["approach_m"]]
    if upcoming:
        return min(upcoming, key=lambda item: item["entry_s_m"])["command"]
    passed = [item for item in navigation if 0 < station - item["exit_s_m"] <= item["exit_hold_m"]]
    return passed[-1]["command"] if passed else "follow_road"


def sensor_windows(path):
    """저장된 driver 코드에서 실제 센서 신선도 제한을 읽는다."""
    wanted = {"age_ego", "age_camera", "delta"}
    result = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id in wanted and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Gt) and isinstance(node.comparators[0], ast.Constant)):
            result[node.left.id] = float(node.comparators[0].value)
    if set(result) != wanted:
        raise ValueError("cannot verify the driver sensor freshness windows")
    return result


def load_trial(folder, protocol, base, audits):
    settings, summary = [read_json(folder / name) for name in ("settings.json", "summary.json")]
    course = settings["course"]
    if course not in protocol["reported_courses"] or summary["course"] != course:
        raise ValueError("trial course is outside the fixed experiment")
    if summary.get("used_for_training") is not False or settings.get("no_retraining") is not True:
        raise ValueError("trial does not identify a frozen model-only rollout")
    if "steering_assist" in settings or "steering_assist" in summary:
        raise ValueError("an assisted diagnostic cannot substitute for a model-only trial")
    if summary.get("test_only") is not True:
        raise ValueError("a demonstration cannot substitute for the reserved test")
    require_close(settings["max_cte_m"], protocol["measurement"]["max_cte_m"], "CTE cutoff")
    require_close(settings["sensor_config"]["collection"]["speed_cap_kph"], protocol["measurement"]["speed_cap_kph"], "speed cap")
    route_info = protocol["reported_courses"][course]
    route = resolve(settings["route"]["route_csv"], base)
    if settings["route_sha256"] != route_info["sha256"] or file_hash(route) != route_info["sha256"]:
        raise ValueError("trial reference differs from the fixed protocol")
    if settings["route"]["route_sha256"] != route_info["sha256"]:
        raise ValueError("route metadata hash differs from the protocol")
    checkpoint_path = resolve(settings["checkpoint"], base)
    digest = file_hash(checkpoint_path)
    if digest != settings["checkpoint_sha256"] or digest != summary["checkpoint_sha256"]:
        raise ValueError("trial checkpoint hash mismatch")
    if digest not in audits:
        audits[digest] = training_audit(checkpoint_path, protocol, base)
    if summary["epoch"] != settings["epoch"] or summary["epoch"] != audits[digest]["checkpoint_epoch"]:
        raise ValueError("trial epoch differs from the saved checkpoint")
    for name in ("model.py", "dataset.py", "config.json"):
        if file_hash(folder / "software" / name) != protocol["immutable_inputs_sha256"][name]:
            raise ValueError("trial source snapshot changed: " + name)
    windows = sensor_windows(folder / "software/drive.py")
    if settings["record_screen"] and not (folder / str(summary.get("screen_video"))).is_file():
        raise ValueError("requested actual-screen recording is missing")
    rows = read_rows(folder / "telemetry.csv")
    if not rows or len(rows) != summary["telemetry_rows"]:
        raise ValueError("missing or truncated telemetry")
    fields = ("elapsed_s", "route_s_m", "cte_m", "heading_error_deg", "speed_kph", "raw_steer", "steer")
    data = {key: np.array([float(row[key]) for row in rows]) for key in fields}
    if not all(np.isfinite(value).all() for value in data.values()) or np.any(np.diff(data["elapsed_s"]) < 0):
        raise ValueError("non-finite or non-monotone telemetry")
    require_close(data["route_s_m"][-1], summary["terminal_station_m"], "terminal station")
    require_close(np.max(np.abs(data["cte_m"])), summary["max_abs_cte_m"], "max CTE")
    require_close(np.sqrt(np.mean(data["cte_m"] ** 2)), summary["cte_rmse_m"], "CTE RMSE")
    numbers = settings["sensor_config"]["commands"]
    for row in rows:
        expected = command_for(float(row["route_s_m"]), settings["route"]["navigation"])
        if row["command_name"] != expected or int(row["command"]) != numbers[expected]:
            raise ValueError("telemetry command does not follow the fixed navigation policy")
    initial = rows[0]
    if abs(float(initial["route_s_m"])) > 1 or abs(float(initial["cte_m"])) > .15 or abs(float(initial["heading_error_deg"])) > 2 or abs(float(initial["speed_kph"])) > .5:
        raise ValueError("trial initial pose or speed differs from the normal-start condition")

    result = {"run": relative(folder, base), "course": course, "status": summary["status"],
              "checkpoint_sha256": digest, "epoch": summary["epoch"],
              "completion_fraction": summary["completion_fraction"],
              "terminal_station_m": summary["terminal_station_m"], "terminal_cte_m": data["cte_m"][-1].item(),
              "max_abs_cte_m": summary["max_abs_cte_m"], "cte_rmse_m": summary["cte_rmse_m"],
              "mean_speed_kph": summary["mean_speed_kph"], "duration_s": summary["elapsed_s"],
              "max_raw_applied_steer_difference": float(np.max(np.abs(data["raw_steer"] - data["steer"]))),
              "collision_metric": None, "first_right_exit_reached": None,
              "first_right_gate_s_m": None, "first_turn_max_abs_cte_m": None,
              "first_turn_cte_rmse_m": None, "first_turn_observations": {}}
    if course == "EAST_OUTER":
        right = settings["route"]["navigation"][0]
        if right["command"] != "right" or right["link_id"] != "A2256W000078":
            raise ValueError("first junction is not the fixed J right turn")
        gate = protocol["measurement"]["target_first_right_exit_hold_m"]
        require_close(gate, right["exit_s_m"] + right["exit_hold_m"], "first-right exit gate")
        hits = np.flatnonzero(data["route_s_m"] >= gate)
        limit = int(hits[0]) if len(hits) else len(rows) - 1
        result["first_right_gate_s_m"] = gate
        result["first_right_exit_reached"] = bool(len(hits) and np.max(np.abs(data["cte_m"][:limit + 1])) < settings["max_cte_m"])
        mask = (data["route_s_m"] >= right["entry_s_m"]) & (np.arange(len(rows)) <= limit)
        if mask.any():
            result["first_turn_max_abs_cte_m"] = float(np.max(np.abs(data["cte_m"][mask])))
            result["first_turn_cte_rmse_m"] = float(np.sqrt(np.mean(data["cte_m"][mask] ** 2)))
        stages = {"command_start": right["entry_s_m"] - right["approach_m"],
                  "entry": right["entry_s_m"], "middle": (right["entry_s_m"] + right["exit_s_m"]) / 2,
                  "exit": right["exit_s_m"], "exit_hold_end": gate}
        for name, station in stages.items():
            hit = np.flatnonzero(data["route_s_m"] >= station)
            result["first_turn_observations"][name] = None if not len(hit) else {
                **{key: float(data[key][hit[0]]) for key in fields},
                "command": rows[hit[0]]["command_name"], "reason": rows[hit[0]]["reason"]}
    return {"settings": settings, "summary": summary, "metrics": result, "sensor_windows": windows,
            "runtime_source_sha256": {name: file_hash(folder / "software" / name)
                                      for name in ("drive.py", "morai.py", "recording.py", "live_command_banner.py")
                                      if (folder / "software" / name).is_file()},
            "source_sha256": {name: file_hash(folder / name) for name in ("settings.json", "summary.json", "telemetry.csv")}}


def verify_conditions(trial, original, matched_runtime=False):
    course = trial["metrics"]["course"]
    for key in ("route_sha256", "max_cte_m", "seconds", "control_hz", "camera", "model_controls", "actuator_policy", "stall_timeout_s", "record_screen"):
        if trial["settings"][key] != original["settings"][key]:
            raise ValueError(f"{course}: changed driving policy {key}")
    for key in ("commands", "camera_common", "cameras", "collection"):
        if trial["settings"]["sensor_config"][key] != original["settings"]["sensor_config"][key]:
            raise ValueError(f"{course}: changed sensor/control configuration {key}")
    if trial["settings"]["route"]["navigation"] != original["settings"]["route"]["navigation"]:
        raise ValueError("categorical command geometry or timing changed")
    if trial["sensor_windows"] != original["sensor_windows"]:
        raise ValueError("driver sensor freshness windows changed")
    endpoint_match = trial["settings"]["completion_definition"] == original["settings"]["completion_definition"]
    if not endpoint_match and not (not matched_runtime and course == "U2" and original["metrics"]["terminal_station_m"] < original["settings"]["route"]["length_m"] - 2):
        raise ValueError("endpoint policy changed and affects the comparison")
    if matched_runtime:
        for key in ("device", "screen_command_overlay", "live_command_banner", "screen_command_source"):
            if trial["settings"].get(key) != original["settings"].get(key):
                raise ValueError(f"{course}: matched repeat differs in runtime/display setting {key}")
        if trial["runtime_source_sha256"] != original["runtime_source_sha256"]:
            raise ValueError("matched repeat differs in driver/recording source snapshots")
    return endpoint_match


def compare_trials(candidate_folders, output, base=BASE, baseline_repeat_folders=None):
    protocol_path = base / "runs/east_outer_data_protocol_v1.json"
    protocol = read_json(protocol_path)
    for name, digest in protocol["immutable_inputs_sha256"].items():
        if file_hash(base / name) != digest:
            raise ValueError("original immutable input changed: " + name)
    audits = {}
    baseline_names = {**BASELINE_RUNS, "EAST_OUTER": protocol["baseline_east_outer_run"]}
    baseline = {course: load_trial(resolve(folder, base), protocol, base, audits) for course, folder in baseline_names.items()}
    baseline_digest = protocol["baseline_checkpoint"]["checkpoint_sha256"]
    if any(record["metrics"]["checkpoint_sha256"] != baseline_digest for record in baseline.values()):
        raise ValueError("baseline trial does not use the frozen baseline checkpoint")
    repeats = {}
    for folder in baseline_repeat_folders or []:
        repeat = load_trial(resolve(folder, base), protocol, base, audits)
        course = repeat["metrics"]["course"]
        if course in repeats:
            raise ValueError("provide exactly one matched baseline repeat per course")
        if repeat["metrics"]["checkpoint_sha256"] != baseline_digest:
            raise ValueError("baseline repeat does not use the frozen baseline checkpoint")
        if repeat["metrics"]["run"] in [record["metrics"]["run"] for record in baseline.values()]:
            raise ValueError("a historical run cannot be relabeled as a fresh baseline repeat")
        verify_conditions(repeat, baseline[course])
        repeats[course] = repeat
    candidates = [load_trial(resolve(folder, base), protocol, base, audits) for folder in candidate_folders]
    if not candidates:
        raise ValueError("no candidate trials supplied")
    comparisons = []
    for candidate in candidates:
        course = candidate["metrics"]["course"]
        if baseline_repeat_folders is not None and course not in repeats:
            raise ValueError("matched baseline repeat missing for candidate course: " + course)
        original = repeats.get(course, baseline[course])
        endpoint_match = verify_conditions(candidate, original, matched_runtime=bool(repeats))
        comparisons.append({"course": course, "baseline": original["metrics"], "candidate": candidate["metrics"],
                            "baseline_source": "matched_repeat" if course in repeats else "historical",
                            "historical_baseline": baseline[course]["metrics"],
                            "matched_baseline_repeat": repeats[course]["metrics"] if course in repeats else None,
                            "same_driving_policy": True, "sensor_windows_s": original["sensor_windows"],
                            "same_endpoint_definition": endpoint_match,
                            "endpoint_note": None if endpoint_match else "기존 U2는 종점 전에 종료돼 이전 종점 조건 차이가 기존 실패 결과에 영향을 주지 않았다.",
                            "same_checkpoint": candidate["metrics"]["checkpoint_sha256"] == baseline_digest})
    report = {"scope": "고정 Validation으로 선택한 모델의 조건별 1회 주행 기록 비교. 통계적 성공률·개선 원인·독립 도시 일반화를 증명하지 않는다.",
              "protocol": relative(protocol_path, base), "protocol_sha256": file_hash(protocol_path),
              "first_right_definition": "EAST_OUTER s93.7022765394m에 CTE 7m 중단 전에 도달했는지 관찰한다. 정밀한 차선 유지나 전체 코스 완주와 구분한다.",
              "comparisons": comparisons, "training_audits": audits,
              "historical_baseline_runs": {course: record["metrics"] for course, record in baseline.items()},
              "baseline_repeat_runs": {course: record["metrics"] for course, record in repeats.items()},
              "source_sha256": {record["metrics"]["run"]: record["source_sha256"] for record in [*baseline.values(), *repeats.values(), *candidates]},
              "collision_metric": None, "model_selection_uses_closed_loop": False,
              "notes": ["신규 시뮬레이터 제어·모델 추론·학습을 실행하지 않았다.", "학습 CSV 해시와 실제 좌표로 네 시험 구역의 학습 유입 여부를 확인했다.",
                        "좌우회전 명령은 지도 related_signal 문자열이 아닌 저장된 출구 기하와 시간 정책으로 확인했다.", "모델 선택은 고정 Validation 기준이며 이 주행 결과로 체크포인트를 선택하지 않는다.",
                        "재실행을 지정하면 후보 비교는 해당 재실행을 사용하고 과거 결과도 별도로 보존한다. 더 좋은 결과를 골라 사용하지 않는다.",
                        "날씨·시각의 실제 readback은 기존 기록에 없다. 동일 reset 절차 사용 여부와 별도로 구분해야 한다."]}
    output_path = resolve(output, base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        if output_path.suffix.lower() == ".csv":
            rows = [{"role": role, **record[role]} for record in comparisons
                    for role in (("historical_baseline", "matched_baseline_repeat", "candidate")
                                 if record["matched_baseline_repeat"] is not None else ("baseline", "candidate"))]
            rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value for key, value in row.items()} for row in rows]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", required=True, help="CIL 상대 또는 절대 경로의 완료된 후보 주행 폴더")
    parser.add_argument("--baseline-repeat", nargs="+", help="같은 현재 조건에서 다시 기록한 동결 Baseline 주행 폴더; 과거 결과도 함께 보존")
    parser.add_argument("--output", required=True, help="새 JSON 또는 CSV 파일; 기존 파일은 덮어쓰지 않음")
    args = parser.parse_args()
    report = compare_trials(args.candidate, args.output, baseline_repeat_folders=args.baseline_repeat)
    print(json.dumps({"comparisons": len(report["comparisons"]), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
