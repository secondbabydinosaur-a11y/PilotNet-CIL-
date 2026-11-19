"""저장된 CIL 시험 로그를 표·궤적으로 요약한다. 주행·학습·모델 재평가는 하지 않는다."""

import argparse  # 집계할 원본 실행 폴더와 새 결과 폴더를 받는다.
import csv  # 실제 주행 telemetry를 읽는다.
import json  # 기록의 설정과 요약을 읽고 집계를 저장한다.
from pathlib import Path

import numpy as np  # 저장된 좌표·오차만 계산한다.

try:
    from .summarize_training import BASE, file_hash, read_json, require_close
except ImportError:
    from summarize_training import BASE, file_hash, read_json, require_close

COURSES = ("U2", "J_straight", "J_left", "J_right", "COMPETITION", "COMPETITION_REVERSE")
DEFAULT_COURSES = COURSES[:4] + ("COMPETITION_REVERSE",)  # 현재 공개 목록은 구간 시험 4개와 돌아오는 연속 주행이다.


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def display_path(path, base):
    try:
        return str(path.relative_to(base))  # CIL 내부 파일은 공유하기 쉬운 상대경로로 남긴다.
    except ValueError:
        return str(path)


def load_trial(folder, base, checkpoint_cache):
    required = [folder / "settings.json", folder / "summary.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("missing required run records: " + ", ".join(missing))
    settings, summary = map(read_json, required)
    course = summary["course"]
    if course not in COURSES or settings["course"] != course:
        raise ValueError(f"{folder}: unexpected or inconsistent course")
    if settings["checkpoint_sha256"] != summary["checkpoint_sha256"] or settings["epoch"] != summary["epoch"]:
        raise ValueError(f"{folder}: settings and summary identify different checkpoints")
    checkpoint = resolve_path(settings["checkpoint"], base)
    if checkpoint not in checkpoint_cache:
        checkpoint_cache[checkpoint] = file_hash(checkpoint)  # 같은 모델 파일은 한 번만 읽는다.
    if checkpoint_cache[checkpoint] != summary["checkpoint_sha256"]:
        raise ValueError(f"{folder}: checkpoint hash mismatch")
    route_path = resolve_path(settings["route"]["route_csv"], base)
    route_sha = file_hash(route_path)
    if route_sha != settings["route_sha256"] or route_sha != settings["route"]["route_sha256"]:
        raise ValueError(f"{folder}: reference route hash mismatch")
    reference = np.loadtxt(route_path, delimiter=",", skiprows=1, ndmin=2)
    if reference.shape[1] != 4 or len(reference) < 2 or not np.isfinite(reference).all() or reference[-1, 0] <= reference[0, 0]:
        raise ValueError(f"{folder}: invalid reference coordinates")

    telemetry_path = folder / "telemetry.csv"
    missing_logs = []
    if telemetry_path.is_file():
        with telemetry_path.open(newline="", encoding="utf-8") as stream:
            telemetry = list(csv.DictReader(stream))  # 원본 행 순서대로 실제 주행 궤적을 읽는다.
    else:
        telemetry = []
        missing_logs.append("telemetry.csv")  # 출발 전 실패를 실제 주행 성공으로 처리하지 않는다.
    if len(telemetry) != int(summary["telemetry_rows"]):
        raise ValueError(f"{folder}: telemetry row count differs from summary; missing or truncated log")
    columns = ("elapsed_s", "x_m", "y_m", "route_s_m", "cte_m", "speed_kph")
    values = np.array([[float(row[key]) for key in columns] for row in telemetry], dtype=float).reshape(-1, len(columns))
    if not np.isfinite(values).all() or (len(values) and np.any(np.diff(values[:, 0]) < 0)):
        raise ValueError(f"{folder}: non-finite or non-monotone telemetry")
    if len(values):
        expected = {
            "elapsed_s": values[-1, 0], "terminal_station_m": values[-1, 3],
            "max_abs_cte_m": np.max(np.abs(values[:, 4])),
            "cte_rmse_m": np.sqrt(np.mean(values[:, 4] ** 2)),
            "mean_speed_kph": np.mean(values[:, 5]),
            "completion_fraction": np.clip((values[-1, 3] - reference[0, 0]) / (reference[-1, 0] - reference[0, 0]), 0., 1.),
        }
        for key, value in expected.items():
            require_close(value, summary[key], f"{folder.name} {key}", rel_tol=1e-5)

    videos = {}
    for key in ("camera_video", "screen_video"):
        filename = summary.get(key)
        if filename:
            path = resolve_path(filename, folder)
            videos[key] = display_path(path, base)
            if not path.is_file() or path.stat().st_size == 0:
                missing_logs.append(key + ": " + videos[key])
        else:
            videos[key] = None
            if key == "screen_video" and settings.get("record_screen"):
                missing_logs.append("requested screen recording was not created")
    row = {
        "course": course, "status": summary["status"],
        "completion_fraction": summary["completion_fraction"],
        "max_abs_cte_m": summary["max_abs_cte_m"], "cte_rmse_m": summary["cte_rmse_m"],
        "mean_speed_kph": summary["mean_speed_kph"], "duration_s": summary["elapsed_s"],
        "checkpoint": display_path(checkpoint, base), "checkpoint_sha256": summary["checkpoint_sha256"],
        "epoch": summary["epoch"], **videos, "run": display_path(folder, base),
        "telemetry_rows": len(telemetry), "missing_logs": "; ".join(missing_logs),
        "cleanup_errors": "; ".join(summary.get("cleanup_errors", [])), "error": summary.get("error"),
    }
    sources = {path.name: file_hash(path) for path in required}
    if telemetry_path.is_file():
        sources["telemetry.csv"] = file_hash(telemetry_path)
    return {"row": row, "summary": summary, "reference": reference, "actual": values,
            "route_sha256": route_sha, "sources": sources}


def summarize_runs(run_folders, output_folder, base=BASE):
    if not run_folders:
        raise ValueError("no actual driving runs provided")
    folders = [resolve_path(value, base) for value in run_folders]
    if len(folders) != len(set(folders)):
        raise ValueError("duplicate run folder")
    checkpoint_cache = {}
    trials = [load_trial(folder, base, checkpoint_cache) for folder in folders]
    if not any(len(trial["actual"]) for trial in trials):
        raise ValueError("no actual driving telemetry; all supplied runs contain zero samples")
    if len({trial["row"]["checkpoint_sha256"] for trial in trials}) != 1 or len({trial["row"]["epoch"] for trial in trials}) != 1:
        raise ValueError("trials use different model checkpoints")
    courses = [trial["row"]["course"] for trial in trials]
    if len(courses) != len(set(courses)):
        raise ValueError("provide one recorded trial per course; keep repeat trials in separate summaries")
    output = resolve_path(output_folder, base)
    if output.exists():
        raise FileExistsError(f"output must be a new folder: {output}")
    output.mkdir(parents=True)

    rows = [trial["row"] for trial in trials]
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")  # 화면 창 없이 실제 좌표 그래프를 저장한다.
    from matplotlib import pyplot as plt

    plot_courses = COURSES if all(key in courses for key in COURSES[-2:]) else (COURSES[:5] if "COMPETITION" in courses else DEFAULT_COURSES)
    figure, axes = plt.subplots(1, len(plot_courses), figsize=(3.8 * len(plot_courses), 4.8))
    by_course = {trial["row"]["course"]: trial for trial in trials}
    for axis, course in zip(axes, plot_courses):
        trial = by_course.get(course)
        if trial is None:
            axis.set_title(course + "\nNot provided")
            axis.text(.5, .5, "No trial supplied", ha="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        row, reference, actual = trial["row"], trial["reference"], trial["actual"]
        axis.plot(reference[:, 1], reference[:, 2], color="0.7", linewidth=2.5, label="Reference")
        if len(actual):
            color = "#13815a" if row["status"] == "completed" else "#bf3547"
            axis.plot(actual[:, 1], actual[:, 2], color=color, linewidth=1.5, label="CIL actual")
            axis.scatter(actual[0, 1], actual[0, 2], color=color, marker="o", s=25)
            axis.scatter(actual[-1, 1], actual[-1, 2], color=color, marker="x", s=45, label="Actual end")
        axis.scatter(reference[-1, 1], reference[-1, 2], color="0.4", marker="s", s=18, label="Reference end")
        axis.set(title=course + "\n" + row["status"], xlabel="Map X (m)", ylabel="Map Y (m)")
        axis.axis("equal")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.suptitle("CIL: held-out road trials and competition-route continuous driving")
    figure.tight_layout(rect=(0, .04, 1, .93))
    figure.text(.5, .01, "Endpoint reached is not a zero-collision claim. Collision events were not instrumented.", ha="center", fontsize=9)
    figure.savefig(output / "trajectory.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed", "evaluation_scope": "four held-out road trials and a competition-route demonstration with training-road overlap; no new-town or repeated-trial success-rate claim",
        "trials": rows, "provided_courses": courses, "missing_courses": [course for course in plot_courses if course not in courses],
        "trial_count": len(trials), "endpoint_completed_trials": sum(row["status"] == "completed" for row in rows),
        "checkpoint_sha256": rows[0]["checkpoint_sha256"], "epoch": rows[0]["epoch"],
        "collision_metric": None, "collision_note": "not instrumented; do not interpret endpoint completion as zero collisions",
        "has_missing_logs": any(row["missing_logs"] for row in rows),
        "verified_sources": {trial["row"]["run"]: {**trial["sources"], "reference_route": trial["route_sha256"]} for trial in trials},
        "outputs": ["table.csv", "summary.json", "trajectory.png"],
        "notes": ["Original run records, models and training data were not modified.",
                  "No model was re-evaluated and no vehicle control or training was performed.",
                  "Status and cleanup errors are preserved. Missing videos or logs are explicitly listed."],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="CIL-relative or absolute recorded run folders")
    parser.add_argument("--output", required=True, help="new CIL-relative or absolute output folder")
    args = parser.parse_args()
    report = summarize_runs(args.runs, args.output)
    print(json.dumps({key: report[key] for key in ("trial_count", "provided_courses", "missing_courses", "has_missing_logs", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
