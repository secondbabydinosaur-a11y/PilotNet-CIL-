"""새 7회 주행만으로 기존 공개 영상 4개와 closed-loop 표를 교체한다.

사용 예: bash run.sh pilotnet -m tools.evaluation.replace_selected_videos --runs <폴더 7개>
학습·시뮬레이터 제어는 하지 않는다. open_loop.csv와 README는 수정하지 않는다.
"""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from data_utils import BASE, read_json, write_json
from tools.evaluation.curate_results import encode, duration, digest


OUTPUT = BASE / "results"
CHECKPOINTS = {
    "center": BASE / "checkpoints/center_camera/best.pt",
    "three": BASE / "checkpoints/three_cameras/best.pt",
    "three_curves": BASE / "checkpoints/three_cameras_more_curves/best.pt",
}
HISTORICAL_CHECKPOINT_KEYS = {
    "center": "runs/pilotnet_center_v1/best.pt",
    "three": "runs/pilotnet_three_v1/best.pt",
}  # 고정 당시 프로토콜의 이름은 유지하고 실제 파일은 공개 폴더에서 읽는다.
MODELS = {
    "center": "가운데 카메라로 학습",
    "three": "카메라 3개로 학습",
    "three_curves": "카메라 3개 · 곡선 비중 증가 학습",
}
COURSES = {"U1": "학습하지 않은 직선", "U2": "학습하지 않은 곡선", "J": "교차로 J"}
EXPECTED = {
    ("center", "U1"), ("center", "U2"), ("three", "U1"), ("three", "U2"),
    ("three_curves", "U1"), ("three_curves", "U2"), ("three", "J"),
}
NAMES = (
    "01_new_road_straight", "02_new_road_curve", "03_more_curve_training", "04_intersection",
)
METRICS = ("driving_trials.csv", "curve_sampling_comparison.csv", "intersection_example.csv")
CTE_DEFINITION = "abs(cte_m) >= max_cte_m"


def result_name(status):
    """예전 1m 라벨을 재사용하지 않고 실제 새 종료 상태를 표시한다."""
    names = {
        "completed": "완주 (경로 종점 도달)",
        "intervention_cte": "횡오차 7m 이상으로 종료",
        "intervention_sensor": "센서 수신 지연으로 종료",
        "timeout": "시간 초과",
    }
    if status not in names:
        raise ValueError("publication requires a finished trial, got status=" + str(status))
    return names[status]


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite " + name)
    return value


def relative(path):
    """저장소 내부 원본은 상대경로로, 외부 지정 원본은 절대경로로 기록한다."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def load_runs(paths, hashes):
    """폴더 이름 대신 실제 설정·summary·체크포인트 SHA로 7개 조건을 확인한다."""
    if len(paths) != 7 or len({Path(p).resolve() for p in paths}) != 7:
        raise ValueError("exactly seven different new run directories are required")
    if len(set(hashes.values())) != 3:
        raise ValueError("three distinct frozen checkpoints are required")
    by_hash = {value: mode for mode, value in hashes.items()}
    trials = {}
    for item in paths:
        run = Path(item).resolve()
        settings = read_json(run / "settings.json")
        summary = read_json(run / "summary.json")
        initial = read_json(run / "initial_state.json")
        sha = settings["checkpoint_sha256"]
        mode = by_hash.get(sha)
        course = settings["course"]
        key = (mode, course)
        if key not in EXPECTED or key in trials:
            raise ValueError("unknown or repeated model/course condition: " + str(key))
        if summary.get("checkpoint_sha256") != sha or summary.get("course") != course:
            raise ValueError("settings and summary differ: " + str(run))
        if not settings.get("control") or not summary.get("control") or not settings.get("record_screen"):
            raise ValueError("actual controlled MORAI GUI recording required: " + str(run))
        if (finite(settings["max_cte_m"], "max_cte_m") != 7.0
                or settings.get("cte_intervention_definition") != CTE_DEFINITION
                or settings.get("standalone_heading_intervention") is not False):
            raise ValueError("only new >=7m CTE trials without a separate heading cutoff are allowed")
        if (finite(settings["initial_offset_m"], "initial_offset_m") != 0
                or finite(settings["initial_heading_offset_deg"], "initial_heading_offset_deg") != 0):
            raise ValueError("only nominal initial conditions are allowed")
        if (abs(finite(initial["cte_m"], "initial CTE")) > .1
                or abs(finite(initial["heading_error_deg"], "initial heading")) > 1):
            raise ValueError("measured initial pose differs from nominal: " + str(run))
        expected_mode = "center" if mode == "center" else "three"
        if settings.get("model_camera_mode") != expected_mode:
            raise ValueError("checkpoint camera mode differs: " + str(run))
        result_name(summary["status"])  # 오류·중단 기록을 성공/실패 시험으로 포장하지 않는다.
        for field in ("completion_fraction", "cte_rmse_m", "max_abs_cte_m", "mean_speed_kph"):
            finite(summary[field], field)
        if summary["status"] == "intervention_cte" and float(summary["max_abs_cte_m"]) < 7:
            raise ValueError("CTE intervention occurred below the configured threshold")
        video = run / "morai_screen.mp4"
        if not video.is_file() or duration(video) <= 0:
            raise ValueError("missing or unreadable actual MORAI recording: " + str(video))
        if not (run / "telemetry.csv").is_file():
            raise FileNotFoundError(run / "telemetry.csv")
        trials[key] = dict(run=run, settings=settings, summary=summary, initial=initial,
                           mode=mode, course=course, video=video)
    if set(trials) != EXPECTED:
        raise ValueError("incomplete set of seven requested conditions")
    return trials


def source(trial):
    return dict(
        source=trial["video"], model=MODELS[trial["mode"]],
        condition=COURSES[trial["course"]] + " · 정상 출발 · 중단 기준: 횡오차 ≥7m",
        result=result_name(trial["summary"]["status"]),
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(trial):
    s = trial["summary"]
    return {
        "학습에 사용한 카메라": MODELS[trial["mode"]],
        "도로": COURSES[trial["course"]], "출발 조건": "중앙에서 방향 0°",
        "결과": result_name(s["status"]),
        "진행률 (%)": round(100 * finite(s["completion_fraction"], "completion"), 2),
        "횡오차 RMSE (m)": s["cte_rmse_m"], "최대 횡오차 (m)": s["max_abs_cte_m"],
        "평균 속도 (km/h)": s["mean_speed_kph"], "CTE 중단 기준 (m 이상)": 7.0,
        "체크포인트 SHA256": s["checkpoint_sha256"], "원본 주행 폴더": relative(trial["run"]),
    }


def provenance(trial):
    s, settings, run = trial["summary"], trial["settings"], trial["run"]
    return dict(
        model=MODELS[trial["mode"]], model_mode=trial["mode"], course=trial["course"],
        condition=source(trial)["condition"],
        original_file=relative(trial["video"]), original_sha256=digest(trial["video"]),
        run_directory=relative(run), checkpoint_sha256=s["checkpoint_sha256"],
        settings_sha256=digest(run / "settings.json"), summary_sha256=digest(run / "summary.json"),
        initial_state_sha256=digest(run / "initial_state.json"),
        telemetry_sha256=digest(run / "telemetry.csv"),
        status=s["status"], result=result_name(s["status"]), initial_condition="nominal",
        initial_offset_m=0.0, initial_heading_offset_deg=0.0,
        measured_initial_cte_m=trial["initial"]["cte_m"],
        measured_initial_heading_deg=trial["initial"]["heading_error_deg"],
        cte_intervention_threshold_m=7.0, cte_intervention_definition=CTE_DEFINITION,
        standalone_heading_intervention=False, target_kph=settings["target_kph"],
        reference_station_m=settings["reference_station_m"],
    )


def render(stage, trials):
    """기존 영상 4개와 같은 파일명만 임시 영역에 생성한다."""
    definitions = [
        (NAMES[0], "학습하지 않은 직선에서 비교", [[trials[m, "U1"] for m in ("center", "three")]]),
        (NAMES[1], "학습하지 않은 곡선에서 비교", [[trials[m, "U2"] for m in ("center", "three")]]),
        (NAMES[2], "곡선 비중을 높여 다시 학습한 결과",
         [[trials["three_curves", c]] for c in ("U2", "U1")]),
        (NAMES[3], "명령 입력 없는 PilotNet의 교차로 주행", [[trials["three", "J"]]]),
    ]
    videos = []
    for name, title, chapters in definitions:
        destination = stage / "videos" / (name + ".mp4")
        with tempfile.TemporaryDirectory(prefix="selected_video_") as temporary:
            directory = Path(temporary)
            parts = []
            for index, chapter in enumerate(chapters):
                chapter_dir = directory / str(index)
                chapter_dir.mkdir()
                part = directory / (str(index) + ".mp4")
                encode([source(t) for t in chapter], part, chapter_dir)
                parts.append(part)
            if len(parts) == 1:
                shutil.copy2(parts[0], destination)
            else:
                listing = directory / "concat.txt"
                listing.write_text("".join("file '{}'\n".format(p) for p in parts), encoding="utf-8")
                subprocess.run([
                    "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(destination),
                ], check=True)
        preview = stage / "images" / (name + ".jpg")
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-y", "-ss", "0",
            "-i", str(destination), "-frames:v", "1", "-vf", "scale=960:-2", str(preview),
        ], check=True)
        videos.append(dict(title=title, video="videos/" + destination.name,
                           duration_s=duration(destination), sha256=digest(destination),
                           chapters=[[provenance(t) for t in chapter] for chapter in chapters]))
        print("STAGED", name, flush=True)
    return videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=7, required=True, metavar="RUN")
    args = parser.parse_args()
    hashes = {mode: digest(path) for mode, path in CHECKPOINTS.items()}
    protocol = read_json(BASE / "docs/evaluation_protocol.json")
    for mode in ("center", "three"):
        path = HISTORICAL_CHECKPOINT_KEYS[mode]
        if hashes[mode] != protocol["source_sha256"][path]:
            raise ValueError("frozen baseline checkpoint changed: " + mode)
    trials = load_runs(args.runs, hashes)
    for trial in trials.values():
        expected = protocol["held_out_sections"].get(trial["course"], [40.0, 108.0])
        if trial["settings"]["reference_station_m"] != expected:
            raise ValueError("evaluation route station range changed")

    preserved = OUTPUT / "metrics/open_loop.csv"
    open_loop_hash = digest(preserved)  # open-loop 결과는 이번 교체에서 변경하지 않는다.
    backup_root = BASE / "runs/publication_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    backup = backup_root / tag

    with tempfile.TemporaryDirectory(prefix="pilotnet_publication_") as temporary:
        stage = Path(temporary)
        for folder in ("videos", "images", "metrics"):
            (stage / folder).mkdir()
        videos = render(stage, trials)
        write_csv(stage / "metrics/driving_trials.csv", [
            metric_row(trials[mode, course]) for mode in ("center", "three") for course in ("U1", "U2")
        ])
        write_csv(stage / "metrics/curve_sampling_comparison.csv", [
            metric_row(trials[mode, course]) for course in ("U2", "U1") for mode in ("three", "three_curves")
        ])
        intersection = metric_row(trials["three", "J"])
        intersection.update({
            "평가 목표": "좌회전 경로 추종",
            "관찰된 주행": "진행 방향은 실제 영상 참조; 결과는 새 주행 summary 기준",
            "시험 성격": "학습 구간과 일부 겹치는 교차로 사례; 미학습 도로 일반화 시험과 구분",
        })
        write_csv(stage / "metrics/intersection_example.csv", [intersection])
        write_json(stage / "video_sources.json", dict(
            generated_utc=datetime.now(timezone.utc).isoformat(),
            scope="seven new nominal controlled trials only; no previous 1m-threshold trials reused",
            evaluation_scope=dict(
                held_out_sections=protocol["held_out_sections"],
                training_capture_station_ranges=protocol["training_capture_station_ranges"],
                intersection_demonstration=dict(course="J", reference_station_m=[40.0, 108.0],
                                               overlaps_training_capture=True),
            ),
            cte_intervention_threshold_m=7.0, cte_intervention_definition=CTE_DEFINITION,
            standalone_heading_intervention=False, initial_condition="nominal",
            editing="actual MORAI recordings at original speed; paired clips freeze the earlier ending with an explicit label; curve-weighted clips concatenate U2 then U1",
            interpretation="route completion is not proof of lane compliance or zero collisions; lower CTE thresholds from old trials are not mixed in these current tables",
            originals_preserved_locally=True, previous_results_backup=relative(backup),
            unchanged_open_loop_sha256=open_loop_hash, videos=videos,
        ))

        shutil.copytree(OUTPUT, backup / "results")  # 모든 예전 공개 파일을 교체 전에 보존한다.
        files = sorted(path.relative_to(stage) for path in stage.rglob("*") if path.is_file())
        write_json(backup / "replacement_manifest.json", dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            replacement_files=[str(p) for p in files],
            previous_sha256={str(p): digest(OUTPUT / p) for p in files if (OUTPUT / p).is_file()},
            reason="replace seven nominal PilotNet GUI trials using >=7m CTE intervention rule",
        ))
        try:
            # index는 마지막에 교체해 원본 출처 목록을 완성된 영상과 함께 공개한다.
            for item in sorted(files, key=lambda p: p.name == "video_sources.json"):
                destination = OUTPUT / item
                destination.parent.mkdir(parents=True, exist_ok=True)
                pending = destination.with_name(destination.name + ".replacement.tmp")
                shutil.copy2(stage / item, pending)
                pending.replace(destination)
            if digest(preserved) != open_loop_hash:
                raise RuntimeError("open_loop.csv changed during publication")
        except BaseException:
            for item in files:  # 도중 실패하면 교체 대상만 원래 공개 상태로 복구한다.
                original, destination = backup / "results" / item, OUTPUT / item
                if original.exists():
                    shutil.copy2(original, destination)
                elif destination.exists():
                    destination.unlink()
            raise
    print("REPLACED 4 existing videos; updated 4+4+1 closed-loop rows; open-loop unchanged")
    print("BACKUP", backup)


if __name__ == "__main__":
    main()
