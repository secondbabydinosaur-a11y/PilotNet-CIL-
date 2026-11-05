"""검증한 원시 결과를 GitHub용 표·그림·실제 MORAI 영상 모음으로 복사한다."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from data_utils import BASE, read_csv, read_json, write_json
from learning.assessment import recovery_metrics, telemetry_metrics
from learning.common import file_hash
from learning.live import Reference
from tools.collection.collect_unseen import verify_frozen

OUTPUT = BASE / "github_artifacts"
MATRICES = {
    "base_lateral": "matrix.json",
    "base_nominal_gui_rerun": "matrix_screen.json",
    "seen_heading": "matrix_heading_screen.json",
    "unseen_lateral": "matrix_unseen_screen.json",
    "unseen_heading": "matrix_heading_unseen_screen.json",
}
EXPECTED = {"base_lateral": 18, "base_nominal_gui_rerun": 6, "seen_heading": 8,
            "unseen_lateral": 12, "unseen_heading": 8}


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def probe(path):
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-threads", "2", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate:format=duration",
        "-of", "json", str(path)], text=True))
    stream = info["streams"][0]
    if int(stream["nb_read_frames"]) < 2 or float(info["format"]["duration"]) <= 0:
        raise RuntimeError("empty video: " + str(path))
    return {**stream, "duration_s": float(info["format"]["duration"])}


def export_test_teacher(session, output):
    frames = sorted((r for r in read_csv(session / "frames.csv") if r["camera"] == "center"),
                    key=lambda r: int(r["frame_id"]))
    if [int(r["frame_id"]) for r in frames] != list(range(len(frames))):
        raise RuntimeError("test teacher export needs continuous saved frame IDs")
    span = (int(frames[-1]["stamp_ns"]) - int(frames[0]["stamp_ns"])) / 1e9
    fps = (len(frames) - 1) / span
    destination = output / "teacher.mp4"
    if not destination.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-framerate", str(fps), "-start_number", "0", "-i", str(session / "images/center/%08d.jpg"),
            "-frames:v", str(len(frames)), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True)
    write_json(output / "video_source.json", dict(original_session=str(session.relative_to(BASE)),
        human_driving=False, used_for_training=False, evaluation_only=True,
        original_frames=len(frames), native_span_s=span, video_fps=fps,
        timing="CFR at measured mean camera rate; native timestamps in frames.csv"))


def collect_closed():
    records, used = [], set()
    for cohort, filename in MATRICES.items():
        path = BASE / "runs/closed_loop" / filename
        matrix = read_json(path)
        if len(matrix) != EXPECTED[cohort]:
            raise RuntimeError("incomplete matrix " + filename)
        for item in matrix:
            run = Path(item["path"])
            settings, summary = read_json(run / "settings.json"), read_json(run / "summary.json")
            if summary["status"] in ("failed", "interrupted", "not_started"):
                raise RuntimeError("unfinished trial included: " + run.name)
            if settings["checkpoint_sha256"] != summary["checkpoint_sha256"]:
                raise RuntimeError("trial checkpoint mismatch")
            logs = read_csv(run / "telemetry.csv")
            start, end = settings["reference_station_m"]
            record = {"cohort": cohort, "mode": item["mode"], "course": item["course"],
                "trial": item["trial"], "initial_offset_m": item["initial_offset_m"],
                "requested_heading_deg": settings.get("initial_heading_offset_deg", 0.),
                "measured_initial_cte_m": float(logs[0]["cte_m"]),
                "measured_initial_heading_deg": float(logs[0]["heading_error_deg"]),
                "status": summary["status"], "completed": int(summary["status"] == "completed"),
                "completion_fraction": summary["completion_fraction"],
                "elapsed_s": summary["elapsed_s"], "station_start_m": start, "station_end_m": end,
                **telemetry_metrics(logs, summary.get("terminal_cte_m")),
                **recovery_metrics(logs, start), "checkpoint_sha256": summary["checkpoint_sha256"],
                "run_id": run.name,
                "front_video": "trials/" + run.name + "/drive.mp4",
                "morai_video": "trials/" + run.name + "/morai_screen.mp4" if (run / "morai_screen.mp4").exists() else ""}
            records.append(record)
            used.add(run.name)
            for name in ("settings.json", "initial_state.json", "summary.json", "telemetry.csv", "drive.mp4", "morai_screen.mp4"):
                if (run / name).exists():
                    copy_file(run / name, OUTPUT / "trials" / run.name / name)
    excluded = []
    for run in sorted((BASE / "runs/closed_loop").iterdir()):
        if not run.is_dir() or run.name in used or not (run / "summary.json").exists():
            continue
        summary = read_json(run / "summary.json")
        if summary["status"] in ("failed", "interrupted", "not_started") or (run / "infrastructure_incident.json").exists():
            excluded.append(dict(run_id=run.name, status=summary["status"], error=summary.get("error"),
                reason="operator stop or initial-pose/infrastructure check; excluded from model success denominator"))
            for name in ("summary.json", "settings.json", "initial_state.json", "telemetry.csv", "infrastructure_incident.json"):
                if (run / name).exists():
                    copy_file(run / name, OUTPUT / "excluded_attempts" / run.name / name)
    write_json(OUTPUT / "reports/excluded_attempts.json", excluded)
    return records


def collect_open():
    rows = []
    for mode in ("center", "three"):
        source = BASE / ("runs/pilotnet_" + mode + "_v1")
        for name in ("best.pt", "settings.json", "summary.json", "history.csv", "learning_curve.png"):
            copy_file(source / name, OUTPUT / "models" / mode / name)
        for cohort, folder in (("seen", "open_loop_test"), ("unseen", "open_loop_unseen")):
            result = read_json(source / folder / "metrics.json")
            for name in ("metrics.json", "predictions.csv", "open_loop.png", "open_loop.svg"):
                copy_file(source / folder / name, OUTPUT / "open_loop" / mode / cohort / name)
            for course, metrics in [("all", result["metrics"])] + sorted(result["by_course"].items()):
                baseline = (result["zero_curvature_baseline"] if course == "all" else
                            result.get("zero_curvature_baseline_by_course", {}).get(course, {}))
                rows.append(dict(cohort=cohort, mode=mode, course=course, **metrics,
                    zero_baseline_rmse_1pm=baseline.get("rmse_1pm")))
    return rows


def plots(records, open_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})
    colors = {"center": "tab:blue", "three": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for j, cohort in enumerate(("seen", "unseen")):
        group = [r for r in open_rows if r["cohort"] == cohort and r["course"] == "all"]
        axes[0, j].bar([r["mode"] for r in group], [r["rmse_1pm"] for r in group],
                       color=[colors[r["mode"]] for r in group])
        axes[0, j].set(title=cohort + " roads: open loop", ylabel="Curvature RMSE (1/m)")
    for j, cohort in enumerate(("base_lateral", "unseen_lateral")):
        group = [r for r in records if r["cohort"] == cohort and r["course"] != "J"]
        for i, mode in enumerate(colors):
            selected = [r for r in group if r["mode"] == mode]
            passed = sum(r["completed"] for r in selected)
            axes[1, j].bar(i, passed / len(selected), color=colors[mode])
            axes[1, j].text(i, passed / len(selected) + .02, "{}/{}".format(passed, len(selected)), ha="center")
        axes[1, j].set(xticks=[0, 1], xticklabels=list(colors), ylim=(0, 1.2),
                       title=cohort + ": completed trials", ylabel="Fraction")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / ("plots/overview." + extension), dpi=170)
    plt.close(fig)
    # 각 조건에서의 시간 경과를 그대로 보여 준다. 중단 이후 가짜 궤적을 덧붙이지 않는다.
    for course in ("A", "D", "J", "U1", "U2"):
        subset = [r for r in records if r["course"] == course and r["cohort"] != "base_nominal_gui_rerun"]
        names = sorted({r["trial"] for r in subset})
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.6 * len(names)), squeeze=False)
        for j, name in enumerate(names):
            for row in (r for r in subset if r["trial"] == name):
                log = read_csv(OUTPUT / "trials" / row["run_id"] / "telemetry.csv")
                t = [float(r["elapsed_s"]) for r in log]
                for ax, key in zip(axes[j], ("cte_m", "heading_error_deg")):
                    ax.plot(t, [float(r[key]) for r in log], label=row["mode"] + " / " + row["status"],
                            color=colors[row["mode"]])
                    ax.set(xlabel="Time (s)", ylabel=key, title=course + " / " + name)
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT / ("plots/tracking_" + course + ".png"), dpi=150)
        plt.close(fig)
    route = np.loadtxt(BASE / "assets/evaluation_route.csv", delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(route[:, 1], route[:, 2], color="lightgray", label="official route (reference)")
    for name, low, high, color in (("train A", 0, 65, "tab:blue"), ("train D", 640, 866, "tab:blue"),
                                    ("unseen U1", 1300, 1500, "tab:red"), ("unseen U2", 978.337, 1210, "tab:orange")):
        points = route[(route[:, 0] >= low) & (route[:, 0] <= high)]
        ax.plot(points[:, 1], points[:, 2], linewidth=4, color=color, label=name + " s={:.1f}-{:.1f}m".format(low, high))
        ax.annotate(name, points[len(points)//2, 1:3], xytext=(10, 12), textcoords="offset points")
    ax.set(aspect="equal", xlabel="Local X (m)", ylabel="Local Y (m)", title="Training capture and held-out road sections")
    ax.legend(loc="best")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "plots/course_map.png", dpi=180)
    fig.savefig(OUTPUT / "plots/course_map.svg")
    plt.close(fig)


def documentation(records, open_rows):
    groups = defaultdict(list)
    for row in records:
        groups[(row["cohort"], row["mode"], row["course"])].append(row)
    aggregate = []
    lines = ["| 시험 묶음 | 모델 | 코스 | 완주 | 평균 진행률 |", "|---|---|---|---|---|"]
    for (cohort, mode, course), values in sorted(groups.items()):
        passed = sum(r["completed"] for r in values)
        progress = float(np.mean([r["completion_fraction"] for r in values]))
        distance = float(sum(r["distance_driven_m"] for r in values))
        interventions = sum(r["status"].startswith("intervention") for r in values)
        aggregate.append(dict(cohort=cohort, mode=mode, course=course, completed=passed,
                              trials=len(values), mean_completion_fraction=progress,
                              distance_driven_m=distance, interventions=interventions,
                              interventions_per_km=1000*interventions/distance if distance else None))
        lines.append("| {} | {} | {} | {}/{} | {:.1%} |".format(cohort, mode, course, passed, len(values), progress))
    write_csv(OUTPUT / "reports/closed_loop_aggregate.csv", aggregate)
    video_lines = ["# 시험별 영상", "", "## PP 자동 teacher 원본", "",
        "- [학습에 사용한 직선](teacher/01_teacher_straight.mp4)",
        "- [학습에 사용한 곡선](teacher/02_teacher_curves.mp4)",
        "- [시험에만 사용한 미학습 직선 U1](teacher/U1/teacher.mp4)",
        "- [시험에만 사용한 미학습 곡선 U2](teacher/U2/teacher.mp4)", "",
        "## PilotNet 주행", "", "GUI는 실제 MORAI 창, front는 실시간 전방 카메라와 진단 표시다.", "",
        "| 시험 묶음 | 모델 | 코스 | 초기 조건 | 결과 | MORAI GUI | front |", "|---|---|---|---|---|---|---|"]
    for r in records:
        gui = "[MP4](" + r["morai_video"] + ")" if r["morai_video"] else "미녹화"
        video_lines.append("| {cohort} | {mode} | {course} | {trial} | {status} | ".format(**r)
                           + gui + " | [MP4](" + r["front_video"] + ") |")
    (OUTPUT / "videos.md").write_text("\n".join(video_lines) + "\n", encoding="utf-8")
    table = "\n".join(lines)
    olines = ["| 도로 | 모델 | 이미지 수 | 곡률 RMSE (1/m) | 곡률 MAE (1/m) |",
              "|---|---|---|---|---|"]
    for r in open_rows:
        if r["course"] == "all":
            olines.append("| {} | {} | {} | {:.6f} | {:.6f} |".format(r["cohort"], r["mode"], r["samples"], r["rmse_1pm"], r["mae_1pm"]))
    text = """# PilotNet 실험 기록

가운데 카메라로 학습한 모델(`center`)과 좌·중·우 카메라로 학습한 모델(`three`)을 비교했다.
**실시간 조향 입력은 두 모델 모두 가운데 카메라 한 장이다.** 원본은 사람이 아닌 지도 PP 자동 teacher 주행이다.
기존 모델과 train/val/test CSV를 고정한 뒤 미학습 도로와 헤딩 복귀 시험을 추가했다.

- [GitHub에서 시험별 영상 보기](videos.md): 실제 MORAI GUI 영상과 전방 센서 영상 구분
- [로컬 영상 갤러리](index.html): 저장소를 내려받아 브라우저로 열기
- [코스 지도](plots/course_map.png)
- [open-loop CSV](reports/open_loop.csv), [closed-loop CSV](reports/closed_loop.csv)
- [고정한 시험 계획](reports/protocol.json), [제외한 설정 실패·사용자 중단](reports/excluded_attempts.json)
- [모델·학습 기록](models), [원본 teacher 영상·출처](teacher)
- [전체 파일 SHA256 및 영상 검사](reports/files.json)

## 시험 조건

| 코스 | 범위 | 의미 |
|---|---|---|
| A | s=0~55m | 학습했던 직선 |
| D | s=650~850m | 학습했던 연속 곡선 |
| J | s=40~108m | 미학습 좌회전 교차로, 기존 코스 일부와 겹침 |
| U1 | s=1300~1500m, 200m | 학습에 없는 긴 직선 |
| U2 | s=978.337~1210m, 약 232m | 학습에 없는 진입 곡선과 연결 직선 |

초기 조건은 정상, 횡위치 +0.3m/−0.3m, 헤딩 +5°/−5°다. 헤딩 시험은 위치를 가운데에 둔다.
각 조건당 한 번이며 횡오프셋 3개를 독립 반복 3회로 해석하지 않는다. 새로운 헤딩·미학습 시험은
실제 UDP 상태로 초기 각도·위치를 확인했다. 기존 시험의 요청/측정 초기값도 CSV에 구분했다.
`base_nominal_gui_rerun`은 기존 정상 출발을 GUI 녹화로 다시 실행한 묶음이며 원래 시험과 합산하지 않는다.

목표 19km/h, 같은 맑은 정오, 같은 센서/차량이다. 고속도로 위치의 시험도 19km/h로 진행했다.
신경망은 조향만 결정하고 종방향은 고정 속도 피드백이다. 지도는 초기 배치와 평가에만 사용한다.
종료점 1m 이내면 완주, |횡오차|>1m이면 중단한다. 센서 지연과 헤딩 이탈 중단도 기록한다.
진행률은 마지막 제어 로그 기준이어서 완주도 약 98~99.5%로 표시될 수 있다.

복귀는 3m 이상 전진한 뒤 |횡오차|≤0.15m와 |헤딩 오차|≤2°를 1초 연속 유지한 첫 구간이다.
복귀 시간은 제어 시작부터 측정하며 정차 출발 가속을 포함한다. 복귀한 뒤의 재이탈/최종 중단은 별도로 본다.

## Open-loop

OL_TABLE

`seen`은 같은 A/D 도로의 다른 주행 그룹(g06), `unseen`은 새 U1/U2 도로의 시험 전용 teacher다.
두 평가 모두 이미지마다 실제 조향 피드백을 시간 보간한 뒤 `−tan(조향각)/3m`로 만든 곡률을 정답으로 쓴다.
이는 자전거 모델 가정이며 실제 궤적 곡률을 독립 계측한 정답은 아니다. 곡률이 거의 0인 직선의 R²는
작은 오차에도 불안정할 수 있으므로 MAE/RMSE와 zero-curvature baseline을 함께 본다.

## Closed-loop

CL_TABLE

중단 시험의 횡오차 통계는 중단 전 구간만 포함한다. 짧게 실패한 모델의 낮은 RMSE를 우수한 완주로 해석하지 않는다.
누적 주행거리와 intervention/km도 CSV에 기록한다. 구간 끝에서 시험을 종료하므로 무한 연속주행의
평균 무개입 거리나 원 논문의 autonomy 점수로 해석하지 않는다.
조건별 시계열 그림은 [plots](plots)에 있다. 영상은 실제 실행 속도이며 종료 후 제동 구간이 포함될 수 있다.
GUI 영상은 `morai_screen.mp4`, 실제 전방 카메라+진단 표시는 `drive.mp4`다.
각 영상은 프레임 수·길이·해상도를 ffprobe로 확인했다. 초기 18개 시험에는 전방 영상만 있고
정상 출발 재시험·추가 헤딩·미학습 시험에는 실제 MORAI 창 영상도 있다.
추론 시간은 새 프레임의 모델 forward와 scalar 출력 동기화 구간이며 JPEG 디코딩·전처리·전송 시간은 포함하지 않는다.

## 해석 범위

- 같은 K-City 맵의 미학습 구간 시험이다. 새 도시·날씨·차량 일반화를 검증한 결과는 아니다.
- 3개 카메라 학습은 같은 30 epoch에서 더 많은 이미지와 optimizer step을 사용했다.
- 좌우 카메라 복귀 라벨은 2초 뒤 teacher 위치에 대한 우리의 PP-delta 근사다. yaw 시점 증강은 학습에 없다.
- 교차로 실패에는 미학습 형상과 목적지 명령 부재가 함께 영향을 줄 수 있다. 모든 교차로 통과가
  PilotNet 구조상 불가능하다는 근거로 사용하지 않는다. CIL은 아직 구현하지 않았다.
- 충돌 ground truth를 별도로 계측하지 않았다. 충돌 0회를 검증한 결과로 표시하지 않는다.
- 단일 학습 seed와 조건별 한 번의 실행이다. 실험 조건에서 관측한 결과로 해석한다.

## 재현과 파일

저장소 루트의 `README.md`와 `docs/collection.md`에 수집·학습 명령이 있다. 모델 가중치는
`github_artifacts/models/{center,three}/best.pt`에 있다. 기존 원본 경로의 체크포인트와 SHA256이 같다.
원본 JPEG 약 13GB와 MORAI 설치/지도 파일은 이 결과 폴더에 복사하지 않았다. 데이터 원본은 로컬
`dataset/raw`와 `dataset/unseen_raw`에 보존한다. CSV의 이미지 경로는 저장소 루트 기준이며,
이미지 단위 평가 재현에는 해당 원본이 필요하다. 원본이 포함된 공개 데이터셋이라고 표기하지 않는다.

```bash
# 모델 고정 확인 후 추가 시험을 재실행할 때: 기존 완료된 주행 기록은 재사용한다.
bash run.sh -m tools.evaluation.run_matrix --courses A D --conditions heading_left heading_right --record-screen
bash run.sh -m tools.evaluation.run_matrix --courses U1 U2 --repeats 3 --record-screen --tag unseen
bash run.sh -m tools.evaluation.run_matrix --courses U1 U2 --conditions heading_left heading_right --record-screen --tag unseen
bash run.sh -m tools.evaluation.build_github_artifacts
```

논문: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
게시 대상: [secondbabydinosaur-a11y/PilotNet-CIL-](https://github.com/secondbabydinosaur-a11y/PilotNet-CIL-).
""".replace("OL_TABLE", "\n".join(olines)).replace("CL_TABLE", table)
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")
    cards = []
    for r in records:
        videos = []
        for key, title in (("morai_video", "Actual MORAI window"), ("front_video", "Live center camera + diagnostics")):
            if r[key]:
                videos.append('<p>{}</p><video controls preload="none" width="640" src="{}"></video>'.format(title, html.escape(r[key])))
        title = "{cohort} / {mode} / {course} / {trial}".format(**r)
        cards.append('<article data-course="{}"><h3>{}</h3><p>{} | progress {:.1%} | CTE RMSE {:.3f}m | recovered {}</p>{}<p><a href="trials/{}/telemetry.csv">telemetry CSV</a></p></article>'.format(
            r["course"], html.escape(title), r["status"], r["completion_fraction"], r["cte_rmse_m"],
            r["recovered"], "".join(videos), r["run_id"]))
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PilotNet experiment archive</title>
<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;background:#111827;color:#e5e7eb}a{color:#93c5fd}article{padding:20px;margin:20px 0;background:#1f2937;border-radius:10px}video{max-width:100%}select{font-size:18px}img{max-width:100%}</style>
<h1>PilotNet experiment archive</h1><p>Both models steer using only the center camera. Teacher = automatic map PP. Frozen models; same-map unseen roads.</p>
<p><a href="README.md">실험 조건과 해석 범위</a> · <a href="reports/closed_loop.csv">Closed-loop CSV</a> · <a href="reports/open_loop.csv">Open-loop CSV</a></p>
<img src="plots/overview.png" width="900"><p>Course filter: <select onchange="document.querySelectorAll('article').forEach(a=>a.hidden=this.value!='all'&&a.dataset.course!=this.value)"><option>all</option><option>A</option><option>D</option><option>J</option><option>U1</option><option>U2</option></select></p>
''' + "\n".join(cards) + "</html>"
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")


def main():
    protocol = verify_frozen()
    OUTPUT.mkdir(exist_ok=True)
    for directory in ("reports", "plots", "teacher", "data_manifests"):
        (OUTPUT / directory).mkdir(exist_ok=True)
    records, open_rows = collect_closed(), collect_open()
    write_csv(OUTPUT / "reports/closed_loop.csv", records)
    write_csv(OUTPUT / "reports/open_loop.csv", open_rows)
    write_json(OUTPUT / "reports/protocol.json", protocol)
    write_json(OUTPUT / "reports/frozen_verification.json", dict(
        checked_utc=datetime.now(timezone.utc).isoformat(), matched=True,
        source_sha256={name: file_hash(BASE / name) for name in protocol["source_sha256"]},
        model_updated_after_test=False))
    for name in ("teacher_sources.json", "videos/01_teacher_straight.mp4", "videos/02_teacher_curves.mp4"):
        copy_file(BASE / "results/pilotnet_comparison_20261105" / name, OUTPUT / "teacher" / Path(name).name)
    teacher_sources = read_json(OUTPUT / "teacher/teacher_sources.json")
    for source in teacher_sources:
        source["original_export_video_field"] = source["video"]
        source["video"] = Path(source["video"]).name
    write_json(OUTPUT / "teacher/teacher_sources.json", teacher_sources)
    for path in ("dataset/splits/groups.json", "dataset/paired/summary.json", "dataset/paired_three/summary.json",
                 "dataset/unseen_sessions.json", "dataset/unseen_paired/summary.json", "dataset/unseen_paired/test.csv"):
        source = BASE / path
        if source.exists():
            copy_file(source, OUTPUT / "data_manifests" / path)
    for course, session in read_json(BASE / "dataset/unseen_sessions.json").items():
        for name in ("session.json", "quality_report.json", "frames.csv", "vehicle.csv", "commands.csv", "expert.csv"):
            copy_file(BASE / session / name, OUTPUT / "teacher" / course / name)
        export_test_teacher(BASE / session, OUTPUT / "teacher" / course)
    selected_sessions = set(read_json(BASE / "dataset/unseen_sessions.json").values())
    rejected = []
    for meta_path in sorted((BASE / "dataset/unseen_raw").glob("*/session.json")):
        if str(meta_path.parent.relative_to(BASE)) not in selected_sessions:
            meta = read_json(meta_path)
            rejected.append(dict(session_id=meta["session_id"], status=meta["status"],
                fatal_errors=meta.get("fatal_errors", []), used_for_evaluation=False,
                used_for_training=False, reason="incomplete test-only capture; original preserved locally"))
    write_json(OUTPUT / "data_manifests/excluded_test_captures.json", rejected)
    plots(records, open_rows)
    documentation(records, open_rows)
    files = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path != OUTPUT / "reports/files.json":
            entry = dict(path=str(path.relative_to(OUTPUT)), bytes=path.stat().st_size, sha256=file_hash(path))
            files.append(entry)
    videos = [entry for entry in files if entry["path"].endswith(".mp4")]
    # 녹화가 모두 끝난 뒤 독립된 파일 검사만 병렬로 수행한다.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for entry, metadata in zip(videos, pool.map(probe, [OUTPUT / entry["path"] for entry in videos])):
            entry["video"] = metadata
    print("VERIFIED VIDEO FILES", len(videos), flush=True)
    write_json(OUTPUT / "reports/files.json", files)
    verify_frozen()
    print("GITHUB ARTIFACTS", OUTPUT, "trials", len(records), "videos", sum("video" in r for r in files), flush=True)


if __name__ == "__main__":
    main()
