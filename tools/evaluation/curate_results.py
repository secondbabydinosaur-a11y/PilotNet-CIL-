"""PilotNet 교차로 사례 영상 1개와 기존 새 도로 평가 표를 로컬 아카이브에서 만든다.

학습/시뮬레이터 실행은 하지 않는다. 좌회전 평가 경로에서 직진 방향으로
진행한 실제 MORAI 녹화를 원래 속도로 보여 준다.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from data_utils import BASE, read_csv, read_json, write_json

ARCHIVE = BASE / "github_artifacts"
OUTPUT = BASE / "results"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
MODELS = {"center": "가운데 카메라로 학습", "three": "카메라 3개로 학습"}
COURSES = {"U1": "학습하지 않은 직선", "U2": "학습하지 않은 곡선"}
CONDITIONS = {"nominal": "중앙에서 방향 0°", "left": "왼쪽으로 0.3m 치우쳐 출발",
              "right": "오른쪽으로 0.3m 치우쳐 출발", "heading_left": "진행 방향을 왼쪽으로 5° 틀어 출발",
              "heading_right": "진행 방향을 오른쪽으로 5° 틀어 출발"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)], text=True))


def result_name(status):
    return {"completed": "완주", "intervention_cte": "횡오차 1m 초과로 종료",
            "intervention_heading": "방향 오차로 종료", "intervention_sensor": "센서 수신 중단",
            "timeout": "시간 초과"}.get(status, status)


def text_file(directory, name, value):
    path = directory / name
    path.write_text(value, encoding="utf-8")
    return path


def overlay(text_path, size, y, color="white", extra=""):
    return "drawtext=fontfile={}:textfile={}:fontsize={}:fontcolor={}:x=20:y={}{}".format(
        FONT, text_path, size, color, y, extra)


def encode(sources, output, directory):
    """한 장면: single 또는 side by side. sources는 표시 문구와 원본 경로다."""
    paired = len(sources) == 2
    width, height, viewport = (960, 760, 600) if paired else (1280, 800, 660)
    lengths = [duration(item["source"]) for item in sources]
    filters, command = [], ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-filter_complex_threads", "2"]
    for source in sources:
        command += ["-threads", "2", "-i", str(source["source"])]
    for i, (source, seconds) in enumerate(zip(sources, lengths)):
        label = text_file(directory, "label_" + str(i), source["model"])
        description = text_file(directory, "condition_" + str(i), source["condition"])
        result = text_file(directory, "result_" + str(i), "결과: " + source["result"])
        chain = ("[{}:v]fps=20,scale={}:{}:force_original_aspect_ratio=decrease,"
                 "pad={}:{}:(ow-iw)/2:88+({}-ih)/2:color=0x101827,setsar=1,").format(
                     i, width, viewport, width, height, viewport)
        chain += ",".join([overlay(label, 30, 10), overlay(description, 21, 52),
                            overlay(result, 22, height-58, "0xffd166")])
        if seconds < max(lengths) - .05:
            note = text_file(directory, "ended_" + str(i), "시험 종료 · 마지막 화면 고정")
            chain += ",tpad=stop_mode=clone:stop_duration={:.4f},{}".format(
                max(lengths)-seconds, overlay(note, 18, height-28, "0xff8b8b", ":enable='gte(t,{:.4f})'".format(seconds)))
        chain += "[pane{}]".format(i)
        filters.append(chain)
    if paired:
        filters.append("[pane0][pane1]hstack=inputs=2[out]")
    else:
        filters.append("[pane0]null[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-threads", "4", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluation_scope():
    protocol = read_json(BASE / "docs/evaluation_protocol.json")
    sections = {course: protocol["held_out_sections"][course] for course in COURSES}
    for start, end in sections.values():
        for low, high in protocol["training_capture_station_ranges"]:
            if start - 8 <= high and end + 8 >= low:
                raise ValueError("public test overlaps a training capture")
    return dict(held_out_sections=sections,
                training_capture_station_ranges=protocol["training_capture_station_ranges"],
                description="U1/U2 are held-out road tests; J is a separate intersection demonstration",
                intersection_demonstration=dict(course="J", reference_station_m=[40., 108.],
                    expected_route="left turn", observed_behavior="continued towards straight; CTE intervention",
                    overlaps_training_capture=True,
                    interpretation="illustrates missing navigation-command input; not a disjoint-road test or proof that every intersection is impossible"))


def tables(rows):
    evaluation_scope()
    output = OUTPUT / "metrics"
    output.mkdir(exist_ok=True)
    open_rows = []
    for r in read_csv(ARCHIVE / "reports/open_loop.csv"):
        if r["course"] != "all" or r["cohort"] != "unseen":
            continue
        open_rows.append({"학습에 사용한 카메라": MODELS[r["mode"]],
            "평가 도로": "학습하지 않은 도로",
            "이미지 수": r["samples"], "곡률 RMSE (1/m)": r["rmse_1pm"], "곡률 MAE (1/m)": r["mae_1pm"]})
    write_csv(output / "open_loop.csv", open_rows)
    trials = []
    for r in rows:
        if r["course"] not in COURSES or r["cohort"] not in ("unseen_lateral", "unseen_heading"):
            continue  # 새 도로의 모든 출발 조건을 포함하고, 기존 도로 기록은 로컬에 보존한다.
        trials.append({"학습에 사용한 카메라": MODELS[r["mode"]], "도로": COURSES[r["course"]],
            "출발 조건": CONDITIONS[r["trial"]], "결과": result_name(r["status"]),
            "진행률 (%)": round(100*float(r["completion_fraction"]), 2),
            "횡오차 RMSE (m)": r["cte_rmse_m"], "최대 횡오차 (m)": r["max_abs_cte_m"],
            "복귀 여부": "해당 없음 (정상 출발)" if r["trial"] == "nominal" else
                         "복귀" if r["recovered"] == "True" else "미복귀",
            "복귀 시간 (s)": "" if r["trial"] == "nominal" else r["recovery_time_s"]})
    write_csv(output / "driving_trials.csv", trials)
    extra = []
    for r in read_csv(ARCHIVE / "curve_sampling_v2/comparison.csv"):
        if r["course"] not in COURSES:
            continue
        extra.append({"학습 조건": "기존 카메라 3개 학습" if r["version"] == "v1_target_weight" else "곡선 비중을 높인 카메라 3개 학습",
            "도로": COURSES[r["course"]], "출발 조건": CONDITIONS["nominal"], "결과": result_name(r["status"]),
            "진행률 (%)": round(100*float(r["completion_fraction"]), 2), "횡오차 RMSE (m)": r["cte_rmse_m"]})
    write_csv(output / "curve_sampling_comparison.csv", extra)
    intersection = read_json(ARCHIVE / "trials/three_J_nominal_screen/summary.json")
    write_csv(output / "intersection_example.csv", [{
        "학습에 사용한 카메라": MODELS["three"], "도로": "교차로 J (s40~108m)",
        "평가 목표": "좌회전 경로 추종", "관찰된 주행": "직진 방향으로 진행 후 중단",
        "결과": result_name(intersection["status"]),
        "횡오차 RMSE (m)": intersection["cte_rmse_m"],
        "최대 횡오차 (m)": intersection["max_abs_cte_m"],
        "시험 성격": "학습 구간과 일부 겹치는 사례 영상; 새 도로 일반화 시험과 별도"}])


def main():
    scope = evaluation_scope()
    if not FONT.exists():
        raise RuntimeError("Korean font needed: " + str(FONT))
    (OUTPUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "images").mkdir(exist_ok=True)
    index = OUTPUT / "video_sources.json"
    previous = read_json(index) if index.exists() else dict(videos=[])
    rows = read_csv(ARCHIVE / "reports/closed_loop.csv")
    intersection = next(r for r in rows if r["mode"] == "three" and r["course"] == "J"
                        and r["trial"] == "nominal" and r["cohort"] == "base_nominal_gui_rerun")
    definitions = [
        ("01_intersection", "PilotNet 교차로 사례: 좌회전 경로에서 직진 방향으로 진행", [[dict(
            source=ARCHIVE / intersection["morai_video"], model=MODELS["three"],
            condition="교차로 · " + CONDITIONS["nominal"], result=result_name(intersection["status"]))]]),
    ]
    def create(definition):
        name, title, chapters = definition
        destination = OUTPUT / "videos" / (name + ".mp4")
        with tempfile.TemporaryDirectory(prefix="pilotnet_selected_") as tmp:
            directory = Path(tmp)
            parts = []
            for i, sources in enumerate(chapters):
                part_dir = directory / str(i);part_dir.mkdir()
                part = directory / (str(i) + ".mp4")
                encode(sources, part, part_dir)
                parts.append(part)
            if len(parts) == 1:
                shutil.copy2(str(parts[0]), str(destination))
            else:
                listing = directory / "concat.txt"
                listing.write_text("".join("file '{}'\n".format(p) for p in parts))
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(destination)], check=True)
        preview = OUTPUT / "images" / (name + ".jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-y", "-ss", "2", "-i", str(destination),
            "-frames:v", "1", "-vf", "scale=960:-2", str(preview)], check=True)
        print("SELECTED", title, flush=True)
        return dict(title=title, video=str(destination.relative_to(OUTPUT)), duration_s=duration(destination),
            sha256=digest(destination), chapters=[[dict(**{k:v for k,v in item.items() if k != "source"},
                original_file=str(item["source"].relative_to(BASE)), original_sha256=digest(item["source"])) for item in sources] for sources in chapters])
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = list(pool.map(create, definitions))
    # 이전 공개 목록에서 빠진 영상만 로컬로 옮긴다. 재생성해도 예전 코스가 남지 않는다.
    current_paths = {item["video"] for item in sources}
    retired = [OUTPUT / item["video"] for item in previous["videos"]
               if item["video"] not in current_paths]
    if retired:
        (BASE / "runs").mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix="retired_public_videos_", dir=str(BASE / "runs")))
        for path in retired:
            if path.parent != OUTPUT / "videos":
                raise ValueError("unexpected video path in previous public index")
            for old in (path, OUTPUT / "images" / (path.stem + ".jpg")):
                if old.exists():
                    destination = backup / old.relative_to(OUTPUT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old), str(destination))
    write_json(index, dict(evaluation_scope=scope,
        editing="actual MORAI window recording at original speed; explanatory labels added",
        originals_preserved_locally=True, videos=sources))
    tables(rows)
    print("CURATED", len(sources), "videos", flush=True)


if __name__ == "__main__":
    main()
