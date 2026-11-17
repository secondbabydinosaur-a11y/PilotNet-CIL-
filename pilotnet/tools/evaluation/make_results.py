"""과거 학습 도로의 내부 점검 기록을 로컬에 복원한다. 공개 시험은 curate_results 사용."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from data_utils import BASE, read_csv, read_json, write_json
from learning.common import file_hash


def teacher_videos(output):
    videos = output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    sources = []
    for sid, name, title in (
        ("20261103T234006Z_A_50527a", "01_teacher_straight.mp4", "Automatic PP teacher: straight"),
        ("20261103T234027Z_D_f8b565", "02_teacher_curves.mp4", "Automatic PP teacher: curves")):
        session = BASE / "dataset/raw" / sid
        meta = read_json(session / "session.json")
        frames = sorted((r for r in read_csv(session / "frames.csv") if r["camera"] == "center"),
                        key=lambda r: int(r["frame_id"]))
        if [int(r["frame_id"]) for r in frames] != list(range(len(frames))):
            raise ValueError("teacher export expects continuous JPEG frame IDs")
        seconds = (int(frames[-1]["stamp_ns"]) - int(frames[0]["stamp_ns"])) / 1e9
        fps = (len(frames) - 1) / seconds
        destination = videos / name
        if not destination.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-framerate", str(fps), "-start_number", "0", "-i", str(session / "images/center/%08d.jpg"),
                "-frames:v", str(len(frames)), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True)
        sources.append(dict(title=title, video="videos/" + name, original_session=str(session),
            group=meta["group"], used_in_training=True, human_driving=False,
            provenance=meta["control"], original_frames=len(frames), native_span_s=seconds,
            presentation_fps=fps, presentation_timing="CFR at measured mean camera rate; native stamps retained in original CSV",
            session_sha256=file_hash(session / "session.json")))
        print("TEACHER {} ({:.1f}s)".format(destination, seconds), flush=True)
    write_json(output / "teacher_sources.json", sources)
    return sources


def plots(output, matrix, open_loop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ("center", "three")
    axes[0].bar(labels, [open_loop[m]["metrics"]["rmse_1pm"] for m in labels])
    axes[0].set(ylabel="Curvature RMSE (1/m)", title="Open-loop: same held-out center images")
    courses = ("A", "D", "J")
    for i, mode in enumerate(labels):
        values = [np.mean([r["status"] == "completed" for r in matrix
                           if r["mode"] == mode and r["course"] == c]) for c in courses]
        axes[1].bar(np.arange(3) + (i - .5) * .35, values, width=.35, label=mode)
    axes[1].set(xticks=np.arange(3), xticklabels=("Straight", "Curves", "Junction"),
                ylim=(0, 1.1), ylabel="Completed trials / attempted trials", title="MORAI closed-loop")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "runs/legacy_results/pilotnet_comparison_20261105")
    parser.add_argument("--teachers-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = teacher_videos(output)
    if args.teachers_only:
        return
    matrix = read_json(BASE / "runs/closed_loop/matrix.json")
    screen_matrix = read_json(BASE / "runs/closed_loop/matrix_screen.json")
    expected = {(m, c, t) for m in ("center", "three") for c in ("A", "D", "J")
                for t in ("nominal", "left", "right")}
    if {(r["mode"], r["course"], r["trial"]) for r in matrix} != expected:
        raise ValueError("the three-offset closed-loop matrix is incomplete")
    if {(r["mode"], r["course"]) for r in screen_matrix} != {(m, c) for m in ("center", "three") for c in ("A", "D", "J")}:
        raise ValueError("the actual MORAI GUI video matrix is incomplete")
    records = list(sources)
    for number, mode, course, label in (
        (3, "center", "A", "straight"), (4, "center", "D", "curves"), (5, "center", "J", "junction"),
        (6, "three", "A", "straight"), (7, "three", "D", "curves"), (8, "three", "J", "junction")):
        trial = next(r for r in screen_matrix if r["mode"] == mode and r["course"] == course and r["trial"] == "nominal")
        # 수집 당시 절대경로 대신 이동한 PilotNet 폴더 안에서 같은 실행을 찾는다.
        trial_dir = BASE / "runs/closed_loop" / Path(trial["path"]).name
        original = trial_dir / "morai_screen.mp4"
        name = "{:02d}_{}_{}.mp4".format(number, mode, label)
        destination = output / "videos" / name
        if not destination.exists():
            shutil.copyfile(original, destination)
        front = output / "videos/front" / name
        front.parent.mkdir(exist_ok=True)
        if not front.exists():
            shutil.copyfile(trial_dir / "drive.mp4", front)
        records.append(dict(title="PilotNet {}: {}".format(mode, label), video="videos/" + name,
                            trial=trial, front_camera_video="videos/front/" + name,
                            selection="predefined zero-offset nominal GUI-recorded trial; not best-trial selection"))
    open_loop = {mode: read_json(BASE / "runs" / ("pilotnet_" + mode + "_v1") / "open_loop_test/metrics.json")
                 for mode in ("center", "three")}
    report = dict(status="experiment_complete", teacher_sources=sources, presentation_videos=records,
        open_loop=open_loop, closed_loop_trials=matrix, presentation_trials=screen_matrix,
        comparison_notes=["Both models run on center camera at inference.",
            "Three-camera train labels use our PP-delta 2s approximation, not the unpublished NVIDIA formula.",
            "Same A/D roads are repeated across groups. Test scores are not unseen-road generalization.",
            "J overlaps training capture stations 40-65m; it is an internal diagnostic, not a disjoint-road test.",
            "An unconditional PilotNet may traverse a junction; it cannot select arbitrary user intent from identical images alone.",
            "Both use 30 epochs; three-camera training has more images and optimizer steps.",
            "Collision ground truth was not instrumented; CTE intervention threshold is 1m.",
            "CIL is not implemented or evaluated.",
            "Primary model videos are actual X11 recordings of the MORAI window; front-camera videos are also preserved.",
            "Closed-loop chart uses the three-offset matrix; presentation GUI videos are separate zero-offset repetitions."])
    write_json(output / "results.json", report)
    fields = ("mode", "course", "trial", "initial_offset_m", "status", "completion_fraction",
              "cte_rmse_m", "max_abs_cte_m", "interventions", "elapsed_s", "path")
    with (output / "closed_loop.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in matrix)
    with (output / "open_loop.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "samples", "rmse_1pm", "mae_1pm", "bicycle_steer_mae_deg"))
        writer.writeheader()
        writer.writerows({"model": mode, **{k: result["metrics"][k] for k in writer.fieldnames[1:]}}
                         for mode, result in open_loop.items())
    plots(output, matrix, open_loop)
    cards = []
    for record in records:
        title = html.escape(record["title"])
        caption = "Automatic teacher; original JPEG/CSV retained; this run is in train."
        if "trial" in record:
            t = record["trial"]
            caption = "{} | CTE RMSE {:.3f}m | nominal offset=0".format(t["status"], t["cte_rmse_m"])
        cards.append('<article><h2>{}</h2><video controls preload="metadata" src="{}"></video><p>{}</p></article>'.format(
            title, html.escape(record["video"]), html.escape(caption)))
    notes = "".join("<li>" + html.escape(n) + "</li>" for n in report["comparison_notes"])
    document = '''<!doctype html><meta charset="utf-8"><title>PilotNet KASA experiment</title>
<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#17202a}
main{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;padding:18px;border-radius:12px}
video,img{width:100%}h2{font-size:18px}li{margin:9px 0} @media(max-width:750px){main{grid-template-columns:1fr}}</style>
<h1>PilotNet KASA: center vs three-camera training</h1>
<p>Actual MORAI GUI recordings for model trials. Original camera recordings for teacher runs.</p>
<p><a href="results.json">Full results JSON</a> · <a href="open_loop.csv">Open-loop CSV</a> · <a href="closed_loop.csv">All trials CSV</a></p>
<img src="comparison.png" alt="Experiment metrics"><main>''' + "".join(cards) + "</main><h2>Interpretation</h2><ul>" + notes + "</ul>"
    (output / "index.html").write_text(document)
    print("RESULTS " + str(output), flush=True)


if __name__ == "__main__":
    main()
