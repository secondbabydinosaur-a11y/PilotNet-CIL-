"""두 실제 MORAI 화면을 자르지 않고 나란히 보여 주는 CPU 전용 비교 영상 도구.

예: python3 cil/tools/compare_morai_videos.py --left old/morai_screen.mp4 \
    --right new/morai_screen.mp4 --output comparison.mp4 \
    --left-training-seconds 1104 --right-training-seconds 5400 \
    --left-result '경로 이탈로 중단' --right-result '종점 도달'

시작 정렬이 필요하면 각 원본에서 주행이 시작되는 영상 시각을 확인해
--left-start-s / --right-start-s로 지정한다. 기록에 연결 시각이 없으므로
도구가 녹화 시작과 주행 시작이 같다고 추정하지는 않는다.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata


FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("유한한 0 이상의 수를 입력하세요.")
    return number


def probe(path):
    """ffprobe로 읽을 수 있는 첫 번째 영상의 크기와 실제 길이를 확인한다."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,duration,nb_frames:format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=30, check=True)
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError("영상 스트림이 없습니다: " + str(path))
    stream = info["streams"][0]
    duration = float(stream.get("duration", info.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("영상 길이가 올바르지 않습니다: " + str(path))
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration_s": duration}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_label(seconds):
    """표시 시간은 학습 연산시간이 아니라 학습에 사용한 주행 데이터의 길이다."""
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return ("{}시간 ".format(hours) if hours else "") + "{}분 {:02d}초".format(minutes, seconds)


def checked_text(value):
    if not value or len(value) > 160 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("표시문은 1~160자의 한 줄 문자열이어야 합니다.")
    return value


def text_filter(path, text, width, y, color="white", enable=None):
    """사용자 문구는 textfile로 전달하므로 콜론·퍼센트·따옴표도 안전하다."""
    # 한글은 넓은 글자, 영문은 약 0.7칸으로 계산해 긴 제목도 패널 안에 맞춘다.
    units = sum(1.0 if unicodedata.east_asian_width(char) in "WF" else .7 for char in text)
    size = min(38 if y == 10 else 30, max(14, math.floor((width - 48) / max(1, units))))
    if units * 14 > width - 48:
        raise ValueError("패널에 비해 문구가 너무 깁니다. 문구를 줄이거나 --panel-width를 늘리세요.")
    expression = ("drawtext=fontfile={}:textfile={}:expansion=none:fontcolor={}:"
                  "fontsize={}:x=24:y={}").format(FONT, path, color, size, y)
    if enable:
        expression += ":enable='" + enable + "'"
    return expression


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True, help="왼쪽 실제 MORAI 화면 MP4")
    parser.add_argument("--right", type=Path, required=True, help="오른쪽 실제 MORAI 화면 MP4")
    parser.add_argument("--output", type=Path, required=True, help="새 비교 MP4; 기존 파일은 덮어쓰지 않음")
    for side, name in (("left", "기존 모델"), ("right", "추가 데이터 모델")):
        parser.add_argument("--" + side + "-label", default=name, help="해당 패널의 모델 이름")
        parser.add_argument("--" + side + "-training-seconds", type=nonnegative,
                            help="학습에 사용한 유효 주행 데이터 길이(초); 실제 집계값 입력")
        parser.add_argument("--" + side + "-start-s", type=nonnegative, default=0.,
                            help="해당 원본의 주행 시작 시각(초); 이 시점들을 서로 맞춰 재생")
        parser.add_argument("--" + side + "-result", default="",
                            help="끝난 뒤 표시할 실제 결과 문구; 도구는 성공 여부를 추정하지 않음")
    parser.add_argument("--panel-width", type=int, default=0,
                        help="각 패널의 폭. 0이면 두 원본 중 큰 폭을 사용해 원본 자막 크기 유지")
    parser.add_argument("--fps", type=int, default=20, help="출력 FPS; 재생 배속은 바꾸지 않음 (기본 20)")
    parser.add_argument("--end-hold-s", type=nonnegative, default=2., help="긴 영상도 끝난 뒤 결과를 보여 줄 시간")
    parser.add_argument("--crf", type=int, default=18, help="H.264 화질 0~51; 낮을수록 고화질 (기본 18)")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg와 ffprobe가 필요합니다.")
    if not 1 <= args.fps <= 60 or not 0 <= args.crf <= 51 or args.panel_width < 0:
        raise ValueError("FPS는 1~60, CRF는 0~51, panel-width는 0 이상이어야 합니다.")
    global FONT
    FONT = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if FONT is None:
        raise FileNotFoundError("Noto Sans CJK 또는 DejaVu Sans Bold 폰트가 필요합니다.")
    sources = [args.left.resolve(), args.right.resolve()]
    output = args.output.resolve()
    report_path = output.with_suffix(".comparison.json")
    if output.suffix.lower() != ".mp4" or output in sources or output.exists() or report_path.exists():
        raise ValueError("원본과 다른 새 .mp4 경로를 사용하세요. 기존 영상·보고서는 덮어쓰지 않습니다.")
    infos = [probe(path) for path in sources]
    width = args.panel_width or max(info["width"] for info in infos)
    width += width % 2  # yuv420p와 H.264 출력의 가로·세로 크기를 짝수로 맞춘다.
    if width < 640:
        raise ValueError("원본 자막을 읽을 수 있도록 패널 폭은 640px 이상이어야 합니다.")
    height = math.ceil(max(info["height"] * width / info["width"] for info in infos))
    height += height % 2
    offsets = [args.left_start_s, args.right_start_s]
    durations = [info["duration_s"] - start for info, start in zip(infos, offsets)]
    if min(durations) <= 1. / args.fps:
        raise ValueError("시작 시각 뒤에 최소 한 프레임 이상의 영상이 남아야 합니다.")
    total = math.ceil((max(durations) + args.end_hold_s) * args.fps) / args.fps
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptions = []
    # 임시 텍스트는 ASCII 경로를 사용하므로 ffmpeg 필터 안에서 경로를 재해석하지 않는다.
    with tempfile.TemporaryDirectory(prefix="cil_video_compare_") as directory:
        temporary = Path(directory)
        filters = []
        for index, side in enumerate(("left", "right")):
            label = checked_text(getattr(args, side + "_label"))
            seconds = getattr(args, side + "_training_seconds")
            if seconds is not None:
                label += " | 학습 주행 데이터 " + duration_label(seconds)
            result = getattr(args, side + "_result")
            final = "결과: " + checked_text(result) if result else "영상 종료 · 마지막 프레임 유지"
            texts = (label, "주행 영상 재생 중", final)
            paths = []
            for number, text in enumerate(texts):
                path = temporary / "{}_{}.txt".format(side, number)
                path.write_text(text, encoding="utf-8")
                paths.append(path)
            duration = durations[index]
            graph = ("[{}:v]trim=start={:.9f},setpts=PTS-STARTPTS,fps={},"
                     "scale={}:{}:force_original_aspect_ratio=decrease:flags=lanczos,"
                     "pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                     "tpad=stop_mode=clone:stop_duration={:.9f},trim=duration={:.9f},"
                     "pad={}:{}:0:112:color=0x101827,").format(
                         index, offsets[index], args.fps, width, height, width, height,
                         total - duration, total, width, height + 112)
            graph += text_filter(paths[0], texts[0], width, 10)
            graph += "," + text_filter(paths[1], texts[1], width, 65, enable="lt(t\\,{:.9f})".format(duration))
            graph += "," + text_filter(paths[2], texts[2], width, 65, "0xFFEA00",
                                       enable="gte(t\\,{:.9f})".format(duration))
            if index == 0:
                graph += ",pad={}:{}:0:0:color=0x64748B".format(width + 8, height + 112)
            graph += "[{}]".format(side)
            filters.append(graph)
            descriptions.append({"source": str(sources[index]), "sha256": sha256(sources[index]),
                "source_video": infos[index], "label": label, "training_data_seconds": seconds,
                "trim_start_s": offsets[index], "retained_duration_s": duration,
                "freeze_duration_s": total - duration, "supplied_result": result or None})
        filters.append("[left][right]hstack=inputs=2:shortest=1,format=yuv420p[out]")
        # 완성되기 전의 MP4는 별도 폴더에 두어 실패한 인코딩을 결과로 남기지 않는다.
        with tempfile.TemporaryDirectory(prefix=".comparison_", dir=str(output.parent)) as staging:
            encoded = Path(staging) / "comparison.mp4"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
            for source in sources:
                command += ["-threads", "2", "-i", str(source)]
            command += ["-filter_complex_threads", "2", "-filter_complex", ";".join(filters),
                        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
                        "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)]
            subprocess.run(command, check=True)
            actual = probe(encoded)
            if abs(actual["duration_s"] - total) > 2. / args.fps:
                raise RuntimeError("출력 길이가 마지막 프레임 유지 계획과 다릅니다.")
            report = {"schema_version": 1, "output": str(output), "video": actual,
                "left": descriptions[0], "right": descriptions[1],
                "alignment": "explicit source start offsets; no automatic event alignment",
                "playback_speed": 1.0, "original_overlays": "preserved; full frame, no crop or repaint",
                "shorter_video_policy": "freeze final source frame; continue longer video",
                "audio": "omitted", "output_sha256": sha256(encoded),
                "interpretation": "각 1회 영상 비교이며 통계적 성공률이나 알고리즘 단독 효과를 의미하지 않음"}
            if output.exists() or report_path.exists():
                raise FileExistsError("인코딩 중 같은 결과 경로가 생성되었습니다.")
            encoded.replace(output)
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    print(output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
