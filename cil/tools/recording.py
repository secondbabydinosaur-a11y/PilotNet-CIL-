"""CIL 주행 기록: 전방 BGR 영상과 실제 Simulator X11 창을 MP4로 저장한다."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

import numpy as np


COMMAND_LABELS = {
    "follow_road": (0, "FOLLOW ROAD"),
    "left": (1, "LEFT"),
    "straight": (2, "STRAIGHT"),
    "right": (3, "RIGHT"),
}
OVERLAY_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
INTERVENTION_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
STEERING_INTERVENTION_LABEL = "STEERING INTERVENTION ACTIVE"


def command_overlay_text(name, command_id, elapsed_s, reason="model", course=""):
    """실제 적용 중인 명령을 사람이 읽을 수 있는 두 줄로 표시한다."""
    if name not in COMMAND_LABELS:
        raise ValueError("unknown command name: " + str(name))
    expected_id, label = COMMAND_LABELS[name]
    if (isinstance(command_id, (bool, np.bool_))
            or not isinstance(command_id, (int, np.integer)) or command_id != expected_id):
        raise ValueError("command name and command_id do not match")
    if not np.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("elapsed_s must be finite and nonnegative")
    for key, value in (("reason", reason), ("course", course)):
        if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(key + " must be a single line of text")
    if reason == STEERING_INTERVENTION_LABEL:
        return "{}\nNAV: {} [{}] | t={:.1f}s\n".format(
            STEERING_INTERVENTION_LABEL, label, expected_id, float(elapsed_s)
        )  # 개입 중에는 두 줄 안에서 경고·진행 명령·시간을 우선 표시한다.
    return "COMMAND: {} [{}]\nCOURSE: {} | t={:.1f}s | CONTROL: {}\n".format(
        label, expected_id, course[:48] or "-", float(elapsed_s), reason[:48]
    )  # textfile와 expansion=none을 사용하므로 %나 콜론도 ffmpeg 식으로 실행되지 않는다.


def video_overlay_parts(text):
    """전체 표시문은 Tk에 제공하고, 영상의 경고만 별도 노란색 층으로 분리한다."""
    lines = text.splitlines()
    active = bool(lines and lines[0] == STEERING_INTERVENTION_LABEL)
    body = "\n" + "\n".join(lines[1:]) + "\n" if active else text
    warning = STEERING_INTERVENTION_LABEL + "\n" if active else "\n"
    return body, warning


def atomic_overlay_write(path, text):
    """동일 폴더의 임시 파일을 완성한 뒤 교체해 drawtext의 빈 파일 읽기를 막는다."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".command_overlay_", suffix=".tmp",
            dir=str(path.parent), delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)  # 같은 파일시스템 안의 rename은 원자적으로 이루어진다.
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def command_overlay_filter(text_path, intervention_path=None):
    """필터 문법과 충돌하지 않는 절대경로만 허용한다. 쉘은 사용하지 않는다."""
    text_path = Path(text_path).resolve()
    paths = [text_path, OVERLAY_FONT]
    if intervention_path is not None:
        intervention_path = Path(intervention_path).resolve()
        paths += [intervention_path, INTERVENTION_FONT]
    for path in paths:
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
            raise ValueError("overlay path must contain only ASCII letters, digits, / . _ -: " + str(path))
    if not OVERLAY_FONT.is_file():
        raise FileNotFoundError("overlay font missing: " + str(OVERLAY_FONT))
    expression = (
        "drawbox=x=0:y=0:w=iw:h=90:color=0x101827:t=fill,"
        "drawtext=fontfile={}:textfile={}:reload=1:expansion=none:"
        "fontcolor=white:fontsize=34:x=18:y=8:line_spacing=8"
    ).format(OVERLAY_FONT, text_path)
    if intervention_path is not None:
        if not INTERVENTION_FONT.is_file():
            raise FileNotFoundError("bold overlay font missing: " + str(INTERVENTION_FONT))
        expression += (
            ",drawtext=fontfile={}:textfile={}:reload=1:expansion=none:"
            "fontcolor=0xFFEA00:fontsize=min(40\\,(w-36)/20):x=18:y=6:"
            "box=1:boxcolor=0x493700@0.95:boxborderw=4:borderw=1:bordercolor=black"
        ).format(INTERVENTION_FONT, intervention_path)  # 작은 창에서도 경고가 잘리지 않도록 크기를 제한한다.
    return expression


class _Recorder:
    """ffmpeg 종료를 기다려 MP4 인덱스까지 기록하는 공통 처리."""

    def _start(self, path, arguments):
        self.path = Path(path).resolve()  # 결과 위치를 오류 메시지에도 사용한다.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():  # 이전 주행 영상을 덮어쓰지 않는다.
            raise FileExistsError(self.path)
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("x", encoding="utf-8")
        self.frames = 0
        self.closed = False
        try:
            self.process = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n"]
                + arguments + [str(self.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log,
            )  # stderr를 파일에 기록해 PIPE가 가득 차는 문제를 피한다.
        except BaseException:
            self.log.close()
            raise

    def check(self):
        """주행 루프에서도 호출하면 화면 녹화의 조기 종료를 감지할 수 있다."""
        if self.closed:
            raise RuntimeError("recorder is already closed: " + str(self.path))
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                "ffmpeg stopped early ({}); inspect {}".format(code, self.log_path)
            )

    def _finish(self, screen=False):
        if self.closed:  # finally에서 다시 호출해도 파일을 건드리지 않는다.
            return
        self.closed = True
        timed_out = False
        try:
            try:
                if screen:
                    self.process.communicate(input=b"q\n", timeout=30)
                else:
                    try:
                        self.process.stdin.close()  # EOF 이후 남은 프레임을 인코딩한다.
                    except BrokenPipeError:
                        pass  # 아래 returncode 검사에서 실제 오류를 보고한다.
                    self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.process.send_signal(signal.SIGINT)  # 먼저 MP4를 마무리할 기회를 준다.
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            finally:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
        finally:
            self.log.close()

        if timed_out or self.process.returncode:
            raise RuntimeError(
                "ffmpeg finalization failed (code={}, timeout={}); inspect {}".format(
                    self.process.returncode, timed_out, self.log_path
                )
            )
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError("recording file is missing or empty: " + str(self.path))

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames,width,height", "-of", "json", str(self.path)],
            capture_output=True, text=True, timeout=15, check=True,
        )  # 종료 후 실제로 읽을 수 있는 MP4인지 확인한다.
        streams = json.loads(result.stdout).get("streams", [])
        if not streams or int(streams[0].get("nb_frames", 0)) <= 0:
            raise RuntimeError("recording contains no readable frames: " + str(self.path))
        self.frames = int(streams[0]["nb_frames"])  # 실제 저장된 프레임 수다.


class Video(_Recorder):
    """주행 루프에서 전달하는 uint8 BGR 영상을 일정 FPS로 저장한다."""

    def __init__(self, path, width, height, hz=20):
        self.width, self.height = int(width), int(height)
        if (self.width <= 0 or self.height <= 0 or self.width % 2
                or self.height % 2 or not np.isfinite(hz) or hz <= 0):
            raise ValueError("video size must be positive/even and FPS positive")
        self._start(path, [
            "-nostdin", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self.width, self.height), "-r", str(hz),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        self.written = 0

    def write(self, image):
        self.check()  # 인코더가 죽었는데 주행 기록이 계속되는 상황을 막는다.
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError("expected uint8 BGR frame matching configured video size")
        try:
            self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("video encoder failed; inspect " + str(self.log_path)) from exc
        self.written += 1
        self.frames = self.written  # close 전에는 전달 완료한 프레임 수를 제공한다.

    def close(self):
        if self.closed:
            return
        self._finish()
        if self.frames != self.written:
            raise RuntimeError("encoded frame count differs from written frame count")


class ScreenVideo(_Recorder):
    """생성 시점의 실제 Simulator 창 영역을 X11에서 직접 녹화한다."""

    def __init__(self, path, command_overlay=False):
        self.command_overlay = bool(command_overlay)
        self.overlay_text_path = None  # 켜진 경우 영상 옆의 최신 표시문 파일을 가리킨다.
        self.overlay_video_text_path = None
        self.overlay_intervention_text_path = None
        self.overlay_style = None
        self._overlay_last_text = None
        overlay_arguments = []
        if self.command_overlay:
            overlay_path = Path(path).resolve().with_suffix(".commands.txt")
            video_text_path = overlay_path.with_suffix(".video.txt")
            intervention_path = overlay_path.with_suffix(".intervention.txt")
            overlay_filter = command_overlay_filter(video_text_path, intervention_path)  # 경로·폰트는 창 조작 전에 검사한다.
            for text_path in (overlay_path, video_text_path, intervention_path):
                if text_path.exists():
                    raise FileExistsError(text_path)
            overlay_arguments = ["-vf", overlay_filter]
        display = os.environ.get("DISPLAY")
        if not display:
            raise RuntimeError("DISPLAY is missing; actual Simulator window recording unavailable")
        try:
            ids = subprocess.check_output(
                ["xdotool", "search", "--onlyvisible", "--name", "^Simulator$"],
                text=True, timeout=5,
            ).split()  # 해당 창이 없으면 다른 화면으로 대체하지 않는다.
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("visible Simulator X11 window was not found") from exc
        if not ids:
            raise RuntimeError("visible Simulator X11 window was not found")
        self.window_id = ids[-1]
        focus = subprocess.run(
            ["xdotool", "windowactivate", "--sync", self.window_id],
            capture_output=True, timeout=5,
        )  # 녹화 시작 시점에만 창을 앞으로 가져온다.
        if focus.returncode != 0:
            subprocess.run(["xdotool", "windowraise", self.window_id], check=True, timeout=5)
        geometry = subprocess.check_output(
            ["xwininfo", "-id", self.window_id], text=True, timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
        values = {key.strip(): int(value) for key, value in re.findall(
            r"^ *([A-Za-z -]+):\s*(-?\d+)\s*$", geometry, re.MULTILINE,
        )}
        x, y = values["Absolute upper-left X"], values["Absolute upper-left Y"]
        width, height = values["Width"] // 2 * 2, values["Height"] // 2 * 2
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise RuntimeError("Simulator window has an unsupported screen geometry")
        self.geometry = dict(x=x, y=y, width=width, height=height, display=display)
        if self.command_overlay:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            waiting = "COMMAND: WAITING\nCOURSE: - | t=0.0s | CONTROL: waiting\n"
            atomic_overlay_write(overlay_path, waiting)
            atomic_overlay_write(video_text_path, waiting)
            atomic_overlay_write(intervention_path, "\n")
            self.overlay_text_path = overlay_path
            self.overlay_video_text_path = video_text_path
            self.overlay_intervention_text_path = intervention_path
            self.overlay_style = dict(
                banner_height_px=90, font=str(OVERLAY_FONT), font_size_px=34,
                position="top 90px of actual Simulator capture", text_reload_frames=1,
                text_expansion="none", source="live command selected by driving loop",
                intervention_label=STEERING_INTERVENTION_LABEL, intervention_font=str(INTERVENTION_FONT),
                intervention_color="#FFEA00", intervention_background="#493700",
            )  # CNN 입력이나 차량 명령을 바꾸지 않는 표시 전용 설정이다.
        self._start(path, [
            "-f", "x11grab", "-draw_mouse", "0", "-framerate", "20",
            "-video_size", "{}x{}".format(width, height),
            "-i", "{}+{},{}".format(display, x, y),
        ] + overlay_arguments + [
            "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "21", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ])
        try:
            time.sleep(0.35)  # 접속 직후 인코더가 실패하면 출발 전에 감지한다.
            self.check()
        except BaseException:
            try:
                self._finish(screen=True)  # 생성 중 실패해도 녹화 프로세스를 남기지 않는다.
            except Exception:
                pass  # 최초 시작 실패 원인을 유지한다.
            raise

    def update_command(self, name, command_id, elapsed_s, reason="model", course=""):
        """주행 루프의 현재 명령만 녹화 자막으로 갱신한다."""
        self.update_overlay(command_overlay_text(name, command_id, elapsed_s, reason=reason, course=course))

    def update_overlay(self, text):
        """두 줄의 주행 상태를 영상 및 실시간 배너에 전달한다."""
        self.check()  # 녹화가 먼저 종료되었으면 주행 루프에서 즉시 알 수 있다.
        if not self.command_overlay:
            raise RuntimeError("command overlay was not enabled for this screen recording")
        if text != self._overlay_last_text:
            if getattr(self, "overlay_video_text_path", None) is not None:
                body, warning = video_overlay_parts(text)
                atomic_overlay_write(self.overlay_video_text_path, body)
                atomic_overlay_write(self.overlay_intervention_text_path, warning)
            atomic_overlay_write(self.overlay_text_path, text)
            self._overlay_last_text = text  # 표시 문구가 같으면 불필요한 디스크 쓰기를 생략한다.

    def close(self):
        self._finish(screen=True)  # q로 종료해 MP4 인덱스까지 저장한다.
