"""CIL 주행 기록: 전방 BGR 영상과 실제 Simulator X11 창을 MP4로 저장한다."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np


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

    def __init__(self, path):
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
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", self.window_id],
            check=True, timeout=5,
        )  # 녹화 시작 시점에만 창을 앞으로 가져온다.
        geometry = subprocess.check_output(
            ["xdotool", "getwindowgeometry", "--shell", self.window_id],
            text=True, timeout=5,
        )
        values = dict(line.split("=", 1) for line in geometry.splitlines() if "=" in line)
        x, y = int(values["X"]), int(values["Y"])
        width, height = int(values["WIDTH"]) // 2 * 2, int(values["HEIGHT"]) // 2 * 2
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise RuntimeError("Simulator window has an unsupported screen geometry")
        self.geometry = dict(x=x, y=y, width=width, height=height, display=display)
        self._start(path, [
            "-f", "x11grab", "-draw_mouse", "0", "-framerate", "20",
            "-video_size", "{}x{}".format(width, height),
            "-i", "{}+{},{}".format(display, x, y), "-an", "-c:v", "libx264",
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

    def close(self):
        self._finish(screen=True)  # q로 종료해 MP4 인덱스까지 저장한다.
