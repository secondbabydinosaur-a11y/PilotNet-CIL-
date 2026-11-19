"""Simulator를 열거나 차량을 제어하지 않고 녹화 자막과 ffmpeg 필터를 검사한다."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.recording import (
    ScreenVideo, atomic_overlay_write, command_overlay_filter, command_overlay_text,
    STEERING_INTERVENTION_LABEL, video_overlay_parts,
)
from tools.live_command_banner import LiveCommandBanner


class RecordingOverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cil_overlay_test_")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "screen.commands.txt"

    def test_command_names_ids_and_elapsed_are_validated(self):
        expected = (("follow_road", 0, "FOLLOW ROAD"), ("left", 1, "LEFT"),
                    ("straight", 2, "STRAIGHT"), ("right", 3, "RIGHT"))
        for name, number, label in expected:
            text = command_overlay_text(name, number, 12.34, "model", "COMPETITION")
            self.assertEqual(text.splitlines()[0], "COMMAND: {} [{}]".format(label, number))
            self.assertIn("COURSE: COMPETITION | t=12.3s | CONTROL: model", text)
        for name, number, elapsed in (("unknown", 0, 1), ("left", 3, 1),
                                      ("left", True, 1), ("left", 1, -1),
                                      ("left", 1, float("nan")), ("left", 1, float("inf"))):
            with self.subTest(name=name, number=number, elapsed=elapsed), self.assertRaises(ValueError):
                command_overlay_text(name, number, elapsed)
        with self.assertRaises(ValueError):
            command_overlay_text("left", 1, 0, course="first\nsecond")

    def test_replace_is_atomic_and_failed_replace_keeps_old_text(self):
        self.path.write_text("previous complete text", encoding="utf-8")
        replace = Path.replace
        observed = []

        def inspect_replace(source, destination):
            observed.append((self.path.read_text(), source.read_text()))
            return replace(source, destination)

        with patch.object(Path, "replace", autospec=True, side_effect=inspect_replace):
            atomic_overlay_write(self.path, "new complete text")
        self.assertEqual(observed, [("previous complete text", "new complete text")])
        self.assertEqual(self.path.read_text(), "new complete text")
        with patch.object(Path, "replace", side_effect=OSError("simulated rename failure")):
            with self.assertRaises(OSError):
                atomic_overlay_write(self.path, "must not become visible")
        self.assertEqual(self.path.read_text(), "new complete text")
        self.assertFalse(list(self.path.parent.glob(".command_overlay_*.tmp")))

    def test_update_checks_process_and_writes_only_changed_display_text(self):
        screen = ScreenVideo.__new__(ScreenVideo)  # 실제 X11 창에는 접근하지 않는다.
        screen.check = Mock()
        screen.command_overlay = True
        screen.overlay_text_path = self.path
        screen._overlay_last_text = None
        with patch("tools.recording.atomic_overlay_write", wraps=atomic_overlay_write) as write:
            screen.update_command("left", 1, 1.11, course="J_left")
            screen.update_command("left", 1, 1.12, course="J_left")
            self.assertEqual(write.call_count, 1)  # 둘 다 화면에는 t=1.1s로 표시된다.
            screen.update_command("straight", 2, 1.12, reason="speed_cap", course="J_straight")
            self.assertEqual(write.call_count, 2)
        self.assertEqual(screen.check.call_count, 3)
        self.assertIn("COMMAND: STRAIGHT [2]", self.path.read_text())
        screen.check.side_effect = RuntimeError("encoder stopped")
        with self.assertRaisesRegex(RuntimeError, "encoder stopped"):
            screen.update_command("right", 3, 2)

    def test_filter_disables_expansion_and_rejects_filter_syntax_in_paths(self):
        expression = command_overlay_filter(self.path)
        self.assertIn("h=90", expression)
        self.assertIn("reload=1:expansion=none", expression)
        self.assertIn("fontsize=34", expression)
        for filename in ("bad:name.txt", "bad'name.txt", "bad,name.txt", "has space.txt"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                command_overlay_filter(self.path.parent / filename)

    def test_intervention_keeps_navigation_and_time_in_two_lines(self):
        text = command_overlay_text("left", 1, 24.56, STEERING_INTERVENTION_LABEL, "EAST_OUTER")
        self.assertEqual(text.splitlines(), [STEERING_INTERVENTION_LABEL, "NAV: LEFT [1] | t=24.6s"])
        body, warning = video_overlay_parts(text)
        self.assertEqual(body.splitlines(), ["", "NAV: LEFT [1] | t=24.6s"])
        self.assertEqual(warning.strip(), STEERING_INTERVENTION_LABEL)
        normal = command_overlay_text("left", 1, 25, "model", "EAST_OUTER")
        self.assertEqual(video_overlay_parts(normal), (normal, "\n"))

    def test_screen_sidecars_and_tk_colors_return_to_normal(self):
        screen = ScreenVideo.__new__(ScreenVideo)  # X11·실제 Simulator에는 접근하지 않는다.
        screen.check = Mock()
        screen.command_overlay = True
        screen.overlay_text_path = self.path
        screen.overlay_video_text_path = self.path.with_suffix(".video.txt")
        screen.overlay_intervention_text_path = self.path.with_suffix(".intervention.txt")
        screen._overlay_last_text = None
        screen.update_command("left", 1, 10, reason=STEERING_INTERVENTION_LABEL)
        self.assertEqual(screen.overlay_intervention_text_path.read_text().strip(), STEERING_INTERVENTION_LABEL)
        self.assertTrue(screen.overlay_video_text_path.read_text().startswith("\nNAV: LEFT"))

        banner = LiveCommandBanner.__new__(LiveCommandBanner)  # Tk 창 대신 위젯 모의 객체만 사용한다.
        banner.args = SimpleNamespace(text_file=self.path, width=640)
        banner.root, banner.main, banner.detail = Mock(), Mock(), Mock()
        banner.main_font, banner.detail_font = Mock(), Mock()
        banner.main_font.measure.side_effect = lambda text: len(text) * 10
        banner.detail_font.measure.side_effect = lambda text: len(text) * 8
        banner.stopping, banner.last_text = False, None
        banner.refresh()
        self.assertEqual(banner.main.configure.call_args.kwargs["fg"], "#FFEA00")
        self.assertEqual(banner.main.configure.call_args.kwargs["bg"], "#493700")
        self.assertIn("NAV: LEFT [1] | t=10.0s", banner.detail.configure.call_args.kwargs["text"])
        screen.update_command("straight", 2, 11, reason="model")
        self.assertEqual(screen.overlay_intervention_text_path.read_text(), "\n")
        banner.refresh()
        self.assertEqual(banner.main.configure.call_args.kwargs["fg"], "white")
        self.assertEqual(banner.main.configure.call_args.kwargs["bg"], "#101827")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_ffmpeg_renders_bold_yellow_intervention(self):
        body_path = self.path.with_suffix(".video.txt")
        warning_path = self.path.with_suffix(".intervention.txt")
        images = []
        for reason in (STEERING_INTERVENTION_LABEL, "model"):
            body, warning = video_overlay_parts(command_overlay_text("left", 1, 24.5, reason))
            atomic_overlay_write(body_path, body)
            atomic_overlay_write(warning_path, warning)
            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "lavfi", "-i", "color=c=white:s=640x240:r=1:d=1",
                "-vf", command_overlay_filter(body_path, warning_path), "-frames:v", "1", "-an",
                "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], capture_output=True, timeout=10, check=True)
            images.append(np.frombuffer(result.stdout, np.uint8).reshape(240, 640, 3))
        active, normal = images
        yellow = lambda image: (image[:, :, 0] > 170) & (image[:, :, 1] > 150) & (image[:, :, 2] < 120)
        self.assertGreater(np.count_nonzero(yellow(active[:48])), 1000)
        self.assertEqual(np.count_nonzero(yellow(normal[:48])), 0)
        self.assertGreater(np.count_nonzero(np.all(active[45:88] > 210, axis=2)), 300)
        self.assertTrue(np.all(active[100:] > 250))  # 경고는 배너만 바꾸고 실제 장면을 가리지 않는다.

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_ffmpeg_reloads_live_text_on_synthetic_scene(self):
        atomic_overlay_write(self.path, command_overlay_text("follow_road", 0, 0))
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-re",
            "-f", "lavfi", "-i", "color=c=white:s=640x240:r=10:d=1.2",
            "-vf", command_overlay_filter(self.path), "-an", "-threads", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(.4)
            atomic_overlay_write(self.path, command_overlay_text("left", 1, .4, reason="100% model"))
            output, error = process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, error.decode(errors="replace"))
        frames = np.frombuffer(output, np.uint8).reshape(-1, 240, 640, 3)
        self.assertGreaterEqual(len(frames), 8)
        self.assertGreater(np.count_nonzero(frames[0, :85] > 210), 500)  # 배너 위 흰 글자가 보인다.
        self.assertLess(frames[-1, 88].mean(), 100)  # 위쪽 90px만 어둡게 표시한다.
        self.assertGreater(frames[-1, 120:].mean(), 250)  # 실제 장면 아래쪽은 그대로 남는다.
        self.assertGreater(np.mean(np.abs(frames[0, :85].astype(float) - frames[-1, :85])), 1)


if __name__ == "__main__":
    unittest.main()
