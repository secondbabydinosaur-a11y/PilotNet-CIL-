"""Governor/controller tests; outbound packets go only to ephemeral fake endpoints."""
from collections import deque
import socket
import struct
import threading
import time
import unittest

from data_utils import BASE, HumanCommands, load_config
from tools.collection.manual_control import ControlWorker, Governor, encode_command


def config():
    return load_config(BASE / "config.json")["manual_control"]


def status_at(speed):
    return {"speed_kph": speed, "ctrl_mode": 2}


class GovernorTests(unittest.TestCase):
    def test_wire_format_and_mutually_exclusive_pedals(self):
        packet = encode_command(.2, 0., -.1)
        self.assertEqual(len(packet), 55)
        self.assertEqual(packet[:14], b"#MoraiCtrlCmd$")
        self.assertEqual(struct.unpack_from("<I", packet, 14)[0], 23)
        fields = struct.unpack_from("<BBB5f", packet, 30)
        self.assertEqual(fields[:3], (2, 4, 1))
        self.assertAlmostEqual(fields[5], .2)
        self.assertAlmostEqual(fields[7], -.1)
        self.assertEqual(packet[-2:], b"\r\n")
        with self.assertRaises(ValueError):
            encode_command(.1, .1, 0.)

    def test_state_loss_focus_loss_and_rearming(self):
        controller = Governor(config())
        status = status_at(0.)
        self.assertEqual(controller.step({"forward"}, True, None, 100., .02)["accel"], 0.)
        self.assertEqual(controller.step({"forward"}, True, status, 0., .02)["reason"], "release_forward_to_arm")
        controller.step(set(), True, status, 0., .02)
        self.assertGreater(controller.step({"forward"}, True, status, 0., .02)["accel"], 0.)
        self.assertEqual(controller.step({"forward"}, False, status, 0., .02)["brake"], 1.)
        self.assertEqual(controller.step({"forward"}, True, status, 0., .02)["accel"], 0.)
        controller.step(set(), True, status, 0., .02)
        self.assertEqual(controller.step({"forward"}, True, status, .3, .02)["reason"], "stale_ego")
        self.assertEqual(controller.step({"forward"}, True, status, 0., .02)["accel"], 0.)

    def test_brake_priority_and_steering_bounds(self):
        controller = Governor(config())
        status = status_at(10.)
        controller.step(set(), True, status, 0., .02)
        for _ in range(200):
            row = controller.step({"forward", "left"}, True, status, 0., .02)
        self.assertLessEqual(abs(row["steer_norm"]), .5)
        row = controller.step({"forward", "back"}, True, status, 0., .02)
        self.assertEqual(row["accel"], 0.)
        self.assertEqual(row["brake"], 1.)
        row = controller.step({"forward"}, True, status_at(25.), 0., .02)
        self.assertEqual(row["accel"], 0.)
        self.assertGreaterEqual(row["brake"], .4)

    def test_sustained_forward_with_delayed_vehicle_feedback(self):
        # Simple independent longitudinal plant with actuator delay, drag, and
        # low/high response gains. This validates logic, not MORAI calibration.
        for gain in (4., 6.):
            controller = Governor(config())
            controller.step(set(), True, status_at(0.), 0., .02)
            speed = 0.
            delay = deque([(0., 0.)] * 6)
            observed = []
            for _ in range(2000):
                row = controller.step({"forward"}, True, status_at(speed), .02, .02)
                delay.append((row["accel"], row["brake"]))
                accel, brake = delay.popleft()
                speed = max(0., speed + (gain * accel - 8. * brake - .12) * .02 * 3.6)
                observed.append(speed)
            self.assertLessEqual(max(observed), 20.)
            self.assertGreater(speed, 18.)
            self.assertLess(speed, 19.5)

    def test_native_keyboard_mode_does_not_get_throttle(self):
        controller = Governor(config())
        controller.step(set(), True, status_at(0.), 0., .02)
        row = controller.step({"forward"}, True, {"speed_kph": 0., "ctrl_mode": 1}, 0., .02)
        self.assertEqual(row["accel"], 0.)
        self.assertEqual(row["reason"], "wrong_mode_select_AV_ExternalCtrl")

    def test_command_provenance_clock_and_focus(self):
        rows = [{"recv_ns": 1_000_000_000, "sent": 1, "focused": 1, "ego_fresh": 1, "reason": "drive"},
                {"recv_ns": 1_100_000_000, "sent": 1, "focused": 0, "ego_fresh": 1, "reason": "focus_lost"}]
        commands = HumanCommands(rows)
        self.assertFalse(commands.valid_at(999_999_999))
        self.assertTrue(commands.valid_at(1_050_000_000))
        self.assertFalse(commands.valid_at(1_150_000_000))
        self.assertFalse(commands.valid_at(9_000_000_000))


class FakeKeys:
    def __init__(self):
        self.closed = False

    def poll(self):
        return set(), True  # Full brake, zero throttle throughout integration.

    def close(self):
        self.closed = True


class ControlNetworkTests(unittest.TestCase):
    def test_no_state_no_send_then_brake_and_clean_shutdown(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(.1)
            cfg = config()
            cfg.update(sim_ip="127.0.0.1", sim_ctrl_host_port=receiver.getsockname()[1], shutdown_brake_s=.05)
            ready = threading.Event()
            snapshot = lambda: (status_at(0.), time.monotonic_ns()) if ready.is_set() else (None, None)
            queued, errors, keys = [], [], FakeKeys()
            worker = ControlWorker(cfg, snapshot, queued.append, errors.append, key_source=keys)
            worker.start()
            try:
                with self.assertRaises(socket.timeout):
                    receiver.recvfrom(1000)
                ready.set()
                packet, _ = receiver.recvfrom(1000)
                values = struct.unpack_from("<BBB5f", packet, 30)
                self.assertEqual(values[:3], (2, 4, 1))
                self.assertEqual(values[5], 0.)
                self.assertGreater(values[6], 0.)
            finally:
                worker.close()
            self.assertFalse(worker.thread.is_alive())
            self.assertTrue(keys.closed)
            self.assertFalse(errors)
            self.assertTrue(any(row[1]["sent"] == 0 for row in queued))
            self.assertTrue(any(row[1]["sent"] == 1 for row in queued))


if __name__ == "__main__":
    unittest.main()
