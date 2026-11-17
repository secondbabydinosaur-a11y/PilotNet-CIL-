"""모델 계약·곡률 부호·실시간 수신을 확인한다. 실제 MORAI에는 송신하지 않는다."""
import copy
import math
from pathlib import Path
import socket
import tempfile
import time
import unittest

import numpy as np
import torch

from data_utils import BASE, read_json, steering_to_curvature
from dataset import PilotNetDataset
from learning.common import data_contract, load_checkpoint, regression_metrics, save_checkpoint
from learning.live import LatestSensors, Reference, curvature_to_degrees, speed_command
from learning.data import curve_sampler, sampling_curvature
from model import PilotNet, PixelNormalization
from test_pipeline import camera_packets, ego_packet, image_bytes


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_input_normalization_output_and_backward(self):
        self.assertTrue(torch.equal(PixelNormalization()(torch.tensor([0., 127.5, 255.])),
                                    torch.tensor([-1., 0., 1.])))
        model = PilotNet()
        output = model(torch.zeros(2, 3, 66, 200))
        self.assertEqual(output.shape, (2, 1))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 200, 66))

    def test_checkpoint_roundtrip(self):
        model = PilotNet().eval()
        image = torch.full((1, 3, 66, 200), 100.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, model, data_contract=data_contract(), epoch=1)
            restored, _ = load_checkpoint(path, torch.device("cpu"), check_data=True)
            torch.testing.assert_close(model(image), restored(image))

    def test_curvature_inverse_and_metrics(self):
        contract = dict(wheelbase_m=3., steer_left_sign=1, road_wheel_gain=1.)
        for angle in (-20., -3., 0., 3., 20.):
            self.assertAlmostEqual(curvature_to_degrees(steering_to_curvature(angle, 3.), contract), angle)
        metrics = regression_metrics([-.02, 0., .02], [-.02, 0., .02])
        self.assertEqual(metrics["rmse_1pm"], 0.)
        self.assertEqual(metrics["r2"], 1.)
        with self.assertRaises(ValueError):
            regression_metrics([1.], [float("nan")])

    def test_longitudinal_limit_and_reference(self):
        for speed in (0., 5., 19., 19.8, 21.):
            throttle, brake = speed_command(speed, 19.)
            self.assertFalse(throttle > 0 and brake > 0)
            if speed >= 19.8:
                self.assertEqual(throttle, 0.)
                self.assertGreater(brake, 0.)
        for name in Reference.limits:
            ref = Reference(name)
            x, y, _, yaw = ref.pose(.3)
            _, cte, heading = ref.project(dict(x_m=x, y_m=y, yaw_deg=yaw))
            self.assertAlmostEqual(cte, .3, places=2)
            self.assertLess(abs(heading), 1.)

    def test_heading_reset_rotates_without_moving_position(self):
        for name in Reference.limits:
            ref = Reference(name)
            nominal = ref.pose(.3)
            for yaw in (-5., 5.):
                tilted = ref.pose(.3, yaw)
                np.testing.assert_allclose(tilted[:3], nominal[:3])
                station, cte, heading = ref.project(dict(x_m=tilted[0], y_m=tilted[1], yaw_deg=tilted[3]))
                self.assertAlmostEqual(cte, .3, places=2)
                self.assertAlmostEqual(heading, yaw, places=1)
                self.assertLess(abs(station - ref.start), .02)
        train_ranges = ((0, 65), (640, 866))
        for name in ("U1", "U2"):
            start, end = Reference.limits[name]
            self.assertGreaterEqual(end - start, 200.)
            for low, high in train_ranges:
                self.assertFalse(start <= high and end >= low)

    def test_three_cameras_are_sampled_separately(self):
        if not (BASE / "dataset/paired_three/train.csv").exists():
            self.skipTest("side labels not built")
        center = PilotNetDataset("train")
        three = PilotNetDataset("train", BASE / "dataset/paired_three", allow_side=True)
        self.assertEqual(sum(r["camera"] == "center" for r in three.rows), len(center))
        self.assertEqual({r["camera"] for r in three.rows}, {"left", "center", "right"})
        self.assertTrue(all(r["group"] in ("expert_g01", "expert_g02", "expert_g03", "expert_g04") for r in three.rows))
        side_index = next(i for i, r in enumerate(three.rows) if r["camera"] == "left")
        if not (BASE / three.rows[side_index]["image_path"]).exists():
            self.skipTest("raw JPEG dataset is retained locally, not in the Git repository")
        image, target = three[side_index]
        self.assertEqual(tuple(image.shape), (3, 66, 200))
        self.assertTrue(torch.isfinite(target).all().item())

    def test_side_recovery_is_not_used_as_road_curve_proxy(self):
        class Samples:
            rows = [dict(camera="left", y_curvature_1pm=".02", center_curvature_1pm=".0001"),
                    dict(camera="right", y_curvature_1pm=".001", center_curvature_1pm=".02")]
        source = Samples()
        self.assertEqual(sampling_curvature(source.rows[0], "target"), .02)
        sampler = curve_sampler(source, 2., 42, basis="center")
        self.assertEqual(sampler.weights.tolist(), [1., 2.])
        # 정답은 그대로 두고 선택 확률만 바뀐다.
        self.assertEqual(source.rows[0]["y_curvature_1pm"], ".02")
        with self.assertRaises(ValueError):
            sampling_curvature(dict(camera="left", y_curvature_1pm=".02"), "center")

    def test_latest_sensors_with_synthetic_udp(self):
        cfg = copy.deepcopy(read_json(BASE / "config.json"))
        reservations = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
        for sock in reservations:
            sock.bind(("127.0.0.1", 0))
        cfg["bind_ip"] = "127.0.0.1"
        cfg["cameras"]["center"]["port"] = reservations[0].getsockname()[1]
        cfg["ego"]["port"] = reservations[1].getsockname()[1]
        for sock in reservations:
            sock.close()
        receiver = LatestSensors(cfg)
        stamp = 1_000_000_000
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(ego_packet(stamp), ("127.0.0.1", cfg["ego"]["port"]))
                for packet in camera_packets(image_bytes(), stamp, 60000):
                    sender.sendto(packet, ("127.0.0.1", cfg["cameras"]["center"]["port"]))
                    time.sleep(.001)
                deadline = time.monotonic() + 2.
                while time.monotonic() < deadline:
                    state, _, frame = receiver.snapshot()
                    if state and frame:
                        break
                    time.sleep(.01)
                self.assertIsNotNone(frame)
                self.assertEqual(state["stamp_ns"], stamp)
                self.assertEqual(frame[0], stamp)
        finally:
            receiver.close()


if __name__ == "__main__":
    unittest.main()
