"""수집 정답 분리·재현 가능한 외란·명령 조건·파일 보존을 검사한다. MORAI는 조작하지 않는다."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect import Route, SessionWriter, SteeringNoise, COMMAND_FIELDS
from morai import BASE, encode_control
from tools.inspect_collection import match_fraction


class CollectionTests(unittest.TestCase):
    def test_image_bytes_and_writer_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            writer = SessionWriter(output)
            event = dict(kind="camera", camera="center", stamp_ns=123, recv_ns=456, jpeg=b"\xff\xd8raw\xff\xd9")
            writer.add(event)
            writer.close()
            self.assertEqual((output / "images/center/123.jpg").read_bytes(), event["jpeg"])
            with (output / "images.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["stamp_ns"], "123")
            broken = SessionWriter(output / "other")
            broken.add(event)
            broken.add(event)
            with self.assertRaises(FileExistsError): broken.close()

    def test_noisy_command_is_not_the_teacher_label(self):
        teacher = -0.12
        added = 0.08
        actual = teacher + added
        self.assertNotEqual(actual, teacher)
        self.assertIn("teacher_steer", COMMAND_FIELDS)
        self.assertIn("applied_steer", COMMAND_FIELDS)
        self.assertEqual(len(encode_control(actual, 0.1, 0.0)), 55)
        with self.assertRaises(ValueError): encode_control(0, 0.1, 0.1)

    def test_noise_reproducibility_and_cancel(self):
        first, second = SteeringNoise(True, 19), SteeringNoise(True, 19)
        for t in np.arange(0, 60, .02):
            self.assertEqual(first.value(t, True), second.value(t, True))
        noise = SteeringNoise(True, 7)
        noise.value(8.0, True)
        peak, episode, recovery = noise.value(9.0, True)
        self.assertGreater(abs(peak), 0.01)
        self.assertLessEqual(abs(peak), .1)
        self.assertTrue(recovery)
        cancelled, same_episode, recovering = noise.value(9.1, False)
        self.assertEqual(cancelled, 0.0)
        self.assertEqual(same_episode, episode)
        self.assertTrue(recovering)
        self.assertEqual(SteeringNoise(False).value(10, True), (0.0, 0, False))

    def test_same_approach_has_three_distinct_commands(self):
        plan = json.loads((BASE / "routes/plan.json").read_text())
        routes = {}
        for name in ("CENTRAL_LEFT", "CENTRAL_STRAIGHT", "CENTRAL_RIGHT"):
            info = plan["routes"][name]
            routes[name] = Route(BASE / info["route_csv"], info["navigation"])
        for route in routes.values(): np.testing.assert_allclose(route.xy[0], routes["CENTRAL_LEFT"].xy[0])
        self.assertEqual({route.command(60.0) for route in routes.values()}, {"left", "straight", "right"})
        self.assertEqual({route.command(5.0) for route in routes.values()}, {"follow_road"})
        self.assertEqual(plan["routes"]["T5"]["split"], "calibration")

    def test_matching_requires_brackets_and_small_gap(self):
        reference = np.array([0, 20, 40, 120]) * 1_000_000
        self.assertEqual(match_fraction([10_000_000, 20_000_000], reference), 1.0)
        self.assertEqual(match_fraction([80_000_000], reference), 0.0)
        self.assertEqual(match_fraction([-10_000_000], reference), 0.0)

    def test_normal_road_curve_keeps_follow_road(self):
        plan = json.loads((BASE / "routes/plan.json").read_text())
        info = plan["routes"]["T4"]
        route = Route(BASE / info["route_csv"], info["navigation"])
        self.assertGreater(route.kappa.max(), .02)
        for s in np.linspace(0, route.s[-1], 30): self.assertEqual(route.command(float(s)), "follow_road")


if __name__ == "__main__":
    unittest.main()
