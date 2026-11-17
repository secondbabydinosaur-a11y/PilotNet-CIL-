"""실제 센서·차량을 연결하지 않고 시험 경로와 CIL 출력 적용 규칙을 검사한다."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from drive import Reference, actuator_control


class DrivingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cil_driving_test_")
        self.addCleanup(self.directory.cleanup)  # 시험용 CSV만 /tmp에 만들고 삭제한다.
        self.route_path = Path(self.directory.name) / "route.csv"

    def reference(self, points, navigation=None):
        np.savetxt(self.route_path, points, delimiter=",", header="s_m,x_m,y_m,z_m", comments="")
        return Reference({"route_csv": str(self.route_path), "navigation": navigation or []})

    def test_projection_lateral_sign_heading_wrap_and_monotone_search_progress(self):
        reference = self.reference([[0, 0, 0, 0], [5, 5, 0, 0], [10, 10, 0, 0], [15, 15, 0, 0]])
        station, cte, heading = reference.project({"x_m": 7., "y_m": 2., "yaw_deg": 190.})
        self.assertAlmostEqual(station, 7.)
        self.assertAlmostEqual(cte, 2.)  # 전방을 향할 때 왼쪽은 양의 횡오차다.
        self.assertAlmostEqual(heading, -170.)
        station, cte, heading = reference.project({"x_m": 6., "y_m": -1., "yaw_deg": 0.})
        self.assertAlmostEqual(station, 6.)  # 현재 위치의 투영값은 뒤로 이동할 수도 있다.
        self.assertAlmostEqual(reference.progress, 7.)  # 검색 기준 진행률은 뒤로 돌아가지 않는다.
        self.assertAlmostEqual(cte, -1.)
        self.assertAlmostEqual(heading, 0.)

    def test_projection_does_not_jump_to_later_overlapping_segment(self):
        points = [[0, 0, 0, 0], [10, 10, 0, 0], [20, 10, 10, 0],
                  [30, 0, 10, 0], [40, 0, 0, 0], [50, 10, 0, 0]]
        reference = self.reference(points)
        station, cte, _ = reference.project({"x_m": 1., "y_m": .1, "yaw_deg": 0.})
        self.assertAlmostEqual(station, 1.)  # 같은 좌표의 s=41m 구간을 선택하면 안 된다.
        self.assertAlmostEqual(cte, .1)

    def test_command_approach_inside_and_exit_hold_boundaries(self):
        navigation = [{"entry_s_m": 10., "exit_s_m": 20., "approach_m": 5.,
                       "exit_hold_m": 3., "command": "left"}]
        reference = self.reference([[0, 0, 0, 0], [40, 40, 0, 0]], navigation)
        cases = [(4.999, "follow_road"), (5., "left"), (10., "left"),
                 (20., "left"), (23., "left"), (23.001, "follow_road")]
        for station, expected in cases:
            with self.subTest(station=station):
                self.assertEqual(reference.command(station), expected)
        self.assertFalse(hasattr(reference, "steering"))  # 평가 경로는 Teacher 조향을 생성하지 않는다.

    def test_next_junction_command_precedes_previous_exit_hold(self):
        navigation = [
            {"entry_s_m": 10., "exit_s_m": 20., "approach_m": 5., "exit_hold_m": 6., "command": "left"},
            {"entry_s_m": 25., "exit_s_m": 30., "approach_m": 8., "exit_hold_m": 3., "command": "right"},
        ]
        reference = self.reference([[0, 0, 0, 0], [40, 40, 0, 0]], navigation)
        self.assertEqual(reference.command(19.), "left")  # 실제 교차로 내부 명령이 우선이다.
        self.assertEqual(reference.command(22.), "right")  # 통과 후에는 다음 교차로를 준비한다.

    def test_near_duplicate_points_removed_and_invalid_routes_rejected(self):
        reference = self.reference([[0, 0, 0, 0], [.00001, .00001, 0, 0], [5, 5, 0, 0]])
        self.assertEqual(len(reference.s), 2)
        for points in ([[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, 5, 0, 0]],
                       [[0, 0, 0, 0], [1, float("nan"), 0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.reference(points)

    def test_control_clips_normalized_outputs_and_uses_dominant_pedal(self):
        self.assertEqual(actuator_control([2., 2., -.1], 10., 20.), (1., 1., 0., "model"))
        self.assertEqual(actuator_control([-2., .1, 2.], 10., 20.), (-1., 0., 1., "model"))
        self.assertEqual(actuator_control([.2, .5, .5], 10., 20.), (.2, 0., .5, "model"))
        self.assertEqual(actuator_control([.2, -.1, -.2], 10., 20.), (.2, 0., 0., "model"))

    def test_speed_cap_removes_throttle_and_preserves_stronger_brake(self):
        self.assertEqual(actuator_control([.2, .8, .1], 20., 20.), (.2, 0., .15, "speed_cap"))
        self.assertEqual(actuator_control([.2, .1, .6], 25., 20.), (.2, 0., .6, "speed_cap"))
        self.assertEqual(actuator_control([.2, .8, .1], 19.99, 20.), (.2, .8, 0., "model"))

    def test_nonfinite_or_wrong_shape_controls_and_invalid_speed_are_rejected(self):
        for raw in ([0., 0.], [[0., 0., 0.]], [float("nan"), 0., 0.], [0., float("inf"), 0.]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                actuator_control(raw, 10., 20.)
        for speed, cap in ((float("nan"), 20.), (float("inf"), 20.), (10., float("nan")),
                           (10., float("inf")), (10., 0.), (10., -1.)):
            with self.subTest(speed=speed, cap=cap), self.assertRaises(ValueError):
                actuator_control([0., .2, 0.], speed, cap)

    def test_selected_baseline_weights_model_and_data_sources_are_preserved(self):
        selection_path = BASE / "checkpoints/selected.json"
        if not selection_path.exists():
            self.skipTest("local baseline selection is not included in the public repository")
        selection = json.loads(selection_path.read_text())
        checkpoint = BASE / selection["checkpoint"]
        if not checkpoint.exists() or not (BASE / "data/prepared/train.csv").exists():
            self.skipTest("local checkpoint and prepared data are required for this integrity check")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.assertEqual(digest.hexdigest(), selection["checkpoint_sha256"])
        self.assertEqual(selection["epoch"], 19)
        self.assertEqual(selection["run_name"], "cil_baseline_20261117T223956Z")
        settings = json.loads((BASE / "runs" / selection["run_name"] / "settings.json").read_text())
        for name in ("model.py", "dataset.py", "config.json", "train.csv", "val.csv"):
            path = BASE / "data/prepared" / name if name.endswith(".csv") else BASE / name
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), settings["hashes"][name])


if __name__ == "__main__":
    unittest.main()
