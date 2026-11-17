"""시각 연결·Teacher 정답·가상 카메라 복귀 보정을 독립된 예제로 검증한다."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import (
    Match, Rejected, Timeline, correct_side_steering,
    interpolate_ego, interpolate_teacher, pp_angle, virtual_pose,
)


MS = 1_000_000


def teacher(**changes):
    row = dict(
        ego_stamp_ns=0, sent_ns=100, command=0,
        teacher_steer=.2, teacher_throttle=.25, teacher_brake=0,
        applied_steer=.8, applied_throttle=.25, applied_brake=0,
        noise_steer=.6, route_s_m=10, control_phase="drive",
        label_valid=1, noise_episode=2, recovery_active=1,
    )
    row.update(changes)
    return {key: str(value) for key, value in row.items()}  # 실제 CSV와 같은 문자열이다.


def ego(**changes):
    row = dict(
        stamp_ns=0, recv_ns=200, ctrl_mode=2, gear=4,
        speed_kph=10, x_m=5, y_m=0, z_m=28, yaw_deg=0,
    )
    row.update(changes)
    return {key: str(value) for key, value in row.items()}


def paired(left, right, alpha=.5):
    return Match(left=left, right=right, alpha=alpha, left_ns=0, right_ns=20 * MS)


def rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


class TimelineTests(unittest.TestCase):
    def test_exact_hits_and_interpolation_brackets(self):
        rows = [ego(stamp_ns=20 * MS), ego(stamp_ns=0)]  # UDP 도착 순서와 무관해야 한다.
        timeline = Timeline(rows, "stamp_ns")
        exact = timeline.bracket(20 * MS)
        self.assertIsNotNone(exact)
        self.assertEqual(exact.left_ns, 20 * MS)
        self.assertEqual(exact.right_ns, 20 * MS)
        self.assertEqual(exact.alpha, 0)
        middle = timeline.bracket(5 * MS)
        self.assertEqual((middle.left_ns, middle.right_ns), (0, 20 * MS))
        self.assertAlmostEqual(middle.alpha, .25)

    def test_no_extrapolation_or_large_sensor_gaps(self):
        timeline = Timeline([ego(stamp_ns=0), ego(stamp_ns=20 * MS)], "stamp_ns")
        self.assertIsNone(timeline.bracket(-1))
        self.assertIsNone(timeline.bracket(20 * MS + 1))
        gap = Timeline([ego(stamp_ns=0), ego(stamp_ns=60 * MS)], "stamp_ns")
        self.assertIsNone(gap.bracket(30 * MS))
        far = Timeline([ego(stamp_ns=0), ego(stamp_ns=100 * MS)], "stamp_ns")
        self.assertIsNone(far.bracket(50 * MS, max_gap_ns=100 * MS))

    def test_duplicate_uses_earliest_pc_receive_or_send(self):
        timeline = Timeline([ego(recv_ns=900), ego(recv_ns=100)], "stamp_ns")
        self.assertEqual(timeline.duplicates, 1)
        self.assertEqual(int(timeline.bracket(0).left["recv_ns"]), 100)
        sent = Timeline([teacher(sent_ns=900), teacher(sent_ns=100)], "ego_stamp_ns")
        self.assertEqual(int(sent.bracket(0).left["sent_ns"]), 100)

    def test_conflicting_duplicate_cannot_silently_supply_a_label(self):
        timeline = Timeline(
            [teacher(teacher_steer=.1), teacher(teacher_steer=.2)],
            "ego_stamp_ns", compare_fields=("teacher_steer",),
        )
        self.assertEqual(timeline.duplicates, 1)
        self.assertIn(0, timeline.conflicts)
        self.assertIsNone(timeline.bracket(0))


class InterpolationTests(unittest.TestCase):
    def test_interpolates_teacher_and_never_applied_noisy_command(self):
        result = interpolate_teacher(paired(
            teacher(teacher_steer=.1, applied_steer=.8, route_s_m=10),
            teacher(teacher_steer=.3, applied_steer=.9, route_s_m=14),
            alpha=.25,
        ))
        self.assertAlmostEqual(result["steer"], .15)
        self.assertAlmostEqual(result["throttle"], .25)
        self.assertEqual(result["brake"], 0)
        self.assertEqual(result["command"], 0)
        self.assertAlmostEqual(result["route_s_m"], 11)
        self.assertEqual(result["noise_episode"], 2)
        self.assertTrue(result["recovery_active"])

    def test_rejects_command_boundary_or_non_driving_label(self):
        for change in (dict(command=1), dict(control_phase="terminal_stop"), dict(label_valid=0)):
            with self.subTest(change=change), self.assertRaises(Rejected):
                interpolate_teacher(paired(teacher(), teacher(**change)))
            with self.subTest(left=change), self.assertRaises(Rejected):
                interpolate_teacher(paired(teacher(**change), teacher()))

    def test_rejects_interpolation_across_accelerator_brake_transition(self):
        with self.assertRaises(Rejected):
            interpolate_teacher(paired(
                teacher(teacher_throttle=.2, teacher_brake=0),
                teacher(teacher_throttle=0, teacher_brake=.2),
            ))  # 보간으로 가속과 브레이크가 동시에 양수가 되면 안 된다.

    def test_ego_yaw_uses_short_rotation_across_wrap(self):
        result = interpolate_ego(paired(
            ego(speed_kph=10, x_m=2, yaw_deg=179),
            ego(speed_kph=14, x_m=6, yaw_deg=-179),
        ))
        self.assertAlmostEqual(result["speed_kph"], 12)
        self.assertAlmostEqual(result["x_m"], 4)
        self.assertAlmostEqual(abs(result["yaw_deg"]), 180)

    def test_ego_requires_forward_external_control(self):
        for change in (dict(ctrl_mode=1), dict(gear=2), dict(speed_kph=-1)):
            with self.subTest(change=change), self.assertRaises(Rejected):
                interpolate_ego(paired(ego(), ego(**change)))
        nearly_stopped = interpolate_ego(paired(ego(speed_kph=-.01), ego(speed_kph=-.01)))
        self.assertEqual(nearly_stopped["speed_kph"], 0)


class CameraRecoveryTests(unittest.TestCase):
    def setUp(self):
        s = np.arange(0., 101.)
        self.route = np.column_stack((s, s, np.zeros_like(s), np.full_like(s, 28)))
        self.center = dict(xyz_m=[2.6, 0., 1.18], rpy_deg=[0., 2., 0.])
        self.left = dict(xyz_m=[2.6, .4, 1.18], rpy_deg=[0., 2., 30.])
        self.right = dict(xyz_m=[2.6, -.4, 1.18], rpy_deg=[0., 2., -30.])
        self.state = {key: float(value) for key, value in ego().items()}

    def test_identical_extrinsics_keep_original_teacher(self):
        corrected = correct_side_steering(
            self.state, .17, 5., self.route, self.center, self.center,
        )
        self.assertAlmostEqual(corrected, .17, places=12)

    def test_mirrored_straight_views_require_opposite_recovery(self):
        left = correct_side_steering(self.state, 0., 5., self.route, self.left, self.center)
        right = correct_side_steering(self.state, 0., 5., self.route, self.right, self.center)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        self.assertAlmostEqual(left, -right, places=12)

    def test_virtual_pose_reconstructs_side_camera_world_origin_and_heading(self):
        state = dict(self.state, x_m=17., y_m=-31., yaw_deg=46.)
        center = dict(xyz_m=[2.6, .1, 1.18], rpy_deg=[0., 2., 3.])
        pose = virtual_pose(state, self.left, center)
        yaw = math.radians(state["yaw_deg"])
        physical_origin = np.array([state["x_m"], state["y_m"]]) + rotation(yaw) @ np.array(self.left["xyz_m"][:2])
        reconstructed = pose[:2] + rotation(pose[2]) @ np.array(center["xyz_m"][:2])
        np.testing.assert_allclose(reconstructed, physical_origin, atol=1e-12)
        self.assertAlmostEqual(pose[2] + math.radians(center["rpy_deg"][2]),
                               yaw + math.radians(self.left["rpy_deg"][2]))

    def test_correction_is_invariant_to_world_translation_and_rotation(self):
        state = dict(self.state, y_m=.1, yaw_deg=5.)
        first = correct_side_steering(state, .05, 5., self.route, self.left, self.center)
        angle = math.radians(67.)
        offset = np.array([102., -55.])
        route = self.route.copy()
        route[:, 1:3] = route[:, 1:3] @ rotation(angle).T + offset
        position = rotation(angle) @ np.array([state["x_m"], state["y_m"]]) + offset
        shifted = dict(state, x_m=position[0], y_m=position[1], yaw_deg=state["yaw_deg"] + 67.)
        second = correct_side_steering(shifted, .05, 5., route, self.left, self.center)
        self.assertAlmostEqual(first, second, places=11)

    def test_forward_goal_and_available_route_tail_are_required(self):
        with self.assertRaises(Rejected):
            pp_angle(np.array([0., 0., 0.]), np.array([-2., 0.]), 3.)
        with self.assertRaises(Rejected):
            correct_side_steering(self.state, 1.1, 5., self.route, self.left, self.center)
        with self.assertRaises(Rejected):
            correct_side_steering(self.state, 0., 99., self.route, self.left, self.center)


if __name__ == "__main__":
    unittest.main()
