import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import EgoTimeline, ExpertCommands, HumanCommands
from tools.collection.expert_collect import Route
from tools.collection.protocol import parse_ego
from test_pipeline import ego_packet


class LiveProtocolTests(unittest.TestCase):
    def test_extended_payload_preserves_verified_prefix(self):
        original = ego_packet(12_345_678_901, steer=-4.5, yaw=-30.)
        extended = bytearray(original[:-2] + bytes(48) + b"\r\n")
        struct.pack_into("<I", extended, 11, 200)
        self.assertEqual(parse_ego(bytes(extended)), parse_ego(original))
        struct.pack_into("<I", extended, 11, 152)
        with self.assertRaises(ValueError):
            parse_ego(bytes(extended))

    def test_interpolation_nearest_distance_and_gap_are_both_bounded(self):
        a = parse_ego(ego_packet(100_000_000, steer=0))
        b = parse_ego(ego_packet(134_000_000, steer=3.4))
        timeline = EgoTimeline([a, b])
        # 60Hz render sampling can give 34ms distinct-state gaps at 50Hz UDP.
        self.assertIsNone(timeline.at(103_000_000, 25_000_000))  # legacy strict mode
        self.assertAlmostEqual(timeline.at(103_000_000, 25_000_000, 50_000_000)["steer_raw_deg"], .3)
        b["stamp_ns"] = 200_000_000
        self.assertIsNone(EgoTimeline([a, b]).at(101_000_000, 25_000_000, 50_000_000))
        self.assertIsNone(timeline.at(90_000_000, 25_000_000, 50_000_000))

    def test_expert_is_not_human_command_provenance(self):
        row = {"recv_ns": "1000000000", "sent": "1", "focused": "0", "ego_fresh": "1",
               "reason": "expert:s=651.2 cte=-0.1"}
        self.assertTrue(ExpertCommands([row]).valid_at(1_100_000_000))
        self.assertFalse(HumanCommands([row]).valid_at(1_100_000_000))
        self.assertFalse(ExpertCommands([row]).valid_at(1_300_000_000))
        self.assertFalse(ExpertCommands([dict(row, reason="expert:stale_ego_braking")]).valid_at(1_000_000_000))


class TeacherGeometryTests(unittest.TestCase):
    def test_left_and_right_recovery_signs_and_real_course_curves(self):
        route = Route("A")
        x, y, z, yaw = route.pose(20.)
        rad = math.radians(yaw)
        ego = {"x_m": x - math.sin(rad) * .3, "y_m": y + math.cos(rad) * .3, "yaw_deg": yaw}
        station, cte, heading, steering = route.command(ego)
        self.assertAlmostEqual(station, 20., delta=.02)
        self.assertGreater(cte, .29)
        self.assertLess(steering, 0.)
        route = Route("D")
        angles = []
        for station in (700., 780.):
            x, y, z, yaw = route.pose(station)
            angles.append(route.command({"x_m": x, "y_m": y, "yaw_deg": yaw})[-1])
        self.assertGreater(angles[0], 0.)
        self.assertLess(angles[1], 0.)


if __name__ == "__main__":
    unittest.main()
