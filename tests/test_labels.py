"""Label-only checks: native clocks, curvature units/sign, split integrity, no image processing."""
import csv
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation.build_labels import build, matched_status
from data_utils import BASE, EXPERT_KIND, EgoTimeline, read_csv, read_json, steering_to_curvature, write_json


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ego(stamp, angle, **updates):
    row = dict(stamp_ns=stamp, recv_ns=5_000_000_000, gear=4, ctrl_mode=2,
               speed_kph=20., steer_raw_deg=angle, x_m=0., y_m=0., z_m=0.,
               yaw_deg=0., yaw_rate_deg_s=0.)
    row.update(updates)
    return row


class NumericLabelTests(unittest.TestCase):
    def test_degrees_wheelbase_and_paper_sign(self):
        self.assertEqual(steering_to_curvature(0., 3.), 0.)
        self.assertAlmostEqual(steering_to_curvature(3., 3.), -.017469259761013735)
        self.assertAlmostEqual(steering_to_curvature(-3., 3.), .017469259761013735)
        self.assertAlmostEqual(steering_to_curvature(3., 6.), steering_to_curvature(3., 3.) / 2)
        self.assertAlmostEqual(steering_to_curvature(3., 3., -1), steering_to_curvature(-3., 3.))
        for args in [(1., 0.), (float("nan"), 3.), (90., 3.), (1., 3., 0), (1., 3., 1, 0.)]:
            with self.assertRaises(ValueError):
                steering_to_curvature(*args)

    def test_real_epoch_integer_brackets_and_exact_values(self):
        base = 1_793_749_227_960_000_000  # larger than exact float integer precision
        a, b = ego(base + 8_008_000_000, 2.405254364013672), ego(base + 8_039_000_000, 2.3521015644073486)
        tl = EgoTimeline([a, b])
        quality = {"max_pair_delta_ms": 25., "max_interpolation_gap_ms": 50.}
        status, audit, reason = matched_status(tl, base + 8_026_000_000, quality)
        self.assertIsNone(reason)
        self.assertEqual(audit["before_stamp_ns"], a["stamp_ns"])
        self.assertAlmostEqual(audit["interpolation_weight"], 18 / 31)
        self.assertAlmostEqual(status["steer_raw_deg"], 2.374391448113226)
        self.assertEqual(audit["nearest_delta_ms"], 13.)
        exact, info, _ = matched_status(tl, a["stamp_ns"], quality)
        self.assertEqual(exact["steer_raw_deg"], a["steer_raw_deg"])
        self.assertEqual(info["match_method"], "exact")
        self.assertEqual(info["bracket_gap_ms"], 0.)

    def test_no_extrapolation_no_large_gaps_no_mode_transition(self):
        q = {"max_pair_delta_ms": 25., "max_interpolation_gap_ms": 50.}
        self.assertIsNone(matched_status(EgoTimeline([ego(0, 0), ego(100_000_000, 1)]), 1_000_000, q)[0])
        self.assertIsNone(matched_status(EgoTimeline([ego(0, 0), ego(20_000_000, 1)]), 30_000_000, q)[0])
        self.assertIsNone(matched_status(EgoTimeline([ego(0, 0), ego(20_000_000, 1, gear=1)]), 10_000_000, q)[0])


class LabelBuildTests(unittest.TestCase):
    def fixture(self, root):
        splits = root / "dataset/splits"
        splits.mkdir(parents=True)
        groups = {"train": ["g01"], "val": ["g05"], "test": ["g06"]}
        write_json(splits / "groups.json", groups)
        cfg = read_json(BASE / "config.json")
        origin = 1_793_749_000_000_000_000
        for split, names in groups.items():
            group = names[0]
            sid = "synthetic_" + group
            session = root / "dataset/raw" / sid
            for camera in ("center", "left", "right"):
                (session / "images" / camera).mkdir(parents=True)
            write_json(session / "session.json", {"session_id": sid, "group": group,
                "course": "D", "status": "capture_finished", "config": cfg,
                "control": {"kind": EXPERT_KIND, "human_steering": False}})
            write_json(session / "quality_report.json", {"data_checks_passed": True})
            save_csv(session / "vehicle.csv", [ego(origin, 2., speed_kph=0.),
                        ego(origin + 20_000_000, 4.), ego(origin + 40_000_000, 6.)])
            save_csv(session / "commands.csv", [{"recv_ns": 9_999_999_999, "sent": 1,
                        "ego_fresh": 1, "focused": 0, "reason": "expert:s=700 cte=0"}])
            raw_frames, index = [], []
            for camera in ("center", "left", "right"):
                for i, delta in enumerate((0, 10_000_000, 20_000_000, 60_000_000)):
                    relative = "images/{}/{:08d}.jpg".format(camera, i)
                    (session / relative).write_bytes(b"original bytes; never decode or alter in this stage")
                    frame = dict(camera=camera, frame_id=i, stamp_ns=origin + delta,
                                 recv_ns=10_000_000_000 + delta, path=relative)
                    raw_frames.append(frame)
                    index.append({"session_id": sid, "group": group, "course": "D", **{
                        k: frame[k] for k in ("camera", "frame_id", "stamp_ns", "recv_ns")},
                        "image_path": str((session / relative).relative_to(root)),
                        "vehicle_csv": str((session / "vehicle.csv").relative_to(root))})
            save_csv(session / "frames.csv", raw_frames)
            save_csv(splits / (split + ".csv"), index)
        return splits

    def test_writes_pairs_without_decoding_or_changing_any_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = self.fixture(root)
            original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            output = root / "dataset/paired"
            with patch("cv2.imread", side_effect=AssertionError("images must not be decoded")), \
                 patch("data_utils.preprocess", side_effect=AssertionError("preprocessing is on hold")):
                report = build(splits, output, root)
            for split in ("train", "val", "test"):
                rows = read_csv(output / (split + ".csv"))
                self.assertEqual(len(rows), 3)
                self.assertEqual(set(r["camera"] for r in rows), {"center"})
                self.assertAlmostEqual(float(rows[1]["steer_raw_deg"]), 3.)
                self.assertAlmostEqual(float(rows[1]["y_curvature_1pm"]), -math.tan(math.radians(3)) / 3)
                self.assertEqual(rows[0]["speed_eligible"], "0")  # stops retained, flagged
                self.assertEqual(report["splits"][split]["side_images_deferred"], 8)
                self.assertEqual(sum(report["splits"][split]["rejected_center"].values()), 1)
            self.assertEqual(len(read_csv(output / "rejected_center.csv")), 3)
            self.assertEqual(list(output.glob("*.npy")), [])
            for path, digest in original.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, "already exists"):
                build(splits, output, root)

    def test_repeated_image_and_tampered_timestamp_leave_no_output(self):
        for corruption in ("duplicate", "timestamp"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                splits = self.fixture(root)
                rows = read_csv(splits / "train.csv")
                if corruption == "duplicate":
                    rows.append(rows[0])
                else:
                    rows[0]["stamp_ns"] = str(int(rows[0]["stamp_ns"]) + 1)
                save_csv(splits / "train.csv", rows)
                output = root / "dataset/paired"
                with self.assertRaises(ValueError):
                    build(splits, output, root)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
