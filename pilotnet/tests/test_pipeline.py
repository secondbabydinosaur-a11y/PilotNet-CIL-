"""Synthetic receive-only UDP integration and dataset-integrity tests.

No MORAI connection or vehicle command is used. Every dataset lives in a temp dir.
Run: python3 -m unittest discover -s tests -v
"""
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from tools.collection.collect import Recorder, bind_sockets
from data_utils import BASE, EgoTimeline, load_config, read_csv, read_json, write_json
from tools.collection.inspect_session import inspect
from tools.preparation.prepare_dataset import export, side_curvature_paper, validate_splits
from tools.collection.protocol import CameraAssembler, parse_ego
from tools.collection.manual_control import COMMAND_FIELDS, CONTROL_KIND


def ego_packet(stamp_ns, steer=4.0, x=0.0, y=0.0, yaw=0.0):
    sec, nsec = divmod(stamp_ns, 1_000_000_000)
    # Construct in field order independently of decoder offsets. Angular velocity
    # and acceleration are deliberately different from steering to catch misoffsets.
    floats = [.1, 0., 4.635, 1.892, 2.434, .845, 3., .79,
              x, y, 28.44, 0., 0., yaw, 20., 0., 0., .11, .22, 1.2,
              .5, .6, .7, steer]
    payload = struct.pack("<IIbbfi24f38s", sec, nsec, 1, 4, 20., 0, *floats, b"A2256W000304")
    return b"#MoraiInfo$" + struct.pack("<I", len(payload)) + bytes(12) + payload + b"\r\n"


def camera_packets(jpeg, stamp_ns, chunk_size=6000):
    sec, nsec = divmod(stamp_ns, 1_000_000_000)
    chunks = [jpeg[i:i + chunk_size] for i in range(0, len(jpeg), chunk_size)]
    return [b"MOR" + struct.pack("<IIII", sec, nsec, i, len(chunk)) + chunk + b"\x00\x00EI"
            for i, chunk in enumerate(chunks)]


def image_bytes():
    tiny = np.random.default_rng(7).integers(0, 256, (45, 80, 3), dtype=np.uint8)
    image = cv2.resize(tiny, (1280, 720))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return encoded.tobytes()


class ProtocolTests(unittest.TestCase):
    def test_ego_fields_and_unsupported(self):
        row = parse_ego(ego_packet(12_345_678_901, steer=-7., x=1.467, y=129.745, yaw=101.367))
        self.assertEqual(row["stamp_ns"], 12_345_678_901)
        self.assertAlmostEqual(row["steer_raw_deg"], -7.)
        self.assertAlmostEqual(row["yaw_rate_deg_s"], 1.2, places=5)
        self.assertAlmostEqual(row["x_m"], 1.467, places=5)
        self.assertEqual(row["link_id"], "A2256W000304")
        self.assertEqual(row["speed_kph"], 20.)
        for packet in (b"bad", ego_packet(0)[:-1], b"#MoraiInfo$" + bytes(150)):
            with self.assertRaises(ValueError):
                parse_ego(packet)

    def test_camera_reorder_duplicates_and_bounded_loss(self):
        jpeg = image_bytes()
        packets = camera_packets(jpeg, 1_000_000_000)
        assembler = CameraAssembler(max_pending=2)
        self.assertIsNone(assembler.push(packets[-1]))
        self.assertIsNone(assembler.push(packets[-1]))
        result = None
        for packet in reversed(packets[:-1]):
            result = assembler.push(packet) or result
        self.assertEqual(result, (1_000_000_000, jpeg))
        self.assertIsNone(assembler.push(packets[0]))
        for stamp in (2_000_000_000, 3_000_000_000, 4_000_000_000):
            assembler.push(camera_packets(jpeg, stamp)[0])
        self.assertEqual(assembler.evicted, 1)
        self.assertEqual(len(assembler.pending), 2)

    def test_interpolation_uses_native_clocks_and_short_yaw_arc(self):
        a = parse_ego(ego_packet(100_000_000, steer=2., yaw=179))
        b = parse_ego(ego_packet(120_000_000, steer=4., yaw=-179))
        a["recv_ns"], b["recv_ns"] = 9_000_000_000, 9_100_000_000
        row = EgoTimeline([a, b]).at(110_000_000, 20_000_000)
        self.assertEqual(row["steer_raw_deg"], 3.)
        self.assertEqual(row["yaw_deg"], 180.)
        self.assertIsNone(EgoTimeline([a, b]).at(130_000_000, 20_000_000))

    def test_side_correction_direction_and_zero_offset(self):
        now = {"x_m": 0., "y_m": 0., "yaw_deg": 0.}
        future = {"x_m": 10., "y_m": 0.}
        self.assertEqual(side_curvature_paper(now, future, 0., .01), -.01)
        self.assertGreater(side_curvature_paper(now, future, .4, 0.), 0.)  # right correction
        self.assertLess(side_curvature_paper(now, future, -.4, 0.), 0.)

    def test_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits({"train": ["g1"], "val": ["g2"], "test": ["g1"]})


class FailureTests(unittest.TestCase):
    def test_empty_ego_is_capture_error_not_finished(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = load_config(BASE / "config.json")
            cfg["manual_control"]["enabled"] = False
            cfg["bind_ip"] = "127.0.0.1"
            reservations = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(4)]
            for sock in reservations:
                sock.bind(("127.0.0.1", 0))
            for item, sock in zip(list(cfg["cameras"].values()) + [cfg["ego"]], reservations):
                item["port"] = sock.getsockname()[1]
            for sock in reservations:
                sock.close()
            output = Path(temporary) / "empty"
            recorder = Recorder(cfg, BASE / "config.json", output, "calibration", "trial")
            self.assertEqual(recorder.run(.1, False), 1)
            meta = read_json(output / "session.json")
            self.assertEqual(meta["status"], "capture_error")
            self.assertEqual(meta["saved"]["ego_raw"], 0)

    def test_queue_overflow_stops(self):
        cfg = load_config(BASE / "config.json")
        cfg["recording"]["queue_size"] = 1
        recorder = Recorder(cfg, BASE / "config.json", Path("unused"), "A", "trial")
        self.assertTrue(recorder.enqueue(("event", 0, "one")))
        self.assertFalse(recorder.enqueue(("event", 1, "two")))
        self.assertTrue(recorder.stop.is_set())

    def test_writer_failure_drains_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = load_config(BASE / "config.json")
            recorder = Recorder(cfg, BASE / "config.json", Path(temporary), "A", "trial")
            recorder.enqueue(("image", "center", 0, 0, 0, b"jpeg"))  # nonexistent images directory
            recorder.enqueue(("event", 1, "must drain"))
            recorder.q.put(None)
            thread = threading.Thread(target=recorder.writer, args=({}, {}), daemon=True)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(recorder.stop.is_set())
            self.assertTrue(recorder.fatal_errors)

    def test_port_conflict_is_not_shared(self):
        cfg = load_config(BASE / "config.json")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            cfg["bind_ip"] = "127.0.0.1"
            cfg["cameras"]["left"]["port"] = occupied.getsockname()[1]
            with self.assertRaises(OSError):
                bind_sockets(cfg)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cfg = load_config(BASE / "config.json")
        cfg["manual_control"]["enabled"] = False  # Never send commands to a live MORAI during tests.
        cfg["bind_ip"] = "127.0.0.1"
        # Reserve all ports at once so no two streams receive the same free port.
        reservations = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(4)]
        for sock in reservations:
            sock.bind(("127.0.0.1", 0))
        for item, sock in zip(list(cfg["cameras"].values()) + [cfg["ego"]], reservations):
            item["port"] = sock.getsockname()[1]
        for sock in reservations:
            sock.close()
        config_path = cls.root / "config.json"
        write_json(config_path, cfg)
        raw = cls.root / "raw"
        process = subprocess.Popen([sys.executable, "-m", "tools.collection.collect", "--config", str(config_path),
                    "--output-root", str(raw), "--course", "D", "--group", "g01",
                    "--seconds", "4", "--no-preview"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(BASE))
        first = process.stdout.readline()
        if not first.startswith("SESSION "):
            output = first + process.communicate(timeout=8)[0]
            raise AssertionError(output)
        cls.session = Path(first.strip()[8:])
        jpeg = image_bytes()
        reference = [r for r in read_csv(BASE / "assets/course_reference.csv") if r["course"] == "D"]
        s = np.array([float(r["route_s_m"]) for r in reference])
        base_stamp = 1_700_000_000_000_000_000
        # 3 seconds of native-clock input, paced in real time. All three camera
        # streams run at 30 Hz and ordinary Ego at 50 Hz; include one bad raw Ego.
        events = [(i / 50, "ego", i) for i in range(151)]
        events += [(i / 30, camera, i) for camera in cfg["cameras"] for i in range(90)]
        events.sort()
        start = time.monotonic()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"malformed raw ego", ("127.0.0.1", cfg["ego"]["port"]))
            for offset, name, index in events:
                remaining = start + offset - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                stamp = base_stamp + round(offset * 1e9)
                if name == "ego":
                    station = 650 + offset * 20 / 3.6
                    x, y, yaw = [float(np.interp(station, s, [float(r[k]) for r in reference]))
                                 for k in ("x_m", "y_m", "yaw_deg")]
                    sender.sendto(ego_packet(stamp, x=x, y=y, yaw=yaw), ("127.0.0.1", cfg["ego"]["port"]))
                else:
                    packets = camera_packets(jpeg, stamp, 60000)
                    for packet in reversed(packets):
                        sender.sendto(packet, ("127.0.0.1", cfg["cameras"][name]["port"]))
                        time.sleep(.0003)
        output = first + process.communicate(timeout=8)[0]
        if process.returncode:
            raise AssertionError(output)
        cls.log = output

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_udp_capture_and_full_inspection(self):
        meta = read_json(self.session / "session.json")
        self.assertEqual(meta["status"], "capture_finished", self.log)
        self.assertFalse(meta["validated"])
        self.assertEqual(meta["saved"]["ego"], 151)
        self.assertEqual(meta["saved"]["ego_raw"], 152)
        self.assertEqual(meta["parse_errors"]["ego"], 1)
        for camera in ("left", "center", "right"):
            self.assertEqual(meta["saved"][camera], 90, self.log)
        self.assertEqual(len((self.session / "ego_raw.jsonl").read_text().splitlines()), 152)
        report = inspect(self.session, decode_all=True)
        self.assertTrue(report["data_checks_passed"], report["errors"])
        for camera in ("left", "center", "right"):
            self.assertEqual(report["streams"][camera]["match_fraction"], 1.)
            self.assertEqual(report["streams"][camera]["bad_images"], 0)

    def test_export_and_bad_data_rejection(self):
        export_root = self.root / "export_raw"
        for i in range(1, 4):
            dest = export_root / ("synthetic_g0" + str(i))
            shutil.copytree(self.session, dest)
            meta = read_json(dest / "session.json")
            meta.update(session_id=dest.name, group="g0" + str(i))
            if i == 1:
                meta["control"] = {"kind": CONTROL_KIND, "human_steering": True}
                rows = read_csv(dest / "vehicle.csv")
                with (dest / "vehicle.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    for row in rows:
                        row["ctrl_mode"] = 2
                        writer.writerow(row)
                with (dest / "commands.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=COMMAND_FIELDS)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(recv_ns=int(row["recv_ns"]), sent=1, keys="forward", focused=1,
                                             ego_fresh=1, reason="drive", speed_kph=20, target_kph=19,
                                             accel=0, brake=0, steer_norm=.1))
            write_json(dest / "session.json", meta)
        splits = self.root / "splits.json"
        write_json(splits, {"train": ["g01"], "val": ["g02"], "test": ["g03"]})
        output = self.root / "processed"
        result = export(export_root, splits, output, 1, side_labels=True)
        self.assertEqual(result["status"], "prepared")
        # A frame arriving before the first causal human command is a warmup
        # frame and must be excluded even though a native Ego timestamp matches.
        self.assertGreaterEqual(result["counts"]["train/center"], 28)
        self.assertLessEqual(result["counts"]["train/center"], 30)
        self.assertGreater(result["counts"]["train/left"], 0)
        for split, group in (("train", "g01"), ("val", "g02"), ("test", "g03")):
            rows = read_csv(output / (split + ".csv"))
            self.assertEqual({r["group"] for r in rows}, {group})
            if split != "train":
                self.assertEqual({r["camera"] for r in rows}, {"center"})
            array = np.load(output / rows[0]["array_path"], mmap_mode="r")
            self.assertEqual(array.shape[1:], (66, 200, 3))
            self.assertEqual(array.dtype, np.uint8)
            self.assertAlmostEqual(float(rows[0]["curvature_paper_1pm"]), -np.tan(np.radians(4)) / 3)
        bad_session = export_root / "synthetic_g01"
        meta = read_json(bad_session / "session.json")
        saved_control = meta["control"]
        meta["control"] = {"kind": "unknown_external_controller"}
        write_json(bad_session / "session.json", meta)
        with self.assertRaisesRegex(ValueError, "no eligible samples"):
            export(export_root, splits, self.root / "reject_unknown_auto", 1)
        meta["control"] = saved_control
        write_json(bad_session / "session.json", meta)
        frame = read_csv(bad_session / "frames.csv")[0]
        (bad_session / frame["path"]).write_bytes(b"corrupt")
        self.assertFalse(inspect(bad_session, decode_all=True)["data_checks_passed"])
        with self.assertRaises(ValueError):
            export(export_root, splits, self.root / "must_not_exist", 1)
        self.assertFalse((self.root / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
