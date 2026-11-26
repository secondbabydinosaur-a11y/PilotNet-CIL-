from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai import LegacyCameraAssembler, parse_ego


class LegacyMoraiTests(unittest.TestCase):
    def test_ego_packet_without_sensor_timestamp(self):
        payload = bytearray(132)
        payload[:2] = bytes([2, 4])
        struct.pack_into("<f", payload, 2, 15.)
        struct.pack_into("<fff", payload, 42, 120., -32., 1.3)
        struct.pack_into("<f", payload, 62, -59.)
        struct.pack_into("<f", payload, 90, 14.)
        payload[94:99] = b"A219A"
        packet = b"#MoraiInfo$" + struct.pack("<I", 132) + bytes(12) + payload + b"\r\n"
        status = parse_ego(packet, 123456)
        self.assertEqual(status["stamp_ns"], 123456)
        self.assertEqual(status["link_id"], "A219A")
        self.assertEqual((status["speed_kph"], status["x_m"], status["yaw_deg"], status["steer_raw_deg"]),
                         (15., 120., -59., 14.))
        with self.assertRaises(ValueError):
            parse_ego(packet)

    def test_camera_restarts_after_a_lost_fragment(self):
        def packet(index, payload, marker):
            return b"MOR" + struct.pack("<II", index, len(payload)) + payload + bytes(10) + marker

        camera = LegacyCameraAssembler()
        start = packet(0, b"\xff\xd8s", b"AI")
        self.assertIsNone(camera.push(start, 10))
        self.assertIsNone(camera.push(packet(2, b"x\xff\xd9", b"EI"), 11))
        self.assertIsNone(camera.push(start, 20))
        self.assertEqual(camera.push(packet(1, b"e\xff\xd9", b"EI"), 21), (20, b"\xff\xd8se\xff\xd9"))


if __name__ == "__main__":
    unittest.main()
