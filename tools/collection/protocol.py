"""Receive-only MORAI UDP decoding. No vehicle command generation.

Reference: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1
The ordinary 152-byte and extended 200-byte payloads are supported. Competition
status may share the same length: select the ordinary publisher and check the HUD.
"""
from collections import OrderedDict, deque
import math
import struct


def sensor_stamp(sec, nsec):
    if not (0 <= nsec < 1_000_000_000):
        raise ValueError("invalid sensor nanoseconds")
    return sec * 1_000_000_000 + nsec


def parse_ego(packet):
    if len(packet) not in (181, 229) or packet[:11] != b"#MoraiInfo$":
        raise ValueError("expected ordinary Ego Vehicle Status (181 or 229 bytes)")
    size = struct.unpack_from("<I", packet, 11)[0]
    if size != len(packet) - 29 or packet[-2:] != b"\r\n":
        raise ValueError("invalid Ego payload size or tail")
    payload = packet[27:-2]
    sec, nsec = struct.unpack_from("<II", payload)
    ctrl_mode, gear = struct.unpack_from("<bb", payload, 8)
    fields = {
        "speed_kph": 10, "accel": 18, "brake": 22,
        "x_m": 50, "y_m": 54, "z_m": 58,
        "roll_deg": 62, "pitch_deg": 66, "yaw_deg": 70,
        "vx_kph": 74, "vy_kph": 78, "vz_kph": 82,
        "yaw_rate_deg_s": 94, "steer_raw_deg": 110,
    }
    values = {name: struct.unpack_from("<f", payload, offset)[0]
              for name, offset in fields.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite Ego value")
    if abs(values["speed_kph"]) > 400 or abs(values["steer_raw_deg"]) > 90:
        raise ValueError("implausible speed/steer: check ordinary Ego publisher")
    values.update(stamp_ns=sensor_stamp(sec, nsec), ctrl_mode=ctrl_mode, gear=gear,
                  link_id=payload[114:152].split(b"\x00", 1)[0].decode("ascii", errors="replace"))
    return values


class CameraAssembler:
    """Reassemble JPEGs by sensor timestamp, supporting out-of-order fragments."""

    def __init__(self, max_pending=6):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self.pending = OrderedDict()
        self.completed = deque(maxlen=128)
        self.evicted = 0
        self.duplicates = 0

    def push(self, packet):
        if len(packet) < 19 or packet[:3] != b"MOR":
            raise ValueError("invalid camera header")
        sec, nsec, index, length = struct.unpack_from("<IIII", packet, 3)
        stamp = sensor_stamp(sec, nsec)
        if index >= 512 or length < 1 or length > 65507 - 19 or 19 + length > len(packet):
            raise ValueError("invalid camera fragment length/index")
        if stamp in self.completed:
            self.duplicates += 1
            return None
        if stamp not in self.pending:
            if len(self.pending) >= self.max_pending:
                self.pending.popitem(last=False)
                self.evicted += 1
            self.pending[stamp] = {}
        chunks = self.pending[stamp]
        fragment = packet[19:19 + length]
        if index in chunks:
            if chunks[index] != fragment:
                del self.pending[stamp]
                raise ValueError("conflicting duplicate camera fragment")
            self.duplicates += 1
            return None
        chunks[index] = fragment
        # JPEG end, rather than UDP padding/EI, determines frame completion.
        if 0 not in chunks or not chunks[0].startswith(b"\xff\xd8"):
            return None
        last = max(chunks)
        if not chunks[last].endswith(b"\xff\xd9") or len(chunks) != last + 1:
            return None
        jpeg = b"".join(chunks[i] for i in range(last + 1))
        del self.pending[stamp]
        self.completed.append(stamp)
        return stamp, jpeg
