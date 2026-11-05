"""Shared dataset IO and clock-safe interpolation, independent of any network."""
from bisect import bisect_left, bisect_right
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
EGO_FIELDS = ["stamp_ns", "recv_ns", "ctrl_mode", "gear", "speed_kph", "steer_raw_deg",
              "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg", "vx_kph",
              "vy_kph", "vz_kph", "yaw_rate_deg_s", "accel", "brake", "link_id"]
FRAME_FIELDS = ["camera", "frame_id", "stamp_ns", "recv_ns", "path"]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def steering_to_curvature(steer_raw_deg, wheelbase_m, steer_left_sign=1,
                          road_wheel_gain=1.0):
    """Reported degrees -> bicycle-model curvature in paper sign (left negative).

    road_wheel_gain=1 explicitly treats reported steering as equivalent road-wheel
    steering. This is a model assumption, not measured trajectory curvature.
    """
    if not all(math.isfinite(x) for x in (steer_raw_deg, wheelbase_m, road_wheel_gain)):
        raise ValueError("steering model requires finite numbers")
    if wheelbase_m <= 0 or road_wheel_gain <= 0 or steer_left_sign not in (-1, 1):
        raise ValueError("invalid wheelbase, steering sign, or road-wheel gain")
    delta = math.radians(steer_left_sign * steer_raw_deg * road_wheel_gain)
    if abs(delta) >= math.pi / 2:
        raise ValueError("equivalent steering must be between -90 and 90 degrees")
    return -math.tan(delta) / wheelbase_m


def load_config(path):
    cfg = read_json(path)
    ports = [item["port"] for item in cfg["cameras"].values()] + [cfg["ego"]["port"]]
    if len(set(ports)) != len(ports) or any(not 0 < int(p) < 65536 for p in ports):
        raise ValueError("all destination ports must be distinct and in 1..65535")
    if set(cfg["cameras"]) != {"left", "center", "right"}:
        raise ValueError("three camera roles required: left, center, right")
    if cfg["ego"]["format"] not in ("normal_ego_152", "normal_ego_152_or_200"):
        raise ValueError("only ordinary Ego payloads of 152 or 200 bytes are supported")
    for key in ("queue_size", "socket_buffer_bytes", "max_pending_frames"):
        if cfg["recording"][key] <= 0:
            raise ValueError("recording." + key + " must be positive")
    if cfg["vehicle"]["wheelbase_m"] <= 0 or cfg["preprocessing"]["sample_hz"] <= 0:
        raise ValueError("wheelbase and sample_hz must be positive")
    manual = cfg.get("manual_control", {})
    if manual.get("enabled"):
        if not 0 < manual["target_margin_kph"] < manual["speed_limit_kph"] <= 20:
            raise ValueError("governor requires 0 < target_margin < speed_limit <= 20kph")
        if manual["command_left_sign"] not in (-1, 1):
            raise ValueError("command_left_sign must be +/-1")
        if not 0 < manual["manual_steer_limit_deg"] <= manual["max_wheel_angle_deg"]:
            raise ValueError("manual steering limit must fit max wheel angle")
        for key in ("hz", "max_ego_age_s", "max_accel_mps2", "steer_rate_deg_s", "steer_return_deg_s"):
            if manual[key] <= 0:
                raise ValueError("manual_control." + key + " must be positive")
        for key in ("sim_ctrl_host_port", "sim_ctrl_destination_port"):
            if not 0 < manual[key] < 65536 or manual[key] in ports:
                raise ValueError("control ports must be distinct from sensor/Ego receive ports")
        if manual["sim_ctrl_host_port"] == manual["sim_ctrl_destination_port"]:
            raise ValueError("control Host and Destination ports must be distinct")
        if not 0 < manual["max_throttle"] <= 1 or manual["shutdown_brake_s"] < 0:
            raise ValueError("invalid governor throttle/shutdown settings")
    return cfg


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class EgoTimeline:
    def __init__(self, rows):
        # A clock reset is reported by inspection before preparation; sorting here
        # only supports normal UDP reordering, and does not make a reset valid.
        unique = {int(row["stamp_ns"]): row for row in rows}
        self.stamps = sorted(unique)
        self.rows = [unique[t] for t in self.stamps]

    def at(self, stamp, max_delta_ns, max_bracket_ns=None):
        i = bisect_left(self.stamps, stamp)
        if i < len(self.stamps) and self.stamps[i] == stamp:
            return self.numeric(self.rows[i])
        if i == 0 or i == len(self.stamps):
            return None
        before, after = self.stamps[i - 1], self.stamps[i]
        if max_bracket_ns is None:
            if stamp - before > max_delta_ns or after - stamp > max_delta_ns:
                return None
        elif (min(stamp - before, after - stamp) > max_delta_ns
              or after - before > max_bracket_ns):
            return None
        a, b = self.numeric(self.rows[i - 1]), self.numeric(self.rows[i])
        # Do not interpolate through gear/control-mode transitions.
        if a["gear"] != b["gear"] or a["ctrl_mode"] != b["ctrl_mode"]:
            return None
        weight = (stamp - before) / (after - before)
        out = dict(a)
        for key in ("speed_kph", "steer_raw_deg", "x_m", "y_m", "z_m", "yaw_rate_deg_s"):
            out[key] = a[key] + weight * (b[key] - a[key])
        yaw_delta = (b["yaw_deg"] - a["yaw_deg"] + 180) % 360 - 180
        out["yaw_deg"] = a["yaw_deg"] + weight * yaw_delta
        out["stamp_ns"] = stamp
        return out

    @staticmethod
    def numeric(row):
        out = dict(row)
        for key in ("speed_kph", "steer_raw_deg", "x_m", "y_m", "z_m", "yaw_deg", "yaw_rate_deg_s"):
            out[key] = float(row[key])
        for key in ("stamp_ns", "gear", "ctrl_mode"):
            out[key] = int(row[key])
        return out


class HumanCommands:
    """Causal command provenance on the host monotonic clock, not sensor time."""
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: int(row["recv_ns"]))
        self.stamps = [int(row["recv_ns"]) for row in self.rows]

    def valid_at(self, recv_ns, max_age_ns=250_000_000):
        i = bisect_right(self.stamps, recv_ns) - 1
        if i < 0 or recv_ns - self.stamps[i] > max_age_ns:
            return False
        row = self.rows[i]
        return (str(row.get("sent")) == "1" and str(row.get("focused")) == "1"
                and str(row.get("ego_fresh")) == "1" and row.get("reason") in ("drive", "brake", "idle"))


EXPERT_KIND = "map_pure_pursuit_expert_v1"


class ExpertCommands(HumanCommands):
    """Explicit automatic-teacher provenance; focus/keys do not establish it."""
    def valid_at(self, recv_ns, max_age_ns=250_000_000):
        i = bisect_right(self.stamps, recv_ns) - 1
        if i < 0 or recv_ns - self.stamps[i] > max_age_ns:
            return False
        row = self.rows[i]
        return (str(row.get("sent")) == "1" and str(row.get("ego_fresh")) == "1"
                and row.get("reason", "").startswith("expert:s="))


def preprocess(image, settings):
    x0, y0, x1, y1 = settings["crop_xyxy"]
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop falls outside the recorded image")
    resized = cv2.resize(image[y0:y1, x0:x1], (settings["width"], settings["height"]),
                         interpolation=cv2.INTER_AREA)
    if settings["color"] != "YUV":
        raise ValueError("expected YUV preprocessing")
    # uint8 HWC, Y/U/V channel order. Normalization belongs to the future model.
    return cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
