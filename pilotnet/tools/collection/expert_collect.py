#!/usr/bin/env python3
"""Collect measured MORAI demonstrations from a classical map Pure Pursuit teacher.

This is NOT human driving, PilotNet inference, or closed-loop model evaluation.
Teleports occur only between sessions. The trained model will not receive the map.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import socket
import threading
import time
import uuid

import numpy as np

from .collect import Recorder
from data_utils import BASE, EXPERT_KIND, load_config, write_json
from .manual_control import encode_command
from .sim_api import Simulator
from .inspect_session import inspect

TELEMETRY_FIELDS = ["recv_ns", "ego_stamp_ns", "route_s_m", "cte_m", "heading_error_deg",
                    "speed_kph", "steer_raw_deg", "command_deg", "target_kph"]


class Route:
    def __init__(self, course):
        raw = np.loadtxt(BASE / "assets/expert_route.csv", delimiter=",", skiprows=1)
        self.points = raw[raw[:, 0] < 100] if course == "A" else raw[raw[:, 0] > 600]
        self.s = self.points[:, 0]
        self.xy = self.points[:, 1:3]
        self.yaw = np.unwrap(np.arctan2(np.gradient(self.xy[:, 1]), np.gradient(self.xy[:, 0])))

    def pose(self, station):
        return [float(np.interp(station, self.s, self.points[:, i])) for i in (1, 2, 3)] + [
            math.degrees(float(np.interp(station, self.s, self.yaw)))]

    def command(self, ego, lookahead=4.5):
        pos = np.array([ego["x_m"], ego["y_m"]])
        i = int(np.argmin(np.sum((self.xy - pos) ** 2, axis=1)))
        # Rear-axle path projection onto the closest segment (0.5m sampled).
        candidates = []
        for j in range(max(0, i - 1), min(len(self.xy) - 1, i + 1)):
            edge = self.xy[j + 1] - self.xy[j]
            ratio = float(np.clip(np.dot(pos - self.xy[j], edge) / np.dot(edge, edge), 0, 1))
            projected = self.xy[j] + ratio * edge
            candidates.append((np.linalg.norm(pos - projected), j, ratio, projected))
        _, j, ratio, projected = min(candidates, key=lambda item: item[0])
        station = float(self.s[j] + ratio * (self.s[j + 1] - self.s[j]))
        tangent = math.atan2(*(self.xy[j + 1] - self.xy[j])[::-1])
        cte = float(np.dot(pos - projected, [-math.sin(tangent), math.cos(tangent)]))
        target = np.array(self.pose(min(station + lookahead, self.s[-1]))[:2])
        diff = target - pos
        yaw = math.radians(ego["yaw_deg"])
        local_y = -math.sin(yaw) * diff[0] + math.cos(yaw) * diff[1]
        curvature = 2 * local_y / max(float(np.dot(diff, diff)), 1.)
        angle = math.degrees(math.atan(3. * curvature))
        heading_error = (ego["yaw_deg"] - math.degrees(tangent) + 180) % 360 - 180
        return station, cte, heading_error, float(np.clip(angle, -20, 20))


class ExpertWorker:
    def __init__(self, recorder, route, target, lookahead, end_station):
        self.recorder, self.route = recorder, route
        self.target, self.lookahead, self.end_station = target, lookahead, end_station
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="map-expert")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cfg = recorder.cfg["manual_control"]
        self.destination = cfg["sim_ip"], cfg["sim_ctrl_host_port"]
        self.last_reason = "waiting_ego"
        self.reached_end = False

    def start(self):
        self.thread.start()

    def send(self, throttle, brake, angle):
        self.sock.sendto(encode_command(throttle, brake, angle / 40.), self.destination)

    def run(self):
        previous_speed, previous_recv = None, None
        start = time.monotonic()
        try:
            with (self.recorder.output / "expert.csv").open("x", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS)
                writer.writeheader()
                while not self.stop.wait(.02) and not self.recorder.stop.is_set():
                    ego, recv = self.recorder.ego_snapshot()
                    now = time.monotonic_ns()
                    fresh = ego is not None and (now - recv) < 200_000_000
                    throttle, brake, angle = 0., 1., 0.
                    target = self.target
                    if fresh:
                        if ego["ctrl_mode"] != 2:
                            raise RuntimeError("expert requires Auto control mode")
                        station, cte, heading, angle = self.route.command(ego, self.lookahead)
                        if abs(cte) > .8 or abs(heading) > 25:
                            raise RuntimeError("teacher left course: cte={:.2f}, heading={:.1f}".format(cte, heading))
                        if ego["speed_kph"] > 20.3:
                            raise RuntimeError("teacher exceeded speed limit: {:.2f}".format(ego["speed_kph"]))
                        remaining = self.end_station - station
                        target = min(self.target, 3.6 * math.sqrt(max(0., 2. * .8 * (remaining - 1.))))
                        speed = ego["speed_kph"] / 3.6
                        accel = 0.
                        if previous_recv is not None and recv != previous_recv:
                            accel = (speed - previous_speed) / ((recv - previous_recv) / 1e9)
                        if recv != previous_recv:
                            previous_speed, previous_recv = speed, recv
                        demand = 1.5 * (target / 3.6 - speed) - .18 * accel
                        throttle = min(.25, max(0., demand / 5.2))
                        brake = min(.5, max(0., -demand / 9.))
                        if target < .2:
                            throttle, brake = 0., .6
                        self.last_reason = "s={:.1f} cte={:.3f}".format(station, cte)
                        writer.writerow(dict(zip(TELEMETRY_FIELDS, [now, ego["stamp_ns"], station,
                            cte, heading, ego["speed_kph"], ego["steer_raw_deg"], angle, target])))
                        if remaining < 1.5 and ego["speed_kph"] < .8:
                            self.reached_end = True
                            self.recorder.stop.set()
                    else:
                        self.last_reason = "stale_ego_braking"
                        if time.monotonic() - start > 3:
                            raise RuntimeError("expert has no fresh Ego")
                    self.send(throttle, brake, angle)
                    self.recorder.enqueue(("command", {"recv_ns": now, "sent": 1, "keys": "",
                        "focused": 0, "ego_fresh": int(fresh), "reason": "expert:" + self.last_reason,
                        "speed_kph": ego["speed_kph"] if ego else 0., "target_kph": target,
                        "accel": throttle, "brake": brake, "steer_norm": angle / 40.}))
        except Exception as exc:
            self.recorder.fail("expert: " + str(exc))
        finally:
            self.send(0., 1., 0.)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        for _ in range(25):
            self.send(0., 1., 0.)
            time.sleep(.02)
        self.sock.close()


def collect_one(sim, cfg, course, group, variant=0):
    route = Route(course)
    start_station, end_station = (0., 65.) if course == "A" else (640., 866.)
    pose = route.pose(start_station)
    # Small deterministic initial-pose variation, explicitly recorded.
    offset = [0., .12, -.12][variant % 3]
    yaw_offset = [0., .5, -.5][variant % 3]
    yaw = math.radians(pose[3])
    pose[0] -= math.sin(yaw) * offset
    pose[1] += math.cos(yaw) * offset
    pose[3] += yaw_offset
    sim.automatic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _ in range(15):
            sock.sendto(encode_command(0., 1., 0.), (cfg["manual_control"]["sim_ip"], cfg["manual_control"]["sim_ctrl_host_port"]))
            time.sleep(.02)
        sim.teleport(*pose)
        for _ in range(75):
            sock.sendto(encode_command(0., 1., 0.), (cfg["manual_control"]["sim_ip"], cfg["manual_control"]["sim_ctrl_host_port"]))
            time.sleep(.02)
    actual = sim.state()["transform"]["location"]
    if np.hypot(actual["x"] - pose[0], actual["y"] - pose[1]) > .5:
        raise RuntimeError("teleport readback did not match requested starting pose")
    config = copy.deepcopy(cfg)
    config["manual_control"]["enabled"] = False
    target = [19., 18.5, 19.5][variant % 3]
    lookahead = 4.5
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + course + "_" + uuid.uuid4().hex[:6]
    output = BASE / cfg["recording"]["root"] / run_id
    recorder = Recorder(config, BASE / "config.json", output, course, group,
            "Automatic classical teacher; not a human demonstration; no neural model involved.",
            control_factory=lambda rec: ExpertWorker(rec, route, target, lookahead, end_station))
    recorder.manifest["control"] = {"kind": EXPERT_KIND, "human_steering": False,
        "teacher": "map Pure Pursuit + speed feedback", "start_pose_xyzyaw": pose,
        "initial_offset_m": offset, "initial_yaw_offset_deg": yaw_offset,
        "target_speed_kph": target, "lookahead_m": lookahead, "speed_ceiling_kph": 20.,
        "end_station_m": end_station, "variant": variant,
        "route_sha256": hashlib.sha256((BASE / "assets/expert_route.csv").read_bytes()).hexdigest()}
    result = recorder.run(seconds=100 if course == "D" else 45, preview=False)
    if not recorder.control.reached_end:
        recorder.manifest["status"] = "capture_error"
        recorder.manifest.setdefault("fatal_errors", []).append("teacher did not reach course end")
        write_json(output / "session.json", recorder.manifest)
        raise RuntimeError("teacher did not complete " + str(output))
    if result:
        raise RuntimeError("capture failed: " + str(output))
    report = inspect(output, decode_all=True)
    write_json(output / "quality_report.json", report)
    if not report["data_checks_passed"]:
        raise RuntimeError("inspection failed: {}: {}".format(output, report["errors"]))
    print("VERIFIED {} matching={} tracking={}".format(output.name,
          {name: round(report["streams"][name]["match_fraction"], 4) for name in cfg["cameras"]},
          report["expert_tracking"]), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--courses", nargs="+", choices=["A", "D"], default=["A", "D"])
    parser.add_argument("--groups", nargs="+", default=["expert_pilot"])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(BASE / "config.json")
    sim = Simulator()
    sim.resume()
    sim.conditions()
    for group_index, group in enumerate(args.groups):
        for repeat in range(args.repeats):
            for course in args.courses:
                path = collect_one(sim, cfg, course, group, variant=group_index + repeat)
                print("COMPLETE " + str(path), flush=True)


if __name__ == "__main__":
    main()
