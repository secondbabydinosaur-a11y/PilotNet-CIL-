"""미학습 도로의 시험 전용 PP teacher 기록. train/val 목록을 수정하지 않는다."""
import copy
from datetime import datetime, timezone
import math
import socket
import time
import uuid

import numpy as np

from data_utils import BASE, EXPERT_KIND, load_config, read_json, write_json
from drive import brake
from learning.common import file_hash
from learning.live import Reference
from tools.collection.collect import Recorder
from tools.collection.expert_collect import ExpertWorker, Route
from tools.collection.inspect_session import inspect
from tools.collection.sim_api import Simulator
from tools.evaluation.run_matrix import park


def verify_frozen():
    protocol = read_json(BASE / "runs/github_protocol.json")
    for path, expected in protocol["source_sha256"].items():
        if file_hash(BASE / path) != expected:
            raise RuntimeError("frozen model or original split changed: " + path)
    for course, limits in protocol["held_out_sections"].items():
        if list(Reference.limits[course]) != limits:
            raise RuntimeError("held-out route differs from frozen protocol")
        for low, high in protocol["training_capture_station_ranges"]:
            if limits[0] - 8 <= high and limits[1] + 8 >= low:
                raise RuntimeError("new test capture overlaps training capture")
    return protocol


class TestRoute(Route):
    def __init__(self, course):
        start, end = Reference.limits[course]
        raw = np.loadtxt(BASE / "assets/evaluation_route.csv", delimiter=",", skiprows=1)
        self.points = raw[(raw[:, 0] >= start - 20) & (raw[:, 0] <= end + 25)]
        self.s, self.xy = self.points[:, 0], self.points[:, 1:3]
        self.yaw = np.unwrap(np.arctan2(np.gradient(self.xy[:, 1]), np.gradient(self.xy[:, 0])))


def collect(sim, cfg, course):
    start, end = Reference.limits[course]
    route = TestRoute(course)
    capture_start, capture_end = start - 8., end + 8.
    pose = route.pose(capture_start)
    destination = (cfg["manual_control"]["sim_ip"], cfg["manual_control"]["sim_ctrl_host_port"])
    sim.automatic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        brake(sock, destination, .4)
        sim.teleport(*pose)
        brake(sock, destination, 1.)
    location = sim.state()["transform"]["location"]
    if math.hypot(location["x"] - pose[0], location["y"] - pose[1]) > .5:
        raise RuntimeError("test teacher reset failed")
    config = copy.deepcopy(cfg)
    config["manual_control"]["enabled"] = False
    config["recording"]["root"] = "dataset/unseen_raw"
    sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + course + "_" + uuid.uuid4().hex[:6]
    output = BASE / config["recording"]["root"] / sid
    recorder = Recorder(config, BASE / "config.json", output, course, "unseen_test_g01",
        "TEST ONLY: new road, frozen models, automatic PP teacher; never train/validation.",
        control_factory=lambda rec: ExpertWorker(rec, route, 19., 4.5, capture_end))
    recorder.manifest["evaluation_only"] = True
    recorder.manifest["evaluation_station_m"] = [start, end]
    recorder.manifest["control"] = dict(kind=EXPERT_KIND, human_steering=False,
        teacher="map Pure Pursuit + speed feedback", start_pose_xyzyaw=pose,
        initial_offset_m=0., initial_yaw_offset_deg=0., target_speed_kph=19.,
        lookahead_m=4.5, speed_ceiling_kph=20., end_station_m=capture_end,
        capture_station_m=[capture_start, capture_end],
        route_sha256=file_hash(BASE / "assets/evaluation_route.csv"))
    code = recorder.run(seconds=100, preview=False)
    if code or not recorder.control.reached_end:
        raise RuntimeError("test teacher incomplete: " + str(output))
    report = inspect(output, decode_all=True)
    write_json(output / "quality_report.json", report)
    if not report["data_checks_passed"]:
        raise RuntimeError("test capture inspection failed: " + str(report["errors"]))
    print("VERIFIED TEST ONLY {} {}".format(course, report["expert_tracking"]), flush=True)
    return str(output.relative_to(BASE))


def main():
    verify_frozen()
    index = BASE / "dataset/unseen_sessions.json"
    sessions = read_json(index) if index.exists() else {}
    sim = Simulator()
    try:
        sim.resume()
        sim.conditions()
        for course in ("U1", "U2"):
            if course in sessions:
                report = read_json(BASE / sessions[course] / "quality_report.json")
                meta = read_json(BASE / sessions[course] / "session.json")
                if not report["data_checks_passed"] or meta.get("evaluation_station_m") != list(Reference.limits[course]):
                    raise RuntimeError("existing unseen capture is invalid")
                print("EXISTING TEST ONLY " + course, flush=True)
                continue
            sessions[course] = collect(sim, load_config(BASE / "config.json"), course)
            write_json(index, sessions)
    finally:
        sim.channel.close()
        park()
        verify_frozen()


if __name__ == "__main__":
    main()
