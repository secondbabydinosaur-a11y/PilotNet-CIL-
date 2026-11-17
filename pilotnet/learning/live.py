"""MORAI 최신 센서 수신, 평가용 지도 투영, 곡률 명령 변환.

지도는 시작 위치·진행률·횡오차 측정에만 사용한다. CNN 조향에 지도 경로를
섞지 않는다. 통신과 계산은 테스트할 수 있도록 분리했다.
"""
import math
from pathlib import Path
import select
import socket
import threading
import time

import numpy as np

from data_utils import BASE
from tools.collection.protocol import CameraAssembler, parse_ego


class LatestSensors:
    def __init__(self, cfg):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ego, self.ego_recv, self.frame, self.error = None, 0, None, None
        self.errors = 0
        self.sockets = {}
        try:
            for name, port in (("camera", cfg["cameras"]["center"]["port"]), ("ego", cfg["ego"]["port"])):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets[sock] = name
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 ** 2)
                sock.bind((cfg["bind_ip"], port))
                sock.setblocking(False)
        except BaseException:
            for sock in self.sockets:
                sock.close()
            raise
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        assembler = CameraAssembler()
        try:
            while not self.stop.is_set():
                ready, _, _ = select.select(list(self.sockets), [], [], .05)
                for sock in ready:
                    for _ in range(256):
                        try:
                            packet = sock.recv(65535)
                        except BlockingIOError:
                            break
                        recv = time.monotonic_ns()
                        try:
                            if self.sockets[sock] == "ego":
                                state = parse_ego(packet)
                                with self.lock:
                                    if self.ego is None or state["stamp_ns"] >= self.ego["stamp_ns"]:
                                        self.ego, self.ego_recv = state, recv
                            else:
                                frame = assembler.push(packet)
                                if frame:
                                    with self.lock:
                                        if self.frame is None or frame[0] > self.frame[0]:
                                            self.frame = (frame[0], recv, frame[1])
                        except ValueError:
                            self.errors += 1
        except BaseException as exc:
            self.error = str(exc)

    def snapshot(self):
        with self.lock:
            return self.ego, self.ego_recv, self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        for sock in self.sockets:
            sock.close()


class Reference:
    limits = {"A": (0., 55.), "D": (650., 850.), "J": (40., 108.),
              "U1": (1300., 1500.), "U2": (978.337, 1210.)}

    def __init__(self, course):
        self.start, self.end = self.limits[course]
        raw = np.loadtxt(BASE / "assets/evaluation_route.csv", delimiter=",", skiprows=1)
        self.points = raw[(raw[:, 0] >= max(0., self.start - 15)) & (raw[:, 0] <= self.end + 20)]
        self.s, self.xy = self.points[:, 0], self.points[:, 1:3]
        self.edges = np.diff(self.xy, axis=0)
        self.length2 = np.sum(self.edges ** 2, axis=1)
        if np.any(self.length2 <= 0):
            raise ValueError("degenerate reference segments")

    def pose(self, lateral=0., heading_offset_deg=0.):
        i = min(len(self.s) - 2, max(0, int(np.searchsorted(self.s, self.start) - 1)))
        yaw = math.atan2(self.edges[i, 1], self.edges[i, 0])
        xyz = [float(np.interp(self.start, self.s, self.points[:, j])) for j in (1, 2, 3)]
        xyz[0] -= math.sin(yaw) * lateral
        xyz[1] += math.cos(yaw) * lateral
        # 횡이동은 도로 접선 기준, 헤딩만 별도로 회전한다. 위치를 함께 회전하지 않는다.
        return xyz + [math.degrees(yaw) + heading_offset_deg]

    def project(self, state):
        pos = np.array([state["x_m"], state["y_m"]])
        ratio = np.clip(np.sum((pos - self.xy[:-1]) * self.edges, axis=1) / self.length2, 0, 1)
        projected = self.xy[:-1] + ratio[:, None] * self.edges
        j = int(np.argmin(np.sum((pos - projected) ** 2, axis=1)))
        tangent = math.atan2(self.edges[j, 1], self.edges[j, 0])
        cte = float(np.dot(pos - projected[j], [-math.sin(tangent), math.cos(tangent)]))
        station = float(self.s[j] + ratio[j] * (self.s[j+1] - self.s[j]))
        heading = (state["yaw_deg"] - math.degrees(tangent) + 180) % 360 - 180
        return station, cte, heading


def curvature_to_degrees(curvature, contract):
    if not math.isfinite(curvature):
        raise ValueError("non-finite curvature")
    return math.degrees(math.atan(-curvature * contract["wheelbase_m"])) / (
        contract["steer_left_sign"] * contract["road_wheel_gain"])


def speed_command(speed_kph, target_kph, limit_kph=20.):
    # 종방향은 고정 목표속도 피드백이다. 신경망은 조향만 결정한다.
    if not all(math.isfinite(v) for v in (speed_kph, target_kph)):
        return 0., 1.
    demand = 1.5 * ((target_kph - speed_kph) / 3.6)
    if speed_kph >= limit_kph - .3:
        demand = min(demand, -.6)
    if speed_kph >= limit_kph:
        demand = min(demand, -3.2)
    return (min(.25, max(0., demand / 5.2)), min(.7, max(0., -demand / 9.)))
