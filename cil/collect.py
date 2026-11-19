import argparse  # 터미널 실행 옵션을 읽는다.
import csv  # 이미지 목록·차량 상태·제어값을 CSV로 저장한다.
import hashlib  # 사용한 경로 파일을 해시로 식별한다.
import json  # 수집 설정과 종료 상태를 JSON으로 저장한다.
import math  # 조향각·거리·정지 속도를 계산한다.
import time  # 제어 주기와 센서 경과 시간을 측정한다.
import uuid  # 수집 폴더 이름의 중복을 방지한다.
import random  # 재현 가능한 시간상관 조향 교란을 만든다.
import shutil  # 수집에 사용한 코드 사본을 세션에 보존한다.

from collections import Counter, deque  # 저장 개수와 미완료 작업을 관리한다.
from concurrent.futures import ThreadPoolExecutor  # 저장 작업을 별도 스레드에서 실행한다.
from contextlib import ExitStack  # 여러 CSV 파일을 함께 정리한다.
from datetime import datetime, timezone  # UTC 기준으로 수집 ID를 만든다.

import numpy as np  # 경로 투영과 벡터 계산에 사용한다.

from morai import BASE, MoraiClient, load_config  # 앞에서 작성한 통신 기능을 가져온다.


IMAGE_FIELDS = [  # images.csv에 기록할 열 이름이다.
    "stamp_ns",  # 이미지의 MORAI 센서 시각이다.
    "recv_ns",  # PC에서 수신한 시각이다.
    "camera",  # left·center·right 중 하나다.
    "image_path",  # 세션 폴더 기준 이미지 상대경로다.
]

VEHICLE_FIELDS = [  # vehicle.csv에 기록할 열 이름이다.
    "stamp_ns",  # 차량 상태의 MORAI 센서 시각이다.
    "recv_ns",  # PC에서 차량 상태를 수신한 시각이다.
    "ctrl_mode",  # 차량 제어 모드다.
    "gear",  # 현재 기어다.
    "speed_kph",  # 차량 속도다.
    "accel",  # 상태 패킷의 가속 페달 값이다.
    "brake",  # 상태 패킷의 브레이크 값이다.
    "x_m",  # 지도 좌표계의 차량 X 위치다.
    "y_m",  # 지도 좌표계의 차량 Y 위치다.
    "z_m",  # 지도 좌표계의 차량 Z 위치다.
    "roll_deg",  # 차량 Roll이다.
    "pitch_deg",  # 차량 Pitch다.
    "yaw_deg",  # 차량 헤딩이다.
    "vx_kph",  # 속도 벡터 X 성분이다.
    "vy_kph",  # 속도 벡터 Y 성분이다.
    "vz_kph",  # 속도 벡터 Z 성분이다.
    "yaw_rate_deg_s",  # 초당 헤딩 변화량이다.
    "steer_raw_deg",  # 상태 패킷의 실제 조향 값이다.
    "link_id",  # 상태 패킷의 지도 링크 ID다.
]

COMMAND_FIELDS = [  # commands.csv에 기록할 열 이름이다.
    "ego_stamp_ns",  # 제어 계산에 사용한 차량 상태의 센서 시각이다.
    "sent_ns",  # 제어 명령을 보낸 직후의 PC 시각이다.
    "command",  # follow_road·left·straight·right의 명령 번호다.
    "teacher_steer",  # Teacher가 계산한 조향 명령이다.
    "teacher_throttle",  # Teacher가 계산한 가속 페달이다.
    "teacher_brake",  # Teacher가 계산한 브레이크다.
    "noise_steer",  # 조향에 추가한 교란이다. 현재 버전에서는 0이다.
    "applied_steer",  # 실제 MORAI에 전송한 조향이다.
    "applied_throttle",  # 실제 MORAI에 전송한 가속 페달이다.
    "applied_brake",  # 실제 MORAI에 전송한 브레이크다.
    "route_s_m",  # 경로 시작부터 진행한 거리다.
    "cte_m",  # 경로 중심 대비 횡오차다.
    "heading_error_deg",  # 경로 진행 방향 대비 헤딩 오차다.
    "target_kph",  # 해당 제어 주기의 목표 속도다.
    "control_phase",  # drive와 임의 경로 끝 정차를 구분한다.
    "label_valid",  # 목표 정차처럼 입력만으로 설명할 수 없는 정답은 제외한다.
    "noise_episode",  # 같은 교란·복귀 사건을 묶는 번호다.
    "recovery_active",  # 교란과 그 직후 복귀 구간을 표시한다.
]


def write_json(path, data):  # 세션 정보를 JSON 파일로 저장한다.
    text = json.dumps(data, ensure_ascii=False, indent=2)  # 한글과 들여쓰기를 유지한다.
    path.write_text(text + "\n", encoding="utf-8")  # UTF-8 파일로 저장한다.


def wrap_deg(angle):  # 각도 차이를 -180~180도 범위로 정리한다.
    return (angle + 180.0) % 360.0 - 180.0  # 179도와 -179도의 차이를 작게 처리한다.


class Route:  # 경로 CSV와 현재 차량 위치 사이의 관계를 계산한다.
    def __init__(self, path, navigation=None):  # 경로 CSV와 명령 구간을 받는다.
        points = np.loadtxt(  # 경로 좌표를 NumPy 배열로 읽는다.
            path,  # 입력 CSV 경로다.
            delimiter=",",  # 쉼표로 구분된 파일이다.
            skiprows=1,  # s_m,x_m,y_m,z_m 헤더를 건너뛴다.
            ndmin=2,  # 결과를 항상 2차원 배열로 유지한다.
        )

        if points.shape[1] != 4 or len(points) < 2:  # 열 개수와 경로점 수를 검사한다.
            raise ValueError("invalid route CSV shape")  # 잘못된 경로를 거부한다.

        if not np.isfinite(points).all():  # NaN이나 무한대 좌표를 검사한다.
            raise ValueError("non-finite route coordinate")  # 비정상 경로를 거부한다.

        keep = np.r_[True, np.linalg.norm(np.diff(points[:, 1:3], axis=0), axis=1) > 1e-3]  # 반올림으로 생긴 1mm 미만 중복 점을 제거한다.
        points = points[keep]  # 누적 거리는 원본 값을 유지한다.
        if len(points) < 2:  # 모든 점이 같은 위치인 경로를 거부한다.
            raise ValueError("route needs two distinct points")  # 투영과 곡률 계산에 두 점 이상이 필요하다.

        self.s = points[:, 0]  # 경로 시작부터의 누적 거리다.
        self.xy = points[:, 1:3]  # 경로의 평면 XY 좌표다.
        self.edges = np.diff(self.xy, axis=0)  # 인접 경로점 사이의 벡터다.
        self.length2 = np.sum(self.edges ** 2, axis=1)  # 각 선분 길이의 제곱이다.

        if np.any(np.diff(self.s) <= 0):  # 누적 거리가 증가하는지 검사한다.
            raise ValueError("route station must increase")  # 역순·중복 거리를 거부한다.

        if np.any(self.length2 <= 1e-8):  # 거의 같은 위치의 인접 점을 검사한다.
            raise ValueError("route contains a zero-length segment")  # 투영 시 0 나눗셈을 막는다.

        self.progress = float(self.s[0])  # 처음에는 경로 시작에 있다고 가정한다.
        self.navigation = navigation or []  # 실제 교차로 링크에 지정한 명령이다.
        self.lookahead_m = 3.0  # 도심 급커브에서 코너를 과하게 자르지 않게 한다.
        back = np.maximum(self.s[0], self.s - 1.5)  # 곡률 계산용 뒤쪽 거리다.
        front = np.minimum(self.s[-1], self.s + 1.5)  # 곡률 계산용 앞쪽 거리다.
        chord = np.column_stack([np.interp(front, self.s, self.xy[:, k]) - np.interp(back, self.s, self.xy[:, k]) for k in (0, 1)])  # 3m 접선이다.
        yaw = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))  # 각도 경계를 연속화한다.
        self.kappa = np.abs(np.gradient(yaw, self.s))  # 지도 곡률의 크기다.

    def command(self, station):  # 숫자를 보간하지 않고 현재 구간의 범주 명령을 고른다.
        inside = [n for n in self.navigation if n["entry_s_m"] <= station <= n["exit_s_m"]]  # 교차로 내부다.
        if inside:  # 실제로 진입 중인 교차로 명령을 우선한다.
            return inside[0]["command"]  # 해당 링크의 선택된 출구다.
        upcoming = [n for n in self.navigation if 0 < n["entry_s_m"] - station <= n["approach_m"]]  # 접근 중이다.
        if upcoming:  # 가까운 다음 교차로의 명령을 미리 준다.
            return min(upcoming, key=lambda n: n["entry_s_m"])["command"]  # 경로 순서를 따른다.
        passed = [n for n in self.navigation if 0 < station - n["exit_s_m"] <= n["exit_hold_m"]]  # 출구 정착 구간이다.
        return passed[-1]["command"] if passed else "follow_road"  # 일반 커브는 도로 추종이다.

    def make_speed_profile(self, cruise_kph):  # 곡선 진입 전에 감속할 속도 프로파일이다.
        profile = np.minimum(cruise_kph / 3.6, np.sqrt(1.8 / np.maximum(self.kappa, 1e-4)))  # 횡가속 기준이다.
        for i in range(len(profile) - 2, -1, -1):  # 뒤에서 앞으로 제동 여유를 계산한다.
            profile[i] = min(profile[i], math.sqrt(profile[i + 1] ** 2 + 2 * 1.2 * (self.s[i + 1] - self.s[i])))  # 완만한 선감속이다.
        self.speed_profile = profile * 3.6  # 제어기가 사용할 km/h로 보관한다.

    def track(self, ego, wheelbase, max_angle):  # 현재 상태로 Teacher 조향을 계산한다.
        position = np.array([ego["x_m"], ego["y_m"]])  # 차량 후륜축의 지도 위치다.

        lo = max(  # 검색할 경로의 뒤쪽 인덱스를 구한다.
            0,  # 배열 시작보다 뒤로 가지 않게 한다.
            int(np.searchsorted(self.s, self.progress - 3.0)) - 1,  # 약 3m 뒤까지 본다.
        )

        hi = min(  # 검색할 경로의 앞쪽 인덱스를 구한다.
            len(self.edges),  # 마지막 선분을 넘지 않게 한다.
            int(np.searchsorted(self.s, self.progress + 15.0)) + 1,  # 약 15m 앞까지 본다.
        )

        edges = self.edges[lo:hi]  # 현재 진행 위치 주변의 선분만 검사한다.
        delta = position - self.xy[lo:hi]  # 각 선분 시작점에서 차량까지의 벡터다.

        ratios = np.clip(  # 차량 위치를 각 선분에 투영하는 비율을 계산한다.
            np.sum(delta * edges, axis=1) / self.length2[lo:hi],  # 내적을 이용한 투영이다.
            0.0,  # 선분 시작점보다 뒤로 나가지 않게 한다.
            1.0,  # 선분 끝점보다 앞으로 나가지 않게 한다.
        )

        projected = self.xy[lo:hi] + ratios[:, None] * edges  # 선분 위의 투영점들이다.
        distances2 = np.sum((projected - position) ** 2, axis=1)  # 투영점까지의 거리 제곱이다.
        index = int(np.argmin(distances2))  # 차량에서 가장 가까운 선분을 선택한다.
        j = lo + index  # 전체 경로에서의 선분 번호로 변환한다.

        station = float(  # 투영된 위치의 누적 거리를 계산한다.
            self.s[j] + ratios[index] * (self.s[j + 1] - self.s[j])
        )

        self.progress = max(self.progress, station)  # 경로 진행 거리가 뒤로 점프하지 않게 한다.

        tangent = math.atan2(  # 선택한 경로 선분의 진행 방향이다.
            self.edges[j, 1],  # 선분의 Y 방향 성분이다.
            self.edges[j, 0],  # 선분의 X 방향 성분이다.
        )

        normal = np.array(  # 경로의 왼쪽을 향하는 단위 벡터다.
            [-math.sin(tangent), math.cos(tangent)]
        )

        cte = float(  # 경로 중심에서 차량까지의 횡방향 거리를 구한다.
            np.dot(position - projected[index], normal)  # 경로 왼쪽이 양수다.
        )

        heading = wrap_deg(  # 현재 차량과 경로 사이의 헤딩 차이를 구한다.
            ego["yaw_deg"] - math.degrees(tangent)
        )

        target_s = min(  # PP가 바라볼 목표점의 누적 거리다.
            self.progress + self.lookahead_m,  # 도심 Teacher의 목표점 거리다.
            float(self.s[-1]),  # 목표점이 경로 끝을 넘지 않게 한다.
        )

        target = np.array(  # 목표 거리에서의 XY 좌표를 보간한다.
            [
                np.interp(target_s, self.s, self.xy[:, k])  # X와 Y를 각각 보간한다.
                for k in (0, 1)  # 두 좌표 성분을 순회한다.
            ]
        )

        diff = target - position  # 차량에서 목표점까지의 지도 좌표 벡터다.
        yaw = math.radians(ego["yaw_deg"])  # 차량 헤딩을 라디안으로 변환한다.

        local_y = (  # 목표점의 차량 기준 왼쪽 방향 거리를 구한다.
            -math.sin(yaw) * diff[0] + math.cos(yaw) * diff[1]
        )

        distance2 = max(float(np.dot(diff, diff)), 1.0)  # 목표점 거리의 제곱에 최솟값을 둔다.
        curvature = 2.0 * local_y / distance2  # Pure Pursuit 곡률을 계산한다.
        angle = math.atan(wheelbase * curvature)  # 자전거 모델로 바퀴 조향각을 구한다.

        steer = float(  # MORAI의 정규화 조향 범위로 변환한다.
            np.clip(angle / math.radians(max_angle), -1.0, 1.0)
        )

        return self.progress, cte, heading, steer  # 추종 상태와 Teacher 조향을 반환한다.


class SessionWriter:  # 파일 저장을 통신·제어 루프와 분리한다.
    def __init__(self, output):  # 이번 수집의 저장 폴더를 받는다.
        self.output = output  # 이미지와 CSV의 기준 폴더다.
        self.stack = ExitStack()  # 열린 CSV 파일을 한 번에 정리한다.
        self.pending = deque()  # 완료 여부를 확인할 저장 작업 목록이다.
        self.counts = Counter()  # 저장이 끝난 데이터 개수다.
        self.writers = {}  # 종류별 CSV writer를 보관한다.
        self.files = []  # 주기적으로 flush할 파일을 보관한다.
        self.last_flush = time.monotonic()  # 마지막 flush 시각이다.

        try:  # 초기화 도중 실패해도 열린 파일을 정리한다.
            for name in ("left", "center", "right"):  # 카메라별 폴더를 만든다.
                folder = output / "images" / name  # 해당 카메라의 이미지 저장 위치다.
                folder.mkdir(parents=True, exist_ok=True)  # 필요한 상위 폴더도 만든다.

            schemas = {  # CSV 파일별 열 구성을 지정한다.
                "images": IMAGE_FIELDS,  # 이미지 메타데이터다.
                "vehicle": VEHICLE_FIELDS,  # 차량 상태다.
                "commands": COMMAND_FIELDS,  # Teacher와 실제 전송 명령이다.
            }

            for name, fields in schemas.items():  # CSV 세 개를 생성한다.
                path = output / (name + ".csv")  # CSV 파일 위치다.

                stream = self.stack.enter_context(  # 종료할 때 자동으로 닫도록 등록한다.
                    path.open("x", newline="", encoding="utf-8")  # 기존 파일은 덮어쓰지 않는다.
                )

                writer = csv.DictWriter(stream, fieldnames=fields)  # 열 이름을 고정한다.
                writer.writeheader()  # 첫 행에 열 이름을 기록한다.
                self.writers[name] = writer  # 이후 저장 작업에서 사용할 writer다.
                self.files.append(stream)  # flush할 파일 목록에도 추가한다.

            self.pool = ThreadPoolExecutor(max_workers=1)  # 한 스레드에서 순서대로 저장한다.

        except Exception:  # 초기화에 실패했다면
            self.stack.close()  # 이미 연 파일을 닫는다.
            raise  # 원래 오류를 전달한다.

    def add(self, event):  # 새로운 저장 작업을 등록한다.
        while self.pending and self.pending[0].done():  # 완료된 작업을 확인한다.
            self.pending.popleft().result()  # 저장 오류가 있었다면 여기서 예외가 발생한다.

        if len(self.pending) >= 128:  # 저장이 수신 속도를 따라가지 못한다면
            raise RuntimeError("disk writer backlog: stop capture")  # 데이터를 버리지 않고 알린다.

        future = self.pool.submit(self._write, event)  # 저장 작업을 별도 스레드에 맡긴다.
        self.pending.append(future)  # 나중에 완료 여부와 오류를 확인하도록 보관한다.

    def _write(self, event):  # 저장 스레드에서 실행하는 함수다.
        if event["kind"] == "camera":  # 카메라 이미지라면
            relative = (  # 세션 폴더 기준 상대경로를 만든다.
                f"images/{event['camera']}/{event['stamp_ns']}.jpg"
            )

            with (self.output / relative).open("xb") as stream:  # 중복 파일을 덮어쓰지 않는다.
                stream.write(event["jpeg"])  # 재압축하지 않고 원본 JPEG를 저장한다.

            row = {  # 이미지 목록에 기록할 시각과 카메라 역할이다.
                key: event[key]  # 이벤트에서 필요한 값을 가져온다.
                for key in ("stamp_ns", "recv_ns", "camera")  # 이미지 메타데이터 항목이다.
            }

            row["image_path"] = relative  # 저장한 이미지 경로를 추가한다.
            self.writers["images"].writerow(row)  # images.csv에 기록한다.
            self.counts[event["camera"]] += 1  # 카메라별 저장 개수를 증가시킨다.

        elif event["kind"] == "ego":  # 차량 상태 이벤트라면
            row = dict(  # 차량 상태와 PC 수신 시각을 합친다.
                event["data"],  # 속도·위치·조향 등의 상태다.
                recv_ns=event["recv_ns"],  # PC 수신 시각을 추가한다.
            )

            self.writers["vehicle"].writerow(row)  # vehicle.csv에 기록한다.
            self.counts["vehicle"] += 1  # 저장한 상태 개수를 증가시킨다.

        else:  # 이 파일에서 만든 제어 이벤트라면
            self.writers["commands"].writerow(event["data"])  # commands.csv에 기록한다.
            self.counts["commands"] += 1  # 저장한 제어 개수를 증가시킨다.

        if time.monotonic() - self.last_flush >= 1.0:  # 약 1초마다 CSV 버퍼를 비운다.
            for stream in self.files:  # 열려 있는 CSV마다
                stream.flush()  # Python의 쓰기 버퍼를 비운다.

            self.last_flush = time.monotonic()  # 마지막 flush 시각을 갱신한다.

    def close(self):  # 미완료 저장을 끝내고 파일을 닫는다.
        self.pool.shutdown(wait=True)  # 등록한 저장 작업이 끝날 때까지 기다린다.

        try:  # 종료 직전의 저장 오류도 확인한다.
            for future in self.pending:  # 아직 확인하지 않은 작업마다
                future.result()  # 저장 실패가 있으면 예외를 발생시킨다.

        finally:  # 저장 성공 여부와 관계없이
            self.stack.close()  # CSV 파일을 모두 닫는다.


def pedals(speed_kph, target_kph, cap_kph):  # 현재·목표 속도로 페달 값을 계산한다.
    demand = 1.5 * (target_kph - speed_kph) / 3.6  # 속도 오차를 가속도 요구량으로 바꾼다.

    if speed_kph >= cap_kph - 0.2:  # 상한 근처부터 가속을 억제한다.
        demand = min(demand, -0.8)  # 감속 요구를 적용한다.

    if target_kph < 0.2:  # 경로 끝에서 정차할 때는
        return 0.0, 0.6  # 가속을 끄고 브레이크를 유지한다.

    throttle = min(0.25, max(0.0, demand / 5.2))  # 가속 요구일 때만 페달을 누른다.
    brake = min(0.6, max(0.0, -demand / 9.0))  # 감속 요구일 때만 브레이크를 누른다.

    return throttle, brake  # 두 페달이 동시에 양수가 되지 않는 결과다.


def stop_vehicle(client, steer, hz):  # 수집 종료 시 차량을 제동한다.
    started = time.monotonic()  # 정지 제어를 시작한 시각이다.

    while time.monotonic() - started < 2.0:  # 최대 2초 동안 정지 제어를 수행한다.
        client.poll(0.0)  # 정지 여부를 확인할 최신 차량 상태를 받는다.
        client.send_control(steer, 0.0, 1.0)  # 마지막 조향을 유지하면서 제동한다.
        ego = client.latest_ego  # 최신 차량 상태다.

        age = (  # 마지막 정상 차량 상태를 받은 뒤 지난 시간이다.
            float("inf")  # 차량 상태가 없으면 오래된 상태로 취급한다.
            if ego is None  # 아직 수신한 상태가 없다면
            else (time.monotonic_ns() - client.latest_ego_recv_ns) / 1e9  # 초 단위다.
        )

        stopped = (  # 신선한 상태에서 충분히 느려졌는지 확인한다.
            ego is not None  # 차량 상태가 있어야 한다.
            and age < 0.2  # 최근 수신한 상태여야 한다.
            and abs(ego["speed_kph"]) < 0.5  # 거의 정지한 속도여야 한다.
            and time.monotonic() - started > 0.3  # 최소 제동 시간을 유지한다.
        )

        if stopped:  # 정지를 확인했다면
            break  # 제동 반복을 종료한다.

        time.sleep(1.0 / hz)  # 설정한 제어 주기를 유지한다.


class SteeringNoise:  # 시간에 따라 연속적으로 변하는 삼각형 조향 교란을 만든다.
    def __init__(self, enabled=False, seed=7, *, first_time_s=8.0, peak_range=(0.07, 0.10)):
        self.enabled = enabled  # Validation에서는 교란을 사용하지 않는다.
        self.rng = random.Random(seed)  # 같은 seed로 같은 교란 순서를 재현한다.
        self.next_time = first_time_s
        self.peak_range = peak_range
        self.active = None  # 현재 진행 중인 교란 사건이다.
        self.episode = 0  # 교란 사건의 일련번호다.
        self.recovery_until = -1.0  # 교란 이후 복귀를 관측할 마지막 시각이다.

    def value(self, elapsed, allowed):  # 센서 기준 경과 시간과 현재 적용 조건을 받는다.
        if not self.enabled:  # 정상 수집이나 Validation이면
            return 0.0, 0, False  # 조향 교란을 더하지 않는다.
        if self.active is not None:  # 교란이 진행 중이라면
            start, duration, peak = self.active  # 삼각파의 시작 시각·길이·크기다.
            if elapsed >= start + duration or not allowed:  # 지속 시간이 끝났거나 적용 조건을 벗어나면
                self.active = None  # 교란 추가를 중단한다.
                self.recovery_until = max(self.recovery_until, elapsed + 4.0)  # Teacher의 복귀 과정을 더 기록한다.
        if self.active is None and elapsed >= self.next_time and allowed:  # 다음 교란을 적용할 수 있는 구간이라면
            duration = self.rng.uniform(2.0, 2.8)  # 교란의 지속 시간이다.
            peak = self.rng.choice((-1, 1)) * self.rng.uniform(*self.peak_range)
            self.active = (elapsed, duration, peak)  # 새 삼각파를 시작한다.
            self.episode += 1  # 새 교란 사건 번호다.
            self.next_time = elapsed + duration + self.rng.uniform(18.0, 25.0)  # 교란을 연속해서 주입하지 않는다.
            self.recovery_until = elapsed + duration + 4.0  # 복귀 관측 구간의 끝이다.
        noise = 0.0  # 교란 중이 아니면 0이다.
        if self.active is not None:  # 진행 중인 삼각파를 계산한다.
            start, duration, peak = self.active  # 현재 교란의 설정값이다.
            fraction = min(1.0, max(0.0, (elapsed - start) / duration))  # 교란의 진행률이다.
            noise = peak * (1.0 - abs(2.0 * fraction - 1.0))  # 0에서 peak까지 증가한 뒤 0으로 줄어드는 연속 신호다.
        return noise, self.episode, elapsed <= self.recovery_until  # 교란값·사건 번호·복귀 상태를 반환한다.


def main():  # 경로별 Teacher 주행 데이터를 수집한다.
    parser = argparse.ArgumentParser()  # 실행 옵션을 정의한다.
    parser.add_argument("--config", default=BASE / "config.json")  # 센서와 저장 설정 파일이다.
    parser.add_argument("--route", default="T1")  # 수집할 경로 이름이다.
    parser.add_argument("--seconds", type=float, default=600.0)  # 주행 시작 후 최대 수집 시간이다.
    parser.add_argument("--describe", action="store_true")  # 설정과 경로 정보만 표시한다.
    parser.add_argument("--auto-reset", action="store_true")  # 로컬 MORAI에서 출발 위치로 배치한다.
    parser.add_argument("--noise", action="store_true")  # 학습 경로에서만 조향 교란을 활성화한다.
    recovery_mode = parser.add_mutually_exclusive_group()
    recovery_mode.add_argument("--turn-recovery", action="store_true", help="inject a small perturbation before a left or right turn")
    recovery_mode.add_argument("--junction-recovery", action="store_true", help="also collect recovery before straight-through junctions")
    recovery_mode.add_argument("--curve-recovery", action="store_true", help="collect teacher recovery inside a junction")
    parser.add_argument("--curve-noise-scale", type=float, default=1.0,
                        help="multiply the collection-only perturbation strength during a turn")
    parser.add_argument("--seed", type=int, default=7)  # 교란을 재현하기 위한 seed다.
    parser.add_argument("--speed", type=float, default=None)  # 상한 안에서 순항 목표를 지정한다.
    parser.add_argument("--hour", type=int, choices=range(9, 16), default=12, help="daylight hour for MORAI auto-reset")
    parser.add_argument("--initial-offset", type=float, default=0.0)  # 학습용 복귀 시범의 초기 횡오프셋이다.
    parser.add_argument("--initial-heading-offset", type=float, default=0.0,
                        help="initial heading perturbation in degrees for training collection")
    args = parser.parse_args()  # 실행 옵션을 읽는다.
    if not math.isfinite(args.seconds) or args.seconds <= 0:  # 수집 시간의 범위를 확인한다.
        parser.error("--seconds must be finite and positive")  # 잘못된 시간을 거부한다.
    if not 1.0 <= args.curve_noise_scale <= 2.0 or (args.curve_noise_scale != 1.0 and not args.curve_recovery):
        parser.error("--curve-noise-scale requires --curve-recovery and must be in [1, 2]")
    cfg = load_config(args.config)  # 설정 파일을 읽는다.
    plan = json.loads((BASE / cfg["paths"]["route_plan"]).read_text(encoding="utf-8"))  # 경로와 시험 구역을 정의한 계획이다.
    if args.route not in plan["routes"]:  # 존재하지 않는 경로라면
        parser.error("unknown route: " + args.route)  # 경로 이름을 확인하도록 알린다.
    info = plan["routes"][args.route]  # 경로의 용도·좌표·명령 구간이다.
    route_path = BASE / info["route_csv"]  # 실제로 실행할 경로 CSV다.
    route = Route(route_path, info.get("navigation", []))  # 교차로 명령이 연결된 경로다.
    print("Route:", args.route, "Split:", info["split"], "Start XYZ:", info["start_xyz_m"], "Yaw:", info["initial_yaw_deg_map"], flush=True)  # 출발 배치 정보를 출력한다.
    if args.describe:  # 설명만 요청했다면
        print("Navigation:", info.get("navigation", []))  # 명령 구간도 함께 표시한다.
        return 0  # 차량을 제어하지 않고 종료한다.
    if info["split"] == "test":  # 예약한 시험 경로는
        raise ValueError("test routes are excluded from collection")  # 데이터 수집에 사용하지 않는다.
    approach_recovery = args.turn_recovery or args.junction_recovery
    junction_noise = approach_recovery or args.curve_recovery
    cfg["collection"]["noise_enabled"] = bool(args.noise or junction_noise or cfg["collection"]["noise_enabled"])
    noisy = cfg["collection"]["noise_enabled"]  # 이번 수집의 교란 활성화 여부다.
    if (noisy or args.initial_offset != 0 or args.initial_heading_offset != 0) and info["split"] != "train":  # Validation과 점검용 경로에서는
        raise ValueError("noise and initial offsets are train-only")  # 학습용 교란이나 초기 위치 변화를 적용하지 않는다.
    if not math.isfinite(args.initial_offset) or abs(args.initial_offset) > 0.6:
        raise ValueError("initial offset must be within +/-0.6m")
    if not math.isfinite(args.initial_heading_offset) or abs(args.initial_heading_offset) > 8.0:
        raise ValueError("initial heading offset must be within +/-8deg")
    cap = float(cfg["collection"]["speed_cap_kph"])  # 설정된 속도 상한이다.
    hz = float(cfg["control"]["hz"])  # 제어 명령을 보낼 주기다.
    cruise = cap - 1.0 if args.speed is None else args.speed  # 기본 순항 목표는 19km/h다.
    if not 1.0 < cap <= 20.0 or not 1.0 <= hz <= 100.0 or not 0 < cruise < cap:  # 검증할 수집 설정 범위다.
        raise ValueError("require 0<cruise<cap<=20 kph and control rate 1~100 Hz")  # 잘못된 설정을 거부한다.
    route.make_speed_profile(cruise)  # 곡선 진입 전 감속 프로파일을 준비한다.
    tag = "noise" if noisy else "normal"  # 수집 조건을 폴더 이름에도 남긴다.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.route + "_" + tag + "_" + uuid.uuid4().hex[:6]  # 수집 실행의 고유 ID다.
    output = BASE / cfg["paths"]["raw_data"] / run_id  # 원본 데이터의 저장 위치다.
    output.mkdir(parents=True, exist_ok=False)  # 기존 수집 데이터를 덮어쓰지 않는다.
    if args.curve_recovery:
        peak_range = (0.07 * args.curve_noise_scale, 0.12 * args.curve_noise_scale)
    else:
        peak_range = (0.04, 0.06) if approach_recovery else (0.07, 0.10)
    metadata = {  # 수집 조건을 기록한다.
        "schema_version": 2, "status": "recording", "route_id": args.route, "split": info["split"],
        "teacher": "map_pure_pursuit", "human_driving": False, "noise_enabled": noisy,
        "seed": args.seed, "initial_offset_m": args.initial_offset, "cruise_kph": cruise,
        "initial_heading_offset_deg": args.initial_heading_offset,
        "daylight_hour": args.hour if args.auto_reset else None,
        "command_policy": "map_intersection_link_with_selected_exit", "route_csv": info["route_csv"],
        "route_sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(), "navigation": info.get("navigation", []),
        "config": cfg, "software_sha256": {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in ("collect.py", "morai.py", "tools/simulator.py")},
        "noise_parameters": {"signal": "triangular", "duration_s": [2.0, 2.8], "peak_normalized": list(peak_range), "recovery_s": 4.0,
                             "location": "junction entry and interior" if args.curve_recovery
                             else "3–15m before a junction" if args.junction_recovery
                             else "3–15m before a turn" if args.turn_recovery else "straight follow_road"},
        "collision_ground_truth": "not_instrumented; tracking and images are checked separately",
    }
    if args.auto_reset:  # 옵션을 지정했을 때만 시뮬레이터에서 차량을 재배치한다.
        from tools.simulator import Simulator  # CIL 수집용 로컬 MORAI API 어댑터다.
        metadata["actual_start_state"] = Simulator().setup(info, cfg, args.initial_offset, hour=args.hour,
                                                         heading_offset_deg=args.initial_heading_offset)
    rig = BASE / "runs/collection_setup/sensor_rig.json"  # 확인한 센서 설정의 기록이다.
    if rig.exists():  # 센서 설정 증거가 있다면
        metadata["rig_snapshot"] = json.loads(rig.read_text())  # 실행 조건과 함께 보존한다.
    write_json(output / "session.json", metadata)  # 수집 시작 상태를 저장한다.
    for name in metadata["software_sha256"]:  # 나중에 코드가 바뀌어도 실행 버전을 재현한다.
        snapshot = output / "software" / name  # 이 세션에서 사용한 코드 사본 위치다.
        snapshot.parent.mkdir(parents=True, exist_ok=True)  # tools 하위 폴더도 만든다.
        shutil.copy2(BASE / name, snapshot)  # 실제 실행 파일을 그대로 보존한다.
    writer = SessionWriter(output)  # 비동기 파일 저장기를 만든다.
    noise = SteeringNoise(noisy, args.seed, first_time_s=0.0 if junction_noise else 8.0, peak_range=peak_range)
    recovery_commands = ("left", "right", "straight") if args.junction_recovery or args.curve_recovery else ("left", "right")
    reason, failure, capture_stats = "starting", "", {}  # 종료 상태와 오류·수신 통계를 준비한다.
    print("Saving:", output, flush=True)  # 실제 저장 위치를 출력한다.
    try:  # 제어·저장 실패에도 최종 정리를 수행한다.
        with MoraiClient(cfg) as client:  # 센서 수신과 제어 송신 소켓을 연다.
            armed, last_steer = False, 0.0  # 아직 주행을 시작하지 않은 상태다.
            cameras, stamps = {}, {}  # 스트림별 최근 시각을 보관한다.
            waiting = report = time.monotonic()  # 센서 대기와 상태 출력의 기준 시각이다.
            try:  # 주행이 중단돼도 정지 제어를 실행한다.
                while True:  # 경로 완주 또는 오류까지 반복한다.
                    for receive_attempt in range(2):  # PC 실행이 잠깐 밀렸다면 큐에 도착한 최신 패킷을 한 번 더 읽는다.
                        for event in client.poll(0.002):  # 도착한 센서 데이터를 수신한다.
                            key = "ego" if event["kind"] == "ego" else event["camera"]  # 데이터가 들어온 센서 이름이다.
                            previous = stamps.get(key, -1)  # 이전 센서 시각이다.
                            if event["stamp_ns"] < previous - 1_000_000_000:  # 시뮬레이터의 시각이 크게 뒤로 돌아갔다면
                                raise RuntimeError("sensor clock went backwards; start a new session")  # 서로 다른 시뮬레이션 실행의 데이터를 섞지 않는다.
                            stamps[key] = max(previous, event["stamp_ns"])  # 작은 UDP 도착 순서 뒤바뀜은 허용한다.
                            if event["kind"] == "camera": cameras[key] = event["recv_ns"]  # 영상이 최근에 수신됐는지 확인할 시각이다.
                            writer.add(event)  # 수신 원본을 저장한다.
                        now, now_ns = time.monotonic(), time.monotonic_ns()  # PC의 현재 시각이다.
                        ego = client.latest_ego  # 최신 차량 상태다.
                        age = float("inf") if ego is None else (now_ns - client.latest_ego_recv_ns) / 1e9  # 차량 상태를 받은 뒤 경과한 시간이다.
                        images_fresh = all((now_ns - cameras.get(name, 0)) / 1e9 < 0.35 for name in cfg["cameras"])  # 카메라 세 개의 수신 상태를 확인한다.
                        fresh = age <= cfg["collection"]["max_ego_age_s"] and images_fresh  # 모든 센서의 준비 상태다.
                        if fresh or not armed: break  # 센서 공백이 실제로 계속되는 경우에만 아래에서 중단한다.
                    if not armed:  # 주행 시작 전이라면
                        if not fresh:  # 센서 데이터가 모이기를 기다린다.
                            if now - waiting > 10.0: raise RuntimeError("missing fresh cameras/Ego")  # 오랫동안 센서가 없으면 수집을 중단한다.
                            continue  # 아직 주행 제어를 시작하지 않는다.
                        if ego["ctrl_mode"] != 2 or ego["gear"] != 4: raise RuntimeError("select AV-ExternalCtrl / Auto and D")  # 외부 제어 모드와 D 기어를 확인한다.
                        distance = math.hypot(ego["x_m"] - info["start_xyz_m"][0], ego["y_m"] - info["start_xyz_m"][1])  # 목표 출발점과 현재 위치의 거리다.
                        heading0 = abs(wrap_deg(ego["yaw_deg"] - info["initial_yaw_deg_map"] - args.initial_heading_offset))
                        if distance > 1.0 or heading0 > 10.0 or abs(ego["speed_kph"]) > 1.0: raise RuntimeError("place the stopped vehicle at route start")  # 정차한 차량이 출발점에 있는지 확인한다.
                        if abs(ego["z_m"] - info["start_xyz_m"][2]) > 1.0: raise RuntimeError("start height mismatch")  # 입체 도로에서 다른 높이의 도로를 구분한다.
                        armed, started, next_tick, initial_stamp = True, now, now, ego["stamp_ns"]  # 주행 제어를 시작한다.
                    if not fresh or ego["ctrl_mode"] != 2 or ego["gear"] != 4:  # 주행 중에도 센서와 제어 모드를 확인한다.
                        camera_ages = {name: round((now_ns - cameras.get(name, 0)) / 1e9, 3) for name in cfg["cameras"]}  # 중단한 센서를 구분할 진단값이다.
                        raise RuntimeError(f"sensor/control check failed: ego_age={age:.3f}s, cameras={camera_ages}, mode={ego['ctrl_mode']}, gear={ego['gear']}")  # 실제 원인을 기록하고 제동한다.
                    if now - started >= args.seconds: reason = "time_limit"; break  # 시간 제한 종료를 완주와 구분해 기록한다.
                    if now < next_tick: continue  # 제어 주기 사이에도 센서 수신을 계속한다.
                    next_tick = now + 1.0 / hz  # 지연이 생겨도 명령을 한꺼번에 몰아 보내지 않는다.
                    station, cte, heading, teacher_steer = route.track(ego, cfg["vehicle"]["wheelbase_m"], cfg["vehicle"]["max_wheel_angle_deg"])  # 현재 차량 상태에서 Teacher 출력을 다시 계산한다.
                    if abs(cte) > 0.8 or abs(heading) > 25 or ego["speed_kph"] < -0.5: raise RuntimeError(f"teacher tracking error: cte={cte:.3f}, heading={heading:.2f}")  # 정상 시범으로 인정할 추종 오차의 상한이다.
                    if ego["speed_kph"] > cap + 1.0: raise RuntimeError("teacher exceeded speed ceiling")  # 속도 제어 이상을 확인한다.
                    remaining = max(0.0, float(route.s[-1]) - station)  # 경로 끝까지 남은 거리다.
                    command_name = route.command(station)  # 선택한 교차로 출구의 명령을 읽는다.
                    normal_target = float(np.interp(station, route.s, route.speed_profile))  # 곡선 감속을 반영한 목표 속도다.
                    stopping_target = 3.6 * math.sqrt(1.6 * max(0.0, remaining - 0.8))  # 경로 끝에서 정차하기 위한 목표 속도다.
                    target = min(normal_target, stopping_target)  # 가장 낮은 속도 제한을 적용한다.
                    phase = "terminal_stop" if stopping_target < normal_target - 0.05 else "drive"  # 임의 경로 끝에서의 정차를 일반 주행과 구분한다.
                    throttle, brake = pedals(ego["speed_kph"], target, cap)  # Teacher가 계산한 가속·브레이크다.
                    elapsed = max(0.0, (ego["stamp_ns"] - initial_stamp) / 1e9)  # 센서 시간축 기준 경과 시간이다.
                    local_kappa = float(np.interp(station, route.s, route.kappa))  # 현재 경로의 곡률이다.
                    allowed = command_name == "follow_road" and phase == "drive" and ego["speed_kph"] > 10 and abs(cte) < 0.45 and local_kappa < 0.035 and remaining > 25  # 조향 교란을 적용할 조건이다.
                    if approach_recovery:
                        # 회전 전 직선에서 작게 벗어나게 하고, 진입 이후에는 Teacher 복귀만 기록한다.
                        approaching_turn = any(n["command"] == command_name and 3 < n["entry_s_m"] - station <= 15
                                               for n in route.navigation)
                        allowed = (command_name in recovery_commands and approaching_turn and phase == "drive"
                                   and ego["speed_kph"] > 8 and abs(cte) < 0.40 and local_kappa < 0.035)
                    elif args.curve_recovery:
                        in_turn = any(n["command"] == command_name
                                      and n["entry_s_m"] - 1. <= station < n["exit_s_m"] - 4.
                                      for n in route.navigation)
                        allowed = (command_name in recovery_commands and in_turn and phase == "drive"
                                   and ego["speed_kph"] > 8 and abs(cte) < 0.40)
                    requested_noise, episode, recovery = noise.value(elapsed, allowed)  # 삼각형 교란과 복귀 상태를 계산한다.
                    applied_steer = float(np.clip(teacher_steer + requested_noise, -1.0, 1.0))  # 실제로 전송할 조향 명령이다.
                    sent_ns = client.send_control(applied_steer, throttle, brake)  # 교란이 포함된 명령을 전송한다.
                    last_steer = applied_steer  # 종료 제동에서 유지할 마지막 조향이다.
                    row = {  # Teacher 정답과 실제 전송값을 구분해 기록한다.
                        "ego_stamp_ns": ego["stamp_ns"], "sent_ns": sent_ns, "command": cfg["commands"][command_name],
                        "teacher_steer": teacher_steer, "teacher_throttle": throttle, "teacher_brake": brake,
                        "noise_steer": applied_steer - teacher_steer, "applied_steer": applied_steer, "applied_throttle": throttle, "applied_brake": brake,
                        "route_s_m": station, "cte_m": cte, "heading_error_deg": heading, "target_kph": target,
                        "control_phase": phase, "label_valid": int(phase == "drive"), "noise_episode": episode, "recovery_active": int(recovery),
                    }
                    writer.add({"kind": "control", "data": row})  # 제어 명령을 파일에 기록한다.
                    if now - report > 2.0:  # 약 2초마다 진행 상황을 표시한다.
                        print(f"s={station:.1f}/{route.s[-1]:.1f}m speed={ego['speed_kph']:.1f} cte={cte:.3f} cmd={command_name} noise={row['noise_steer']:.3f}", flush=True)  # 실제 수신한 주행 상태다.
                        report = now  # 마지막 출력 시각을 갱신한다.
                    if remaining < 1.5 and abs(ego["speed_kph"]) < 0.8: reason = "route_finished"; break  # 경로 끝에서 정지했으면 완주다.
            finally:  # 성공·오류·Ctrl+C에 공통으로 실행할 정리 작업이다.
                capture_stats = {"received": dict(client.counts), "parse_errors": dict(client.errors), "incomplete_frames": {k: a.evicted for k, a in client.assemblers.items()}}  # 정지 제어 전까지의 수신 품질 통계다.
                if armed: stop_vehicle(client, last_steer, hz)  # 통신 소켓을 닫기 전에 제동한다.
    except KeyboardInterrupt: reason = "interrupted"  # 사용자 중단을 완주와 구분한다.
    except Exception as exc: reason, failure = "error", str(exc)  # 오류 원인을 남긴다.
    finally:  # 저장을 마치고 최종 상태를 기록한다.
        try: writer.close()  # 대기 중인 저장 작업을 끝낸다.
        except Exception as exc: reason, failure = "error", (failure + " " + str(exc)).strip()  # 디스크 저장 실패도 오류로 기록한다.
        metadata.update(status=reason, error=failure, saved=dict(writer.counts), training_ready=False, capture_stats=capture_stats, noise_episodes=noise.episode)  # 학습용 시각 매칭과 품질 검사는 별도로 진행한다.
        write_json(output / "session.json", metadata)  # 최종 세션 상태를 저장한다.
    print("Finished:", reason, failure, "Saved:", dict(writer.counts), flush=True)  # 수집 결과를 출력한다.
    return 0 if reason == "route_finished" else 1  # 완주 여부를 종료 코드로도 구분한다.


if __name__ == "__main__":  # 다른 파일에서 import할 때는 주행하지 않는다.
    raise SystemExit(main())  # 직접 실행했을 때만 수집을 시작한다.
