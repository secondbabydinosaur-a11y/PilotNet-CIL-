"""CIL의 가운데 영상·속도·명령 기반 MORAI 주행과 실제 창 녹화.

기본 시험에서는 지도 좌표를 배치·진행 명령·평가에만 사용한다.
--assist-east-outer-left는 첫 좌회전에서만 지도 조향을 쓰는 별도 진단이다.
--check는 수신만, --describe는 시험 조건만 확인하며 --control부터 실제 제어한다.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np
import torch

from dataset import CROP, IMAGE_SIZE, SPEED_SCALE_KPH, preprocess_image
from model import CILModel
from morai import BASE, MoraiClient, load_config
from tools.simulator import Simulator


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()  # 평가에 사용한 파일을 식별한다.


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class LatestSensors:
    """추론·영상 저장 중에도 별도 스레드에서 최신 가운데 영상과 Ego를 받는다."""
    def __init__(self, config):
        receive_config = copy.deepcopy(config)  # 원래 센서 설정 파일은 수정하지 않는다.
        receive_config["cameras"] = {"center": config["cameras"]["center"]}
        self.client = MoraiClient(receive_config)  # 실제 추론에는 가운데 카메라만 사용한다.
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.ego, self.ego_recv, self.frame, self.error = None, 0, None, None
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        try:
            while not self.stop.is_set():
                for event in self.client.poll(.02):
                    with self.lock:
                        if event["kind"] == "ego":
                            if self.ego is None or event["stamp_ns"] >= self.ego["stamp_ns"]:
                                self.ego, self.ego_recv = event["data"], event["recv_ns"]
                        elif self.frame is None or event["stamp_ns"] > self.frame[0]:
                            self.frame = (event["stamp_ns"], event["recv_ns"], event["jpeg"])
        except BaseException as error:
            self.error = str(error)  # 수신 스레드 실패를 주행 루프에 전달한다.

    def snapshot(self):
        with self.lock:
            return self.ego, self.ego_recv, self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.client.close()  # 다음 시험이 같은 UDP 포트를 사용할 수 있게 닫는다.


class Reference:
    """시험 경로에 대한 위치·횡오차·진행 명령만 계산한다. 조향 계산은 없다."""
    def __init__(self, info):
        self.info = info
        self.path = BASE / info["route_csv"]
        points = np.loadtxt(self.path, delimiter=",", skiprows=1, ndmin=2)
        if points.shape[1] != 4 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("invalid evaluation route")
        keep = np.r_[True, np.linalg.norm(np.diff(points[:, 1:3], axis=0), axis=1) > .001]
        self.points = points[keep]
        self.s, self.xy = self.points[:, 0], self.points[:, 1:3]
        self.edges = np.diff(self.xy, axis=0)
        self.length2 = np.sum(self.edges ** 2, axis=1)
        if len(self.s) < 2 or np.any(np.diff(self.s) <= 0) or np.any(self.length2 <= 1e-8):
            raise ValueError("degenerate evaluation route")
        self.progress = float(self.s[0])

    def project(self, ego):
        position = np.array([ego["x_m"], ego["y_m"]])
        lo = max(0, int(np.searchsorted(self.s, self.progress - 3.)) - 1)
        hi = min(len(self.edges), int(np.searchsorted(self.s, self.progress + 15.)) + 1)
        delta, edges = position - self.xy[lo:hi], self.edges[lo:hi]
        ratios = np.clip(np.sum(delta * edges, axis=1) / self.length2[lo:hi], 0., 1.)
        projected = self.xy[lo:hi] + ratios[:, None] * edges
        index = int(np.argmin(np.sum((projected - position) ** 2, axis=1)))
        j = lo + index
        station = float(self.s[j] + ratios[index] * (self.s[j + 1] - self.s[j]))
        self.progress = max(self.progress, station)  # 자기 교차점으로 진행률이 점프하지 않게 검색 범위를 제한했다.
        tangent = math.atan2(self.edges[j, 1], self.edges[j, 0])
        cte = float(np.dot(position - projected[index], [-math.sin(tangent), math.cos(tangent)]))
        heading = (ego["yaw_deg"] - math.degrees(tangent) + 180.) % 360. - 180.
        return station, cte, heading

    def command(self, station):
        navigation = self.info.get("navigation", [])  # 수집 때와 같은 교차로 명령 유지 규칙이다.
        inside = [n for n in navigation if n["entry_s_m"] <= station <= n["exit_s_m"]]
        if inside:
            return inside[0]["command"]
        upcoming = [n for n in navigation if 0 < n["entry_s_m"] - station <= n["approach_m"]]
        if upcoming:
            return min(upcoming, key=lambda n: n["entry_s_m"])["command"]
        passed = [n for n in navigation if 0 < station - n["exit_s_m"] <= n["exit_hold_m"]]
        return passed[-1]["command"] if passed else "follow_road"


def actuator_control(raw, speed_kph, cap_kph):
    """선형 출력을 제어 범위로 제한하고, 큰 페달 하나만 적용한다."""
    if not math.isfinite(speed_kph) or not math.isfinite(cap_kph) or cap_kph <= 0:
        raise ValueError("invalid speed or speed cap")
    values = np.asarray(raw, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("invalid CIL control output")
    steer = float(np.clip(values[0], -1., 1.))  # 정규화 조향이며 곡률 변환은 하지 않는다.
    throttle, braking = np.clip(values[1:], 0., 1.)
    if braking >= throttle:
        throttle = 0.
    else:
        braking = 0.  # 작은 회귀 오차 때문에 항상 브레이크가 우선하지 않게 한다.
    reason = "model"
    if speed_kph >= cap_kph:
        throttle, braking = 0., max(float(braking), .15)
        reason = "speed_cap"  # 속도 상한만 제한하며 목표속도 PI 제어를 섞지 않는다.
    return steer, float(throttle), float(braking), reason


def load_model(checkpoint_arg, device):
    selection = json.loads((BASE / "checkpoints/selected.json").read_text()) if checkpoint_arg is None else None
    path = Path(checkpoint_arg or selection["checkpoint"])
    path = path if path.is_absolute() else BASE / path
    digest = sha256(path)
    if selection is not None and digest != selection["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    expected = {"crop": list(CROP), "image_size_wh": list(IMAGE_SIZE), "color": "RGB",
                "pixel_range": [0, 1], "speed_scale_kph": SPEED_SCALE_KPH}
    if (settings["architecture"] != CILModel.architecture
            or settings["command_names"] != list(CILModel.command_names)
            or settings["control_order"] != ["steer", "throttle", "brake"]
            or settings["steering_unit"] != "normalized_-1_1"
            or json.loads(json.dumps(settings["preprocess"])) != expected):
        raise ValueError("checkpoint input/output contract mismatch")
    for name in ("model.py", "dataset.py", "config.json"):
        if sha256(BASE / name) != settings["hashes"][name]:
            raise ValueError(name + " changed since training")
    model = CILModel().to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.inference_mode():
        model(torch.zeros(1, 3, 88, 200, device=device), torch.zeros(1, 1, device=device),
              torch.zeros(1, dtype=torch.long, device=device))  # GPU 준비는 출발 전에 끝낸다.
    return model, checkpoint, path, digest


def hold_brake(client, seconds=1.5, steer=0.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        client.send_control(steer, 0., 1.)
        time.sleep(.02)


def wait_sensors(sensors, seconds=10.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if sensors.error:
            raise RuntimeError(sensors.error)
        ego, received, frame = sensors.snapshot()
        now = time.monotonic_ns()
        if ego is not None and frame is not None and (now - received) < 200_000_000 and (now - frame[1]) < 250_000_000:
            return ego, received, frame
        time.sleep(.02)
    raise RuntimeError("fresh center camera or Ego packets missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    east_outer_courses = ("EAST_OUTER", "EAST_OUTER_EXTENDED", "EAST_OUTER_MIDDLE_LEFT_RIGHT")
    parser.add_argument("--course", choices=("U2", "J_straight", "J_left", "J_right", "COMPETITION",
                                             "COMPETITION_REVERSE", *east_outer_courses,
                                             "EAST_OUTER_EXTENDED_RIGHT"), default="U2")
    parser.add_argument("--checkpoint")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--record-screen", action="store_true")
    parser.add_argument("--show-command", action="store_true", help="show the recorded command banner at the top of the live MORAI window")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seconds", type=float, default=240.)
    parser.add_argument("--max-cte", type=float, default=7.)  # 사용자가 추종 경향을 볼 수 있도록 정한 중단 기준이다.
    parser.add_argument("--speed-cap-kph", type=float,
                        help="lower the speed cap for this evaluation run only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--assist-east-outer-left", action="store_true",
                        help="diagnostic only: assist steering through the first EAST_OUTER left turn")
    args = parser.parse_args()
    if args.assist_east_outer_left and (args.course not in east_outer_courses or not args.control
                                       or args.check or args.describe or not args.output):
        parser.error("left assist requires an EAST_OUTER course, --control, and a dedicated --output")
    if args.show_command:
        args.record_screen = True  # 실제 화면 녹화와 실시간 창이 같은 명령 원본을 사용한다.
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.max_cte)):
        parser.error("seconds and max-cte must be finite and positive")
    config = load_config()
    if args.speed_cap_kph is not None:
        default_cap = config["collection"]["speed_cap_kph"]
        if not math.isfinite(args.speed_cap_kph) or not 1.0 < args.speed_cap_kph <= default_cap:
            parser.error(f"--speed-cap-kph must be greater than 1 and at most {default_cap:g}")
        config["collection"]["speed_cap_kph"] = args.speed_cap_kph
    if args.check:
        sensors = LatestSensors(config)
        try:
            ego, received, frame = wait_sensors(sensors)
            image = cv2.imdecode(np.frombuffer(frame[2], np.uint8), cv2.IMREAD_COLOR)
            preprocess_image(image, augment=False)  # 학습 때와 같은 크기의 영상이 수신되는지 확인한다.
            print("SENSORS OK", {k: ego[k] for k in ("ctrl_mode", "gear", "speed_kph", "x_m", "y_m", "yaw_deg")},
                  "image_shape=", image.shape, "counts=", dict(sensors.client.counts))
        finally:
            sensors.close()
        return 0
    manifest = json.loads((BASE / "routes/test/manifest.json").read_text())
    info = manifest["routes"][args.course]
    reference = Reference(info)
    if info.get("route_sha256") and sha256(reference.path) != info["route_sha256"]:
        raise ValueError("evaluation route hash mismatch")
    assist_route, assist_info = None, None
    if args.assist_east_outer_left:
        from collect import Route

        turn = next(item for item in info["navigation"] if item["command"] == "left")
        if turn["link_id"] != "A2256W000597":
            raise ValueError("unexpected first left junction on EAST_OUTER")
        assist_route = Route(reference.path)
        assist_info = {"link_id": turn["link_id"], "mode": "pure_pursuit_steering_only",
                       "start_s_m": turn["entry_s_m"] - 5., "end_s_m": turn["exit_s_m"] + 5.,
                       "lookahead_m": assist_route.lookahead_m, "longitudinal_control": "CIL",
                       "blend_in_m": 2., "blend_out_m": 1.5}
    if args.describe:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    if not args.control:
        parser.error("use --check, --describe, or --control to drive")
    from tools.recording import ScreenVideo, Video  # 실제 주행을 실행할 때만 녹화 도구를 불러온다.

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, checkpoint, checkpoint_path, digest = load_model(args.checkpoint, device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.course + "_" + uuid.uuid4().hex[:6]
    run = Path(args.output) if args.output else BASE / "runs/closed_loop" / run_id
    run = run if run.is_absolute() else BASE / run
    run.mkdir(parents=True, exist_ok=False)
    settings = {"course": args.course, "checkpoint": str(checkpoint_path.relative_to(BASE)),
                "checkpoint_sha256": digest, "epoch": checkpoint["epoch"], "sensor_config": config,
                "route": info, "route_sha256": sha256(reference.path), "device": str(device),
                "max_cte_m": args.max_cte, "seconds": args.seconds, "record_screen": args.record_screen,
                "control_hz": 20, "camera": "center", "model_controls": ["steer", "throttle", "brake"],
                "map_use": "reset, categorical navigation and evaluation only; no map steering or speed profile",
                "actuator_policy": "clip raw outputs; larger pedal wins; speed cap overrides; stop on endpoint/intervention",
                "stall_timeout_s": 12., "completion_definition": "reference station reaches final1m while absCTE<max_cte; same endpoint condition as PilotNet; then brake",
                "no_retraining": True, "collision_metric": "not instrumented",
                "screen_command_overlay": bool(args.record_screen),
                "live_command_banner": args.show_command,
                "screen_command_source": "same categorical command passed to the model; display only"}
    if assist_info:
        settings["steering_assist"] = assist_info
        settings["map_use"] = "reset, categorical navigation, evaluation, and steering only in the declared assist interval"
    write_json(run / "settings.json", settings)
    for source in ("drive.py", "model.py", "dataset.py", "morai.py", "config.json"):
        (run / "software").mkdir(exist_ok=True)
        shutil.copy2(BASE / source, run / "software" / source)
    shutil.copy2(BASE / "tools/recording.py", run / "software" / "recording.py")  # 명령 배너의 실제 구현도 보존한다.
    if args.show_command:
        shutil.copy2(BASE / "tools/live_command_banner.py", run / "software" / "live_command_banner.py")
    if assist_info:
        shutil.copy2(BASE / "collect.py", run / "software" / "collect.py")

    sensors = screen = video = simulator = live_banner = None
    status, error, rows = "not_started", None, []
    image, previous_stamp, last_command = None, None, None
    raw, applied = np.zeros(3), (0., 0., 1.)
    start = None
    assist_started, assist_handoff = None, None
    try:
        if args.reset:
            simulator = Simulator()
            write_json(run / "reset_state.json", simulator.setup(info, config))
        sensors = LatestSensors(config)
        ego, ego_recv, frame = wait_sensors(sensors)
        station, cte, heading = reference.project(ego)
        if abs(station - reference.s[0]) > 1. or abs(cte) > .15 or abs(heading) > 2. or ego["speed_kph"] > .5:
            raise RuntimeError("initial pose/speed mismatch; use --reset and verify simulator placement")
        if ego["ctrl_mode"] != 2 or ego["gear"] != 4:
            raise RuntimeError("MORAI must be in AV external control, D gear")
        write_json(run / "initial_state.json", dict(ego, route_s_m=station, cte_m=cte, heading_error_deg=heading))
        if args.record_screen:
            screen = ScreenVideo(run / "morai_screen.mp4", command_overlay=True)
            initial_command = reference.command(station)
            screen.update_command(initial_command, config["commands"][initial_command], 0., reason="ready", course=args.course)
            write_json(run / "screen_recording.json", {"geometry": screen.geometry,
                       "command_overlay": True, "source": "actual model navigation input",
                       "input_image_modified": False, "update_hz": 20,
                       "text_file": screen.overlay_text_path.name, "overlay_style": screen.overlay_style})
            if args.show_command:
                geometry = screen.geometry
                ready = run / "live_command_banner.ready"
                with (run / "live_command_banner.log").open("x", encoding="utf-8") as log:
                    live_banner = subprocess.Popen([
                        sys.executable, str(BASE / "tools/live_command_banner.py"),
                        "--text-file", str(screen.overlay_text_path), "--ready-file", str(ready),
                        "--x", str(geometry["x"]), "--y", str(geometry["y"]), "--width", str(geometry["width"]),
                    ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 3.0
                while not ready.exists():
                    if live_banner.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("live command banner did not start; see live_command_banner.log")
                    time.sleep(.02)
        video = Video(run / "center_camera.mp4", 1280, 720, 20)
        fields = ("elapsed_s", "ego_stamp_ns", "camera_stamp_ns", "x_m", "y_m", "yaw_deg", "speed_kph",
                  "route_s_m", "cte_m", "heading_error_deg", "command", "command_name",
                  "raw_steer", "raw_throttle", "raw_brake", "steer", "throttle", "brake",
                  "camera_age_s", "ego_age_s", "sensor_delta_s", "inference_ms", "new_frame", "reason")
        if assist_info:
            fields += ("steering_source", "steering_assist_weight")
        start, next_tick = time.monotonic(), time.monotonic()
        last_motion, motion_station = start, station
        last_print, stale_since, status = -1, None, "running"
        with (run / "telemetry.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while time.monotonic() - start < args.seconds:
                if sensors.error:
                    raise RuntimeError(sensors.error)
                ego, ego_recv, frame = sensors.snapshot()
                now = time.monotonic_ns()
                age_ego, age_camera = (now - ego_recv) / 1e9, (now - frame[1]) / 1e9
                delta = abs(ego["stamp_ns"] - frame[0]) / 1e9
                station, cte, heading = reference.project(ego)
                name = reference.command(station)
                command = config["commands"][name]
                inference_ms, new_frame = None, False
                reason = "model"
                steering_source = "CIL"
                assist_weight = 0.
                if ego["ctrl_mode"] != 2 or ego["gear"] != 4:
                    raise RuntimeError("external control mode or D gear lost")
                if age_ego > .20 or age_camera > .25 or delta > .25:
                    reason = "stale_sensor"
                if reason == "model" and (frame[0] != previous_stamp or command != last_command):
                    image = cv2.imdecode(np.frombuffer(frame[2], np.uint8), cv2.IMREAD_COLOR)
                    tensor = preprocess_image(image, augment=False).unsqueeze(0).to(device)
                    speed = torch.tensor([[ego["speed_kph"] / SPEED_SCALE_KPH]], dtype=torch.float32, device=device)
                    command_tensor = torch.tensor([command], dtype=torch.long, device=device)
                    inference_start = time.monotonic_ns()
                    with torch.inference_mode():
                        raw = model(tensor, speed, command_tensor)["control"][0].cpu().numpy()
                    inference_ms = (time.monotonic_ns() - inference_start) / 1e6
                    previous_stamp, last_command, new_frame = frame[0], command, True
                if (time.monotonic_ns() - frame[1]) / 1e9 > .25:
                    reason = "stale_inference"
                if reason in ("stale_sensor", "stale_inference"):
                    applied = (float(applied[0]), 0., 1.)
                    stale_since = stale_since or time.monotonic()
                    if time.monotonic() - stale_since > 2.:
                        status = "intervention_sensor"
                else:
                    stale_since = None
                    steer, throttle, braking, reason = actuator_control(raw, ego["speed_kph"], config["collection"]["speed_cap_kph"])
                    applied = (steer, throttle, braking)
                if abs(cte) >= args.max_cte:
                    status = "intervention_cte"
                elif ego["speed_kph"] < -.5:
                    status = "intervention_reverse"
                elif station >= reference.s[-1] - 1.:
                    status = "completed"
                if station - motion_station >= .5:
                    last_motion, motion_station = time.monotonic(), station
                elif time.monotonic() - last_motion > 12. and ego["speed_kph"] < .5 and status == "running":
                    status = "intervention_stalled"
                if status != "running":
                    applied, reason = (float(applied[0]), 0., 1.), status
                    steering_source = "safety_stop"
                elif reason in ("stale_sensor", "stale_inference"):
                    steering_source = "safety_hold"
                elif assist_info:
                    progress = reference.progress
                    if assist_info["start_s_m"] <= progress < assist_info["end_s_m"]:
                        # Use the same local projection window and steering as the collection teacher.
                        assist_route.progress = progress
                        _, _, _, guide = assist_route.track(
                            ego, config["vehicle"]["wheelbase_m"], config["vehicle"]["max_wheel_angle_deg"])
                        assist_weight = min(1., (progress - assist_info["start_s_m"]) / assist_info["blend_in_m"],
                                            (assist_info["end_s_m"] - progress) / assist_info["blend_out_m"])
                        applied = (steer + assist_weight * (guide - steer), throttle, braking)
                        steering_source = "map_assist"
                        if assist_started is None:
                            assist_started = dict(route_s_m=station, elapsed_s=time.monotonic() - start)
                    elif progress >= assist_info["end_s_m"] and assist_started and not assist_handoff:
                        assist_handoff = dict(route_s_m=station, elapsed_s=time.monotonic() - start,
                                              cte_m=cte, heading_error_deg=heading)
                sensors.client.send_control(*applied)
                elapsed = time.monotonic() - start
                overlay_reason = reason
                if assist_info and steering_source == "map_assist":
                    overlay_reason = "STEERING INTERVENTION ACTIVE"
                elif assist_handoff and steering_source == "CIL":
                    overlay_reason = "CIL RESUMED" if reason == "model" else "CIL RESUMED | SPEED CAP"
                if screen is not None:
                    screen.update_command(name, command, elapsed, reason=overlay_reason, course=args.course)
                row = dict(zip(fields, (elapsed, ego["stamp_ns"], frame[0], ego["x_m"], ego["y_m"], ego["yaw_deg"], ego["speed_kph"],
                    station, cte, heading, command, name, *map(float, raw), *applied, age_camera, age_ego, delta,
                    inference_ms, int(new_frame), reason)))
                if assist_info:
                    row["steering_source"] = steering_source
                    row["steering_assist_weight"] = assist_weight
                writer.writerow(row)
                rows.append(row)
                if image is not None:
                    display = image.copy()
                    cv2.rectangle(display, (0, 0), (1280, 88), (15, 15, 15), -1)
                    cv2.putText(display, f"CIL | {args.course} | {name} | {overlay_reason}", (15, 31), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                    cv2.putText(display, f"v={ego['speed_kph']:.1f} km/h  CTE={cte:+.2f} m  steer={applied[0]:+.3f}",
                                (15, 66), cv2.FONT_HERSHEY_SIMPLEX, .75, (80, 230, 230), 2)
                    video.write(display)
                if int(elapsed) != last_print:
                    last_print = int(elapsed)
                    stream.flush()
                    print(f"{args.course} t={elapsed:.0f}s s={station:.1f}/{reference.s[-1]:.1f} cte={cte:+.3f} v={ego['speed_kph']:.1f} {name} {overlay_reason}", flush=True)
                if status != "running":
                    break
                next_tick += .05
                if next_tick < time.monotonic() - .1:
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick - time.monotonic()))
            else:
                status = "timeout"
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException as exception:
        status, error = "failed", str(exception)
    finally:
        cleanup_errors = []
        if sensors:
            try:
                hold_brake(sensors.client, steer=float(applied[0]))  # 실패한 시험도 즉시 정지한다.
            except Exception as exception:
                cleanup_errors.append("brake: " + str(exception))
            try:
                sensors.close()
            except Exception as exception:
                cleanup_errors.append("sensors: " + str(exception))
        for name, recorder in (("camera_video", video), ("screen_video", screen)):
            if recorder:
                try:
                    recorder.close()
                except Exception as exception:
                    cleanup_errors.append(name + ": " + str(exception))
        if live_banner:
            try:
                if live_banner.poll() is None:
                    live_banner.terminate()
                    try:
                        live_banner.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        live_banner.kill()
                        live_banner.wait(timeout=2.)
                if live_banner.returncode not in (0, -15):
                    cleanup_errors.append("live command banner exited with code " + str(live_banner.returncode))
            except Exception as exception:
                cleanup_errors.append("live command banner: " + str(exception))
        if simulator:
            try:
                simulator.channel.close()
            except Exception as exception:
                cleanup_errors.append("simulator: " + str(exception))
        ctes = np.array([row["cte_m"] for row in rows])
        inference = [row["inference_ms"] for row in rows if row["inference_ms"] is not None]
        last = rows[-1] if rows else {}
        completion = np.clip((last.get("route_s_m", reference.s[0]) - reference.s[0]) / (reference.s[-1] - reference.s[0]), 0., 1.)
        summary = {"status": status, "error": error, "course": args.course, "checkpoint_sha256": digest,
                   "epoch": checkpoint["epoch"], "telemetry_rows": len(rows), "elapsed_s": last.get("elapsed_s", 0.),
                   "completion_fraction": float(completion), "cte_rmse_m": float(np.sqrt(np.mean(ctes ** 2))) if len(ctes) else None,
                   "max_abs_cte_m": float(np.max(np.abs(ctes))) if len(ctes) else None,
                   "terminal_station_m": last.get("route_s_m"), "terminal_heading_deg": last.get("heading_error_deg"),
                   "mean_speed_kph": float(np.mean([row["speed_kph"] for row in rows])) if rows else None,
                   "inference_mean_ms": float(np.mean(inference)) if inference else None,
                   "interventions": int(status.startswith("intervention")), "cleanup_errors": cleanup_errors,
                   "camera_video": "center_camera.mp4" if video else None,
                   "screen_video": "morai_screen.mp4" if screen else None,
                   "collision_metric": None, "collision_note": "not instrumented; no zero-collision claim",
                   "evaluation_scope": info.get("evaluation_scope", "held-out roads on the same K-City map"),
                   "test_only": info.get("split") == "test", "used_for_training": False}
        if assist_info:
            summary["evaluation_scope"] = ("assisted_diagnostic_extended_route_with_training_proximity"
                                           if "extension_proximity" in info else "assisted_diagnostic_on_held_out_road")
            summary["steering_assist"] = dict(assist_info, start=assist_started, handoff=assist_handoff,
                                              frames=sum(row["steering_source"] == "map_assist" for row in rows))
        write_json(run / "summary.json", summary)
        print("RESULT", run, json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if status == "completed" and not cleanup_errors else 1 if status == "failed" or cleanup_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
