"""PilotNet MORAI closed-loop 실험. 지도는 평가에만, 조향은 가운데 영상 CNN만 사용.

--control을 지정하면 일반 MORAI UDP 제어를 송신한다. --reset은 실험 시작점으로
배치한다. 실행별 영상/제어 CSV/결과 JSON을 새 폴더에 보존한다.
"""
import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import cv2
import numpy as np
import torch

from data_utils import BASE, preprocess, read_json, write_json
from learning.common import choose_device, file_hash, load_checkpoint
from learning.live import LatestSensors, Reference, curvature_to_degrees, speed_command
from tools.collection.manual_control import encode_command
from tools.collection.sim_api import Simulator


class Video:
    def __init__(self, path, width, height, hz):
        self.frames = 0
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "{}x{}".format(width, height),
            "-r", str(hz), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, image):
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        if code:
            raise RuntimeError("ffmpeg failed: " + str(code))


class ScreenVideo:
    """실제 MORAI 창이 표시된 X11 화면 영역을 녹화한다."""
    def __init__(self, path):
        import gi
        gi.require_version("Gdk", "3.0")
        gi.require_version("GdkX11", "3.0")
        from gi.repository import Gdk, GdkX11
        windows = subprocess.check_output(["xdotool", "search", "--name", "^Simulator$"], text=True).split()
        window_id = int(windows[-1])
        subprocess.run(["xdotool", "windowactivate", "--sync", str(window_id)], check=True)
        time.sleep(.2)
        window = GdkX11.X11Window.foreign_new_for_display(Gdk.Display.get_default(), window_id)
        _, x, y = window.get_origin()
        width, height = window.get_width() // 2 * 2, window.get_height() // 2 * 2
        self.log = path.with_suffix(".ffmpeg.log").open("w")
        self.process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "x11grab", "-draw_mouse", "0", "-framerate", "20", "-video_size", "{}x{}".format(width, height),
            "-i", "{}+{},{}".format(os.environ.get("DISPLAY", ":1"), x, y),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=self.log)
        time.sleep(.3)
        if self.process.poll() is not None:
            self.log.close()
            raise RuntimeError("MORAI screen recorder failed; inspect ffmpeg log")

    def close(self):
        self.process.communicate(input=b"q\n", timeout=20)
        self.log.close()
        if self.process.returncode:
            raise RuntimeError("screen recorder returned " + str(self.process.returncode))


def brake(sock, destination, seconds=.8, angle=0.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sock.sendto(encode_command(0., 1., angle / 40.), destination)
        time.sleep(.02)


def reset_simulator(ref, lateral, destination, sock, heading_offset_deg=0.):
    sim = Simulator()
    try:
        sim.conditions()
        sim.automatic()
        sim.resume()
        brake(sock, destination, .4)
        desired_pose = ref.pose(lateral, heading_offset_deg)
        commanded_yaw = desired_pose[3]
        sim.teleport(*desired_pose)
        brake(sock, destination, 1.)
        actual = sim.state()["transform"]["location"]
        desired = ref.pose(lateral)
        if np.hypot(actual["x"] - desired[0], actual["y"] - desired[1]) > .5:
            raise RuntimeError("simulator reset pose did not match")
        # SetTransform 요청 yaw와 읽힌 yaw 사이의 좌표 변환 차이도 readback으로 보정한다.
        # 직전 '요청' yaw를 수정해야 한다. 반환 yaw에 보정하면 같은 오프셋이 반복된다.
        # 이 초기 배치 과정은 주행 전 정차 중에만 실행한다.
        for _ in range(3):
            transform = sim.state()["transform"]
            pos, rotation = transform["location"], transform["rotation"]
            _, _, measured = ref.project(dict(x_m=pos["x"], y_m=pos["y"], yaw_deg=rotation["z"]))
            delta = (heading_offset_deg - measured + 180) % 360 - 180
            if abs(delta) <= .3:
                break
            commanded_yaw += delta
            sim.teleport(pos["x"], pos["y"], pos["z"], commanded_yaw)
            brake(sock, destination, .8)
    finally:
        sim.channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--course", choices=tuple(Reference.limits), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--record-screen", action="store_true", help="also record the actual MORAI GUI window")
    parser.add_argument("--initial-offset", type=float, default=0.)
    parser.add_argument("--initial-heading-deg", type=float, default=0.,
                        help="yaw relative to route tangent; positive turns left")
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--speed", type=float, default=19.)
    parser.add_argument("--max-cte", type=float, default=7.)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.speed, args.seconds, args.max_cte,
                                         args.initial_offset, args.initial_heading_deg))
            or not 0 < args.speed <= 19.5 or args.seconds <= 0 or args.max_cte <= 0
            or abs(args.initial_offset) > .6 or abs(args.initial_heading_deg) > 10.):
        parser.error("invalid speed, duration, CTE or initial offset")
    if args.reset and not args.control:
        parser.error("--reset requires --control")
    torch.set_num_threads(4)
    cfg = read_json(BASE / "config.json")
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device, check_data=True)
    contract = checkpoint["data_contract"]
    if cfg["camera_common"] != contract["camera_common"] or cfg["cameras"]["center"] != contract["center_camera"]:
        raise ValueError("camera configuration differs from training")
    ref = Reference(args.course)
    run = args.output or BASE / "runs/closed_loop" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.course + "_" + uuid.uuid4().hex[:6])
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "settings.json", dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_hash(args.checkpoint), course=args.course, control=args.control,
        initial_offset_m=args.initial_offset, target_kph=args.speed, max_cte_m=args.max_cte,
        initial_heading_offset_deg=args.initial_heading_deg,
        cte_intervention_definition="abs(cte_m) >= max_cte_m",
        completion_definition="route_station_m >= reference_end_m - 1.0; CTE limit checked first",
        standalone_heading_intervention=False,
        record_screen=args.record_screen,
        seconds=args.seconds, reference_station_m=[ref.start, ref.end],
        steering="CNN only; reference used only for reset and evaluation",
        model_camera_mode=checkpoint.get("training_data", {}).get("camera_mode", "center")))
    destination = (cfg["manual_control"]["sim_ip"], cfg["manual_control"]["sim_ctrl_host_port"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensors, video, screen, image, previous_stamp = None, None, None, None, None
    rows, status, error_message = [], "not_started", None
    sent = False
    try:
        if args.reset:
            reset_simulator(ref, args.initial_offset, destination, sock, args.initial_heading_deg)
        if args.record_screen:
            screen = ScreenVideo(run / "morai_screen.mp4")
        sensors = LatestSensors(cfg)
        # GPU 첫 호출과 커널 로딩은 출발 전에 끝낸다.
        with torch.inference_mode():
            model(torch.zeros((1, 3, 66, 200), device=device))
        waiting = time.monotonic()
        while True:
            state, recv, frame = sensors.snapshot()
            if state is not None and frame is not None:
                break
            if time.monotonic() - waiting > 10:
                raise RuntimeError("camera/Ego packets missing")
            time.sleep(.02)
        station, cte, heading = ref.project(state)
        if abs(station - ref.start) > 8 or abs(cte) > args.max_cte or abs(heading) > 15:
            raise RuntimeError("place vehicle at course start, or use --reset")
        write_json(run / "initial_state.json", dict(route_s_m=station, cte_m=cte,
            heading_error_deg=heading, requested_offset_m=args.initial_offset,
            requested_heading_offset_deg=args.initial_heading_deg, speed_kph=state["speed_kph"]))
        # 실제 UDP readback으로 시험 조건을 확인한다. 요청 값만 저장하고 넘어가지 않는다.
        if args.reset and (abs(cte - args.initial_offset) > .1
                           or abs((heading - args.initial_heading_deg + 180) % 360 - 180) > 1.):
            raise RuntimeError("measured initial lateral/heading error differs from requested reset")
        start, next_tick, last_print = time.monotonic(), time.monotonic(), -1
        kappa, angle, fail_since, complete = 0., 0., None, False
        mode = checkpoint.get("training_data", {}).get("camera_mode", "center")
        status = "running"
        fields = ("elapsed_s", "ego_stamp_ns", "camera_stamp_ns", "speed_kph", "x_m", "y_m", "yaw_deg",
                  "route_s_m", "cte_m", "heading_error_deg", "prediction_1pm", "steer_command_deg",
                  "accel", "brake", "reason", "camera_age_s", "ego_age_s", "sent",
                  "inference_ms", "new_frame")
        with (run / "telemetry.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while time.monotonic() - start < args.seconds:
                state, recv, frame = sensors.snapshot()
                now = time.monotonic_ns()
                age_e, age_c = (now - recv) / 1e9, (now - frame[1]) / 1e9
                reason = "drive"
                if sensors.error:
                    raise RuntimeError(sensors.error)
                if age_e > .2 or age_c > .25 or abs(state["stamp_ns"] - frame[0]) > 250_000_000:
                    reason = "stale_sensor"
                if args.control and state["ctrl_mode"] != 2:
                    raise RuntimeError("vehicle must be in AV-ExternalCtrl mode")
                if state["speed_kph"] < -.5:
                    raise RuntimeError("unexpected reverse motion")
                station, cte, heading = ref.project(state)
                if abs(cte) >= args.max_cte:
                    status = "intervention_cte"
                    break
                if station >= ref.end - 1.:
                    complete, status = True, "completed"
                    break
                inference_ms, new_frame = None, False
                if reason == "drive" and frame[0] != previous_stamp:
                    image = cv2.imdecode(np.frombuffer(frame[2], np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (cfg["camera_common"]["height"], cfg["camera_common"]["width"]):
                        raise RuntimeError("invalid camera frame/size")
                    x = preprocess(image, contract["preprocessing"])
                    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
                    inference_start = time.monotonic_ns()
                    with torch.inference_mode():
                        kappa = float(model(x).item())
                    inference_ms = (time.monotonic_ns() - inference_start) / 1e6
                    new_frame = True
                    requested = float(np.clip(curvature_to_degrees(kappa, contract), -20., 20.))
                    angle += float(np.clip(requested - angle, -3., 3.))
                    previous_stamp = frame[0]
                # 추론 후에도 age를 확인해 지연된 영상으로 가속하지 않게 한다.
                if (time.monotonic_ns() - frame[1]) / 1e9 > .25:
                    reason = "stale_inference"
                remaining = ref.end - station
                target = min(args.speed, 3.6 * math.sqrt(max(0., 2. * 1.2 * (remaining - 1.))))
                throttle, braking = speed_command(state["speed_kph"], target)
                if reason != "drive":
                    throttle, braking = 0., 1.
                    fail_since = fail_since or time.monotonic()
                    if time.monotonic() - fail_since > 2:
                        status = "intervention_sensor"
                        break
                else:
                    fail_since = None
                if args.control:
                    sock.sendto(encode_command(throttle, braking, angle / 40.), destination)
                    sent = True
                elapsed = time.monotonic() - start
                row = dict(zip(fields, (elapsed, state["stamp_ns"], frame[0], state["speed_kph"],
                    state["x_m"], state["y_m"], state["yaw_deg"], station, cte, heading, kappa, angle,
                    throttle, braking, reason, age_c, age_e, int(args.control), inference_ms, int(new_frame))))
                writer.writerow(row)
                rows.append(row)
                if image is not None:
                    if video is None:
                        video = Video(run / "drive.mp4", image.shape[1], image.shape[0], 20)
                    display = image.copy()
                    cv2.rectangle(display, (0, 0), (display.shape[1], 87), (15, 15, 15), -1)
                    cv2.putText(display, "PilotNet {} | Course {} | {}".format(mode, args.course, reason),
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                    cv2.putText(display, "v={:.1f} km/h  CTE={:+.2f} m  steer={:+.2f} deg".format(
                        state["speed_kph"], cte, angle), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 230, 230), 2)
                    video.write(display)
                if int(elapsed) != last_print:
                    last_print = int(elapsed)
                    print("{} {} t={:.0f}s s={:.1f} cte={:+.3f} v={:.1f} {}".format(
                        mode, args.course, elapsed, station, cte, state["speed_kph"], reason), flush=True)
                next_tick += .05
                if next_tick < time.monotonic() - .1:
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick - time.monotonic()))
            else:
                status = "timeout"
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException as exc:
        status, error_message = "failed", str(exc)
    finally:
        if args.control and sent:
            brake(sock, destination, 1.5)
        sock.close()
        if sensors:
            sensors.close()
        if video:
            video.close()
        if screen:
            screen.close()
        ctes = np.array([r["cte_m"] for r in rows])
        terminal_cte = cte if 'cte' in locals() else None
        observed = np.r_[ctes, terminal_cte] if terminal_cte is not None else ctes
        inference = [r["inference_ms"] for r in rows if r.get("inference_ms") is not None]
        summary = dict(status=status, error=error_message, control=args.control, course=args.course,
            frames=video.frames if video else 0, telemetry_rows=len(rows),
            elapsed_s=rows[-1]["elapsed_s"] if rows else 0.,
            completion_fraction=float(np.clip((rows[-1]["route_s_m"] - ref.start) / (ref.end - ref.start), 0, 1)) if rows else 0.,
            cte_rmse_m=float(np.sqrt(np.mean(ctes ** 2))) if len(ctes) else None,
            max_abs_cte_m=float(np.max(np.abs(observed))) if len(observed) else None,
            p95_abs_cte_m=float(np.percentile(np.abs(ctes), 95)) if len(ctes) else None,
            mean_speed_kph=float(np.mean([r["speed_kph"] for r in rows])) if rows else None,
            max_speed_kph=float(max(r["speed_kph"] for r in rows)) if rows else None,
            inference_mean_ms=float(np.mean(inference)) if inference else None,
            inference_p95_ms=float(np.percentile(inference, 95)) if inference else None,
            interventions=int(status.startswith("intervention")),
            collision_metric=None, collision_note="not instrumented; no zero-collision claim",
            terminal_station_m=station if 'station' in locals() else None,
            terminal_cte_m=cte if 'cte' in locals() else None,
            checkpoint_sha256=file_hash(args.checkpoint))
        write_json(run / "summary.json", summary)
        print("RESULT {} {}".format(run, summary), flush=True)
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
