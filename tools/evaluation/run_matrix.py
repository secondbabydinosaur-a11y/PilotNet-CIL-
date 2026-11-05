"""두 고정 모델을 동일한 코스·초기 위치·헤딩 조건에서 순차 시험한다."""
import argparse
from pathlib import Path
import socket
import subprocess

from data_utils import BASE, read_json, write_json
from drive import brake
from learning.common import file_hash
from learning.live import Reference
from tools.collection.sim_api import Simulator

CONDITIONS = {"nominal": (0., 0.), "left": (.3, 0.), "right": (-.3, 0.),
              "heading_left": (0., 5.), "heading_right": (0., -5.)}


def park():
    cfg = read_json(BASE / "config.json")["manual_control"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        brake(sock, (cfg["sim_ip"], cfg["sim_ctrl_host_port"]), 1.)
    sim = Simulator()
    try:
        sim.actor_call("SetVehicleGear", "VehicleGear", gear=1)
        sim.actor_call("SetVehicleDynamicsSpeedLimit", "VehicleDynamicsSpeedLimit",
                       speed_limiter=False, speed_limit=20., reset=False)
        sim.actor_call("SetVehicleControlMode", "VehicleControlModeParam", mode=1)
    finally:
        sim.channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=3)
    parser.add_argument("--record-screen", action="store_true")
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS),
                        help="explicit conditions; otherwise --repeats selects nominal or 3 lateral conditions")
    parser.add_argument("--courses", nargs="+", choices=tuple(Reference.limits), default=["A", "D", "J"])
    parser.add_argument("--tag", default="", choices=("", "unseen"))
    args = parser.parse_args()
    root = BASE / "runs/closed_loop"
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    models = {"center": BASE / "runs/pilotnet_center_v1/best.pt",
              "three": BASE / "runs/pilotnet_three_v1/best.pt"}
    trials = args.conditions or list(CONDITIONS)[:args.repeats]
    results = []
    suffix = ("_" + args.tag if args.tag else "") + ("_screen" if args.record_screen else "")
    if all(name.startswith("heading_") for name in trials):
        suffix = "_heading" + suffix
    elif args.conditions:
        suffix = "_conditions_" + "_".join(trials) + suffix
    write_json(root / ("plan" + suffix + ".json"), dict(models={k: str(v) for k, v in models.items()},
        courses={name: list(Reference.limits[name]) for name in args.courses},
        conditions={name: dict(offset_m=CONDITIONS[name][0], heading_deg=CONDITIONS[name][1])
                    for name in trials}, target_speed_kph=19., intervention_cte_m=1.,
        intersection_note="unseen geometry and missing intent are both factors; failure is not architectural impossibility",
        control="CNN steering, fixed-speed feedback; map only resets and measures"))
    try:
        for trial in trials:
            offset, heading = CONDITIONS[trial]
            for mode, checkpoint in models.items():
                for course in args.courses:
                    name = "{}_{}_{}{}".format(mode, course, trial, suffix)
                    run = root / name
                    summary = run / "summary.json"
                    # 사용자 중단/통신 실패 기록은 보존하고 새 디렉터리에서 다시 시험한다.
                    attempt = 0
                    while summary.exists() and (read_json(summary)["status"] in ("interrupted", "failed", "not_started")
                                                or (run / "infrastructure_incident.json").exists()):
                        attempt += 1
                        run = root / (name + "_retry" + str(attempt))
                        summary = run / "summary.json"
                    if summary.exists():
                        result = read_json(summary)
                        if result["checkpoint_sha256"] != file_hash(checkpoint):
                            raise RuntimeError("existing trial has another checkpoint: " + name)
                        settings = read_json(run / "settings.json")
                        if (settings["reference_station_m"] != list(Reference.limits[course])
                                or settings.get("initial_heading_offset_deg", 0.) != heading
                                or settings["initial_offset_m"] != offset):
                            raise RuntimeError("existing trial conditions differ: " + name)
                        print("EXISTING {} {}".format(name, result["status"]), flush=True)
                    else:
                        print("START {} offset={:+.2f} heading={:+.1f}".format(name, offset, heading), flush=True)
                        command = ["bash", str(BASE / "run.sh"), "drive.py", "--checkpoint", str(checkpoint),
                            "--course", course, "--control", "--reset", "--initial-offset", str(offset),
                            "--initial-heading-deg", str(heading),
                            "--output", str(run)]
                        if args.record_screen:
                            command.append("--record-screen")
                        with (root / "logs" / (run.name + ".log")).open("x") as log:
                            completed = subprocess.run(command, cwd=str(BASE), stdout=log, stderr=subprocess.STDOUT)
                        if not summary.exists():
                            raise RuntimeError("trial crashed before summary: " + name)
                        result = read_json(summary)
                        if completed.returncode or result["status"] == "failed":
                            raise RuntimeError("trial infrastructure failed {}: {}".format(name, result["error"]))
                    results.append({**result, "name": name, "mode": mode, "course": course,
                                    "trial": trial, "initial_offset_m": offset,
                                    "initial_heading_offset_deg": heading, "path": str(run)})
                    write_json(root / ("matrix" + suffix + ".json"), results)
                    print("DONE {} {} cte_rmse={} progress={:.1%}".format(
                        name, result["status"], result["cte_rmse_m"], result["completion_fraction"]), flush=True)
    finally:
        park()
        print("Simulator parked: P / Keyboard; collection speed limiter disabled.", flush=True)


if __name__ == "__main__":
    main()
