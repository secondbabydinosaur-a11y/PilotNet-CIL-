"""Collect disjoint CIL demonstrations until the balanced 90-minute training ledger is full."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from collect import Route
from tools.collect_east_outer_data import capture, read_json, write_json

COMMANDS = ("follow_road", "left", "straight", "right")
TARGET = 13_500
FROZEN_ROUTE = BASE / "routes/test/EAST_OUTER_EXTENDED.csv"
LEDGER_TOOL = BASE / "tools/prepare_east_outer_90min.py"


def save_progress(path, data):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def route_options(plan, frozen):
    reference = np.asarray(plan["test_reservations"]["EAST_OUTER_EXTENDED"]["guard_reference_xyz"])
    guard = cKDTree(reference[:, :2])
    if float(guard.query(frozen[:, 1:3])[0].max()) > .26:
        raise ValueError("collection plan does not guard the whole frozen evaluation route")
    original = set(read_json(BASE / "routes/plan.json")["routes"])
    options = {}
    for name, info in plan["routes"].items():
        if name in original or info["split"] != "train":
            continue
        route = Route(BASE / info["route_csv"], info.get("navigation", []))
        route.make_speed_profile(19.)
        if float(guard.query(route.xy)[0].min()) < 20.5:
            print(f"SKIP {name}: planned course is too close to the full evaluation path", flush=True)
            continue
        stations = np.arange(0., max(0., route.s[-1] - 5.), .5)
        weights = .5 / np.maximum(1.5, np.interp(stations, route.s, route.speed_profile) / 3.6)
        counts = [0.] * 4
        for station, seconds in zip(stations, weights):
            counts[COMMANDS.index(route.command(station))] += float(seconds)
        options[name] = {"seconds": counts, "timeout": max(75, int(sum(counts) * 1.75 + 35))}
    if not options or any(not any(item["seconds"][i] >= 2. for item in options.values()) for i in range(4)):
        raise ValueError("the disjoint collection plan must cover all four commands")
    return options


def choose_route(options, counts, accepted, failures):
    visits = Counter(entry["route_id"] for entry in accepted)
    attempts = Counter(entry["route_id"] for entry in failures)
    deficits = [max(0., TARGET - count) / 10. for count in counts]
    if not any(deficits):
        raise ValueError("there are no command deficits")
    max_deficit = max(deficits)

    def score(name):
        seconds = options[name]["seconds"]
        useful = sum(min(need, have) * math.sqrt(need / max_deficit) for need, have in zip(deficits, seconds))
        # Visit each useful geometry once, then prefer routes that fill remaining command quotas efficiently.
        return useful / (9. + sum(seconds)) - .08 * math.log1p(visits[name]) - .08 * attempts[name]

    candidates = [name for name, settings in options.items()
                  if attempts[name] < 3 + visits[name]
                  and any(need > 0 and have >= 2. for need, have in zip(deficits, settings["seconds"]))]
    if not candidates:
        raise RuntimeError("no quality-checked route options remain for the commands still needed")
    unused = [name for name in candidates if not visits[name]]
    name = max(unused or candidates, key=score)
    return name, visits[name], attempts[name]


def rough_counts(accepted):
    # A temporary estimate to start captures while the preparation ledger is being created.
    counts, noisy = [0] * 4, [0] * 4
    for run in accepted:
        for i in range(4):
            frames = round(float(run["command_seconds"].get(str(i), 0.)) * 10.)
            counts[i] += frames
            if run["noise_enabled"]:
                noisy[i] += frames
    return counts, noisy


def ledger(plan_path, progress_path, output):
    args = [sys.executable, str(LEDGER_TOOL), "--plan", str(plan_path), "--completion", str(progress_path),
            "--output", str(output)]
    subprocess.run(args, cwd=BASE, check=True)
    return read_json(output / "ledger.json"), args


def collect(plan_path, output):
    progress_path = output / "capture_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"status": "collecting", "accepted": [], "failures": []}
    if progress["status"] == "complete":
        print("90-minute collection already finalized", flush=True)
        return
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    configs = output / "capture_settings"
    configs.mkdir(exist_ok=True)
    config = read_json(BASE / "config.json")
    frozen = np.loadtxt(FROZEN_ROUTE, delimiter=",", skiprows=1)
    prior_plan_text, options, config_path = None, {}, None
    accepted, failures = progress["accepted"], progress["failures"]
    while True:
        if (output / "stop_after_session").exists():
            print("Collection paused after a complete session", flush=True)
            return
        plan_text = plan_path.read_text(encoding="utf-8")
        if plan_text != prior_plan_text:
            digest = hashlib.sha256(plan_text.encode()).hexdigest()[:12]
            plan = json.loads(plan_text)
            options = route_options(plan, frozen)
            snapshot = configs / f"plan_{digest}.json"
            snapshot.write_text(plan_text, encoding="utf-8")
            config_path = configs / f"config_{digest}.json"
            config["paths"]["route_plan"] = str(snapshot.relative_to(BASE))
            config["paths"]["raw_data"] = str(raw.relative_to(BASE))
            write_json(config_path, config)
            prior_plan_text = plan_text
        save_progress(progress_path, progress)
        if LEDGER_TOOL.exists():
            audit, prepare_args = ledger(plan_path, progress_path, output)
            if audit["target_met"]:
                subprocess.run(prepare_args + ["--finalize"], cwd=BASE, check=True)
                progress["status"] = "complete"
                save_progress(progress_path, progress)
                print("FINALIZED 90-minute training data", flush=True)
                return
            counts = [audit["center_counts"][str(i)] for i in range(4)]
            noisy = [audit["noise_session_center_counts"][str(i)] for i in range(4)]
        else:
            counts, noisy = rough_counts(accepted)

        name, visits, failed = choose_route(options, counts, accepted, failures)
        route = options[name]
        dominant = max(range(4), key=lambda i: route["seconds"][i] * max(0., TARGET - counts[i]))
        ratio = noisy[dominant] / max(1, counts[dominant])
        noise = sum(counts) >= 800 and ratio < .10 and route["seconds"][dominant] > 3 and failed < 2
        junction_recovery = noise and dominant != 0
        speed = (19., 17., 18., 16.5)[(visits + failed) % 4]
        hour = (12, 10, 14, 11, 13, 9, 15)[visits % 7]
        if noise:
            speed = min(speed, 17.5)
        number = progress.get("next_attempt", 1)
        progress["next_attempt"] = number + 1
        label = f"{number:04d}_{name}_{'recovery' if noise else 'normal'}"
        progress["active"] = {"route_id": name, "label": label, "speed_kph": speed, "daylight_hour": hour}
        save_progress(progress_path, progress)
        try:
            run = capture(config_path, raw, output, name, junction_recovery, 700 + number, label=label,
                          seconds=route["timeout"], speed=speed, noise=noise, junction_recovery=junction_recovery,
                          hour=hour)
            accepted.append(run)
        except ValueError as error:
            failures.append({"route_id": name, "label": label, "error": str(error)})
            print("EXCLUDED", error, flush=True)
        finally:
            progress.pop("active", None)
            save_progress(progress_path, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "collector.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect((BASE / args.plan).resolve(), output)


if __name__ == "__main__":
    main()
