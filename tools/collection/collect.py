#!/usr/bin/env python3
"""Record three cameras/Ego, optionally with human keyboard steering + speed limit."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import select
import socket
import sys
import threading
import time
import uuid

import cv2
import numpy as np

from data_utils import BASE, EGO_FIELDS, FRAME_FIELDS, load_config, write_json
from .protocol import CameraAssembler, parse_ego
from .manual_control import COMMAND_FIELDS, CONTROL_KIND, ControlWorker, X11Keys


def bind_sockets(cfg):
    sockets = {}
    try:
        destinations = {name: item["port"] for name, item in cfg["cameras"].items()}
        destinations["ego"] = cfg["ego"]["port"]
        for name, port in destinations.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets[name] = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                            cfg["recording"]["socket_buffer_bytes"])
            # No SO_REUSEADDR: a second collector should fail instead of stealing data.
            try:
                sock.bind((cfg["bind_ip"], port))
            except OSError as exc:
                raise OSError(exc.errno, "{} receive port {}:{}: {}. Check MORAI Host/Destination ports and other receivers".format(
                    name, cfg["bind_ip"], port, exc.strerror)) from exc
            sock.settimeout(0.2)
        return sockets
    except Exception:
        for sock in sockets.values():
            sock.close()
        raise


def probe_ego(cfg, seconds=10.):
    """Read-only diagnostic: prove receipt before trying any assisted driving."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg["bind_ip"], cfg["ego"]["port"]))
    sock.settimeout(.2)
    raw, parsed, errors = 0, 0, 0
    start = previous = time.monotonic()
    previous_raw = 0
    latest = None
    print("EGO PROBE: UDP {}:{} for {}s; no commands sent".format(cfg["bind_ip"], cfg["ego"]["port"], seconds), flush=True)
    try:
        while time.monotonic() - start < seconds:
            try:
                packet, source = sock.recvfrom(65535)
                raw += 1
                try:
                    latest = parse_ego(packet)
                    parsed += 1
                except ValueError as exc:
                    errors += 1
                    if errors <= 3:
                        print("raw from {} bytes={} parse error: {}".format(source, len(packet), exc), flush=True)
            except socket.timeout:
                pass
            now = time.monotonic()
            if now - previous >= 1:
                detail = "no parsed Ego" if latest is None else "speed={:.1f} steer={:.2f} xyz=({:.2f},{:.2f},{:.2f}) mode={} gear={}".format(
                    latest["speed_kph"], latest["steer_raw_deg"], latest["x_m"], latest["y_m"], latest["z_m"], latest["ctrl_mode"], latest["gear"])
                print("raw={} parsed={} errors={} {:.1f}Hz | {}".format(raw, parsed, errors, (raw - previous_raw) / (now - previous), detail), flush=True)
                previous, previous_raw = now, raw
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    if not raw:
        print("NO UDP PACKETS: MORAI Ego Network > Publisher > ordinary Ego Vehicle Status: UDP, Host9300, Destination9301, Destination IP=this PC, Connect/Send ON; simulator playing.")
    elif not parsed:
        print("UDP arrives but format is unsupported; select ordinary Ego Vehicle Status and share the error above.")
    return 0 if parsed else 1


class Recorder:
    def __init__(self, cfg, config_path, output, course, group, note="", control_factory=None):
        self.cfg, self.output = cfg, Path(output)
        self.stop = threading.Event()
        self.q = queue.Queue(maxsize=cfg["recording"]["queue_size"])
        self.lock = threading.Lock()
        self.counts = {name: 0 for name in ("left", "center", "right", "ego", "ego_raw")}
        self.saved = dict(self.counts)
        self.errors = {name: 0 for name in ("left", "center", "right", "ego")}
        self.first_errors, self.fatal_errors = [], []
        self.latest = {}
        self.last_ego_recv = None
        self.last_ego = None
        self.control = None
        self.control_factory = control_factory
        self.assemblers = {name: CameraAssembler(cfg["recording"]["max_pending_frames"])
                           for name in cfg["cameras"]}
        self.manifest = {
            "schema_version": 1, "session_id": self.output.name,
            "course": course, "group": group, "note": note, "config": cfg,
            "config_source": str(config_path), "status": "recording", "validated": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "clock": "stamp_ns=MORAI sensor sec/nsec; recv_ns=host monotonic_ns; never subtract these clocks",
            "software": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__},
            "ego_source_requires_operator_confirmation": True,
            "control": {"kind": CONTROL_KIND if cfg.get("manual_control", {}).get("enabled") else "native_manual",
                        "human_steering": True},
        }

    def ego_snapshot(self):
        with self.lock:
            return self.last_ego, self.last_ego_recv

    def fail(self, message):
        with self.lock:
            if message not in self.fatal_errors:
                self.fatal_errors.append(message)
        self.stop.set()

    def enqueue(self, item):
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            self.fail("writer queue overflow: session incomplete; lower load/check disk")
            return False

    def receive(self, name, sock):
        try:
            while not self.stop.is_set():
                try:
                    packet = sock.recv(65535)
                except socket.timeout:
                    continue
                recv_ns = time.monotonic_ns()
                if name == "ego":
                    with self.lock:
                        self.counts["ego_raw"] += 1
                    try:
                        status = parse_ego(packet)
                    except ValueError as exc:
                        status = None
                        self.record_parse_error(name, exc)
                    if status is not None:
                        status["recv_ns"] = recv_ns
                        with self.lock:
                            self.counts["ego"] += 1
                            self.last_ego, self.last_ego_recv = status, recv_ns
                    self.enqueue(("ego", packet, recv_ns, status))
                else:
                    try:
                        result = self.assemblers[name].push(packet)
                    except ValueError as exc:
                        self.record_parse_error(name, exc)
                        continue
                    if result is None:
                        continue
                    stamp, jpeg = result
                    with self.lock:
                        frame_id = self.counts[name]
                        self.counts[name] += 1
                        self.latest[name] = jpeg
                    self.enqueue(("image", name, frame_id, stamp, recv_ns, jpeg))
        except Exception as exc:
            self.fail("receiver {}: {}".format(name, exc))

    def record_parse_error(self, name, exc):
        with self.lock:
            self.errors[name] += 1
            if len(self.first_errors) < 8:
                self.first_errors.append("{}: {}".format(name, exc))

    def writer(self, streams, csv_writers):
        last_flush = time.monotonic()
        failed = False
        while True:
            item = self.q.get()
            try:
                if item is None:
                    break
                if failed:
                    continue  # Drain on IO failure so shutdown cannot deadlock.
                if item[0] == "image":
                    _, name, frame_id, stamp, recv_ns, jpeg = item
                    relative = "images/{}/{:08d}.jpg".format(name, frame_id)
                    with (self.output / relative).open("xb") as stream:
                        stream.write(jpeg)
                    csv_writers["frames"].writerow(dict(zip(FRAME_FIELDS,
                        [name, frame_id, stamp, recv_ns, relative])))
                    self.saved[name] += 1
                elif item[0] == "ego":
                    _, packet, recv_ns, status = item
                    streams["raw"].write(json.dumps({"recv_ns": recv_ns, "packet_hex": packet.hex()}) + "\n")
                    self.saved["ego_raw"] += 1
                    if status is not None:
                        csv_writers["vehicle"].writerow(status)
                        self.saved["ego"] += 1
                elif item[0] == "command":
                    csv_writers["commands"].writerow(item[1])
                else:
                    _, recv_ns, event = item
                    csv_writers["events"].writerow({"recv_ns": recv_ns, "event": event})
                if time.monotonic() - last_flush > 1.0:
                    for stream in streams.values():
                        stream.flush()
                    last_flush = time.monotonic()
            except Exception as exc:
                failed = True
                self.fail("writer: {}".format(exc))
            finally:
                self.q.task_done()

    def preview(self):
        with self.lock:
            latest = dict(self.latest)
        tiles = []
        for name in ("left", "center", "right"):
            tile = np.zeros((180, 320, 3), dtype=np.uint8)
            if name in latest:
                decoded = cv2.imdecode(np.frombuffer(latest[name], np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    tile = cv2.resize(decoded, (320, 180))
            cv2.putText(tile, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            tiles.append(tile)
        cv2.imshow("PilotNet collection | q: stop, m: marker", np.hstack(tiles))
        return cv2.waitKey(1) & 0xff

    def run(self, seconds, preview):
        sockets = bind_sockets(self.cfg)
        streams, workers = {}, []
        writer = None
        start = time.monotonic()
        created = False
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            created = True
            for name in self.cfg["cameras"]:
                (self.output / "images" / name).mkdir(parents=True)
            write_json(self.output / "session.json", self.manifest)
            for key, filename in (("frames", "frames.csv"), ("vehicle", "vehicle.csv"),
                                  ("events", "events.csv"), ("raw", "ego_raw.jsonl"),
                                  ("commands", "commands.csv")):
                streams[key] = (self.output / filename).open("x", newline="", encoding="utf-8")
            csv_writers = {}
            for key, fields in (("frames", FRAME_FIELDS), ("vehicle", EGO_FIELDS),
                                ("events", ["recv_ns", "event"]), ("commands", COMMAND_FIELDS)):
                csv_writers[key] = csv.DictWriter(streams[key], fieldnames=fields)
                csv_writers[key].writeheader()
            self.enqueue(("event", time.monotonic_ns(), "start"))
            writer = threading.Thread(target=self.writer, args=(streams, csv_writers), name="writer")
            writer.start()
            for name, sock in sockets.items():
                worker = threading.Thread(target=self.receive, args=(name, sock), name=name)
                workers.append(worker)
                worker.start()
            if self.control_factory is not None:
                self.control = self.control_factory(self)
                self.control.start()
            elif self.cfg.get("manual_control", {}).get("enabled"):
                self.control = ControlWorker(self.cfg["manual_control"], self.ego_snapshot,
                                             self.enqueue, self.fail)
                self.control.start()
            print("SESSION {}".format(self.output), flush=True)
            if self.control_factory is not None:
                print("AUTOMATIC CLASSICAL EXPERT: measured Ego labels; provenance in session.json.", flush=True)
            elif self.control:
                print("HUMAN CONTROL: MORAI AV-ExternalCtrl; focus Simulator, release W once. W/up: forward <=20kph; A/D/arrows: steer; S/down/space: brake.", flush=True)
            else:
                print("Receive only (--record-only).", flush=True)
            print("Enter / preview m: marker. Ctrl+C / preview q: stop.", flush=True)
            previous, last_report, last_preview = dict(self.counts), start, 0.0
            while not self.stop.wait(.02):
                now = time.monotonic()
                if seconds and now - start >= seconds:
                    break
                if now - start >= 5.0 and self.counts["ego"] == 0:
                    self.fail("no parsed Ego after 5s: enable ordinary Ego Vehicle Status UDP to this PC, Destination Port {}; --check only tests binding".format(self.cfg["ego"]["port"]))
                    break
                if self.control and self.last_ego_recv is not None and (time.monotonic_ns() - self.last_ego_recv) / 1e9 > 2.0:
                    self.fail("Ego stream lost for >2s; assisted collection stopped with braking")
                    break
                if self.control and self.last_ego is not None and self.last_ego["ctrl_mode"] != 2:
                    self.fail("select MORAI AV-ExternalCtrl (Auto, mode=2) before assisted collection; native Keyboard mode cannot receive this governor")
                    break
                if preview and now - last_preview >= .1:
                    key = self.preview()
                    last_preview = now
                    if key in (ord("q"), 27):
                        break
                    if key == ord("m"):
                        self.enqueue(("event", time.monotonic_ns(), "operator_mark"))
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline():
                        self.enqueue(("event", time.monotonic_ns(), "operator_mark"))
                if now - last_report >= 1.0:
                    with self.lock:
                        counts, status = dict(self.counts), self.last_ego
                        ego_recv, errors = self.last_ego_recv, dict(self.errors)
                    dt = now - last_report
                    rates = " ".join("{}={:.1f}Hz".format(k, (counts[k] - previous[k]) / dt)
                                     for k in ("left", "center", "right", "ego"))
                    detail = "no parsed Ego"
                    if status is not None:
                        detail = "v={:.1f} steer={:.2f}deg Ego_age={:.0f}ms".format(
                            status["speed_kph"], status["steer_raw_deg"],
                            (time.monotonic_ns() - ego_recv) / 1e6)
                    assist = " assist=" + self.control.last_reason if self.control else ""
                    print("{} | {}{} | ego_raw={} queue={} parse_errors={}".format(
                        rates, detail, assist, counts["ego_raw"], self.q.qsize(), sum(errors.values())), flush=True)
                    previous, last_report = counts, now
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self.fail("collection: {}".format(exc))
        finally:
            if self.control is not None:
                self.control.close()
            self.stop.set()
            for worker in workers:
                worker.join()
            for sock in sockets.values():
                sock.close()
            if writer is not None:
                self.q.put(("event", time.monotonic_ns(), "stop"))
                self.q.put(None)
                writer.join()
            for stream in streams.values():
                try:
                    stream.close()
                except OSError as exc:
                    self.fail("close: {}".format(exc))
            if preview:
                cv2.destroyAllWindows()
            if created:
                if self.counts["ego"] == 0 and not self.fatal_errors:
                    self.fail("no Ego labels received: images alone are not a training session")
                self.manifest.update(
                    status="capture_error" if self.fatal_errors else "capture_finished",
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_wall_s=time.monotonic() - start,
                    received=self.counts, saved=self.saved, parse_errors=self.errors,
                    first_errors=self.first_errors, fatal_errors=self.fatal_errors,
                    incomplete_evicted={k: a.evicted for k, a in self.assemblers.items()},
                    unfinished_frames={k: len(a.pending) for k, a in self.assemblers.items()},
                )
                write_json(self.output / "session.json", self.manifest)
        print("{} saved={}".format(self.manifest["status"], self.saved), flush=True)
        for error in self.fatal_errors:
            print("ERROR: " + error, file=sys.stderr)
        return 1 if self.fatal_errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    parser.add_argument("--course", choices=["calibration", "A", "D"])
    parser.add_argument("--group", help="shared ID for correlated runs; never split a group")
    parser.add_argument("--seconds", type=float, default=0, help="0: until Ctrl+C")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--note", default="")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--record-only", action="store_true", help="disable assisted keyboard/20kph governor; native MORAI manual mode")
    parser.add_argument("--probe-ego", action="store_true", help="receive Ego only for 10s (or --seconds); no control packets")
    parser.add_argument("--check", action="store_true", help="check config/dependencies and bind ports, then exit")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be nonnegative")
    cfg = load_config(args.config)
    if args.record_only:
        cfg.setdefault("manual_control", {})["enabled"] = False
    if args.probe_ego:
        return probe_ego(cfg, args.seconds or 10.)
    if args.check:
        sockets = bind_sockets(cfg)
        for name, sock in sockets.items():
            print("{}: {} RCVBUF={}".format(name, sock.getsockname(),
                  sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)))
            sock.close()
        print("Config and ports OK. This does not confirm live sensor packets.")
        if cfg.get("manual_control", {}).get("enabled"):
            keyboard = X11Keys(cfg["manual_control"]["window_title_contains"])
            try:
                _, focused = keyboard.poll()
                print("X11 keyboard OK; focused window={!r}, driving_focus={}".format(keyboard.last_title, focused))
                print("Human-control destination: {}:{} (no packets sent by --check)".format(
                    cfg["manual_control"]["sim_ip"], cfg["manual_control"]["sim_ctrl_host_port"]))
            finally:
                keyboard.close()
        return 0
    if args.course is None or not args.group or not re.fullmatch(r"[A-Za-z0-9_-]+", args.group):
        parser.error("--course and --group required; group uses letters, digits, _ or -")
    root = args.output_root or BASE / cfg["recording"]["root"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.course + "_" + uuid.uuid4().hex[:6]
    recorder = Recorder(cfg, args.config.resolve(), root.resolve() / run_id, args.course, args.group, args.note)
    preview = not args.no_preview and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return recorder.run(args.seconds, preview)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
