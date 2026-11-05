"""Human keyboard steering with a longitudinal governor for MORAI simulation.

Requires AV-ExternalCtrl (Auto) and ordinary Ego state. X11 is queried for only
the driving keys; no keyboard injection, grabs, root privileges, or ML is used.
The measured Ego steering remains the training label. Mode 2 is transport, not
an autonomous steering policy. Nothing sends until ControlWorker.run is invoked.
"""
import ctypes
import ctypes.util
import math
import os
import socket
import struct
import threading
import time

COMMAND_FIELDS = ["recv_ns", "sent", "keys", "focused", "ego_fresh", "reason",
                  "speed_kph", "target_kph", "accel", "brake", "steer_norm"]
CONTROL_KIND = "human_keyboard_speed_limited_v1"


def encode_command(accel, brake, steer_norm):
    """Ordinary MoraiCmdController UDP: Auto, D, throttle control, 55 bytes."""
    if not all(math.isfinite(x) for x in (accel, brake, steer_norm)):
        raise ValueError("non-finite command")
    if not (0 <= accel <= 1 and 0 <= brake <= 1 and -1 <= steer_norm <= 1):
        raise ValueError("command outside MORAI ranges")
    if accel > 0 and brake > 0:
        raise ValueError("simultaneous throttle and brake")
    payload = struct.pack("<BBB5f", 2, 4, 1, 0., 0., accel, brake, steer_norm)
    return b"#MoraiCtrlCmd$" + struct.pack("<I", 23) + bytes(12) + payload + b"\r\n"


class X11Keys:
    """Poll driving key state, allowing actuation only in intended windows."""
    SYMBOLS = {"forward": ("w", "W", "Up"), "back": ("s", "S", "Down"),
               "left": ("a", "A", "Left"), "right": ("d", "D", "Right"),
               "brake": ("space",), "escape": ("Escape",)}

    def __init__(self, title_fragments):
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            raise RuntimeError("keyboard governor requires an Xorg session; use --record-only for native manual input")
        path = ctypes.util.find_library("X11")
        if not path:
            raise RuntimeError("libX11 not found")
        self.x = ctypes.CDLL(path)
        p, window = ctypes.c_void_p, ctypes.c_ulong
        self.x.XOpenDisplay.argtypes, self.x.XOpenDisplay.restype = [ctypes.c_char_p], p
        self.x.XCloseDisplay.argtypes = [p]
        self.x.XStringToKeysym.argtypes, self.x.XStringToKeysym.restype = [ctypes.c_char_p], ctypes.c_ulong
        self.x.XKeysymToKeycode.argtypes, self.x.XKeysymToKeycode.restype = [p, ctypes.c_ulong], ctypes.c_ubyte
        self.x.XQueryKeymap.argtypes = [p, ctypes.c_void_p]
        self.x.XGetInputFocus.argtypes = [p, ctypes.POINTER(window), ctypes.POINTER(ctypes.c_int)]
        self.x.XFetchName.argtypes = [p, window, ctypes.POINTER(p)]
        self.x.XQueryTree.argtypes = [p, window, ctypes.POINTER(window), ctypes.POINTER(window),
                                      ctypes.POINTER(p), ctypes.POINTER(ctypes.c_uint)]
        self.x.XFree.argtypes = [p]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("cannot open X11 DISPLAY; start from the local desktop")
        self.fragments = [name.casefold() for name in title_fragments]
        self.codes = {}
        for name, symbols in self.SYMBOLS.items():
            self.codes[name] = [self.x.XKeysymToKeycode(self.display, self.x.XStringToKeysym(s.encode()))
                                for s in symbols]
        self.last_title = ""

    def window_title(self):
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        self.x.XGetInputFocus(self.display, ctypes.byref(window), ctypes.byref(revert))
        for _ in range(8):
            if window.value <= 1:
                return ""
            name = ctypes.c_void_p()
            if self.x.XFetchName(self.display, window, ctypes.byref(name)) and name.value:
                try:
                    title = ctypes.string_at(name.value).decode("utf-8", errors="replace")
                finally:
                    self.x.XFree(name)
                if title:
                    return title
            root, parent, children, count = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p(), ctypes.c_uint()
            ok = self.x.XQueryTree(self.display, window, ctypes.byref(root), ctypes.byref(parent),
                                   ctypes.byref(children), ctypes.byref(count))
            if children.value:
                self.x.XFree(children)
            if not ok or parent.value == window.value:
                return ""
            window = parent
        return ""

    def poll(self):
        bitmap = (ctypes.c_ubyte * 32)()
        self.x.XQueryKeymap(self.display, bitmap)
        keys = {name for name, codes in self.codes.items()
                if any(code and bitmap[code // 8] & (1 << (code % 8)) for code in codes)}
        self.last_title = self.window_title()
        focused = any(fragment in self.last_title.casefold() for fragment in self.fragments)
        return keys, focused

    def close(self):
        if self.display:
            self.x.XCloseDisplay(self.display)
            self.display = None


class Governor:
    """Pure human-input/feedback controller; no IO, geometry, or path following."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.steer_deg = 0.
        self.integral = 0.
        self.armed = False

    def step(self, keys, focused, status, age_s, dt):
        cfg = self.cfg
        dt = min(max(dt, .001), .1)
        fresh = (status is not None and age_s <= cfg["max_ego_age_s"]
                 and math.isfinite(status["speed_kph"]))
        speed = float(status["speed_kph"]) if status is not None else 0.
        target = cfg["speed_limit_kph"] - cfg["target_margin_kph"]
        out = dict(keys="+".join(sorted(keys)), focused=int(focused), ego_fresh=int(fresh),
                   speed_kph=speed, target_kph=target, accel=0., brake=1.,
                   steer_norm=self.steer_deg / cfg["max_wheel_angle_deg"] * cfg["command_left_sign"])
        if not fresh or not focused or "escape" in keys:
            self.armed = False
            self.integral = 0.
            out["reason"] = "stale_ego" if not fresh else ("focus_lost" if not focused else "escape")
            return out
        if status.get("ctrl_mode") != 2:
            self.armed = False
            self.integral = 0.
            out["reason"] = "wrong_mode_select_AV_ExternalCtrl"
            return out
        # Require W release after starting, losing focus, or a stale-state stop.
        if "forward" not in keys:
            self.armed = True
        direction = int("left" in keys) - int("right" in keys)
        if direction:
            self.steer_deg += direction * cfg["steer_rate_deg_s"] * dt
        else:
            amount = min(abs(self.steer_deg), cfg["steer_return_deg_s"] * dt)
            self.steer_deg -= math.copysign(amount, self.steer_deg)
        cap = cfg["manual_steer_limit_deg"]
        self.steer_deg = min(cap, max(-cap, self.steer_deg))
        out["steer_norm"] = self.steer_deg / cfg["max_wheel_angle_deg"] * cfg["command_left_sign"]
        if "brake" in keys or "back" in keys or speed < -.5:
            self.integral = 0.
            out.update(reason="brake", brake=1.)
        elif not self.armed:
            out.update(reason="release_forward_to_arm", brake=1.)
        elif "forward" not in keys:
            self.integral = 0.
            out.update(reason="idle", brake=.5)
        else:
            error = (target - speed) / 3.6
            # Anti-windup: integrate near the target, reset during saturation.
            self.integral = min(.5, max(-.5, self.integral + error * dt)) if abs(error) < 1 else 0.
            requested = min(cfg["max_accel_mps2"], max(-3., 1.2 * error + .35 * self.integral))
            # Brake before the ceiling. This is physical feedback, not a clamp of
            # reported speed or recorded labels; transient overshoot is measurable.
            if speed >= cfg["speed_limit_kph"] - .3:
                requested = min(requested, -.6)
            if speed >= cfg["speed_limit_kph"]:
                requested = min(requested, -3.2)
            out.update(reason="drive", brake=0., accel=0.)
            if requested >= 0:
                out["accel"] = min(cfg["max_throttle"], requested / 5.)
            else:
                out["brake"] = min(1., -requested / 8.)
        return out


class ControlWorker:
    def __init__(self, cfg, snapshot, enqueue, fail, key_source=None):
        self.cfg, self.snapshot, self.enqueue, self.fail = cfg, snapshot, enqueue, fail
        self.keys = key_source or X11Keys(cfg["window_title_contains"])
        self.stop_event = threading.Event()
        self.controller = Governor(cfg)
        self.thread = None
        self.last_reason = "starting"
        self.sent_count = 0

    def start(self):
        self.thread = threading.Thread(target=self.run, name="human-control")
        self.thread.start()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        destination = (self.cfg["sim_ip"], self.cfg["sim_ctrl_host_port"])
        interval = 1 / self.cfg["hz"]
        previous = time.monotonic()
        last = None
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                status, recv_ns = self.snapshot()
                age = (time.monotonic_ns() - recv_ns) / 1e9 if recv_ns is not None else float("inf")
                keys, focused = self.keys.poll()
                command = self.controller.step(keys, focused, status, age, now - previous)
                previous = now
                self.last_reason = command["reason"]
                # No outbound packets before the first usable Ego state. After
                # a state loss, braking stays available using the last steering.
                sent = status is not None and status.get("ctrl_mode") == 2
                if sent:
                    sock.sendto(encode_command(command["accel"], command["brake"], command["steer_norm"]), destination)
                    self.sent_count += 1
                command.update(recv_ns=time.monotonic_ns(), sent=int(sent))
                self.enqueue(("command", command))
                last = command
                self.stop_event.wait(max(0, interval - (time.monotonic() - now)))
        except Exception as exc:
            self.fail("manual control: {}".format(exc))
        finally:
            # Keep receiving Ego state while braking at normal shutdown; duration
            # is bounded. Do not switch back to native Keyboard with W held.
            if self.sent_count:
                deadline = time.monotonic() + self.cfg["shutdown_brake_s"]
                minimum = time.monotonic() + min(.3, self.cfg["shutdown_brake_s"])
                while time.monotonic() < deadline:
                    try:
                        steer = last["steer_norm"] if last else 0.
                        sock.sendto(encode_command(0., 1., steer), destination)
                        status, _ = self.snapshot()
                        if time.monotonic() >= minimum and status is not None and abs(status["speed_kph"]) < .5:
                            break
                    except OSError:
                        break
                    time.sleep(interval)
            sock.close()
            self.keys.close()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        else:
            self.keys.close()
