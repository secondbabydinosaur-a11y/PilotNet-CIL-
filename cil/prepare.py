"""검사한 원본 -> 고정된 train/val 입력·정답 CSV. 영상 전처리는 Dataset에서 한다.

좌우 정답은 동일한 카메라 시점을 만드는 가상 차량의 PP 조향 차이로 보정한다.
보정 수식과 2초·최소 6m는 이번 구현의 선택이며 원 논문의 공개 공식이 아니다.
"""
import argparse  # 실행 옵션을 읽는다.
import ast  # Dataset 상수를 실행 없이 기록한다.
import bisect  # 이미지 시각 앞뒤의 상태를 찾는다.
import csv  # 원본과 정답 목록을 읽고 쓴다.
import hashlib  # 입력 자료·코드 버전을 식별한다.
import json  # 검사 기록과 준비 결과를 저장한다.
import math  # 회전과 조향각을 계산한다.
import shutil  # 실패한 임시 출력만 정리한다.
import tempfile  # 완료 전 결과를 기존 데이터와 분리한다.
from collections import Counter, defaultdict  # 선택·제외 사유를 집계한다.
from dataclasses import dataclass  # 시각 매칭 결과를 묶는다.
from datetime import datetime, timezone  # 준비 시각을 기록한다.
from pathlib import Path  # CIL 기준 경로를 사용한다.

import numpy as np  # 경로 보간과 좌표 계산에 사용한다.
from scipy.spatial import cKDTree  # 시험 구역과의 거리를 확인한다.

BASE = Path(__file__).resolve().parent  # CIL 폴더다.
CAMERAS = ("center", "left", "right")  # 카메라 역할이다.
TEACHER_COMPARE = ("command", "teacher_steer", "teacher_throttle", "teacher_brake", "control_phase", "label_valid", "route_s_m", "noise_episode", "recovery_active")  # 중복 시각에서 동일해야 하는 정답이다.
FIELDS = ["image_path", "camera", "speed_kph", "command", "steer", "throttle", "brake", "label_source", "split", "session_id", "route_id", "stamp_ns", "recv_ns", "ego_left_ns", "ego_right_ns", "ego_alpha", "teacher_left_ns", "teacher_right_ns", "teacher_alpha", "center_teacher_steer", "steer_correction_deg", "route_s_m", "x_m", "y_m", "yaw_deg", "noise_episode", "recovery_active", "noise_steer", "label_method"]  # 입력·정답과 추적용 정보다.


class Rejected(ValueError):  # 해당 프레임을 제외할 이유다.
    pass


def read_csv(path):  # 열 이름으로 CSV를 읽는다.
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path):  # 입력 자료를 식별한다.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Match:  # 앞뒤 표본과 보간 계수다.
    left: dict
    right: dict
    alpha: float
    left_ns: int
    right_ns: int


class Timeline:
    def __init__(self, rows, stamp_key, compare_fields=()):
        unique = {}  # 같은 센서 시각은 하나의 상태다.
        self.duplicates, self.conflicts = 0, set()  # 중복과 모순을 구분한다.
        previous = None  # 시계 재설정을 정렬로 숨기지 않는다.
        for row in rows:
            stamp = int(row[stamp_key])  # 정수 나노초를 유지한다.
            if previous is not None and stamp < previous - 1_000_000_000:
                raise ValueError("sensor clock reset inside session")
            previous = max(stamp, previous) if previous is not None else stamp
            if stamp in unique:
                self.duplicates += 1
                if any(row[key] != unique[stamp][key] for key in compare_fields):
                    self.conflicts.add(stamp)  # 서로 다른 정답이면 사용하지 않는다.
                old = unique[stamp]
                if int(row.get("sent_ns", row.get("recv_ns", 0))) >= int(old.get("sent_ns", old.get("recv_ns", 0))):
                    continue  # 가장 먼저 대응한 행을 보존한다.
            unique[stamp] = row
        self.stamps = sorted(unique)
        self.rows = [unique[stamp] for stamp in self.stamps]

    def bracket(self, stamp_ns, max_delta_ns=25_000_000, max_gap_ns=50_000_000):
        i = bisect.bisect_left(self.stamps, stamp_ns)  # 이미지 직후의 표본 번호다.
        if i < len(self.stamps) and self.stamps[i] == stamp_ns:
            if stamp_ns in self.conflicts: return None
            return Match(self.rows[i], self.rows[i], 0.0, stamp_ns, stamp_ns)
        if i == 0 or i == len(self.stamps): return None  # 관측 시간 밖으로 외삽하지 않는다.
        left, right = self.stamps[i - 1], self.stamps[i]
        if left in self.conflicts or right in self.conflicts: return None
        if right - left > max_gap_ns or min(stamp_ns - left, right - stamp_ns) > max_delta_ns:
            return None  # 긴 관측 공백을 보간으로 채우지 않는다.
        return Match(self.rows[i - 1], self.rows[i], (stamp_ns - left) / (right - left), left, right)


def linear(match, key):  # 큰 절대 시각 대신 값만 실수로 보간한다.
    a, b = float(match.left[key]), float(match.right[key])
    if not math.isfinite(a) or not math.isfinite(b): raise Rejected("nonfinite_value")
    return a + match.alpha * (b - a)


def interpolate_ego(match):
    for row in (match.left, match.right):
        if int(row["ctrl_mode"]) != 2 or int(row["gear"]) != 4:
            raise Rejected("not_forward_auto")
        if not math.isfinite(float(row["speed_kph"])) or float(row["speed_kph"]) < -0.05:
            raise Rejected("invalid_ego")  # 후진 표본을 양의 속도와 섞어 감추지 않는다.
    result = {key: linear(match, key) for key in ("speed_kph", "x_m", "y_m", "z_m")}
    yaw0, yaw1 = float(match.left["yaw_deg"]), float(match.right["yaw_deg"])
    if not math.isfinite(yaw0 + yaw1) or result["speed_kph"] < -0.05: raise Rejected("invalid_ego")
    result["yaw_deg"] = yaw0 + match.alpha * ((yaw1 - yaw0 + 180.0) % 360.0 - 180.0)  # 각도 경계에서 짧은 방향으로 보간한다.
    result["speed_kph"] = max(0.0, result["speed_kph"])  # 정지 중 미세한 부호 잡음만 제거한다.
    return result


def interpolate_teacher(match):
    a, b = match.left, match.right
    if any(int(row["label_valid"]) != 1 or row["control_phase"] != "drive" for row in (a, b)):
        raise Rejected("invalid_control_phase")  # 임의 경로 끝의 정차를 제외한다.
    if a["command"] != b["command"]: raise Rejected("command_transition")  # 명령 번호는 보간하지 않는다.
    if int(a["command"]) not in (0, 1, 2, 3): raise Rejected("unknown_command")
    for row in (a, b):
        steer, gas, brake = [float(row[key]) for key in ("teacher_steer", "teacher_throttle", "teacher_brake")]
        if not all(math.isfinite(value) for value in (steer, gas, brake)) or abs(steer) > 1 or not 0 <= gas <= 1 or not 0 <= brake <= 1 or gas * brake > 0:
            raise Rejected("invalid_teacher_range")
    if float(a["teacher_throttle"]) * float(b["teacher_brake"]) > 0 or float(b["teacher_throttle"]) * float(a["teacher_brake"]) > 0:
        raise Rejected("pedal_transition")  # 두 페달이 동시에 양수가 되는 보간을 제외한다.
    return {
        "steer": linear(match, "teacher_steer"),  # 교란이 포함된 applied_steer는 정답이 아니다.
        "throttle": linear(match, "teacher_throttle"), "brake": linear(match, "teacher_brake"),
        "command": int(a["command"]), "route_s_m": linear(match, "route_s_m"),
        "noise_episode": int(a["noise_episode"]), "recovery_active": max(int(a["recovery_active"]), int(b["recovery_active"])),
        "noise_steer": linear(match, "noise_steer"),
    }


def rotation(yaw):  # 차량 평면의 회전 행렬이다.
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def virtual_pose(ego, camera, center):
    delta = math.radians(float(camera["rpy_deg"][2]) - float(center["rpy_deg"][2]))
    yaw = math.radians(ego["yaw_deg"])
    shift = np.asarray(camera["xyz_m"][:2]) - rotation(delta) @ np.asarray(center["xyz_m"][:2])  # 전방 2.6m 장착 거리도 회전에 반영한다.
    xy = np.array([ego["x_m"], ego["y_m"]]) + rotation(yaw) @ shift
    return np.r_[xy, yaw + delta]  # 이 자세의 가운데 카메라와 실제 좌우 카메라의 평면 시점이 같다.


def pp_angle(pose, goal, wheelbase):
    difference = np.asarray(goal) - pose[:2]
    local = rotation(-pose[2]) @ difference
    if local[0] <= 0 or float(local @ local) < 0.01: raise Rejected("recovery_goal_behind")
    return math.atan(2.0 * wheelbase * local[1] / float(local @ local))


def correct_side_steering(ego, teacher_steer, station_m, route, camera, center, wheelbase=3.0, max_angle_deg=40.0, recovery_s=2.0, min_lookahead_m=6.0):
    if not math.isfinite(teacher_steer) or abs(teacher_steer) > 1:
        raise Rejected("invalid_teacher_range")
    distance = max(min_lookahead_m, recovery_s * ego["speed_kph"] / 3.6)
    target_s = station_m + distance  # 같은 미래 경로점을 두 자세에서 바라본다.
    if station_m < route[0, 0] - 0.01 or target_s > route[-1, 0]: raise Rejected("missing_recovery_route_context")
    goal = np.array([np.interp(target_s, route[:, 0], route[:, k]) for k in (1, 2)])
    original = np.array([ego["x_m"], ego["y_m"], math.radians(ego["yaw_deg"])])
    shifted = virtual_pose(ego, camera, center)
    max_angle = math.radians(max_angle_deg)
    label = teacher_steer + (pp_angle(shifted, goal, wheelbase) - pp_angle(original, goal, wheelbase)) / max_angle
    if not math.isfinite(label) or abs(label) > 1: raise Rejected("side_steering_out_of_range")  # 포화 정답을 만들지 않고 제외한다.
    return float(label)


def preprocessing_constants(path):  # Torch를 실행하지 않고 실제 Dataset 설정을 기록한다.
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CROP", "IMAGE_SIZE", "SPEED_SCALE_KPH"):
                    result[target.id] = ast.literal_eval(node.value)
    return result


def build(base=BASE, output=None, sample_hz=10.0, recovery_s=2.0, min_lookahead_m=6.0, *, plan=None, completion=None):
    base = Path(base).resolve()
    cfg = json.loads((base / "config.json").read_text())
    plan_path = base / (plan if plan is not None else cfg["paths"]["route_plan"])
    plan = json.loads(plan_path.read_text())
    completion_path = base / (completion if completion is not None else "data/collection_completion.json")
    completion = json.loads(completion_path.read_text())
    if completion["status"] != "complete" or not completion["accepted"]: raise ValueError("finish collection and quality checks first")
    if not all(math.isfinite(v) and v > 0 for v in (sample_hz, recovery_s, min_lookahead_m)) or sample_hz > cfg["camera_common"]["fps"]:
        raise ValueError("invalid sampling or recovery parameters")
    output = base / cfg["paths"]["prepared_data"] if output is None else Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("prepared output already exists; choose another --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepared-", dir=str(output.parent)))
    sources, reports, seen, totals = {}, [], set(), Counter()
    period_ns = round(1e9 / sample_hz)
    max_delta = round(cfg["quality"]["max_pair_delta_ms"] * 1e6)
    max_gap = round(cfg["quality"]["max_interpolation_gap_ms"] * 1e6)
    protected = {name: np.array(value["guard_reference_xyz"])[:, :2] for name, value in plan["test_reservations"].items()}
    for name in ("P2Q", "V1", "V2"):
        protected[name] = np.loadtxt(base / plan["routes"][name]["route_csv"], delimiter=",", skiprows=1)[:, 1:3]
    trees = {name: cKDTree(points) for name, points in protected.items()}

    def remember(path):  # 입력 색인·설정이 준비 도중 바뀌지 않는지 확인한다.
        sources[str(path.relative_to(base))] = sha256(path)

    try:
        for path in (base / "config.json", base / "prepare.py", base / "dataset.py", completion_path, plan_path): remember(path)
        with (stage / "train.csv").open("x", newline="", encoding="utf-8") as train_file, (stage / "val.csv").open("x", newline="", encoding="utf-8") as val_file:
            writers = {"train": csv.DictWriter(train_file, fieldnames=FIELDS), "val": csv.DictWriter(val_file, fieldnames=FIELDS)}
            for writer in writers.values(): writer.writeheader()
            for item in completion["accepted"]:
                session = (base / item["session"]).resolve()
                if session in seen: raise ValueError("session included twice")
                seen.add(session)
                paths = [session / name for name in ("session.json", "quality_report.json", "images.csv", "vehicle.csv", "commands.csv")]
                for path in paths: remember(path)
                meta, quality = [json.loads(path.read_text()) for path in paths[:2]]
                if meta["status"] != "route_finished" or not meta.get("collection_accepted") or not quality["data_checks_passed"]:
                    raise ValueError("source session did not pass collection quality checks")
                split = {"train": "train", "validation": "val"}.get(meta["split"])
                info = plan["routes"][meta["route_id"]]
                if split is None or item["split"] != meta["split"] or info["split"] != meta["split"]:
                    raise ValueError("source split changed; no random frame splitting is allowed")
                for key in ("vehicle", "cameras", "camera_common", "commands", "quality"):
                    if meta["config"][key] != cfg[key]: raise ValueError("capture and preparation config differ: " + key)
                for camera in cfg["cameras"].values():
                    if camera["rpy_deg"][:2] != cfg["cameras"]["center"]["rpy_deg"][:2] or camera["xyz_m"][2] != cfg["cameras"]["center"]["xyz_m"][2]:
                        raise ValueError("planar side correction requires common height/roll/pitch")
                route_path = base / info["route_csv"]
                remember(route_path)
                if sha256(route_path) != meta["route_sha256"]: raise ValueError("capture route changed")
                route = np.loadtxt(route_path, delimiter=",", skiprows=1)
                ego_table = Timeline(read_csv(session / "vehicle.csv"), "stamp_ns")
                teacher_table = Timeline(read_csv(session / "commands.csv"), "ego_stamp_ns", TEACHER_COMPARE)
                image_rows = read_csv(session / "images.csv")
                origin = min(int(row["stamp_ns"]) for row in image_rows)
                groups = defaultdict(list)
                for row in image_rows:
                    if row["camera"] not in CAMERAS: raise ValueError("unknown camera in raw image index")
                    if split == "val" and row["camera"] != "center": continue  # 실제 추론과 같은 가운데 영상만 검증한다.
                    slot = (int(row["stamp_ns"]) - origin) // period_ns
                    groups[(row["camera"], slot)].append(row)
                accepted, rejected, commands, selected_xy = Counter(), Counter(), Counter(), []
                for (camera, slot), candidates in sorted(groups.items()):
                    target_stamp = origin + slot * period_ns + period_ns // 2
                    for image_row in sorted(candidates, key=lambda row: (abs(int(row["stamp_ns"]) - target_stamp), int(row["stamp_ns"]))):
                        stamp = int(image_row["stamp_ns"])
                        try:
                            ego_match = ego_table.bracket(stamp, max_delta, max_gap)
                            teacher_match = teacher_table.bracket(stamp, max_delta, max_gap)
                            if ego_match is None: raise Rejected("ego_time_match")
                            if teacher_match is None: raise Rejected("teacher_time_match")
                            ego = interpolate_ego(ego_match)
                            teacher = interpolate_teacher(teacher_match)
                            steer = teacher["steer"] if camera == "center" else correct_side_steering(ego, teacher["steer"], teacher["route_s_m"], route, cfg["cameras"][camera], cfg["cameras"]["center"], cfg["vehicle"]["wheelbase_m"], cfg["vehicle"]["max_wheel_angle_deg"], recovery_s, min_lookahead_m)
                        except Rejected as exc:
                            rejected[camera + ":" + str(exc)] += 1
                            continue  # 같은 시간칸의 다른 프레임도 확인한다.
                        image_path = (session / image_row["image_path"]).resolve()
                        image_path.relative_to(session)  # 원본 세션 밖을 참조하지 않는다.
                        if not image_path.is_file(): raise FileNotFoundError(image_path)
                        row = dict(image_path=str(image_path.relative_to(base)), camera=camera, speed_kph=ego["speed_kph"], command=teacher["command"], steer=steer, throttle=teacher["throttle"], brake=teacher["brake"], label_source="teacher" if camera == "center" else "camera_recovery", split=split, session_id=session.name, route_id=meta["route_id"], stamp_ns=stamp, recv_ns=image_row["recv_ns"], ego_left_ns=ego_match.left_ns, ego_right_ns=ego_match.right_ns, ego_alpha=ego_match.alpha, teacher_left_ns=teacher_match.left_ns, teacher_right_ns=teacher_match.right_ns, teacher_alpha=teacher_match.alpha, center_teacher_steer=teacher["steer"], steer_correction_deg=(steer - teacher["steer"]) * cfg["vehicle"]["max_wheel_angle_deg"], route_s_m=teacher["route_s_m"], x_m=ego["x_m"], y_m=ego["y_m"], yaw_deg=ego["yaw_deg"], noise_episode=teacher["noise_episode"], recovery_active=teacher["recovery_active"], noise_steer=teacher["noise_steer"], label_method="teacher_time_interpolation" if camera == "center" else "virtual_center_camera_pp_angle_delta_v1")
                        writers[split].writerow(row)
                        accepted[camera] += 1
                        commands[str(teacher["command"])] += 1
                        totals[split + ":" + camera] += 1
                        selected_xy.append([ego["x_m"], ego["y_m"]])
                        break  # 100ms 시간칸마다 카메라별 유효 영상 한 장을 선택한다.
                guards = {name: float(tree.query(np.array(selected_xy))[0].min()) for name, tree in trees.items() if selected_xy and (split == "train" or name not in ("V1", "V2"))}
                if not selected_xy or any(value < 20.0 for value in guards.values()): raise ValueError("empty session or selected sample entered a protected guard")
                reports.append(dict(session=session.name, route_id=meta["route_id"], split=split, accepted=dict(accepted), rejected_candidates=dict(rejected), command_counts=dict(commands), ego_duplicates=ego_table.duplicates, teacher_duplicates=teacher_table.duplicates, conflicting_teacher_stamps=len(teacher_table.conflicts), protected_min_distances_m=guards))
                print(session.name, split, dict(accepted), flush=True)
        if any(totals[key] == 0 for key in ("train:center", "train:left", "train:right", "val:center")): raise ValueError("required train/validation cameras are empty")
        if any(sha256(base / name) != value for name, value in sources.items()): raise ValueError("source changed during preparation")
        summary = dict(status="prepared", created_utc=datetime.now(timezone.utc).isoformat(), counts=dict(totals), sessions=reports, nominal_sample_hz=sample_hz, sampling="one valid frame nearest bin center per camera per time bin; actual frame intervals may differ", teacher_reference="ego_stamp_ns sensor clock; first identical duplicate", validation="center-only, spatially separate routes", test="U1/U2/J/P2Q held out; no test frames generated", side_label_method="virtual_center_camera_pp_angle_delta_v1", recovery_s=recovery_s, min_lookahead_m=min_lookahead_m, side_label_note="our planar geometric approximation, not the unpublished original paper correction; over-range labels rejected", image_preprocessing=preprocessing_constants(base / "dataset.py"), source_sha256=sources, cil_model_trained=False)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if output.exists(): output.rmdir()  # 기존의 빈 폴더만 교체한다.
        stage.rename(output)  # 검사를 마친 결과를 한 번에 확정한다.
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)  # 원본과 기존 준비 결과는 수정하지 않는다.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)  # 다시 만들 때는 별도 버전 폴더를 지정한다.
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--recovery-s", type=float, default=2.0)
    parser.add_argument("--min-lookahead-m", type=float, default=6.0)
    parser.add_argument("--plan", type=Path, help="use a separate CIL-relative route plan")
    parser.add_argument("--completion", type=Path, help="use a separate CIL-relative accepted session list")
    args = parser.parse_args()
    if (args.plan or args.completion) and not args.output:
        parser.error("a separate --output is required with --plan or --completion")
    summary = build(output=args.output, sample_hz=args.sample_hz, recovery_s=args.recovery_s,
                    min_lookahead_m=args.min_lookahead_m, plan=args.plan, completion=args.completion)
    print("PREPARED:", summary["counts"])


if __name__ == "__main__":
    main()
