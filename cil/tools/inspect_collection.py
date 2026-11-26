"""원본 수집의 영상·시각 정렬·Teacher 기록·구역 분리를 검사한다. 학습용 라벨은 만들지 않는다."""
import argparse  # 검사할 세션을 지정한다.
import csv  # 수집 원본 CSV를 읽는다.
import json  # 품질 보고서를 저장한다.
from collections import Counter  # 명령별 표본을 센다.
from concurrent.futures import ThreadPoolExecutor  # 원본 이미지 검사를 병렬화한다.
from pathlib import Path  # CIL 경로를 찾는다.
import cv2  # JPEG가 실제로 해독되는지 검사한다.
import numpy as np  # 타임스탬프와 오차 통계를 계산한다.
from scipy.spatial import cKDTree  # 시험 구역과의 실제 거리를 검사한다.

BASE = Path(__file__).resolve().parents[1]  # cil 폴더다.


def read_csv(path):  # 열 이름을 유지해 CSV를 읽는다.
    with path.open(newline="", encoding="utf-8-sig") as stream:  # BOM도 허용한다.
        return list(csv.DictReader(stream))  # 행 목록을 반환한다.


def stream_stats(stamps):  # 수신 주기와 긴 공백을 계산한다.
    stamps = np.unique(np.asarray(stamps, dtype=np.int64))  # 정상적인 UDP 순서 뒤바뀜과 중복을 정리한다.
    if len(stamps) < 2:  # 주기를 계산할 수 없다면
        return {"samples": int(len(stamps)), "hz": 0.0, "max_gap_s": 0.0}  # 부족한 데이터다.
    delta = np.diff(stamps) / 1e9  # 센서 시각 차이를 초로 바꾼다.
    return {"samples": int(len(stamps)), "hz": float((len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1e9)), "max_gap_s": float(delta.max())}  # 통계다.


def match_fraction(image_stamps, reference, max_delta_ms=25.0, max_gap_ms=50.0):  # 보간 가능한 센서·Teacher 대응 비율이다.
    images = np.asarray(image_stamps, dtype=np.int64)  # 검사할 이미지 시각이다.
    ref = np.unique(np.asarray(reference, dtype=np.int64))  # 대응할 상태·제어 시각이다.
    if not len(images) or len(ref) < 2:  # 보간 기준이 부족하면
        return 0.0  # 대응 실패다.
    right = np.searchsorted(ref, images)  # 이미지 직후의 표본 번호다.
    ri = np.clip(right, 0, len(ref) - 1)  # 배열 범위를 제한한다.
    li = np.clip(right - 1, 0, len(ref) - 1)  # 직전 표본 번호다.
    exact = ref[ri] == images  # 정확히 같은 시각은 보간할 필요가 없다.
    bracket = (right > 0) & (right < len(ref))  # 앞뒤 표본이 모두 있어야 한다.
    nearest = np.minimum(np.abs(ref[ri] - images), np.abs(ref[li] - images)) / 1e6  # 최근접 시간 차이다.
    gap = (ref[ri] - ref[li]) / 1e6  # 앞뒤 표본 사이 시간이다.
    good = exact | (bracket & (nearest <= max_delta_ms) & (gap <= max_gap_ms))  # 조건을 함께 검사한다.
    return float(good.mean())  # 유효 대응 비율이다.


def inspect(session, decode_all=True):  # 세션 하나의 원본 품질을 검사한다.
    session = Path(session)  # 검사할 저장 폴더다.
    metadata = json.loads((session / "session.json").read_text())  # 수집 상태와 설정이다.
    cfg = metadata["config"]  # 해당 실행의 실제 설정이다.
    images = read_csv(session / "images.csv")  # 원본 이미지 목록이다.
    vehicle = read_csv(session / "vehicle.csv")  # 차량 상태다.
    commands = read_csv(session / "commands.csv")  # Teacher·실제 전송값이다.
    errors = []  # 검사 실패 사유다.
    if metadata["status"] != "route_finished": errors.append("capture did not finish route: " + metadata["status"])  # 부분 수집을 완주로 처리하지 않는다.
    if metadata["split"] not in ("train", "validation"): errors.append("session is not train/validation")  # 점검 기록을 제외한다.
    streams = {"ego": stream_stats([int(row["stamp_ns"]) for row in vehicle])}  # 차량 상태 주기다.
    ego_stamps = [int(row["stamp_ns"]) for row in vehicle]  # 영상과 대응할 기준이다.
    teacher_stamps = [int(row["ego_stamp_ns"]) for row in commands]  # Teacher가 참조한 센서 시각이다.
    quality = cfg["quality"]  # 이번 실행의 시각 정렬 기준이다.
    max_stream_gap_s = float(cfg["collection"].get("max_capture_gap_s", 0.25))
    for name in cfg["cameras"]:  # 세 카메라를 따로 검사한다.
        stamps = [int(row["stamp_ns"]) for row in images if row["camera"] == name]  # 이미지 센서 시각이다.
        stat = stream_stats(stamps)  # 수신 주기와 공백이다.
        stat["ego_match_fraction"] = match_fraction(stamps, ego_stamps, quality["max_pair_delta_ms"], quality["max_interpolation_gap_ms"])  # 차량 상태 대응이다.
        stat["teacher_match_fraction"] = match_fraction(stamps, teacher_stamps, quality["max_pair_delta_ms"], quality["max_interpolation_gap_ms"])  # Teacher 대응이다.
        streams[name] = stat  # 카메라별 보고서다.
        if stat["hz"] < cfg["camera_common"]["fps"] * 0.85: errors.append(name + ": low camera rate")  # 저주기 수신을 검출한다.
        if stat["max_gap_s"] > max_stream_gap_s: errors.append(name + ": long camera gap")  # 긴 영상 누락을 검출한다.
        if stat["ego_match_fraction"] < 0.95: errors.append(name + ": insufficient Ego matches")  # 잘못된 시각 대응이다.
        if stat["teacher_match_fraction"] < 0.94: errors.append(name + ": insufficient Teacher matches")  # 발진 전 짧은 준비 영상은 허용한다.
    if streams["ego"]["hz"] < cfg["ego"]["hz"] * 0.85: errors.append("low Ego rate")  # 상태 수신 주기를 검사한다.
    if streams["ego"]["max_gap_s"] > max_stream_gap_s: errors.append("long Ego gap")  # 상태 누락을 검사한다.
    valid = [row for row in commands if row.get("label_valid") == "1"]  # 임의 경로 끝 정차를 제외한다.
    if not valid: errors.append("no valid Teacher driving samples")  # 학습 가능한 주행이 있어야 한다.
    cte = np.array([abs(float(row["cte_m"])) for row in commands])  # 실제 추종 오차다.
    heading = np.array([abs(float(row["heading_error_deg"])) for row in commands])  # 실제 헤딩 오차다.
    speed = np.array([float(row["speed_kph"]) for row in vehicle])  # 실제 차속이다.
    duration = 0.0  # 유효 주행 시각의 길이다.
    command_seconds = Counter()  # 진행 명령별 실제 시간이다.
    for row, following in zip(commands, commands[1:]):  # 인접 상태 기준으로 시간을 합산한다.
        dt = max(0.0, min(0.1, (int(following["ego_stamp_ns"]) - int(row["ego_stamp_ns"])) / 1e9))  # 중복과 공백을 제한한다.
        if row.get("label_valid") == "1": duration += dt; command_seconds[row["command"]] += dt  # 유효 주행만 센다.
    for row in commands:  # 제어 정답과 실제 전송값의 관계를 검사한다.
        values = np.array([float(row[k]) for k in ("teacher_steer", "teacher_throttle", "teacher_brake", "noise_steer", "applied_steer", "applied_throttle", "applied_brake")])  # 숫자다.
        if not np.isfinite(values).all() or abs(values[0]) > 1.00001 or abs(values[4]) > 1.00001 or np.any(values[[1,2,5,6]] < 0) or np.any(values[[1,2,5,6]] > 1): errors.append("invalid control range"); break  # 범위를 검사한다.
        if abs(values[0] + values[3] - values[4]) > 1e-6: errors.append("Teacher/noise/applied mismatch"); break  # 학습 정답이 뒤섞이지 않아야 한다.
        if values[5] > 0 and values[6] > 0: errors.append("simultaneous pedals"); break  # 동시 페달을 거부한다.
    if cte.size and cte.max() > 0.8: errors.append("tracking error exceeded limit")  # 추종 품질 기준이다.
    if heading.size and heading.max() > 25: errors.append("heading error exceeded limit")  # 헤딩 품질 기준이다.
    if speed.size and speed.max() > cfg["collection"]["speed_cap_kph"] + 1: errors.append("speed exceeded limit")  # 속도 제어 이상이다.
    requested = images if decode_all else images[::max(1, len(images) // 60)]  # 전체 또는 표본을 검사한다.
    expected = (cfg["camera_common"]["height"], cfg["camera_common"]["width"], 3)  # 저장한 해상도다.
    def decode(row):  # JPEG가 원본 해상도로 읽히는지 확인한다.
        image = cv2.imread(str(session / row["image_path"]))  # 원본 파일을 해독한다.
        return image is not None and image.shape == expected  # 손상·해상도 오류를 검사한다.
    cv2.setNumThreads(1)  # 디코더 내부의 과도한 병렬 실행을 막는다.
    with ThreadPoolExecutor(max_workers=4) as pool:  # JPEG 검사를 병렬화한다.
        decode_results = list(pool.map(decode, requested))  # 모든 결과를 확인한다.
    bad_images = sum(not good for good in decode_results)  # 잘못된 이미지 수다.
    if bad_images: errors.append(f"{bad_images} missing/undecodable/wrong-size images")  # 파일 손실을 검출한다.
    plan = json.loads((BASE / cfg["paths"]["route_plan"]).read_text())  # 시험 구역 정의다.
    protected = {name: np.array(value["guard_reference_xyz"]) for name, value in plan["test_reservations"].items()}  # 기존 시험 구역이다.
    for name, info in plan["routes"].items():
        if info["split"] == "test" or (info["split"] == "validation" and metadata["split"] == "train"):
            protected[name] = np.loadtxt(BASE / info["route_csv"], delimiter=",", skiprows=1)[:, 1:4]
    actual = np.array([[float(row["x_m"]), float(row["y_m"])] for row in vehicle])  # 실제 수집 위치다.
    separations = {name: float(cKDTree(xyz[:,:2]).query(actual)[0].min()) for name,xyz in protected.items()} if len(actual) else {}  # 실제 최소 거리다.
    if any(value < 20 for value in separations.values()): errors.append("actual capture entered a protected spatial guard")  # 후보 경로만 보고 통과시키지 않는다.
    episodes = []  # 외란과 복귀 사건의 통계다.
    for number in sorted({int(row.get("noise_episode", 0)) for row in commands} - {0}):  # 외란 번호별로 묶는다.
        rows = [row for row in commands if int(row.get("noise_episode",0)) == number and row.get("recovery_active") == "1"]  # 복귀 관측 창이다.
        injected = [row for row in rows if abs(float(row["noise_steer"])) > 1e-4]  # 실제로 외란을 더한 표본이다.
        if injected: episodes.append({"episode":number,"injected_samples":len(injected),"max_abs_cte_m":max(abs(float(row["cte_m"])) for row in rows),"last_abs_cte_m":abs(float(rows[-1]["cte_m"]))})  # 관측한 복귀 상태다.
    if metadata.get("noise_enabled") and not episodes: errors.append("noise collection contains no actual perturbation")  # 외란 없이 복귀 수집으로 표시하지 않는다.
    report = {  # 원본 수집 품질을 기록한다.
        "session_id": session.name, "route_id": metadata["route_id"], "split": metadata["split"],
        "data_checks_passed": not errors, "errors": errors, "streams": streams,
        "valid_driving_s": duration, "command_seconds": dict(command_seconds),
        "teacher_rows": len(commands), "valid_teacher_rows": len(valid),
        "max_abs_cte_m": float(cte.max()) if cte.size else None,
        "p95_abs_cte_m": float(np.percentile(cte,95)) if cte.size else None,
        "max_abs_heading_deg": float(heading.max()) if heading.size else None,
        "max_speed_kph": float(speed.max()) if speed.size else None,
        "decoded_images": len(requested), "bad_images": bad_images, "protected_min_distances_m": separations,
        "noise_enabled": metadata.get("noise_enabled",False), "recovery_episodes": episodes,
        "terminal_stop_excluded_from_labels": True, "side_camera_recovery_labels": "not_generated_at_collection_stage",
        "collision_ground_truth": "not_instrumented", "training_ready": False,
    }
    (session / "quality_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")  # 원본 품질 검사 결과다.
    metadata["collection_accepted"] = report["data_checks_passed"]  # 수집 품질 통과 여부다.
    metadata["quality_report"] = "quality_report.json"  # 검사 보고서의 상대경로다.
    (session / "session.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")  # 메타데이터도 갱신한다.
    return report  # 배치 수집기에서 통과 여부를 판단한다.


def main():  # 터미널에서 단일 세션을 검사한다.
    parser = argparse.ArgumentParser()  # 검사 옵션이다.
    parser.add_argument("session", type=Path)  # 원본 폴더를 지정한다.
    parser.add_argument("--sample-decode", action="store_true")  # 빠른 재점검 때만 일부 영상을 해독한다.
    args = parser.parse_args()  # 옵션을 읽는다.
    report = inspect(args.session, decode_all=not args.sample_decode)  # 검사를 수행한다.
    print(json.dumps(report,ensure_ascii=False,indent=2))  # 결과를 출력한다.
    return 0 if report["data_checks_passed"] else 1  # 실패를 종료 코드로도 알린다.


if __name__ == "__main__":  # 직접 실행할 때만 검사한다.
    raise SystemExit(main())  # CLI 진입점이다.
