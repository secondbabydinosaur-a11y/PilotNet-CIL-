"""지도 기반 명령 구간·같은 접근로의 좌/직/우 시범 경로를 만들고 공간 분리를 검사한다."""
import csv  # 경로 CSV를 저장한다.
import hashlib  # 지도·경로 원본을 식별한다.
import json  # 계획 파일을 읽고 갱신한다.
import math  # 경로의 실제 진입·진출 방향을 비교한다.
from pathlib import Path  # CIL 기준 경로를 계산한다.
import numpy as np  # 좌표와 누적 거리를 계산한다.
from scipy.spatial import cKDTree  # 시험·Validation 구역과의 거리를 검사한다.

BASE = Path(__file__).resolve().parents[1]  # cil 폴더다.


def clip(points, end_m):  # 다음 교차로 전에 시범 경로를 종료할 수 있게 자른다.
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]  # 거리다.
    q = np.unique(np.r_[s[s < end_m], min(end_m, s[-1])])  # 끝점을 포함한다.
    return np.column_stack([np.interp(q, s, points[:, i]) for i in range(3)])  # XYZ를 보간한다.


def main():  # 수집 경로의 명령 구간을 생성한다.
    path = BASE / "routes/plan.json"  # 기존 후보 계획이다.
    plan = json.loads(path.read_text())  # 기존 시험 구역과 경로를 유지한다.
    map_dir = Path(plan["source"])  # 로컬 MGeo 원본 위치다.
    links = {link["idx"]: link for link in json.loads((map_dir / "link_set.json").read_text()) if not link["lazy_init"]}  # 실제 링크다.
    variants = {  # 같은 북향 접근로에서 서로 다른 출구를 선택한다.
        "CENTRAL_LEFT": (["A2256W000315", "A2256W000092", "A2256W001150"], 40.0),  # 좌회전이다.
        "CENTRAL_STRAIGHT": (["A2256W000315", "A2256W000093", "A2256W000322"], None),  # 직진이다.
        "CENTRAL_RIGHT": (["A2256W000315", "A2256W000318", "A2256W000308"], 19.0),  # 우회전이다.
    }
    for name, (keys, final_length) in variants.items():  # 대회 시험 교차로 J는 사용하지 않는다.
        segments = [np.array(links[key]["points"]) for key in keys]  # 지도 링크를 연결한다.
        if final_length is not None:  # 다음 교차로를 포함하지 않게 한다.
            segments[-1] = clip(segments[-1], final_length)  # 마지막 직선만 자른다.
        xyz = np.concatenate(segments)  # 전체 점열이다.
        xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) > 1e-6]]  # 중복 끝점을 제거한다.
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]  # 새 경로의 거리다.
        relative = "routes/train/" + name + ".csv"  # 다른 학습 경로와 함께 둔다.
        with (BASE / relative).open("w", newline="") as stream:  # 좌표를 저장한다.
            writer = csv.writer(stream); writer.writerow(["s_m", "x_m", "y_m", "z_m"])  # 열 이름이다.
            for station, point in zip(s, xyz):  # 경로점마다
                writer.writerow([f"{station:.6f}"] + [f"{v:.6f}" for v in point])  # 좌표를 기록한다.
        plan["routes"][name] = {  # 경로 설명과 출발 위치다.
            "name_ko": "중앙 교차로 동일 접근로 " + name.split("_")[-1], "split": "train",
            "length_m": float(s[-1]), "start_xyz_m": xyz[0].tolist(), "end_xyz_m": xyz[-1].tolist(),
            "initial_yaw_deg_map": math.degrees(math.atan2(*(xyz[1, :2] - xyz[0, :2])[::-1])),
            "source_links": keys, "route_csv": relative, "last_link_clip_m": final_length,
            "same_approach_group": "central_northbound_node_757", "junction_command": name.split("_")[-1].lower(),
        }
    plan["routes"]["T5"]["split"] = "calibration"  # 차선 없는 도로는 수신·추종 점검용으로만 보존한다.
    plan["routes"]["T5"]["collection_note"] = "Unmarked broad road: excluded from train and validation at user feedback."
    protected = {name: np.array(value["guard_reference_xyz"]) for name, value in plan["test_reservations"].items()}  # 기존 시험이다.
    for name in ("V1", "V2", "P2Q"):  # 독립 검증·새 시험 경로도 보호한다.
        protected[name] = np.loadtxt(BASE / plan["routes"][name]["route_csv"], delimiter=",", skiprows=1)[:, 1:4]  # 좌표다.
    checks = []  # 재계산한 공간 분리 증거다.
    for name, info in plan["routes"].items():  # 원래 경로와 추가 시범을 검사한다.
        route = np.loadtxt(BASE / info["route_csv"], delimiter=",", skiprows=1)  # 실제 실행할 CSV다.
        targets = protected if info["split"] == "train" else {k: v for k, v in protected.items() if k not in ("V1", "V2", name)}  # 용도별 분리다.
        if info["split"] == "calibration":  # 점검 경로는 학습에 넣지 않는다.
            targets = {}  # 수집 대상에서 제외한다.
        for target, xyz in targets.items():  # 실제 거리 기준으로 검사한다.
            distance = float(cKDTree(xyz[:, :2]).query(route[:, 1:3])[0].min())  # 최소 표본 거리다.
            if distance <= 20.5:  # 기존 20m 보호 여유를 지킨다.
                raise ValueError((name, target, distance))  # 겹치는 경로를 수집하지 않는다.
            checks.append({"route": name, "protected": target, "nearest_sample_m": distance})  # 증거다.
        navigation = []  # 교차로의 물리 링크 범위를 명령 구간으로 기록한다.
        lower = 0  # 같은 좌표가 다시 나와도 경로 순서대로 찾는다.
        keys = info.get("source_links") or []  # V2처럼 경로 잘라내기만 한 경우에는 비어 있다.
        for i, key in enumerate(keys):  # 선택한 링크를 순서대로 처리한다.
            link = links[key]  # 원본 지도 링크다.
            p = np.array(link["points"])  # 지도 점열이다.
            start = lower + int(np.argmin(np.linalg.norm(route[lower:, 1:3] - p[0, :2], axis=1)))  # 경로의 진입점이다.
            end = start + int(np.argmin(np.linalg.norm(route[start:, 1:3] - p[-1, :2], axis=1)))  # 경로의 진출점이다.
            lower = start  # 다음 링크의 검색 시작점이다.
            if str(link.get("link_type")) != "1":  # 일반 도로 곡선은 follow_road다.
                continue  # 곡률만으로 좌·우회전 명령을 만들지 않는다.
            s0, s1 = route[start, 0], route[end, 0]  # 교차로 링크의 거리 범위다.
            def point(station):  # 경로에서 특정 거리의 XY를 읽는다.
                return np.array([np.interp(station, route[:, 0], route[:, j]) for j in (1, 2)])  # 보간이다.
            before = point(s0) - point(max(0.0, s0 - 3.0))  # 진입 방향이다.
            after = point(min(route[-1, 0], s1 + 3.0)) - point(s1)  # 진출 방향이다.
            if np.linalg.norm(before) < 0.5:  # 경로가 교차로 내부에서 시작한다면
                before = point(min(s1, s0 + 3.0)) - point(s0)  # 첫 접선을 사용한다.
            if np.linalg.norm(after) < 0.5:  # 경로가 교차로 끝에서 종료한다면
                after = point(s1) - point(max(s0, s1 - 3.0))  # 마지막 접선을 사용한다.
            turn = (math.degrees(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0])) + 180) % 360 - 180  # 방향 변화다.
            signal = link.get("related_signal") or ""  # 지도 값도 증거로 함께 남긴다.
            if signal.startswith("left"): command = "left"  # 명시적 좌회전 링크다.
            elif signal.startswith("right"): command = "right"  # 명시적 우회전 링크다.
            elif turn > 45: command = "left"  # straight 오기와 실제 좌회전 기하를 구분한다.
            elif turn < -45: command = "right"  # 실제 우회전 기하다.
            else: command = "straight"  # 교차로 통과 방향을 유지한다.
            navigation.append({"link_id": key, "entry_s_m": float(s0), "exit_s_m": float(s1), "approach_m": 18.0, "exit_hold_m": 6.0, "command": command, "map_related_signal": signal, "measured_turn_deg": round(turn, 2)})  # 검토 가능한 명령 정의다.
        info["navigation"] = navigation  # 경로와 명령을 함께 보관한다.
        info["route_sha256"] = hashlib.sha256((BASE / info["route_csv"]).read_bytes()).hexdigest()  # 좌표 원본이다.
    plan["status"] = "collection_in_progress_lane_marked_routes"  # 실제 수집 단계로 갱신한다.
    plan["collection_separation_checks"] = checks  # 추가 경로까지 검사한 결과다.
    plan["collection_note"] = "T5 calibration only; collect train/validation roads, preserve U1/U2/J/P2Q. Navigation reviewed using map intersection links and entry/exit geometry."
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")  # 갱신한 계획을 저장한다.
    for name, info in plan["routes"].items():  # 명령 목록을 출력한다.
        print(name, info["split"], round(info["length_m"]), [(n["link_id"][-4:], n["command"]) for n in info["navigation"]])  # 점검용이다.


if __name__ == "__main__":  # 직접 실행한 경우에만 경로를 만든다.
    main()  # 수집 경로 준비를 수행한다.
