"""학습·Validation 경로를 순차 수집하고 원본 품질 검사를 통과한 세션만 목록에 넣는다."""
import argparse  # 재시작 옵션을 읽는다.
import json  # 수집 진행 상태를 저장한다.
import signal  # 중단 시 자식 수집기를 정상 종료한다.
import socket  # 예외 종료에서 정지 명령을 보낼 수 있게 한다.
import subprocess  # 경로별 수집을 독립 프로세스로 실행한다.
import sys  # 현재 Python 실행기를 사용한다.
import time  # 진행 상황을 주기적으로 출력한다.
from pathlib import Path  # CIL 기준 경로를 계산한다.

BASE = Path(__file__).resolve().parents[1]  # cil 폴더다.
sys.path.insert(0, str(BASE))  # CIL 모듈을 가져온다.
from tools.inspect_collection import inspect  # 원본 품질 검사다.
from morai import encode_control, load_config  # 검증한 통신 형식을 사용한다.

TASKS = (  # 목적 없이 같은 코스를 반복하지 않고 정상·명령·복귀를 확보한다.
    ("T1", False, 7, 0.0), ("T2", False, 7, 0.0), ("T3", False, 7, 0.0),
    ("T4", False, 7, 0.0), ("T6", False, 7, 0.0),
    ("CENTRAL_LEFT", False, 7, 0.0), ("CENTRAL_STRAIGHT", False, 7, 0.0), ("CENTRAL_RIGHT", False, 7, 0.0),
    ("V1", False, 7, 0.0), ("V2", False, 7, 0.0),
    ("T4", True, 31, 0.18), ("T6", True, 47, -0.18), ("T2", True, 59, 0.20), ("T1", True, 71, -0.20),
)


def emergency_brake(config):  # 예외 상황에서 제동을 돕는다.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:  # 송신 소켓이다.
        for _ in range(40):  # 약 0.8초 동안 반복한다.
            sock.sendto(encode_control(0.0, 0.0, 1.0), (config["network"]["morai_ip"], config["control"]["host_port"]))  # 제동 명령이다.
            time.sleep(0.02)  # 정상 제어 주기로 보낸다.


def main():  # 배치 수집을 실행한다.
    parser = argparse.ArgumentParser()  # 실행 옵션이다.
    parser.add_argument("--resume", action="store_true")  # 이미 통과한 수집은 재사용한다.
    args = parser.parse_args()  # 옵션을 읽는다.
    cfg = load_config()  # 저장 위치와 통신 설정이다.
    raw = BASE / cfg["paths"]["raw_data"]  # 원본 폴더다.
    logs = BASE / "runs/collection"  # 실행 로그를 모은다.
    logs.mkdir(parents=True, exist_ok=True)  # 필요한 폴더를 만든다.
    completion = BASE / "data/collection_completion.json"  # 진행·완료 상태다.
    accepted = []  # 품질 검사 통과 세션만 넣는다.
    def save(status, current=None, error=""):  # 중간 결과도 계속 보존한다.
        completion.write_text(json.dumps({"status":status,"current":current,"error":error,"requested_tasks":len(TASKS),"accepted":accepted,"excluded_calibration_route":"T5","models_trained":False},ensure_ascii=False,indent=2)+"\n")  # 상태 파일이다.
    try:  # 실패하면 다음 경로로 무작정 넘어가지 않는다.
        for number, (route, noisy, seed, offset) in enumerate(TASKS, 1):  # 목적별 수집을 순서대로 진행한다.
            tag = route + ("_noise" if noisy else "_normal")  # 로그 이름이다.
            save("in_progress", tag)  # 현재 작업을 기록한다.
            session = None  # 이번 조건에 맞는 기존 원본이다.
            if args.resume:  # 정상적으로 수집한 원본을 재활용한다.
                for path in sorted(raw.glob("*/session.json"), reverse=True):  # 최근 세션부터 본다.
                    meta = json.loads(path.read_text())  # 실제 수집 조건이다.
                    matches = meta.get("route_id") == route and bool(meta.get("noise_enabled")) == noisy and meta.get("status") == "route_finished" and abs(meta.get("initial_offset_m",0)-offset)<1e-6 and (not noisy or meta.get("seed")==seed)  # 조건 일치다.
                    if matches: session = path.parent; break  # 같은 목적의 성공 원본을 사용한다.
            if session is None:  # 새 수집이 필요하다면
                before = set(raw.glob("*/session.json"))  # 생성 전 세션 목록이다.
                command = [sys.executable, "-u", str(BASE / "collect.py"), "--route", route, "--auto-reset", "--seed", str(seed), "--initial-offset", str(offset)]  # 경로별 실행이다.
                if noisy: command.append("--noise")  # 복귀 수집에만 교란을 켠다.
                print(f"RUN {number}/{len(TASKS)} {tag}", flush=True)  # 현재 진행을 알린다.
                with (logs / (tag + ".log")).open("w") as log:  # 세부 출력은 로그에 보관한다.
                    proc = subprocess.Popen(command, cwd=str(BASE), stdout=log, stderr=subprocess.STDOUT)  # 수집기를 실행한다.
                    last_report = time.monotonic()  # 진행 출력 주기다.
                    try:  # 중단 시 자식 수집기도 정리한다.
                        while proc.poll() is None:  # 현재 경로가 끝날 때까지 기다린다.
                            time.sleep(1.0)  # 짧게 대기해 중단을 처리할 수 있게 한다.
                            if time.monotonic() - last_report > 20:  # 약 20초마다 상태를 표시한다.
                                lines = (logs / (tag + ".log")).read_text().splitlines()  # 최근 출력이다.
                                print("PROGRESS", tag, lines[-1] if lines else "starting", flush=True)  # 진행 상황이다.
                                last_report = time.monotonic()  # 주기를 갱신한다.
                    except BaseException:  # 사용자 중단이나 실행 오류라면
                        proc.send_signal(signal.SIGINT)  # finally 제동을 실행하도록 요청한다.
                        try: proc.wait(timeout=6)  # 정상 종료를 잠시 기다린다.
                        except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=3)  # 멈춘 프로세스를 정리한다.
                        emergency_brake(cfg)  # 정지 명령도 보낸다.
                        raise  # 배치를 중단한다.
                added = set(raw.glob("*/session.json")) - before  # 이번 실행의 원본 폴더다.
                if proc.returncode != 0 or len(added) != 1:  # 실패하거나 원본 식별이 잘못됐다면
                    raise RuntimeError(tag + ": collection failed; see " + str(logs / (tag + ".log")))  # 원인을 확인한다.
                session = next(iter(added)).parent  # 이번 세션이다.
            report_path = session / "quality_report.json"  # 기존 검사가 있는지 확인한다.
            report = json.loads(report_path.read_text()) if report_path.exists() else inspect(session, decode_all=True)  # 모든 JPEG를 검사한다.
            if not report["data_checks_passed"]:  # 품질 실패를 데이터에 포함하지 않는다.
                raise RuntimeError(tag + ": quality checks failed: " + str(report["errors"]))  # 배치를 중단한다.
            accepted.append({"route_id":route,"noise_enabled":noisy,"seed":seed,"initial_offset_m":offset,"session":str(session.relative_to(BASE)),"report":str(report_path.relative_to(BASE)),"split":report["split"],"valid_driving_s":report["valid_driving_s"],"command_seconds":report["command_seconds"],"max_abs_cte_m":report["max_abs_cte_m"]})  # 통과 기록이다.
            print(f"PASS {number}/{len(TASKS)} {tag}: {report['valid_driving_s']:.1f}s, max CTE {report['max_abs_cte_m']:.3f}m", flush=True)  # 결과다.
            save("in_progress", tag)  # 다음 경로 전에도 기록을 남긴다.
        save("complete")  # 요청한 정상·명령·복귀 수집을 마쳤다.
        print("COLLECTION COMPLETE", len(accepted), "accepted sessions", flush=True)  # 최종 수집 상태다.
        return 0  # 성공 종료다.
    except BaseException as exc:  # 실패·중단 상태를 보존한다.
        save("incomplete", error=str(exc))  # 완료했다고 표시하지 않는다.
        print("COLLECTION STOPPED:", str(exc), flush=True)  # 재시작할 위치를 알린다.
        raise  # 원래 오류를 유지한다.


if __name__ == "__main__":  # 직접 실행할 때만 시뮬레이터를 조작한다.
    raise SystemExit(main())  # 배치 진입점이다.
