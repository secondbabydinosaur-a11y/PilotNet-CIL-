"""원본 품질 검사를 통과한 세션만 집계한다. 모델 성능 지표와 구분한다."""
import csv  # 세션별 요약 표를 저장한다.
import json  # 진행 기록과 원본 검사 결과를 읽는다.
from collections import Counter  # 용도·명령별 시간을 합산한다.
from pathlib import Path  # CIL 폴더를 기준으로 파일을 찾는다.

BASE = Path(__file__).resolve().parents[1]  # CIL 작업 폴더다.


def main():  # 수집 상태를 요약한다.
    progress = json.loads((BASE / "data/collection_completion.json").read_text())  # 통과 목록이다.
    rows, commands, seconds, images, episodes = [], {}, Counter(), Counter(), 0  # 집계할 값들이다.
    all_decoded = True  # 각 보고서가 전체 영상 검사인지 확인한다.
    for item in progress["accepted"]:  # 실패·점검 세션은 포함하지 않는다.
        report = json.loads((BASE / item["report"]).read_text())  # 실제 원본 검사 결과다.
        if not report["data_checks_passed"]:  # 검사 통과 상태를 다시 확인한다.
            raise ValueError("accepted list contains a failed session: " + item["session"])  # 잘못된 합산을 막는다.
        split = item["split"]  # 학습과 Validation을 구분한다.
        seconds[split] += report["valid_driving_s"]  # 임의 종료 제동을 제외한 시간이다.
        commands.setdefault(split, Counter()).update(report["command_seconds"])  # 명령별 시간이다.
        episode_count = len(report["recovery_episodes"])  # 실제 교란이 기록된 사건 수다.
        episodes += episode_count  # 전체 교란 사건을 합산한다.
        image_counts = {camera: report["streams"][camera]["samples"] for camera in ("left", "center", "right")}  # 원본 영상 수다.
        all_decoded = all_decoded and report["decoded_images"] == sum(image_counts.values())  # 표본 검사와 전체 검사를 구분한다.
        images.update(image_counts)  # 전체 원본 수를 합산한다.
        rows.append({  # 사람이 읽을 세션별 표다.
            "route": item["route_id"], "split": split,
            "condition": "perturbation_and_recovery" if item["noise_enabled"] else "normal",
            "session": item["session"], "valid_driving_s": round(report["valid_driving_s"], 3),
            "left_images": image_counts["left"], "center_images": image_counts["center"], "right_images": image_counts["right"],
            "max_abs_cte_m": round(report["max_abs_cte_m"], 4), "p95_abs_cte_m": round(report["p95_abs_cte_m"], 4),
            "max_speed_kph": round(report["max_speed_kph"], 3), "noise_episodes": episode_count,
            "decoded_images": report["decoded_images"], "bad_images": report["bad_images"],
        })
    summary = {  # 모델 평가 결과로 오해하지 않도록 범위를 명시한다.
        "status": progress["status"], "measurement": "automatic_teacher_raw_collection_quality",
        "accepted_sessions": len(rows), "valid_driving_s_by_split": dict(seconds),
        "command_seconds_by_split": {key: dict(value) for key, value in commands.items()},
        "raw_images_by_camera": dict(images), "actual_noise_episodes": episodes,
        "rejected_and_calibration_sessions_included": False, "excluded_route": "T5",
        "reserved_test_regions": ["U1", "U2", "J_left_straight_right", "P2Q"],
        "human_driving": False, "all_accepted_images_decoded": all_decoded,
        "side_camera_recovery_labels_generated": False, "prepared_training_csv_generated": False,
        "cil_model_trained": False, "collision_ground_truth": "not_instrumented",
        "sessions": rows,
    }
    destination = BASE / "data"  # 작은 집계 결과만 공개용 파일로 만든다.
    (destination / "collection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")  # 상세 집계다.
    if rows:  # 통과 세션이 있을 때는 CSV도 만든다.
        with (destination / "collection_summary.csv").open("w", newline="", encoding="utf-8") as stream:  # 표를 저장할 위치다.
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))  # 열 순서를 고정한다.
            writer.writeheader(); writer.writerows(rows)  # 제목과 세션별 결과를 저장한다.
    print(json.dumps({key:value for key,value in summary.items() if key != "sessions"}, ensure_ascii=False, indent=2))  # 합산 결과다.


if __name__ == "__main__":  # 직접 실행했을 때만 집계한다.
    main()  # 명령행 진입점이다.
