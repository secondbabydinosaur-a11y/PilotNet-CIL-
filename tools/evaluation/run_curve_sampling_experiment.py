"""곡선 프레임 선택 기준만 바꿔 학습하고, 헤딩 0도 곡선/직선 딱 2회 시험한다."""
import csv
from pathlib import Path
import shutil
import subprocess

import numpy as np

from data_utils import BASE, read_csv, read_json, write_json
from dataset import PilotNetDataset
from learning.assessment import telemetry_metrics
from learning.common import file_hash
from learning.data import sampling_curvature
from tools.collection.collect_unseen import verify_frozen
from tools.evaluation.run_matrix import park


def main():
    verify_frozen()
    run = BASE / "runs/pilotnet_three_center_weight_v2"
    output = BASE / "github_artifacts/curve_sampling_v2"
    output.mkdir(parents=True, exist_ok=True)
    data = PilotNetDataset("train", BASE / "dataset/paired_three", allow_side=True)
    curves = np.array([abs(sampling_curvature(r, "center")) >= .003 for r in data.rows])
    c, n = int(curves.sum()), len(data)
    old_weights = np.array([2. if abs(float(r["y_curvature_1pm"])) >= .003 else 1. for r in data.rows])
    audit = dict(source="original train/val only; no unseen teacher used",
        train_samples=n, curve_proxy_samples=c, straight_proxy_samples=n-c,
        curve_basis="uncorrected center teacher curvature, not side recovery targets",
        threshold_abs_1pm=.003, curve_weight=2., straight_weight=1.,
        raw_curve_proxy_fraction=c/n, expected_sampled_curve_proxy_fraction=2*c/(n+c),
        v1_expected_sampled_center_curve_proxy_fraction=float(old_weights[curves].sum()/old_weights.sum()),
        paper="https://arxiv.org/pdf/1604.07316", paper_section="5.1 Data Selection",
        paper_specifies_numeric_ratio=False, weight_2_is_our_choice=True,
        caveat="center steering includes teacher corrections; proxy rather than independently annotated road geometry",
        tests=[dict(course="U2", heading_deg=0, offset_m=0), dict(course="U1", heading_deg=0, offset_m=0)],
        interpretation="post-baseline exploratory comparison on the same already-tested held-out roads")
    write_json(output / "sampling.json", audit)
    if not (run / "best.pt").exists():
        subprocess.run(["bash", str(BASE / "run.sh"), "train.py", "--run-dir", str(run),
            "--camera-mode", "three", "--curve-weight", "2", "--curve-basis", "center",
            "--epochs", "30", "--cache", "--workers", "4"], cwd=str(BASE), check=True)
    settings = read_json(run / "settings.json")["settings"]
    if read_json(run / "summary.json")["status"] != "trained":
        raise RuntimeError("training did not finish; partial checkpoints are not tested")
    if settings["camera_mode"] != "three" or settings["curve_basis"] != "center" or settings["curve_weight"] != 2:
        raise RuntimeError("existing experiment settings differ")
    checkpoint_hash = file_hash(run / "best.pt")
    write_json(output / "frozen_before_two_tests.json", dict(checkpoint_sha256=checkpoint_hash,
        only_two_closed_loop_trials=True, retrain_after_test=False))
    for name in ("best.pt", "settings.json", "summary.json", "history.csv", "learning_curve.png"):
        shutil.copy2(str(run / name), str(output / name))
    results = []
    try:
        # 사용자가 지정한 두 조건만 실행한다. 추가 횡오프셋/헤딩/교차로/재학습은 없다.
        for course in ("U2", "U1"):
            trial = BASE / "runs/curve_sampling_v2_trials" / course
            if not (trial / "summary.json").exists():
                trial.parent.mkdir(exist_ok=True)
                with (trial.parent / (course + ".log")).open("x") as log:
                    subprocess.run(["bash", str(BASE / "run.sh"), "drive.py", "--checkpoint", str(run / "best.pt"),
                        "--course", course, "--control", "--reset", "--initial-offset", "0",
                        "--initial-heading-deg", "0", "--record-screen", "--output", str(trial)],
                        cwd=str(BASE), stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read_json(trial / "summary.json")
            if result["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("two-test checkpoint differs")
            if result["status"] in ("failed", "not_started", "interrupted"):
                raise RuntimeError("two-test trial incomplete; no silent retry")
            if file_hash(run / "best.pt") != checkpoint_hash:
                raise RuntimeError("experiment model changed during test")
            for name in ("summary.json", "settings.json", "initial_state.json", "telemetry.csv", "morai_screen.mp4", "drive.mp4"):
                destination = output / course / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(str(trial / name), str(destination))
            result.update(telemetry_metrics(read_csv(trial / "telemetry.csv"), result.get("terminal_cte_m")))
            results.append(result)
            write_json(output / "results.json", results)
            print("TWO TESTS", course, result["status"], result["completion_fraction"], flush=True)
    finally:
        park()
        verify_frozen()
    rows = []
    for result in results:
        course = result["course"]
        old = read_json(BASE / ("runs/closed_loop/three_" + course + "_nominal_unseen_screen/summary.json"))
        for version, r in (("v1_target_weight", old), ("v2_center_proxy_weight", result)):
            rows.append(dict(version=version, course=course, status=r["status"],
                completion_fraction=r["completion_fraction"], cte_rmse_m=r["cte_rmse_m"],
                max_abs_cte_m=r["max_abs_cte_m"], checkpoint_sha256=r["checkpoint_sha256"]))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table = "\n".join("| {} | {} | {} | {:.1%} | {:.3f} |".format(
        r["version"], r["course"], r["status"], r["completion_fraction"], r["cte_rmse_m"]) for r in rows)
    (output / "README.md").write_text("""# 곡선 프레임 선택 기준 추가 비교

원 논문 5.1절은 곡선 프레임 비중을 높인다고 설명하며 숫자 비율은 제시하지 않는다.
기존 v1도 곡선 정답에 2배 sampling weight를 사용했다. 다만 좌우 복귀 보정도 곡선으로 분류했다.
v2는 보정 전 가운데 teacher 곡률을 기준으로 가중치를 적용한다. **2배는 우리의 선택이며 논문 권장 수치가 아니다.**
입력·CNN·원래 train/val·seed 42·30 epoch·optimizer는 유지했다. 기존 모델은 변경하지 않았다.
가운데 teacher 조향은 도로 곡률의 근사이며 실제 도로 형상 annotation과 동일하지 않다.

원래 train의 곡선 근사 비중 {:.1%} → 가중 추출 기대 비중 {:.1%}. 자세한 수치는 [sampling.json](sampling.json).
미학습 구간 teacher는 train/val에 쓰지 않았다. 새 학습 모델 선택도 validation MSE만 사용했다.

closed-loop는 **3개 카메라 학습 모델, 가운데 카메라 추론, 헤딩 0°, 횡위치 0m, 19km/h**로 곡선 U2와 직선 U1을 각 1회만 실행했다.
이미 baseline 시험을 본 뒤 수행한 탐색 비교이므로 처음 보는 도로에 대한 새로운 blind test로 부르지 않는다.

| 버전 | 코스 | 결과 | 진행률 | CTE RMSE (m) |
|---|---|---|---|---|
{}

- [곡선 실제 MORAI 영상](U2/morai_screen.mp4), [곡선 전방 영상](U2/drive.mp4)
- [직선 실제 MORAI 영상](U1/morai_screen.mp4), [직선 전방 영상](U1/drive.mp4)
- [학습 기록](history.csv), [모델](best.pt), [비교 CSV](comparison.csv)

중단 시험의 RMSE는 중단 전 구간만 포함한다. 진행률·완주 여부와 함께 해석한다.
출처: [End to End Learning for Self-Driving Cars, §5.1](https://arxiv.org/pdf/1604.07316).
""".format(c/n, 2*c/(n+c), table), encoding="utf-8")


if __name__ == "__main__":
    main()
