# 상암 후속 시험 · 명령별 조향 손실 비교

모든 명령에 조향 분포를 적용한 첫 실험에서는 우회전 출구 정렬이 좋아진 코스가 있었고, 직진과 연속 주행에서는 MSE가 더 나았다. 후속 실험은 회전에만 순서형 조향 손실을 사용했다. [첫 상암 실험](sangam_study.md) · [두 코스의 전체 영상과 수치](results/README.md#그래서-어디까지-갔나)

## 추가 본시험

새 두 코스를 정한 뒤 가까운 수집 경로 여섯 곳의 12,869장을 제외했다. MSE와 명령별 모델을 나머지 81,208장으로 각각 다시 학습했다. 평가 경로와 학습 프레임의 최단 거리는 51.5 m였다.

직진과 도로 추종 분기는 분포의 평균 조향에 MSE를 적용했다. 좌·우회전 분기는 순서형 조향 손실을 적용했다. 주행에서는 조향값 순서로 이어진 세 구간씩 훑어 확률 합이 가장 큰 묶음을 고르고, 그 안의 예측값을 확률로 가중 평균했다. 두 모델의 체크포인트는 같은 검증 기준으로 골랐다. 모델마다 두 코스를 두 번씩 달린 뒤 같은 가중치로 한 번씩 재확인했다. [최종 조향값을 고르는 숫자 예시](results/README.md#회전할-때-최종-조향값을-고르는-예)

주 지표는 절대 횡오차가 처음 2.2 m 이상으로 계측된 프레임의 기준 경로상 거리다. 기존 1·2회차에는 기존 근접 판정값 2.2 m를 적용해 이 거리를 사후 계산했고, 이후 같은 정의로 3회차를 평가했다. 좌·우회전 출구 정렬, 목적지 도달, 경로 근처의 가장 먼 위치도 함께 기록했다.

| 모델 | 첫 횡오차 2.2 m까지 평균 거리 | 좌·우회전 정렬 | 직진 정렬 | 목적지 도달 |
|---|---:|---:|---:|---:|
| K-City CIL 그대로 적용 | 17.1 m | 0/6 | 0/6 | 0/6 |
| 상암 MSE CIL | 43.2 m | 0/6 | 0/6 | 0/6 |
| 상암 명령별 조향 손실 CIL | **91.8 m** | 0/6 | 0/6 | 0/6 |

초기 출발 자세와 16 km/h 속도 상한, 횡오차 15 m 중단 기준을 맞췄다. 조향 보조 없이 주행했으며 정체·후진·시간 초과 종료 규칙도 공통이다. 목적지 판정에는 경로 마지막 1 m 구간, 종점 반경 2 m, 절대 횡오차 1.5 m 미만을 함께 요구했다. 출구 정렬은 출구 후 6 m 이내에서 절대 횡오차 1.5 m와 경로 방향 오차 25° 기준을 만족하는지로 판정했다. 신호 준수와 충돌률은 별도 평가로 남겼다. [회차별 CSV](results/metrics/sangam_repeated_trials.csv) · [실험 설정과 집계](results/metrics/sangam_repeated_evaluation.json)

<details>
<summary>경로 근처 진행률과 방향 오차</summary>

경로 근처 진행률은 2.2 m 이내에서 기록한 최장 위치를 코스 길이로 나눈 값이다. 각 셀은 거리 조건만 적용한 3회 평균과, 방향 오차 25° 이내도 요구한 3회 평균이다.

| 코스 | K-City CIL | 상암 MSE CIL | 상암 제안 CIL |
|---|---:|---:|---:|
| 우회전 → 직진 | 57.1% / 57.1% | 63.1% / 40.2% | 28.9% / 28.9% |
| 직진 → 좌회전 | 3.5% / 3.4% | 8.5% / 8.5% | 75.6% / 74.5% |

우회전 코스에서 기본 CIL의 2·3회차는 약 59 m에 처음 이탈한 뒤 약 297 m에서 다시 경로 근처에 들어갔다. 그때 방향 오차는 약 42–44°였다. 직진 → 좌회전 코스에서 제안 모델은 경로 근처의 최장 위치에 도달했을 때 방향 오차가 약 53°였고, 이후 세 번 모두 경로 반대 방향으로 움직였다. [프레임별 분석 기록](results/metrics/sangam_repeated_telemetry.csv.gz)

</details>

## 재현

`train.py --model baseline`과 `train.py --model turn_ordinal`로 두 정책을 학습했다. 120장 명령 균형 배치, Adam 학습률 0.0002, seed 7로 모델별 14 epoch를 학습했다(8,120회 업데이트). 기본 CIL 구현은 논문의 CNN과 명령별 분기, 시범 제어값 회귀와 조향 교란을 따른다. 논문은 CARLA에서 사람이 수집한 시범 2시간을 사용했다. 이번에는 MORAI의 Pure Pursuit 교사가 시범을 수집했고 모델 두 개에 같은 81,208장을 썼다. [CIL 논문](https://arxiv.org/abs/1710.02410) · [학습 분할과 체크포인트](results/metrics/sangam_command_loss_results.json)

두 정책의 추론 입력은 200 × 88 전방 영상, 현재 속도, 네 가지 범주형 진행 명령이다. 지도는 시범 수집, 출발 배치, 명령 안내, 평가에 사용한다. 원본 학습 프레임과 가중치는 로컬 실험 기록에 있다. [손실과 출력 코드](turn_ordinal_model.py) · [공통 MORAI 설정](routes/experiments/sangam_distribution/config.json) · [두 평가 경로](routes/experiments/sangam_distribution/command_loss_manifest.json)

```bash
bash run.sh cil drive.py \
    --config routes/experiments/sangam_distribution/config.json \
    --manifest routes/experiments/sangam_distribution/command_loss_manifest.json \
    --course command_loss_wide_2 --checkpoint checkpoints/sangam_turn_ordinal_command_loss/best.pt \
    --speed-cap-kph 16 --max-cte 15 --goal-max-cte 1.5 --goal-radius-m 2 \
    --steering-local-mode --control --reset --record-screen
```

## 남은 문제

첫 횡오차 2.2 m까지의 개선은 이번 두 경로와 한 학습 시드에서 관찰했다. 직진 → 좌회전에서는 이 지표가 좌회전 교차로 전에 집계된다. 회전 성공에 대한 판단은 출구 정렬과 목적지 도달로 따로 확인했다.

출력 표현, 학습 손실, 주행 때의 후보 선택을 함께 바꿨다. 후속 실험에서는 세 요소의 영향을 나눠 비교하고, 추가 경로와 학습 시드에서 첫 이탈 거리와 출구 정렬을 함께 확인할 계획이다.

진행 명령을 언제 전달할지 불명확한 갈림길에서는 이전 분포 모델이 직진 차로를 유지했다. 이 구간에서는 분기 선택에 필요한 명령 전달 시점을 따로 검증할 필요가 있다. [초기 갈림길 비교 영상](results/videos/sangam_fork_failure.mp4) · [실제 궤적](results/images/sangam_fork_trajectory.png)
