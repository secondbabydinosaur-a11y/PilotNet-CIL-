# 상암 후속 시험 · 명령별 조향 손실 비교

모든 명령에 조향 분포를 적용한 첫 실험에서는 우회전 출구 정렬이 좋아진 코스가 있었고, 직진과 연속 주행에서는 MSE가 더 나았다. 후속 실험은 회전에만 순서형 조향 손실을 사용했다. [첫 상암 실험](sangam_study.md) · [두 코스의 전체 영상과 수치](results/README.md#그래서-어디까지-갔나)

## 추가 본시험

새 두 코스를 정한 뒤 가까운 수집 경로 여섯 곳의 12,869장을 제외했다. MSE와 명령별 모델을 나머지 81,208장으로 각각 다시 학습했다. 평가 경로와 학습 프레임의 최단 거리는 51.5 m였다.

직진과 도로 추종 분기는 분포의 평균 조향에 MSE를 적용했다. 좌·우회전 분기는 순서형 조향 손실을 적용했다. 주행에서는 조향값 순서로 이어진 세 구간씩 훑어 확률 합이 가장 큰 묶음을 고르고, 그 안의 예측값을 확률로 가중 평균했다. 두 모델의 체크포인트는 같은 검증 기준으로 골랐다. 횡오차 5 m 기준으로 모델마다 두 코스를 세 번씩 다시 달렸다. [최종 조향값을 고르는 숫자 예시](results/README.md#회전할-때-최종-조향값을-고르는-예)

초기 출발 자세, 16 km/h 속도 상한, 조향 보조 0회 조건을 사용했다. 절대 횡오차가 처음 5 m 이상인 제어 프레임에서 가속 0과 최대 제동을 명령하고 종료했다. 공통 정체·후진·시간 초과 규칙도 적용했다. 목적지 판정은 마지막 1 m 구간, 종점 반경 2 m, 절대 횡오차 1.5 m 미만을 함께 요구했다. 출구 정렬은 출구 후 6 m 이내에서 절대 횡오차 1.5 m와 방향 오차 25° 기준으로 판정했다.

아래 거리는 현재 영상과 같은 5 m 재주행에서 계산했다. 2.2 m 지표는 첫 계측 위치, 종료 위치는 마지막 계측 위치의 코스별 평균이다. 두 거리 모두 기준 경로에 투영했다. 교차로 출구 정렬은 세 모델 모두 0/12회였다.

| 모델 | 첫 횡오차 2.2 m까지 평균 거리 | 우회전 → 직진 종료 위치 | 직진 → 좌회전 종료 위치 | 목적지 도달 |
|---|---:|---:|---:|---:|
| 기존 K-City CIL | 15.3 m | 49.2 m | 20.0 m | 0/6 |
| 상암 MSE CIL | 42.8 m | 125.7 m | 166.3 m | 0/6 |
| 상암 명령별 조향 손실 CIL | 90.4 m | 115.8 m | 106.9 m | 0/6 |

[5 m 실행별 CSV](results/metrics/sangam_cte5_trials.csv) · [5 m 프레임 기록](results/metrics/sangam_cte5_telemetry.csv.gz) · [이전 15 m 조건 기록](results/metrics/sangam_repeated_evaluation.json)

## 재현

`train.py --model baseline`과 `train.py --model turn_ordinal`로 두 정책을 학습했다. 120장 명령 균형 배치, Adam 학습률 0.0002, seed 7로 모델별 14 epoch를 학습했다(8,120회 업데이트). 기본 CIL 구현은 논문의 CNN과 명령별 분기, 시범 제어값 회귀와 조향 교란을 따른다. 논문은 CARLA에서 사람이 수집한 시범 2시간을 사용했다. 이번에는 MORAI의 Pure Pursuit 교사가 시범을 수집했고 모델 두 개에 같은 81,208장을 썼다. [CIL 논문](https://arxiv.org/abs/1710.02410) · [학습 분할과 체크포인트](results/metrics/sangam_command_loss_results.json)

두 정책의 추론 입력은 200 × 88 전방 영상, 현재 속도, 네 가지 범주형 진행 명령이다. 지도는 시범 수집, 출발 배치, 명령 안내, 평가에 사용한다. 원본 학습 프레임과 가중치는 로컬 실험 기록에 있다. [손실과 출력 코드](turn_ordinal_model.py) · [공통 MORAI 설정](routes/experiments/sangam_distribution/config.json) · [두 평가 경로](routes/experiments/sangam_distribution/command_loss_manifest.json)

```bash
bash run.sh cil drive.py \
    --config routes/experiments/sangam_distribution/config.json \
    --manifest routes/experiments/sangam_distribution/command_loss_manifest.json \
    --course command_loss_wide_2 --checkpoint checkpoints/sangam_turn_ordinal_command_loss/best.pt \
    --speed-cap-kph 16 --max-cte 5 --goal-max-cte 1.5 --goal-radius-m 2 \
    --steering-local-mode --control --reset --record-screen
```

## 남은 문제

직진 → 좌회전 코스에서 제안 모델의 첫 2.2 m 지점은 세 회차 모두 첫 직진 교차로 구간이었다. 5 m 중단 기준으로 재주행하자 최종 좌회전 교차로에 도달하기 전에 멈췄다. 이번에 비교한 범위는 두 경로와 한 학습 시드다.

출력 표현, 학습 손실, 주행 때의 후보 선택을 함께 바꿨다. 후속 실험에서는 세 요소의 영향을 나눠 비교하고, 추가 경로와 학습 시드에서 첫 이탈 거리와 출구 정렬을 함께 확인할 계획이다.

[초기 갈림길 5 m 비교](../README.md#한계) · [이전 2.2 m 시험의 궤적](results/images/sangam_fork_trajectory.png)
