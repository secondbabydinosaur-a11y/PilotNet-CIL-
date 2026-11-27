# 상암 초기 실험 · CIL의 조향 분포 비교

K-City에서 명령형 CIL을 구현한 뒤, 새 MORAI 상암 맵에서 고정된 K-City 모델과 상암에서 학습한 모델을 달렸다. 상암 안에서도 학습·검증·시험 도로를 위치로 나눴다. [상암 주행 결과 요약](results/README.md#상암에서-어디를-달렸나) · [두 코스 후속 비교](sangam_command_loss.md#추가-본시험) · [논문](https://arxiv.org/abs/1710.02410)

K-City의 정면 `RIGHT` 표본에서 우조향 20° 이상인 표본 비중은 보강 전 12.9%, 보강 후 16.2%였다. 보강 데이터의 학습 위치와 452.2 m 시험 경로의 최단 거리는 26.7 m였다. 같은 맵에서 실패를 보고 샘플 비중을 조정한 경험을 새 장소에서 재검증하는 계기로 삼았다. [K-City 재학습 기록](results/README.md#point-to-point-주행)

## 학습 조건

| 항목 | 이번 구현 |
|---|---|
| 입력과 구조 | 200 × 88 전방 영상, 실측 속도, 네 진행 명령에 따른 분기, 8층 CNN |
| 시범 수집 | MORAI 상암에서 Pure Pursuit 교사, 학습 시 중앙과 좌우 ±30° 카메라, 추론 시 중앙 카메라 |
| 동일 데이터 대조 | 학습 이미지 94,077장, 촬영 위치 42개 주행 경로, 검증 중앙 이미지 1,796장 |
| 시범 시간 | 실제 유효 조향 기록 53.0분, 조향 교란이 활성화된 수집 7.8분 |
| 최적화 | Adam, 초기 학습률 0.0002, 120장 배치에서 명령별 30장, 모델별 8,120회 업데이트 |
| 이미지 증강 | 밝기·대비·톤 변화, 블러와 잡음, 일부 사각 영역 가리기 |

논문은 CARLA Town 1에서 사람이 조종한 2시간의 시범을 수집했고 약 10%에는 조향 교란을 적용했다. 이번에는 학습 도로를 늘리고 시험 도로를 예약한 상암에서 자동 시범을 수집했다. MORAI 제어에 맞춰 조향·가속·제동을 각각 예측하고 보조 속도 예측도 사용했다. 시범 교사는 지도에서 구한 경로와 복귀 표본을 이용한다. 본시험에서 지도는 출발 배치, 범주형 진행 명령 제공, 평가에 사용했다. 주행 정책의 입력은 영상·속도·진행 명령이다. [논문의 네트워크·데이터 조건](https://arxiv.org/pdf/1710.02410)

[![시험 도로와 학습 도로의 분리](results/images/sangam_arterial_split.png)](results/images/sangam_arterial_split.png)

## 가설과 모델 선택

동일한 관측에 여러 조향이 대응할 때, 제곱오차의 최적 예측은 조건부 평균이다. 같은 `RIGHT`에도 도로 영상과 차량 자세에서 조향 강도를 구분할 수 있는지 확인하고 싶었다. 조향을 17개 각도 구간으로 나누고 각 구간의 확률과 구간 내 조향을 학습했다. 정답에서 먼 각도에 높은 확률을 줄수록 더 큰 손실을 적용했다. 주행에서는 확률의 합이 가장 큰 인접 세 구간 안에서 각도를 확률로 가중해 선택했다.

초기 11,228장 검증에서 가우시안 혼합분포·일반 각도 구간·채널 어텐션을 시험했다. 평균 진행률은 MSE 33.8%, 혼합분포 15.0%, 각도 구간 46.2%, 채널 어텐션 10.3%였다. 경로점을 영상에서 예측한 모델도 5.3%에서 멈췄다. 이후 22,480장으로 확장한 검증에서 순서형 조향 분포의 평균 진행률은 44.5%, 같은 데이터의 MSE는 16.0%였다. 초기 수치는 횡오차 2.2 m 조건의 별도 실험이다.

각도 구간을 학습할 때 순서를 반영하면 멀리 잘못 배정한 확률에 더 큰 손실을 줄 수 있다. 정답 0°에 70%를 준 두 예측에서 나머지 30%가 한 칸 옆에 있을 때는 누적확률 오차가 경계 하나에 나타나고, 네 칸 옆에 있을 때는 경계 네 곳에 나타난다. [손실 그림](results/images/sangam_ordinal_loss.png) · [순서형 손실 코드](ordinal_model.py) · [Ranked Probability Score](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

추가 수집 이후에는 MSE·단순 구간·순서형 분포를 같은 데이터로 다시 학습했다. 회전 중 선택한 조향이 크게 뛰는 장면을 관찰해 연속 프레임의 확률을 평활화하는 방식과 인접한 각도 후보를 결합하는 방식도 비교했다. 별도 검증의 목적지 도달·회전 출구 정렬·첫 횡오차 2.2 m까지의 진행률 순으로 분포 모델을 정했다. MSE의 최저 검증 손실 체크포인트와 마지막 체크포인트도 같은 검증 도로에서 비교해 기준형을 골랐다.

| 모델 | 검증 도달 | 좌·우회전 출구 정렬 | 첫 횡오차 2.2 m까지의 평균 진행률 |
|---|---:|---:|---:|
| 상암 MSE CIL | 1/3 | 0 | 10.1% |
| 상암 MSE CIL · 마지막 epoch | 0/3 | 2 | 59.4% |
| 상암 순서형 조향 분포 CIL | 0/3 | 2 | 50.6% |
| 상암 순서형 조향 분포 CIL + 시간 평활화 | 0/3 | 2 | 50.7% |
| 상암 조향 구간 CIL | 0/3 | 1 | 33.0% |
| 상암 순서형 조향 분포 CIL + 인접 후보 결합 | 0/3 | 2 | 51.9% |

주행 검증 경로 한 곳은 학습 위치에 6.0 m까지 가까워지는 구간을 포함했다. 본시험 세 경로는 학습과 검증 이미지 위치에서 각각 30 m 이상 떨어져 있다.

## 본시험

일반 왕복 4차선 이상의 서로 다른 세 경로에서 같은 출발 자세, 16 km/h 속도 상한, 횡오차 15 m 중단 기준으로 각 모델을 두 번씩 주행했다. 정체·후진·제한 시간 기준도 세 모델에 똑같이 적용했다. 정렬 통과는 교차로 출구 직후 6 m 구간에서 횡오차 1.5 m와 방향 오차 25°를 충족한 경우다. 목적지 판정에는 기준 경로 마지막 1 m 구간과 종점 반경 2 m를 함께 사용했다. 신호 준수와 충돌률은 별도 평가로 남겼다.

우회전 → 직진 코스는 초기 짧은 시연에 나온 교차로 한 곳을 포함한다. 일부 구간의 인접 차로에는 자율버스 표시가 있고 기준 경로는 일반 차로를 사용한다. 세 긴 경로는 학습 데이터를 추가 수집하기 전에 고정했고, 모델 선택은 별도 검증 도로에서 진행했다.

| 모델 | 목적지 도달 | 좌·우회전 정렬 | 직진 정렬 | 경로 근처 진행률 | 평균 최대 횡오차 |
|---|---:|---:|---:|---:|---:|
| K-City CIL 그대로 적용 | 0/6 | 0/8 | 0/6 | 8.3% | 10.17 m |
| 상암 MSE CIL | 0/6 | 1/8 | 4/6 | 56.8% | 8.26 m |
| 상암 순서형 조향 분포 CIL + 인접 후보 결합 | 0/6 | 3/8 | 0/6 | 50.4% | 14.80 m |

아래 셀은 목적지 도달 횟수와 두 번의 평균 진행 거리다. 진행 거리는 기준 경로에서 2.2 m 이내에 있었던 최장 지점으로 계산했다. 영상은 첫 번째 실행의 동일 데이터 모델 비교다.

| 코스 | 거리 | 기존 K-City | 같은 데이터 MSE | 조향 분포 | 비교 |
|---|---:|---:|---:|---:|---|
| 우회전 → 직진 | 451.7 m | 0/2 · 55.3 m | 0/2 · 25.8 m | 0/2 · 373.9 m | [영상](results/videos/sangam_same_data_comparison.mp4) |
| 직진 → 직진 → 우회전 | 367.8 m | 0/2 · 38.0 m | 0/2 · 275.2 m | 0/2 · 55.7 m | [영상](results/videos/arterial_straight_right_vs_baseline.mp4) |
| 좌회전 → 좌회전 | 423.9 m | 0/2 · 9.4 m | 0/2 · 380.6 m | 0/2 · 225.8 m | [영상](results/videos/arterial_two_lefts_vs_baseline.mp4) |

[첫 회전 궤적](results/images/sangam_turn_exit_trajectories.png) · [횡오차 그래프](results/images/sangam_arterial_tracking.png) · [실행별 JSON](results/metrics/sangam_arterial_study.json) · [실행별 CSV](results/metrics/sangam_arterial_trials.csv) · [K-City 모델 비교 영상](results/videos/sangam_original_comparison.mp4)

### 추가 확인

345.3 m 좌회전 → 직진 코스에서 같은 체크포인트를 모델별 두 번씩 시험했다. 학습 위치와 기준 경로는 최소 64.3 m 떨어져 있었다. 목적지 도달은 K-City 0/2회, 같은 데이터 MSE 2/2회, 분포 0/2회였다. 첫 좌회전 정렬은 K-City 0/2회, 같은 데이터 MSE 0/2회, 분포 0/2회였다. 평균 경로 근처 진행 거리는 각각 9.0 m, 344.3 m, 90.6 m였다. [추가 코스 영상](results/videos/arterial_left_straight_confirmation_vs_baseline.mp4)


## 남은 문제

본시험에서 분포 모델은 첫 우회전의 출구 정렬을 두 차례 재현했다. 연속 좌회전 코스에서는 한 번만 첫 출구를 맞췄고, 직진이 이어지는 코스에서는 두 번 모두 첫 교차로에서 이탈했다. 후속 실험에서는 같은 분포에서 직진은 기댓값을 쓰고 회전은 확률이 높은 각도 영역을 쓰는 방식과, 연속 프레임을 함께 보는 방식을 독립된 경로에서 검증할 수 있다.

초기 상암의 별도 시험에서 기준 경로가 오른쪽으로 갈라졌고 CIL은 직진 차로를 유지했다. 분포 모델의 횡오차는 기준 경로에서 2.26 m로 늘어나 당시 중단 기준 2.2 m를 넘었다. 그 구간에는 `FOLLOW ROAD`가 들어왔다. 갈림길의 원하는 출구를 언제 전달해야 하는지, 여러 프레임의 차선 움직임을 어떻게 학습할지 후속 실험에서 살펴볼 부분이다.

[갈림길 실패 비교 영상](results/videos/sangam_fork_failure.mp4) · [갈림길 궤적과 횡오차](results/images/sangam_fork_trajectory.png)

## 재현

상암 지도와 레거시 MORAI 카메라·UDP 연결 설정을 확인한다. [설정](routes/experiments/sangam_distribution/config.json) · [수집 경로](routes/experiments/sangam_distribution/manifest.json) · [주행 검증·시험 경로](routes/experiments/sangam_distribution/evaluation_manifest.json)

```bash
bash run.sh cil drive.py \
    --config routes/experiments/sangam_distribution/config.json \
    --manifest routes/experiments/sangam_distribution/evaluation_manifest.json \
    --course arterial_right_straight --describe

bash run.sh cil drive.py \
    --config routes/experiments/sangam_distribution/config.json \
    --manifest routes/experiments/sangam_distribution/evaluation_manifest.json \
    --course arterial_right_straight --checkpoint checkpoints/sangam_ordinal_v3/best.pt \
    --speed-cap-kph 16 --max-cte 15 --goal-max-cte 1.5 --goal-radius-m 2 \
    --control --reset --record-screen --steering-local-mode
```

학습용 이미지는 `collect.py`로 도로별 수집 후 `tools/inspect_collection.py`와 `prepare.py`로 검증·정렬한다. 각 모델 유형은 `train.py --model baseline|bins|ordinal`에서 선택하고, 공통 인자는 `--seed 7 --epochs 14 --patience 14 --steps-per-epoch 580 --batch-size 120 --lr 0.0002 --cover-samples --cache-images --paper-photometric`이다. 원본 주행 로그와 학습 가중치는 로컬 실험 기록에 저장했다.
