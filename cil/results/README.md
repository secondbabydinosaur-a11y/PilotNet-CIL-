# CIL 주행과 학습 결과

MORAI K-City에서 기본 모델을 짧은 곡선·교차로에서 시험하고, 데이터를 보강한 모델로 동쪽 외곽 경로의 연속 주행을 기록했다. [최종 point-to-point 영상](#point-to-point-주행)

## 주행 영상

곡선과 교차로 시험에는 기본 모델의 19번째 epoch 가중치를 사용했다. 모두 학습과 검증에 쓰지 않은 도로에서 조건별로 한 번씩 주행했다.

### 시험 교차로

`J`는 K-City 기준 경로 초반에 있는 교차로다. 세 번 모두 같은 접근로의 같은 위치에서 출발해 각각 직진·좌회전·우회전 명령을 줬다. 좌회전은 첫 화면의 PilotNet–CIL 비교와 같은 경로다. [비교 영상](../../README.md#교차로-주행-데모)

![K-City 기준 경로에서 시험 교차로의 위치와, 공통 출발점에서 갈라지는 직진·좌회전·우회전 경로](images/test_junction.png)

| 시험 구간 | 관찰 결과 | 최대 횡오차 |
|---|---|---:|
| 곡선·연결도로 (`U2`) | 약 116/232 m에서 중단 | 7.016 m |
| 시험 교차로 · 직진 | 종점 구간 도달 | 2.358 m |
| 시험 교차로 · 좌회전 | 종점 구간 도달 | 1.321 m |
| 시험 교차로 · 우회전 | 약 45/61 m에서 중단 | 7.006 m |

횡오차는 차량이 기준 경로에서 옆으로 벗어난 거리다. 횡오차가 7 m 이상이면 주행을 중단하고, 그 전에 기준 경로의 마지막 1 m 구간에 들어가면 종점 도달로 기록했다. 차선 준수와 충돌은 별도로 계측해야 한다. [시험별 기록](metrics/closed_loop.csv) · [시험 조건](metrics/closed_loop.json)

<details>
<summary>곡선에서 직선으로 이어지는 시험 구간</summary>

https://github.com/user-attachments/assets/d41911e4-ea73-4870-9b94-67535cfd0919

</details>

<details>
<summary>시험 교차로 · 직진</summary>

https://github.com/user-attachments/assets/63fbbd4e-ff00-4bc6-b5d7-b65d74dbff27

</details>

<details>
<summary>시험 교차로 · 좌회전</summary>

https://github.com/user-attachments/assets/b9abc75d-7825-44f3-a835-31c9f7326e68

</details>

<details>
<summary>시험 교차로 · 우회전</summary>

https://github.com/user-attachments/assets/c0cd68c8-ff82-48b2-9535-f95c82ce94d7

</details>

## Point-to-point 주행

### 최종 동쪽 외곽 경로

좌·우회전과 직진 시범에서 얻은 6,110개 샘플을 보강해 같은 구조의 CIL을 33,003개 샘플로 다시 학습했다. 기존 검증 세트에서 선택한 16번째 epoch를 사용했다. 연장 전 경로에서 한 번씩 시험했을 때 기본 모델은 첫 우회전 도중 중단됐고, 재학습한 모델은 그 출구를 통과했다.

https://github.com/user-attachments/assets/3467f322-307e-4b68-b2ce-49dba66a92b3

| 항목 | 주행 기록 |
|---|---|
| 도달 지점 | 기준 경로 452.2 m 중 383.4 m (84.8%) |
| 조향 보조 | 초반 좌회전 한 구간에서 Pure Pursuit 사용 |
| 종료 | 뒤쪽 좌회전에서 횡오차 7.21 m로 중단. 남은 경로 약 68.8 m |

영상에 `STEERING INTERVENTION ACTIVE`가 표시된 동안 조향을 보조했고, 가속·제동은 CIL이 예측했다. 보조 구간을 벗어나자 조향도 CIL로 돌아왔다. 학습 주행 위치와 가까운 도로가 포함된 1회 시연이다.

### 이전 대회 경로

대회 출발점에서 마지막 좌회전 출구 뒤까지 약 645 m를 주행해 종점 구간에 도달했다. 기본 모델을 사용했고, 학습에 사용한 도로 가까이를 지나는 구간이 포함되어 있다.

<details>
<summary>이전 대회 경로 주행 영상</summary>

https://github.com/user-attachments/assets/6216f718-3e21-4d6c-bed9-246a4d155673

</details>

아래 그림은 기본 모델의 곡선·교차로 시험 네 건과 이전 대회 경로를 보여 준다. 회색 선은 기준 경로, 색 선은 차량이 실제로 지나간 궤적이다.

![기본 모델의 곡선·교차로 시험과 이전 대회 경로의 주행 궤적](images/closed_loop_trajectory.png)

## 학습과 검증

![기본 CIL 모델의 학습 및 검증 곡선](images/baseline_training_curve.png)

| 기본 모델 | 결과 |
|---|---:|
| 학습 / 검증 샘플 수 | 26,893 / 1,549 |
| 완료 / 선택 epoch | 27 / 19 |
| 검증 제어 MSE · 명령별 평균 | 0.005748 |
| 검증 조향 MAE · 정규화값 | 0.083502 |

제어 MSE는 조향·가속·제동의 제곱오차를 사용한다. 네 진행 명령의 MSE를 같은 비중으로 평균해 가중치를 선택했다. 검증 입력은 학습에 사용하지 않은 도로의 가운데 카메라 이미지다. [명령별 평가](metrics/baseline_validation.csv) · [학습 요약](metrics/baseline_training_summary.json)

## 하이퍼파라미터 비교

같은 학습·검증 데이터로 학습률, 가중치 감쇠, 드롭아웃 설정을 비교했다. 모델 선택에 사용한 명령별 평균 MSE는 기본 설정이 가장 낮았다.

| 설정 | 선택 epoch | 검증 제어 MSE · 명령별 평균 | 종료 |
|---|---:|---:|---|
| 기본 | 19 | **0.005748** | 조기 종료 |
| 낮은 학습률·L2 | 17 | 0.005894 | 수동 중단 |
| 강한 드롭아웃 | 6 | 0.005771 | 조기 종료 |
| 약한 L2 | 6 | 0.006394 | 수동 중단 |

[학습 설정과 세부 지표](metrics/hyperparameter_comparison.csv) · [CIL 학습 및 실행 방법](../README.md) · [PilotNet 주행 결과](../../pilotnet/results/README.md)
