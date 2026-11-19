# PilotNet & CIL in MORAI K-City

카메라 영상으로 주행을 학습하면 어디까지 갈 수 있을까? MORAI K-City에서 직접 시범 주행 데이터를 수집하고, PilotNet과 Conditional Imitation Learning(CIL)을 PyTorch로 구현했다. 먼저 영상으로 조향을 예측해 직선과 곡선에서 주행했고, 이후 교차로에서 원하는 방향을 전달하는 모델을 시험했다.

## 교차로 주행 데모

https://github.com/user-attachments/assets/29a55b91-5d2b-4a81-ab3a-1780c281a0f6

같은 교차로의 좌회전 경로에서 왼쪽 PilotNet은 직진 방향으로 진행했고, 오른쪽 CIL은 `LEFT` 명령에 따라 좌회전해 종점 구간에 도달했다. 두 주행은 따로 녹화했으며, 먼저 끝난 왼쪽 영상은 마지막 화면에서 멈춘다.

## PilotNet으로 조향 학습하기

PilotNet 논문은 카메라 영상에서 조향에 필요한 값을 학습하는 방법을 제안했다. MORAI에서는 Pure Pursuit로 시범 주행을 수집하고, 영상과 차량 상태의 시각을 맞춰 curvature target을 만들었다.

<img src="docs/figures/pilotnet-pipeline.svg" width="620" alt="PilotNet 주행 흐름. 중앙 카메라 영상 한 장으로 곡률을 예측한 뒤 MORAI 조향으로 변환한다.">

중앙 카메라만으로 학습한 모델과 좌·중·우 카메라로 학습한 모델을 비교했다. 두 모델 모두 주행 중에는 중앙 영상 한 장만 입력받는다. 주행 속도는 별도의 목표속도 제어기가 맞춘다.

| 학습에 쓰지 않은 시험 구간 | 중앙 카메라로 학습 | 3카메라로 학습 |
|---|---|---|
| 직선 200 m | 횡오차로 중단 | 종점 구간 도달 |
| 곡선·연결도로 약 232 m | 횡오차로 중단 | 횡오차로 중단 |

각 모델로 구간별 한 번씩 시험했다. 아래 두 비교 영상에서 왼쪽은 중앙 카메라로, 오른쪽은 3카메라로 학습한 모델이다.

직선 200 m에서는 중앙 카메라 모델은 경로를 이탈했고, 3카메라 모델은 종점 구간에 도달했다.

https://github.com/user-attachments/assets/096a658d-96da-4768-b581-1f41538266f7

곡선·연결도로에서는 두 모델 모두 경로를 이탈했다.

https://github.com/user-attachments/assets/8a0d8432-1fd0-4297-a44d-84c379cf6832

교차로에서는 진행 방향을 지정하는 문제가 남았다. 좌회전 경로를 기준으로 시험했지만 차량은 직진 방향으로 진행했고, 모델에는 좌회전 목표를 전달할 입력이 없었다. [PilotNet 전체 결과와 주행 영상](pilotnet/results/README.md)

## CIL에 진행 방향 전달하기

CIL 논문은 교차로에서 주행 정책에 원하는 방향을 알려주기 위해 high-level command를 추가했다. MORAI에서는 영상, 속도, command, control targets를 같은 시각에 맞춰 학습했다.

<img src="docs/figures/cil-pipeline.svg" width="620" alt="CIL 주행 흐름. 중앙 영상과 현재 속도에서 특징을 추출하고, MGeo 경로에서 얻은 command로 branch를 골라 조향·가속·제동을 예측한다.">

`MGeo` 도로 지도의 경로에서 얻은 command로 도로 추종, 교차로 직진, 좌회전, 우회전 중 어느 branch를 사용할지 정한다.

학습과 검증에 쓰지 않은 교차로에서 같은 위치를 출발점으로 두고, 기본 CIL 모델에 직진·좌회전·우회전 명령을 각각 한 번씩 줬다.

| Command | 주행 결과 | 최대 횡오차 |
|---|---|---:|
| 직진 | 종점 구간 도달 | 2.36 m |
| 좌회전 | 종점 구간 도달 | 1.32 m |
| 우회전 | 약 61 m 중 45 m 지점에서 중단 | 7.01 m |

**CIL 곡선 주행 영상**

CIL은 시범 수집 때 조향에 Noise Injection을 적용해 경로에서 벗어난 차량을 되돌리는 조향도 학습했다. 다른 도로의 시범을 추가해 재학습한 모델은 PilotNet이 이탈한 곡선·연결도로 약 232 m를 조향 보조 없이 주행해 종점 구간에 도달했다. 최대 횡오차는 0.65 m였다.

https://github.com/user-attachments/assets/6d53fb0e-0b20-41b6-929e-d573456649d6

종점 도달은 기준 경로의 마지막 1 m 구간을 기준으로 판정했다. 횡오차 7 m는 주행 관찰을 위한 중단 기준이었다. 신호 대응, 차선 준수, 충돌 여부는 이번 평가 범위에 포함하지 않았다. [CIL 전체 결과와 주행 영상](cil/results/README.md#주행-영상)

## Point-to-point 주행

MORAI K-City 동쪽 외곽의 452.2 m 코스에서 출발 위치를 고정하고, 여러 교차로의 command를 연속으로 따를 수 있는지 시험했다.

<img src="docs/figures/kcity-test-course.png" width="680" alt="452.2 m 테스트 코스와 두 모델의 중단 위치. 기존 모델은 첫 우회전의 85.6 m 지점, 향상된 모델은 마지막 우회전의 431.7 m 지점에서 중단했다.">

### 기존 모델

CIL 논문의 branched architecture를 MORAI에 구현했다. 학습 배치에서는 네 명령을 균등하게 뽑았고, Noise Injection으로 수집한 recovery data도 사용했다. 기존 모델은 20 km/h 속도 상한에서 첫 우회전 중 경로를 이탈해 **85.6 m(18.9%)**에서 멈췄다. [CIL 논문](https://arxiv.org/abs/1710.02410)

https://github.com/user-attachments/assets/156c8f93-f8c0-4e34-8bc5-36f4f54151ac

### 향상된 모델

실패 영상을 보고, 우회전 명령 안에서 급한 코너 진입 장면과 회전 중의 recovery data를 더 자주 학습하면 첫 우회전을 통과할 수 있다고 가설을 세웠다. **CIL 구조와 네 명령의 배치 비율(각 25%)은 유지하고, 명령 안에서 뽑는 장면을 바꿨다.** 정면 영상은 명령별 13,500장씩 총 54,000장으로 유지했다.

1. **급우회전과 복귀 보강.** 시험 경로 밖의 급우회전 정면 영상 1,288장과 Noise Injection으로 수집한 recovery data 304장으로 기존 우회전 장면 일부를 교체했다. 보강 장면을 우회전 샘플 안에서 더 자주 뽑았다. 재학습한 모델은 **기존 모델이 실패한 첫 우회전을 같은 20 km/h 상한에서 조향 보조 없이 통과했다.** 15 km/h와 20 km/h 주행 모두 약 173.1 m 지점의 직진 교차로에서 이탈했다.
2. **직진 교차로 보강.** 다른 교차로의 진입·횡단·출구 장면과 recovery data를 합쳐 903장을 모았다. 이 영상으로 기존 직진 장면 일부를 교체하고 같은 방식으로 이어 학습했다. **같은 15 km/h 상한에서 진행 거리가 173.1 m에서 431.7 m로 258.6 m 늘었다.** 앞서 실패한 직진 교차로를 포함해 두 직진 교차로와 후반 좌회전을 지나 코스의 **95.5%**까지 진행했다.

https://github.com/user-attachments/assets/d6fde611-3539-4ab5-a517-2aa1d2b0bbb3

431.7 m 주행에서는 초반 좌회전에서 Pure Pursuit 조향 보조를 한 차례 적용했다. 이후 CIL로 조향을 넘겼고, 마지막 우회전에서 횡오차가 7 m에 도달해 종료했다. 같은 최종 모델도 20 km/h 상한에서는 첫 우회전에서 멈췄다. 처음과 마지막 주행 사이에 데이터 구성, 학습량, 속도 상한이 달랐다. [주행 조건과 구간별 결과](cil/results/README.md#point-to-point-주행) · [실행별 기록](cil/results/metrics/point_to_point.json)

### 실험에서 배운 점과 다음 구현

주행 중 조향이 조금만 어긋나도 다음 프레임에 들어오는 장면이 달라진다. 시범에서 드물었던 위치나 방향에 놓이면 다음 조향도 틀어져 오차가 커질 수 있다. 모방학습에서 말하는 **covariate shift**와 오차 누적이다. 이번 실험을 통해 명령별 샘플 수를 맞춘 뒤에도, 각 명령 안에서 어떤 도로와 차량 상태를 학습하는지 따로 살펴야 한다는 점을 배웠다. [모방학습의 분포 이동과 DAgger](https://proceedings.mlr.press/v15/ross11a.html)

현재 CIL은 카메라 영상 한 장과 현재 속도, command를 바탕으로 매 순간의 조향·가속·제동을 예측한다. 예를 들어 `RIGHT`는 어느 방향으로 돌아야 하는지 알려 주지만, 모델이 코너의 어느 지점을 향하는지는 제어값만으로 파악하기 어려웠다. 이번 실험을 하면서 같은 우회전 명령을 따르더라도, 코너를 도는 과정에서 생긴 오차를 줄이고 경로로 복귀하는 데에는 별도의 어려움이 있다는 점을 느꼈다. 앞으로는 차량이 이어서 따라갈 경로와 주행 중 누적되는 오차를 E2E 모델이 어떻게 다루는지 공부해 보고 싶다.

## 실행과 재현

- [PilotNet 실행 및 재현 절차](pilotnet/docs/reproduction.md)
- [CIL 설치·학습·주행 절차](cil/README.md#설치와-경로-확인)

설치 후 저장소 루트에서 `bash run.sh cil drive.py --course J_left --describe`를 실행하면 좌회전 시험 경로 설정을 확인할 수 있다. CIL 차량 제어에는 MORAI 센서·통신 설정과 학습 가중치가 필요하다.

## 참고 논문

- Bojarski et al., [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- Codevilla et al., [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
