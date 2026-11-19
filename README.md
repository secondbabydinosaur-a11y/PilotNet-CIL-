# PilotNet & CIL in MORAI K-City

PilotNet과 Conditional Imitation Learning(CIL) 논문의 모델을 PyTorch로 재구현해 MORAI K-City에 적용한 개인 프로젝트다. 시뮬레이터에서 주행 데이터를 직접 수집하고, 학습한 모델을 차량 제어에 연결해 직선·곡선·교차로에서 시험했다.

## 교차로 주행 데모

https://github.com/user-attachments/assets/29a55b91-5d2b-4a81-ab3a-1780c281a0f6

같은 교차로의 좌회전 경로에서 왼쪽의 PilotNet은 직진 방향으로 진행했고, 오른쪽의 CIL은 `LEFT` 명령을 받아 좌회전했다. 두 번의 MORAI 녹화를 나란히 편집했으며, 먼저 끝난 왼쪽 영상은 마지막 화면에서 멈춘다.

[전체 PilotNet 주행 영상](pilotnet/results/README.md#주행-영상) · [전체 CIL 주행 영상](cil/results/README.md#주행-영상)

## 직접 구현한 것

- MORAI에서 카메라 영상과 차량 상태를 받고, 지도 경로를 따라가는 Pure Pursuit로 학습용 주행을 수집했다. 센서 시각을 맞춰 모델 입력과 제어 정답을 만들었다.
- PilotNet 논문의 3카메라 학습과 곡률 예측을 구현했다. 주행 시에는 가운데 카메라 영상에서 예측한 곡률을 MORAI 조향으로 변환한다. 종방향은 목표속도 제어기로 처리했다.
- CIL 논문의 명령별 분기 구조를 구현했다. 가운데 카메라 영상과 속도를 입력받고, MGeo 경로에서 얻은 진행 명령으로 분기를 골라 조향·가속·제동을 계산한다.

## MORAI 주행 결과

**PilotNet.** 3카메라로 학습한 모델과 가운데 카메라 한 대로 학습한 비교 모델을 같은 미학습 구간에서 시험했다. 직선에서는 3카메라 모델이 종점 구간에 도달했고 비교 모델은 중단됐다. 곡선에서는 두 모델 모두 횡오차 기준으로 중단됐다. [PilotNet 학습 방식과 결과](pilotnet/README.md#3카메라-학습과-주행-결과)

**CIL.** 학습과 검증에 사용하지 않은 교차로 J에서 방향별로 한 번씩 주행했다.

| CIL 교차로 J 시험 | 결과 | 최대 횡오차 |
|---|---|---:|
| 직진 | 종점 구간 도달 | 2.36 m |
| 좌회전 | 종점 구간 도달 | 1.32 m |
| 우회전 | 횡오차 기준으로 중단 | 7.01 m |

[교차로 영상과 시험 조건](cil/results/README.md#주행-영상)

## Point-to-point 주행

K-City 동쪽 외곽 기준 경로 452.2 m 중 **383.4 m 지점(84.8%)**까지 진행했다. 초반 좌회전 구간에서 조향 보조를 한 차례 적용했다. 이후 좌회전에서 횡오차가 7 m를 넘어 종료됐고, 종점까지 약 68.8 m가 남았다.

https://github.com/user-attachments/assets/3467f322-307e-4b68-b2ce-49dba66a92b3

[추가 학습과 주행 상세 기록](cil/results/README.md#point-to-point-주행)

## 실행과 상세 기록

| 모델 | 구현과 실행 | 주행 기록 |
|---|---|---|
| PilotNet | [구현 안내](pilotnet/README.md) · [실행·재현](pilotnet/docs/reproduction.md) | [영상·지표](pilotnet/results/README.md) |
| CIL | [구현·실행 안내](cil/README.md) | [영상·지표](cil/results/README.md) |

저장소 루트에서 `bash run.sh cil drive.py --course J_left --describe`를 실행하면 좌회전 시험 조건을 확인할 수 있다. CIL 차량 제어에는 MORAI 센서·통신 설정과 별도 학습 가중치가 필요하다.

## 참고 논문

- Bojarski et al., [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- Codevilla et al., [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)

## 한계

이번 CIL 구현은 경로에서 받은 진행 명령을 따라 차를 제어하는 범위까지 다뤘다. 여러 차선 중 어느 차선을 탈지 정하고, 신호등 상태에 맞춰 멈췄다가 출발하고, 주변 차량과 보행자에 대응해야 하는 복잡한 도심에서는 이 구성만으로 안정적인 point-to-point 주행이 어렵다. 같은 K-City에서도 곡선과 교차로에서 경로 이탈이 관찰됐다.

공개 시험은 K-City의 제한된 조건에서 진행했다. 종점 도달은 기준경로의 마지막 1 m 구간을 기준으로 판정했고, 횡오차 7 m는 주행을 관찰하기 위한 중단 기준으로 사용했다. 차선 준수와 충돌 여부는 별도로 계측하는 평가가 필요하다.
