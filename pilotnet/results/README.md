# PilotNet 주행 결과

두 학습 조건의 차량은 모두 가운데 카메라 한 장으로 주행했다. 미학습 직선에서는 3카메라로 학습한 모델이 종점 구간에 도달했고, 미학습 곡선에서는 공개한 세 모델 모두 중단됐다.

## 주행 영상

<details>
<summary>01. 미학습 직선 · 중앙 카메라로만 학습한 모델은 중단, 3카메라 학습 모델은 종점 도달</summary>

https://github.com/user-attachments/assets/096a658d-96da-4768-b581-1f41538266f7

</details>

<details>
<summary>02. 미학습 곡선 · 두 모델 모두 횡오차 기준으로 중단</summary>

https://github.com/user-attachments/assets/8a0d8432-1fd0-4297-a44d-84c379cf6832

</details>

<details>
<summary>03. 곡선 샘플을 늘린 모델 · 곡선에서는 중단, 직선에서는 종점 도달</summary>

https://github.com/user-attachments/assets/0d5447ed-05c4-49b4-8b58-5e06ff542bfb

</details>

<details>
<summary>04. 교차로 J · 좌회전 경로에서 직진 방향으로 진행하다 중단</summary>

https://github.com/user-attachments/assets/248675d3-e24b-4e05-9902-c5ea20747ace

</details>

01·02번 영상의 왼쪽은 가운데 카메라로만 학습한 모델, 오른쪽은 세 카메라로 학습한 모델이다. 서로 다른 주행의 녹화이며, 먼저 종료한 화면은 마지막 장면에서 멈춘다. 03번은 같은 모델의 곡선·직선 시험을 이어 붙였다.

교차로 J는 PilotNet 학습 도로와 일부 겹친다. 교차로에서 실제로 선택한 방향은 루트의 CIL 좌회전 영상과 나란히 볼 수 있다. [교차로 비교](../../README.md#교차로-주행-데모) · [영상 파일](videos/)

## 시험 구간

| 구분 | K-City 기준 경로의 진행거리 |
|---|---|
| 학습 데이터 수집 | 직선 0~65m, 곡선 640~866m |
| 새 직선 시험 (`U1`) | **1300~1500m** |
| 새 곡선과 연결 직선 시험 (`U2`) | **978.337~1210m** |

학습 구간과 시험 구간은 겹치지 않는다. 같은 K-City 맵·차량·맑은 낮 조건을 사용했다.
[학습 도로와 시험 도로 지도](images/course_map.png)

가중치를 고정한 상태에서 경로 중앙·헤딩 오프셋 0°·목표속도 19 km/h로 총 7회 주행해 영상을 기록했다. 횡오차 7 m 이상이면 중단하고, 그 전에 기준경로의 마지막 1 m 구간에 도달하면 종점 도달로 기록했다.

## 평가 표

- [이미지 예측 평가](metrics/open_loop.csv): 새 도로의 동일한 가운데 카메라 이미지 830장, 두 모델 비교.
- 학습하지 않은 직선·곡선에서 두 학습 조건을 한 번씩 주행한 결과. [주행 결과 CSV](metrics/driving_trials.csv)
- 곡선 샘플 비중을 바꾼 모델과 기존 3카메라 모델의 비교. 기존 모델의 두 주행은 주행 결과 CSV와 같다. [추가 실험 CSV](metrics/curve_sampling_comparison.csv)
- 교차로 J 좌회전 경로에서 3카메라로 학습한 모델의 주행 한 건. [교차로 CSV](metrics/intersection_example.csv)

곡선 비중을 높인 모델은 기존 학습·검증 데이터로 학습했고, 기본 모델을 평가한 도로에서 탐색적으로 다시 시험했다. 중단된 주행의 횡오차 RMSE는 종료 전 구간에 대해서만 계산한 값이다.

공개한 7회 주행의 원본 녹화와 프레임별 로그는 `runs/closed_loop_cte7_public/`에 있다. 과거 횡오차 1 m 종료 기준의 20회 시험은 로컬 아카이브에 있다. [영상별 원본 경로와 SHA256](video_sources.json) · [시험 재현](../docs/reproduction.md) · [곡선 샘플 설정](curve_sampling.json)
