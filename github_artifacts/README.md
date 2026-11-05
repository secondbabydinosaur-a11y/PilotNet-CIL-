# PilotNet 실험 기록

가운데 카메라로 학습한 모델(`center`)과 좌·중·우 카메라로 학습한 모델(`three`)을 비교했다.
**실시간 조향 입력은 두 모델 모두 가운데 카메라 한 장이다.** 원본은 사람이 아닌 지도 PP 자동 teacher 주행이다.
기존 모델과 train/val/test CSV를 고정한 뒤 미학습 도로와 헤딩 복귀 시험을 추가했다.

- [GitHub에서 시험별 영상 보기](videos.md): 실제 MORAI GUI 영상과 전방 센서 영상 구분
- [로컬 영상 갤러리](index.html): 저장소를 내려받아 브라우저로 열기
- [코스 지도](plots/course_map.png)
- [open-loop CSV](reports/open_loop.csv), [closed-loop CSV](reports/closed_loop.csv)
- [고정한 시험 계획](reports/protocol.json), [제외한 설정 실패·사용자 중단](reports/excluded_attempts.json)
- [모델·학습 기록](models), [원본 teacher 영상·출처](teacher)
- [전체 파일 SHA256 및 영상 검사](reports/files.json)

## 시험 조건

| 코스 | 범위 | 의미 |
|---|---|---|
| A | s=0~55m | 학습했던 직선 |
| D | s=650~850m | 학습했던 연속 곡선 |
| J | s=40~108m | 미학습 좌회전 교차로, 기존 코스 일부와 겹침 |
| U1 | s=1300~1500m, 200m | 학습에 없는 긴 직선 |
| U2 | s=978.337~1210m, 약 232m | 학습에 없는 진입 곡선과 연결 직선 |

초기 조건은 정상, 횡위치 +0.3m/−0.3m, 헤딩 +5°/−5°다. 헤딩 시험은 위치를 가운데에 둔다.
각 조건당 한 번이며 횡오프셋 3개를 독립 반복 3회로 해석하지 않는다. 새로운 헤딩·미학습 시험은
실제 UDP 상태로 초기 각도·위치를 확인했다. 기존 시험의 요청/측정 초기값도 CSV에 구분했다.
`base_nominal_gui_rerun`은 기존 정상 출발을 GUI 녹화로 다시 실행한 묶음이며 원래 시험과 합산하지 않는다.

목표 19km/h, 같은 맑은 정오, 같은 센서/차량이다. 고속도로 위치의 시험도 19km/h로 진행했다.
신경망은 조향만 결정하고 종방향은 고정 속도 피드백이다. 지도는 초기 배치와 평가에만 사용한다.
종료점 1m 이내면 완주, |횡오차|>1m이면 중단한다. 센서 지연과 헤딩 이탈 중단도 기록한다.
진행률은 마지막 제어 로그 기준이어서 완주도 약 98~99.5%로 표시될 수 있다.

복귀는 3m 이상 전진한 뒤 |횡오차|≤0.15m와 |헤딩 오차|≤2°를 1초 연속 유지한 첫 구간이다.
복귀 시간은 제어 시작부터 측정하며 정차 출발 가속을 포함한다. 복귀한 뒤의 재이탈/최종 중단은 별도로 본다.

## Open-loop

| 도로 | 모델 | 이미지 수 | 곡률 RMSE (1/m) | 곡률 MAE (1/m) |
|---|---|---|---|---|
| seen | center | 1653 | 0.000119 | 0.000090 |
| unseen | center | 830 | 0.009216 | 0.005665 |
| seen | three | 1653 | 0.000158 | 0.000119 |
| unseen | three | 830 | 0.007449 | 0.005595 |

`seen`은 같은 A/D 도로의 다른 주행 그룹(g06), `unseen`은 새 U1/U2 도로의 시험 전용 teacher다.
두 평가 모두 이미지마다 실제 조향 피드백을 시간 보간한 뒤 `−tan(조향각)/3m`로 만든 곡률을 정답으로 쓴다.
이는 자전거 모델 가정이며 실제 궤적 곡률을 독립 계측한 정답은 아니다. 곡률이 거의 0인 직선의 R²는
작은 오차에도 불안정할 수 있으므로 MAE/RMSE와 zero-curvature baseline을 함께 본다.

## Closed-loop

| 시험 묶음 | 모델 | 코스 | 완주 | 평균 진행률 |
|---|---|---|---|---|
| base_lateral | center | A | 3/3 | 98.1% |
| base_lateral | center | D | 3/3 | 99.5% |
| base_lateral | center | J | 0/3 | 56.2% |
| base_lateral | three | A | 3/3 | 98.1% |
| base_lateral | three | D | 3/3 | 99.5% |
| base_lateral | three | J | 0/3 | 54.9% |
| base_nominal_gui_rerun | center | A | 1/1 | 98.1% |
| base_nominal_gui_rerun | center | D | 1/1 | 99.5% |
| base_nominal_gui_rerun | center | J | 0/1 | 56.1% |
| base_nominal_gui_rerun | three | A | 1/1 | 98.1% |
| base_nominal_gui_rerun | three | D | 1/1 | 99.5% |
| base_nominal_gui_rerun | three | J | 0/1 | 54.8% |
| seen_heading | center | A | 1/2 | 64.8% |
| seen_heading | center | D | 1/2 | 51.5% |
| seen_heading | three | A | 2/2 | 98.1% |
| seen_heading | three | D | 2/2 | 99.5% |
| unseen_heading | center | U1 | 0/2 | 5.9% |
| unseen_heading | center | U2 | 0/2 | 4.6% |
| unseen_heading | three | U1 | 1/2 | 52.6% |
| unseen_heading | three | U2 | 0/2 | 13.2% |
| unseen_lateral | center | U1 | 0/3 | 11.8% |
| unseen_lateral | center | U2 | 0/3 | 9.9% |
| unseen_lateral | three | U1 | 3/3 | 99.5% |
| unseen_lateral | three | U2 | 0/3 | 14.8% |

중단 시험의 횡오차 통계는 중단 전 구간만 포함한다. 짧게 실패한 모델의 낮은 RMSE를 우수한 완주로 해석하지 않는다.
누적 주행거리와 intervention/km도 CSV에 기록한다. 구간 끝에서 시험을 종료하므로 무한 연속주행의
평균 무개입 거리나 원 논문의 autonomy 점수로 해석하지 않는다.
조건별 시계열 그림은 [plots](plots)에 있다. 영상은 실제 실행 속도이며 종료 후 제동 구간이 포함될 수 있다.
GUI 영상은 `morai_screen.mp4`, 실제 전방 카메라+진단 표시는 `drive.mp4`다.
각 영상은 프레임 수·길이·해상도를 ffprobe로 확인했다. 초기 18개 시험에는 전방 영상만 있고
정상 출발 재시험·추가 헤딩·미학습 시험에는 실제 MORAI 창 영상도 있다.
추론 시간은 새 프레임의 모델 forward와 scalar 출력 동기화 구간이며 JPEG 디코딩·전처리·전송 시간은 포함하지 않는다.

## 해석 범위

- 같은 K-City 맵의 미학습 구간 시험이다. 새 도시·날씨·차량 일반화를 검증한 결과는 아니다.
- 3개 카메라 학습은 같은 30 epoch에서 더 많은 이미지와 optimizer step을 사용했다.
- 좌우 카메라 복귀 라벨은 2초 뒤 teacher 위치에 대한 우리의 PP-delta 근사다. yaw 시점 증강은 학습에 없다.
- 교차로 실패에는 미학습 형상과 목적지 명령 부재가 함께 영향을 줄 수 있다. 모든 교차로 통과가
  PilotNet 구조상 불가능하다는 근거로 사용하지 않는다. CIL은 아직 구현하지 않았다.
- 충돌 ground truth를 별도로 계측하지 않았다. 충돌 0회를 검증한 결과로 표시하지 않는다.
- 단일 학습 seed와 조건별 한 번의 실행이다. 실험 조건에서 관측한 결과로 해석한다.

## 재현과 파일

저장소 루트의 `README.md`와 `docs/collection.md`에 수집·학습 명령이 있다. 모델 가중치는
`github_artifacts/models/{center,three}/best.pt`에 있다. 기존 원본 경로의 체크포인트와 SHA256이 같다.
원본 JPEG 약 13GB와 MORAI 설치/지도 파일은 이 결과 폴더에 복사하지 않았다. 데이터 원본은 로컬
`dataset/raw`와 `dataset/unseen_raw`에 보존한다. CSV의 이미지 경로는 저장소 루트 기준이며,
이미지 단위 평가 재현에는 해당 원본이 필요하다. 원본이 포함된 공개 데이터셋이라고 표기하지 않는다.

```bash
# 모델 고정 확인 후 추가 시험을 재실행할 때: 기존 완료된 주행 기록은 재사용한다.
bash run.sh -m tools.evaluation.run_matrix --courses A D --conditions heading_left heading_right --record-screen
bash run.sh -m tools.evaluation.run_matrix --courses U1 U2 --repeats 3 --record-screen --tag unseen
bash run.sh -m tools.evaluation.run_matrix --courses U1 U2 --conditions heading_left heading_right --record-screen --tag unseen
bash run.sh -m tools.evaluation.build_github_artifacts
```

논문: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
게시 대상: [secondbabydinosaur-a11y/PilotNet-CIL-](https://github.com/secondbabydinosaur-a11y/PilotNet-CIL-).
