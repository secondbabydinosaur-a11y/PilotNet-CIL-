# PilotNet KASA

현재 데이터로 **가운데 카메라 모델과 3개 카메라 학습 모델**을 구현·학습했다.
두 모델 모두 실행할 때는 가운데 카메라 한 장으로 곡률을 예측한다. CIL은 아직 구현하지 않았다.

## 먼저 볼 결과

- [GitHub 실험 기록과 결과 표](github_artifacts/README.md)
- [실제 MORAI 영상 모음](github_artifacts/index.html): 내려받아 브라우저에서 열면 조건별 재생 가능
- [open-loop 지표](github_artifacts/reports/open_loop.csv)
- [closed-loop 시험별 지표](github_artifacts/reports/closed_loop.csv)
- [학습/미학습 코스 지도](github_artifacts/plots/course_map.png)
- [원본 teacher 출처](github_artifacts/teacher/teacher_sources.json)
- [곡선 프레임 선택 기준 추가 학습·2회 시험](github_artifacts/curve_sampling_v2/README.md)

결과 모음은 `tools.evaluation.build_github_artifacts`가 생성한다. GitHub에서는 Markdown과 MP4 링크를 보고,
HTML 갤러리는 저장소를 내려받아 연다. 추가 시험 영상은 **실제 MORAI 창 녹화**이며 전방 카메라 영상도 보존한다.
원본 주행은 지도 PP 자동 teacher다. 사람 수동 운전으로 표기하지 않는다.

기존 학습 모델을 고정한 뒤 같은 K-City 맵의 **학습하지 않은 직선 200m와 곡선 약 232m**를 추가했다.
정상 출발, 횡위치 ±0.3m, 헤딩 ±5°를 구분해서 기록한다. 새 구간 teacher는 시험용으로만 보관한다.
정상 출발 성공과 자세가 틀어진 상태에서의 복귀 성공은 각각 따로 확인한다.

## 파일 구조와 수정 위치

```text
pilotnet_kasa/
├── dataset.py             사용자가 작성한 Dataset·전처리 연결
├── model.py               PilotNet 구조: 다음에 모델을 바꿀 곳
├── train.py               학습·validation·best/last 체크포인트
├── evaluate.py            기록 영상 open-loop 평가
├── drive.py               MORAI 실시간 추론·제어·영상 녹화
├── check_model.py         실제 한 배치로 순전파·역전파 확인
├── data_utils.py          공통 IO·시간 보간·곡률·이미지 전처리
├── config.json            센서·crop·속도 조건
├── learning/              체크포인트·지표·캐시·실시간 수신 보조 코드
├── tools/
│   ├── collection/        기존 수집·검수·MORAI 통신
│   ├── preparation/       가운데/좌우 카메라 정답 생성
│   └── evaluation/        실험 자동 실행·영상과 결과 모음 생성
├── dataset/
│   ├── raw/               원본 JPEG·차량 상태·제어·teacher 기록
│   ├── splits/            그룹 분할 목록
│   ├── paired/            가운데 이미지–곡률 CSV
│   └── paired_three/      3개 카메라 train + 동일한 가운데 val/test
├── runs/                  학습 체크포인트·원시 실험 로그·모든 주행 영상
├── results/               기존 영상 모음의 로컬 생성물
├── github_artifacts/      업로드할 가중치·지표·그림·실제 MORAI 영상
├── assets/                코스 경로·참고 그림·MORAI descriptor
├── tests/                 수집·라벨·모델·실시간 수신 검사
└── docs/collection.md     수집과 데이터 준비 상세 기록
```

## 실행 환경

프로젝트 루트에서 실행한다. 이 PC는 CUDA 11/12 패키지 경로가 겹쳐 기본 `python3`의 torch import가
실패한다. `run.sh`는 실행한 프로세스에서 경로 우선순위를 지정한다. 설치 파일과 전역 설정은 수정하지 않는다.
확인한 환경은 Python 3.8, PyTorch 2.4.1+cu121, RTX 4080 SUPER다.

```bash
cd /home/kuuve/catkin_ws/src/pilotnet_kasa
bash run.sh check_model.py
bash run.sh -m unittest discover -s tests -v

# GitHub에서 처음 내려받았을 때: 보관한 baseline 가중치와 계획을 runs/에 복원
# 원본 JPEG가 없는 환경에서는 실제 데이터 로딩 검사를 건너뛴다.
bash run.sh -m tools.evaluation.restore_artifacts
```

별도 환경을 새로 만들 때는 공식 PyTorch cu121 wheel과 `requirements-train.txt`를 참고한다.
현재 설치 환경에 다시 설치할 필요는 없다.

## 학습과 open-loop 평가

```bash
# 기존 폴더는 덮어쓰지 않는다. 다시 학습할 때 v2 등 새 이름을 지정한다.
bash run.sh train.py --run-dir runs/pilotnet_center_v2 --camera-mode center --epochs 30 --cache

# 좌우 복귀 라벨을 처음 만드는 경우만 실행한다. 현재는 이미 생성되어 있다.
# bash run.sh -m tools.preparation.build_side_labels
bash run.sh train.py --run-dir runs/pilotnet_three_v2 --camera-mode three --epochs 30 --cache

bash run.sh evaluate.py --checkpoint runs/pilotnet_center_v2/best.pt --split test
bash run.sh evaluate.py --checkpoint runs/pilotnet_three_v2/best.pt --split test
```

기본 학습은 Adam, batch size 64, seed 42, 최대 30 epoch다. `|곡률| >= 0.003` 샘플에
2배 sampling weight를 주며 train에만 적용한다. validation/test는 원래 분포의 가운데 영상이다.
이는 v1의 보정 후 정답 기준이다. 추가 v2 실험은 `--curve-basis center`로 좌우 복귀 보정 전
teacher 곡률을 사용한다. 논문 5.1절은 곡선 비중 증가를 설명하며 고정된 숫자 비율은 제시하지 않는다.
따라서 2배는 우리 구현의 선택이다. v2는 원래 train/val만 사용하고 기존 가중치를 덮어쓰지 않는다.
validation MSE로 best를 고르고 test는 모델 선택에 사용하지 않는다. `--cache`는 전처리 uint8 영상을 RAM에
보관해 반복 JPEG 디코딩을 줄인다. 원본 이미지는 수정하지 않는다.
곡률의 수치가 작아 최적화 loss에만 100²을 곱한다. CSV/로그/평가 지표의 단위는 원래 `1/m`이다.

입력 계약: YUV `[B,3,66,200]`, float32 0~255 → 모델 내부에서 `/127.5-1` → 곡률 `[B,1]`.
Conv/FC 크기는 논문 그림 4를 따르고, ReLU·정규화 수식·출력층 초기화·학습 설정은 이번 구현의 선택이다.
Flatten은 실제 feature 크기에 맞춰 1152다.

체크포인트에는 전처리·카메라·라벨 부호·휠베이스·원본 CSV 해시를 저장한다.
평가/주행 시 현재 데이터 계약과 다르면 중단하므로 서로 다른 센서/라벨 설정이 조용히 섞이지 않는다.

## MORAI closed-loop 시험

MORAI는 `R_KR_PR_K-city_2025`, Ioniq5, 수집 당시 센서 설정을 사용한다.
Destination port는 가운데 카메라 9105, 일반 Ego 9301이고 제어 Host port는 9093이다.
시뮬레이터는 다른 제어 코드가 실행되지 않은 상태에서 사용한다.

```bash
# 실제 차량을 시작점으로 옮기고 CNN으로 주행한다. 종료 시 제동한다.
bash run.sh drive.py --checkpoint runs/pilotnet_center_v1/best.pt --course A --control --reset --record-screen

# 두 모델 × 세 코스 × 초기 횡오프셋 0/+0.3/-0.3m. 완료된 같은 체크포인트 시험은 재사용한다.
bash run.sh -m tools.evaluation.run_matrix --repeats 3

# 발표용 실제 MORAI GUI 녹화: 같은 조건에서 초기 횡오프셋 0m로 다시 시험한다.
bash run.sh -m tools.evaluation.run_matrix --repeats 1 --record-screen

# 전체 순차 실행 및 결과 모으기
bash run.sh -m tools.evaluation.run_all

# 학습했던 구간의 헤딩 복귀: 중앙에서 +5° 또는 -5°로 정차 출발
bash run.sh -m tools.evaluation.run_matrix --courses A D --conditions heading_left heading_right --record-screen

# 미학습 구간의 기록 수집(시험 전용), open-loop, closed-loop를 순차 실행
bash run.sh -m tools.evaluation.run_extended

# 결과 파일을 github_artifacts/에 모으기
bash run.sh -m tools.evaluation.build_github_artifacts

# 별도 v2 학습과 헤딩 0° 곡선/직선 각 1회 비교. 완료된 동일 시험은 재사용한다.
bash run.sh -m tools.evaluation.run_curve_sampling_experiment
```

A=직선 s0~55m, D=연속 곡선 s650~850m, J=미학습 좌회전 교차로 s40~108m다.
U1=미학습 직선 s1300~1500m, U2=미학습 진입 곡선 s978.337~1210m다.
목표속도는 19km/h이고 끝부분에 제동한다. 종료점 1m 이내면 완료로 판정한다.
횡오차 1m 초과 시 intervention으로 중단한다. 카메라/Ego 상태가 오래되면 가속을 중단하고 제동한다.

지도는 배치·진행률·횡오차 평가·종료에 사용한다. 조향은 CNN 출력만 사용하며 PP 조향을 섞지 않는다.
종방향은 속도 피드백 제어다. 실행별 `drive.mp4`는 전방 센서 영상, `morai_screen.mp4`는 실제 MORAI 창이다.
창 녹화는 X11에서 표시된 Simulator 영역을 캡처하므로 녹화 중 창을 가리거나 이동하지 않는다.
실험 행렬 종료 후 차량을 P/Keyboard로 두고 수집용 추가 속도 제한을 해제한다.

헤딩 조건은 `--initial-heading-deg 5` 또는 `-5`다. 좌우 횡오프셋과 별도로 설정한다.
정차 상태에서 요청한 위치·헤딩을 실제 Ego 상태로 확인한 뒤 출발하며 `initial_state.json`에 보존한다.
복귀는 3m 이상 전진한 뒤 |횡오차|≤0.15m, |헤딩 오차|≤2°를 1초 유지하는지로 측정한다.
복귀 후 재이탈이 생길 수 있으므로 복귀 시간·거리와 최종 완주 여부를 함께 본다.

시험 결과를 보고 모델을 다시 학습하거나 센서·전처리·조향 계수를 조정하지 않는다.
`runs/github_protocol.json`에 모델과 기존 CSV의 SHA256을 고정하고 추가 시험 전후에 확인한다.
원본 JPEG는 `dataset/raw`, 새 구간 JPEG는 `dataset/unseen_raw`에 남으며 Git에 넣지 않는다.
이미지 평가를 다시 실행하려면 해당 로컬 데이터가 필요하다. 학습된 가중치는
`github_artifacts/models/{center,three}/best.pt`에도 같은 SHA256으로 복사한다.

## 결과 해석 범위

- 기존 train/val/test는 같은 A/D 도로의 다른 주행 그룹이다. U1/U2만 추가 도로 일반화 시험이다.
  이 역시 같은 맵·맑은 정오이므로 새 도시나 날씨 일반화를 검증한 결과는 아니다.
- 좌우 카메라 정답은 2초 뒤 teacher 위치를 이용한 **우리의 PP-delta 복귀 근사**다.
  원 논문의 비공개 보정식을 그대로 재현한 것은 아니며 yaw 시점 증강도 포함하지 않는다.
- 교차로 J는 미학습 형상이다. 실패 원인에 새로운 장면과 목적지 명령 부재가 함께 있을 수 있다.
  PilotNet의 모든 교차로 통과가 구조적으로 불가능하다는 결론으로 확대하지 않는다.
- 두 모델은 동일한 30 epoch를 사용하므로 3개 카메라 모델이 더 많은 이미지와 optimizer step을 본다.
- 충돌 ground truth는 별도 계측하지 않았다. 0회 충돌을 검증했다고 주장하지 않는다.
- CIL 구현·학습·평가는 아직 진행하지 않았다.

시뮬레이터의 native GPU crash로 끊긴 주행과 초기 배치 검사 실패는 원본을 보존하고,
설정/인프라 중단으로 분류한 뒤 같은 모델·조건으로 다시 실행했다. 제외 사유도 결과 폴더에 남긴다.

원 논문: https://arxiv.org/abs/1604.07316
수집·검수·라벨 가정의 상세 출처는 [수집 안내](docs/collection.md)에 있다.
