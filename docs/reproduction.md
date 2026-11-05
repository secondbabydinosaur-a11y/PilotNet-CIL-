# 실행과 재현

## 환경

프로젝트 루트에서 실행한다. 확인한 환경은 Python 3.8, PyTorch 2.4.1+cu121,
RTX 4080 SUPER다. 의존성은 `requirements.txt`, `requirements-train.txt`에 있다.
`run.sh`는 이 PC의 CUDA 패키지 경로 충돌을 해당 프로세스 안에서 처리한다.

```bash
bash run.sh -m unittest discover -s tests -v
bash run.sh -m tools.evaluation.restore_artifacts
```

복원 명령은 `checkpoints/`의 baseline 가중치와 `docs/evaluation_protocol.json`을
기존 실행 코드가 사용하는 로컬 `runs/` 경로에 복사한다. 학습이나 주행은 실행하지 않는다.

## 데이터

수집 절차와 센서 포트는 [수집 안내](collection.md)에 있다.
원본 JPEG는 로컬 `dataset/raw/`에, 시험 전용 새 도로 기록은 `dataset/unseen_raw/`에 보존한다.
GitHub에는 이미지 대신 분할 목록과 이미지–곡률 연결 CSV를 보관한다.
**학습·이미지 평가는 원본 JPEG가 있어야 재현할 수 있다.**

원본은 지도 Pure Pursuit 자동 teacher의 주행이다. 사람이 직접 운전한 데이터로 표시하지 않는다.
그룹 기준 train:validation:test는 4:1:1이며 인접 프레임을 무작위로 서로 다른 세트에 섞지 않았다.
가운데/3개 카메라 모델은 같은 가운데 카메라 validation/test로 평가했다.
추가 도로 기록은 학습에 사용하지 않았다.

## 학습과 이미지 평가

```bash
# 새 폴더 이름을 사용한다. 아래는 재현 명령이며 현재 가중치를 덮어쓰지 않는다.
bash run.sh train.py --run-dir runs/center_reproduction --camera-mode center --epochs 30 --cache
bash run.sh train.py --run-dir runs/three_reproduction --camera-mode three --epochs 30 --cache
bash run.sh evaluate.py --checkpoint runs/center_reproduction/best.pt --split test
bash run.sh evaluate.py --checkpoint runs/three_reproduction/best.pt --split test
```

기본은 Adam, learning rate 0.001, batch size 64, seed 42, 최대 30 epoch다.
정답 곡률의 절댓값이 0.003 이상인 프레임은 2배 sampling weight를 준다.
validation 오차로 모델을 선택하고 test를 모델 선택에 사용하지 않았다.

추가 학습 모델은 `--curve-basis center`로 보정 전 가운데 teacher 곡률을 선택 기준으로 사용한다.
곧은 도로에서도 좌우 카메라의 복귀 보정값은 0이 아니므로 이 값을 분리한 것이다.
가운데 teacher 조향 역시 도로 형상의 근사 기준이다. 논문 5.1절은 곡선 비중 증가를 설명하지만
고정된 숫자 비율을 제시하지 않는다. 2배는 우리 구현에서 선택했다.

## 실제 MORAI 주행

MORAI `R_KR_PR_K-city_2025`, Ioniq5를 사용한다. 카메라·Ego UDP 설정은 `config.json`과 맞춘다.
다른 제어 프로그램이 실행되지 않는 상태에서 아래 명령을 사용한다.

```bash
# 학습한 곡선에서 카메라 3개 학습 모델을 실행한다.
bash run.sh drive.py --checkpoint checkpoints/three_cameras/best.pt --course D --control --reset --record-screen
```

| 코드 인자 | 도로 | 지도 진행거리 |
|---|---|---|
| `A` | 학습한 직선 | 0~55m |
| `D` | 학습한 곡선 | 650~850m |
| `J` | 좌회전 교차로 | 40~108m |
| `U1` | 학습하지 않은 직선 | 1300~1500m |
| `U2` | 학습하지 않은 곡선과 연결 직선 | 978.337~1210m |

목표 19km/h, |횡오차|>1m이면 시험을 종료한다. 끝점 1m 이내에 도달하면 완주로 판정한다.
정차 출발에서 `--initial-offset 0.3`은 왼쪽 0.3m, `--initial-heading-deg -5`는 오른쪽 5°다.
복귀는 3m 이상 진행한 뒤 |횡오차|≤0.15m와 |방향 오차|≤2°를 1초 유지하는지로 판단한다.
정상 출발은 복귀 여부를 따로 평가하지 않는다. 복귀 후 재이탈할 수도 있어 완주 결과와 함께 본다.
조향은 CNN만 결정한다. 지도는 초기 배치와 평가에, 속도 피드백은 종방향 제어에 사용한다.
실제 창 녹화 중 MORAI 창을 가리거나 이동하지 않는다.

## 결과 파일 구성

공개 결과는 [results/](../results/README.md)의 대표 영상 7개와 읽기 쉬운 표다.
모든 성공·실패 조건은 요약 CSV에 남기고, 같은 조건의 GUI 녹화용 재시험은 중복 합산하지 않는다.
원본 주행별 로그와 전체 영상은 로컬 `runs/`, `github_artifacts/`에 보존한다.

```bash
# 기존 로컬 기록에서 대표 비교 영상과 표만 편집한다. 새 주행은 실행하지 않는다.
bash run.sh -m tools.evaluation.curate_results
```

이 명령은 로컬 전체 아카이브가 있는 작업 PC에서 사용한다.
`build_github_artifacts`는 전체 검수 아카이브를 만드는 보조 도구이며 출력 폴더는 Git에서 제외한다.

## 해석 범위

- 같은 K-City 맵에서 시험했다. 새 도시·날씨 일반화 시험은 아니다.
- 조건별 1회·학습 seed 1개다. 실패 전에 기록한 짧은 구간의 RMSE와 완주 RMSE를 단독 비교하지 않는다.
- 좌우 카메라의 복귀 정답은 2초 뒤 teacher 위치를 이용한 PP-delta 근사다. yaw 영상 증강은 포함하지 않았다.
- 두 모델은 같은 epoch 수로 학습해 카메라 3개 모델의 이미지 수와 optimizer step이 더 많다.
- 교차로 실패에는 새 형상과 목적지 명령 부재가 함께 영향을 줄 수 있다. 모든 교차로 통과가 불가능하다는 근거는 아니다.
- 추가 곡선 비중 실험은 baseline 시험을 본 뒤 같은 도로에서 진행한 탐색 비교다.
- 충돌 ground truth와 원 논문의 autonomy 점수를 별도로 계측하지 않았다.
