# 실행과 재현

## 환경

`pilotnet/` 폴더에서 실행한다. 확인한 환경은 Python 3.8, PyTorch 2.4.1+cu121,
RTX 4080 SUPER다. 의존성은 `requirements.txt`, `requirements-train.txt`에 있다.
`run.sh`는 이 PC의 CUDA 패키지 경로 충돌을 해당 프로세스 안에서 처리한다.

```bash
cd pilotnet  # 저장소 루트에서 시작
bash run.sh -m unittest discover -s tests -v
bash run.sh -m tools.evaluation.restore_artifacts
```

복원 명령은 `checkpoints/`의 baseline 가중치와 `docs/evaluation_protocol.json`을
기존 실행 코드가 사용하는 로컬 `runs/` 경로에 복사한다. 학습이나 주행은 실행하지 않는다.

## 데이터와 시험 구간

수집 절차와 센서 포트는 [수집 안내](collection.md)에 있다.
기존 실행 설정·메타데이터의 절대경로는 수집 당시의 기록이며 수정하지 않았다.
현재 데이터 경로는 `pilotnet/` 기준 상대경로로 해석한다.

원본 JPEG는 로컬 `dataset/raw/`에, 시험 전용 새 도로 기록은 `dataset/unseen_raw/`에 보존한다.
GitHub에는 이미지 대신 분할 목록과 이미지–곡률 연결 CSV를 보관한다.
**학습·이미지 평가는 원본 JPEG가 있어야 재현할 수 있다.**

원본은 지도 Pure Pursuit 자동 teacher의 주행이다. 사람이 직접 운전한 데이터로 표시하지 않는다.
학습용 데이터는 직선 s=0~65m, 곡선 s=640~866m에서 수집했다.
이 수집 그룹을 4:1:1로 나눴으며, 인접 프레임을 무작위로 서로 다른 세트에 섞지 않았다.
기존 `dataset/paired/test.csv`는 같은 A/D 도로의 다른 수집 그룹이므로 내부 점검용이다.
파일 이름과 원본 CSV는 기존 체크포인트의 데이터 계약을 유지하기 위해 보존한다.

**공개 시험 데이터는 별도 수집한 `dataset/unseen_paired/test.csv`다.**
학습에 사용하지 않은 직선 U1과 곡선 U2의 가운데 카메라 이미지 830장을 두 모델에 동일하게 입력한다.
이 이미지는 train/validation에 넣지 않았다. 시험 전에 고정한 baseline 모델로 평가했다.

| 시험 인자 | 도로 | K-City 기준 경로의 진행거리 |
|---|---|---|
| `U1` | 새 직선 200m | 1300~1500m |
| `U2` | 새 곡선과 연결 직선 약 232m | 978.337~1210m |

모델·기존 분할의 SHA256과 학습/시험 구간의 비중첩은 `verify_frozen()`으로 확인한다.
[고정 당시 프로토콜](evaluation_protocol.json)은 과거 내부 점검 조건도 포함한 원본 기록이다.
새 도로 정량 평가는 위 U1/U2에 한정한다. 교차로 J(s40~108m) 사례 영상은 별도로 추가했다.
J는 학습 수집 구간과 일부 겹치므로 새 도로 일반화 시험과 구분한다. [도로 지도](../results/images/course_map.png)

## 학습

```bash
# 새 폴더 이름을 사용한다. 아래는 재현 명령이며 현재 가중치를 덮어쓰지 않는다.
bash run.sh train.py --run-dir runs/center_reproduction --camera-mode center --epochs 30 --cache
bash run.sh train.py --run-dir runs/three_reproduction --camera-mode three --epochs 30 --cache
```

기본은 Adam, learning rate 0.001, batch size 64, seed 42, 최대 30 epoch다.
정답 곡률의 절댓값이 0.003 이상인 프레임은 2배 sampling weight를 준다.
같은 가운데 카메라 validation 오차로 모델을 선택하고, 시험 결과는 baseline 모델 선택에 사용하지 않았다.

추가 학습 모델은 `--curve-basis center`로 보정 전 가운데 teacher 곡률을 선택 기준으로 사용한다.
곧은 도로에서도 좌우 카메라의 복귀 보정값은 0이 아니므로 이 값을 분리한 것이다.
가운데 teacher 조향 역시 도로 형상의 근사 기준이다. 논문 5.1절은 곡선 비중 증가를 설명하지만
고정된 숫자 비율을 제시하지 않는다. 2배는 우리 구현에서 선택했다.

## 새 도로 이미지 평가 (open-loop)

```bash
# 원본 시험 JPEG와 dataset/unseen_sessions.json이 있는 작업 PC에서 실행한다.
# 고정된 가운데 카메라 / 카메라 3개 baseline 모델을 함께 평가한다.
bash run.sh -m tools.evaluation.evaluate_unseen
```

이 명령은 시험 전용 세션의 품질과 학습 데이터와의 분리를 확인하고,
`dataset/unseen_paired/test.csv`와 각 baseline 모델의 `open_loop_unseen/` 결과를 만든다.
동일한 모델/데이터의 결과가 이미 있으면 재사용한다. 가중치를 업데이트하지 않는다.
`evaluate.py`는 같은 학습 도로의 내부 검증 도구이며 기본 split은 `val`이다.
`--split test`를 명시해도 과거 A/D 기록을 점검하는 것이며 공개 새 도로 평가와 구분한다.

## 새 도로 MORAI 주행 (closed-loop)

MORAI `R_KR_PR_K-city_2025`, Ioniq5를 사용한다. 카메라·Ego UDP 설정은 `config.json`과 맞춘다.
다른 제어 프로그램이 실행되지 않는 상태에서 아래 명령을 사용한다.

```bash
# 현재 공개 영상과 같은 7m 종료 조건으로 새 직선을 실행한다.
bash run.sh drive.py --checkpoint checkpoints/three_cameras/best.pt --course U1 \
  --control --reset --record-screen --initial-offset 0 --initial-heading-deg 0 \
  --speed 19 --max-cte 7 --seconds 90
```

현재 공개 영상에 필요한 실행은 아래 **7회**다. 기존 모델 세 개를 그대로 사용하며 재학습하지 않는다.

| 고정 체크포인트 | 코스 | 실행 수 |
|---|---|---:|
| `checkpoints/center_camera/best.pt` | U1, U2 | 2 |
| `checkpoints/three_cameras/best.pt` | U1, U2, J | 3 |
| `checkpoints/three_cameras_more_curves/best.pt` | U2, U1 | 2 |

모두 정차 상태의 정상 위치·헤딩 0°에서 출발하며 목표속도는 19km/h다.
`--max-cte 7`로 절대 횡오차가 7m 이상이면 종료하고, 헤딩 오차만으로는 종료하지 않는다.
횡오차 종료 조건을 먼저 검사한 뒤 기준 경로의 진행거리가 종점 1m 전까지 도달하면 완주로 판정한다.
7m는 이탈 이후 동작을 관찰하기 위한 종료 기준이며 차선 유지나 안전 기준을 뜻하지 않는다.
J는 같은 s40~108m 좌회전 목표를 다시 실행한 사례이며 새 도로 일반화 시험과 구분한다.

기존 `run_matrix`는 과거 여러 출발 조건을 실행하는 보조 도구다. 과거 1m 종료·다섯 출발 조건의
20회 기록은 로컬 아카이브에 보존하지만 현재 공개 영상·표에는 포함하지 않는다. 이번 교체를 위해 추가로 20회 실행하지 않았다.
조향은 CNN만 결정한다. 지도는 초기 배치와 평가에, 속도 피드백은 종방향 제어에 사용한다.
실제 창 녹화 중 MORAI 창을 가리거나 이동하지 않는다.

## 결과 파일 구성

공개 결과는 [results/](../results/README.md)의 기존 새 도로 영상 3개, 추가 교차로 사례 영상 1개와 평가 표다.
현재 `driving_trials.csv`는 두 baseline 모델 × U1/U2의 정상 출발 4행이다.
`curve_sampling_comparison.csv`는 기존 카메라 3개 모델과 곡선 비중 증가 모델 × U1/U2의 4행이며,
기존 모델 2회는 앞 표와 같은 실행이다. `intersection_example.csv`는 J 1행이다.
이번 7회 원본과 프레임별 로그는 로컬 `runs/closed_loop_cte7_public/`에, 과거 기록은 기존 `runs/`, `github_artifacts/`에 보존한다.

대표 영상 4개의 파일명과 비교 배치는 유지하며 새 원본의 실제 결과·종료 기준·출처 해시로 갱신했다.
기존 `tools.evaluation.curate_results`는 과거 아카이브를 읽는 도구이므로 현재 공개 결과를 복원하는 명령으로 사용하지 않는다.
이전에 공개했다가 목록에서 제외된 영상도 로컬 `runs/` 아래에 보존한다.
`build_github_artifacts`는 전체 검수 아카이브를 만드는 보조 도구이며 출력 폴더는 Git에서 제외한다.
과거 학습 도로 결과를 만드는 `make_results`도 기본 출력이 로컬 `runs/legacy_results/`다.

## 해석 범위

- 같은 K-City 맵에서 시험했다. 새 도시·날씨 일반화 시험은 아니다.
- 조건별 1회·학습 seed 1개다. 실패 전에 기록한 짧은 구간의 RMSE와 완주 RMSE를 단독 비교하지 않는다.
- 좌우 카메라의 복귀 정답은 2초 뒤 teacher 위치를 이용한 PP-delta 근사다. yaw 영상 증강은 포함하지 않았다.
- 두 모델은 같은 epoch 수로 학습해 카메라 3개 모델의 이미지 수와 optimizer step이 더 많다.
- 추가 곡선 비중 실험은 baseline 시험을 본 뒤 같은 도로에서 진행한 탐색 비교다.
- 충돌 ground truth와 원 논문의 autonomy 점수를 별도로 계측하지 않았다.
