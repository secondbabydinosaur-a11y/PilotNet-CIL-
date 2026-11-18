# Conditional Imitation Learning in MORAI K-City

CIL 논문의 명령별 분기 구조를 PyTorch로 구현했다. 가운데 카메라 영상과 현재 속도에서 주행 특징을 추출하고, MGeo 경로에서 얻은 도로 추종·좌회전·직진·우회전 명령으로 조향·가속·제동을 예측할 분기를 고른다. 학습과 MORAI 실시간 추론에 같은 전처리를 사용했다. [CIL 논문 IV-A절](https://arxiv.org/pdf/1710.02410#page=3)

교차로 J 시험에서 직진과 좌회전은 종점 구간에 도달했고, 우회전은 횡오차 기준으로 중단됐다. [교차로 비교 데모](../README.md#교차로-주행-데모) · [영상과 주행 지표](results/README.md#주행-영상)

## 구현과 실행

| 작업 | 코드 |
|---|---|
| 센서 수신과 차량 제어 송신 | [morai.py](morai.py), [config.json](config.json) |
| 정상 주행 및 조향 교란·복귀 수집 | [collect.py](collect.py), [수집 도구](tools/) |
| 시각 맞춤, 복귀 정답 생성, 전처리 | [prepare.py](prepare.py), [dataset.py](dataset.py) |
| 모델 학습과 저장한 가중치 평가 | [model.py](model.py), [train.py](train.py), [evaluate.py](evaluate.py) |
| 진행 명령 선택, MORAI 제어, 영상 기록 | [drive.py](drive.py), [경로](routes/) |

아래 명령은 저장소 루트에서 실행한다. `--describe`는 차량을 움직이지 않고 좌회전 시험 경로와 명령 구간을 출력한다.

```bash
python3 -m pip install -r requirements-train.txt
bash run.sh cil drive.py --course J_left --describe
```

MORAI에서 차량을 제어하려면 센서·포트를 `config.json`에 맞추고 학습한 CIL 가중치를 준비한다. 학습용 원본과 전처리 데이터, 주행 로그, CIL 체크포인트는 로컬에 보관한다. [모델 학습](#초기-모델-학습) · [주행 명령](#morai-주행-시험)

## 카메라 설정

| 역할 | MORAI 센서 | X (m) | Y (m) | Z (m) | Roll | Pitch | Yaw | 수신 포트 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 왼쪽 | Camera_2 | 2.60 | +0.40 | 1.18 | 0° | 2° | +30° | 9096 |
| 가운데 | Camera_4 | 2.60 | 0.00 | 1.18 | 0° | 2° | 0° | 9105 |
| 오른쪽 | Camera_3 | 2.60 | −0.40 | 1.18 | 0° | 2° | −30° | 9097 |

가운데 카메라의 장착 위치·회전·화각은 PilotNet 수집 설정과 같다.
세 카메라 공통 설정은 1280×720, Horizontal FOV 90°, 30Hz, Compression Ratio 90, 일반 RGB다.
좌우 카메라가 각각 왼쪽 앞·오른쪽 앞을 보는지 View에서 확인한다.

좌우 회전 영상의 정답에는 카메라의 횡위치와 Yaw를 모두 고려해야 한다.
수집 중에는 교란을 더한 실제 명령과, 교란된 현재 상태에서 teacher가 계산한 복귀 제어를 따로 기록한다.

## 수집 경로와 시험 구역

![초기 수집 후보 6개; 현재 T5는 점검 전용](assets/training_routes.png)

[경로와 좌표 안내](routes/README.md)

- 학습: 차선이 있는 T1·T2·T3·T4·T6와, 같은 접근로에서 출구가 다른 좌회전·직진·우회전 시범을 사용한다.
- T5는 차선 없는 넓은 도로로, 15초 통신·추종 점검 기록만 보존하며 학습·validation에서 제외한다. 초기 후보 CSV는 보존한다.
- Validation: 별도 V1·V2 경로를 사용한다.
- 기존 시험: U1(s1300~1500m), U2(s978.337~1210m), J(s40~108m 및 세 방향 출구)를 CIL 학습·validation에서 제외한다.
- 기존 연속 시연은 대회 출발점부터 마지막 좌회전 출구 뒤 10 m까지의 `COMPETITION` 경로에서 기록했다. 학습 도로가 포함된다. [Point-to-point 기록](results/README.md#point-to-point-주행)
- PilotNet 학습 도로는 CIL 학습에 사용할 수 있다. J와 실제로 겹치는 부분은 제외한다.

명령은 일반 곡선의 조향 부호가 아니라, 선택한 교차로 링크의 진입·진출 방향으로 부여한다.
일반 도로 곡선에서는 `follow_road`를 유지한다. 교차로 18m 전부터 선택한 명령을 기록한다.
지도 경로와 실제 수집 위치 모두 시험·validation 보호 구역과의 최소 거리를 검사한다.

## 수집과 원본 검사

2026-11-16 기준 [수집 요약](data/collection_summary.json), [세션별 표](data/collection_summary.csv):

| 용도 | 세션 | 유효 주행시간 |
|---|---:|---:|
| 학습 정상 주행 | 8 | 8분 44초 |
| 학습 교란·복귀 | 4 | 6분 16초 |
| Validation | 2 | 2분 35초 |

총 14세션의 원본 이미지 101,745장(카메라별 33,915장)을 모두 해독·해상도 검사했다.
실제 교란이 들어간 사건은 15건이다. 이는 자동 Teacher 원본의 품질 기록이며 CIL 평가 성능이 아니다.
학습 시간은 임의 수집 종료를 위한 제동을 제외한 값이고, 보관한 원본 영상 수에는 그 구간도 포함한다.

```bash
# 차량을 움직이지 않고 실제 센서 수신 확인
bash run.sh cil morai.py

# 실제 주행 없이 경로·명령 구간 확인
bash run.sh cil collect.py --route CENTRAL_LEFT --describe

# 로컬 MORAI를 출발점에 배치한 뒤 정상 Teacher 수집
bash run.sh cil collect.py --route T1 --auto-reset

# 학습 경로의 교란·복귀 시범 수집
bash run.sh cil collect.py --route T4 --auto-reset --noise --seed 31

# 정상·명령·복귀 시범을 수집하고 모든 JPEG 검사; 기존 성공 수집은 재사용
bash run.sh cil tools/collect_all.py --resume
```

수집기는 최고 20km/h, 기본 순항 목표 19km/h로 지도 PP Teacher를 실행한다.
곡선은 감속하며, 실제 상태에서 Teacher를 다시 계산한다. 사람이 운전한 데이터로 표시하지 않는다.
교란은 학습 경로의 적합한 일반 도로에서만 주입하고, Teacher 정답과 교란이 더해진 전송 명령을 분리한다.
시각은 센서 시각과 PC 수신·송신 시각을 각각 기록한다. 센서 중단·추종 이상·저장 실패는 수집을 중단한다.

각 원본 세션은 `images/{left,center,right}/`, `images.csv`, `vehicle.csv`, `commands.csv`,
`session.json`, 실행 코드 사본 `software/`를 포함한다. 검사 후 `quality_report.json`도 생성한다.
명령 CSV의 `teacher_*`가 학습 정답 원본이고 `applied_*`는 실제 전송값이다.
임의로 정한 수집 경로 끝에서의 감속은 `label_valid=0`으로 표시해 일반 주행 정답에서 제외한다.

통과 조건은 모든 JPEG 해독·해상도, 센서 주기·시각 대응, 제어 범위, 추종 오차,
실제 수집 위치와 보호 구역의 거리다. 충돌 이벤트의 정답 로그는 수집하지 않았으므로
이 검사 통과를 무충돌 증명이나 CIL 모델 주행 성능으로 해석하지 않는다.

## 입력·정답 준비 완료

`prepare.py`로 100ms 시간칸마다 유효 영상을 고르고 상태·Teacher를 같은 센서 시각으로 맞췄다.
가장 가까운 상태의 허용 차이는 25ms, 앞뒤 보간 표본의 허용 간격은 50ms다.
명령 전환·임의 경로 끝 정차·가속과 제동 사이의 보간은 제외한다.
시각이 같은 중복 상태는 가장 먼저 대응한 행을 사용하고, 서로 다른 정답의 중복은 제외한다.

| 용도 | 카메라 | 샘플 수 |
|---|---|---:|
| 학습 | 가운데 | 8,988 |
| 학습 | 왼쪽 | 8,939 |
| 학습 | 오른쪽 | 8,966 |
| Validation | 가운데 | 1,549 |

학습 총 26,893샘플이다. Validation은 실제 추론 입력과 같은 가운데 영상만 사용한다.
별도 도로·세션 분할을 유지하며 U1·U2·J·P→Q의 시험 영상을 이 목록에 넣지 않았다.
샘플은 10Hz 시간칸 기준 선택이므로 실제 카메라 프레임 간격은 약간 달라질 수 있다.
원본 영상은 복사하지 않고 `data/prepared/train.csv`, `val.csv`에서 참조한다.
준비 조건·입력 해시는 `data/prepared/summary.json`, 실제 배치 로딩 확인은 `verification.json`에 보관한다.
수집 시점의 `collection_summary.json`과 이후 준비 결과를 구분한다.

좌우 정답은 카메라와 같은 평면 시점을 갖는 가상 차량에서 공통 경로점을 바라본 PP 조향 차이로 보정했다.
목표점은 진행거리 `max(6m, 현재 속도×2초)` 앞이다. 전방 장착 거리 2.6m, 횡위치 ±0.4m와 Yaw ±30°를 모두 반영한다.
원래 가운데 Teacher 조향에 이 차이를 더하고, 제어 범위를 벗어난 보정값은 잘라 넣지 않고 제외한다.
**보정 공식·2초·6m는 이번 구현의 기하학적 근사이며 원 논문에 공개된 보정식을 재현한 것은 아니다.**

`dataset.py`는 원본의 Y=280~720 영역을 잘라 RGB 200×88, 픽셀 0~1로 변환한다.
속도는 20km/h로 나누고, 정답은 곡률이 아닌 정규화 조향·가속·브레이크다.
train에서만 영상 증강을 적용한다. 학습 후 실제 주행에서도 같은 전처리를 사용해야 한다.
시각·정답·기하 보정 단위 검사 20개와 실제 2-worker DataLoader 배치 검사를 통과했다.
Validation 1,549장의 전처리·로딩도 모두 확인했다.

```bash
# 첫 준비 실행. 기존 준비 결과가 있으면 덮어쓰지 않는다.
bash run.sh cil prepare.py

# 보정 설정이나 원본이 달라졌다면 새 버전 폴더로 생성한다.
bash run.sh cil prepare.py --output data/prepared_v2

# Dataset 전처리와 실제 준비 CSV 확인
bash run.sh cil dataset.py --self-test
bash run.sh cil dataset.py --csv data/prepared/val.csv --split val
```

## 초기 모델 학습

`train.py`는 `data/prepared/train.csv`와 `val.csv`를 읽어 현재 `CILModel`을 학습한다.
학습에는 세 카메라의 각 영상을 개별 샘플로 사용하고, validation에는 가운데 영상만 사용한다.
학습 중에만 `dataset.py`의 영상 증강을 적용한다. 원본과 고정 분할은 바꾸지 않는다.

| 항목 | 설정 |
|---|---|
| Optimizer / 학습률 | Adam / 0.0002 |
| Batch | 120: `follow_road`·좌회전·직진·우회전 각 30개 |
| 명령 균형 | 명령별 복원 추출; 1 epoch = 225배치·27,000회 선택 |
| Loss | 세 정규화 제어값의 평균 MSE + 0.05 × 보조 속도 MSE |
| 최대 epoch / 조기 종료 | 30 / 검증 선택 점수 8 epoch 연속 미개선 |
| 모델 선택 | Validation의 명령별 제어 MSE 네 개를 같은 비중으로 평균 |
| Seed | 7 |

Adam·학습률·배치 크기·명령별 균등 배치는 논문을 참고했다.
제어값 세 개의 동등 가중치, 보조 속도 loss 비중, epoch·조기 종료·모델 선택 기준은 이번 구현의 설정이다.
보조 속도 정답은 이미지 시각의 실제 속도/20이며 목표속도가 아니다.
균형 복원 추출이므로 한 epoch에서 모든 학습 이미지가 반드시 한 번씩 선택되는 것은 아니다.

```bash
# 저장소 루트에서 실행한다. 같은 이름의 기존 실행은 덮어쓰지 않는다.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash run.sh cil train.py \
  --epochs 30 --batch-size 120 --lr 0.0002 --aux-weight 0.05 \
  --patience 8 --workers 4 --seed 7 --device cuda --name cil_baseline_v1

# 명령 균형·손실 집계·선택한 branch의 gradient 검사
bash run.sh cil -m unittest discover -s tests -p test_training.py -v
```

`runs/<실행 이름>/settings.json`, `history.csv`, `summary.json`에 조건·지표·완료 상태를 남긴다.
`software/`에는 학습 당시 코드와 센서 설정을 복사하며 설정 파일에 코드·CSV 해시를 기록한다.
`checkpoints/<실행 이름>/best.pt`는 검증 기준 최선, `last.pt`는 마지막 epoch의 상태다.
학습 기록의 제어 MAE는 정규화 조향·페달 기준이다. 실제 차로 추종 오차나 완주율과 구분한다.

2026-11-18(KST) 초기 학습 `cil_baseline_20261117T223956Z`를 완료했다.
약 15분 13초 동안 27 epoch를 학습한 뒤 조기 종료했고, 검증 기준으로 19 epoch의 `best.pt`를 선택했다.
명령별 제어 MSE 평균은 0.005748, 전체 Validation의 정규화 조향 MAE는 0.083502다.
저장 모델을 다시 불러와 가운데 영상 1,549장을 재계산했으며 저장된 검증 지표와 모두 일치했다.
학습 로직 검사 5개도 통과했다. [학습 곡선과 상세 Validation 표](results/README.md).

## 하이퍼파라미터 비교

하이퍼파라미터 비교에서는 `model.py`를 유지하고 `train.py`의 `--lr`, `--weight-decay`,
`--conv-dropout`, `--fc-dropout`, `--epochs`, `--patience`로 학습 조건만 조정한다.
Dropout은 기존 모듈의 확률만 실행 시 변경하며 층·채널·branch·활성화는 바꾸지 않는다.
기본 모델의 19/27 epoch를 가운데 카메라·증강 OFF·평가 모드로 비교했을 때,
학습 데이터의 명령별 평균 MSE는 0.001001→0.000611로 개선됐지만 Validation은 0.005748→0.006909로 커졌다.
이는 후반 학습의 일반화 악화 경향을 보여주며, 두 세트의 도로·복귀 사건 분포 차이는 남는다.
진단 수치는 기본 실행의 `generalization_diagnostic.json`에 보관한다.

비교 후보는 세 개로 한정했다. 데이터 목록·증강·명령별 배치·seed·보조 loss를 고정했다.

| 조건 | 학습률 | Weight decay | Conv / FC Dropout | 최대 epoch / patience |
|---|---:|---:|---|---|
| 기본 | 0.0002 | 0 | 0.20 / 0.50 | 30 / 8 |
| 후보 1: 낮은 학습률 + L2 | 0.0001 | 0.0001 | 0.20 / 0.50 | 40 / 10 |
| 후보 2: 후보 1 + 강한 Dropout | 0.0001 | 0.0001 | 0.25 / 0.60 | 40 / 10 |
| 후보 3: 기본 학습률 + 약한 L2 | 0.0002 | 0.00001 | 0.20 / 0.50 | 40 / 10 |

모델 선택은 기존과 같은 Validation의 명령별 평균 MSE로 했다.
후보 1·3은 수동으로 종료했고, 완료한 epoch에서 얻은 최선의 검증 점수를 사용했다.

| 조건 | 완료 / Best epoch | Validation 명령 평균 MSE | 종료 상태 |
|---|---:|---:|---|
| **기본 모델** | 27 / **19** | **0.005748** | 조기 종료·선택 유지 |
| 후보 1: 낮은 학습률 + L2 | 23 / 17 | 0.005894 | 수동 중단 |
| 후보 2: 강한 Dropout | 16 / 6 | 0.005771 | 조기 종료 |
| 후보 3: 약한 L2 | 10 / 6 | 0.006394 | 수동 중단 |

일부 후보의 전체 조향 MAE는 줄었지만, 고정한 선택 기준인 명령 평균 MSE는 기본 모델보다 낮아지지 않았다.
따라서 **`cil_baseline_20261117T223956Z/best.pt`(19 epoch)를 유지**한다.
이는 같은 Validation을 활용한 모델 선택 결과이며 독립 시험 성능이 아니다.
[비교 그래프·표·상태 기록](results/README.md)을 공개하고 전체 실행·소스 사본은 로컬에 보존한다.

```bash
# 완료된 실행의 기록·체크포인트 일치 확인과 그래프·표 생성
bash run.sh cil tools/summarize_training.py --run cil_baseline_20261117T223956Z

# 완료한 epoch의 결과만 사용해 기본 모델과 후보 세 개 비교
bash run.sh cil tools/compare_training.py \
  --runs cil_baseline_20261117T223956Z cil_tune_lr_l2_20261118 \
         cil_tune_stronger_dropout_20261118 cil_tune_mild_l2_20261118 \
  --output-name cil_hparam_tuning
```

전체 학습 기록·소스 사본·재로딩 검사 결과는 `runs/cil_baseline_20261117T223956Z/`,
가중치는 `checkpoints/cil_baseline_20261117T223956Z/`에 로컬로 보관한다.
수집·준비 검증 JSON은 각 단계 당시의 상태를 담은 기록이며 학습 완료 상태는 해당 실행의 `summary.json`으로 확인한다.

## MORAI 주행 시험

기본 설정의 19번째 epoch 가중치로 미학습 곡선 U2와 교차로 J의 세 방향을 시험했다. 지도는 초기 배치, 진행 명령 선택, 진행률과 횡오차 측정에 사용했다. 조향·가속·제동은 모델이 예측하며, 실행 중에는 속도 상한과 중단 조건을 적용했다. [시험 조건](routes/test/protocol.json) · [영상과 기록](results/README.md)

실제 주행 명령은 센서 설정을 마친 MORAI K-City에서 실행한다. 아래 `--checkpoint` 파일은 직접 학습한 가중치가 있는 경로로 바꾼다. 상대경로는 `cil/` 기준이다.

```bash
# 주행 없이 U2 시험 경로 확인
bash run.sh cil drive.py --course U2 --describe

# 시작점에 차량을 배치하고 U2 주행 녹화
bash run.sh cil drive.py --course U2 --checkpoint checkpoints/my_run/best.pt \
  --control --reset --record-screen --max-cte 7

# 기존 대회 연속 주행 경로 확인
bash run.sh cil drive.py --course COMPETITION --describe
```

횡오차 7 m에 도달하면 중단하고, 그 전에 기준경로의 마지막 1 m 구간에 들어가면 종점 도달로 기록한다. 실행별 설정·차량 상태 로그·녹화는 `runs/closed_loop/`에 저장된다. [Point-to-point 주행 기록](results/README.md#point-to-point-주행)
