# PilotNet & Conditional Imitation Learning

MORAI K-City에서 이미지 기반 모방학습 주행을 구현하는 프로젝트다.
**PilotNet 구현·실험은 `pilotnet/`, CIL 구현·실험은 `cil/`에 각각 보관한다.**

| 모델 | 현재 상태 | 코드·실험 안내 | 결과 |
|---|---|---|---|
| **PilotNet** | 학습 및 미학습 도로의 open-loop·closed-loop 평가 완료 | [PilotNet 안내](pilotnet/README.md) | [대표 영상 4개와 평가 표](pilotnet/results/README.md) |
| **CIL** | 학습·open-loop·MORAI 5조건 평가 완료, 19 epoch 고정 | [CIL 안내](cil/README.md) | [실제 영상 5개·궤적·평가 표](cil/results/README.md) |

## PilotNet 실제 주행 결과

[![학습하지 않은 직선에서 가운데 카메라 학습과 카메라 3개 학습 비교](pilotnet/results/images/01_new_road_straight.jpg)](pilotnet/results/videos/01_new_road_straight.mp4)

왼쪽은 가운데 카메라로 학습한 모델, 오른쪽은 좌·중·우 카메라 3개로 학습한 모델이다.
두 모델 모두 주행할 때는 가운데 카메라 한 장을 사용한다. 같은 미학습 직선의 정상 출발에서
가운데 카메라 모델은 횡오차 기준을 넘었고, 카메라 3개 모델은 완주했다.

| 영상 | 확인할 내용 |
|---|---|
| [미학습 직선 비교](pilotnet/results/videos/01_new_road_straight.mp4) | 가운데 카메라 학습과 카메라 3개 학습의 주행 비교 |
| [미학습 곡선 비교](pilotnet/results/videos/02_new_road_curve.mp4) | 두 모델 모두 횡오차 기준을 넘어 종료된 사례 |
| [곡선 비중을 높인 추가 학습](pilotnet/results/videos/03_more_curve_training.mp4) | 기존 모델을 보존하고 수행한 탐색 실험 |
| [교차로 사례](pilotnet/results/videos/04_intersection.mp4) | 좌회전 평가 경로에서 직진 방향으로 진행한 사례 |

PilotNet 공개 영상은 목표속도 19km/h, **횡오차 7m 이상 시 중단**하는 조건으로 다시 기록했다.
영상에 해당하는 정상 출발 7회만 재실행했으며, 모델 가중치는 그대로 유지했다.
7m는 추종 경향을 관찰하기 위한 종료 기준이다. 종점 도달을 차선 유지나 안전성의 증명으로 해석하지 않는다.
교차로 J는 PilotNet 학습 구간과 일부 겹치므로 미학습 도로 정량 평가와 구분한다.
실험 조건과 한계는 [PilotNet 상세 안내](pilotnet/README.md)에 정리했다.

## 저장소 구조

```text
.
├── README.md                    # 두 모델의 전체 안내와 대표 결과
├── requirements.txt             # 공통 수집·통신 환경
├── requirements-train.txt       # PyTorch·학습·평가 환경
├── run.sh                       # 실행할 모델을 명시하는 공통 실행기
│
├── pilotnet/                    # 기존 PilotNet 구현과 실험 전체
│   ├── README.md
│   ├── config.json
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── drive.py
│   ├── learning/                # PilotNet 학습·실시간 주행 보조 코드
│   ├── tools/                   # PilotNet 수집·정답 생성·평가 도구
│   ├── tests/
│   ├── dataset/                 # PilotNet 분할·정답 CSV와 로컬 원본
│   ├── checkpoints/             # 공개한 학습 모델 3개
│   ├── results/                 # 기존 대표 영상 4개·그림·평가 표
│   ├── docs/                    # 수집 절차·고정 시험 조건·재현 방법
│   └── assets/                  # PilotNet 경로와 센서 API 자료
│
└── cil/                         # CIL 구현과 실험을 위한 별도 공간
    ├── README.md
    ├── assets/training_routes.png
    ├── config.json              # CIL 센서·명령·통신 설정
    ├── morai.py                 # 카메라·Ego 수신과 제어 송신
    ├── collect.py               # Teacher 정상·교란·복귀 수집
    ├── prepare.py               # 시각 매칭·좌우 정답 보정·고정 분할 CSV 생성
    ├── dataset.py               # 학습 중 영상 전처리·증강·텐서 로딩
    ├── model.py                 # 이미지·현재 속도와 명령별 제어 branch
    ├── train.py                 # 명령별 균형 배치·학습·validation·모델 저장
    ├── evaluate.py              # 저장 모델의 이미지별 예측·Validation 오차 분석
    ├── tools/                   # 수집 검사·학습 요약·진단·하이퍼파라미터 비교
    ├── tests/                   # 통신·수집·정답 준비·학습 로직 검사
    ├── routes/                  # 학습·validation·시험 경로
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── data/                    # CIL 전용 원본·정답 데이터 (로컬)
    ├── runs/                    # CIL 실행·학습 기록 (로컬)
    ├── checkpoints/             # CIL 학습 가중치 (로컬)
    └── results/                 # 실제 실험 후 선정할 공개 결과
```

CIL은 초기 학습과 하이퍼파라미터 비교를 마쳤고, 기존 Baseline의 19 epoch 모델을 유지한다.
비교 후보 두 개는 사용자 요청으로 중지했으며 완료한 epoch의 기록만 남겼다.
다음은 선택 모델의 Validation 재평가다. 미학습 시험 구역과 MORAI 실제 주행은 아직 평가하지 않았다.
파일별 역할과 작성 순서는 [CIL 안내](cil/README.md)에 있다.
원본 센서 기록과 전체 실행 아카이브는 로컬에 보관하고, GitHub에는 코드·경로·선정한 결과를 올린다.

## 실행 위치

저장소 루트에서 실행할 모델 이름을 먼저 지정한다.

```bash
# PilotNet 데이터·모델 입출력 확인
bash run.sh pilotnet check_model.py

# PilotNet 기존 테스트
bash run.sh pilotnet -m unittest discover -s tests -v
```

공통 실행기는 해당 모델 폴더로 이동한 뒤 Python을 실행한다.
직접 `pilotnet/` 폴더로 이동해서 기존 `bash run.sh ...` 명령을 사용해도 된다.
CIL 수신 점검은 `bash run.sh cil morai.py`로 실행한다. [PilotNet 실행·재현 안내](pilotnet/docs/reproduction.md)

## 논문

- PilotNet: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- CIL: [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)

원 논문의 구조와 절차를 참고하며, MORAI 환경에 맞게 선택한 설정과 실험 범위는 모델별 안내에 기록한다.
