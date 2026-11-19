# Conditional Imitation Learning in MORAI K-City

CIL은 가운데 카메라 영상과 차량 속도를 보고, **도로 추종·직진·좌회전·우회전** 중 주어진 진행 명령에 따라 조향·가속·제동을 예측한다. 이 구현에서는 MORAI의 도로 지도(MGeo)에서 계획한 경로로 진행 명령을 정하고, 논문의 명령별 분기 구조를 PyTorch로 구현했다. [CIL 논문](https://arxiv.org/abs/1710.02410)

[곡선과 교차로 주행 결과](results/README.md#주행-영상) · [최종 point-to-point 영상](results/README.md#point-to-point-주행)

## 구현 흐름

- [morai.py](morai.py)는 UDP로 카메라 영상과 차량 상태를 받고 제어값을 보낸다. [config.json](config.json)에 MORAI 센서와 통신 설정이 있다.
- [collect.py](collect.py)는 지도 경로를 따라가는 Pure Pursuit 제어기로 시범 주행을 수집한다. 같은 접근로에서 출구를 달리한 교차로 주행과, 차량이 경로로 되돌아오는 복귀 주행도 기록했다. [경로 안내](routes/README.md)
- [prepare.py](prepare.py)는 영상·속도·진행 명령·제어 정답을 센서 시각으로 맞춘다. 학습에는 좌·중·우 카메라를 사용하고, 검증과 실시간 주행에는 가운데 카메라를 사용한다. [데이터 구성](data/README.md)
- [model.py](model.py)는 영상과 속도의 특징을 합치고, 명령에 해당하는 출력 분기로 조향·가속·제동을 예측한다. [train.py](train.py)는 네 명령의 샘플을 배치에 같은 수로 넣어 학습한다.

## 설치와 경로 확인

저장소 루트에서 실행한다. `--describe`는 같은 시험 교차로에서 좌회전하는 경로의 설정을 출력한다. 이 명령에는 MORAI 연결이나 가중치가 필요 없다.

```bash
python3 -m pip install -r requirements-train.txt
bash run.sh cil drive.py --course J_left --describe
```

## 초기 모델 학습

원본 카메라 이미지와 CIL 가중치는 로컬에 보관했다. 새로 학습하려면 MORAI K-City에서 카메라·차량 상태·제어 포트를 [config.json](config.json)에 맞춘 뒤 데이터를 수집한다. `--resume`은 로컬에 품질 검사를 통과한 수집 기록이 있으면 재사용한다.

```bash
bash run.sh cil tools/collect_all.py --resume
bash run.sh cil prepare.py
bash run.sh cil train.py --name cil_baseline
```

학습과 검증에는 서로 다른 도로의 데이터를 사용한다. 모델은 네 진행 명령의 검증 제어 MSE를 같은 비중으로 평균한 값으로 선택한다. 결과 파일과 `best.pt`는 각각 `cil/runs/cil_baseline/`, `cil/checkpoints/cil_baseline/`에 저장된다. [학습 곡선과 지표](results/README.md#학습과-검증)

## MORAI 주행 시험

MORAI에서 Ioniq5를 외부 제어 모드와 D 기어로 설정한 뒤 [config.json](config.json)과 같은 센서·통신 환경을 사용한다. `--checkpoint`에는 같은 설정에서 학습한 가중치를 지정한다. 상대경로는 `cil/` 기준이다.

```bash
# 센서가 들어오는지 확인
bash run.sh cil drive.py --check

# 시험 교차로 좌회전 경로에서 주행하고 화면을 녹화
bash run.sh cil drive.py --course J_left --checkpoint checkpoints/cil_baseline/best.pt \
  --control --reset --record-screen
```

[drive.py](drive.py)는 추론 결과를 MORAI에 보내고, 주행 영상과 차량 로그를 `cil/runs/closed_loop/`에 저장한다. 시험에서 사용하는 경로와 출발 조건은 [시험 경로 목록](routes/test/manifest.json)에 있다.
