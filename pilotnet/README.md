# PilotNet in MORAI K-City

가운데 카메라 이미지 한 장에서 도로 곡률을 예측하고, 예측값을 MORAI 차량의 조향으로 바꾼다. 차량 속도는 별도의 제어기가 맞춘다. 학습에 쓰인 영상과 조향은 기준 경로를 따라가는 Pure Pursuit 제어기로 수집했다. [PilotNet 논문](https://arxiv.org/abs/1604.07316)

## 3카메라 학습과 주행 결과

가운데 카메라로만 학습한 모델과 좌·중·우 카메라로 학습한 모델을 비교했다. 좌우 영상에는 차량이 경로 중심으로 돌아오는 데 필요한 곡률을 정답으로 붙였다. 주행 중에는 두 모델 모두 가운데 영상 한 장씩을 받는다. [수집과 정답 생성](docs/collection.md)

학습에 사용하지 않은 직선과 곡선에서 각각 주행했다. 같은 시험 이미지에서의 예측 오차, 주행 진행률, 곡선 샘플링을 바꾼 모델의 결과를 한 페이지에 정리했다. [주행 영상과 비교 표](results/README.md)

## 실행

저장소 루트에서 아래 명령으로 의존성을 설치하고 모델·데이터 처리 테스트를 실행할 수 있다. 테스트에는 MORAI 연결이 필요 없다.

```bash
python3 -m pip install -r requirements-train.txt
bash run.sh pilotnet -m unittest discover -s tests -p test_learning.py
```

MORAI 주행에는 K-City와 차량·센서 설정이 필요하다. 저장소에는 학습된 가중치와 데이터 목록이 들어 있다. 학습과 기록 이미지 평가는 목록에서 참조하는 원본 JPEG가 있어야 재현할 수 있다. [MORAI 실행 및 재현](docs/reproduction.md) · [가중치 선택](checkpoints/README.md)

## 코드 읽기

- [model.py](model.py)와 [dataset.py](dataset.py)에서 모델 구조, 이미지 처리, 곡률 정답을 확인할 수 있다.
- [train.py](train.py)는 모델을 학습하고, [drive.py](drive.py)는 실시간 영상에서 조향을 계산해 MORAI에 보낸다.
