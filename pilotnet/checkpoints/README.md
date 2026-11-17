# 학습된 모델

| 폴더 | 학습 조건 | 실제 주행 입력 |
|---|---|---|
| `center_camera` | 가운데 카메라로 학습 | 가운데 카메라 1장 |
| `three_cameras` | 좌·중·우 카메라 3개로 학습 | 가운데 카메라 1장 |
| `three_cameras_more_curves` | 카메라 3개, 곡선 프레임 선택 기준 변경 | 가운데 카메라 1장 |

각 폴더의 `best.pt`는 validation 오차로 선택한 가중치다.
`settings.json`은 실제 학습 조건, `history.csv`와 `learning_curve.png`는 학습 경과다.
기존 모델을 덮어쓰지 않고 별도 모델로 추가 실험했다. 이번 저장소 정리에서는 가중치를 변경하지 않았다.

모델 구조는 [model.py](../model.py), 실행 절차는 [재현 안내](../docs/reproduction.md)에 있다.
