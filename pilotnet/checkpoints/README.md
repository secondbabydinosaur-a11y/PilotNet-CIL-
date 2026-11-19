# PilotNet 학습 가중치

각 `best.pt`는 검증 이미지의 예측 오차로 선택한 모델이다. 아래 모델은 모두 주행 중에 가운데 카메라 영상 한 장을 입력으로 받는다.

| 가중치 | 학습에 사용한 데이터 |
|---|---|
| [가운데 카메라](center_camera/best.pt) | 가운데 영상 |
| [3카메라](three_cameras/best.pt) | 좌·중·우 영상과 좌우 영상의 복귀 정답 |
| [3카메라·곡선 샘플링 변경](three_cameras_more_curves/best.pt) | 같은 3카메라 학습 데이터에서 곡선 프레임을 판별하는 기준 변경 |

각 폴더의 `settings.json`과 `history.csv`에 학습 설정과 기록이 있다. [모델별 주행 결과](../results/README.md) · [가중치로 MORAI 실행](../docs/reproduction.md#새-도로-morai-주행-closed-loop)
