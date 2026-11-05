# 곡선 프레임 선택 기준 추가 비교

원 논문 5.1절은 곡선 프레임 비중을 높인다고 설명하며 숫자 비율은 제시하지 않는다.
기존 v1도 곡선 정답에 2배 sampling weight를 사용했다. 다만 좌우 복귀 보정도 곡선으로 분류했다.
v2는 보정 전 가운데 teacher 곡률을 기준으로 가중치를 적용한다. **2배는 우리의 선택이며 논문 권장 수치가 아니다.**
입력·CNN·원래 train/val·seed 42·30 epoch·optimizer는 유지했다. 기존 모델은 변경하지 않았다.
가운데 teacher 조향은 도로 곡률의 근사이며 실제 도로 형상 annotation과 동일하지 않다.

원래 train의 곡선 근사 비중 63.8% → 가중 추출 기대 비중 77.9%. 자세한 수치는 [sampling.json](sampling.json).
미학습 구간 teacher는 train/val에 쓰지 않았다. 새 학습 모델 선택도 validation MSE만 사용했다.

closed-loop는 **3개 카메라 학습 모델, 가운데 카메라 추론, 헤딩 0°, 횡위치 0m, 19km/h**로 곡선 U2와 직선 U1을 각 1회만 실행했다.
이미 baseline 시험을 본 뒤 수행한 탐색 비교이므로 처음 보는 도로에 대한 새로운 blind test로 부르지 않는다.

| 버전 | 코스 | 결과 | 진행률 | CTE RMSE (m) |
|---|---|---|---|---|
| v1_target_weight | U2 | intervention_cte | 14.7% | 0.217 |
| v2_center_proxy_weight | U2 | intervention_cte | 12.7% | 0.309 |
| v1_target_weight | U1 | completed | 99.5% | 0.221 |
| v2_center_proxy_weight | U1 | completed | 99.5% | 0.154 |

- [곡선 실제 MORAI 영상](U2/morai_screen.mp4), [곡선 전방 영상](U2/drive.mp4)
- [직선 실제 MORAI 영상](U1/morai_screen.mp4), [직선 전방 영상](U1/drive.mp4)
- [학습 기록](history.csv), [모델](best.pt), [비교 CSV](comparison.csv)

중단 시험의 RMSE는 중단 전 구간만 포함한다. 진행률·완주 여부와 함께 해석한다.
출처: [End to End Learning for Self-Driving Cars, §5.1](https://arxiv.org/pdf/1604.07316).
