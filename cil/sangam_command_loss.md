# 상암 후속 시험 · 명령별 조향 손실 비교

모든 명령에 조향 분포를 적용한 첫 실험에서는 우회전 출구 정렬이 좋아진 코스가 있었고, 직진과 연속 주행에서는 MSE가 더 나았다. 첫 모델의 가설·학습 조건·실패 영상을 별도로 기록했다. [첫 상암 실험](sangam_study.md) · [두 코스 영상과 진행률 요약](results/README.md#그래서-어디까지-갔나)

## 추가 본시험

새 두 코스를 정한 뒤 가까운 수집 경로 여섯 곳의 12,869장을 제외했다. MSE와 명령별 모델을 나머지 81,208장으로 각각 다시 학습했다. 평가 경로와 학습 프레임의 최단 거리는 51.5 m였다.

직진과 차선 유지 분기는 분포의 평균 조향에 MSE를 적용했다. 좌·우회전 분기는 앞서 사용한 순서형 조향 손실을 적용했다. 주행에서는 조향값 순서로 이어진 세 구간씩 훑어 확률 합이 가장 큰 묶음을 고르고, 그 안의 예측값을 확률로 가중 평균했다. [최종 조향값을 고르는 숫자 예시](results/README.md#회전할-때-최종-조향값을-고르는-예) 두 모델의 체크포인트는 같은 검증 기준으로 골랐고, 두 평가 코스를 각 모델로 두 번씩 달렸다.

| 모델 | 목적지 도달 | 좌·우회전 정렬 | 직진 정렬 | 경로 근처 진행률 | 첫 횡오차 2.2 m까지 평균 거리 |
|---|---:|---:|---:|---:|---:|
| K-City CIL 그대로 적용 | 0/4 | 0/4 | 0/4 | 43.2% | 17.9 m |
| 상암 MSE CIL | 0/4 | 0/4 | 0/4 | 30.1% | 43.3 m |
| 상암 명령별 조향 손실 CIL | 0/4 | 0/4 | 0/4 | 52.2% | 91.7 m |

코스별 첫 횡오차 2.2 m까지의 평균 거리다.

| 코스 | 기존 K-City | 같은 데이터 MSE | 명령별 모델 |
|---|---:|---:|---:|
| 우회전 → 직진 | 24.2 m | 59.3 m | 99.7 m |
| 직진 → 좌회전 | 11.5 m | 27.3 m | 83.7 m |

아래 표는 목적지 도달 횟수와 경로 근처 평균 진행 거리다.

| 코스 | 거리 | 기존 K-City | 같은 데이터 MSE | 명령별 모델 | 비교 |
|---|---:|---:|---:|---:|---|
| 우회전 → 직진 | 344.8 m | 0/2 · 285.2 m | 0/2 · 178.0 m | 0/2 · 99.5 m | [영상](results/videos/command_loss_wide_1_vs_baseline.mp4) |
| 직진 → 좌회전 | 315.5 m | 0/2 · 11.3 m | 0/2 · 27.0 m | 0/2 · 238.2 m | [영상](results/videos/command_loss_wide_2_vs_baseline.mp4) |

초기 출발 자세와 16 km/h 속도 상한, 횡오차 15 m 중단 기준을 맞췄다. 경로 2.2 m 이내의 최장 지점을 진행 거리로 기록했다. 목적지 도달은 경로 마지막 1 m 구간, 실제 종점 반경 2 m, 횡오차 1.5 m를 모두 적용했다. 신호 준수와 충돌률은 별도 평가로 남겼다. [횡오차 그래프](results/images/sangam_command_loss_tracking.png) · [실행별 CSV](results/metrics/sangam_command_loss_trials.csv) · [해시와 궤적 요약 JSON](results/metrics/sangam_command_loss_results.json)

## 재현

`train.py --model baseline`과 `train.py --model turn_ordinal`로 두 정책을 학습했다. 120장 명령 균형 배치, Adam 학습률 0.0002, seed 7을 쓰고 모델별 14 epoch에서 8,120회 업데이트했다. 원본 프레임과 학습 가중치는 로컬 실험 기록에 있다. 두 정책은 200 × 88 전방 영상·속도·진행 명령을 받는다. 지도는 시범 수집, 출발 배치, 범주형 명령 안내, 평가에 사용한다. [손실과 출력 코드](turn_ordinal_model.py) · [공통 MORAI 설정](routes/experiments/sangam_distribution/config.json) · [두 평가 경로](routes/experiments/sangam_distribution/command_loss_manifest.json)

```bash
bash run.sh cil drive.py \
    --config routes/experiments/sangam_distribution/config.json \
    --manifest routes/experiments/sangam_distribution/command_loss_manifest.json \
    --course command_loss_wide_2 --checkpoint checkpoints/sangam_turn_ordinal_command_loss/best.pt \
    --speed-cap-kph 16 --max-cte 15 --goal-max-cte 1.5 --goal-radius-m 2 \
    --steering-local-mode --control --reset --record-screen
```

## 남은 문제

첫 횡오차 2.2 m 지점은 두 코스의 두 반복에서 모두 뒤로 밀렸다. 우회전 → 직진에서는 분포 모델이 첫 이탈 뒤 기준 경로 근처로 복귀하지 못했고, 기존 K-City 모델은 더 멀리 진행했다. 교차로 출구에서 원하는 차로에 맞춰 복귀하는 동작을 연속 프레임에서 학습하는 것이 다음 실험이다.

진행 명령을 언제 전달할지 불명확한 갈림길에서는 이전 분포 모델이 직진 차로를 유지했다. 이 구간에서는 분기 선택에 필요한 명령 전달 시점을 따로 검증할 필요가 있다. [초기 갈림길 비교 영상](results/videos/sangam_fork_failure.mp4) · [실제 궤적](results/images/sangam_fork_trajectory.png)
