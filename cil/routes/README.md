# CIL 수집 경로

`train/`, `val/`, `test/`를 물리적인 도로 구역에 따라 나눴다.
지도에서 경로와 명령을 정하고 실제 MORAI Teacher 수집으로 확인한다.
수집 완료·품질 통과 여부는 [실제 수집 기록](../data/collection_completion.json)에 보관한다.
선정에 사용한 지도 파일의 SHA256, 기존 시험 구역과의 분리 검사 및 경로별 출발·종료 좌표는
[plan.json](plan.json)에 기록했다.

| 용도 | 경로 | 내용 | 길이 |
|---|---|---|---:|
| 학습 | [T1](train/T1.csv) | 중앙 도심 우회전·직선 연결 | 406m |
| 학습 | [T2](train/T2.csv) | 남하·좌회전·직진 | 312m |
| 학습 | [T3](train/T3.csv) | 중앙 도심 좌회전·직선 연결 | 421m |
| 학습 | [T4](train/T4.csv) | 중앙 서쪽 연속 곡선·연결도로 | 619m |
| 점검 전용 | [T5 초기 후보](train/T5.csv) | 차선 없는 도로; 학습·validation 제외 | 1,186m |
| 학습 | [T6](train/T6.csv) | 남쪽 도심에서 북쪽 곡선까지 | 621m |
| 학습 | [같은 접근로 좌회전](train/CENTRAL_LEFT.csv) | 북향 접근 후 왼쪽 출구 | 141m |
| 학습 | [같은 접근로 직진](train/CENTRAL_STRAIGHT.csv) | 동일 접근로에서 직진 | 110m |
| 학습 | [같은 접근로 우회전](train/CENTRAL_RIGHT.csv) | 동일 접근로에서 오른쪽 출구 | 117m |
| Validation | [V1](val/V1.csv) | 북쪽 연속 곡선·교차로 | 526m |
| Validation | [V2](val/V2.csv) | 남쪽 외곽 곡선 | 280m |
| 시험 | [P→Q](test/P2Q.csv) | 북서쪽 외곽 미학습 주행 | 636m |

CSV 열은 `s_m,x_m,y_m,z_m`이다.
`s_m`은 각 후보 경로의 시작부터 누적한 거리이며, 기존 대회 경로의 station과 다르다.
XYZ는 MGeo 월드 좌표(m)다. X는 동쪽, Y는 북쪽이며 UTM 52N 원점은 `[302595, 4124145, 0]`이다.
센서 장착 위치를 나타내는 차량 기준 XYZ와 구분한다.
`plan.json`의 `route_csv`는 `cil/` 기준 상대경로다.

U1·U2와 교차로 J의 세 출구는 `plan.json`의 `test_reservations`에 시험 구역으로 보존했다.
이 예약 구역의 경계·여유 점열을 그대로 시험용 주행 경로로 사용하지 않는다.
J 직진·좌회전·우회전의 정확한 시험 실행 경로는 평가 단계에서 준비한다.
학습의 `CENTRAL_*`는 J와 떨어진 별도 교차로다. 동일한 접근 영상에 서로 다른 명령을 부여한다.
T5 CSV는 초기 후보를 보존한 파일이며, 실제 용도는 `plan.json`의 `split=calibration`이다.

그림과 경로는 [CIL 안내](../README.md)에서 함께 볼 수 있다.
