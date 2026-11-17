# CIL 데이터

`raw/`는 실제 MORAI에서 받은 원본, `prepared/`는 준비한 입력·정답 목록이다.
원본 이미지를 train·val·test 폴더로 다시 복사하지 않고, 정리된 CSV에서 원본 경로를 참조한다.
원본과 전체 프레임 CSV는 로컬에 보관한다. 이 안내와 작은 수집 요약만 Git 공개 대상으로 둔다.

## 현재 원본

```text
data/raw/<세션 ID>/
├── images/left/                # 원본 JPEG
├── images/center/
├── images/right/
├── images.csv                  # 카메라·이미지 경로·센서 시각·PC 수신 시각
├── vehicle.csv                 # 센서 시각의 차량 상태
├── commands.csv                # 명령·Teacher 정답·교란·실제 전송값
├── session.json                # 실제 설정·수집 상태·소프트웨어 해시
├── quality_report.json         # 수집 후 생성하는 원본 검사 결과
└── software/                   # 해당 실행에 사용한 코드 사본
```

`vehicle.csv`의 `steer_raw_deg`는 차량 상태 피드백이다.
이번 CIL의 제어 정답 원본은 `commands.csv`의 `teacher_steer`, `teacher_throttle`, `teacher_brake`다.
조향 정답은 정규화된 -1~1이며 PilotNet의 곡률 단위와 다르다.
`applied_steer`에는 수집용 교란이 포함되므로 이를 그대로 Teacher 정답으로 사용하지 않는다.

`stamp_ns`는 MORAI 센서 시간이다. `recv_ns`·`sent_ns`는 PC 단조 증가 시간으로,
두 시간축의 숫자를 직접 빼서 영상과 정답을 매칭하지 않는다.
`commands.csv`의 `ego_stamp_ns`는 Teacher가 참조한 차량 상태의 센서 시각이다.

## prepare.py의 출력

`prepare.py`에서 검사를 통과한 세션을 선택하고 이미지 시각의 속도·Teacher 정답을 맞췄다.
학습 26,893샘플(가운데 8,988·왼쪽 8,939·오른쪽 8,966), Validation 가운데 1,549샘플이다.
`prepared/train.csv`, `val.csv`에 아래 열과 추적용 정보를 저장했다.

| 열 | 값 |
|---|---|
| `image_path` | `cil/` 기준 원본 이미지 경로 |
| `camera` | `center`, `left`, `right` |
| `speed_kph` | 이미지 시각의 실제 속도 |
| `command` | 도로 따라가기 0, 좌회전 1, 직진 2, 우회전 3 |
| `steer` | Teacher 조향 -1~1; 좌우 영상은 카메라 위치·Yaw에 따른 복귀 보정 필요 |
| `throttle`, `brake` | 각각 0~1 |
| `label_source` | 가운데 `teacher`, 좌우 보정 완료 후 `camera_recovery` |

추적에 필요한 세션·센서 시각·교란 사건 번호도 추가 열로 보존한다.
임의 수집 종료를 위한 `label_valid=0` 제동과 시각 대응 실패 프레임을 제외하고,
준비 단계에서 100ms 시간칸마다 유효 프레임 한 장을 고른다. 서로 이어진 프레임을 임의로 나눠
train·val에 섞지 않고, 미리 정한 도로·세션 용도를 따른다.

좌우 정답은 전방 장착 거리·횡위치·Yaw를 반영한 가상 차량의 PP 조향 차이로 보정한다.
공통 경로 목표점까지 거리는 `max(6m, 속도×2초)`다. 원 논문에 공개된 공식이 아닌 이번 구현의 근사다.
범위를 벗어난 보정 조향과 명령·페달 전환을 가로지르는 보간은 제외한다.
`summary.json`은 준비 설정·입력 해시·세션별 선택 및 후보 제외 사유를 기록한다.
제외 사유의 숫자는 대체 후보를 포함한 시도 횟수이며 유실된 최종 샘플 수가 아니다.
`verification.json`은 실제 Dataset 배치 로딩과 Validation 전체 로딩 검사 결과다.
새 원본이나 보정 설정으로 다시 준비할 때는 `prepare.py --output data/prepared_v2`처럼 버전을 나눈다.

## dataset.py의 역할

준비 CSV의 한 행을 읽어 이미지 한 장을 Crop·RGB 변환·200×88 크기 변경·0~1 변환한다.
영상 증강은 train에만 적용한다. 좌우 카메라도 각각 한 장씩 별도 샘플이다.
반환값은 이미지 `[3,88,200]`, 현재 속도 `[1]`, 명령 번호, 정답 제어 `[3]`다.
실제 주행에는 가운데 영상 한 장과 현재 속도·명령을 넣는다.

수집 요약의 시간·추종 오차는 자동 Teacher 원본의 품질 기록이며 학습한 CIL의 시험 결과가 아니다.
T5 점검 기록·중단된 수집은 통과 목록에서 제외하고, U1·U2·J·P→Q는 시험용으로 보존한다.
