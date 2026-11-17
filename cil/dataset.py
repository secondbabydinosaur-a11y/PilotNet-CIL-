###### 여기서는 수집한 데이터를 실제 CIL 모델에 넣을 텐서로 바꾸는 파일임.(이미지 crop, data augmentation등등 여기서 진행)
##### 그래서 입력: 이미지+현재속도+명령 // 정답: teacher의 조향+ 가속 + 브레이크



import argparse
import csv #csv 읽을떄 사용
from pathlib import Path  #운영체제에 맞게 파일 경로 처리

import cv2 #opencv
import numpy as np # 배열 계산에 사용
import torch #numpy 배열을 pytorch 텐서로 변환
from torch.utils.data import Dataset

BASE=Path(__file__).resolve().parent #이 파일이 들어있는 cil 폴더다.
CROP= (0,280,1280,720) #왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y(즉, y축으로만 720->440으로 crop)
IMAGE_SIZE=(200,88)
SPEED_SCALE_KPH=20.0 
CAMERAS = ("center", "left", "right")  # 사용할 수 있는 카메라 역할이다.

REQUIRED = {
    "image_path",
    "camera",
    "speed_kph",
    "command",
    "steer",
    "throttle",
    "brake",
    "label_source",
}  # 준비 CSV에 필요한 열 이름이다.


def augment_image(image): #이미지 전처리 함수!!!!!!!

    image=image.copy() #image 직접 수정 x, copy하고 진행
    height,width=image.shape[:2]

    #일부 이미지에 Gaussian blur를 적용(이미지 흐릿....)
    if np.random.random() < 0.20:
        kernel=int(np.random.choice([3,5])) #홀수 크기의 커널을 선택
        sigma=float(np.random.uniform(0.1,1.2)) #흐림의 세기를 선택(시그마 커질수록 이미지 더 흐려짐)
        image=cv2.GaussianBlur(
            image, (kernel, kernel), sigmaX=sigma
        ) #image 변수에 가우시안 블러 저장됨.

    #일부 이미지에 Gaussian noise 적용(이미지에 잡음 혹은 자글자글한 주름같은거...)
    if np.random.random() < 0.20:
        sigma=float(np.random.uniform(0.005, 0.025)) #noise의 표준편차
        noise= np.random.normal(
            0.0, sigma, image.shape
        ).astype(np.float32) #픽셀 별로 서로 다른 잡음을 만듦
        image += noise #이미지에 noise 추가.

    #일부 이미지에 salt and pepper noise 적용(카메라에 빗물 튀는것처럼 이미지 augmentation)
    if np.random.random() < 0.10:
        amount=float(np.random.uniform(0.001, 0.005))
        mask = np.random.random((height,width)) 
        image[mask < amount / 2.0] = 0.0
        image[mask > 1.0 - amount / 2.0] = 1.0


    if np.random.random() < 0.15:  # 일부 영상에 작은 가림 영역을 만든다.
        for _ in range(int(np.random.randint(1, 3))):  # 한두 개 영역만 가린다.
            block_h = int(np.random.randint(3, 10))  # 가림 영역의 높이다.
            block_w = int(np.random.randint(4, 17))  # 가림 영역의 너비다.
            top = int(np.random.randint(0, height - block_h + 1))  # 세로 위치다.
            left = int(np.random.randint(0, width - block_w + 1))  # 가로 위치다.
            image[
                top:top + block_h, left:left + block_w
            ] = 0.0  # 해당 영역을 검게 가리는 영상 dropout이다.

    return np.clip(image, 0.0, 1.0).astype(np.float32)  # 잡음으로 벗어난 범위를 제한한다.


def preprocess_image(bgr, augment=False): #학습과 실제 주행에서 공통으로 사용할 이미지 전처리...(텐서형태로 실제 신경망에..)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3: # Defensive Programming(이렇게 안하면 어디서 에러 났는지 모름)
        raise ValueError("컬러 영상 안 들어왔음...") # 빠르게 에러를 찾기 위해 이렇게 코드 사이사이 Defensive programming 박아놔야함
    
    if bgr.shape[:2] != (720,1280):
        raise ValueError(f"expected 1280*720 image, got {bgr.shape}")
    
    x1,y1,x2,y2 = CROP #앞에서 정의한 crop 좌표 가져옴
    cropped =bgr[y1:y2, x1:x2] 

    resized=cv2.resize(
        cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA # crop한 영상의 크기를 200x88로 변경
    )

    rgb=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) #Opencv의 BGR을 RGB로 바꾼다(opencv에서 이미지 불러올때:BGR, pytorch:RGB가 표준)
    image=rgb.astype(np.float32) / 255.0 

    if augment:
        image=augment_image(image)

    chw=np.ascontiguousarray(image.transpose(2,0,1)) #(height,width,channel) -> (channel,height,width) 순서로 변환
    return torch.from_numpy(chw) #numpy 배열에서 텐서로 반환


#이미지,속도,명령을 입력 / teacher의 제어값을 정답으로 제공

class CILDataset(Dataset):
    """이미지·속도·명령을 입력으로, Teacher 제어값을 정답으로 제공한다."""

    def __init__(
        self,
        csv_path,
        split="train",
        cameras=CAMERAS,
        augment=True,
        root_dir=BASE,
    ):
        if split not in ("train", "val", "test"):  # 데이터 용도 이름을 확인한다.
            raise ValueError("split must be train, val or test")

        self.root = Path(root_dir).resolve()  # 이미지 상대경로를 해석할 기준 폴더다.
        csv_path = Path(csv_path)  # CSV 경로를 Path 객체로 바꾼다.

        if not csv_path.is_absolute():  # 상대경로로 받았다면
            csv_path = self.root / csv_path  # cil 폴더를 기준으로 찾는다.

        selected = set(cameras)  # 이번 실험에서 사용할 카메라들의 집합이다.
        if not selected or not selected.issubset(CAMERAS):  # 이름이 올바른지 확인한다.
            raise ValueError(f"cameras must be selected from {CAMERAS}")

        self.augment = bool(augment and split == "train")  # Val·Test에는 항상 증강을 끈다.
        self.samples = []  # 영상 자체가 아니라 경로·속도·정답 정보만 저장한다.

        with csv_path.open(newline="", encoding="utf-8") as stream:  # CSV를 연다.
            reader = csv.DictReader(stream)  # 각 행을 열 이름으로 읽는다.
            missing = REQUIRED - set(reader.fieldnames or [])  # 필요한 열이 빠졌는지 확인한다.

            if missing:
                raise ValueError(f"missing CSV columns: {sorted(missing)}")

            for line, row in enumerate(reader, start=2):  # 첫 줄은 제목이므로 2행부터다.
                camera = row["camera"]  # 이미지가 어느 카메라에서 왔는지 읽는다.

                if camera not in CAMERAS:  # 알 수 없는 카메라 이름은 거부한다.
                    raise ValueError(f"line {line}: unknown camera {camera}")

                if camera not in selected:  # 이번 실험에서 사용하지 않는 카메라는
                    continue  # 데이터셋에 넣지 않는다.

                source = (
                    "teacher" if camera == "center" else "camera_recovery"
                )  # 좌우 영상에는 카메라 위치·Yaw에 따른 정답 보정이 필요하다.

                if row["label_source"] != source:  # 보정하지 않은 좌우 정답을 막는다.
                    raise ValueError(
                        f"line {line}: {camera} requires label_source={source}"
                    )

                speed = float(row["speed_kph"])  # 이미지 시각에 맞춘 실제 속도다.
                command = int(row["command"])  # 사용할 명령 branch 번호다.

                target = np.array(
                    [row["steer"], row["throttle"], row["brake"]],
                    dtype=np.float32,
                )  # 정답 순서는 조향·가속·브레이크다.

                if not np.isfinite(speed) or speed < 0 or not np.isfinite(target).all():
                    raise ValueError(f"line {line}: invalid speed or target")

                if command not in (0, 1, 2, 3):  # 허용한 명령 번호인지 확인한다.
                    raise ValueError(f"line {line}: invalid command {command}")

                if (
                    abs(target[0]) > 1
                    or np.any(target[1:] < 0)
                    or np.any(target[1:] > 1)
                ):  # 조향은 -1~1, 가속·브레이크는 0~1이어야 한다.
                    raise ValueError(f"line {line}: target outside control range")

                if target[1] > 0 and target[2] > 0:  # 현재 Teacher는 동시 페달을 쓰지 않는다.
                    raise ValueError(
                        f"line {line}: throttle and brake are both positive"
                    )

                image_path = Path(row["image_path"])  # 원본 이미지 경로를 읽는다.

                if not image_path.is_absolute():  # 상대경로라면
                    image_path = self.root / image_path  # cil 폴더 기준으로 해석한다.

                if not image_path.is_file():  # 파일이 없으면 학습 전에 알려준다.
                    raise FileNotFoundError(f"line {line}: {image_path}")

                self.samples.append(
                    (image_path, camera, speed, command, target)
                )  # 필요한 정보만 메모리에 저장한다.

        if not self.samples:  # 선택한 카메라의 데이터가 하나도 없다면
            raise ValueError("no samples after camera selection")

    def __len__(self):  # DataLoader가 전체 데이터 개수를 확인할 때 호출한다.
        return len(self.samples)  # 사용할 수 있는 샘플 수를 반환한다.

    def __getitem__(self, index):  # DataLoader가 샘플 하나를 요청할 때 호출한다.
        path, camera, speed, command, target = self.samples[index]  # 해당 샘플 정보다.
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # 필요한 순간에만 이미지를 읽는다.

        if bgr is None:  # 파일이 손상되어 영상으로 읽히지 않는다면
            raise RuntimeError(f"cannot decode image: {path}")

        return {
            "image": preprocess_image(
                bgr, augment=self.augment
            ),  # float32 [3, 88, 200], RGB 0~1이다.

            "speed": torch.tensor(
                [speed / SPEED_SCALE_KPH], dtype=torch.float32
            ),  # float32 [1], 현재 속도를 기준값 20km/h로 나눈다.

            "command": torch.tensor(
                command, dtype=torch.long
            ),  # int64 스칼라로 사용할 branch 번호다.

            "target": torch.from_numpy(
                target.copy()
            ),  # float32 [3], 조향·가속·브레이크 정답이다.

            "camera": camera,  # 모델에 넣는 값이 아니라 샘플 확인용 정보다.
            "image_path": str(path),  # 오류 분석 때 원본을 찾기 위한 정보다.
        }
    
def main():
    """준비 CSV가 없어도 전처리부터 점검할 수 있게 한다."""

    parser = argparse.ArgumentParser()  # 실행 옵션을 정의한다.
    parser.add_argument("--self-test", action="store_true")  # 가상 영상으로 점검한다.
    parser.add_argument("--csv")  # 나중에 준비한 CSV를 지정한다.

    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )  # 데이터 용도를 선택한다.

    parser.add_argument(
        "--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS)
    )  # 사용할 카메라를 선택한다.

    args = parser.parse_args()  # 실행 옵션을 읽는다.

    if args.self_test:
        bgr = np.full(
            (720, 1280, 3), (10, 20, 30), dtype=np.uint8
        )  # BGR 순서가 구별되는 가상 영상을 만든다.

        image = preprocess_image(bgr)  # 증강 없이 전처리한다.

        assert image.shape == (3, 88, 200)  # 모델 입력 크기를 확인한다.
        assert image.dtype == torch.float32  # 자료형을 확인한다.

        assert torch.allclose(
            image[:, 0, 0], torch.tensor([30, 20, 10]) / 255.0
        )  # RGB 변환과 0~1 정규화를 확인한다.

        print("SELF-TEST PASS:", tuple(image.shape), image.dtype, "RGB, 0~1")

    if args.csv:
        dataset = CILDataset(
            args.csv, split=args.split, cameras=args.cameras
        )  # 준비된 CSV를 읽는다.

        sample = dataset[0]  # 첫 번째 샘플을 불러온다.
        print("samples:", len(dataset))  # 전체 샘플 개수를 출력한다.

        print({
            key: (tuple(value.shape), str(value.dtype))
            if torch.is_tensor(value) else value
            for key, value in sample.items()
        })  # 입력과 정답의 형태를 출력한다.

    if not args.self_test and not args.csv:
        parser.error("use --self-test or --csv data/prepared/train.csv")


if __name__ == "__main__":  # 다른 파일에서 import할 때는 점검을 실행하지 않는다.
    main()  # 직접 실행했을 때만 점검한다.



        
        
        

