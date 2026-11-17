from pathlib import Path # 운영체제 걱정 없이 컴퓨터 안에 폴더경로를 쉽게 가져올 수 있게 도와주는 도구

import cv2  # opencv 라이브러리 가져옴
import numpy as np 
import torch
from torch.utils.data import Dataset #pytorch에서 데이터 다루는 tool

from data_utils import read_json, read_csv, preprocess

ROOT=Path(__file__).resolve().parent  #프로젝트 루트 경로 지정.
                                      #터미널 위치 상관없이 데이터 경로 찾게 해줌...!!! 이거 없으면 이상한 경로에서 터미널열고 명령어 실행하면 에러 발생.
                                      # ex) src에서 코드 실행해야하는데 src/data 라는 폴더에서 코드 실행해도 이거 있으면 실행됨


class PilotNetDataset(Dataset):  #dataset 클래스
    def __init__(self, split, paired_dir=None, allow_side=False):
        if split not in ("train", "val", "test"):     # 예외처리때문에 쓰는것. split 오타로 잘못된 CSV 경로를 찾는 일을 미리 방지하는 거임
            raise ValueError("split must be train, val or test")
        
        config = read_json(ROOT / "config.json")  #config.json 파일 읽기
        self.settings = config["preprocessing"] # config 파일에서 preprocessing 파트(crop 어느사이즈로 할지 색상 어떻게 바꿀지 등등 저장돼있음)를 가져옴

        if (self.settings["height"], self.settings["width"]) != (66, 200): #나중에 config 파일 만질수도 있으니깐 preproecessing 관련 수치들 만지면 오류메세지 뜨도록 방어 코드 심은거
            raise ValueError("PilotNet input size must be 66*200")
        
        csv_path= (Path(paired_dir) if paired_dir is not None else ROOT / "dataset" / "paired") / f"{split}.csv" #dataset/paired/{split}.csv 차례대로 폴더 안에 들어감. split은 아까 나눈 train,val,test
        rows=read_csv(csv_path)   #지정한 split의 CSV 하나만 읽음. train,val,test는 각각 따로 객체를 만들어서 사용함

        rows = [
            row for row in rows
            if row["speed_eligible"]=="1"  #미리 train,val,test.csv에 속도 10~30km면 speed_eligible=1이라고 기록함
                                           # 그니깐 속도 10~30km인 데이터 새로운 목록(rows)로 담음
        ]

        rows.sort(      #위에서 지정한 row들을 촬영시각 순서로 정렬시킴
            key=lambda row: ( 
                row["session_id"],
                int(row["stamp_ns"]),
            )
        )

        period_ns=round(
            1_000_000_000 / self.settings["sample_hz"] #약 30Hz 원본에서 10Hz로 선택하려고 한 구간 길이(0.1초)를 나노초로 계산하는 거임
        )

        origins={}
        used_bins=set()
        self.rows=[]

        for row in rows:
            if row["camera"] not in (("center", "left", "right") if allow_side else ("center",)):
                raise ValueError(
                    "This baseline expects center-camera labels"
                )
            
            session = row["session_id"] #이 사진이 어떤 주행(수집 1회)에 속하는지. 코스 종류는 별도 course 필드에 있음
            stamp=int(row["stamp_ns"])  #이 사진의 촬영시각(stamp_ns)을 나노초(ns) 단위로 가져옴

            origin = origins.setdefault(session, stamp) #처음 보는 세션이면 첫 유효 프레임의 촬영시각을 기준점으로 등록하고, 이후에는 그 기준점을 계속 사용함

            time_bin=(stamp - origin) // period_ns #시작점(origin)을 기준으로 몇번째 0.1초 구간인지 계산함
            key = (session, row["camera"], time_bin)

            if key in used_bins: #만약에 같은 카메라의 해당 시간 구간에 뽑아둔 사진이 있으면 건너뜀
                continue

            used_bins.add(key)
            self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No usable samples in {split}")

            ### 여기까지 아는 거. 지정한 split에서 주행별 0.1초 구간마다 최대 한 장만 선택목록에 넣음. 원본 파일은 그대로 남음 ###


    def __len__(self): #전체 데이터셋 크기 나오게 함
        return len(self.rows)
    
    def __getitem__(self,  index): 
        row=self.rows[index] #전체목록(rows) 중에서 pytorch가 지정한 index번째 데이터 한줄 집어옴
        image_path=ROOT / row["image_path"] #그 줄에 적혀 있는 상대경로와 전체 프로젝트 ROOT를 합쳐서 절대적인 전체 주소를 완성함

        image=cv2.imread( #이미지 다룰때 쓰는 opencv 라이브러리의 이미지 읽기 함수
            str(image_path), #이미지 전체 경로를 파이썬이 알아듣기 쉬운 문자열(string)로 변환해서 넣는거
            cv2.IMREAD_COLOR, # 이미지를 컬러(BGR) 사진으로 읽어와라
        )
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image=preprocess(image, self.settings) #config.json파일에 적은 preprocess 옵션들(crop, bgr을 yuv로 변환)을 image에 적용시킴

        image=np.ascontiguousarray(  #[66,200,3] -> [3,66,200]으로 배열 정리.
            image.transpose(2,0,1)
        )        

        image=torch.from_numpy(image).float() #numpy 상태였던 image를 tensor로 바꿔줌

        target=torch.tensor( #csv에서 진짜 정답값(곡률) 가져옴
            [float(row["y_curvature_1pm"])],
            dtype=torch.float32,
        )

        return image,target #최종 image랑 target
        
if __name__ == "__main__":
    dataset = PilotNetDataset("train")
    image, target = dataset[0]

    print("samples:", len(dataset))
    print("image shape:", tuple(image.shape))
    print("image dtype:", image.dtype)
    print("target shape:", tuple(target.shape))
    print("target curvature:", target.item(), "1/m")
