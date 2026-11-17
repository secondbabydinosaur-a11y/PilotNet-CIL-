#CIL 모델 구현...(설계만!!!!!!!!!)
#실제 학습은 여기서는 안하고 train.py 파일에서 진행함. train.py에서 dataset.py랑 model.py import해서....

#Pilotnet에서는 구조가 간단해서 하나의 클래스 안에 fc layer, conv layer 다 몰아두었는데 그러면 가독성 너무 떨어짐
# -> 그래서 반복사용하는 conv쪽이랑 fc쪽이랑 나눠서 클래스 정의. 그래서 최종 cilmodel에서 이 클래스를 불러와 전체 신경망 구현(전보다 훨씬 가독성 있음)


##############################################################################################


import torch
from torch import nn #conv,linear같은 층 사용



############################# CNN에서 Convolution layer 선언 ######################################

class ConvBlock(nn.Module): #nn.Module 상속받아서 진행 -> pytorch에서 신경망 만들떄 사용하는 클래스인 nn.Module 상속

    def __init__(self, in_channels, out_channels, kernel_size, stride): # 밑에 있는 CILModel에서 layer 쌓을떄 조정할 수 있는 하이퍼파라미터 선언함.!
                        #in_channels=입력채널수, out_channels=출력 채널 수 
        super().__init__()


        self.layers=nn.Sequential( #CNN 층 묶기(나중에 ConBlock.layers라고 선언하면 밑에 정의한 층들의 묶음 사용 가능)
            nn.Conv2d(         #층1: convolution layer
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0
            ),
            nn.BatchNorm2d(out_channels), #채널별로 feature 정규화
            nn.Dropout(p=0.20), #dropout 진행(과대적합 방지)
            nn.ReLU(), #활성화함수로 relu 사용
        )

    def forward(self, x): #앞에서 만든 생성자를 불러와서 실제로 실행시키는 역할
        return self.layers(x)

############################ CNN에서 FC layer 선언 ########################################

class FCBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.layers=nn.Sequential(
            nn.Linear(in_features, out_features), #conv layer층 아니니깐 커널이랑 stride 필요 x
            nn.Dropout(p=0.50),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
    


##################### 이제 본격적인 신경망 구현 #####################

class CILModel(nn.Module):
    architecture="cil_branched_architecture" #체크포인트에 기록할 구조 이름.
    command_names=(
        "follow_road", "left", "straight", "right"
    )

    def __init__(self):
        super().__init__()

        self.image_conv=nn.Sequential(
            ConvBlock(3, 32, 5, 2),  # [B, 32, 42, 98]
            ConvBlock(32, 32, 3, 1),  # [B, 32, 40, 96]
            ConvBlock(32, 64, 3, 2),  # [B, 64, 19, 47]
            ConvBlock(64, 64, 3, 1),  # [B, 64, 17, 45]
            ConvBlock(64, 128, 3, 2),  # [B, 128, 8, 22]
            ConvBlock(128, 128, 3, 1),  # [B, 128, 6, 20]
            ConvBlock(128, 256, 3, 1),  # [B, 256, 4, 18]
            ConvBlock(256, 256, 3, 1),  # [B, 256, 2, 16]
        ) #cnn에서 convolution layer를 의미

        self.image_fc=nn.Sequential(
            FCBlock(256*2*16,512), #flatten한 cnn 출력이 input
            FCBlock(512,512), 
        )

        self.speed_encoder=nn.Sequential(  #CIL은 카메라 이미지 말고도 차량 현재 속도도 input임
            FCBlock(1,128), #정규화된 현재 속도 하나 -> 속도 feature 128개
            FCBlock(128, 128), 
        )

        self.fusion=FCBlock(
            512+128,512              #이미지특징:512+속도특징:128 합쳐야함!
        ) 
        self.branches=nn.ModuleList([ #앞에서 정의한 command_names마다 출력 다르게!
            nn.Sequential(
                FCBlock(512,256),
                FCBlock(256,256),
                nn.Linear(256, 3),   #256->3(steering,throttle,brake를 의미)
            )
            for _ in self.command_names  
        ])


        self.speed_head = nn.Sequential(
            FCBlock(512, 256),  # 이미지 feature만 사용한다.
            FCBlock(256, 256),  # 보조 속도 예측용 hidden feature다.
            nn.Linear(256, 1),  # 실제 정규화 속도 하나를 예측한다.
        )

    
    def forward(self, image, speed, command):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 88, 200):
            raise ValueError("image must have shape [B, 3, 88, 200]")

        batch_size = image.shape[0]  # 현재 배치의 샘플 수다.

        if batch_size == 0 or tuple(speed.shape) != (batch_size, 1):
            raise ValueError("speed must have shape [B, 1] with B > 0")

        if tuple(command.shape) != (batch_size,) or command.dtype != torch.long:
            raise ValueError("command must be int64 with shape [B]")

        if speed.device != image.device or command.device != image.device:
            raise ValueError("image, speed and command must be on the same device")

        if torch.any((command < 0) | (command >= len(self.command_names))):
            raise ValueError("command must be 0, 1, 2 or 3")

        conv_features = self.image_conv(image)  # [B, 256, 2, 16]

        flat_features = torch.flatten(
            conv_features, start_dim=1
        )  # 배치 차원을 유지하면서 [B, 8192]로 펼친다.

        image_features = self.image_fc(flat_features)  # [B, 512]
        speed_features = self.speed_encoder(speed)  # [B, 128]

        combined = torch.cat(
            [image_features, speed_features], dim=1
        )  # feature 축으로 연결해 [B, 640]을 만든다.

        joint_features = self.fusion(combined)  # 결합 feature [B, 512]다.

        all_controls = torch.stack(
            [branch(joint_features) for branch in self.branches],
            dim=1,
        )  # [B, 4, 3]: 배치·명령·제어값 순서다.

        batch_indices = torch.arange(
            batch_size, device=image.device
        )  # 각 샘플의 배치 번호다.

        control = all_controls[
            batch_indices, command
        ]  # 샘플별 명령에 해당하는 결과를 선택해 [B, 3]을 만든다.

        pred_speed = self.speed_head(
            image_features
        )  # 이미지 기반 보조 속도 예측 [B, 1]이다.

        return {
            "control": control,  # Teacher 제어값과 비교할 예측이다.
            "all_controls": all_controls,  # 네 branch의 출력을 확인할 때 사용한다.
            "pred_speed": pred_speed,  # 실제 속도/20과 비교할 보조 출력이다.
        }
    

def main():
    """직접 실행했을 때 가상 입력으로 입출력 형태를 점검한다."""

    torch.set_num_threads(1)  # CPU 점검에서 과도한 스레드 생성을 줄인다.
    torch.manual_seed(7)  # 초기 가중치와 가상 입력을 재현한다.

    model = CILModel().eval()  # Dropout을 끄고 BatchNorm을 평가 모드로 사용한다.

    image = torch.rand(
        1, 3, 88, 200
    ).repeat(4, 1, 1, 1)  # 같은 가상 영상 네 장이다.

    speed = torch.full((4, 1), 0.5)  # 실제 10km/h에 해당하는 정규화 속도다.
    command = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 명령만 다르게 준다.

    with torch.no_grad():  # 형태 점검에는 gradient가 필요하지 않다.
        output = model(image, speed, command)

    assert output["control"].shape == (4, 3)
    assert output["all_controls"].shape == (4, 4, 3)
    assert output["pred_speed"].shape == (4, 1)

    assert torch.equal(
        output["control"],
        output["all_controls"][torch.arange(4), command],
    )  # 명령에 맞는 branch를 선택했는지도 확인한다.

    print("MODEL SELF-TEST PASS")

    for name, value in output.items():
        print(name + ":", tuple(value.shape))  # 출력별 크기를 보여준다.

    print(
        "parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )  # 학습 가능한 파라미터 개수를 보여준다.


if __name__ == "__main__":  # train.py에서 import할 때는 점검을 실행하지 않는다.
    main()
