"""실제 데이터 한 배치로 입력/출력과 역전파를 확인한다. 가중치는 갱신하지 않는다."""
import torch
from torch.utils.data import DataLoader

from dataset import PilotNetDataset
from learning.common import choose_device, seed_everything
from model import PilotNet


def main():
    seed_everything(42)
    torch.set_num_threads(4)
    device = choose_device()
    dataset = PilotNetDataset("train")
    images, targets = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))
    model = PilotNet().to(device)
    prediction = model(images.to(device))
    assert prediction.shape == targets.shape == (32, 1)
    loss = torch.nn.functional.mse_loss(prediction, targets.to(device))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
               for p in model.parameters())
    print("device:", device)
    print("train samples:", len(dataset))
    print("images:", tuple(images.shape), images.dtype)
    print("targets / predictions:", tuple(targets.shape), tuple(prediction.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("initial MSE:", loss.item())
    print("forward / backward OK")


if __name__ == "__main__":
    main()
