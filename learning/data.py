"""Dataset 원본 코드/주석을 유지하면서 선택적인 RAM 캐시와 학습 sampler 제공."""
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from learning.common import seed_worker


class CachedDataset(Dataset):
    def __init__(self, source, workers=4):
        self.rows, self.settings = source.rows, source.settings
        # 전처리 결과만 uint8로 캐시한다. 원본 JPEG는 그대로 보존한다.
        self.images = torch.empty((len(source), 3, 66, 200), dtype=torch.uint8)
        self.targets = torch.empty((len(source), 1), dtype=torch.float32)
        loader = DataLoader(source, batch_size=128, shuffle=False, num_workers=workers,
                            worker_init_fn=seed_worker)
        offset = 0
        for images, targets in loader:
            n = len(images)
            self.images[offset:offset+n].copy_(images.to(torch.uint8))
            self.targets[offset:offset+n].copy_(targets)
            offset += n
        print("Cached {} images ({:.0f} MiB)".format(
            len(source), self.images.numel() / 1024 ** 2), flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.images[index].float(), self.targets[index]


def sampling_curvature(row, basis="target"):
    if basis not in ("target", "center"):
        raise ValueError("curve basis must be target or center")
    # 좌우 이미지의 복귀 조향은 도로가 직선이어도 0이 아니다.
    # center는 보정 전 teacher 곡률을 사용한다. 실제 지도 곡률의 독립 정답은 아니다.
    if basis == "center" and row.get("camera") != "center":
        if row.get("center_curvature_1pm") in (None, ""):
            raise ValueError("side camera needs uncorrected center curvature")
        return float(row["center_curvature_1pm"])
    return float(row["y_curvature_1pm"])


def curve_sampler(dataset, weight, seed, basis="target"):
    if weight < 1:
        raise ValueError("curve weight must be >= 1")
    if weight == 1:
        return None
    weights = [weight if abs(sampling_curvature(row, basis)) >= .003 else 1.
               for row in dataset.rows]
    return WeightedRandomSampler(weights, len(weights), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))
