"""Keep decoded CIL crops in memory while applying the original augmentation on access."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import IMAGE_SIZE, SPEED_SCALE_KPH, augment_image, preprocess_image


def augment_tone(image):
    """Vary brightness, contrast and tone without changing road geometry."""
    if np.random.random() < .35:
        image += np.random.uniform(-.10, .10)
    if np.random.random() < .35:
        image = .5 + (image - .5) * np.random.uniform(.8, 1.2)
    image = np.clip(image, 0., 1.)
    if np.random.random() < .30:
        image = image ** np.random.uniform(.85, 1.2)
    return image


class CachedImages(Dataset):
    def __init__(self, source, workers, *, photometric=False):
        self.samples = source.samples
        self.augment = source.augment
        self.photometric = photometric
        self.path_label = getattr(source, "path_label", None)
        self.images = np.empty((len(source), IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)

        def decode(sample):
            path = sample[0]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot decode image: {path}")
            copy = preprocess_image(bgr, augment=False).numpy().transpose(1, 2, 0)
            return np.rint(copy * 255).astype(np.uint8)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, image in enumerate(pool.map(decode, self.samples)):
                self.images[index] = image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, camera, speed, command, target = self.samples[index]
        image = self.images[index].astype(np.float32) / 255.0
        if self.augment:
            if self.photometric:
                image = augment_tone(image)
            image = augment_image(image)
        row = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "speed": torch.tensor([speed / SPEED_SCALE_KPH], dtype=torch.float32),
            "command": torch.tensor(command, dtype=torch.long),
            "target": torch.from_numpy(target.copy()),
            "camera": camera,
            "image_path": str(path),
        }
        if self.path_label is not None:
            row.update(self.path_label(index))
        return row
