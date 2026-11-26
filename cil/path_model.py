"""CIL that predicts a short vehicle-relative path and steers toward its first point."""

import math

import torch
from torch import nn

from model import CILModel


PATH_DISTANCES_M = (3.0, 6.0, 10.0)
PATH_SPEC = {"distances_m": list(PATH_DISTANCES_M), "wheelbase_m": 3.0, "max_angle_deg": 40.0}


class PathCIL(CILModel):
    architecture = "cil_predicted_path_v1"

    def __init__(self):
        super().__init__()
        self.register_buffer("distances", torch.tensor(PATH_DISTANCES_M), persistent=False)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for branch in self.branches:
                original = branch[-1]
                output = nn.Linear(original.in_features, 2 + 2 * len(PATH_DISTANCES_M))
                output.weight[:2].copy_(original.weight[1:])
                output.bias[:2].copy_(original.bias[1:])
                output.bias[2::2].fill_(1.)
                output.bias[3::2].zero_()
                branch[-1] = output

    def forward(self, image, speed, command):
        output = super().forward(image, speed, command)
        raw = output["all_controls"]
        paths = raw[..., 2:].reshape(*raw.shape[:2], len(PATH_DISTANCES_M), 2)
        x, y = (paths[..., 0, :] * self.distances[0]).unbind(dim=-1)
        squared_distance = x.clamp_min(.25).square() + y.square()
        curvature = 2. * y / squared_distance.clamp_min(1.)
        steer = torch.atan(PATH_SPEC["wheelbase_m"] * curvature) / math.radians(PATH_SPEC["max_angle_deg"])
        all_controls = torch.cat((steer.clamp(-1., 1.).unsqueeze(-1), raw[..., :2]), dim=-1)
        rows = torch.arange(image.shape[0], device=image.device)
        output.update(control=all_controls[rows, command], all_controls=all_controls, path=paths[rows, command])
        return output
