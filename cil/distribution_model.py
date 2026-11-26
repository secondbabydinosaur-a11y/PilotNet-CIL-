"""Conditional steering distributions for the existing CIL command branches."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model import CILModel


DISTRIBUTION_ARCHITECTURES = {
    "mixture": "cil_steering_mixture_v1",
    "bins": "cil_steering_bins_v1",
}


class DistributionCIL(CILModel):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution
        self.architecture = DISTRIBUTION_ARCHITECTURES[distribution]
        self.components = 3 if distribution == "mixture" else 17
        self.half_bin_width = 1 / self.components
        columns = 3 if distribution == "mixture" else 2
        self.spec = {
            "distribution": distribution,
            "components": self.components,
            "steering_range": [-1, 1],
            "decode": "highest probability component",
        }
        if distribution == "mixture":
            self.spec["scale_range"] = [0.02, 0.50]

        self.register_buffer(
            "bin_centres", torch.linspace(-1 + self.half_bin_width, 1 - self.half_bin_width, self.components),
            persistent=False,
        )
        # The shared layers and pedal outputs start from the baseline initialization.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for branch in self.branches:
                original = branch[-1]
                head = nn.Linear(original.in_features, 2 + columns * self.components)
                head.weight[:2].copy_(original.weight[1:])
                head.bias[:2].copy_(original.bias[1:])
                if distribution == "mixture":
                    head.bias[2 + self.components:2 + 2 * self.components].copy_(
                        torch.linspace(-0.25, 0.25, self.components).atanh())
                    head.bias[2 + 2 * self.components:].fill_(-1.3)
                branch[-1] = head

    def forward(self, image, speed, command):
        output = super().forward(image, speed, command)
        raw = output["all_controls"]
        logits = raw[..., 2:2 + self.components]
        values = raw[..., 2 + self.components:2 + 2 * self.components].tanh()
        if self.distribution == "bins":
            values = self.bin_centres + self.half_bin_width * values
        preferred = logits.argmax(dim=-1, keepdim=True)
        all_controls = torch.cat((values.gather(-1, preferred), raw[..., :2]), dim=-1)
        rows = torch.arange(image.shape[0], device=command.device)
        output.update(
            control=all_controls[rows, command], all_controls=all_controls,
            steering_logits=logits[rows, command], steering_candidates=values[rows, command],
        )
        if self.distribution == "mixture":
            output["steering_scales"] = 0.02 + 0.48 * raw[rows, command, 2 + 2 * self.components:].sigmoid()
        return output

    def steering_loss(self, output, target):
        logits = output["steering_logits"]
        values = output["steering_candidates"]
        if self.distribution == "bins":
            index = ((target + 1) / (2 * self.half_bin_width)).long().clamp(0, self.components - 1)
            predicted = values.gather(1, index[:, None])[:, 0]
            residual = ((predicted - target) / self.half_bin_width).square()
            return F.cross_entropy(logits, index, reduction="none") + 0.25 * residual

        scales = output["steering_scales"]
        z = (target[:, None] - values) / scales
        # Normalizing on [-1, 1] gives a density on the actuator's valid range.
        mass = torch.special.ndtr((1 - values) / scales) - torch.special.ndtr((-1 - values) / scales)
        log_density = -0.5 * z.square() - scales.log() - 0.5 * math.log(2 * math.pi) - mass.log()
        return -torch.logsumexp(F.log_softmax(logits, dim=1) + log_density, dim=1)


def distribution_from_checkpoint(settings):
    distribution = next(name for name, architecture in DISTRIBUTION_ARCHITECTURES.items()
                        if architecture == settings["architecture"])
    model = DistributionCIL(distribution)
    if settings.get("steering_distribution") != model.spec:
        raise ValueError("checkpoint steering distribution contract mismatch")
    return model
