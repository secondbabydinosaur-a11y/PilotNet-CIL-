"""Order-aware probability loss for the binned steering model."""

import torch

from distribution_model import DistributionCIL


class OrdinalCIL(DistributionCIL):
    architecture = "cil_steering_ordinal_v1"

    def __init__(self):
        super().__init__("bins")
        self.architecture = type(self).architecture
        self.spec.update(classification_weight=.65, ranked_probability_weight=3.)

    def steering_loss(self, output, target):
        probabilities = output["steering_logits"].softmax(dim=1)
        cumulative = probabilities.cumsum(dim=1)[:, :-1]
        cutpoints = torch.arange(1, self.components, device=target.device) * (2 * self.half_bin_width) - 1
        observed_cdf = (target[:, None] < cutpoints).to(cumulative.dtype)
        ranked_probability = (cumulative - observed_cdf).square().mean(dim=1)
        return .65 * super().steering_loss(output, target) + 3. * ranked_probability

    @classmethod
    def from_settings(cls, settings):
        model = cls()
        if settings.get("steering_distribution") != model.spec:
            raise ValueError("checkpoint ordinal steering contract mismatch")
        return model
