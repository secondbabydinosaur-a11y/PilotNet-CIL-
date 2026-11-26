"""Train turn branches on a steering distribution and other branches on mean steering."""

import torch

from model import CILModel
from ordinal_model import OrdinalCIL


class TurnOrdinalCIL(OrdinalCIL):
    architecture = "cil_turn_ordinal_v1"

    def __init__(self):
        super().__init__()
        turn_names = ("left", "right")
        self.spec.update(mode_commands=list(turn_names), other_decode="expected steering")
        self.register_buffer("turn_branch", torch.tensor([name in turn_names for name in self.command_names]),
                             persistent=False)

    def forward(self, image, speed, command):
        output = CILModel.forward(self, image, speed, command)
        raw = output["all_controls"]
        logits = raw[..., 2:2 + self.components]
        candidates = self.bin_centres + self.half_bin_width * raw[..., 2 + self.components:].tanh()
        probability = logits.softmax(dim=-1)
        mean = (probability * candidates).sum(dim=-1, keepdim=True)
        mode = candidates.gather(-1, logits.argmax(dim=-1, keepdim=True))
        steering = torch.where(self.turn_branch[None, :, None], mode, mean)
        controls = torch.cat((steering, raw[..., :2]), dim=-1)
        rows = torch.arange(image.shape[0], device=command.device)
        output.update(
            control=controls[rows, command], all_controls=controls,
            steering_logits=logits[rows, command], steering_candidates=candidates[rows, command],
            turn_branch=self.turn_branch[command],
        )
        return output

    def steering_loss(self, output, target):
        # The trainer applies .10 to the steering loss, leaving point branches at the CIL MSE weight.
        point_loss = 10 * (output["control"][:, 0] - target).square()
        distribution_loss = super().steering_loss(output, target)
        return torch.where(output["turn_branch"], distribution_loss, point_loss)
