"""A CIL variant with one squeeze-and-excitation gate on the visual features."""

import torch
from torch import nn

from model import CILModel


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, channels // 16), nn.ReLU(),
                                  nn.Linear(channels // 16, channels))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        pooled = features.mean(dim=(2, 3))
        scale = 2.0 * self.gate(pooled).sigmoid()
        return features * scale[:, :, None, None]


class AttentionCIL(CILModel):
    architecture = "cil_channel_attention_v1"

    def __init__(self):
        super().__init__()
        channels = self.image_conv[-1].layers[0].out_channels
        with torch.random.fork_rng(devices=[]):
            self.image_conv.add_module("channel_attention", ChannelAttention(channels))
