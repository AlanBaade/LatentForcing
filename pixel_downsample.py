## UTILS

import torch
from torch import nn
import torch.nn.functional as F

class PixelDownsampleEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @torch.compile()
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        # x : b c h w

        x = F.interpolate(x, size=(64, 64), mode='area')
        B, C, H, W = x.shape
        p = 4  # patch size
        z = x.view(B, C, H // p, p, W // p, p).permute(0, 1, 3, 5, 2, 4).reshape(B, 48, 16, 16)

        return z
