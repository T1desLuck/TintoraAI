from typing import Tuple
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class ConvNeXtTiny(nn.Module):
    """
    Минимальная стадия в стиле ConvNeXt-Tiny для получения признаков F1 на H/4, C=96 из входа L.
    """

    def __init__(self, out_channels: int = 96, in_ch: int = 1, depth: int = 2):
        super().__init__()
        # Даунсэмплинг 4x с помощью свёртки со stride=4 (1→out_channels)
        self.stem = nn.Conv2d(in_ch, out_channels, kernel_size=7, stride=4, padding=3)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x  # (B,96,H/4,W/4)
