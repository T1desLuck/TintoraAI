from typing import Tuple
import torch
import torch.nn as nn


class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,HW,C) — преобразование к последовательности
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        res = x_seq
        x_seq = self.norm1(x_seq)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = res + attn_out
        res2 = x_seq
        x_seq = self.norm2(x_seq)
        x_seq = res2 + self.ff(x_seq)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        return x


class CoAtNetLight(nn.Module):
    """
    Облегчённая стадия в стиле CoAtNet: даунсэмплинг 2x (H/4→H/8), свёрточный блок + MHSA.
    Вход: (B, C_in=96, H/4, W/4)
    Выход: (B, C_out=192, H/8, W/8)
    """

    def __init__(self, in_channels: int = 96, out_channels: int = 192, num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([MHSA(out_channels, num_heads=num_heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        for blk in self.blocks:
            x = blk(x)
        return x
