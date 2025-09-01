from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_features, eps=eps)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        # x: (B,C,H,W), gamma/beta: (B,C) — применение FiLM после нормализации
        x = self.norm(x)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ConvFiLMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, film_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.film = FiLMLayer(out_ch)
        self.film_gamma = nn.Linear(film_in, out_ch)
        self.film_beta = nn.Linear(film_in, out_ch)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        g = self.film_gamma(gamma)
        b = self.film_beta(beta)
        x = self.film(x, g, b)
        return x


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class UNetPPDecoder(nn.Module):
    """
    Облегчённый декодер в стиле U-Net++ с FiLM-модуляцией на каждой стадии.
    Стадии соответствуют уровням признаков: F3 (c3, H/16), F2 (c2, H/8), F1 (c1, H/4).
    """

    def __init__(self, c1: int, c2: int, c3: int, mid: int = 128):
        super().__init__()
        self.output_channels = mid
        # Уровень 3 (самое низкое разрешение)
        self.l3 = ConvFiLMBlock(c3, mid, film_in=c3)
        # Поднимаемся до уровня 2 и сливаем с F2
        self.up32 = PixelShuffleUp(mid, mid)
        self.l2 = ConvFiLMBlock(mid + c2, mid, film_in=c2)
        # Поднимаемся до уровня 1 и сливаем с F1
        self.up21 = PixelShuffleUp(mid, mid)
        self.l1 = ConvFiLMBlock(mid + c1, 64, film_in=c1)

        self.head_ab = nn.Conv2d(64, 2, kernel_size=1)
        self.head_sat = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        F1: torch.Tensor,
        F2: torch.Tensor,
        F3: torch.Tensor,
        gammas: list[torch.Tensor],
        betas: list[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        gammas/betas: список тензоров формы (B,C) для стадий [F3, F2, F1] с размерами [c3, c2, ?].
        FiLM применяется в блоках с векторами по стадиям. Для l1 проецируем FiLM к 64 через срез/паддинг.
        """
        # Индексы стадий: 0→F3, 1→F2, 2→F1
        g3, g2, g1 = gammas[0], gammas[1], gammas[2]  # (B,C)
        b3, b2, b1 = betas[0], betas[1], betas[2]

        x3 = self.l3(F3, g3, b3)
        x2in = self.up32(x3)
        # Выравниваем пространственные размеры под F2
        x2in = F.interpolate(
            x2in, size=F2.shape[-2:], mode="bilinear", align_corners=False
        )
        x2 = torch.cat([x2in, F2], dim=1)
        x2 = self.l2(x2, g2, b2)

        x1in = self.up21(x2)
        x1in = F.interpolate(
            x1in, size=F1.shape[-2:], mode="bilinear", align_corners=False
        )
        x1 = torch.cat([x1in, F1], dim=1)
        x1 = self.l1(x1, g1, b1)

        x = F.interpolate(x1, size=out_size, mode="bilinear", align_corners=False)
        ab = self.head_ab(x)
        sat = torch.sigmoid(self.head_sat(x))
        return ab, sat
