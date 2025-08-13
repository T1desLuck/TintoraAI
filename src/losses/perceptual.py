from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Извлекает промежуточные признаки из VGG19 на заданных слоях.
    Параметр `layers` — индексы слоёв фичей; `requires_grad=False` замораживает веса.
    """
    def __init__(self, layers: List[int] = [3, 8, 17, 26, 35], requires_grad: bool = False):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # allowed for perceptual
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l + 1)]))
            prev = l + 1
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = x
        for seq in self.slices:
            h = seq(h)
            feats.append(h)
        return feats


class PerceptualLoss(nn.Module):
    """
    Перцептуальный лосс на основе VGG19: суммирует L1-различия признаков
    между предсказанием и таргетом на нескольких уровнях `layers` с весами `weights`.
    Ожидает входы в RGB диапазоне [0, 1].
    """
    def __init__(self, layers: List[int] = [3, 8, 17, 26, 35], weights: List[float] = None):
        super().__init__()
        self.feat = VGGFeatureExtractor(layers=layers, requires_grad=False)
        if weights is None:
            weights = [1.0] * len(layers)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ожидаются входы в диапазоне [0,1] в RGB
        p = self.feat(pred)
        t = self.feat(target)
        loss = 0.0
        for i, (pf, tf) in enumerate(zip(p, t)):
            loss = loss + self.w[i] * self.l1(pf, tf)
        return loss
