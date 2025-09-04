from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision import models  # type: ignore[import-untyped]


class VGGFeatureExtractor(nn.Module):
    """
    Извлекает промежуточные признаки из VGG19 на заданных слоях.
    Параметр `layers` — индексы слоёв фичей; `requires_grad=False` замораживает веса.
    """

    def __init__(
        self, layers: List[int] = [3, 8, 17, 26, 35], requires_grad: bool = False
    ):
        super().__init__()
        vgg = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        ).features  # allowed for perceptual
        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in layers:
            self.slices.append(
                nn.Sequential(*[vgg[i] for i in range(prev, layer_idx + 1)])
            )
            prev = layer_idx + 1
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

    def __init__(
        self, layers: List[int] = [3, 8, 17, 26, 35], weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.feat = VGGFeatureExtractor(layers=layers, requires_grad=False)
        if weights is None:
            weights = [1.0] * len(layers)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ожидаются входы в диапазоне [0,1] в RGB
        # VGG должна работать в float32. Отключаем автокаст и приводим к float32.
        pred_fp32 = torch.clamp(pred, 0.0, 1.0).to(dtype=torch.float32)
        target_fp32 = torch.clamp(target, 0.0, 1.0).to(dtype=torch.float32)
        # Гарантируем, что сама VGG на том же устройстве и в float32
        self.feat.to(device=pred.device, dtype=torch.float32)
        # Вычисляем признаки в режиме без autocast (важно при AMP на CUDA)
        with torch.cuda.amp.autocast(enabled=False):
            p = self.feat(pred_fp32)
            t = self.feat(target_fp32)
            # Накапливаем лосс в float32 на том же устройстве, где были входы
            loss = torch.zeros((), dtype=torch.float32, device=pred.device)
            for i, (pf, tf) in enumerate(zip(p, t)):
                loss = loss + self.w[i].to(loss.dtype) * self.l1(pf, tf)
        # Приводим тип результата к типу исходного тензора для совместимости
        return loss.to(dtype=pred.dtype)
