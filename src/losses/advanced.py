from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return gx, gy


class PhotometricSmoothnessLoss(nn.Module):
    """
    Край-ориентированная гладкость для цветовых каналов a/b (по аналогии с edge-aware TV),
    взвешенная градиентами яркости L (в диапазоне [-1,1], переводим в [0,1]).
    """

    def __init__(self, weight_edge: float = 10.0):
        super().__init__()
        self.weight_edge = weight_edge

    def forward(
        self, L: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        L01 = (L + 1.0) * 0.5
        Lx, Ly = _sobel(L01)
        ax, ay = _sobel(a)
        bx, by = _sobel(b)
        wx = torch.exp(-self.weight_edge * torch.abs(Lx))
        wy = torch.exp(-self.weight_edge * torch.abs(Ly))
        tv = (
            wx * torch.abs(ax)
            + wy * torch.abs(ay)
            + wx * torch.abs(bx)
            + wy * torch.abs(by)
        ).mean()
        return tv


class DepthSmoothnessLoss(nn.Module):
    """Гладкость для карты глубины D с edge-aware взвешиванием яркостью L."""

    def __init__(self, weight_edge: float = 10.0):
        super().__init__()
        self.weight_edge = weight_edge

    def forward(self, L: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        L01 = (L + 1.0) * 0.5
        Lx, Ly = _sobel(L01)
        Dx, Dy = _sobel(D)
        wx = torch.exp(-self.weight_edge * torch.abs(Lx))
        wy = torch.exp(-self.weight_edge * torch.abs(Ly))
        tv = (wx * torch.abs(Dx) + wy * torch.abs(Dy)).mean()
        return tv


class ColorConsistencyPyrLoss(nn.Module):
    """
    Пирамидальная согласованность цвета: L1 между размытыми версиями предсказанного ab и GT ab на нескольких масштабах.
    Стабилизирует крупномасштабные цвета.
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels

    def forward(self, ab_pred: torch.Tensor, ab_gt: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), dtype=ab_pred.dtype, device=ab_pred.device)
        pred = ab_pred
        gt = ab_gt
        for i in range(self.levels):
            loss = loss + F.l1_loss(pred, gt)
            if i < self.levels - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                gt = F.avg_pool2d(gt, kernel_size=2, stride=2)
        return loss / float(self.levels)


class EntropyLoss(nn.Module):
    """
    Энтропия Бернулли для карты насыщенности sat (в [0,1]): -p log p - (1-p) log(1-p)
    Способствует уверенности (или может служить регуляризатором, вес настраивается).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, sat: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(sat, self.eps, 1.0 - self.eps)
        ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return ent.mean()


class OMMClusterLoss(nn.Module):
    """
    Кластеризационный лосс для OMM: делаем F2 (1/8) ближе к карте памяти, приведённой к тому же разрешению.
    Используем 1 - косинусное сходство как ошибку.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        n = torch.sqrt(torch.clamp((x * x).sum(dim=1, keepdim=True), min=eps))
        return x / n

    def forward(self, F2n: torch.Tensor, mem_map: torch.Tensor) -> torch.Tensor:
        # Приводим mem_map (B,C,H,W) к размеру F2n
        B, C, h2, w2 = F2n.shape
        mem_r = F.interpolate(
            mem_map, size=(h2, w2), mode="bilinear", align_corners=False
        )
        # Нормируем по каналам
        f = self._l2n(F2n)
        m = self._l2n(mem_r)
        # косинусная близость по каналам, усреднение по пространству и батчу
        cos = (f * m).sum(dim=1, keepdim=True)  # (B,1,h2,w2)
        loss = (1.0 - cos).mean()
        return loss
