from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        # Апсемплируем к размеру skip и объединяем по каналам
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DepthHead(nn.Module):
    """
    Голова для предсказания глубины: использует F3 (низкое разрешение) + пропуск F2,
    апсемплирует до полного разрешения H×W.
    Возвращает D в диапазоне [0, 1] через sigmoid.
    """

    def __init__(self, c2: int, c3: int, mid: int = 128):
        super().__init__()
        self.reduce3 = nn.Conv2d(c3, mid, kernel_size=1)
        self.reduce2 = nn.Conv2d(c2, mid, kernel_size=1)
        self.up = UpBlock(mid, mid, mid)
        self.out = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, F2: torch.Tensor, F3: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x3 = self.reduce3(F3)
        x2 = self.reduce2(F2)
        x = self.up(x3, x2, out_hw)
        D = torch.sigmoid(self.out(x))
        return D


class IlluminationHead(nn.Module):
    """
    Голова карты освещённости, аналогичная по структуре голове глубины.
    """

    def __init__(self, c2: int, c3: int, mid: int = 128):
        super().__init__()
        self.reduce3 = nn.Conv2d(c3, mid, kernel_size=1)
        self.reduce2 = nn.Conv2d(c2, mid, kernel_size=1)
        self.up = UpBlock(mid, mid, mid)
        self.out = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, F2: torch.Tensor, F3: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x3 = self.reduce3(F3)
        x2 = self.reduce2(F2)
        x = self.up(x3, x2, out_hw)
        I = torch.sigmoid(self.out(x))
        return I
