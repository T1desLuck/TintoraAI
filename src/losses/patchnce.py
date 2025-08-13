from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    PatchNCE / InfoNCE-стиль лосс для контрастного сопоставления патч-фичей.
    Ожидает L2-нормированные признаки. Если нет — можно включить normalize=True.

    forward(q, k, temperature=None):
      - q: (B, C, N)  — анкоры (например, из текущего вида)
      - k: (B, C, N)  — положительные (например, из аугм. вида)
      - temperature: float | None — если None, используется self.temperature
    Возвращает скалярный loss.
    """

    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        n = torch.sqrt(torch.clamp((x * x).sum(dim=1, keepdim=True), min=eps))
        return x / n

    def forward(self, q: torch.Tensor, k: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        assert q.dim() == 3 and k.dim() == 3, "q,k must be (B,C,N)"
        B, C, N = q.shape
        assert k.shape == (B, C, N), "q and k must have the same shape"
        t = float(self.temperature if temperature is None else temperature)
        # Нормировка, если требуется
        if self.normalize:
            q = self._l2n(q)
            k = self._l2n(k)
        # Перестроим (B,N,C)
        q_ = q.permute(0, 2, 1).contiguous()  # (B,N,C)
        k_ = k.permute(0, 2, 1).contiguous()  # (B,N,C)
        # Логиты: позитивы по диагонали внутри батча, негативы — остальные
        # Соберём матрицу сходств между всеми патчами всех элементов батча
        # Сольём батч в ось пачей: BN x C
        q_flat = q_.reshape(B * N, C)  # (BN,C)
        k_flat = k_.reshape(B * N, C)  # (BN,C)
        logits = torch.matmul(q_flat, k_flat.t()) / t  # (BN, BN)
        # Таргеты — положительные соответствия (один и тот же индекс в пределах элемента батча)
        # В пределах каждого B блока из N, позитив — диагональ
        target = torch.arange(B * N, device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        return loss
