from __future__ import annotations
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

dist_utils: Optional[Any]
try:
    from ...utils import dist as _dist_utils  # type: ignore
    dist_utils = _dist_utils
except Exception:
    dist_utils = None


def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))
    return x / n


class ObjectMemoryModule(nn.Module):
    """
    Object Memory Module (OMM)
    - Банк прототипов размера P×D (буфер без градиентов, обновляется через EMA)
    - Региональный пуллинг признаков в R×D
    - Top-k softmax-внимание по прототипам для получения векторов памяти
    - EMA-обновления выбранных прототипов на основе усреднённых регионов
    - Обслуживание прототипов: min_support, опциональная случайная реинициализация
    - DDP-совместимость: при необходимости обновление только на rank0 и широковещание
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 2048,  # ТЗ: 2048
        momentum: float = 0.995,  # ТЗ: 0.995 (alpha)
        temperature: float = 0.07,  # ТЗ: 0.07 (tau)
        topk: int = 64,  # ТЗ: 64
        grid: int = 7,  # По умолчанию, не в ТЗ
        min_support: int = 15,  # Из конфига
        reinit_prob: float = 0.0,  # По умолчанию
        sync_rank0: bool = True,  # Из конфига
        **kwargs,  # Поглощаем лишние параметры из конфига
    ):
        super().__init__()
        self.dim = dim
        self.P = num_prototypes
        self.momentum = momentum
        self.temperature = temperature
        self.topk = topk
        self.grid = grid
        self.min_support = min_support
        self.reinit_prob = reinit_prob
        self.sync_rank0 = sync_rank0

        prot = torch.randn(self.P, dim)
        prot = l2n(prot)
        self.register_buffer("prototypes", prot)  # (P,D)
        self.register_buffer("support", torch.zeros(self.P, dtype=torch.long))
        # Буферы цветовой статистики на прототип
        self.register_buffer("prot_mu_ab", torch.zeros(self.P, 2))
        self.register_buffer("prot_sigma_ab", torch.zeros(self.P, 2))

    @torch.no_grad()
    def _ddp_should_update_here(self) -> bool:
        if not self.training:
            return False
        if not self.sync_rank0:
            return True
        if dist_utils is None or not dist.is_initialized():
            return True
        return dist_utils.is_main_process()

    def regional_pool(
        self, x: torch.Tensor, target_dim: Optional[int] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Адаптивное среднее усреднение к фиксированной сетке (grid×grid).
        Вход: F2 (B,C,H/8,W/8). Выход: regions (B,R,D) и размеры сетки (h,w).
        """
        B, C, H, W = x.shape
        g = self.grid
        pooled = F.adaptive_avg_pool2d(x, output_size=(g, g))  # (B,C,g,g)
        regions = pooled.flatten(2).transpose(1, 2)  # (B,R,C)
        if (
            target_dim is None
        ):  # L2-нормализация только для признаков, не для цветовых значений
            regions = l2n(regions)  # нормализация по регионам
        return regions, (g, g)

    def forward(
        self,
        F2: torch.Tensor,
        gt_ab: Optional[torch.Tensor] = None,
        read_only: Optional[bool] = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        """
        F2: (B, D, H, W) — L2-нормализованные признаки.
        Возвращает словарь: {attn (B,R,k), idx (B,R,k), mem (B,R,D), grid_hw}
        """
        if read_only is None:
            read_only = not self.training

        regions, grid_hw = self.regional_pool(F2)  # (B,R,D)

        gt_ab_regions = None
        if gt_ab is not None and not read_only:
            # Пулим истинные значения ab к той же сетке
            # Ожидается, что gt_ab в полном разрешении, как и входной L
            gt_ab_pooled = F.adaptive_avg_pool2d(
                gt_ab, output_size=grid_hw
            )  # (B,2,g,g)
            gt_ab_regions = gt_ab_pooled.flatten(2).transpose(1, 2)  # (B,R,2)
        B, R, D = regions.shape
        P = self.P

        # Косинусное сходство с прототипами (оба нормализованы)
        prot = l2n(self.prototypes)  # (P,D)
        sim = regions @ prot.t()  # (B,R,P)
        if self.topk < P:
            topv, topi = torch.topk(sim, k=self.topk, dim=-1)  # (B,R,k)
            attn = F.softmax(topv / self.temperature, dim=-1)
            sel = prot[topi]  # (B,R,k,D)
            mem = torch.sum(attn.unsqueeze(-1) * sel, dim=-2)  # (B,R,D)
        else:
            attn = F.softmax(sim / self.temperature, dim=-1)  # (B,R,P)
            mem = attn @ prot  # (B,R,D)
            topi = None

        out = {
            "attn": attn,
            "idx": topi,
            "mem": mem,
            "grid_hw": grid_hw,
            "prot_mu_ab": self.prot_mu_ab,
            "prot_sigma_ab": self.prot_sigma_ab,
        }

        if not read_only:
            self.ema_update(regions, attn, topi, gt_ab_regions)
        return out

    @torch.no_grad()
    def ema_update(
        self,
        regions: torch.Tensor,
        attn: torch.Tensor,
        topi: Optional[torch.Tensor],
        gt_ab_regions: Optional[torch.Tensor],
    ) -> None:
        """
        EMA-обновление прототипов на основе региональных признаков.
        Если используется top-k, обновляем только выбранные индексы с весами attn.
        Иначе обновляем все прототипы, взвешенные attn.
        DDP: при sync_rank0 обновление только на rank0 и широковещание буферов.
        """
        if not self._ddp_should_update_here():
            return

        m = self.momentum
        if topi is not None:
            # накапливаем обновления на прототип
            B, R, k = attn.shape
            D = regions.shape[-1]
            # Для каждой пары (b,r) распределяем регион по k выбранным прототипам
            flat_idx = topi.reshape(-1)  # (B*R*k)
            flat_w = attn.reshape(-1)
            flat_feat = (
                regions.unsqueeze(2).expand(B, R, k, D).reshape(-1, D)
            )  # (B*R*k,D)
            # агрегируем по индексу прототипа
            delta = torch.zeros_like(self.prototypes)  # (P,D)
            sup = torch.zeros(self.P, device=delta.device, dtype=torch.float32)
            delta.index_add_(0, flat_idx, flat_w.unsqueeze(-1) * flat_feat)
            sup.index_add_(0, flat_idx, flat_w)
            # Обновляем цветовую статистику, если доступна (ветка top-k)
            if gt_ab_regions is not None:
                flat_color = (
                    gt_ab_regions.unsqueeze(2).expand(B, R, k, 2).reshape(-1, 2)
                )  # (B*R*k,2)
                delta_mu = torch.zeros(
                    self.P, 2, device=delta.device, dtype=flat_color.dtype
                )
                delta_sigma = torch.zeros_like(delta_mu)
                delta_mu.index_add_(0, flat_idx, flat_w.unsqueeze(-1) * flat_color)
                delta_sigma.index_add_(
                    0, flat_idx, flat_w.unsqueeze(-1) * (flat_color**2)
                )
        else:
            # случай полного внимания
            # delta = sum_{b,r} attn[b,r,p] * regions[b,r]
            B, R, P = attn.shape
            delta = attn.reshape(-1, P).t() @ regions.reshape(
                -1, regions.shape[-1]
            )  # (P,D)
            sup = attn.sum(dim=(0, 1))  # (P,)

            # Обновление цветовой статистики, если доступна
            if gt_ab_regions is not None:
                # delta_mu = sum_{b,r} attn[b,r,p] * gt_ab_regions[b,r]
                delta_mu = attn.reshape(-1, P).t() @ gt_ab_regions.reshape(
                    -1, 2
                )  # (P,2)
                # delta_sigma = sum_{b,r} attn[b,r,p] * (gt_ab_regions[b,r]**2)
                delta_sigma = attn.reshape(-1, P).t() @ (
                    gt_ab_regions.reshape(-1, 2) ** 2
                )  # (P,2)

        # избегаем деления на ноль
        mask = sup > 0
        updated = self.prototypes.clone()
        updated[mask] = l2n(
            m * updated[mask] + (1 - m) * (delta[mask] / sup[mask].unsqueeze(-1))
        )
        self.prototypes.copy_(updated)

        # обновляем счётчики поддержки
        self.support[mask] += sup[mask].round().to(self.support.dtype)

        # Обновляем цветовую статистику
        if gt_ab_regions is not None:
            # Для дисперсии обновляем второй момент E[X^2]
            # Var(X) = E[X^2] - (E[X])^2
            # В sigma храним E[X^2], в mu — E[X]
            sup_masked = sup[mask].unsqueeze(-1)
            # Обновляем mu (E[X])
            new_mu = m * self.prot_mu_ab[mask] + (1 - m) * (delta_mu[mask] / sup_masked)
            # Обновляем sigma (E[X^2])
            new_sigma = m * self.prot_sigma_ab[mask] + (1 - m) * (
                delta_sigma[mask] / sup_masked
            )

            self.prot_mu_ab[mask] = new_mu
            self.prot_sigma_ab[mask] = new_sigma

        # Обслуживание: опциональная случайная реинициализация слабых прототипов
        if self.reinit_prob > 0 and self.min_support > 0:
            weak = self.support < self.min_support
            if weak.any():
                prob = torch.rand_like(self.support.float())
                to_reinit = weak & (prob < self.reinit_prob)
                if to_reinit.any():
                    # выбираем случайные региональные векторы для реинициализации
                    num = int(to_reinit.sum().item())
                    idxs = torch.nonzero(to_reinit, as_tuple=False).squeeze(-1)
                    # случайный выбор из регионов
                    BxR = regions.reshape(-1, regions.shape[-1])
                    ridx = torch.randint(0, BxR.shape[0], (num,), device=BxR.device)
                    self.prototypes[idxs] = l2n(BxR[ridx])
                    self.support[idxs] = 0
                    # Сбрасываем цветовую статистику для реинициализированных прототипов
                    self.prot_mu_ab[idxs] = 0
                    self.prot_sigma_ab[idxs] = 0

        # DDP-широковещание
        if self.sync_rank0 and dist.is_available() and dist.is_initialized():
            dist.broadcast(self.prototypes, src=0)
            dist.broadcast(self.support, src=0)
            dist.broadcast(self.prot_mu_ab, src=0)
            dist.broadcast(self.prot_sigma_ab, src=0)
