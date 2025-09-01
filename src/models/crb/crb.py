from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """
    Генерирует параметры FiLM (gamma, beta) для нескольких стадий декодера
    на основе слитых контекстных признаков (F3, память, геометрия).
    """

    def __init__(
        self,
        in_dim: int,
        stage_dims: Tuple[int, ...] = (384, 192, 96),
        hidden: int = 256,
    ):
        super().__init__()
        self.stage_dims = stage_dims
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # головки для каждой стадии
        self.gamma = nn.ModuleList([nn.Linear(hidden, d) for d in stage_dims])
        self.beta = nn.ModuleList([nn.Linear(hidden, d) for d in stage_dims])

    def forward(self, ctx: torch.Tensor) -> Dict[str, list[torch.Tensor]]:
        # ctx: (B, in_dim)
        h = self.trunk(ctx)
        out_gamma = []
        out_beta = []
        for g, b in zip(self.gamma, self.beta):
            out_gamma.append(g(h))  # (B, C)
            out_beta.append(b(h))  # (B, C)
        return {"gamma": out_gamma, "beta": out_beta}


class ColorReasoningBlock(nn.Module):
    """
    CRB: объединяет семантику/внешний вид (F3), память (mem_map) и геометрию (D/I/нормали)
    в компактный контекстный вектор и генерирует параметры FiLM для каждой стадии декодера.
    """

    def __init__(
        self,
        c3: int,
        cmem: int,
        film_stage_dims=(384, 192, 96),
        geom_ch: int = 1 + 1 + 3,
        ctx_hidden: int = 256,
    ):
        super().__init__()
        self.film_stage_dims = film_stage_dims
        # Снижение числа каналов и слияние
        self.reduce_f3 = nn.Conv2d(c3, 128, kernel_size=1)
        self.reduce_mem = nn.Conv2d(cmem, 64, kernel_size=1)
        self.reduce_geom = nn.Conv2d(geom_ch, 32, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(128 + 64 + 32, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(160, ctx_hidden)
        self.film = FiLMGenerator(
            in_dim=ctx_hidden, stage_dims=film_stage_dims, hidden=ctx_hidden
        )

    def forward(
        self,
        F3: torch.Tensor,
        mem_map: torch.Tensor,
        D: torch.Tensor,
        illum_map: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        *,
        I: torch.Tensor | None = None,
        guide_ctx: torch.Tensor | None = None,
    ) -> Dict[str, list[torch.Tensor]]:
        B, _, H, W = F3.shape
        # даунсэмплируем память/геометрию к разрешению F3
        mem_ds = F.interpolate(
            mem_map, size=(H, W), mode="bilinear", align_corners=False
        )
        # поддержка синонима параметра освещения через ключевое слово I
        if illum_map is None and I is not None:
            illum_map = I
        if illum_map is None or normals is None:
            raise TypeError("illum_map (or I) and normals must be provided")
        geom = torch.cat([D, illum_map, normals], dim=1)
        geom_ds = F.interpolate(geom, size=(H, W), mode="bilinear", align_corners=False)

        f3r = self.reduce_f3(F3)
        memr = self.reduce_mem(mem_ds)
        geomr = self.reduce_geom(geom_ds)
        x = torch.cat([f3r, memr, geomr], dim=1)
        x = self.fuse(x)
        x = self.pool(x).flatten(1)  # (B,160)
        ctx = self.proj(x)  # (B,ctx_hidden)
        if guide_ctx is not None:
            # ожидаем (B, ctx_hidden); если иначе — приведите на стороне вызова
            if guide_ctx.shape[-1] == ctx.shape[-1]:
                ctx = ctx + guide_ctx
            else:
                raise ValueError(
                    f"guide_ctx dim {guide_ctx.shape[-1]} != ctx_hidden {ctx.shape[-1]}"
                )
        film = self.film(ctx)  # {gamma:(S,B,C), beta:(S,B,C)}
        return film
