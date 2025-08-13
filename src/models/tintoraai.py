from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvNeXtTiny, CoAtNetLight, GATLight
from .heads import DepthHead, IlluminationHead
from .guidenet import GuideNet
from .omm import ObjectMemoryModule
from .crb import ColorReasoningBlock
from .decoder.decoder_unetpp import UNetPPDecoder


def l2_normalize_feature(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # L2-нормализация по каналам для каждой пространственной позиции
    norm = torch.sqrt(torch.clamp((x * x).sum(dim=1, keepdim=True), min=eps))
    return x / norm


class TintoraAI(nn.Module):
    """
    Каркас TintoraAI с интегрированными бэкбонами.
    Вход: L (B,1,H,W). Выходы бэкбона:
      F1: (B,c1,H/4,W/4), F2: (B,c2,H/8,W/8), F3: (B,c3,H/16,W/16)
    F2/F3 проходят L2-нормализацию. Головные модули выступают проекциями,
    и далее апсемплируются из F3.
    """

    def __init__(
        self,
        c1: int = 96,
        c2: int = 192,
        c3: int = 384,
        film_dim: int = 256,
        use_guidenet: bool = False,
        guide_feature_dim: Optional[int] = None,
        guide_out_dim: Optional[int] = None,
        omm_config: Optional[Dict] = None,
        use_saturation_head: bool = True,
    ):
        super().__init__()
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.film_dim = film_dim
        self.use_guidenet = use_guidenet

        # Стадии бэкбона
        self.stage1 = ConvNeXtTiny()
        self.stage2 = CoAtNetLight()
        self.stage3 = GATLight(in_channels=192, out_channels=384)
        self.coatnet_channel_fix = nn.Conv2d(256, 192, kernel_size=1)

        # Модуль объектной памяти (OMM) и проекционный слой
        if omm_config is None:
            omm_config = {}
        self.omm_dim = omm_config.get("D", 256)
        self.omm_proj = nn.Conv2d(c2, self.omm_dim, kernel_size=1) if c2 != self.omm_dim else nn.Identity()

        # Передаём все параметры конфига в OMM — он сам обработает значения по умолчанию
        self.omm = ObjectMemoryModule(
            dim=self.omm_dim,
            num_prototypes=omm_config.get("N", 2048),
            momentum=omm_config.get("alpha", 0.995),
            temperature=omm_config.get("tau", 0.07),
            topk=omm_config.get("top_k", 64),
            min_support=omm_config.get("min_support", 15),
            sync_rank0=omm_config.get("sync", {}).get("enabled", True),
            # Pass other potential params from config
            **omm_config.get("extra_params", {})
        )

        # Вспомогательные головы
        self.depth_head = DepthHead(c2=c2, c3=c3, mid=128)
        self.illum_head = IlluminationHead(c2=c2, c3=c3, mid=128)

        # CRB и декодер (FiLM U-Net++)
        self.ctx_hidden = 256
        self.crb = ColorReasoningBlock(c3=c3, cmem=self.omm_dim, film_stage_dims=(c3, c2, c1), geom_ch=1+1+3, ctx_hidden=self.ctx_hidden)
        self.decoder = UNetPPDecoder(c1=c1, c2=c2, c3=c3, mid=128)
        self.ab_head = nn.Conv2d(self.decoder.output_channels, 2, kernel_size=1)
        self.sat_head = nn.Conv2d(self.decoder.output_channels, 1, kernel_size=1) if use_saturation_head else None
        # GuideNet (опционально)
        if self.use_guidenet:
            g_in = guide_feature_dim if guide_feature_dim is not None else c3
            g_out = guide_out_dim if guide_out_dim is not None else self.ctx_hidden
            self.guide = GuideNet(feature_dim=g_in, out_dim=g_out)
            self.pool_f3 = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.guide = None

    def forward(self, L: torch.Tensor, gt_ab: Optional[torch.Tensor] = None, omm_read_only: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        B, _, H, W = L.shape
        # Бэкбон
        F1 = self.stage1(L)[-1]  # (B,c1,H/4,W/4)
        # CoAtNetLight возвращает карты признаков для стадий 0, 1, 2. Нужна стадия 1 (192 канала).
        F2_list = self.stage2(F1)
        F2 = F2_list[2]  # Фактически (B, 256, H/8, W/8)
        F2 = self.coatnet_channel_fix(F2) # Приводим число каналов к 192
        F3 = self.stage3(F2)[-1]  # (B,c3,H/16,W/16)

        # L2-нормализация F2/F3 согласно ТЗ
        F2n = l2_normalize_feature(F2)
        F3n = l2_normalize_feature(F3)

        # Чтение/обновление OMM
        F2_for_omm = self.omm_proj(F2n)
        omm_out = self.omm(F2_for_omm, gt_ab=gt_ab, read_only=omm_read_only)  # учебный план может «замораживать» OMM
        mem = omm_out["mem"]  # (B,R,D=omm_dim)
        g_h, g_w = omm_out["grid_hw"]
        # преобразуем в карту сетки и апсемплим до размера изображения
        mem_map = mem.transpose(1, 2).reshape(B, self.omm_dim, g_h, g_w)
        mem_map = F.interpolate(mem_map, size=(H, W), mode="bilinear", align_corners=False)

        # Карты глубины и освещённости из голов
        D = self.depth_head(F2n, F3n, (H, W))
        I = self.illum_head(F2n, F3n, (H, W))

        # Вычисляем нормали из глубины через градиенты Собеля
        normals = self.compute_normals(D)

        # CRB генерирует параметры FiLM; декодер предсказывает ab/sat
        guide_ctx = None
        if self.guide is not None:
            # Собираем семантический вектор из F3 и пропускаем через GuideNet
            f3_vec = self.pool_f3(F3n).flatten(1)  # (B,c3)
            # Если вход GuideNet меньше c3, ожидается, что GuideNet настроен под guide_feature_dim
            guide_ctx = self.guide(f3_vec)  # (B, ctx_hidden)

        film = self.crb(F3n, mem_map, D, I, normals, guide_ctx=guide_ctx)  # {gamma:(S,B,C), beta:(S,B,C)}
        x_ab, x_sat_raw = self.decoder(F1, F2n, F3n, film["gamma"], film["beta"], (H, W))
        a, b = torch.chunk(x_ab, 2, dim=1)

        out = {
            "a": a,
            "b": b,
            "D": D,
            "I": I,
            "normals": normals,
            "F2": F2n,
            "F3": F3n,
            "F2_omm": F2_for_omm,
            "mem_map": mem_map,
            "mem": mem,
        }

        if self.sat_head is not None and x_sat_raw is not None:
            s = torch.sigmoid(x_sat_raw)
            out["sat"] = s

        if self.guide is not None:
            out["guide_params"] = guide_ctx

        if omm_out.get("cluster_loss") is not None:
            out["cluster_loss"] = omm_out["cluster_loss"]

        return out

    @staticmethod
    def compute_normals(D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Вычисляет приближённые нормали поверхности по карте глубины D в [0,1].
        Возвращает (B,3,H,W): нормализованные Nx, Ny, Nz.
        """
        B, C, H, W = D.shape
        # Ядра Собеля
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=D.dtype, device=D.device).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=D.dtype, device=D.device).view(1, 1, 3, 3)
        Dx = F.conv2d(D, kx, padding=1)
        Dy = F.conv2d(D, ky, padding=1)
        Nx = -Dx
        Ny = -Dy
        Nz = torch.ones_like(D)
        N = torch.cat([Nx, Ny, Nz], dim=1)
        norm = torch.sqrt(torch.clamp((N * N).sum(dim=1, keepdim=True), min=eps))
        N = N / norm
        return N
