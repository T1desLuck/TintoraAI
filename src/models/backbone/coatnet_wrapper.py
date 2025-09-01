from typing import Tuple, List
import torch
import torch.nn as nn
import timm


class CoAtNetLight(nn.Module):
    """Обёртка для CoAtNet-light на базе timm, поддерживающая одноканальный вход
    и корректный возврат промежуточных карт признаков.
    """

    def __init__(self, in_channels: int = 96, out_indices: Tuple[int, ...] = (0, 1, 2), **kwargs) -> None:
        super().__init__()
        # Используем timm.create_model — надёжный способ для извлечения признаков
        self.model = timm.create_model(
            "coatnet_rmlp_2_rw_224",
            pretrained=False,
            features_only=True,
            in_chans=in_channels,
            out_indices=out_indices,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Прямой проход возвращает список карт признаков с указанных стадий."""
        return self.model(x)
