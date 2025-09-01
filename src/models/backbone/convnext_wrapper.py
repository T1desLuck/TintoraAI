from typing import Tuple, List
import torch
import torch.nn as nn
import timm


class ConvNeXtTiny(nn.Module):
    """Обёртка для ConvNeXt-Tiny с использованием timm, поддерживающая
    одноканальный вход и корректный возврат промежуточных карт признаков.
    """

    def __init__(self, in_channels: int = 1, out_indices: Tuple[int, ...] = (0,), **kwargs) -> None:
        super().__init__()
        # Используем timm.create_model, который надёжно работает для извлечения признаков
        self.model = timm.create_model(
            "convnext_tiny",
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
            pretrained=False,  # Без предобученных весов согласно ТЗ
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Прямой проход возвращает список карт признаков с указанных стадий."""
        return self.model(x)
