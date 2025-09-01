from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Универсальный GAN лосс для дискриминатора/генератора.
    Поддержка: 'bce' и 'hinge'.

    Параметры:
      - loss_type: Literal['bce','hinge']
      - real_label: float (для BCE)
      - fake_label: float (для BCE)
    Использование:
      - для D: loss_D_real = loss(pred_real, True, for_discriminator=True)
               loss_D_fake = loss(pred_fake, False, for_discriminator=True)
      - для G: loss_G = loss(pred_fake, True, for_discriminator=False)
    """

    def __init__(
        self,
        loss_type: Literal["bce", "hinge"] = "hinge",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.real_label = real_label
        self.fake_label = fake_label
        if loss_type == "bce":
            self.crit: Optional[nn.BCEWithLogitsLoss] = nn.BCEWithLogitsLoss()
        else:
            self.crit = None  # hinge реализуем вручную

    def forward(
        self, pred: torch.Tensor, is_real: bool, for_discriminator: bool = True
    ) -> torch.Tensor:
        if self.loss_type == "bce":
            target_val = self.real_label if is_real else self.fake_label
            target = torch.full_like(pred, fill_value=target_val)
            assert self.crit is not None
            return self.crit(pred, target)
        # Версия hinge
        if for_discriminator:
            if is_real:
                # макс(0, 1 - предсказание)
                return F.relu(1.0 - pred).mean()
            else:
                # макс(0, 1 + предсказание)
                return F.relu(1.0 + pred).mean()
        # Генератор (hinge): максимизировать D(fake) = минимизировать -pred
        return (-pred).mean()
