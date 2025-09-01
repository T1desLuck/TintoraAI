import torch
import torch.nn as nn


class L1LabLoss(nn.Module):
    """
    Простой L1-лосс в пространстве Lab для каналов a и b.
    Суммирует L1 по предсказанным `a_pred`, `b_pred` и целевым каналам `ab_gt`.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self, a_pred: torch.Tensor, b_pred: torch.Tensor, ab_gt: torch.Tensor
    ) -> torch.Tensor:
        return self.l1(a_pred, ab_gt[:, :1]) + self.l1(b_pred, ab_gt[:, 1:2])
