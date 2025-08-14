from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GuideNet(nn.Module):
    """
    GuideNet: модуль цветовых советов на основе семантических признаков.
    Параметры:
      - feature_dim: входная размерность семантического вектора (маппится на hidden_dim)
      - num_layers: количество слоёв MLP
      - out_dim: размерность выходного вектора подсказок (например, FiLM/контекст)
    Примечание: по требованию совместимости feature_dim -> hidden_dim.
    """

    def __init__(self, feature_dim: int = 256, num_layers: int = 2, out_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        hidden_dim = feature_dim  # маппинг feature_dim -> hidden_dim
        dims = [feature_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, feature_dim) — семантический вектор/пуллинг признаков.
        return: (B, out_dim) — вектор цветовых советов/контекста.
        """
        return self.mlp(x)
