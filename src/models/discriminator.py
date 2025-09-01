from __future__ import annotations
import torch
import torch.nn as nn


def spectral_norm_if(module: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_sn else module


class PatchDiscriminator(nn.Module):
    """
    PatchGAN-дискриминатор (70x70 style) для адверсариального обучения.

    Параметры:
      - input_nc: число входных каналов (обычно 3 для RGB)
      - ndf: базовое число каналов фильтров
      - n_layers: количество промежуточных свёрточных слоёв
      - use_spectral_norm: использовать ли spectral norm
    Выход: карта (B,1,H/patch,W/patch) логитов.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        kw = 4
        padw = 1
        layers = []
        # первый слой
        layers += [
            spectral_norm_if(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                spectral_norm_if(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                    ),
                    use_spectral_norm,
                ),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            spectral_norm_if(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                ),
                use_spectral_norm,
            ),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # выходной слой (1 канал логитов)
        layers += [
            spectral_norm_if(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                use_spectral_norm,
            )
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
