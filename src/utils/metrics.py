from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from contextlib import contextmanager
import warnings


@contextmanager
def _temp_sys_prefix(path: str):
    old = sys.prefix
    try:
        sys.prefix = path
        yield
    finally:
        sys.prefix = old


def _ensure_dists_weights() -> Optional[Path]:
    """
    Гарантирует наличие официального weights.pt для DISTS.
    Скачивает в пользовательский кэш, если отсутствует.
    Возвращает путь к каталогу, содержащему weights.pt, либо None при неудаче.
    """
    try:
        cache_dir = Path.home() / ".cache" / "tintoraai" / "dists"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / "weights.pt"
        if not weights_path.exists():
            import urllib.request

            url = "https://github.com/dingkeyan93/DISTS/raw/master/weights.pt"
            urllib.request.urlretrieve(url, str(weights_path))
        return cache_dir if weights_path.exists() else None
    except Exception:
        return None


def _gaussian_kernel(
    window_size: int = 11, sigma: float = 1.5, channels: int = 3
) -> torch.Tensor:
    coords = torch.arange(window_size) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).float()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = (kernel_1d.transpose(1, 2) @ kernel_1d).squeeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return kernel


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Вычисляет SSIM для каждого изображения и возвращает среднее по батчу.
    Ожидает img1, img2 в диапазоне [0,1] и формы (B,C,H,W).
    """
    assert img1.shape == img2.shape, "SSIM: input shapes must match"
    B, C, H, W = img1.shape
    device = img1.device
    # Ensure kernel matches input dtype to support float64 inputs
    kernel = _gaussian_kernel(window_size, sigma, C).to(device=device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1_mu2
    )

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    # усредняем по C,H,W, затем по батчу
    return ssim_map.mean(dim=(1, 2, 3)).mean()


class _DISTSWrapper:
    def __init__(self):
        self.enabled = False
        self.metric = None
        try:
            # DISTS expects inputs in [0,1]
            # Некоторые версии DISTS/torchvision порождают Deprecation/UserWarning
            # про параметр 'pretrained' и позиционные 'weights'. Подавим только эти
            # внешние предупреждения на время инициализации, не меняя поведение.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*pretrained.*deprecated.*|.*Arguments other than a weight enum.*",
                    category=UserWarning,
                    module=r"torchvision\.models\._utils",
                )
                from DISTS_pytorch import DISTS  # type: ignore

                try:
                    self.metric = DISTS()
                    self.enabled = True
                except FileNotFoundError:
                    # Официальный пакет ищет weights.pt в sys.prefix. Если недоступно — попробуем скачать и подменить префикс.
                    cache_dir = _ensure_dists_weights()
                    if cache_dir is not None:
                        with _temp_sys_prefix(str(cache_dir)):
                            self.metric = DISTS()
                            self.enabled = True
                    else:
                        self.metric = None
                        self.enabled = False
        except Exception:
            self.metric = None
            self.enabled = False

    def __call__(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.enabled or self.metric is None:
            return None
            
        device = img1.device
        self.metric = self.metric.to(device)
        
        # DISTS принимает тензоры в диапазоне [0,1]
        with torch.no_grad():
            val = self.metric(img1, img2)
        # Возвращаем усреднённое значение по батчу
        return val.mean()


def try_dists() -> _DISTSWrapper:
    return _DISTSWrapper()


def preload_dists() -> None:
    """
    Предзагружает/инициализирует DISTS, чтобы исключить задержки на первой валидации.
    Безопасно к ошибкам: при отсутствии пакета просто ничего не делает.
    """
    try:
        from DISTS_pytorch import DISTS  # type: ignore
    except Exception:
        return
    try:
        try:
            metric = DISTS()
        except FileNotFoundError:
            cache_dir = _ensure_dists_weights()
            if cache_dir is None:
                return
            with _temp_sys_prefix(str(cache_dir)):
                metric = DISTS()
        # Короткий прогон на крошечном тензоре [0,1], чтобы инициализировать внутренние веса/буферы
        with torch.no_grad():
            x = torch.zeros(1, 3, 8, 8, dtype=torch.float32)
            _ = metric(x, x)
    except Exception:
        # Тихо игнорируем, предзагрузка — best-effort
        pass


def ciede2000(
    lab1: torch.Tensor, lab2: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Вычисляет среднее Delta E 2000 (CIEDE2000) между batched Lab-изображениями.

    Ожидается lab тензоры формы (B,3,H,W) в абсолютных единицах:
      - L в [0,100]
      - a,b приблизительно в [-128,127]

    Возвращает скалярный тензор (усреднение по B,H,W).
    Реализация векторизована и следует формуле Sharma et al., 2005.
    """
    assert lab1.shape == lab2.shape, "ciede2000: shapes must match"
    assert lab1.dim() == 4 and lab1.size(1) == 3, "ciede2000: expected (B,3,H,W)"

    # Разворачиваем по каналам
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    # Шаг 1: вычисление C' и h' для обоих цветов
    C1 = torch.sqrt(a1 * a1 + b1 * b1 + eps)
    C2 = torch.sqrt(a2 * a2 + b2 * b2 + eps)
    C_bar = 0.5 * (C1 + C2)

    # G фактор
    C_bar7 = C_bar.pow(7)
    G = 0.5 * (1.0 - torch.sqrt(C_bar7 / (C_bar7 + (25.0**7) + eps)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1 + eps)
    C2p = torch.sqrt(a2p * a2p + b2 * b2 + eps)

    # Углы в градусах 0..360
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0

    # Шаг 2: dL', dC', dH'
    dLp = L2 - L1
    dCp = C2p - C1p

    # dHp с учётом циклической природы угла
    dhp = h2p - h1p
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * torch.sqrt(C1p * C2p + eps) * torch.sin(torch.deg2rad(0.5 * dhp))

    # Шаг 3: средние значения
    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    # Среднее h' корректно по сектору
    hsum = h1p + h2p
    habs = (h1p - h2p).abs()
    h_bar = torch.where(
        (C1p * C2p) < eps,
        hsum * 0.0,  # если хотя бы одна насыщенность почти нулевая — h_bar не влияет
        torch.where(
            habs <= 180.0,
            0.5 * hsum,
            0.5 * (hsum + torch.where(hsum < 360.0, 360.0, -360.0)),
        ),
    )

    # T модификатор
    T = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(h_bar - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * h_bar))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * h_bar + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * h_bar - 63.0))
    )

    # SL, SC, SH весовые функции
    Sl = 1.0 + (0.015 * (Lp_bar - 50.0).pow(2)) / torch.sqrt(
        20.0 + (Lp_bar - 50.0).pow(2) + eps
    )
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T

    # Δθ и RC, RT
    delta_theta = 30.0 * torch.exp(-((h_bar - 275.0) / 25.0).pow(2))
    Rc = 2.0 * torch.sqrt(Cp_bar.pow(7) / (Cp_bar.pow(7) + (25.0**7) + eps))
    Rt = -torch.sin(torch.deg2rad(2.0 * delta_theta)) * Rc

    # Собираем итог
    kL = kC = kH = 1.0
    dE = torch.sqrt(
        (dLp / (kL * Sl + eps)).pow(2)
        + (dCp / (kC * Sc + eps)).pow(2)
        + (dHp / (kH * Sh + eps)).pow(2)
        + Rt * (dCp / (kC * Sc + eps)) * (dHp / (kH * Sh + eps))
        + eps
    )

    return dE.mean()
