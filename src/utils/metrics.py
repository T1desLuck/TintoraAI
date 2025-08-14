from __future__ import annotations
import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window_size) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).float()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = (kernel_1d.transpose(1, 2) @ kernel_1d).squeeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """
    Вычисляет SSIM для каждого изображения и возвращает среднее по батчу.
    Ожидает img1, img2 в диапазоне [0,1] и формы (B,C,H,W).
    """
    assert img1.shape == img2.shape, "SSIM: input shapes must match"
    B, C, H, W = img1.shape
    device = img1.device
    kernel = _gaussian_kernel(window_size, sigma, C).to(device)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    # усредняем по C,H,W, затем по батчу
    return ssim_map.mean(dim=(1, 2, 3)).mean()


class _LPIPSWrapper:
    def __init__(self):
        self.enabled = False
        self.metric = None
        try:
            import lpips  # type: ignore
            self.metric = lpips.LPIPS(net='vgg')
            self.enabled = True
        except Exception:
            self.metric = None
            self.enabled = False

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled or self.metric is None:
            return None
        # LPIPS ожидает диапазон [-1,1]
        x1 = img1 * 2 - 1
        x2 = img2 * 2 - 1
        with torch.no_grad():
            val = self.metric(x1, x2)
        # возвращает (B,1,1,1) или (B)
        return val.mean()


def try_lpips() -> _LPIPSWrapper:
    return _LPIPSWrapper()
