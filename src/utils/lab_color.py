import torch
from kornia.color import lab_to_rgb as kornia_lab_to_rgb


def lab_to_rgb_tensor(
    L: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
     Преобразует отдельные каналы Lab в RGB-тензор в диапазоне [0,1].
     - Ожидается: L в диапазоне [-1, 1] (нормализованный), a/b в исходных единицах Lab.
     - Формы входов: (B,1,H,W)
     Реализация полностью тензорная и векторизованная (Kornia), без NumPy/CPU-циклов.
     """
    # Масштабируем L обратно к [0, 100]
    L_ = (L + 1.0) * 50.0
    # Клиппим допустимые диапазоны Lab
    L_ = torch.clamp(L_, 0.0, 100.0)
    a = torch.clamp(a, -128.0, 127.0)
    b = torch.clamp(b, -128.0, 127.0)

    # Собираем Lab-тензор (B,3,H,W) и конвертируем через Kornia (поддерживает CUDA)
    lab = torch.cat([L_, a, b], dim=1)
    rgb = kornia_lab_to_rgb(lab)
    return torch.clamp(rgb, 0.0, 1.0)
