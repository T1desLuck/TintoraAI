import torch
from skimage import color


def lab_to_rgb_tensor(L: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Преобразует отдельные каналы Lab в RGB-тензор в диапазоне [0,1].
    L нормирован в [-1,1]; сперва переводим обратно в 0..100.
    Формы входов: (B,1,H,W)
    """
    B, _, H, W = L.shape
    L_ = (L + 1.0) * 50.0
    lab = torch.cat([L_, a, b], dim=1)  # (B,3,H,W)
    # skimage ожидает NumPy-массив в формате HWC (Lab)
    out = []
    for i in range(B):
        # Отсоединяем тензор перед NumPy-преобразованием, чтобы избежать ошибок autograd
        lab_i = lab[i].detach().permute(1, 2, 0).cpu().numpy()
        rgb_i = color.lab2rgb(lab_i).astype("float32")
        out.append(torch.from_numpy(rgb_i).permute(2, 0, 1))
    rgb = torch.stack(out, dim=0)
    return torch.clamp(rgb, 0.0, 1.0)
