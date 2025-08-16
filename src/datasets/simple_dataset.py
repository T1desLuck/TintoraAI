from .augmentations import (
    resize_shorter_side_and_center_crop,
    resize_shorter_side_then_random_crop,
    random_resized_crop,
)
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import color


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    lab = color.rgb2lab(img)
    return lab


def to_L_and_ab(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = lab[..., 0:1]  # 0..100 (яркостная компонента)
    a = lab[..., 1:2]
    b = lab[..., 2:3]
    # Нормируем L в диапазон [-1, 1]
    Ln = (L / 50.0) - 1.0
    ab = np.concatenate([a, b], axis=-1)
    return Ln, ab


class SimpleColorizationDataset(Dataset):
    """
    Минимальный датасет: читает изображения из директории и возвращает
    кортеж (L, ab, path). Предназначен для каркаса/тестирования.
    """

    def __init__(self, root_dir: str, image_size: int = 256, geom_mode: str = "center_crop", resize_filter: str = "lanczos"):
        self.root = Path(root_dir)
        self.paths: List[Path] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if self.root.exists():
            for p in self.root.rglob("*"):
                if p.suffix.lower() in exts:
                    self.paths.append(p)
        self.image_size = image_size
        self.geom_mode = (geom_mode or "center_crop").lower()
        self.resize_filter = (resize_filter or "lanczos")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        # Геометрия по выбору (по умолчанию center_crop)
        mode = self.geom_mode
        if mode == "random_resized_crop":
            img = random_resized_crop(img, self.image_size, resample=self.resize_filter)
        elif mode == "random_crop":
            img = resize_shorter_side_then_random_crop(img, self.image_size, resample=self.resize_filter)
        else:
            img = resize_shorter_side_and_center_crop(img, self.image_size, resample=self.resize_filter)
        arr = np.array(img)
        lab = rgb_to_lab(arr)
        Ln, ab = to_L_and_ab(lab)
        # Преобразуем в тензоры PyTorch (BCHW далее)
        L = torch.from_numpy(Ln.transpose(2, 0, 1))  # (1,H,W)
        ab_t = torch.from_numpy(ab.transpose(2, 0, 1))  # (2,H,W)
        return L.float(), ab_t.float(), str(path)
