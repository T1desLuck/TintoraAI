from pathlib import Path
from typing import Tuple, List, Optional
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage import color

from .augmentations import random_horizontal_flip, random_resized_crop, color_jitter_lab


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    lab = color.rgb2lab(img)
    return lab


def to_L_and_ab(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = lab[..., 0:1]  # 0..100 (яркость)
    a = lab[..., 1:2]
    b = lab[..., 2:3]
    # Нормируем L в диапазон [-1, 1]
    Ln = (L / 50.0) - 1.0
    ab = np.concatenate([a, b], axis=-1)
    return Ln, ab


class AdvancedColorizationDataset(Dataset):
    """
    Датасет с базовыми аугментациями для обучения и детерминированным изменением
    размера на валидации. Возвращает кортеж (L, ab, path).
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        train: bool = True,
        aug_flip: float = 0.5,
        aug_crop_scale: Tuple[float, float] = (0.8, 1.0),
        aug_ab_jitter: float = 0.05,
    ):
        self.root = Path(root_dir)
        self.paths: List[Path] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if self.root.exists():
            for p in self.root.rglob("*"):
                if p.suffix.lower() in exts:
                    self.paths.append(p)
        self.image_size = image_size
        self.train = train
        self.aug_flip = aug_flip
        self.aug_crop_scale = aug_crop_scale
        self.aug_ab_jitter = aug_ab_jitter

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path: Path) -> Image.Image:
        # Чтение и приведение к RGB
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = self._load_image(path)
        if self.train:
            # Аугментации для обучения
            img = random_resized_crop(img, self.image_size, self.aug_crop_scale)
            img = random_horizontal_flip(img, self.aug_flip)
        else:
            # Детерминированное изменение размера на валидации
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.array(img)
        lab = rgb_to_lab(arr)
        Ln, ab = to_L_and_ab(lab)
        if self.train and self.aug_ab_jitter > 0:
            # Лёгкая цветовая джиттер-аугментация в Lab (только ab)
            ab = color_jitter_lab(ab, jitter=self.aug_ab_jitter)

        L = torch.from_numpy(Ln.transpose(2, 0, 1)).float()
        ab_t = torch.from_numpy(ab.transpose(2, 0, 1)).float()
        return L, ab_t, str(path)
