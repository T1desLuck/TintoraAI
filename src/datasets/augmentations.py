from typing import Tuple
import random
import numpy as np
from PIL import Image, ImageOps


def random_horizontal_flip(img: Image.Image, p: float = 0.5) -> Image.Image:
    """
    С вероятностью `p` выполняет горизонтальное отражение изображения.
    """
    if random.random() < p:
        return ImageOps.mirror(img)
    return img


def random_resized_crop(img: Image.Image, size: int, scale: Tuple[float, float] = (0.8, 1.0)) -> Image.Image:
    """
    Случайно вырезает прямоугольную область с масштабом из `scale` и
    изменяет размер результата до `size x size`.
    """
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(3 / 4, 4 / 3)
        new_w = int(round((target_area * aspect) ** 0.5))
        new_h = int(round((target_area / aspect) ** 0.5))
        if new_w <= w and new_h <= h:
            x1 = random.randint(0, w - new_w)
            y1 = random.randint(0, h - new_h)
            img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            return img.resize((size, size), Image.BILINEAR)
    return img.resize((size, size), Image.BILINEAR)


def color_jitter_lab(ab: np.ndarray, jitter: float = 0.05) -> np.ndarray:
    """
    Добавляет гауссов шум к каналам `ab` в пространстве Lab с дисперсией,
    задаваемой `jitter` (в долях от диапазона Lab ~128).
    """
    if jitter <= 0:
        return ab
    noise = np.random.normal(0.0, jitter * 128.0, size=ab.shape).astype(np.float32)
    out = ab + noise
    out[..., 0] = np.clip(out[..., 0], -128, 127)
    out[..., 1] = np.clip(out[..., 1], -128, 127)
    return out
