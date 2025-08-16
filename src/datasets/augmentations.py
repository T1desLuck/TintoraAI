from typing import Tuple
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def random_horizontal_flip(img: Image.Image, p: float = 0.5) -> Image.Image:
    """
    С вероятностью `p` выполняет горизонтальное отражение изображения.
    """
    if random.random() < p:
        return ImageOps.mirror(img)
    return img


def _resolve_resample(filter_name: str | int | None) -> int:
    """
    Преобразует строковое имя фильтра ресайза в константу PIL.Image.Resampling.
    Поддерживаемые значения (регистронезависимо): "lanczos" (по умолчанию), "bicubic", "bilinear", "nearest".
    Также допускается передача уже готовой константы PIL (int).
    """
    if isinstance(filter_name, int):
        return filter_name
    name = (filter_name or "lanczos").strip().lower()
    # Совместимость с PIL>=9: константы доступны как Image.LANCZOS и т.д.
    if name in ("lanczos", "l"):  # default
        return Image.LANCZOS
    if name in ("bicubic", "cubic"):
        return Image.BICUBIC
    if name in ("bilinear", "linear"):
        return Image.BILINEAR
    if name in ("nearest", "nn"):
        return Image.NEAREST
    # fallback
    return Image.LANCZOS


def random_resized_crop(img: Image.Image, size: int, scale: Tuple[float, float] = (0.8, 1.0), resample: str | int | None = None) -> Image.Image:
    """
    Случайно вырезает КВАДРАТНУЮ область с площадью из `scale * (w*h)` и
    изменяет размер результата до `size x size` с высоким качеством (LANCZOS).
    Это исключает геометрическое искажение аспектов.
    """
    w, h = img.size
    area = w * h
    side_max = min(w, h)
    res = _resolve_resample(resample)
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        side = int(round(target_area ** 0.5))  # квадратная область
        side = min(side, side_max)
        if side >= 1:
            x1 = random.randint(0, w - side)
            y1 = random.randint(0, h - side)
            img_sq = img.crop((x1, y1, x1 + side, y1 + side))
            return img_sq.resize((size, size), res)
    # Fallback: центр-кроп квадрата по короткой стороне, затем ресайз
    side = side_max
    left = (w - side) // 2
    top = (h - side) // 2
    img_sq = img.crop((left, top, left + side, top + side))
    return img_sq.resize((size, size), res)


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


def resize_shorter_side_and_center_crop(img: Image.Image, size: int, resample: str | int | None = None) -> Image.Image:
    """
    Масштабирует изображение пропорционально так, чтобы КОРОТКАЯ сторона стала `size`,
    затем делает центр-кроп `size x size`. Использует LANCZOS для даунскейла.
    Исключает геометрические искажения.
    """
    w, h = img.size
    if w == 0 or h == 0:
        return img
    res = _resolve_resample(resample)
    # масштабирование по короткой стороне
    if w < h:
        new_w = size
        new_h = int(round(h * (size / w)))
    else:
        new_h = size
        new_w = int(round(w * (size / h)))
    img_resized = img.resize((new_w, new_h), res)
    # центр-кроп до size x size
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    return img_resized.crop((left, top, left + size, top + size))


def resize_shorter_side_then_random_crop(img: Image.Image, size: int, resample: str | int | None = None) -> Image.Image:
    """
    Сначала пропорционально масштабирует изображение так, чтобы короткая сторона стала >= size,
    затем делает СЛУЧАЙНЫЙ кроп `size x size`. Это исключает искажение аспекта, но добавляет вариативность.
    """
    w, h = img.size
    if w == 0 or h == 0:
        return img
    res = _resolve_resample(resample)
    # Масштаб до такой степени, чтобы обе стороны были >= size
    if w < h:
        new_w = size
        new_h = max(size, int(round(h * (size / w))))
    else:
        new_h = size
        new_w = max(size, int(round(w * (size / h))))
    img_resized = img.resize((new_w, new_h), res)
    # Случайный кроп size x size внутри увеличенного изображения
    max_x = new_w - size
    max_y = new_h - size
    x1 = 0 if max_x <= 0 else random.randint(0, max_x)
    y1 = 0 if max_y <= 0 else random.randint(0, max_y)
    return img_resized.crop((x1, y1, x1 + size, y1 + size))


# ------------------------------
# L-channel (grayscale) defects
# ------------------------------

def _to_pil_L_from_Ln(Ln: np.ndarray) -> Image.Image:
    """
    Ln: numpy (H,W,1) float32 in [-1,1]. Convert to 8-bit PIL 'L'.
    """
    L01 = np.clip((Ln + 1.0) * 0.5, 0.0, 1.0)  # [0,1]
    L8 = (L01 * 255.0).astype(np.uint8)
    if L8.ndim == 3 and L8.shape[-1] == 1:
        L8 = L8[..., 0]
    return Image.fromarray(L8, mode="L")


def _from_pil_L_to_Ln(imgL: Image.Image) -> np.ndarray:
    """
    PIL 'L' -> Ln: numpy (H,W,1) float32 in [-1,1].
    """
    arr = np.array(imgL).astype(np.float32) / 255.0
    Ln = arr * 2.0 - 1.0
    return Ln[..., None]


def defect_gaussian_noise(Ln: np.ndarray, std: float) -> np.ndarray:
    """
    Additive Gaussian noise to Ln ([-1,1]). std is in Ln units (recommended 0.005..0.03).
    """
    if std <= 0:
        return Ln
    noise = np.random.normal(0.0, std, size=Ln.shape).astype(np.float32)
    out = Ln + noise
    return np.clip(out, -1.0, 1.0)


def defect_speckle_noise(Ln: np.ndarray, std: float) -> np.ndarray:
    """
    Multiplicative speckle noise: Ln * (1 + N(0,std)). std ~ 0.005..0.03
    """
    if std <= 0:
        return Ln
    mult = 1.0 + np.random.normal(0.0, std, size=Ln.shape).astype(np.float32)
    out = Ln * mult
    return np.clip(out, -1.0, 1.0)


def defect_small_spots(Ln: np.ndarray, p: float = 0.3, max_spots: int = 3, size_range: Tuple[int, int] = (2, 10), strength: float = 0.2) -> np.ndarray:
    """
    Apply small bright/dark square spots on Ln. strength in Ln units (0..1), size in pixels.
    p: probability to apply. max_spots: max number of spots when applied.
    """
    if p <= 0 or random.random() >= p:
        return Ln
    H, W, _ = Ln.shape
    out = Ln.copy()
    n = random.randint(1, max(1, max_spots))
    for _ in range(n):
        sz = random.randint(max(1, size_range[0]), max(size_range[0], size_range[1]))
        x = random.randint(0, max(0, W - sz))
        y = random.randint(0, max(0, H - sz))
        delta = (random.random() * 2 - 1) * strength  # bright or dark
        out[y:y+sz, x:x+sz, 0] = np.clip(out[y:y+sz, x:x+sz, 0] + delta, -1.0, 1.0)
    return out


def defect_blur(Ln: np.ndarray, p: float = 0.2, radius: float = 0.7) -> np.ndarray:
    """
    Slight Gaussian blur on L to simulate minor smearing. radius ~ 0.5..1.5
    """
    if p <= 0 or random.random() >= p or radius <= 0:
        return Ln
    pilL = _to_pil_L_from_Ln(Ln)
    pilB = pilL.filter(ImageFilter.GaussianBlur(radius=radius))
    return _from_pil_L_to_Ln(pilB)


def augment_L_defects(Ln: np.ndarray, cfg: dict | None) -> np.ndarray:
    """
    Apply a suite of small defects on L-channel (Ln in [-1,1]) controlled by cfg dict.
    cfg example:
      {
        "enabled": true,
        "gauss_std": 0.01,
        "speckle_std": 0.0,
        "spots_p": 0.2,
        "spots_max": 2,
        "spots_size": [2, 8],
        "spots_strength": 0.15,
        "blur_p": 0.15,
        "blur_radius": 0.8
      }
    Defaults: no-ops.
    """
    if cfg is None or not bool(cfg.get("enabled", False)):
        return Ln
    out = Ln
    # Gaussian noise
    out = defect_gaussian_noise(out, float(cfg.get("gauss_std", 0.0)))
    # Speckle noise
    out = defect_speckle_noise(out, float(cfg.get("speckle_std", 0.0)))
    # Small spots
    spots_size = cfg.get("spots_size", [2, 8])
    if isinstance(spots_size, (list, tuple)) and len(spots_size) == 2:
        size_rng = (int(spots_size[0]), int(spots_size[1]))
    else:
        size_rng = (2, 8)
    out = defect_small_spots(
        out,
        p=float(cfg.get("spots_p", 0.0)),
        max_spots=int(cfg.get("spots_max", 2)),
        size_range=size_rng,
        strength=float(cfg.get("spots_strength", 0.15)),
    )
    # Blur
    out = defect_blur(out, p=float(cfg.get("blur_p", 0.0)), radius=float(cfg.get("blur_radius", 0.8)))
    return out
