import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
from skimage import color

# Используем существующие утилиты/аугментации проекта
from src.utils import load_config
from src.datasets.augmentations import (
    random_resized_crop,
    random_horizontal_flip,
    color_jitter_lab,
    resize_shorter_side_and_center_crop,
    resize_shorter_side_then_random_crop,
    augment_L_defects,
)
from src.datasets.augmentations import _resolve_resample


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def apply_pipeline(
    img: Image.Image,
    mode: str,
    dataset_kind: str,
    image_size: int,
    flip_p: float,
    crop_scale,
    ab_jitter: float,
    disable_flip: bool,
    disable_jitter: bool,
    enable_defects: bool,
    defects_cfg: dict | None,
    geom_mode: str = "random_crop",
    resize_filter: str = "lanczos",
) -> dict:
    """
    Возвращает словарь с ключами:
    - original_rgb: PIL.Image
    - processed_rgb_geom: PIL.Image (после геометрических препроцессов до конвертации в Lab)
    - processed_rgb_lab: PIL.Image (после конвертации в Lab и обратной сборки с jitter ab, если включён)
    - notes: str (краткое описание применённых шагов)
    """
    notes = []
    original = img.copy()
    notes.append(f"original_size={img.size}")

    res = _resolve_resample(resize_filter)
    sel_mode = (
        geom_mode or ("random_crop" if mode == "train" else "center_crop")
    ).lower()
    if sel_mode == "random_resized_crop":
        img2 = random_resized_crop(
            img, image_size, scale=tuple(crop_scale), resample=res
        )
        notes.append(
            f"random_resized_crop(scale={tuple(crop_scale)}) -> {image_size}x{image_size}"
        )
    elif sel_mode == "random_crop":
        img2 = resize_shorter_side_then_random_crop(img, image_size, resample=res)
        notes.append(
            f"resize_shorter_side_then_random_crop -> {image_size}x{image_size}"
        )
    else:  # center_crop
        img2 = resize_shorter_side_and_center_crop(img, image_size, resample=res)
        notes.append(
            f"resize_shorter_side_and_center_crop -> {image_size}x{image_size}"
        )
    if (dataset_kind == "advanced") and (mode == "train") and (not disable_flip):
        img2 = random_horizontal_flip(img2, p=flip_p)
        notes.append(f"random_horizontal_flip(p={flip_p})")
    processed_geom = img2

    notes.append(f"processed_geom_size={processed_geom.size}")
    # Конвертация в Lab и обратная сборка с jitter ab (как в датасете)
    arr = pil_to_np(processed_geom).astype(np.float32)
    lab = color.rgb2lab(arr / 255.0)
    L = lab[..., 0:1]  # 0..100
    a = lab[..., 1:2]
    b = lab[..., 2:3]
    # Нормализация L в [-1,1] как в датасете (для соответствия пайплайну)
    Ln = (L / 50.0) - 1.0
    ab = np.concatenate([a, b], axis=-1).astype(np.float32)

    # При необходимости — применяем мелкие дефекты на L-канал (как в AdvancedColorizationDataset)
    L_gray_before = None
    if (
        mode == "train"
        and dataset_kind == "advanced"
        and enable_defects
        and isinstance(defects_cfg, dict)
        and bool(defects_cfg.get("enabled", False))
    ):
        # Сохраним до-дефектный L для сравнения
        L_back0 = np.clip((Ln + 1.0) * 50.0, 0.0, 100.0)
        L_gray_8u0 = (np.clip(L_back0, 0.0, 100.0) / 100.0 * 255.0).astype(np.uint8)
        if L_gray_8u0.ndim == 3 and L_gray_8u0.shape[-1] == 1:
            L_gray_8u0 = L_gray_8u0[..., 0]
        L_gray_before = np_to_pil(L_gray_8u0).convert("L")
        # Применяем дефекты
        Ln = augment_L_defects(Ln, defects_cfg)
        notes.append("augment_L_defects(enabled)")

    if (
        (not disable_jitter)
        and ab_jitter > 0
        and mode == "train"
        and dataset_kind == "advanced"
    ):
        ab = color_jitter_lab(ab, jitter=float(ab_jitter))
        notes.append(f"color_jitter_lab(jitter={ab_jitter})")

    # Сборка обратно в RGB для визуальной проверки (через Lab, учитывая нормализацию L)
    L_back = np.clip((Ln + 1.0) * 50.0, 0.0, 100.0)
    lab_back = np.concatenate([L_back, ab[..., 0:1], ab[..., 1:2]], axis=-1)
    rgb_back = (np.clip(color.lab2rgb(lab_back), 0.0, 1.0) * 255.0).astype(np.uint8)

    processed_lab_rgb = np_to_pil(rgb_back)

    # Также подготовим L (яркостной канал) как настоящее ЧБ-изображение для визуальной проверки входа модели
    # Ln в [-1,1] → L_back в [0,100] → масштабируем в 8-битный серый [0,255]
    L_gray_8u = (np.clip(L_back, 0.0, 100.0) / 100.0 * 255.0).astype(np.uint8)
    # Убираем последнюю размерность (H,W,1) → (H,W) для корректного режима 'L'
    if L_gray_8u.ndim == 3 and L_gray_8u.shape[-1] == 1:
        L_gray_8u = L_gray_8u[..., 0]
    L_gray_img = np_to_pil(L_gray_8u).convert("L")

    return {
        "original_rgb": original,
        "processed_rgb_geom": processed_geom,
        "processed_rgb_lab": processed_lab_rgb,
        "L_gray": L_gray_img,
        "L_gray_before": L_gray_before,
        "notes": "; ".join(notes) if notes else "no-op",
    }


def _make_comparison(original: Image.Image, processed: Image.Image) -> Image.Image:
    # Нормализуем высоту для сравнения (не искажая аспект)
    target_h = 256

    def keep_aspect(im, h):
        w, hh = im.size
        new_w = int(round(w * (h / hh)))
        return im.resize((new_w, h), Image.LANCZOS)

    o = keep_aspect(original, target_h)
    p = keep_aspect(processed, target_h)
    canvas = Image.new("RGB", (o.width + p.width + 10, target_h), (20, 20, 20))
    canvas.paste(o, (0, 0))
    canvas.paste(p, (o.width + 10, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"orig {original.size[0]}x{original.size[1]}", fill=(255, 255, 0))
    draw.text(
        (o.width + 15, 5),
        f"proc {processed.size[0]}x{processed.size[1]}",
        fill=(255, 255, 0),
    )
    return canvas


def save_triplet(out_dir: Path, stem: str, data: dict, save_L: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_p = out_dir / f"{stem}_original.png"
    geom_p = out_dir / f"{stem}_processed_geom.png"
    lab_p = out_dir / f"{stem}_processed_lab.png"
    data["original_rgb"].save(orig_p)
    data["processed_rgb_geom"].save(geom_p)
    data["processed_rgb_lab"].save(lab_p)
    l_path = None
    if save_L and ("L_gray" in data):
        l_path = out_dir / f"{stem}_L_gray.png"
        data["L_gray"].save(l_path)
        # Если есть L до дефектов — сохраним её для сравнения
        if data.get("L_gray_before", None) is not None:
            lb_path = out_dir / f"{stem}_L_gray_before.png"
            data["L_gray_before"].save(lb_path)
    # Сравнение original vs processed_geom
    cmp_img = _make_comparison(data["original_rgb"], data["processed_rgb_geom"])
    cmp_p = out_dir / f"{stem}_compare_orig_vs_geom.png"
    cmp_img.save(cmp_p)
    txt = out_dir / f"{stem}_notes.txt"
    txt.write_text(data.get("notes", ""), encoding="utf-8")
    return orig_p, geom_p, lab_p, l_path


def main():
    parser = argparse.ArgumentParser(
        description="Предпросмотр пайплайна подготовки изображений без обучения"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Путь к изображению (одному файлу)"
    )
    parser.add_argument("--config", type=str, default=str(Path("configs/default.yaml")))
    parser.add_argument("--mode", type=str, choices=["train", "val"], default="train")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["advanced", "simple"],
        default=None,
        help="Перекрыть training.dataset из конфига",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Папка для сохранения превью"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_flip", action="store_true")
    parser.add_argument("--disable_jitter", action="store_true")
    parser.add_argument(
        "--save_L",
        action="store_true",
        help="Сохранить L-канал (ЧБ) как <stem>_L_gray.png для проверки входа модели",
    )
    parser.add_argument(
        "--enable_defects",
        action="store_true",
        help="Применить дефекты L-канала из training.aug.defects конфига к предпросмотру (train/advanced)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    ds_kind = (
        args.dataset
        if args.dataset is not None
        else train_cfg.get("dataset", "advanced")
    )
    image_size = int(train_cfg.get("image_size", 256))
    aug = train_cfg.get("aug", {})
    flip_p = float(aug.get("flip_p", 0.5))
    crop_scale = aug.get("crop_scale", [0.8, 1.0])
    ab_jitter = float(aug.get("ab_jitter", 0.05))
    defects_cfg = aug.get("defects", None)
    # Geometry and resize filter from config
    geom_cfg = train_cfg.get("geometry", {})
    geom_train_mode = str(geom_cfg.get("train_mode", "random_crop"))
    geom_val_mode = str(geom_cfg.get("val_mode", "center_crop"))
    resize_filter = str(train_cfg.get("resize", {}).get("filter", "lanczos"))

    inp = Path(args.input)
    if not inp.exists() or not inp.is_file():
        raise FileNotFoundError(f"Файл не найден: {inp}")

    out_dir = (
        Path(args.output)
        if args.output
        else Path(cfg.get("paths", {}).get("experiments", "experiments/exp_default"))
        / "preview_preprocess"
    )

    img = load_rgb(inp)
    result = apply_pipeline(
        img=img,
        mode=args.mode,
        dataset_kind=ds_kind,
        image_size=image_size,
        flip_p=flip_p,
        crop_scale=crop_scale,
        ab_jitter=ab_jitter,
        disable_flip=args.disable_flip,
        disable_jitter=args.disable_jitter,
        enable_defects=args.enable_defects,
        defects_cfg=defects_cfg,
        geom_mode=(geom_train_mode if args.mode == "train" else geom_val_mode),
        resize_filter=resize_filter,
    )

    stem = inp.stem
    o, g, lab_path, lgray = save_triplet(out_dir, stem, result, save_L=args.save_L)

    print("Готово. Сохранены файлы:")
    print(f" - Оригинал:          {o}")
    print(f" - После геометрии:   {g}")
    print(f" - После Lab+jitter:  {lab_path}")
    if lgray is not None:
        print(f" - L (ЧБ вход):       {lgray}")
    print(f"Шаги: {result.get('notes','')}")


if __name__ == "__main__":
    main()
