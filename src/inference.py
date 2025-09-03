import argparse
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch
from PIL import Image
import numpy as np
from .models import TintoraAI
from .utils import lab_to_rgb_tensor, load_config


def load_image_L(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # Преобразуем в Lab и отбрасываем ab, чтобы получить имитацию градаций серого
    from skimage import color

    lab = color.rgb2lab(arr).astype("float32")
    L = lab[..., 0:1]
    Ln = (L / 50.0) - 1.0
    L_t = torch.from_numpy(Ln.transpose(2, 0, 1)).unsqueeze(0)
    return L_t  # (1,1,H,W)


def pad_to_divisible(x: torch.Tensor, div: int = 32) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, H, W = x.shape
    pad_h = (div - (H % div)) % div
    pad_w = (div - (W % div)) % div
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    xpad = torch.nn.functional.pad(
        x, (0, pad_w, 0, pad_h)
    )  # (слева,справа,сверху,снизу); порядок F.pad: (W_left, W_right, H_top, H_bot)
    return xpad, (0, pad_w, 0, pad_h)


@torch.no_grad()
def colorize_single(
    model: TintoraAI, L: torch.Tensor, omm_read_only: bool = False, pad_divisor: int = 32
) -> torch.Tensor:
    Lp, pads = pad_to_divisible(L, pad_divisor)
    out = model(Lp, omm_read_only=omm_read_only)
    a, b = out["a"], out["b"]
    print(
        f"DEBUG shapes before: Lp={tuple(Lp.shape)} a={tuple(a.shape)} b={tuple(b.shape)}",
        flush=True,
    )
    # Унифицируем размерности Lp, a, b: возьмём целевой размер = максимум по каждой размерности
    HL, WL = Lp.shape[-2:]
    Ha, Wa = a.shape[-2:]
    Hb, Wb = b.shape[-2:]
    Ht = max(HL, Ha, Hb)
    Wt = max(WL, Wa, Wb)
    if (HL, WL) != (Ht, Wt):
        Lp = torch.nn.functional.interpolate(
            Lp, size=(Ht, Wt), mode="bilinear", align_corners=False
        )
    if (Ha, Wa) != (Ht, Wt):
        a = torch.nn.functional.interpolate(
            a, size=(Ht, Wt), mode="bilinear", align_corners=False
        )
    if (Hb, Wb) != (Ht, Wt):
        b = torch.nn.functional.interpolate(
            b, size=(Ht, Wt), mode="bilinear", align_corners=False
        )
    print(
        f"DEBUG shapes after:  Lp={tuple(Lp.shape)} a={tuple(a.shape)} b={tuple(b.shape)}",
        flush=True,
    )
    rgb = lab_to_rgb_tensor(Lp, a, b)  # (1,3,H,W)
    # удаляем паддинги
    _, _, H, W = L.shape
    rgb = rgb[:, :, :H, :W]
    return rgb


@torch.no_grad()
def colorize_tiled(
    model: TintoraAI,
    L: torch.Tensor,
    tile: int = 512,
    overlap: int = 32,
    omm_read_only: bool = False,
    pad_divisor: int = 32,
) -> torch.Tensor:
    # Тильная обработка с косинусным (Хэннинг) окном для бесшовного склейки
    _, _, H, W = L.shape
    if tile <= 0 or tile >= max(H, W):
        return colorize_single(
            model, L, omm_read_only=omm_read_only, pad_divisor=pad_divisor
        )
    stride = max(1, tile - overlap)
    out_rgb = torch.zeros((1, 3, H, W), device=L.device)
    weight = torch.zeros((1, 1, H, W), device=L.device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, x0 = y, x
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            patch = L[:, :, y0:y1, x0:x1]
            rgb_patch = colorize_single(
                model, patch, omm_read_only=omm_read_only, pad_divisor=pad_divisor
            )
            # Окно Хэннинга по фактическому размеру патча (на краях тайлы меньше)
            ph, pw = rgb_patch.shape[-2:]
            wy = torch.hann_window(ph, device=L.device).view(1, 1, ph, 1)
            wx = torch.hann_window(pw, device=L.device).view(1, 1, 1, pw)
            w2d = (wy * wx).clamp(min=1e-3)  # избегаем нулей
            out_rgb[:, :, y0:y1, x0:x1] += rgb_patch * w2d
            weight[:, :, y0:y1, x0:x1] += w2d

    out_rgb = out_rgb / torch.clamp(weight, min=1e-6)
    return out_rgb


@torch.no_grad()
def tta_colorize(
    model: TintoraAI,
    L: torch.Tensor,
    tile: int,
    overlap: int,
    pad_divisor: int,
    tta_cfg: Dict[str, Any],
) -> torch.Tensor:
    # Поддерживаем горизонтальный флип и масштабирование, усредняя предсказания
    do_flip = bool(tta_cfg.get("flip", True))
    scales = tta_cfg.get("scales", [1.0])
    if not isinstance(scales, (list, tuple)):
        scales = [1.0]
    H, W = L.shape[-2:]

    def run(Lin: torch.Tensor):
        if tile and tile > 0:
            return colorize_tiled(
                model,
                Lin,
                tile=tile,
                overlap=overlap,
                omm_read_only=False,
                pad_divisor=pad_divisor,
            )
        return colorize_single(model, Lin, omm_read_only=False, pad_divisor=pad_divisor)

    preds = []
    # Базовый
    for s in scales:
        if abs(float(s) - 1.0) < 1e-6:
            rgb = run(L)
        else:
            nh, nw = max(1, int(H * float(s))), max(1, int(W * float(s)))
            Ls = torch.nn.functional.interpolate(
                L, size=(nh, nw), mode="bilinear", align_corners=False
            )
            rgb_s = run(Ls)
            rgb = torch.nn.functional.interpolate(
                rgb_s, size=(H, W), mode="bilinear", align_corners=False
            )
        preds.append(rgb)

    if do_flip:
        Lf = torch.flip(L, dims=[-1])
        for s in scales:
            if abs(float(s) - 1.0) < 1e-6:
                rgb_f = run(Lf)
            else:
                nh, nw = max(1, int(H * float(s))), max(1, int(W * float(s)))
                Lfs = torch.nn.functional.interpolate(
                    Lf, size=(nh, nw), mode="bilinear", align_corners=False
                )
                rgb_fs = run(Lfs)
                rgb_f = torch.nn.functional.interpolate(
                    rgb_fs, size=(H, W), mode="bilinear", align_corners=False
                )
            preds.append(torch.flip(rgb_f, dims=[-1]))

    # Усреднение
    out = torch.stack(preds, dim=0).mean(dim=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs/default.yaml")))
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Размер тайла для тильной инференции. 0 — отключает разбиение на тайлы.",
    )
    parser.add_argument(
        "--overlap", type=int, default=32, help="Размер перекрытия между тайлами."
    )
    parser.add_argument(
        "--pad-div",
        type=int,
        default=None,
        help="Делитель для паддинга входа (по умолчанию из конфига или 32).",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Принудительный запуск на CPU."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    inf_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    cfg_pad_div = (
        int(inf_cfg.get("pad_divisor", 32)) if isinstance(inf_cfg, dict) else 32
    )
    pad_divisor = int(args.pad_div) if args.pad_div is not None else cfg_pad_div
    model_cfg = cfg.get("model", {})
    # Разрешаем конфиг OMM так же, как в train
    if (
        isinstance(model_cfg, dict)
        and "omm" in model_cfg
        and isinstance(model_cfg["omm"], dict)
        and len(model_cfg["omm"]) > 0
    ):
        omm_config_to_use = model_cfg["omm"]
    else:
        omm_config_to_use = cfg.get("omm", {})

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    model = (
        TintoraAI(
            c1=model_cfg.get("c1", 96),
            c2=model_cfg.get("c2", 192),
            c3=model_cfg.get("c3", 384),
            film_dim=model_cfg.get("film_dim", 256),
            use_guidenet=bool(model_cfg.get("use_guidenet", False)),
            guide_feature_dim=model_cfg.get("guide_feature_dim", None),
            guide_out_dim=model_cfg.get("guide_out_dim", None),
            omm_config=omm_config_to_use,
            use_saturation_head=model_cfg.get("use_saturation_head", False),
        )
        .to(device)
        .eval()
    )

    default_ckpt = Path(paths.get("checkpoints", "checkpoints")) / cfg.get(
        "checkpointing", {}
    ).get("latest_names", {}).get("model", "latest.pth")
    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else default_ckpt
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    else:
        print(f"Внимание: чекпойнт {ckpt_path} не найден. Использую случайные веса.")

    input_path = Path(args.input)
    # Определяем путь вывода из YAML, если он не задан флагом
    default_output_dir = (
        Path(paths.get("experiments", paths.get("logs", "outputs"))) / "val_pred"
    )
    output_path = Path(args.output) if args.output is not None else default_output_dir

    if not input_path.exists():
        print(f"Ошибка: путь ввода {input_path} не существует.")
        return

    if input_path.is_dir():
        image_files = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
    else:
        image_files = [input_path]

    if not image_files:
        print(f"Изображения не найдены в {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in image_files:
        try:
            print(f"Обработка {image_file.name}...")
            L = load_image_L(image_file).to(device)
            with torch.no_grad():
                tta_cfg = inf_cfg.get("tta", {}) if isinstance(inf_cfg, dict) else {}
                use_tta = bool(tta_cfg.get("enabled", False))
                if use_tta:
                    rgb = tta_colorize(
                        model,
                        L,
                        tile=int(args.tile) if args.tile else 0,
                        overlap=int(args.overlap),
                        pad_divisor=pad_divisor,
                        tta_cfg=tta_cfg,
                    )
                else:
                    if args.tile and args.tile > 0:
                        rgb = colorize_tiled(
                            model,
                            L,
                            tile=int(args.tile),
                            overlap=int(args.overlap),
                            omm_read_only=False,
                            pad_divisor=pad_divisor,
                        )
                    else:
                        rgb = colorize_single(
                            model, L, omm_read_only=False, pad_divisor=pad_divisor
                        )

            rgb_img = (
                (rgb.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )

            save_path = output_path / image_file.name
            Image.fromarray(rgb_img).save(save_path)
            print(f"Сохранено: {save_path}")
        except Exception as e:
            traceback.print_exc()
            print(f"Не удалось обработать {image_file.name}: {e}")


if __name__ == "__main__":
    main()
