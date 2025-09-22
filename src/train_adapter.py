from __future__ import annotations
"""
Adapter training script (standalone).
Constraints:
- Keeps the base model frozen; optimizes only additive deltas for selected params (decoder/CRB).
- Saves a single adapter checkpoint with parameter-wise deltas into checkpoints/adapters/adapter.pth
- Uses the same dataset and losses as the main training, but omits DDP/EMA for simplicity and robustness.

Usage:
  python -m src.train_adapter --config configs/default.yaml

This script intentionally avoids adding new CLI flags beyond --config to comply with the spec.
"""
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, set_seed
from src.datasets.advanced_dataset import AdvancedColorizationDataset
from src.models import TintoraAI
from src.models.adapter import AdapterCheckpoint, select_default_adapter_targets
from torch.nn.utils import parametrize
from src.losses import L1LabLoss, PerceptualLoss
from src.losses.advanced import (
    PhotometricSmoothnessLoss,
    DepthSmoothnessLoss,
    ColorConsistencyLoss,
)
from src.losses.gan import GANLoss  # type: ignore
from src.models.discriminator import PatchDiscriminator  # type: ignore
from src.utils.lab_color import lab_to_rgb_tensor  # type: ignore
from torch.utils.tensorboard import SummaryWriter


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_dir = cfg.get("paths", {}).get("train_dir", "data/train")
    val_dir = cfg.get("paths", {}).get("val_dir", "data/val")
    bs = int(cfg.get("training", {}).get("batch_size", 2))
    image_size = int(cfg.get("training", {}).get("image_size", 256))
    geom_train = cfg.get("training", {}).get("geometry", {}).get("train_mode", "random_crop")
    geom_val = cfg.get("training", {}).get("geometry", {}).get("val_mode", "center_crop")
    resize_filter = cfg.get("training", {}).get("resize", {}).get("filter", "lanczos")

    aug_cfg = cfg.get("training", {}).get("aug", {}) or {}
    flip_p = float(aug_cfg.get("flip_p", 0.5))
    crop_scale = aug_cfg.get("crop_scale", [0.8, 1.0])
    if isinstance(crop_scale, (list, tuple)) and len(crop_scale) == 2:
        crop_scale_t = (float(crop_scale[0]), float(crop_scale[1]))
    else:
        crop_scale_t = (0.8, 1.0)
    ab_jitter = float(aug_cfg.get("ab_jitter", 0.05))
    defects_cfg = aug_cfg.get("defects", None)

    ds_train = AdvancedColorizationDataset(
        root_dir=train_dir,
        image_size=image_size,
        train=True,
        aug_flip=flip_p,
        aug_crop_scale=crop_scale_t,
        aug_ab_jitter=ab_jitter,
        geom_mode_train=geom_train,
        geom_mode_val=geom_val,
        resize_filter=resize_filter,
    )
    # установить дефекты L при обучении, если сконфигурированы
    if isinstance(defects_cfg, dict) and defects_cfg.get("enabled", False):
        ds_train.aug_defects = defects_cfg

    ds_val = AdvancedColorizationDataset(
        root_dir=val_dir,
        image_size=image_size,
        train=False,
        aug_flip=flip_p,
        aug_crop_scale=crop_scale_t,
        aug_ab_jitter=0.0,  # без jitter на валидации для детерминизма
        geom_mode_train=geom_train,
        geom_mode_val=geom_val,
        resize_filter=resize_filter,
    )

    num_workers = int(cfg.get("runtime", {}).get("num_workers", 2))
    pin = bool(cfg.get("runtime", {}).get("pin_memory", True))
    return (
        DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin),
    )


def load_base_model(cfg: Dict[str, Any], device: torch.device) -> TintoraAI:
    # Важно: передаём параметры модели как именованные аргументы и фильтруем по сигнатуре
    model_cfg = dict(cfg.get("model", {}) or {})
    allowed_keys = {
        "c1",
        "c2",
        "c3",
        "film_dim",
        "use_guidenet",
        "guide_feature_dim",
        "guide_out_dim",
        "use_saturation_head",
    }
    filtered = {k: v for k, v in model_cfg.items() if k in allowed_keys}
    model = TintoraAI(**filtered, omm_config=cfg.get("omm", {}))
    # load base checkpoint if exists
    ck_root = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
    latest = ck_root / cfg.get("checkpointing", {}).get("latest_names", {}).get("model", "latest.pth")
    if latest.exists():
        sd = torch.load(latest, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            model.load_state_dict(sd["model"], strict=False)
        else:
            model.load_state_dict(sd, strict=False)
    return model.to(device)


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


class CosineWithWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup: int = 0, min_lr: float = 0.0):
        self.opt = optimizer
        self.t_total = max(1, int(total_steps))
        self.warmup = int(max(0, warmup))
        self.min_lr = float(min_lr)
        self._step = 0
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        self._step += 1
        t = min(self._step, self.t_total)
        for i, group in enumerate(self.opt.param_groups):
            base_lr = self.base_lrs[i]
            if self.warmup > 0 and t <= self.warmup:
                lr = base_lr * t / self.warmup
            else:
                progress = (t - self.warmup) / max(1, self.t_total - self.warmup)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
                lr = float(lr)
            group["lr"] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("runtime", {}).get("device", "auto") != "cpu" else "cpu")

    # Data
    train_loader, _ = build_dataloaders(cfg)

    # Model
    model = load_base_model(cfg, device)
    # База остаётся замороженной при обучении адаптера, но модель работает в train-режиме
    freeze_model(model)
    model.train()

    # Build adapter parametrizations (additive deltas) for targeted params
    target_names = list(select_default_adapter_targets(model))
    name_to_module: Dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        for p_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{p_name}" if module_name else p_name
            name_to_module[full] = module

    class AdditiveDelta(nn.Module):
        def __init__(self, base: torch.Tensor):
            super().__init__()
            self.delta = nn.Parameter(torch.zeros_like(base))

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return X + self.delta

    # Register parametrizations
    registered: Dict[str, AdditiveDelta] = {}
    for full in target_names:
        mod = name_to_module.get(full)
        if mod is None:
            continue
        base = getattr(mod, full.split(".")[-1])  # parameter tensor
        pd = AdditiveDelta(base.data)
        parametrize.register_parametrization(mod, full.split(".")[-1], pd)
        registered[full] = pd

    # Losses
    losses_cfg = cfg.get("loss", {})
    lam_l1 = float(losses_cfg.get("lambda_l1", 1.0))
    lam_perc = float(losses_cfg.get("lambda_perc", 0.0))
    lam_photo = float(losses_cfg.get("lambda_photo", 0.0))
    lam_ds = float(losses_cfg.get("lambda_ds", 0.0))
    lam_cc = float(losses_cfg.get("lambda_cc", 0.0))

    l1 = L1LabLoss().to(device)
    perc = PerceptualLoss().to(device)
    photo = PhotometricSmoothnessLoss().to(device)
    dsm = DepthSmoothnessLoss().to(device)
    cc = ColorConsistencyLoss().to(device)

    # Optional GAN (Phase 4)
    gan_cfg = cfg.get("gan", {})
    use_gan = bool(gan_cfg.get("enabled", False))
    D = None
    gan_loss = None
    if use_gan:
        try:
            input_nc = int(gan_cfg.get("input_nc", 3))
            ndf = int(gan_cfg.get("ndf", 64))
            n_layers = int(gan_cfg.get("n_layers", 3))
            D = PatchDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers).to(device)
            gan_loss = GANLoss(gan_cfg.get("loss_type", "hinge")).to(device)
            d_opt = torch.optim.AdamW(D.parameters(), lr=float(cfg.get("optimizer", {}).get("lr_decoder_heads", 1.5e-4)))
        except Exception:
            D = None
            gan_loss = None
            use_gan = False

    # Optimizer
    # Optimizer over adapter deltas only
    opt = torch.optim.AdamW(
        (p for pd in registered.values() for p in pd.parameters()),
        lr=float(cfg.get("optimizer", {}).get("lr_decoder_heads", 1.5e-4)),
        weight_decay=float(cfg.get("optimizer", {}).get("weight_decay_other", 1e-3)),
    )

    # Scheduler (cosine with warmup)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(cfg.get("training", {}).get("epochs", 2))
    warmup = int(cfg.get("scheduler", {}).get("warmup_steps", 0))
    sched = CosineWithWarmup(opt, total_steps=total_steps, warmup=warmup)

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.get("runtime", {}).get("amp", True)) and device.type == "cuda"
    )

    # TensorBoard logging
    logs_dir = Path(cfg.get("paths", {}).get("logs", "logs"))
    writer = SummaryWriter(log_dir=str(logs_dir))
    # Период логирования изображений (из YAML: logging.log_images_every)
    log_img_every = int(cfg.get("logging", {}).get("log_images_every", 50))

    # Curriculum: try to read from config; fallback to simple phase gates
    cur = cfg.get("training", {}).get("curriculum", {})
    use_curr = bool(cur.get("enabled", False))
    cur_phases = cur.get("phases", []) if isinstance(cur.get("phases", []), list) else []

    def resolve_phase(epoch: int) -> int:
        """
        Поддержка YAML-расписания с ключами 'from'/'to' как в базовом train.py.
        Если фаз нет или формат иной — используем разумный фолбэк по эпохам.
        """
        if use_curr and cur_phases:
            try:
                for seg in cur_phases:
                    f = int(seg.get("from", 1))
                    t = int(seg.get("to", f))
                    if f <= epoch <= t:
                        return int(seg.get("phase", 0))
                # Если ни один сегмент не совпал — возьмём фазу из последнего сегмента
                return int(cur_phases[-1].get("phase", 0))
            except Exception:
                pass
        # Фолбэк: простое расписание по эпохам
        if epoch < 1:
            return -1
        if epoch < 2:
            return 0
        if epoch < 3:
            return 1
        if epoch < 4:
            return 2
        if epoch < 5:
            return 3
        return 4

    def resolve_omm_read_only(epoch: int, phase: int) -> bool:
        """Возвращает omm_read_only: читаем из YAML, иначе по правилу phase < 3."""
        if use_curr and cur_phases:
            try:
                for seg in cur_phases:
                    f = int(seg.get("from", 1))
                    t = int(seg.get("to", f))
                    if f <= epoch <= t:
                        if seg.get("omm_read_only", None) is not None:
                            return bool(seg.get("omm_read_only"))
                        break
            except Exception:
                pass
        return bool(phase < 3)

    # Простейший curriculum: до фазы 3 не читаем OMM; перцептуалка с фазы 2; CC с фазы 3
    epochs = int(cfg.get("training", {}).get("epochs", 2))
    global_step = 0
    for epoch in range(1, epochs + 1):
        phase = resolve_phase(epoch)
        omm_ro = resolve_omm_read_only(epoch, phase)

        running = 0.0
        batch_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{epochs} (Phase {phase})",
            leave=True,
        )
        for batch in batch_bar:
            L, ab_gt, _ = batch
            L = L.to(device)
            ab_gt = ab_gt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(L, gt_ab=ab_gt, omm_read_only=omm_ro)
                total = 0.0 * (out["a"].sum() * 0)  # safe scalar seed
                # L1
                if lam_l1 > 0:
                    total = total + lam_l1 * l1(out["a"], out["b"], ab_gt)
                # Perceptual (phase >=2)
                if lam_perc > 0 and phase >= 2:
                    pred_lab = torch.cat([L, out["a"], out["b"]], dim=1)
                    gt_lab = torch.cat([L, ab_gt[:, :1], ab_gt[:, 1:2]], dim=1)
                    total = total + lam_perc * perc(pred_lab, gt_lab)
                # Photometric smoothness (ab), phase >=1
                if lam_photo > 0 and phase >= 1:
                    total = total + lam_photo * photo(L, out["a"], out["b"])
                # Depth smoothness, phase >=1
                if lam_ds > 0 and phase >= 1 and ("D" in out):
                    total = total + lam_ds * dsm(L, out["D"])  # (L, D) порядок как в DepthSmoothnessLoss
                # Color Consistency (пирамида L1 между ab_pred и ab_gt), phase >=3
                if lam_cc > 0 and phase >= 3:
                    ab_pred = torch.cat([out["a"], out["b"]], dim=1)
                    total = total + lam_cc * cc(ab_pred, ab_gt)

            # GAN (Phase 4): generator loss
            if use_gan and phase >= 4 and D is not None and gan_loss is not None:
                try:
                    # Prepare fake RGB from separate L,a,b
                    rgb_fake = lab_to_rgb_tensor(L, out["a"], out["b"]).clamp(0, 1)
                    # Generator wants D(fake) to be real
                    d_pred = D(rgb_fake)
                    g_adv = float(losses_cfg.get("lambda_adv", 0.0)) * gan_loss(d_pred, True)
                    total = total + g_adv
                except Exception:
                    pass

            # учёт лосса для прогресса (до backward, detach)
            try:
                running += float(total.detach().item())
            except Exception:
                pass

            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            # обновление строки прогресса
            try:
                avg = running / max(1, (global_step % steps_per_epoch) + 1)
                batch_bar.set_postfix({"loss": f"{avg:.4f}", "phase": phase})
            except Exception:
                pass

            # Discriminator step
            if use_gan and phase >= 4 and D is not None and gan_loss is not None:
                try:
                    d_opt.zero_grad(set_to_none=True)  # type: ignore
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        rgb_fake = lab_to_rgb_tensor(L, out["a"], out["b"]).clamp(0, 1).detach()
                        rgb_real = lab_to_rgb_tensor(L, ab_gt[:, :1], ab_gt[:, 1:2]).clamp(0, 1)
                        d_fake = D(rgb_fake)
                        d_real = D(rgb_real)
                        d_loss = 0.5 * (gan_loss(d_real, True) + gan_loss(d_fake, False))
                    scaler.scale(d_loss).backward()
                    scaler.step(d_opt)  # type: ignore
                    scaler.update()
                except Exception:
                    pass

            # Logging
            if global_step % 50 == 0:
                # Скаляры: фаза, LR и текущий loss
                writer.add_scalar("train/phase", phase, global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
                try:
                    writer.add_scalar("train/loss", float(total.detach().item()), global_step)
                except Exception:
                    pass

            # Изображения: вход L, предсказанный RGB, GT RGB
            if log_img_every > 0 and (global_step % log_img_every == 0):
                try:
                    with torch.no_grad():
                        # Нормализуем L из [-1,1] в [0,1] для визуализации
                        L_vis = ((L[:1] + 1.0) * 0.5).clamp(0, 1)  # (1,1,H,W)
                        rgb_pred = lab_to_rgb_tensor(L[:1], out["a"][:1], out["b"][:1]).clamp(0, 1)  # (1,3,H,W)
                        rgb_gt = lab_to_rgb_tensor(L[:1], ab_gt[:1, :1], ab_gt[:1, 1:2]).clamp(0, 1)
                    # add_image ожидает (C,H,W)
                    writer.add_image("images/L", L_vis.squeeze(0).detach().cpu(), global_step)
                    writer.add_image("images/RGB_pred", rgb_pred.squeeze(0).detach().cpu(), global_step)
                    writer.add_image("images/RGB_gt", rgb_gt.squeeze(0).detach().cpu(), global_step)
                except Exception as e:
                    try:
                        print(f"[WARN][images] skip TB images: {e}")
                    except Exception:
                        pass
            global_step += 1

    # Save adapter checkpoint
    out_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints")) / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Export adapter deltas into checkpoint format
    deltas: Dict[str, torch.Tensor] = {}
    for full, pd in registered.items():
        deltas[full] = pd.delta.detach().cpu().clone()
    ck = AdapterCheckpoint(deltas=deltas, meta={"config": "default", "targets": target_names}).to_state()
    torch.save(ck, out_dir / "adapter.pth")
    try:
        print(f"[OK] Adapter saved to {out_dir / 'adapter.pth'} with {len(ck['deltas'])} tensors")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
