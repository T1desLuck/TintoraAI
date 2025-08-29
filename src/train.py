import os
from pathlib import Path
import argparse
from tqdm import tqdm
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .utils.seed import set_seed
from .utils.lab_color import lab_to_rgb_tensor
from .utils.dist import init_distributed, ddp_available, is_main_process, get_rank, cleanup, get_device
from .utils.metrics import ssim, try_lpips
from .utils import load_config
from .datasets import SimpleColorizationDataset, AdvancedColorizationDataset
from .models import TintoraAI, PatchDiscriminator
from .losses import (
    L1LabLoss,
    PerceptualLoss,
    PhotometricSmoothnessLoss,
    DepthSmoothnessLoss,
    ColorConsistencyPyrLoss,
    EntropyLoss,
    OMMClusterLoss,
    GANLoss,
    PatchNCELoss
)
from .utils.balancer import LossBalancer
from .utils import DynamicLossBalancer





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs/default.yaml")))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    paths = cfg.get("paths", {})
    runtime = cfg.get("runtime", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    gan_cfg = cfg.get("gan", {})
    ssl_cfg = cfg.get("ssl", {})

    # DDP init (if enabled and torchrun provides env)
    ddp_enabled = ddp_available(runtime)
    if ddp_enabled:
        backend_cfg = runtime.get("ddp", {}).get("backend", "auto")
        backend = None if backend_cfg == "auto" else backend_cfg
        init_distributed(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = get_device(runtime, local_rank)
    if torch.cuda.is_available():
        # Ensure device has a specific index
        if device.type == 'cuda' and device.index is None:
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.device_count() > 1 else 'cuda:0')
        torch.cuda.set_device(device)

    # Логирование
    log_dir = Path(paths.get("logs", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if (cfg.get("logging", {}).get("tensorboard", True) and is_main_process()) else None

    # Данные
    # Предпочитаем пути из секции `paths`, с резервом на `data`
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    data_root = paths_cfg.get("data_root", data_cfg.get("base_path", "./data"))
    train_dir = paths_cfg.get("train_dir", os.path.join(data_root, data_cfg.get("train_dir", "train")))
    val_dir = paths_cfg.get("val_dir", os.path.join(data_root, data_cfg.get("val_dir", "val")))
    image_size = train_cfg.get("image_size", 256)
    geom_cfg = train_cfg.get("geometry", {})
    geom_train_mode = str(geom_cfg.get("train_mode", "random_crop")).lower()
    geom_val_mode = str(geom_cfg.get("val_mode", "center_crop")).lower()
    resize_filter = str(train_cfg.get("resize", {}).get("filter", "lanczos")).lower()
    use_adv = (train_cfg.get("dataset", "advanced") == "advanced")
    if use_adv:
        aug = train_cfg.get("aug", {})
        ds = AdvancedColorizationDataset(
            train_dir,
            image_size=image_size,
            train=True,
            aug_flip=float(aug.get("flip_p", 0.5)),
            aug_crop_scale=tuple(aug.get("crop_scale", [0.8, 1.0])),
            aug_ab_jitter=float(aug.get("ab_jitter", 0.05)),
            geom_mode_train=geom_train_mode,
            geom_mode_val=geom_val_mode,
            resize_filter=resize_filter,
        )
        # Передаём конфиг мелких дефектов L-канала (если указан)
        ds.aug_defects = aug.get("defects", None)
    else:
        ds = SimpleColorizationDataset(
            train_dir,
            image_size=image_size,
            geom_mode=geom_train_mode,
            resize_filter=resize_filter,
        )
    if len(ds) == 0:
        print(f"Изображения не найдены в {train_dir}. Добавьте изображения, чтобы начать обучение.")
        return
    sampler = None
    if ddp_enabled:
        sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=runtime.get("num_workers", 4),
        pin_memory=runtime.get("pin_memory", True),
    )
    # Валидационный загрузчик (Validation DataLoader)
    val_cfg = cfg.get("validation", {})
    do_val = bool(val_cfg.get("enabled", True)) and Path(val_dir).exists()
    if do_val:
        if use_adv:
            ds_val = AdvancedColorizationDataset(
                val_dir,
                image_size=image_size,
                train=False,
                geom_mode_train=geom_train_mode,
                geom_mode_val=geom_val_mode,
                resize_filter=resize_filter,
            )
        else:
            from .datasets import SimpleColorizationDataset as _S
            ds_val = _S(
                val_dir,
                image_size=image_size,
                geom_mode=geom_val_mode,
                resize_filter=resize_filter,
            )
        val_sampler = DistributedSampler(ds_val, shuffle=False) if ddp_enabled else None
        dl_val = DataLoader(
            ds_val,
            batch_size=int(val_cfg.get("batch_size", 4)),
            shuffle=False,
            sampler=val_sampler,
            num_workers=runtime.get("num_workers", 4),
            pin_memory=runtime.get("pin_memory", True),
        )
    else:
        dl_val = None

    # Модель
    # Разрешение конфигурации OMM: приоритет model.omm, иначе верхнеуровневый omm
    omm_config_to_use = {}
    if isinstance(model_cfg, dict) and "omm" in model_cfg and isinstance(model_cfg["omm"], dict) and len(model_cfg["omm"]) > 0:
        omm_config_to_use = model_cfg["omm"]
    else:
        omm_config_to_use = cfg.get("omm", {})

    model = TintoraAI(
        c1=model_cfg.get("c1", 96),
        c2=model_cfg.get("c2", 192),
        c3=model_cfg.get("c3", 384),
        film_dim=model_cfg.get("film_dim", 256),
        use_guidenet=bool(model_cfg.get("use_guidenet", False)),
        guide_feature_dim=model_cfg.get("guide_feature_dim", None),
        guide_out_dim=model_cfg.get("guide_out_dim", None),
        omm_config=omm_config_to_use, # Берём из model.omm или fallback на верхний omm
        use_saturation_head=model_cfg.get("use_saturation_head", False),
    ).to(device)
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=runtime.get("ddp", {}).get("find_unused_parameters", False),
            broadcast_buffers=runtime.get("ddp", {}).get("broadcast_buffers", True),
            static_graph=runtime.get("ddp", {}).get("static_graph", False),
        )

    # Оптимизатор из YAML
    optim_cfg = cfg.get("optimizer", {})
    base_lr = float(optim_cfg.get("lr_decoder_heads", 1.0e-4))
    weight_decay = float(optim_cfg.get("weight_decay_other", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # Дискриминатор и его оптимизатор (включаются только при соответствующей конфигурации; используются на фазе 4)
    use_gan = bool(gan_cfg.get("enabled", False))
    if use_gan:
        D = PatchDiscriminator(
            input_nc=int(gan_cfg.get("input_nc", 3)),
            ndf=int(gan_cfg.get("ndf", 64)),
            n_layers=int(gan_cfg.get("n_layers", 3)),
            use_spectral_norm=True,
        ).to(device)
        if ddp_enabled:
            # DDP для дискриминатора обычно не обязателен при 1-GPU на узел, оставим без DDP
            pass
        d_lr = float(gan_cfg.get("lr", base_lr))
        optD = torch.optim.AdamW(D.parameters(), lr=d_lr, weight_decay=weight_decay)
    else:
        D = None
        optD = None

    # Настройка EMA
    ema_cfg = train_cfg.get("ema", {})
    use_ema = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_start = int(ema_cfg.get("start_epoch", 1))

    class ModelEMA:
        def __init__(self, model):
            # создаём FP32-копию на том же устройстве
            # Создаём теневую модель с тем же конфигом, включая OMM
            full_model_cfg = {
                "c1": model_cfg.get("c1", 96),
                "c2": model_cfg.get("c2", 192),
                "c3": model_cfg.get("c3", 384),
                "film_dim": model_cfg.get("film_dim", 256),
                "use_guidenet": bool(model_cfg.get("use_guidenet", False)),
                "guide_feature_dim": model_cfg.get("guide_feature_dim", None),
                "guide_out_dim": model_cfg.get("guide_out_dim", None),
                "omm_config": omm_config_to_use,
                "use_saturation_head": model_cfg.get("use_saturation_head", False),
            }
            self.shadow = TintoraAI(**full_model_cfg).to(device)
            self.shadow.load_state_dict((model.module if isinstance(model, DDP) else model).state_dict())
            self.shadow.eval()
            for p in self.shadow.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def update(self, model, decay: float):
            src = model.module if isinstance(model, DDP) else model
            for (n_s, p_s), (n, p) in zip(self.shadow.state_dict().items(), src.state_dict().items()):
                if not p_s.dtype.is_floating_point:
                    self.shadow.state_dict()[n_s].copy_(p)
                else:
                    self.shadow.state_dict()[n_s].lerp_(p, weight=(1.0 - decay))

        def state_dict(self):
            return self.shadow.state_dict()

    ema = ModelEMA(model) if use_ema else None

    loss_l1 = L1LabLoss()
    base_loss_cfg = cfg.get("loss", {})
    lam_l1 = float(base_loss_cfg.get("lambda_l1", 1.0))
    lam_perc = float(base_loss_cfg.get("lambda_perc", 0.0))
    loss_perc = PerceptualLoss() if lam_perc > 0.0 else None
    # Продвинутые лоссы (веса могут быть 0.0)
    lam_photo = float(base_loss_cfg.get("lambda_photo", 0.0))
    lam_ds = float(base_loss_cfg.get("lambda_ds", 0.0))
    lam_cc = float(base_loss_cfg.get("lambda_cc", 0.0))
    lam_entropy = float(base_loss_cfg.get("lambda_entropy", 0.0))
    lam_cluster = float(base_loss_cfg.get("lambda_cluster", 0.0))
    lam_adv = float(base_loss_cfg.get("lambda_adv", 0.0))

    loss_photo = PhotometricSmoothnessLoss()
    loss_ds = DepthSmoothnessLoss()
    loss_cc = ColorConsistencyPyrLoss()
    loss_entropy = EntropyLoss()
    loss_cluster = OMMClusterLoss()
    loss_gan = GANLoss(loss_type=str(gan_cfg.get("loss_type", "hinge"))) if use_gan else None

    epochs = int(train_cfg.get("epochs", 2))
    scaler = torch.cuda.amp.GradScaler(enabled=runtime.get("amp", True))
    # PatchNCE для SSL-предобучения (Фаза -1)
    ssl_enabled = bool(ssl_cfg.get("enabled", False))
    pncet = float(ssl_cfg.get("patchnce", {}).get("temperature", 0.07))
    pnce_norm = bool(ssl_cfg.get("patchnce", {}).get("normalize", True))
    loss_patchnce = PatchNCELoss(temperature=pncet, normalize=pnce_norm)

    global_step = 0
    # Куррикулум-расписание
    phase = int(train_cfg.get("curriculum_phase", 0))
    cur_cfg = train_cfg.get("curriculum", {}) if train_cfg.get("curriculum", {}).get("enabled", False) else None
    def apply_curriculum(epoch: int):
        nonlocal phase, lam_l1, lam_perc, lam_cc, lam_photo, lam_entropy, lam_ds, lam_cluster, lam_adv
        omm_ro = True if phase <= 0 else None
        if cur_cfg is None:
            return omm_ro
        for seg in cur_cfg.get("phases", []):
            if int(seg.get("from", 0)) <= epoch <= int(seg.get("to", 0)):
                phase = int(seg.get("phase", phase))
                if seg.get("omm_read_only", None) is not None:
                    omm_ro = bool(seg["omm_read_only"])
                # локальные переопределения весов
                ls = seg.get("loss", {})
                lam_l1 = float(ls.get("lambda_l1", lam_l1))
                lam_perc = float(ls.get("lambda_perc", lam_perc))
                lam_cc = float(ls.get("lambda_cc", lam_cc))
                lam_photo = float(ls.get("lambda_photo", lam_photo))
                lam_entropy = float(ls.get("lambda_entropy", lam_entropy))
                lam_ds = float(ls.get("lambda_ds", lam_ds))
                lam_cluster = float(ls.get("lambda_cluster", lam_cluster))
                lam_adv = float(ls.get("lambda_adv", lam_adv))
                break
        return omm_ro
    log_img_every = int(cfg.get("logging", {}).get("log_images_every", 50))
    # DLB (Dynamic Loss Balancer)
    # --- Настройка балансировщика лоссов ---
    dlb_cfg = train_cfg.get("dlb", {})
    use_dlb = bool(dlb_cfg.get("enabled", False))
    dlb_decay = float(dlb_cfg.get("decay", 0.9))
    dynamic_balancer = DynamicLossBalancer(decay=dlb_decay) if use_dlb else None

    # Подготовим curriculum для LossBalancer: ожидается dict {epoch_start: {phase_num:int, losses: {..}}}
    lb_curriculum = {}
    if cur_cfg is not None:
        for seg in cur_cfg.get("phases", []):
            try:
                epoch_start = int(seg.get("from", 1))
            except Exception:
                epoch_start = 1
            phase_num = int(seg.get("phase", 0))
            ls = seg.get("loss", {})
            # Преобразуем имена lambda_* → ключи, ожидаемые LossBalancer/DLB
            losses_map = {
                "l1": float(ls.get("lambda_l1", lam_l1)),
                "perc": float(ls.get("lambda_perc", lam_perc)),
                "cc": float(ls.get("lambda_cc", lam_cc)),
                "photo": float(ls.get("lambda_photo", lam_photo)),
                "entropy": float(ls.get("lambda_entropy", lam_entropy)),
                "ds": float(ls.get("lambda_ds", lam_ds)),
                "cluster": float(ls.get("lambda_cluster", lam_cluster)),
                "adv": float(ls.get("lambda_adv", lam_adv)),
            }
            lb_curriculum[epoch_start] = {"phase_num": phase_num, "losses": losses_map}
    else:
        # Фолбэк: одна фаза с текущими базовыми весами
        lb_curriculum[1] = {
            "phase_num": phase,
            "losses": {
                "l1": lam_l1,
                "perc": lam_perc,
                "cc": lam_cc,
                "photo": lam_photo,
                "entropy": lam_entropy,
                "ds": lam_ds,
                "cluster": lam_cluster,
                "adv": lam_adv,
            },
        }

    loss_balancer = LossBalancer(curriculum_cfg=lb_curriculum, dlb=dynamic_balancer)
    best_score = None
    best_metric = str(ckpt_cfg.get("best_metric", "ssim")).lower()
    save_best = bool(ckpt_cfg.get("save_best", True))
    save_latest = bool(ckpt_cfg.get("save_latest", True))
    latest_names = ckpt_cfg.get("latest_names", {"model": "latest.pth", "ema": "latest_ema.pth"})
    best_names = ckpt_cfg.get("best_names", {"model": "best.pth", "ema": "best_ema.pth"})

    start_epoch = train_cfg.get("start_epoch", 1)

    # Scheduler (optional, supports cosine with warmup over steps)
    sched_cfg = cfg.get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "none")).lower()
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    steps_per_epoch = len(dl)
    total_steps = max(1, steps_per_epoch * max(0, epochs - start_epoch + 1))

    def _lr_lambda(step_idx: int):
        if warmup_steps > 0 and step_idx < warmup_steps:
            return max(1e-8, float(step_idx + 1) / float(max(1, warmup_steps)))
        if sched_type == "cosine":
            t = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
            # cosine from 1.0 -> 0.0 (no restarts)
            import math
            return 0.5 * (1.0 + math.cos(math.pi * t))
        # default: constant LR
        return 1.0

    scheduler = None
    if sched_type in {"cosine", "constant"} or warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        # ПРИМЕНЯЕМ РАСПИСАНИЕ НА НАЧАЛО ЭПОХИ
        omm_epoch_read_only = apply_curriculum(epoch)
        phase_num = loss_balancer.phase_num
        loss_weights = loss_balancer.get_weights(epoch)

        if ddp_enabled and sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs} (Phase {phase_num})")
        for i, (L, ab, *_) in enumerate(pbar):
            L = L.to(device)
            ab = ab.to(device)
            with torch.cuda.amp.autocast(enabled=runtime.get("amp", True)):
                # OMM режим на эпоху
                # Pass ground truth `ab` for OMM color statistics update
                out = model(L, gt_ab=ab, omm_read_only=omm_epoch_read_only)
                # Фаза -1: SSL PatchNCE
                if ssl_enabled and phase <= -1:
                    # Приводим пространственные размеры F3 к F2 для парных патчей
                    F2 = out["F2"]  # (B,c2,H/8,W/8) — нормированы
                    F3 = out["F3"]  # (B,c3,H/16,W/16) — нормированы
                    B, c2, h2, w2 = F2.shape
                    F3_up = torch.nn.functional.interpolate(F3, size=(h2, w2), mode="bilinear", align_corners=False)
                    # Приводим канальные размерности через 1x1, если отличаются
                    if F3_up.shape[1] != F2.shape[1]:
                        # временный 1x1 проектор на лету (без параметров хранить нельзя) — используем свертку с групповым усреднением
                        # чтобы избежать обучаемых параметров, просто усечем/допадим каналы
                        if F3_up.shape[1] > F2.shape[1]:
                            F3_up = F3_up[:, :c2]
                        else:
                            pad = c2 - F3_up.shape[1]
                            F3_up = torch.nn.functional.pad(F3_up, (0,0,0,0,0,pad))
                    # Формируем (B,C,N)
                    q = F3_up.flatten(2)  # (B,C,N)
                    k = F2.flatten(2)     # (B,C,N)
                    l_ssl = loss_patchnce(q, k, temperature=None)
                    loss = l_ssl
                    is_ssl_phase = True
                    # Плейсхолдеры для единообразного логирования ниже
                    l_l1 = torch.zeros((), device=L.device)
                    l_perc = None
                    rgb_pred = None
                    rgb_gt = None
                    l_photo = None
                    l_cc = None
                    l_entropy = None
                    l_ds = None
                    l_cluster = None
                else:
                    is_ssl_phase = False
                    # Сначала считаем все термы (supervised)
                    l_l1 = loss_l1(out["a"], out["b"], ab)
                    l_perc = None
                    rgb_pred = None
                    rgb_gt = None
                    if loss_perc is not None:
                        rgb_pred = lab_to_rgb_tensor(L, out["a"], out["b"])  # (B,3,H,W)
                        rgb_gt = lab_to_rgb_tensor(L, ab[:, :1], ab[:, 1:2])
                        l_perc = loss_perc(rgb_pred, rgb_gt)
                    l_photo = None
                    l_cc = None
                    l_entropy = None
                    l_ds = None
                    l_cluster = None
                    if phase_num >= 1 and loss_weights.get("color_consistency", 0) > 0.0:
                        l_cc = loss_cc(torch.cat([out["a"], out["b"]], dim=1), ab)
                    if phase >= 1 and lam_photo > 0.0:
                        l_photo = loss_photo(L, out["a"], out["b"])
                    if phase >= 2 and lam_entropy > 0.0:
                        l_entropy = loss_entropy(out.get("sat", torch.sigmoid(out["a"])) )
                    if phase >= 3 and lam_ds > 0.0:
                        l_ds = loss_ds(L, out.get("D", torch.zeros_like(L)))
                    if phase >= 4 and lam_cluster > 0.0:
                        # Use OMM-projected F2 (256-D) to match mem_map channels
                        F2_omm = out.get(
                            "F2_omm",
                            torch.zeros(L.size(0), 256, L.size(2)//8, L.size(3)//8, device=L.device),
                        )
                        mem_map = out.get(
                            "mem_map",
                            torch.zeros(L.size(0), 256, L.size(2), L.size(3), device=L.device),
                        )
                        l_cluster = loss_cluster(F2_omm, mem_map)

            # Адверсариальный шаг (фаза 4)
            adv_active = use_gan and (phase >= 4) and (lam_adv > 0.0) and (D is not None) and (loss_gan is not None)
            l_adv_term = None
            if adv_active:
                # Подготовка RGB, если ещё не было
                if rgb_pred is None or rgb_gt is None:
                    with torch.cuda.amp.autocast(enabled=runtime.get("amp", True)):
                        rgb_pred = lab_to_rgb_tensor(L, out["a"], out["b"])  # (B,3,H,W)
                        rgb_gt = lab_to_rgb_tensor(L, ab[:, :1], ab[:, 1:2])
                D.train()
                optD.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=runtime.get("amp", True)):
                    pred_real = D(rgb_gt)
                    pred_fake = D(rgb_pred.detach())
                    loss_D = 0.5 * (loss_gan(pred_real, True, for_discriminator=True) + loss_gan(pred_fake, False, for_discriminator=True))
                scaler.scale(loss_D).backward()
                scaler.step(optD)
                # Генераторный терм
                with torch.cuda.amp.autocast(enabled=runtime.get("amp", True)):
                    pred_fake_forG = D(rgb_pred)
                    l_adv_term = loss_gan(pred_fake_forG, True, for_discriminator=False)
            # Сформировать базовые веса лоссов из конфигурации
            base_lams = {
                "l1": lam_l1,
                "perc": lam_perc,
                "cc": lam_cc,
                "photo": lam_photo,
                "entropy": lam_entropy,
                "ds": lam_ds,
                "cluster": lam_cluster,
                "adv": lam_adv,
            }
            
            # Initialize lams with base values first
            lams = base_lams
            
            if not is_ssl_phase:
                if use_dlb and dynamic_balancer is not None:
                    lams = dynamic_balancer.compute_weights(base_lams)
                # else: lams already equals base_lams

                # Итоговый лосс: сумма по доступным термам
                loss = 0.0 * l_l1
                if lams["l1"] > 0.0:
                    loss = loss + lams["l1"] * l_l1
                if l_perc is not None and lams["perc"] > 0.0:
                    loss = loss + lams["perc"] * l_perc
                if l_cc is not None and lams["cc"] > 0.0:
                    loss = loss + lams["cc"] * l_cc
                if l_photo is not None and lams["photo"] > 0.0:
                    loss = loss + lams["photo"] * l_photo
                if l_entropy is not None and lams["entropy"] > 0.0:
                    loss = loss + lams["entropy"] * l_entropy
                if l_ds is not None and lams["ds"] > 0.0:
                    loss = loss + lams["ds"] * l_ds
                if l_cluster is not None and lams["cluster"] > 0.0:
                    loss = loss + lams["cluster"] * l_cluster
                if l_adv_term is not None and lams["adv"] > 0.0:
                    loss = loss + lams["adv"] * l_adv_term
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            # Обновление EMA после шага оптимизатора
            if use_ema and epoch >= ema_start:
                ema.update(model, ema_decay)

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_l1", l_l1.item(), global_step)
                if l_perc is not None:
                    writer.add_scalar("train/loss_perc", l_perc.item(), global_step)
                if l_cc is not None:
                    writer.add_scalar("train/loss_cc", l_cc.item(), global_step)
                if l_photo is not None:
                    writer.add_scalar("train/loss_photo", l_photo.item(), global_step)
                if l_entropy is not None:
                    writer.add_scalar("train/loss_entropy", l_entropy.item(), global_step)
                if l_ds is not None:
                    writer.add_scalar("train/loss_ds", l_ds.item(), global_step)
                if l_cluster is not None:
                    writer.add_scalar("train/loss_cluster", l_cluster.item(), global_step)
                if use_dlb and dynamic_balancer is not None:
                    writer.add_scalar("train/dlb_w_l1", lams["l1"], global_step)
                    writer.add_scalar("train/dlb_w_perc", lams["perc"], global_step)
                    writer.add_scalar("train/dlb_w_cc", lams["cc"], global_step)
                    writer.add_scalar("train/dlb_w_photo", lams["photo"], global_step)
                    writer.add_scalar("train/dlb_w_entropy", lams["entropy"], global_step)
                    writer.add_scalar("train/dlb_w_ds", lams["ds"], global_step)
                    writer.add_scalar("train/dlb_w_cluster", lams["cluster"], global_step)
                    writer.add_scalar("train/dlb_w_adv", lams["adv"], global_step)
                if adv_active:
                    writer.add_scalar("train/loss_D", loss_D.item(), global_step)
                    writer.add_scalar("train/loss_adv_G", l_adv_term.item() if l_adv_term is not None else 0.0, global_step)
                if log_img_every > 0 and (global_step % log_img_every == 0):
                    with torch.no_grad():
                        rgb = lab_to_rgb_tensor(L, out["a"], out["b"])  # (B,3,H,W)
                        writer.add_images("train/rgb_pred", rgb, global_step)
            global_step += 1
            # пошаговое обновление шедулера на каждой итерации (если включён)
            if scheduler is not None:
                scheduler.step()
        if is_main_process():
            avg_epoch_loss = epoch_loss / len(dl)
            active_losses = " ".join([f"{k}={v:.1f}" for k, v in loss_weights.items() if v > 0])
            log_msg = f"Epoch {epoch} done: loss={avg_epoch_loss:.4f} | phase={phase} | {active_losses} | omm_read_only={omm_epoch_read_only}"
            print(log_msg)

        # Валидация
        if do_val and (epoch % int(train_cfg.get("validate_every_epochs", 1)) == 0):
            model.eval()
            ssim_vals = []
            lpips_wrap = try_lpips() if bool(val_cfg.get("lpips", False)) else None
            lpips_vals = []
            with torch.no_grad():
                for L, ab, _ in dl_val:
                    L = L.to(device)
                    ab = ab.to(device)
                    out = model(L, omm_read_only=True)
                    rgb_pred = lab_to_rgb_tensor(L, out["a"], out["b"])  # (B,3,H,W)
                    rgb_gt = lab_to_rgb_tensor(L, ab[:, :1], ab[:, 1:2])
                    ssim_vals.append(
                        ssim(
                            rgb_pred,
                            rgb_gt,
                            window_size=int(val_cfg.get("ssim_window", 11)),
                            sigma=float(val_cfg.get("ssim_sigma", 1.5)),
                        ).item()
                    )
                    if lpips_wrap is not None and lpips_wrap.enabled:
                        val = lpips_wrap(rgb_pred, rgb_gt)
                        if val is not None:
                            lpips_vals.append(val.item())
            ssim_mean = float(sum(ssim_vals) / max(1, len(ssim_vals)))
            if writer is not None:
                writer.add_scalar("val/ssim", ssim_mean, epoch)
            if lpips_vals and writer is not None:
                writer.add_scalar("val/lpips", float(sum(lpips_vals) / len(lpips_vals)), epoch)

            # Сохранение чекпойнтов
            if is_main_process():
                ckpt_dir = Path(paths.get("checkpoints", "checkpoints"))
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Latest
                if save_latest:
                    to_save = model.module if isinstance(model, DDP) else model
                    torch.save({"model": to_save.state_dict(), "epoch": epoch}, ckpt_dir / latest_names.get("model", "latest.pth"))
                    if use_ema and ema is not None:
                        torch.save({"model": ema.state_dict(), "epoch": epoch}, ckpt_dir / latest_names.get("ema", "latest_ema.pth"))

                # Best
                if save_best:
                    if best_metric == "ssim":
                        score = ssim_mean
                        is_better = (best_score is None) or (score > best_score)
                    elif best_metric == "lpips":
                        if lpips_vals:
                            score = float(sum(lpips_vals) / len(lpips_vals))
                            is_better = (best_score is None) or (score < best_score)
                        else:
                            is_better = False
                            score = None
                    else:
                        is_better = False
                        score = None
                    if is_better:
                        best_score = score
                        to_save = model.module if isinstance(model, DDP) else model
                        torch.save({"model": to_save.state_dict(), "epoch": epoch, "best_score": best_score}, ckpt_dir / best_names.get("model", "best.pth"))
                        if use_ema and ema is not None:
                            torch.save({"model": ema.state_dict(), "epoch": epoch, "best_score": best_score}, ckpt_dir / best_names.get("ema", "best_ema.pth"))

    # Сохранение latest в конце, если не сохранялось на валидации
    if is_main_process() and not do_val:
        ckpt_dir = Path(paths.get("checkpoints", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        to_save = model.module if isinstance(model, DDP) else model
        if save_latest:
            torch.save({"model": to_save.state_dict(), "epoch": epoch}, ckpt_dir / latest_names.get("model", "latest.pth"))
            if use_ema and ema is not None:
                torch.save({"model": ema.state_dict(), "epoch": epoch}, ckpt_dir / latest_names.get("ema", "latest_ema.pth"))

    if writer is not None:
        writer.flush()
        writer.close()
    if ddp_enabled:
        cleanup()


if __name__ == "__main__":
    main()
