#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Скрипт обучения модели колоризации изображений

Данный скрипт предоставляет полный цикл обучения модели колоризации изображений,
включая загрузку и подготовку данных, создание модели по конфигурации,
настройку процесса обучения и системы мотивации, а также валидацию и сохранение
результатов в процессе обучения.

Возможности:
- Конфигурация всего процесса обучения через YAML-файлы
- Поддержка различных архитектур модели из конфигурации
- Расширенный мониторинг и визуализация процесса обучения
- Система наград и наказаний для улучшения качества колоризации
- Поддержка различных стратегий обучения и балансировки потерь
- Автоматическое продолжение обучения после прерывания
- Распределенное обучение на нескольких GPU
"""

import os
import sys
import time
import yaml
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Добавляем корневую директорию проекта в путь поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из модулей проекта
from core.swin_unet import SwinUNet
from core.vit_semantic import ViTSemantic
from core.fpn_pyramid import FPNPyramid
from core.cross_attention_bridge import CrossAttentionBridge
from core.feature_fusion import MultiHeadFeatureFusion

from losses.patch_nce import PatchNCELoss
from losses.vgg_perceptual import VGGPerceptualLoss
from losses.gan_loss import GANLoss
from losses.dynamic_balancer import DynamicLossBalancer

from modules.guide_net import GuideNet
from modules.discriminator import Discriminator
from modules.memory_bank import MemoryBankModule
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.few_shot_adapter import AdaptableColorizer

from training.trainer import ColorizationTrainer, create_trainer, create_model_from_config
from training.validator import ColorizationValidator, create_validator
from training.scheduler import create_scheduler
from training.checkpoints import save_checkpoint, load_checkpoint, find_latest_checkpoint, CheckpointManager

from datasets.train_dataset import create_train_dataset, create_train_dataloader
from datasets.validation_dataset import create_validation_dataset, create_validation_dataloader

from utils.metrics import MetricsCalculator
from utils.visualization import ColorizationVisualizer
from utils.config_parser import load_config, ConfigParser


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Обучение модели колоризации изображений"
    )
    
    # Общие параметры
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Путь к файлу конфигурации обучения")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                        help="Путь к файлу конфигурации модели")
    parser.add_argument("--loss-config", type=str, default="configs/loss_config.yaml",
                        help="Путь к файлу конфигурации функций потерь")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Корневая директория с данными")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Директория для сохранения результатов")
    parser.add_argument("--experiment-name", type=str, default="",
                        help="Название эксперимента (если пусто, будет создано автоматически)")
    
    # Параметры обучения
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Размер батча (перезаписывает значение из конфига)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Количество эпох обучения (перезаписывает значение из конфига)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Скорость обучения (перезаписывает значение из конфига)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Количество рабочих процессов для загрузки данных")
    
    # Параметры управления процессом обучения
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Продолжить обучение с последнего чекпоинта")
    parser.add_argument("--resume-from", type=str, default="",
                        help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Частота запуска валидации (в эпохах)")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Частота сохранения чекпоинтов (в эпохах)")
    parser.add_argument("--log-freq", type=int, default=None,
                        help="Частота логирования метрик (в итерациях)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Фиксированный сид для воспроизводимости")
    
    # Параметры мультипроцессорного и распределенного обучения
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Использовать распределенное обучение")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Количество процессов для распределенного обучения")
    parser.add_argument("--rank", type=int, default=0,
                        help="Ранг процесса для распределенного обучения")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Индекс GPU для обучения (для однопроцессорного режима)")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456",
                        help="URL для инициализации процесса распределенного обучения")
    parser.add_argument("--dist-backend", type=str, default="nccl",
                        help="Бэкенд для распределенного обучения")
    
    # Дополнительные параметры
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Использовать автоматическую смешанную точность (AMP)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Использовать детерминированные алгоритмы для воспроизводимости")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Режим тестирования без фактического обучения")
    
    return parser.parse_args()


def setup_environment(args):
    """
    Настройка окружения для обучения.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        tuple: (device, rank, world_size) - устройство, ранг и размер мира для распределенного обучения
    """
    # Проверка наличия GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        print(f"Доступно GPU: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("GPU недоступен, используется CPU.")
        args.distributed = False  # Отключаем распределенное обучение, если нет GPU
    
    # Фиксируем сиды для воспроизводимости, если указан параметр seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"Установлен режим детерминированного выполнения с seed={args.seed}")
    else:
        torch.backends.cudnn.benchmark = True
        print("Включена оптимизация cudnn.benchmark для ускорения обучения")
    
    # Настройка распределенного обучения
    rank = 0
    world_size = 1
    
    if args.distributed:
        if args.gpu is not None:
            # Для указанного GPU
            rank = args.rank
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(args.gpu)
            
            # Инициализируем процесс распределенного обучения
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank
            )
            world_size = dist.get_world_size()
            print(f"Инициализировано распределенное обучение: ранг {rank}/{world_size}, GPU: {args.gpu}")
        else:
            # Автоматическое распределение по доступным GPU
            if torch.cuda.device_count() <= 1:
                print("Для распределенного обучения требуется более одного GPU, отключаем режим")
                args.distributed = False
            else:
                world_size = torch.cuda.device_count()
                # Инициализация не требуется здесь, так как она будет выполнена в spawn
    
    return device, rank, world_size


def load_configurations(args):
    """
    Загрузка конфигураций из файлов.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        tuple: (train_config, model_config, loss_config) - словари с конфигурациями
    """
    # Загрузка конфигурации обучения
    train_config = load_config(args.config)
    if train_config is None:
        raise ValueError(f"Не удалось загрузить конфигурацию обучения из {args.config}")
    
    # Загрузка конфигурации модели
    model_config = load_config(args.model_config)
    if model_config is None:
        raise ValueError(f"Не удалось загрузить конфигурацию модели из {args.model_config}")
    
    # Загрузка конфигурации функций потерь
    loss_config = load_config(args.loss_config)
    if loss_config is None:
        raise ValueError(f"Не удалось загрузить конфигурацию функций потерь из {args.loss_config}")
    
    # Перезаписываем значения из аргументов командной строки, если они указаны
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
        
    if args.epochs is not None:
        train_config['epochs'] = args.epochs
        
    if args.lr is not None:
        if 'optimizer' in train_config:
            train_config['optimizer']['lr'] = args.lr
        else:
            train_config['optimizer'] = {'lr': args.lr}
            
    if args.eval_freq is not None:
        train_config['eval_freq'] = args.eval_freq
        
    if args.save_freq is not None:
        train_config['save_freq'] = args.save_freq
        
    if args.log_freq is not None:
        train_config['log_freq'] = args.log_freq
        
    if args.workers is not None:
        train_config['num_workers'] = args.workers
            
    return train_config, model_config, loss_config


def setup_directories(args):
    """
    Создание директорий для сохранения результатов.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        tuple: (experiment_dir, checkpoints_dir, logs_dir) - пути к директориям
    """
    # Создаем имя эксперимента, если не указано
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    # Создаем директории
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    logs_dir = os.path.join(experiment_dir, "logs")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Сохраняем копии конфигураций в директории эксперимента
    if args.rank == 0:  # Только для главного процесса
        configs_dir = os.path.join(experiment_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        for config_path, config_name in [
            (args.config, "training_config.yaml"),
            (args.model_config, "model_config.yaml"),
            (args.loss_config, "loss_config.yaml")
        ]:
            if os.path.exists(config_path):
                target_path = os.path.join(configs_dir, config_name)
                # Копируем конфигурационный файл
                with open(config_path, 'r') as src_file:
                    with open(target_path, 'w') as dst_file:
                        dst_file.write(src_file.read())
    
    return experiment_dir, checkpoints_dir, logs_dir


def create_full_model(model_config, device):
    """
    Создает полную модель колоризации на основе конфигурации.
    
    Args:
        model_config (dict): Конфигурация модели
        device (torch.device): Устройство для размещения модели
        
    Returns:
        nn.Module: Созданная модель
    """
    # Извлекаем основные параметры
    img_size = model_config.get('img_size', 256)
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 2)  # a и b каналы для Lab
    
    # Параметры для Swin-UNet
    swin_config = model_config.get('swin_unet', {})
    swin_embed_dim = swin_config.get('embed_dim', 96)
    swin_depths = swin_config.get('depths', [2, 2, 6, 2])
    swin_num_heads = swin_config.get('num_heads', [3, 6, 12, 24])
    swin_window_size = swin_config.get('window_size', 8)
    swin_mlp_ratio = swin_config.get('mlp_ratio', 4.0)
    swin_drop_rate = swin_config.get('drop_rate', 0.0)
    swin_attn_drop_rate = swin_config.get('attn_drop_rate', 0.0)
    
    # Параметры для ViT
    vit_config = model_config.get('vit_semantic', {})
    vit_patch_size = vit_config.get('patch_size', 16)
    vit_embed_dim = vit_config.get('embed_dim', 768)
    vit_depth = vit_config.get('depth', 12)
    vit_num_heads = vit_config.get('num_heads', 12)
    vit_mlp_ratio = vit_config.get('mlp_ratio', 4.0)
    vit_drop_rate = vit_config.get('drop_rate', 0.0)
    
    # Параметры для FPN
    fpn_config = model_config.get('fpn_pyramid', {})
    fpn_in_channels = fpn_config.get('in_channels', [swin_embed_dim, swin_embed_dim*2, swin_embed_dim*4, swin_embed_dim*8])
    fpn_out_channels = fpn_config.get('out_channels', 256)
    fpn_use_pyramid_pooling = fpn_config.get('use_pyramid_pooling', True)
    
    # Параметры для CrossAttentionBridge
    ca_config = model_config.get('cross_attention', {})
    ca_num_heads = ca_config.get('num_heads', 8)
    ca_dropout_rate = ca_config.get('dropout_rate', 0.0)
    
    # Параметры для MultiHeadFeatureFusion
    ff_config = model_config.get('feature_fusion', {})
    ff_out_channels = ff_config.get('out_channels', 512)
    ff_num_heads = ff_config.get('num_heads', 8)
    
    # Создаем компоненты модели
    swin_unet = SwinUNet(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=swin_embed_dim,
        depths=swin_depths,
        num_heads=swin_num_heads,
        window_size=swin_window_size,
        mlp_ratio=swin_mlp_ratio,
        drop_rate=swin_drop_rate,
        attn_drop_rate=swin_attn_drop_rate,
        return_intermediate=True
    )
    
    vit_semantic = ViTSemantic(
        img_size=img_size,
        patch_size=vit_patch_size,
        in_channels=in_channels,
        embed_dim=vit_embed_dim,
        depth=vit_depth,
        num_heads=vit_num_heads,
        mlp_ratio=vit_mlp_ratio,
        drop_rate=vit_drop_rate
    )
    
    fpn = FPNPyramid(
        in_channels_list=fpn_in_channels,
        out_channels=fpn_out_channels,
        use_pyramid_pooling=fpn_use_pyramid_pooling
    )
    
    cross_attention = CrossAttentionBridge(
        swin_dim=fpn_out_channels,
        vit_dim=vit_embed_dim,
        num_heads=ca_num_heads,
        dropout_rate=ca_dropout_rate
    )
    
    feature_fusion = MultiHeadFeatureFusion(
        in_channels_list=[fpn_out_channels, vit_embed_dim],
        out_channels=ff_out_channels,
        num_heads=ff_num_heads
    )
    
    # Финальный слой
    final = nn.Conv2d(
        ff_out_channels,
        out_channels,
        kernel_size=1
    )
    
    # Создаем полную модель
    class FullColorizationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.swin_unet = swin_unet
            self.vit_semantic = vit_semantic
            self.fpn = fpn
            self.cross_attention = cross_attention
            self.feature_fusion = feature_fusion
            self.final = final
            
        def forward(self, x):
            # Swin-UNet обработка
            swin_features = self.swin_unet(x)
            
            # ViT обработка
            vit_features = self.vit_semantic(x)
            
            # FPN обработка
            fpn_features = self.fpn(swin_features[:-1])  # Используем промежуточные выходы
            
            # Cross-Attention между FPN и ViT
            attended_features = self.cross_attention(fpn_features, vit_features)
            
            # Слияние признаков
            fused_features = self.feature_fusion([attended_features, vit_features])
            
            # Финальное предсказание
            output = self.final(fused_features)
            
            # Возвращаем a и b каналы
            return {'a': output[:, 0:1], 'b': output[:, 1:2]}
    
    model = FullColorizationModel()
    model.to(device)
    
    return model


def create_intelligent_modules(model_config, device):
    """
    Создает интеллектуальные модули на основе конфигурации.
    
    Args:
        model_config (dict): Конфигурация модели
        device (torch.device): Устройство для размещения модулей
        
    Returns:
        dict: Словарь с интеллектуальными модулями
    """
    modules = {}
    
    # Настройки GuideNet
    if 'guide_net' in model_config and model_config['guide_net'].get('enabled', False):
        guide_config = model_config['guide_net']
        modules['guide_net'] = GuideNet(
            input_channels=guide_config.get('input_channels', 1),
            advice_channels=guide_config.get('advice_channels', 2),
            feature_dim=guide_config.get('feature_dim', 64),
            num_layers=guide_config.get('num_layers', 6),
            use_attention=guide_config.get('use_attention', True),
            use_confidence=guide_config.get('use_confidence', True),
            device=device
        ).to(device)
    
    # Настройки Discriminator
    if 'discriminator' in model_config and model_config['discriminator'].get('enabled', False):
        disc_config = model_config['discriminator']
        modules['discriminator'] = Discriminator(
            input_nc=disc_config.get('input_nc', 3),
            ndf=disc_config.get('ndf', 64),
            n_layers=disc_config.get('n_layers', 3),
            use_spectral_norm=disc_config.get('use_spectral_norm', True),
            reward_type=disc_config.get('reward_type', 'adaptive'),
            device=device
        ).to(device)
    
    # Настройки Memory Bank
    if 'memory_bank' in model_config and model_config['memory_bank'].get('enabled', False):
        mem_config = model_config['memory_bank']
        modules['memory_bank'] = MemoryBankModule(
            feature_dim=mem_config.get('feature_dim', 256),
            max_items=mem_config.get('max_items', 1000),
            index_type=mem_config.get('index_type', 'flat'),
            use_fusion=mem_config.get('use_fusion', True),
            device=device
        )
    
    # Настройки UncertaintyEstimation
    if 'uncertainty' in model_config and model_config['uncertainty'].get('enabled', False):
        unc_config = model_config['uncertainty']
        modules['uncertainty'] = UncertaintyEstimation(
            method=unc_config.get('method', 'guided'),
            num_samples=unc_config.get('num_samples', 5),
            dropout_rate=unc_config.get('dropout_rate', 0.2),
            device=device
        ).to(device)
    
    # Настройки AdaptableColorizer
    if 'adaptable' in model_config and model_config['adaptable'].get('enabled', False):
        adapt_config = model_config['adaptable']
        if 'model' in modules:  # Если основная модель уже создана
            modules['adaptable'] = AdaptableColorizer(
                adapter_type=adapt_config.get('adapter_type', 'standard'),
                bottleneck_dim=adapt_config.get('bottleneck_dim', 64),
                base_model=modules['model'],
                device=device
            ).to(device)
    
    return modules


def create_loss_functions(loss_config, device):
    """
    Создает функции потерь на основе конфигурации.
    
    Args:
        loss_config (dict): Конфигурация функций потерь
        device (torch.device): Устройство для размещения функций потерь
        
    Returns:
        tuple: (losses_dict, balancer) - словарь с функциями потерь и балансировщик
    """
    losses_dict = {}
    
    # L1 Loss
    if loss_config.get('l1_loss', {}).get('enabled', True):
        l1_config = loss_config.get('l1_loss', {})
        losses_dict['l1'] = {
            'weight': l1_config.get('weight', 10.0),
            'function': nn.L1Loss().to(device)
        }
    
    # MSE Loss
    if loss_config.get('mse_loss', {}).get('enabled', False):
        mse_config = loss_config.get('mse_loss', {})
        losses_dict['mse'] = {
            'weight': mse_config.get('weight', 10.0),
            'function': nn.MSELoss().to(device)
        }
    
    # PatchNCE Loss
    if loss_config.get('patch_nce', {}).get('enabled', True):
        pnce_config = loss_config.get('patch_nce', {})
        losses_dict['patch_nce'] = {
            'weight': pnce_config.get('weight', 1.0),
            'function': PatchNCELoss(
                temperature=pnce_config.get('temperature', 0.07),
                patch_size=pnce_config.get('patch_size', 16),
                n_patches=pnce_config.get('n_patches', 256),
                device=device
            ).to(device)
        }
    
    # VGG Perceptual Loss
    if loss_config.get('vgg_perceptual', {}).get('enabled', True):
        vgg_config = loss_config.get('vgg_perceptual', {})
        try:
            losses_dict['vgg'] = {
                'weight': vgg_config.get('weight', 1.0),
                'function': VGGPerceptualLoss(
                    layers=vgg_config.get('layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']),
                    weights=vgg_config.get('layer_weights', None),
                    criterion=vgg_config.get('criterion', 'l1'),
                    resize=vgg_config.get('resize', True),
                    normalize=vgg_config.get('normalize', True),
                    device=device
                ).to(device)
            }
        except Exception as e:
            print(f"Ошибка при создании VGG Perceptual Loss: {str(e)}")
            print("VGG Perceptual Loss не будет использоваться")
    
    # GAN Loss
    if loss_config.get('gan_loss', {}).get('enabled', True):
        gan_config = loss_config.get('gan_loss', {})
        losses_dict['gan'] = {
            'weight': gan_config.get('weight', 0.1),
            'function': GANLoss(
                gan_mode=gan_config.get('gan_mode', 'lsgan'),
                device=device
            ).to(device)
        }
    
    # Dynamic Loss Balancer
    balancer_config = loss_config.get('dynamic_balancer', {})
    if balancer_config.get('enabled', True):
        # Создаем начальные веса для балансировщика
        initial_weights = {name: config['weight'] for name, config in losses_dict.items()}
        
        balancer = DynamicLossBalancer(
            initial_weights=initial_weights,
            strategy=balancer_config.get('strategy', 'adaptive'),
            target_metric=balancer_config.get('target_metric', 'lpips'),
            learning_rate=balancer_config.get('learning_rate', 0.01)
        )
    else:
        balancer = None
    
    return losses_dict, balancer


def create_optimizer_and_scheduler(model, train_config):
    """
    Создает оптимизатор и планировщик скорости обучения.
    
    Args:
        model (nn.Module): Модель для оптимизации
        train_config (dict): Конфигурация обучения
        
    Returns:
        tuple: (optimizer, scheduler) - оптимизатор и планировщик
    """
    # Извлекаем параметры оптимизатора из конфигурации
    optimizer_config = train_config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('lr', 0.0001)
    weight_decay = optimizer_config.get('weight_decay', 0.0001)
    
    # Создаем оптимизатор
    if optimizer_type == 'adam':
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
    
    # Извлекаем параметры планировщика из конфигурации
    scheduler_config = train_config.get('scheduler', {})
    
    # Создаем планировщик
    scheduler = create_scheduler(optimizer, scheduler_config)
    
    return optimizer, scheduler


def setup_data_loaders(args, train_config, model_config, device, rank, world_size):
    """
    Настройка загрузчиков данных.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        train_config (dict): Конфигурация обучения
        model_config (dict): Конфигурация модели
        device (torch.device): Устройство для обучения
        rank (int): Ранг процесса (для распределенного обучения)
        world_size (int): Размер мира (для распределенного обучения)
        
    Returns:
        tuple: (train_loader, val_loader) - загрузчики для обучения и валидации
    """
    # Извлекаем параметры из конфигурации
    data_config = train_config.get('data', {})
    batch_size = train_config.get('batch_size', 16)
    num_workers = train_config.get('num_workers', 4)
    color_space = data_config.get('color_space', 'lab')
    img_size = model_config.get('img_size', 256)
    
    # Определяем пути к данным
    data_root = args.data_root
    grayscale_dir = data_config.get('grayscale_dir', 'train/grayscale')
    color_dir = data_config.get('color_dir', 'train/color')
    val_grayscale_dir = data_config.get('val_grayscale_dir', 'val/grayscale')
    val_color_dir = data_config.get('val_color_dir', 'val/color')
    
    # Создаем тренировочный датасет
    train_dataset = create_train_dataset(
        data_root=data_root,
        grayscale_dir=grayscale_dir,
        color_dir=color_dir,
        color_space=color_space,
        img_size=img_size,
        augmentation_level=data_config.get('augmentation_level', 'medium'),
        max_dataset_size=data_config.get('max_train_samples', None),
        reference_dataset_path=data_config.get('reference_dataset_path', None)
    )
    
    # Создаем валидационный датасет
    val_dataset = create_validation_dataset(
        data_root=data_root,
        grayscale_dir=val_grayscale_dir,
        color_dir=val_color_dir,
        color_space=color_space,
        img_size=img_size,
        center_crop=True,
        max_dataset_size=data_config.get('max_val_samples', None)
    )
    
    print(f"Создан тренировочный датасет с {len(train_dataset)} изображениями")
    print(f"Создан валидационный датасет с {len(val_dataset)} изображениями")
    
    # Создаем загрузчики данных
    train_loader = create_train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=not args.distributed,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        dynamic_batching=data_config.get('dynamic_batching', False)
    )
    
    val_loader = create_validation_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def setup_logging_and_visualization(args, logs_dir, rank):
    """
    Настройка логирования и визуализации.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        logs_dir (str): Директория для логов
        rank (int): Ранг процесса (для распределенного обучения)
        
    Returns:
        tuple: (logger, visualizer) - объекты для логирования и визуализации
    """
    # Инициализируем логгер
    import logging
    logger = logging.getLogger("TintoraAI")
    
    # Настройка логирования только для главного процесса в распределенном режиме
    if not args.distributed or rank == 0:
        logger.setLevel(logging.INFO)
        
        # Логирование в файл
        log_file = os.path.join(logs_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Логирование в консоль
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # Создаем визуализатор
        results_dir = os.path.join(os.path.dirname(logs_dir), "results")
        os.makedirs(results_dir, exist_ok=True)
        visualizer = ColorizationVisualizer(output_dir=results_dir)
    else:
        # Для остальных процессов - минимальное логирование
        logger.setLevel(logging.ERROR)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        logger.addHandler(console_handler)
        
        # Пустой визуализатор для дочерних процессов
        visualizer = None
    
    return logger, visualizer


def train_model(
    args, model, intelligent_modules, losses_dict, loss_balancer,
    optimizer, scheduler, train_loader, val_loader,
    checkpoints_dir, logs_dir, device, rank, world_size,
    logger, visualizer
):
    """
    Обучение модели колоризации.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        model (nn.Module): Модель для обучения
        intelligent_modules (dict): Словарь с интеллектуальными модулями
        losses_dict (dict): Словарь с функциями потерь
        loss_balancer (DynamicLossBalancer): Балансировщик потерь
        optimizer (torch.optim.Optimizer): Оптимизатор
        scheduler (torch.optim.lr_scheduler._LRScheduler): Планировщик
        train_loader (torch.utils.data.DataLoader): Загрузчик тренировочных данных
        val_loader (torch.utils.data.DataLoader): Загрузчик валидационных данных
        checkpoints_dir (str): Директория для сохранения чекпоинтов
        logs_dir (str): Директория для логов
        device (torch.device): Устройство для обучения
        rank (int): Ранг процесса (для распределенного обучения)
        world_size (int): Размер мира (для распределенного обучения)
        logger (logging.Logger): Логгер
        visualizer (ColorizationVisualizer): Визуализатор
    """
    # Извлекаем параметры обучения из конфигурации
    train_config = load_config(args.config)
    epochs = train_config.get('epochs', 100)
    eval_freq = train_config.get('eval_freq', 1)
    save_freq = train_config.get('save_freq', 1)
    log_freq = train_config.get('log_freq', 10)
    
    # Создаем калькулятор метрик
    metrics_calc = MetricsCalculator(
        metrics=['psnr', 'ssim', 'lpips'],
        device=device
    )
    
    # Создаем менеджер чекпоинтов
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoints_dir,
        max_checkpoints=train_config.get('max_checkpoints', 5),
        save_best=True,
        metric_name='val_loss',
        mode='min'
    )
    
    # Если нужно продолжить обучение с чекпоинта
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume or args.resume_from:
        # Определяем путь к чекпоинту
        if args.resume_from:
            checkpoint_path = args.resume_from
        else:
            checkpoint_path = find_latest_checkpoint(checkpoints_dir)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Загружаем чекпоинт
            checkpoint = load_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_balancer=loss_balancer,
                device=device
            )
            
            # Извлекаем параметры обучения
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            
            if 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
                best_val_loss = checkpoint['metrics']['val_loss']
                
            logger.info(f"Продолжаем обучение с эпохи {start_epoch}, шаг {global_step}")
            logger.info(f"Загружен чекпоинт: {checkpoint_path}")
        else:
            logger.warning(f"Чекпоинт не найден, начинаем обучение с нуля")
    
    # Оборачиваем модель в DDP для распределенного обучения
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        
        # Оборачиваем интеллектуальные модули
        for name, module in intelligent_modules.items():
            if isinstance(module, nn.Module):
                intelligent_modules[name] = DDP(module, device_ids=[args.gpu])
    
    # Инициализируем scaler для автоматической смешанной точности (AMP)
    scaler = GradScaler() if args.amp else None
    
    # Запускаем основной цикл обучения
    for epoch in range(start_epoch, epochs):
        # Устанавливаем режим обучения
        model.train()
        for module in intelligent_modules.values():
            if isinstance(module, nn.Module):
                module.train()
        
        # Сбрасываем сэмплер для распределенного обучения
        if args.distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Инициализируем статистику эпохи
        epoch_losses = {}
        epoch_metrics = {}
        batch_count = 0
        epoch_start_time = time.time()
        
        # Перебираем батчи
        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Извлекаем данные
            grayscale = batch['grayscale'].to(device)
            target_color = batch['color'].to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Используем AMP, если включено
            with torch.cuda.amp.autocast() if args.amp else torch.no_grad():
                # Получаем предсказание от модели
                output = model(grayscale)
                
                # Используем интеллектуальные модули, если они есть
                if 'guide_net' in intelligent_modules:
                    guide_output = intelligent_modules['guide_net'](grayscale)
                    output['color_advice'] = guide_output['color_advice']
                    if 'confidence_map' in guide_output:
                        output['confidence_map'] = guide_output['confidence_map']
                
                # Объединяем L-канал с предсказанными a и b каналами
                predicted_lab = torch.cat([grayscale, output['a'], output['b']], dim=1)
                
                # Вычисляем потери
                batch_losses = {}
                
                # L1 Loss
                if 'l1' in losses_dict:
                    batch_losses['l1'] = losses_dict['l1']['function'](predicted_lab, target_color)
                
                # MSE Loss
                if 'mse' in losses_dict:
                    batch_losses['mse'] = losses_dict['mse']['function'](predicted_lab, target_color)
                
                # PatchNCE Loss
                if 'patch_nce' in losses_dict and hasattr(losses_dict['patch_nce']['function'], 'forward'):
                    batch_losses['patch_nce'] = losses_dict['patch_nce']['function'](grayscale, predicted_lab, target_color)
                
                # VGG Perceptual Loss
                if 'vgg' in losses_dict and hasattr(losses_dict['vgg']['function'], 'forward'):
                    batch_losses['vgg'] = losses_dict['vgg']['function'](predicted_lab, target_color)
                
                # GAN Loss
                if 'gan' in losses_dict and 'discriminator' in intelligent_modules:
                    # Преобразуем Lab или другое цветовое пространство в RGB для дискриминатора
                    # В реальной реализации здесь должно быть правильное преобразование
                    discriminator = intelligent_modules['discriminator']
                    batch_losses['gan'] = losses_dict['gan']['function'](discriminator(predicted_lab), True)
                
                # Применяем балансировку потерь
                if loss_balancer:
                    weights = loss_balancer.get_weights()
                    total_loss = sum(weights[name] * loss for name, loss in batch_losses.items() if name in weights)
                else:
                    # Если нет балансировщика, используем веса из конфигурации
                    total_loss = sum(losses_dict[name]['weight'] * loss for name, loss in batch_losses.items() if name in losses_dict)
            
            # Обратное распространение с AMP или без
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # Обновляем счетчики и статистику
            batch_count += 1
            global_step += 1
            
            # Записываем потери для логирования
            for name, loss in batch_losses.items():
                if name not in epoch_losses:
                    epoch_losses[name] = 0.0
                epoch_losses[name] += loss.item()
            
            # Логируем статистику батча
            if i % log_freq == 0 and (not args.distributed or rank == 0):
                batch_time = time.time() - batch_start_time
                lr = optimizer.param_groups[0]['lr']
                
                # Формируем строку с потерями
                losses_str = ", ".join(f"{name}: {loss.item():.4f}" for name, loss in batch_losses.items())
                logger.info(f"Эпоха [{epoch+1}/{epochs}], Батч [{i}/{len(train_loader)}], "
                           f"LR: {lr:.6f}, Потери: {losses_str}, "
                           f"Время: {batch_time:.2f}с")
            
            # Очищаем память GPU
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
                
        # Конец эпохи, вычисляем средние потери
        for name in epoch_losses:
            epoch_losses[name] /= batch_count
            
        # Выводим статистику эпохи
        if not args.distributed or rank == 0:
            epoch_time = time.time() - epoch_start_time
            losses_str = ", ".join(f"{name}: {loss:.4f}" for name, loss in epoch_losses.items())
            logger.info(f"Эпоха [{epoch+1}/{epochs}] завершена, "
                       f"Потери: {losses_str}, "
                       f"Время: {epoch_time:.2f}с")
        
        # Запускаем валидацию, если нужно
        if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
            val_losses, val_metrics = validate_model(
                model, intelligent_modules, losses_dict, val_loader,
                metrics_calc, visualizer, device, epoch, args.distributed, rank
            )
            
            # Обновляем балансировщик потерь на основе метрик
            if loss_balancer and not args.distributed or rank == 0:
                all_metrics = {**val_metrics}
                for name, loss in val_losses.items():
                    all_metrics[f"val_{name}"] = loss
                loss_balancer.update_weights(all_metrics)
                
                # Логируем обновленные веса
                weights = loss_balancer.get_weights()
                weights_str = ", ".join(f"{name}: {weight:.4f}" for name, weight in weights.items())
                logger.info(f"Обновлены веса потерь: {weights_str}")
            
            # Сохраняем лучшую модель
            val_loss = val_losses.get('total', float('inf'))
            if not args.distributed or rank == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"Новая лучшая модель! Валидационная потеря: {val_loss:.4f}")
        
        # Обновляем планировщик
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses.get('total', 0))
            else:
                scheduler.step()
        
        # Сохраняем чекпоинт
        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            if not args.distributed or rank == 0:
                # Создаем словарь с интеллектуальными модулями для сохранения
                modules_dict = {}
                for name, module in intelligent_modules.items():
                    if isinstance(module, nn.Module):
                        if isinstance(module, DDP):
                            modules_dict[name] = module.module
                        else:
                            modules_dict[name] = module
                
                # Сохраняем чекпоинт
                checkpoint_manager.save(
                    model=model.module if isinstance(model, DDP) else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics={
                        'train_loss': epoch_losses.get('total', 0),
                        'val_loss': val_losses.get('total', 0) if 'val_losses' in locals() else 0,
                        **(val_metrics if 'val_metrics' in locals() else {})
                    },
                    modules=modules_dict,
                    loss_balancer=loss_balancer,
                    scaler=scaler
                )
                logger.info(f"Сохранен чекпоинт для эпохи {epoch+1}")
    
    # Завершаем обучение
    if not args.distributed or rank == 0:
        logger.info("Обучение завершено!")
        
        # Сохраняем финальный чекпоинт
        final_checkpoint_path = os.path.join(checkpoints_dir, "final_model.pth")
        save_checkpoint(
            model=model.module if isinstance(model, DDP) else model,
            optimizer=optimizer,
            path=final_checkpoint_path,
            epoch=epochs-1,
            global_step=global_step
        )
        logger.info(f"Финальная модель сохранена в {final_checkpoint_path}")


def validate_model(model, intelligent_modules, losses_dict, val_loader, metrics_calc, visualizer, device, epoch, distributed, rank):
    """
    Валидация модели.
    
    Args:
        model (nn.Module): Модель для валидации
        intelligent_modules (dict): Словарь с интеллектуальными модулями
        losses_dict (dict): Словарь с функциями потерь
        val_loader (torch.utils.data.DataLoader): Загрузчик валидационных данных
        metrics_calc (MetricsCalculator): Калькулятор метрик
        visualizer (ColorizationVisualizer): Визуализатор
        device (torch.device): Устройство для валидации
        epoch (int): Текущая эпоха
        distributed (bool): Флаг распределенного режима
        rank (int): Ранг процесса
        
    Returns:
        tuple: (val_losses, val_metrics) - потери и метрики валидации
    """
    # Устанавливаем режим оценки
    model.eval()
    for module in intelligent_modules.values():
        if isinstance(module, nn.Module):
            module.eval()
    
    # Инициализируем статистику валидации
    val_losses = {}
    val_batch_count = 0
    all_metrics = {}
    visualize_count = 0
    max_visualize = 8  # Максимальное количество изображений для визуализации
    
    # Собираем данные для визуализации
    vis_grayscales = []
    vis_predictions = []
    vis_targets = []
    
    # Выполняем валидацию без вычисления градиентов
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Извлекаем данные
            grayscale = batch['grayscale'].to(device)
            target_color = batch['color'].to(device)
            
            # Получаем предсказание от модели
            output = model(grayscale)
            
            # Используем интеллектуальные модули, если они есть
            if 'guide_net' in intelligent_modules:
                guide_output = intelligent_modules['guide_net'](grayscale)
                output['color_advice'] = guide_output['color_advice']
                if 'confidence_map' in guide_output:
                    output['confidence_map'] = guide_output['confidence_map']
            
            # Объединяем L-канал с предсказанными a и b каналами
            predicted_lab = torch.cat([grayscale, output['a'], output['b']], dim=1)
            
            # Вычисляем потери
            batch_losses = {}
            
            # L1 Loss
            if 'l1' in losses_dict:
                batch_losses['l1'] = losses_dict['l1']['function'](predicted_lab, target_color)
            
            # MSE Loss
            if 'mse' in losses_dict:
                batch_losses['mse'] = losses_dict['mse']['function'](predicted_lab, target_color)
            
            # PatchNCE Loss
            if 'patch_nce' in losses_dict and hasattr(losses_dict['patch_nce']['function'], 'forward'):
                batch_losses['patch_nce'] = losses_dict['patch_nce']['function'](grayscale, predicted_lab, target_color)
            
            # VGG Perceptual Loss
            if 'vgg' in losses_dict and hasattr(losses_dict['vgg']['function'], 'forward'):
                batch_losses['vgg'] = losses_dict['vgg']['function'](predicted_lab, target_color)
                
            # Вычисляем общую потерю
            total_loss = sum(loss for loss in batch_losses.values())
            batch_losses['total'] = total_loss
            
            # Записываем потери
            for name, loss in batch_losses.items():
                if name not in val_losses:
                    val_losses[name] = 0.0
                val_losses[name] += loss.item()
            
            # Вычисляем метрики
            batch_metrics = metrics_calc.calculate(predicted_lab, target_color)
            
            # Записываем метрики
            for name, value in batch_metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = 0.0
                all_metrics[name] += value
            
            # Собираем данные для визуализации
            if not distributed or rank == 0:
                if visualize_count < max_visualize:
                    # Берем только первое изображение из батча для визуализации
                    vis_grayscales.append(grayscale[0].cpu().numpy())
                    vis_predictions.append(predicted_lab[0].cpu().numpy())
                    vis_targets.append(target_color[0].cpu().numpy())
                    
                    visualize_count += 1
            
            val_batch_count += 1
    
    # Вычисляем средние потери и метрики
    for name in val_losses:
        val_losses[name] /= val_batch_count
        
    for name in all_metrics:
        all_metrics[name] /= val_batch_count
    
    # Создаем визуализации
    if not distributed or rank == 0:
        if visualizer and vis_grayscales:
            # Создаем сетку с результатами колоризации
            visualizer.create_grid(
                grayscales=vis_grayscales,
                colorized=vis_predictions,
                originals=vis_targets,
                filename=f"validation_epoch_{epoch+1}.png"
            )
            
            # Создаем отдельные сравнения для нескольких изображений
            for i in range(min(4, len(vis_grayscales))):
                visualizer.create_comparison(
                    grayscale=vis_grayscales[i],
                    colorized=vis_predictions[i],
                    original=vis_targets[i],
                    filename=f"comparison_epoch_{epoch+1}_sample_{i+1}.png"
                )
    
    return val_losses, all_metrics


def run_training_process(rank, world_size, args):
    """
    Функция для запуска процесса обучения в одном процессе.
    
    Args:
        rank (int): Ранг процесса
        world_size (int): Размер мира (количество процессов)
        args (argparse.Namespace): Аргументы командной строки
    """
    # Устанавливаем GPU для этого процесса
    if args.distributed:
        args.gpu = rank
        torch.cuda.set_device(rank)
        
        # Инициализируем группу процессов
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank
        )
    
    # Настройка окружения
    device, rank, world_size = setup_environment(args)
    
    # Загрузка конфигураций
    train_config, model_config, loss_config = load_configurations(args)
    
    # Создание директорий для сохранения результатов
    experiment_dir, checkpoints_dir, logs_dir = setup_directories(args)
    
    # Настройка логирования и визуализации
    logger, visualizer = setup_logging_and_visualization(args, logs_dir, rank)
    
    # Создаем модель и интеллектуальные модули
    model = create_full_model(model_config, device)
    intelligent_modules = create_intelligent_modules(model_config, device)
    
    # Создаем функции потерь и балансировщик
    losses_dict, loss_balancer = create_loss_functions(loss_config, device)
    
    # Создаем оптимизатор и планировщик
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)
    
    # Настройка загрузчиков данных
    train_loader, val_loader = setup_data_loaders(args, train_config, model_config, device, rank, world_size)
    
    # Запускаем обучение
    if not args.dry_run:
        train_model(
            args, model, intelligent_modules, losses_dict, loss_balancer,
            optimizer, scheduler, train_loader, val_loader,
            checkpoints_dir, logs_dir, device, rank, world_size,
            logger, visualizer
        )
    else:
        logger.info("Режим dry run: проверка конфигурации без обучения")
    
    # Очищаем процессы для распределенного обучения
    if args.distributed:
        dist.destroy_process_group()


def main():
    """
    Основная функция запуска обучения.
    """
    # Парсинг аргументов командной строки
    args = parse_args()
    
    if args.distributed:
        # Многопроцессорный запуск
        world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
        mp.spawn(
            run_training_process,
            args=(world_size, args),
            nprocs=world_size
        )
    else:
        # Однопроцессорный запуск
        run_training_process(0, 1, args)


if __name__ == "__main__":
    main()
