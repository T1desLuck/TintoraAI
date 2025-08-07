"""
Trainer: Основной класс для обучения модели колоризации.

Данный модуль предоставляет функциональность для обучения моделей колоризации,
включая организацию процесса тренировки, работу с лосс-функциями, оптимизацию,
логирование метрик и визуализацию результатов. Он объединяет различные компоненты
системы для эффективного и контролируемого обучения.

Ключевые особенности:
- Гибкая конфигурация процесса обучения через YAML-файлы
- Поддержка различных лосс-функций с динамическим балансированием
- Интеграция с интеллектуальными модулями для улучшения обучения
- Система наград и наказаний для улучшения реалистичности результатов
- Подробное логирование метрик и сохранение чекпоинтов

Преимущества:
- Полный контроль над процессом обучения с обширной настройкой параметров
- Интеграция с системой мониторинга для отслеживания прогресса
- Поддержка смешанной точности для ускорения обучения
- Гибкая интеграция с различными датасетами и модулями
"""

import os
import time
import datetime
import json
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import tqdm

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
from modules.style_transfer import StyleTransfer
from modules.memory_bank import MemoryBankModule
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.few_shot_adapter import AdaptableColorizer

from utils.config_parser import ConfigParser, load_config
from utils.metrics import MetricsCalculator
from utils.visualization import ColorizationVisualizer
from utils.data_loader import prepare_batch_for_colorization

from .scheduler import create_scheduler
from .checkpoints import save_checkpoint, load_checkpoint
from .validator import ColorizationValidator


class ColorizationTrainer:
    """
    Основной класс для обучения модели колоризации.
    
    Args:
        model (nn.Module): Модель колоризации
        config (Dict): Конфигурация обучения
        train_loader (DataLoader): Загрузчик тренировочных данных
        val_loader (DataLoader, optional): Загрузчик данных для валидации
        device (torch.device): Устройство для вычислений
        experiment_dir (str): Директория для сохранения результатов
        distributed (bool): Включить распределенное обучение
        rank (int): Ранг текущего процесса (для распределенного обучения)
        world_size (int): Общее количество процессов (для распределенного обучения)
    """
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        experiment_dir: Optional[str] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Устанавливаем устройство
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Устанавливаем директорию эксперимента
        self.experiment_dir = experiment_dir or os.path.join('./experiments', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        if self.is_main_process:
            os.makedirs(self.experiment_dir, exist_ok=True)
            
        # Настраиваем логирование
        self.logger = self._setup_logging()
        
        # Настраиваем tensorboard
        self.tensorboard_dir = os.path.join(self.experiment_dir, 'tensorboard')
        if self.is_main_process:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        # Настраиваем визуализацию
        self.visualization_dir = os.path.join(self.experiment_dir, 'visualizations')
        if self.is_main_process:
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.visualizer = ColorizationVisualizer(output_dir=self.visualization_dir)
        
        # Настраиваем модель
        self.model = model
        self.model.to(self.device)
        
        # Настраиваем распределенное обучение, если нужно
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            
        # Парсим параметры обучения из конфигурации
        self._parse_training_config()
        
        # Инициализируем оптимизатор и планировщик скорости обучения
        self.optimizer = self._create_optimizer()
        self.scheduler = create_scheduler(self.optimizer, self.lr_scheduler_config)
        
        # Инициализируем лосс-функции
        self.losses = self._create_losses()
        
        # Инициализируем интеллектуальные модули
        self.modules = self._create_modules()
        
        # Инициализируем динамический балансировщик лосса
        self.loss_balancer = self._create_loss_balancer()
        
        # Инициализируем калькулятор метрик
        self.metrics_calculator = MetricsCalculator(
            metrics=self.metrics_config.get('metrics', ['psnr', 'ssim', 'lpips']),
            device=self.device
        )
        
        # Инициализируем валидатор
        if self.val_loader is not None:
            self.validator = ColorizationValidator(
                model=self.model if not self.distributed else self.model.module,
                val_loader=self.val_loader,
                losses=self.losses,
                metrics_calculator=self.metrics_calculator,
                visualizer=self.visualizer if self.is_main_process else None,
                config=self.config,
                device=self.device,
                experiment_dir=self.experiment_dir,
                distributed=self.distributed,
                rank=self.rank,
                world_size=self.world_size
            )
            
        # Инициализируем скейлер для смешанной точности
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Счетчики итераций
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('inf')
        self.epochs_without_improvement = 0
        
        # Путь для сохранения чекпоинта
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Логируем информацию о модели и конфигурации
        if self.is_main_process:
            self._log_configuration()
            
        self.logger.info(f"Инициализирован ColorizationTrainer на устройстве {self.device}")
            
    def train(
        self,
        resume_from: Optional[str] = None,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """
        Обучает модель.
        
        Args:
            resume_from (str, optional): Путь к чекпоинту для восстановления обучения
            start_epoch (int): Начальная эпоха
            
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        # Восстанавливаем обучение, если указан чекпоинт
        if resume_from:
            self._load_checkpoint(resume_from)
            self.logger.info(f"Восстановлено обучение с чекпоинта {resume_from}")
            
        # Устанавливаем начальную эпоху
        self.current_epoch = start_epoch
        
        # Инициализируем трекеры для метрик
        train_losses = []
        val_metrics = []
        
        # Основной цикл обучения
        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                self.logger.info(f"Начало эпохи {epoch + 1}/{self.max_epochs}")
                
                # Обучаем одну эпоху
                epoch_losses = self._train_epoch()
                train_losses.append(epoch_losses)
                
                # Логируем потери
                if self.is_main_process:
                    for loss_name, loss_value in epoch_losses.items():
                        self.writer.add_scalar(f'train/{loss_name}', loss_value, epoch)
                    
                # Сохраняем чекпоинт
                if (epoch + 1) % self.checkpoint_interval == 0 and self.is_main_process:
                    save_checkpoint(
                        model=self.model if not self.distributed else self.model.module,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        global_step=self.global_step,
                        config=self.config,
                        metrics=epoch_losses,
                        path=os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                    )
                    self.logger.info(f"Сохранен чекпоинт для эпохи {epoch + 1}")
                
                # Валидация
                if self.val_loader is not None and (epoch + 1) % self.validation_interval == 0:
                    self.logger.info(f"Валидация для эпохи {epoch + 1}")
                    val_results = self.validator.validate(epoch)
                    val_metrics.append(val_results)
                    
                    if self.is_main_process:
                        # Логируем метрики валидации
                        for metric_name, metric_value in val_results.get('metrics', {}).items():
                            self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
                            
                        # Проверяем улучшение и сохраняем лучшую модель
                        val_score = val_results.get('metrics', {}).get(self.early_stopping_metric, float('inf'))
                        if self._is_improvement(val_score):
                            self.best_val_score = val_score
                            self.epochs_without_improvement = 0
                            
                            # Сохраняем лучшую модель
                            save_checkpoint(
                                model=self.model if not self.distributed else self.model.module,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch=epoch,
                                global_step=self.global_step,
                                config=self.config,
                                metrics=val_results.get('metrics', {}),
                                path=os.path.join(self.checkpoint_dir, 'best_model.pth')
                            )
                            self.logger.info(f"Сохранена лучшая модель с {self.early_stopping_metric}={val_score:.4f}")
                        else:
                            self.epochs_without_improvement += 1
                            self.logger.info(f"Нет улучшения {self.early_stopping_metric} в течение {self.epochs_without_improvement} эпох")
                            
                            # Проверяем ранний останов
                            if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                                self.logger.info(f"Ранний останов после {epoch + 1} эпох")
                                break
                                
                # Обновляем планировщик скорости обучения
                if self.scheduler is not None:
                    if self.scheduler_step_on == 'epoch':
                        if self.scheduler_metric is not None and self.val_loader is not None and (epoch + 1) % self.validation_interval == 0:
                            # Для планировщиков, которые используют метрику (например, ReduceLROnPlateau)
                            val_metric = val_results.get('metrics', {}).get(self.scheduler_metric, None)
                            if val_metric is not None:
                                self.scheduler.step(val_metric)
                        else:
                            # Для обычных планировщиков
                            self.scheduler.step()
                            
                        # Логируем скорость обучения
                        for param_group in self.optimizer.param_groups:
                            if self.is_main_process:
                                self.writer.add_scalar('train/learning_rate', param_group['lr'], epoch)
                
                # Обновляем динамический балансировщик лосса
                if self.loss_balancer is not None and (epoch + 1) % self.loss_balancer_update_interval == 0:
                    if self.val_loader is not None and (epoch + 1) % self.validation_interval == 0:
                        # Используем метрики валидации для обновления весов
                        val_metrics_dict = val_results.get('metrics', {})
                        self.loss_balancer.update_weights(val_metrics_dict)
                    else:
                        # Используем потери обучения для обновления весов
                        self.loss_balancer.update_weights(epoch_losses)
                    
                    # Логируем новые веса
                    if self.is_main_process:
                        weights = self.loss_balancer.get_weights()
                        for loss_name, weight in weights.items():
                            self.writer.add_scalar(f'weights/{loss_name}', weight, epoch)
                            
                self.logger.info(f"Завершена эпоха {epoch + 1}/{self.max_epochs}")
                
            # Сохраняем последний чекпоинт
            if self.is_main_process:
                save_checkpoint(
                    model=self.model if not self.distributed else self.model.module,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    config=self.config,
                    metrics=epoch_losses,
                    path=os.path.join(self.checkpoint_dir, 'final_model.pth')
                )
                self.logger.info(f"Сохранен финальный чекпоинт")
                
            # Закрываем tensorboard writer
            if self.is_main_process:
                self.writer.close()
                
            # Формируем результаты обучения
            results = {
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'best_val_score': self.best_val_score,
                'epochs_trained': self.current_epoch + 1,
                'global_step': self.global_step,
                'best_model_path': os.path.join(self.checkpoint_dir, 'best_model.pth'),
                'final_model_path': os.path.join(self.checkpoint_dir, 'final_model.pth')
            }
            
            return results
            
        except KeyboardInterrupt:
            self.logger.info("Обучение прервано пользователем")
            
            # Сохраняем текущий чекпоинт
            if self.is_main_process:
                save_checkpoint(
                    model=self.model if not self.distributed else self.model.module,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    config=self.config,
                    metrics={},
                    path=os.path.join(self.checkpoint_dir, 'interrupted_model.pth')
                )
                self.logger.info(f"Сохранен чекпоинт прерванного обучения")
                
                # Закрываем tensorboard writer
                self.writer.close()
                
            return {
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'interrupted': True,
                'epochs_trained': self.current_epoch + 1,
                'global_step': self.global_step,
                'interrupted_model_path': os.path.join(self.checkpoint_dir, 'interrupted_model.pth')
            }
            
    def _train_epoch(self) -> Dict[str, float]:
        """
        Обучает модель в течение одной эпохи.
        
        Returns:
            Dict[str, float]: Средние потери за эпоху
        """
        # Переводим модель в режим обучения
        self.model.train()
        
        # Инициализируем трекеры потерь
        epoch_losses = {}
        epoch_samples = 0
        
        # Обновляем набор данных для распределенного обучения
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)
            
        # Создаем прогресс-бар
        progress_bar = tqdm.tqdm(
            self.train_loader,
            desc=f"Эпоха {self.current_epoch + 1}/{self.max_epochs}",
            disable=not self.is_main_process
        )
        
        # Основной цикл обучения
        for batch in progress_bar:
            # Проверяем, является ли пакет кортежем или словарем
            if isinstance(batch, (list, tuple)):
                # Ожидаем кортеж (input_images, target_images)
                input_images, target_images = batch
                batch_size = input_images.shape[0]
            else:
                # Ожидаем словарь с ключами 'grayscale' и 'color'
                input_images = batch.get('grayscale')
                target_images = batch.get('color')
                batch_size = input_images.shape[0]
            
            # Подготавливаем входные данные
            input_images = input_images.to(self.device)
            target_images = target_images.to(self.device)
            
            # Очищаем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход с использованием смешанной точности, если включена
            if self.mixed_precision:
                with autocast():
                    # Прямой проход
                    outputs = self.model(input_images)
                    
                    # Вычисляем потери
                    losses = self._compute_losses(input_images, target_images, outputs)
                    
                    # Общая потеря
                    total_loss = self._compute_total_loss(losses)
                    
                # Обратный проход с использованием скейлера
                self.scaler.scale(total_loss).backward()
                
                # Клиппинг градиентов, если включен
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                # Обновляем веса с использованием скейлера
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Прямой проход
                outputs = self.model(input_images)
                
                # Вычисляем потери
                losses = self._compute_losses(input_images, target_images, outputs)
                
                # Общая потеря
                total_loss = self._compute_total_loss(losses)
                
                # Обратный проход
                total_loss.backward()
                
                # Клиппинг градиентов, если включен
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                # Обновляем веса
                self.optimizer.step()
                
            # Обновляем планировщик, если он работает на основе шага, а не эпохи
            if self.scheduler is not None and self.scheduler_step_on == 'step':
                self.scheduler.step()
                
            # Обновляем счетчик глобальных шагов
            self.global_step += 1
                
            # Обновляем трекеры потерь
            for loss_name, loss_value in losses.items():
                # Преобразуем значения в Python-скаляры
                loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                
                # Добавляем или обновляем значение в словаре потерь
                epoch_losses[loss_name] = epoch_losses.get(loss_name, 0) + loss_value_scalar * batch_size
                
            # Учитываем текущий пакет в общем количестве образцов
            epoch_samples += batch_size
            
            # Обновляем прогресс-бар с текущими потерями
            if self.is_main_process:
                postfix_dict = {
                    'loss': total_loss.item(),
                    **{f"{k}": f"{v/epoch_samples:.4f}" for k, v in epoch_losses.items()}
                }
                progress_bar.set_postfix(postfix_dict)
                
            # Логируем потери и образцы через заданные интервалы
            if self.is_main_process and self.global_step % self.log_interval == 0:
                # Логируем потери в tensorboard
                for loss_name, loss_value in losses.items():
                    loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                    self.writer.add_scalar(f'train/step/{loss_name}', loss_value_scalar, self.global_step)
                
                # Визуализируем результаты через заданные интервалы
                if self.global_step % self.visualization_interval == 0:
                    self._visualize_results(input_images, target_images, outputs)
                
        # Закрываем прогресс-бар
        progress_bar.close()
        
        # Вычисляем средние потери за эпоху
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= max(1, epoch_samples)
            
        # Логируем средние потери за эпоху
        if self.is_main_process:
            self.logger.info(f"Эпоха {self.current_epoch + 1} завершена. Потери: {epoch_losses}")
            
        return epoch_losses
    
    def _compute_losses(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисляет все потери для текущего пакета.
        
        Args:
            input_images (torch.Tensor): Входные изображения
            target_images (torch.Tensor): Целевые изображения
            outputs (torch.Tensor or Dict[str, torch.Tensor]): Выходы модели
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с потерями
        """
        # Извлекаем предсказанные изображения из выхода модели
        if isinstance(outputs, dict):
            pred_images = outputs.get('colorized', outputs.get('output', None))
            
            # Проверяем, есть ли в выходе отдельные каналы a и b (для пространства LAB)
            if pred_images is None and 'a' in outputs and 'b' in outputs:
                # Объединяем L-канал из входа с предсказанными a и b каналами
                a_channel = outputs['a']
                b_channel = outputs['b']
                
                # Проверяем, является ли входное изображение L-каналом
                if input_images.shape[1] == 1:
                    # Объединяем L из входа с предсказанными a и b
                    pred_images = torch.cat([input_images, a_channel, b_channel], dim=1)
                else:
                    # Если вход не L-канал, выводим ошибку
                    raise ValueError("Модель вернула отдельные a и b каналы, но вход не является L-каналом")
        else:
            pred_images = outputs
            
        # Проверка на None
        if pred_images is None:
            raise ValueError("Не удалось извлечь предсказанные изображения из выхода модели")
            
        # Инициализируем словарь потерь
        losses = {}
        
        # L1 потеря
        if self.l1_loss_enabled:
            losses['l1'] = nn.functional.l1_loss(pred_images, target_images)
            
        # L2 (MSE) потеря
        if self.l2_loss_enabled:
            losses['l2'] = nn.functional.mse_loss(pred_images, target_images)
            
        # Перцептивная потеря
        if self.perceptual_loss_enabled and 'perceptual' in self.losses:
            perceptual_loss = self.losses['perceptual'](pred_images, target_images)
            losses['perceptual'] = perceptual_loss
            
        # Потеря PatchNCE
        if self.patch_nce_loss_enabled and 'patch_nce' in self.losses:
            patch_nce_loss = self.losses['patch_nce'](input_images, pred_images, target_images)
            losses['patch_nce'] = patch_nce_loss
            
        # GAN потеря
        if self.gan_loss_enabled and 'gan' in self.losses and 'discriminator' in self.modules:
            # Детектируем реальное и поддельное
            real_detection = self.modules['discriminator'](target_images)
            fake_detection = self.modules['discriminator'](pred_images.detach())
            
            # Потеря дискриминатора
            d_loss_real = self.losses['gan'](real_detection, True)
            d_loss_fake = self.losses['gan'](fake_detection, False)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            losses['d_loss'] = d_loss
            
            # Обратный проход для дискриминатора
            if self.gan_discriminator_enabled:
                self.optimizer_d.zero_grad()
                
                if self.mixed_precision:
                    self.scaler.scale(d_loss).backward()
                    
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer_d)
                        torch.nn.utils.clip_grad_norm_(self.modules['discriminator'].parameters(), self.grad_clip)
                        
                    self.scaler.step(self.optimizer_d)
                else:
                    d_loss.backward()
                    
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.modules['discriminator'].parameters(), self.grad_clip)
                        
                    self.optimizer_d.step()
                
            # Потеря генератора
            fake_detection = self.modules['discriminator'](pred_images)
            g_loss = self.losses['gan'](fake_detection, True)
            losses['g_loss'] = g_loss
            
            # Добавляем награды или наказания в зависимости от качества предсказания
            if self.gan_reward_enabled and hasattr(self.modules['discriminator'], 'compute_rewards'):
                rewards = self.modules['discriminator'].compute_rewards(pred_images, target_images)
                reward_loss = rewards.mean() * self.gan_reward_weight
                losses['reward'] = reward_loss
            
        # Потеря советника GuideNet
        if self.guide_net_enabled and 'guide_net' in self.modules:
            # Получаем советы по цветам
            guide_result = self.modules['guide_net'](input_images)
            guide_advice = guide_result.get('color_advice')
            
            if guide_advice is not None:
                # Если советник возвращает только ab каналы для Lab
                if guide_advice.shape[1] == 2 and pred_images.shape[1] > 2:
                    # Извлекаем ab каналы из предсказания и цели
                    pred_ab = pred_images[:, 1:3]
                    target_ab = target_images[:, 1:3]
                    
                    # Вычисляем потерю советника
                    guide_loss = nn.functional.mse_loss(guide_advice, target_ab)
                    losses['guide_loss'] = guide_loss
                    
                    # Вычисляем потерю согласованности между предсказанием и советом
                    consistency_loss = nn.functional.mse_loss(pred_ab, guide_advice.detach())
                    losses['guide_consistency'] = consistency_loss
                else:
                    # Если советник возвращает полное изображение
                    guide_loss = nn.functional.mse_loss(guide_advice, target_images)
                    losses['guide_loss'] = guide_loss
                    
                    # Вычисляем потерю согласованности между предсказанием и советом
                    consistency_loss = nn.functional.mse_loss(pred_images, guide_advice.detach())
                    losses['guide_consistency'] = consistency_loss
                    
                # Обратный проход для советника
                if self.guide_net_train_separately:
                    self.optimizer_g.zero_grad()
                    
                    if self.mixed_precision:
                        self.scaler.scale(guide_loss).backward()
                        
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer_g)
                            torch.nn.utils.clip_grad_norm_(self.modules['guide_net'].parameters(), self.grad_clip)
                            
                        self.scaler.step(self.optimizer_g)
                    else:
                        guide_loss.backward()
                        
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.modules['guide_net'].parameters(), self.grad_clip)
                            
                        self.optimizer_g.step()
        
        # Потеря банка памяти
        if self.memory_bank_enabled and 'memory_bank' in self.modules:
            memory_result = self.modules['memory_bank'](input_images)
            
            if 'colorized' in memory_result:
                memory_colorized = memory_result['colorized']
                
                # Вычисляем потерю согласованности между предсказанием и банком памяти
                memory_consistency_loss = nn.functional.mse_loss(pred_images, memory_colorized.detach())
                losses['memory_consistency'] = memory_consistency_loss
        
        # Потеря оценки неопределенности
        if self.uncertainty_estimation_enabled and 'uncertainty_estimation' in self.modules:
            uncertainty_result = self.modules['uncertainty_estimation'](pred_images)
            
            if 'uncertainty' in uncertainty_result:
                uncertainty_map = uncertainty_result['uncertainty']
                
                # Стимулируем модель уменьшать неопределенность в областях с большой ошибкой
                l1_error = torch.abs(pred_images - target_images)
                uncertainty_guided_loss = (uncertainty_map * l1_error).mean()
                
                # Добавляем регуляризацию для предотвращения тривиальных решений (все нули)
                uncertainty_regularization = -torch.log(uncertainty_map + 1e-6).mean() * 0.1
                
                losses['uncertainty_guided'] = uncertainty_guided_loss
                losses['uncertainty_reg'] = uncertainty_regularization
        
        # Применяем веса из динамического балансировщика
        if self.loss_balancer is not None:
            weights = self.loss_balancer.get_weights()
            for loss_name in losses:
                if loss_name in weights:
                    losses[loss_name] = losses[loss_name] * weights[loss_name]
        
        # Логируем значения каждой потери через заданные интервалы
        if self.is_main_process and self.global_step % self.log_interval == 0:
            for loss_name, loss_value in losses.items():
                loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                self.writer.add_scalar(f'train/step/{loss_name}_raw', loss_value_scalar, self.global_step)
        
        return losses
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Вычисляет общую потерю на основе отдельных потерь.
        
        Args:
            losses (Dict[str, torch.Tensor]): Словарь с потерями
            
        Returns:
            torch.Tensor: Общая потеря
        """
        # Исключаем потери дискриминатора из общей потери для генератора
        excluded_losses = ['d_loss']
        
        # Суммируем все потери, кроме исключенных
        total_loss = sum(loss for name, loss in losses.items() if name not in excluded_losses)
        
        # Логируем общую потерю через заданные интервалы
        if self.is_main_process and self.global_step % self.log_interval == 0:
            self.writer.add_scalar('train/step/total_loss', total_loss.item(), self.global_step)
            
        return total_loss
    
    def _visualize_results(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """
        Визуализирует результаты колоризации.
        
        Args:
            input_images (torch.Tensor): Входные изображения
            target_images (torch.Tensor): Целевые изображения
            outputs (torch.Tensor or Dict[str, torch.Tensor]): Выходы модели
        """
        if not self.is_main_process:
            return
            
        # Извлекаем предсказанные изображения из выхода модели
        if isinstance(outputs, dict):
            pred_images = outputs.get('colorized', outputs.get('output', None))
            
            # Проверяем, есть ли в выходе отдельные каналы a и b (для пространства LAB)
            if pred_images is None and 'a' in outputs and 'b' in outputs:
                # Объединяем L-канал из входа с предсказанными a и b каналами
                a_channel = outputs['a']
                b_channel = outputs['b']
                
                # Проверяем, является ли входное изображение L-каналом
                if input_images.shape[1] == 1:
                    # Объединяем L из входа с предсказанными a и b
                    pred_images = torch.cat([input_images, a_channel, b_channel], dim=1)
                else:
                    return
        else:
            pred_images = outputs
            
        # Проверка на None
        if pred_images is None:
            return
            
        # Выбираем несколько изображений для визуализации (не более 4)
        num_images = min(4, input_images.shape[0])
        
        for i in range(num_images):
            # Извлекаем изображения
            grayscale = input_images[i].detach().cpu()
            target = target_images[i].detach().cpu()
            pred = pred_images[i].detach().cpu()
            
            # Преобразуем в numpy массивы
            grayscale_np = grayscale.squeeze().numpy() if grayscale.shape[0] == 1 else grayscale.permute(1, 2, 0).numpy()
            target_np = target.permute(1, 2, 0).numpy()
            pred_np = pred.permute(1, 2, 0).numpy()
            
            # Нормализуем, если нужно
            if grayscale_np.max() <= 1.0:
                grayscale_np = grayscale_np
            if target_np.max() <= 1.0:
                target_np = target_np
            if pred_np.max() <= 1.0:
                pred_np = pred_np
                
            # Преобразуем Lab в RGB, если нужно
            if self.color_space == 'lab':
                # Преобразуем из Lab в RGB
                from skimage.color import lab2rgb
                
                # Денормализуем Lab
                target_lab = target_np.copy()
                target_lab[:, :, 0] = target_lab[:, :, 0] * 100.0
                target_lab[:, :, 1:] = target_lab[:, :, 1:] * 127.0
                
                pred_lab = pred_np.copy()
                pred_lab[:, :, 0] = pred_lab[:, :, 0] * 100.0
                pred_lab[:, :, 1:] = pred_lab[:, :, 1:] * 127.0
                
                # Преобразуем в RGB
                target_np = lab2rgb(target_lab)
                pred_np = lab2rgb(pred_lab)
                
            # Создаем сравнение
            comparison = self.visualizer.create_comparison(
                grayscale=grayscale_np,
                colorized=pred_np,
                original=target_np,
                filename=f"epoch_{self.current_epoch + 1}_step_{self.global_step}_img_{i}.png"
            )
            
            # Логируем изображения в TensorBoard
            grayscale_tensor = torch.from_numpy(grayscale_np).unsqueeze(0) if grayscale_np.ndim == 2 else torch.from_numpy(grayscale_np.transpose(2, 0, 1)).unsqueeze(0)
            target_tensor = torch.from_numpy(target_np.transpose(2, 0, 1)).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_np.transpose(2, 0, 1)).unsqueeze(0)
            
            self.writer.add_images(f'images/sample_{i}/grayscale', grayscale_tensor, self.global_step)
            self.writer.add_images(f'images/sample_{i}/target', target_tensor, self.global_step)
            self.writer.add_images(f'images/sample_{i}/prediction', pred_tensor, self.global_step)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Создает оптимизатор на основе конфигурации.
        
        Returns:
            optim.Optimizer: Оптимизатор
        """
        # Получаем параметры модели
        model_parameters = self.model.parameters() if not self.distributed else self.model.module.parameters()
        
        # Создаем оптимизатор в зависимости от типа
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(
                model_parameters,
                lr=self.learning_rate,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model_parameters,
                lr=self.learning_rate,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model_parameters,
                lr=self.learning_rate,
                momentum=self.optimizer_momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Неподдерживаемый тип оптимизатора: {self.optimizer_type}")
            
        # Создаем оптимизаторы для дискриминатора и советника, если они включены
        if self.gan_discriminator_enabled and 'discriminator' in self.modules:
            self.optimizer_d = optim.Adam(
                self.modules['discriminator'].parameters(),
                lr=self.learning_rate * self.gan_lr_factor,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay
            )
            
        if self.guide_net_train_separately and 'guide_net' in self.modules:
            self.optimizer_g = optim.Adam(
                self.modules['guide_net'].parameters(),
                lr=self.learning_rate * self.guide_net_lr_factor,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay
            )
            
        return optimizer
    
    def _create_losses(self) -> Dict[str, Any]:
        """
        Создает лосс-функции на основе конфигурации.
        
        Returns:
            Dict[str, Any]: Словарь с лосс-функциями
        """
        losses = {}
        
        # Перцептивная потеря
        if self.perceptual_loss_enabled:
            perceptual_loss = VGGPerceptualLoss(
                layers=self.perceptual_loss_layers,
                weights=self.perceptual_loss_weights,
                criterion=self.perceptual_loss_criterion,
                resize=self.perceptual_loss_resize,
                normalize=self.perceptual_loss_normalize,
                device=self.device
            )
            losses['perceptual'] = perceptual_loss
            
        # PatchNCE потеря
        if self.patch_nce_loss_enabled:
            patch_nce_loss = PatchNCELoss(
                temperature=self.patch_nce_temperature,
                patch_size=self.patch_nce_patch_size,
                n_patches=self.patch_nce_num_patches,
                device=self.device
            )
            losses['patch_nce'] = patch_nce_loss
            
        # GAN потеря
        if self.gan_loss_enabled:
            gan_loss = GANLoss(
                gan_mode=self.gan_loss_type,
                device=self.device
            )
            losses['gan'] = gan_loss
            
        return losses
    
    def _create_modules(self) -> Dict[str, nn.Module]:
        """
        Создает интеллектуальные модули на основе конфигурации.
        
        Returns:
            Dict[str, nn.Module]: Словарь с интеллектуальными модулями
        """
        modules = {}
        
        # GuideNet
        if self.guide_net_enabled:
            guide_net = GuideNet(
                input_channels=1 if self.color_space == 'lab' else 3,
                advice_channels=2 if self.color_space == 'lab' else 3,
                feature_dim=self.guide_net_feature_dim,
                num_layers=self.guide_net_num_layers,
                use_attention=self.guide_net_use_attention,
                device=self.device
            )
            guide_net.to(self.device)
            modules['guide_net'] = guide_net
            
        # Дискриминатор
        if self.gan_discriminator_enabled:
            discriminator = Discriminator(
                input_nc=3,
                ndf=self.discriminator_ndf,
                n_layers=self.discriminator_n_layers,
                use_spectral_norm=self.discriminator_spectral_norm,
                reward_type=self.discriminator_reward_type,
                device=self.device
            )
            discriminator.to(self.device)
            modules['discriminator'] = discriminator
            
        # Банк памяти
        if self.memory_bank_enabled:
            memory_bank = MemoryBankModule(
                feature_dim=self.memory_bank_feature_dim,
                max_items=self.memory_bank_max_items,
                index_type=self.memory_bank_index_type,
                use_fusion=self.memory_bank_use_fusion,
                device=self.device
            )
            memory_bank.to(self.device)
            modules['memory_bank'] = memory_bank
            
        # Оценка неопределенности
        if self.uncertainty_estimation_enabled:
            uncertainty_estimation = UncertaintyEstimation(
                method=self.uncertainty_estimation_method,
                num_samples=self.uncertainty_estimation_num_samples,
                dropout_rate=self.uncertainty_estimation_dropout_rate,
                device=self.device
            )
            uncertainty_estimation.to(self.device)
            modules['uncertainty_estimation'] = uncertainty_estimation
            
        # Перенос стилей
        if self.style_transfer_enabled:
            style_transfer = StyleTransfer(
                content_weight=self.style_transfer_content_weight,
                style_weight=self.style_transfer_style_weight,
                content_layers=self.style_transfer_content_layers,
                style_layers=self.style_transfer_style_layers,
                device=self.device
            )
            style_transfer.to(self.device)
            modules['style_transfer'] = style_transfer
            
        # Few-shot адаптер
        if self.few_shot_adapter_enabled:
            adapter = AdaptableColorizer(
                adapter_type=self.few_shot_adapter_type,
                bottleneck_dim=self.few_shot_bottleneck_dim,
                base_model=self.model if not self.distributed else self.model.module,
                device=self.device
            )
            adapter.to(self.device)
            modules['few_shot_adapter'] = adapter
            
        return modules
    
    def _create_loss_balancer(self) -> Optional[DynamicLossBalancer]:
        """
        Создает динамический балансировщик лосса на основе конфигурации.
        
        Returns:
            Optional[DynamicLossBalancer]: Балансировщик лосса или None, если отключен
        """
        if not self.loss_balancer_enabled:
            return None
            
        from losses.dynamic_balancer import DynamicLossBalancer
        
        # Определяем начальные веса
        initial_weights = {}
        
        if self.l1_loss_enabled:
            initial_weights['l1'] = self.l1_loss_weight
            
        if self.l2_loss_enabled:
            initial_weights['l2'] = self.l2_loss_weight
            
        if self.perceptual_loss_enabled:
            initial_weights['perceptual'] = self.perceptual_loss_weight
            
        if self.patch_nce_loss_enabled:
            initial_weights['patch_nce'] = self.patch_nce_loss_weight
            
        if self.gan_loss_enabled:
            initial_weights['g_loss'] = self.gan_generator_weight
            initial_weights['d_loss'] = self.gan_discriminator_weight
            
            if self.gan_reward_enabled:
                initial_weights['reward'] = self.gan_reward_weight
                
        if self.guide_net_enabled:
            initial_weights['guide_loss'] = self.guide_net_weight
            initial_weights['guide_consistency'] = self.guide_net_consistency_weight
            
        if self.memory_bank_enabled:
            initial_weights['memory_consistency'] = self.memory_bank_consistency_weight
            
        if self.uncertainty_estimation_enabled:
            initial_weights['uncertainty_guided'] = self.uncertainty_estimation_weight
            initial_weights['uncertainty_reg'] = self.uncertainty_estimation_reg_weight
            
        # Создаем балансировщик
        balancer = DynamicLossBalancer(
            initial_weights=initial_weights,
            strategy=self.loss_balancer_strategy,
            target_metric=self.loss_balancer_target_metric,
            learning_rate=self.loss_balancer_lr
        )
        
        return balancer
    
    def _setup_logging(self) -> logging.Logger:
        """
        Настраивает логирование.
        
        Returns:
            logging.Logger: Объект логгера
        """
        logger = logging.getLogger("ColorizationTrainer")
        
        if not logger.handlers:
            # Добавляем обработчик консоли
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Добавляем обработчик файла для основного процесса
            if self.is_main_process:
                os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(self.experiment_dir, 'logs', 'training.log'))
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
        logger.setLevel(logging.INFO)
        
        # Отключаем передачу логов родительскому логгеру
        logger.propagate = False
        
        return logger
    
    def _log_configuration(self):
        """Логирует информацию о модели и конфигурации."""
        # Логируем информацию о модели
        model_info = {
            'name': type(self.model if not self.distributed else self.model.module).__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Логируем информацию о датасете
        dataset_info = {
            'train_size': len(self.train_loader.dataset),
            'val_size': len(self.val_loader.dataset) if self.val_loader is not None else 0,
            'batch_size': self.train_loader.batch_size,
            'num_batches': len(self.train_loader)
        }
        
        # Логируем информацию о конфигурации обучения
        training_info = {
            'optimizer': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'mixed_precision': self.mixed_precision,
            'distributed': self.distributed,
            'world_size': self.world_size,
            'loss_balancer_enabled': self.loss_balancer_enabled
        }
        
        # Логируем в файл
        self.logger.info(f"Модель: {model_info}")
        self.logger.info(f"Данные: {dataset_info}")
        self.logger.info(f"Обучение: {training_info}")
        
        # Сохраняем конфигурацию в JSON
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model': model_info,
                'dataset': dataset_info,
                'training': training_info,
                'config': self.config
            }, f, indent=2)
            
        self.logger.info(f"Конфигурация сохранена в {config_path}")
    
    def _parse_training_config(self):
        """Парсит параметры обучения из конфигурации."""
        # Получаем конфигурацию обучения
        training_config = self.config.get('training', {})
        
        # Основные параметры обучения
        self.max_epochs = training_config.get('epochs', 100)
        self.checkpoint_interval = training_config.get('checkpoint_interval', 5)
        self.validation_interval = training_config.get('validation_interval', 1)
        self.log_interval = training_config.get('log_interval', 100)
        self.visualization_interval = training_config.get('visualization_interval', 500)
        self.mixed_precision = training_config.get('mixed_precision', True)
        self.grad_clip = training_config.get('grad_clip', 0.0)
        self.color_space = self.config.get('data', {}).get('color_space', 'lab')
        
        # Параметры оптимизатора
        optimizer_config = training_config.get('optimizer', {})
        self.optimizer_type = optimizer_config.get('type', 'adam')
        self.learning_rate = optimizer_config.get('lr', 1e-4)
        self.optimizer_betas = optimizer_config.get('betas', (0.9, 0.999))
        self.optimizer_momentum = optimizer_config.get('momentum', 0.9)
        self.weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        # Параметры планировщика
        self.lr_scheduler_config = training_config.get('scheduler', {})
        self.scheduler_step_on = self.lr_scheduler_config.get('step_on', 'epoch')
        self.scheduler_metric = self.lr_scheduler_config.get('metric', None)
        
        # Параметры раннего останова
        early_stopping_config = training_config.get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', False)
        self.patience = early_stopping_config.get('patience', 10)
        self.early_stopping_metric = early_stopping_config.get('metric', 'lpips')
        self.early_stopping_mode = early_stopping_config.get('mode', 'min')
        
        # Параметры лоссов из конфигурации лоссов
        loss_config = self.config.get('losses', {})
        
        # L1 потеря
        l1_config = loss_config.get('l1_loss', {})
        self.l1_loss_enabled = l1_config.get('enabled', True)
        self.l1_loss_weight = l1_config.get('weight', 10.0)
        
        # L2 потеря
        l2_config = loss_config.get('l2_loss', {})
        self.l2_loss_enabled = l2_config.get('enabled', False)
        self.l2_loss_weight = l2_config.get('weight', 1.0)
        
        # Перцептивная потеря
        perceptual_config = loss_config.get('vgg_perceptual', {})
        self.perceptual_loss_enabled = perceptual_config.get('enabled', True)
        self.perceptual_loss_weight = perceptual_config.get('weight', 1.0)
        self.perceptual_loss_layers = perceptual_config.get('layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        self.perceptual_loss_weights = perceptual_config.get('layer_weights', None)
        self.perceptual_loss_criterion = perceptual_config.get('criterion', 'l1')
        self.perceptual_loss_resize = perceptual_config.get('resize', True)
        self.perceptual_loss_normalize = perceptual_config.get('normalize', True)
        
        # PatchNCE потеря
        patch_nce_config = loss_config.get('patch_nce', {})
        self.patch_nce_loss_enabled = patch_nce_config.get('enabled', True)
        self.patch_nce_loss_weight = patch_nce_config.get('weight', 1.0)
        self.patch_nce_temperature = patch_nce_config.get('temperature', 0.07)
        self.patch_nce_patch_size = patch_nce_config.get('patch_size', 16)
        self.patch_nce_num_patches = patch_nce_config.get('num_patches', 256)
        
        # GAN потеря
        gan_config = loss_config.get('gan_loss', {})
        self.gan_loss_enabled = gan_config.get('enabled', True)
        self.gan_loss_type = gan_config.get('type', 'hinge')
        self.gan_generator_weight = gan_config.get('generator_weight', 1.0)
        self.gan_discriminator_weight = gan_config.get('discriminator_weight', 0.5)
        self.gan_lr_factor = gan_config.get('lr_factor', 1.0)
        
        # GAN награда
        self.gan_reward_enabled = gan_config.get('reward_enabled', True)
        self.gan_reward_weight = gan_config.get('reward_weight', 0.1)
        
        # Параметры интеллектуальных модулей из конфигурации модулей
        modules_config = self.config.get('modules', {})
        
        # Параметры GuideNet
        guide_net_config = modules_config.get('guide_net', {})
        self.guide_net_enabled = guide_net_config.get('enabled', True)
        self.guide_net_feature_dim = guide_net_config.get('feature_dim', 512)
        self.guide_net_num_layers = guide_net_config.get('num_layers', 4)
        self.guide_net_use_attention = guide_net_config.get('use_attention', True)
        self.guide_net_train_separately = guide_net_config.get('train_separately', True)
        self.guide_net_lr_factor = guide_net_config.get('lr_factor', 1.0)
        self.guide_net_weight = guide_net_config.get('weight', 1.0)
        self.guide_net_consistency_weight = guide_net_config.get('consistency_weight', 0.5)
        
        # Параметры дискриминатора
        discriminator_config = modules_config.get('discriminator', {})
        self.gan_discriminator_enabled = discriminator_config.get('enabled', True)
        self.discriminator_ndf = discriminator_config.get('ndf', 64)
        self.discriminator_n_layers = discriminator_config.get('n_layers', 3)
        self.discriminator_spectral_norm = discriminator_config.get('use_spectral_norm', True)
        self.discriminator_reward_type = discriminator_config.get('reward_type', 'adaptive')
        
        # Параметры банка памяти
        memory_bank_config = modules_config.get('memory_bank', {})
        self.memory_bank_enabled = memory_bank_config.get('enabled', True)
        self.memory_bank_feature_dim = memory_bank_config.get('feature_dim', 512)
        self.memory_bank_max_items = memory_bank_config.get('max_items', 100000)
        self.memory_bank_index_type = memory_bank_config.get('index_type', 'flat')
        self.memory_bank_use_fusion = memory_bank_config.get('use_fusion', True)
        self.memory_bank_consistency_weight = memory_bank_config.get('consistency_weight', 0.5)
        
        # Параметры оценки неопределенности
        uncertainty_config = modules_config.get('uncertainty_estimation', {})
        self.uncertainty_estimation_enabled = uncertainty_config