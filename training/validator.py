"""
Validator: Модуль валидации модели колоризации.

Данный модуль предоставляет функциональность для валидации и оценки качества 
моделей колоризации. Он выполняет прогон модели на валидационном наборе данных,
вычисляет различные метрики качества, сохраняет визуализации результатов и 
предоставляет подробные отчеты о производительности модели.

Ключевые особенности:
- Вычисление разнообразных метрик качества колоризации (PSNR, SSIM, LPIPS и др.)
- Визуализация результатов колоризации для качественной оценки
- Поддержка распределенной валидации для ускорения на нескольких GPU
- Отслеживание прогресса валидации в реальном времени
- Создание подробных отчетов для анализа качества модели

Преимущества:
- Комплексная оценка качества колоризации с использованием различных метрик
- Возможность автоматической визуализации наиболее и наименее удачных результатов
- Эффективное использование вычислительных ресурсов через распределенную валидацию
- Детальное логирование результатов для последующего анализа
"""

import os
import time
import datetime
import json
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import autocast
import tqdm
from tensorboardX import SummaryWriter

from utils.metrics import MetricsCalculator
from utils.visualization import ColorizationVisualizer


class ColorizationValidator:
    """
    Валидатор для моделей колоризации.
    
    Args:
        model (nn.Module): Модель колоризации
        val_loader (DataLoader): Загрузчик данных для валидации
        losses (Dict): Словарь с лосс-функциями для валидации
        metrics_calculator (MetricsCalculator): Калькулятор метрик качества
        visualizer (ColorizationVisualizer, optional): Визуализатор результатов
        config (Dict): Конфигурация валидации
        device (torch.device): Устройство для вычислений
        experiment_dir (str): Директория для сохранения результатов
        distributed (bool): Включить распределенную валидацию
        rank (int): Ранг текущего процесса (для распределенной валидации)
        world_size (int): Общее количество процессов (для распределенной валидации)
    """
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        losses: Dict,
        metrics_calculator: MetricsCalculator,
        visualizer: Optional[ColorizationVisualizer] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        experiment_dir: Optional[str] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.val_loader = val_loader
        self.losses = losses
        self.metrics_calculator = metrics_calculator
        self.visualizer = visualizer
        self.config = config or {}
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
            
        # Создаем директории для результатов
        self.validation_dir = os.path.join(self.experiment_dir, 'validation')
        if self.is_main_process:
            os.makedirs(self.validation_dir, exist_ok=True)
            
        # Настраиваем логирование
        self.logger = self._setup_logging()
        
        # Настраиваем tensorboard
        self.tensorboard_dir = os.path.join(self.experiment_dir, 'tensorboard')
        if self.is_main_process:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        # Парсим конфигурацию валидации
        self._parse_validation_config()
        
        self.logger.info(f"Инициализирован ColorizationValidator на устройстве {self.device}")
        
    def validate(
        self,
        epoch: Optional[int] = None,
        visualize: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Выполняет валидацию модели.
        
        Args:
            epoch (int, optional): Текущая эпоха обучения
            visualize (bool): Визуализировать результаты колоризации
            save_results (bool): Сохранять результаты на диск
            
        Returns:
            Dict[str, Any]: Результаты валидации
        """
        # Переводим модель в режим оценки
        self.model.eval()
        
        # Инициализируем трекеры для метрик и потерь
        val_losses = {}
        val_metrics = {}
        all_metrics = []
        
        # Инициализируем трекеры для лучших и худших результатов
        best_results = []
        worst_results = []
        
        # Инициализируем счетчики
        num_samples = 0
        start_time = time.time()
        
        # Создаем прогресс-бар
        progress_bar = tqdm.tqdm(
            self.val_loader,
            desc=f"Валидация {f'Эпоха {epoch}' if epoch is not None else ''}",
            disable=not self.is_main_process
        )
        
        try:
            with torch.no_grad():
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
                    
                    # Прямой проход
                    with autocast(enabled=self.mixed_precision):
                        outputs = self.model(input_images)
                        
                        # Извлекаем предсказанные изображения из выхода модели
                        pred_images = self._extract_predictions(outputs, input_images)
                        
                        # Вычисляем потери
                        batch_losses = self._compute_losses(input_images, target_images, outputs)
                        
                        # Вычисляем метрики
                        batch_metrics = self._compute_metrics(pred_images, target_images)
                        
                    # Обновляем трекеры потерь
                    for loss_name, loss_value in batch_losses.items():
                        # Преобразуем значения в Python-скаляры
                        loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                        
                        # Добавляем или обновляем значение в словаре потерь
                        val_losses[loss_name] = val_losses.get(loss_name, 0) + loss_value_scalar * batch_size
                        
                    # Обновляем трекеры метрик
                    for metric_name, metric_value in batch_metrics.items():
                        # Преобразуем значения в Python-скаляры
                        metric_value_scalar = metric_value.item() if torch.is_tensor(metric_value) else metric_value
                        
                        # Добавляем или обновляем значение в словаре метрик
                        val_metrics[metric_name] = val_metrics.get(metric_name, 0) + metric_value_scalar * batch_size
                        
                    # Сохраняем метрики для каждого изображения
                    for i in range(batch_size):
                        # Создаем словарь с метриками для текущего изображения
                        image_metrics = {}
                        for metric_name, metric_value in batch_metrics.items():
                            if isinstance(metric_value, (list, tuple)):
                                # Если метрика возвращает значения для каждого изображения
                                image_metrics[metric_name] = metric_value[i]
                            else:
                                # Если метрика возвращает скаляр, используем его для всех изображений
                                image_metrics[metric_name] = metric_value
                                
                        all_metrics.append(image_metrics)
                        
                    # Обновляем трекер лучших и худших результатов
                    self._update_best_worst(
                        input_images, target_images, pred_images, 
                        batch_metrics, best_results, worst_results
                    )
                        
                    # Обновляем счетчик образцов
                    num_samples += batch_size
                    
                    # Обновляем прогресс-бар с текущими метриками
                    if self.is_main_process:
                        postfix_dict = {
                            **{f"{k}": f"{v/num_samples:.4f}" for k, v in val_losses.items()},
                            **{f"{k}": f"{v/num_samples:.4f}" for k, v in val_metrics.items()}
                        }
                        progress_bar.set_postfix(postfix_dict)
                        
            # Закрываем прогресс-бар
            progress_bar.close()
            
            # Вычисляем средние значения потерь и метрик
            for loss_name in val_losses:
                val_losses[loss_name] /= max(1, num_samples)
                
            for metric_name in val_metrics:
                val_metrics[metric_name] /= max(1, num_samples)
                
            # Вычисляем статистику по метрикам
            metrics_stats = self._compute_metrics_stats(all_metrics)
            
            # Визуализируем результаты, если нужно
            if visualize and self.visualizer is not None and self.is_main_process:
                self._visualize_validation_results(epoch, best_results, worst_results)
                
            # Сохраняем результаты, если нужно
            if save_results and self.is_main_process:
                self._save_validation_results(epoch, val_losses, val_metrics, metrics_stats)
                
            # Логируем результаты
            if self.is_main_process:
                self.logger.info(f"Валидация завершена. Потери: {val_losses}, Метрики: {val_metrics}")
                
            # Формируем результат
            total_time = time.time() - start_time
            results = {
                "losses": val_losses,
                "metrics": val_metrics,
                "metrics_stats": metrics_stats,
                "num_samples": num_samples,
                "time": total_time,
                "samples_per_second": num_samples / max(0.001, total_time),
                "best_results": best_results,
                "worst_results": worst_results
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _extract_predictions(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Извлекает предсказанные изображения из выхода модели.
        
        Args:
            outputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Выход модели
            input_images (torch.Tensor): Входные изображения
            
        Returns:
            torch.Tensor: Предсказанные изображения
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
            
        return pred_images
        
    def _compute_losses(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Вычисляет потери для текущего пакета.
        
        Args:
            input_images (torch.Tensor): Входные изображения
            target_images (torch.Tensor): Целевые изображения
            outputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): Выход модели
            
        Returns:
            Dict[str, float]: Словарь с потерями
        """
        # Извлекаем предсказанные изображения из выхода модели
        pred_images = self._extract_predictions(outputs, input_images)
            
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
            
        return losses
        
    def _compute_metrics(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Вычисляет метрики качества для текущего пакета.
        
        Args:
            pred_images (torch.Tensor): Предсказанные изображения
            target_images (torch.Tensor): Целевые изображения
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        # Вычисляем метрики с помощью калькулятора метрик
        metrics = self.metrics_calculator.calculate(pred_images, target_images)
        
        return metrics
        
    def _update_best_worst(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        pred_images: torch.Tensor,
        batch_metrics: Dict[str, float],
        best_results: List[Dict],
        worst_results: List[Dict]
    ):
        """
        Обновляет трекеры лучших и худших результатов.
        
        Args:
            input_images (torch.Tensor): Входные изображения
            target_images (torch.Tensor): Целевые изображения
            pred_images (torch.Tensor): Предсказанные изображения
            batch_metrics (Dict[str, float]): Метрики для текущего пакета
            best_results (List[Dict]): Список для хранения лучших результатов
            worst_results (List[Dict]): Список для хранения худших результатов
        """
        # Проверяем, что визуализатор доступен и это основной процесс
        if self.visualizer is None or not self.is_main_process:
            return
            
        # Получаем метрику для ранжирования
        if self.ranking_metric not in batch_metrics:
            return
            
        # Извлекаем значения метрики для каждого изображения
        metric_values = batch_metrics[self.ranking_metric]
        if not isinstance(metric_values, (list, tuple)):
            # Если метрика вернула скаляр, пропускаем обновление
            return
            
        batch_size = len(metric_values)
        
        # Ранжируем изображения по метрике
        if self.ranking_mode == 'min':
            # Чем меньше значение метрики, тем лучше (например, LPIPS)
            indices = np.argsort(metric_values)
        else:
            # Чем больше значение метрики, тем лучше (например, SSIM)
            indices = np.argsort([-v for v in metric_values])
            
        # Отбираем лучшие и худшие изображения из текущего пакета
        for i in range(min(self.num_samples_to_track, batch_size)):
            # Лучшие
            best_idx = indices[i]
            best_results.append({
                'input': input_images[best_idx].detach().cpu(),
                'target': target_images[best_idx].detach().cpu(),
                'pred': pred_images[best_idx].detach().cpu(),
                'metrics': {k: v[best_idx] if isinstance(v, (list, tuple)) else v for k, v in batch_metrics.items()}
            })
            
            # Худшие
            worst_idx = indices[-i-1]
            worst_results.append({
                'input': input_images[worst_idx].detach().cpu(),
                'target': target_images[worst_idx].detach().cpu(),
                'pred': pred_images[worst_idx].detach().cpu(),
                'metrics': {k: v[worst_idx] if isinstance(v, (list, tuple)) else v for k, v in batch_metrics.items()}
            })
            
        # Оставляем только указанное количество лучших и худших результатов
        best_results.sort(key=lambda x: x['metrics'][self.ranking_metric] if self.ranking_mode == 'max' 
                         else -x['metrics'][self.ranking_metric])
        best_results[:] = best_results[:self.num_samples_to_track]
        
        worst_results.sort(key=lambda x: -x['metrics'][self.ranking_metric] if self.ranking_mode == 'max' 
                          else x['metrics'][self.ranking_metric])
        worst_results[:] = worst_results[:self.num_samples_to_track]
        
    def _compute_metrics_stats(self, all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Вычисляет статистику по метрикам.
        
        Args:
            all_metrics (List[Dict[str, float]]): Список словарей с метриками для каждого изображения
            
        Returns:
            Dict[str, Dict[str, float]]: Словарь со статистикой по метрикам
        """
        stats = {}
        
        # Проверяем, что список не пуст
        if not all_metrics:
            return stats
            
        # Для каждой метрики вычисляем статистику
        for metric_name in all_metrics[0]:
            # Собираем значения метрики для всех изображений
            values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
            
            if not values:
                continue
                
            # Вычисляем статистику
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
            
        return stats
        
    def _visualize_validation_results(
        self,
        epoch: Optional[int],
        best_results: List[Dict],
        worst_results: List[Dict]
    ):
        """
        Визуализирует результаты валидации.
        
        Args:
            epoch (int, optional): Текущая эпоха обучения
            best_results (List[Dict]): Список лучших результатов
            worst_results (List[Dict]): Список худших результатов
        """
        # Проверяем, что визуализатор доступен и это основной процесс
        if self.visualizer is None or not self.is_main_process:
            return
            
        # Создаем директорию для визуализаций текущей эпохи
        epoch_dir = os.path.join(self.validation_dir, f"epoch_{epoch}" if epoch is not None else "validation")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Визуализируем лучшие результаты
        for i, result in enumerate(best_results):
            # Извлекаем изображения
            grayscale = result['input']
            target = result['target']
            pred = result['pred']
            
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
                filename=f"best_{i + 1}.png",
                output_dir=epoch_dir
            )
            
            # Логируем метрики для этого результата
            with open(os.path.join(epoch_dir, f"best_{i + 1}_metrics.json"), 'w') as f:
                json.dump(result['metrics'], f, indent=2)
                
        # Визуализируем худшие результаты
        for i, result in enumerate(worst_results):
            # Извлекаем изображения
            grayscale = result['input']
            target = result['target']
            pred = result['pred']
            
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
                filename=f"worst_{i + 1}.png",
                output_dir=epoch_dir
            )
            
            # Логируем метрики для этого результата
            with open(os.path.join(epoch_dir, f"worst_{i + 1}_metrics.json"), 'w') as f:
                json.dump(result['metrics'], f, indent=2)
                
        # Создаем сводную визуализацию
        self.visualizer.create_grid(
            best_results=best_results[:4],
            worst_results=worst_results[:4],
            color_space=self.color_space,
            filename=f"summary.png",
            output_dir=epoch_dir
        )
        
    def _save_validation_results(
        self,
        epoch: Optional[int],
        losses: Dict[str, float],
        metrics: Dict[str, float],
        metrics_stats: Dict[str, Dict[str, float]]
    ):
        """
        Сохраняет результаты валидации.
        
        Args:
            epoch (int, optional): Текущая эпоха обучения
            losses (Dict[str, float]): Словарь с потерями
            metrics (Dict[str, float]): Словарь с метриками
            metrics_stats (Dict[str, Dict[str, float]]): Словарь со статистикой по метрикам
        """
        # Проверяем, что это основной процесс
        if not self.is_main_process:
            return
            
        # Создаем директорию для результатов текущей эпохи
        epoch_dir = os.path.join(self.validation_dir, f"epoch_{epoch}" if epoch is not None else "validation")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Сохраняем потери
        with open(os.path.join(epoch_dir, "losses.json"), 'w') as f:
            json.dump(losses, f, indent=2)
            
        # Сохраняем метрики
        with open(os.path.join(epoch_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Сохраняем статистику по метрикам
        with open(os.path.join(epoch_dir, "metrics_stats.json"), 'w') as f:
            json.dump(metrics_stats, f, indent=2)
            
        # Логируем потери и метрики в tensorboard
        if hasattr(self, 'writer'):
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f'val/{loss_name}', loss_value, epoch)
                
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
                
    def _setup_logging(self) -> logging.Logger:
        """
        Настраивает логирование.
        
        Returns:
            logging.Logger: Объект логгера
        """
        logger = logging.getLogger("ColorizationValidator")
        
        if not logger.handlers:
            # Добавляем обработчик консоли
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Добавляем обработчик файла для основного процесса
            if self.is_main_process:
                os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(self.experiment_dir, 'logs', 'validation.log'))
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
        logger.setLevel(logging.INFO)
        
        # Отключаем передачу логов родительскому логгеру
        logger.propagate = False
        
        return logger
        
    def _parse_validation_config(self):
        """Парсит параметры валидации из конфигурации."""
        # Получаем конфигурацию обучения и данных
        training_config = self.config.get('training', {})
        data_config = self.config.get('data', {})
        
        # Основные параметры валидации
        self.mixed_precision = training_config.get('mixed_precision', True)
        self.color_space = data_config.get('color_space', 'lab')
        
        # Параметры отслеживания лучших и худших результатов
        validation_config = self.config.get('validation', {})
        self.ranking_metric = validation_config.get('ranking_metric', 'lpips')
        self.ranking_mode = validation_config.get('ranking_mode', 'min')
        self.num_samples_to_track = validation_config.get('num_samples_to_track', 10)
        
        # Параметры лосс-функций
        loss_config = self.config.get('losses', {})
        
        # L1 потеря
        l1_config = loss_config.get('l1_loss', {})
        self.l1_loss_enabled = l1_config.get('enabled', True)
        
        # L2 потеря
        l2_config = loss_config.get('l2_loss', {})
        self.l2_loss_enabled = l2_config.get('enabled', False)
        
        # Перцептивная потеря
        perceptual_config = loss_config.get('vgg_perceptual', {})
        self.perceptual_loss_enabled = perceptual_config.get('enabled', True)
        
        # PatchNCE потеря
        patch_nce_config = loss_config.get('patch_nce', {})
        self.patch_nce_loss_enabled = patch_nce_config.get('enabled', True)


def create_validator(
    model: nn.Module,
    val_loader: DataLoader,
    losses: Dict,
    metrics_calculator: MetricsCalculator,
    config_path: Optional[str] = None,
    visualizer: Optional[ColorizationVisualizer] = None,
    device: Optional[torch.device] = None,
    experiment_dir: Optional[str] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> ColorizationValidator:
    """
    Создает валидатор на основе параметров.
    
    Args:
        model (nn.Module): Модель колоризации
        val_loader (DataLoader): Загрузчик данных для валидации
        losses (Dict): Словарь с лосс-функциями для валидации
        metrics_calculator (MetricsCalculator): Калькулятор метрик качества
        config_path (str, optional): Путь к файлу конфигурации
        visualizer (ColorizationVisualizer, optional): Визуализатор результатов
        device (torch.device, optional): Устройство для вычислений
        experiment_dir (str, optional): Директория для сохранения результатов
        distributed (bool): Включить распределенную валидацию
        rank (int): Ранг текущего процесса (для распределенной валидации)
        world_size (int): Общее количество процессов (для распределенной валидации)
        
    Returns:
        ColorizationValidator: Созданный валидатор
    """
    # Загружаем конфигурацию, если указан путь
    config = None
    if config_path:
        from utils.config_parser import load_config
        config = load_config(config_path)
        
    # Определяем устройство
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Создаем визуализатор, если не предоставлен и это основной процесс
    if visualizer is None and rank == 0:
        from utils.visualization import ColorizationVisualizer
        visualizer = ColorizationVisualizer(experiment_dir)
        
    # Создаем валидатор
    validator = ColorizationValidator(
        model=model,
        val_loader=val_loader,
        losses=losses,
        metrics_calculator=metrics_calculator,
        visualizer=visualizer,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    return validator


if __name__ == "__main__":
    import argparse
    import torch.utils.data
    
    parser = argparse.ArgumentParser(description="TintoraAI Validator")
    parser.add_argument("--model", type=str, required=True, help="Путь к модели")
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигурации")
    parser.add_argument("--data", type=str, required=True, help="Путь к данным для валидации")
    parser.add_argument("--output", type=str, help="Директория для сохранения результатов")
    parser.add_argument("--device", type=str, help="Устройство для вычислений (cuda, cpu)")
    parser.add_argument("--distributed", action="store_true", help="Включить распределенную валидацию")
    parser.add_argument("--world-size", type=int, default=1, help="Общее количество процессов")
    parser.add_argument("--rank", type=int, default=0, help="Ранг текущего процесса")
    
    args = parser.parse_args()
    
    try:
        # Определяем устройство
        device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем конфигурацию
        from utils.config_parser import load_config
        config = load_config(args.config)
        
        # Загружаем модель
        from training.checkpoints import load_checkpoint
        model = load_checkpoint(args.model, device)['model']
        
        # Создаем загрузчик данных
        from utils.data_loader import create_datamodule
        datamodule = create_datamodule(
            data_config=config.get('data', {}),
            batch_size=config.get('training', {}).get('batch_size', 16),
            num_workers=config.get('training', {}).get('num_workers', 4),
            distributed=args.distributed,
            rank=args.rank,
            world_size=args.world_size,
            is_train=False
        )
        val_loader = datamodule.val_dataloader()
        
        # Создаем лосс-функции
        losses = {}
        loss_config = config.get('losses', {})
        
        if loss_config.get('vgg_perceptual', {}).get('enabled', True):
            from losses.vgg_perceptual import VGGPerceptualLoss
            perceptual_config = loss_config.get('vgg_perceptual', {})
            losses['perceptual'] = VGGPerceptualLoss(
                layers=perceptual_config.get('layers', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']),
                weights=perceptual_config.get('layer_weights', None),
                criterion=perceptual_config.get('criterion', 'l1'),
                resize=perceptual_config.get('resize', True),
                normalize=perceptual_config.get('normalize', True),
                device=device
            )
            
        if loss_config.get('patch_nce', {}).get('enabled', True):
            from losses.patch_nce import PatchNCELoss
            patch_nce_config = loss_config.get('patch_nce', {})
            losses['patch_nce'] = PatchNCELoss(
                temperature=patch_nce_config.get('temperature', 0.07),
                patch_size=patch_nce_config.get('patch_size', 16),
                n_patches=patch_nce_config.get('num_patches', 256),
                device=device
            )
            
        # Создаем калькулятор метрик
        from utils.metrics import MetricsCalculator
        metrics_calculator = MetricsCalculator(
            metrics=config.get('metrics', {}).get('metrics', ['psnr', 'ssim', 'lpips']),
            device=device
        )
        
        # Создаем визуализатор
        from utils.visualization import ColorizationVisualizer
        visualizer = ColorizationVisualizer(args.output) if args.rank == 0 else None
        
        # Создаем валидатор
        validator = create_validator(
            model=model,
            val_loader=val_loader,
            losses=losses,
            metrics_calculator=metrics_calculator,
            config_path=args.config,
            visualizer=visualizer,
            device=device,
            experiment_dir=args.output,
            distributed=args.distributed,
            rank=args.rank,
            world_size=args.world_size
        )
        
        # Выполняем валидацию
        results = validator.validate(save_results=True)
        
        # Выводим результаты
        if args.rank == 0:
            print("Валидация завершена.")
            print(f"Метрики: {results['metrics']}")
            
    except Exception as e:
        print(f"Ошибка при валидации: {str(e)}")
        traceback.print_exc()