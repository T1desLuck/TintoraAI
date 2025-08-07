"""
Checkpoints: Модуль для сохранения и загрузки чекпоинтов моделей колоризации.

Данный модуль предоставляет функциональность для эффективного сохранения и загрузки
состояний модели колоризации, оптимизатора и других компонентов процесса обучения.
Он поддерживает различные стратегии сохранения чекпоинтов и позволяет восстанавливать
обучение с последней сохраненной точки.

Ключевые особенности:
- Сохранение полного состояния модели, оптимизатора и планировщика
- Хранение дополнительных метаданных и метрик для анализа
- Поддержка различных стратегий сохранения чекпоинтов
- Функции для управления жизненным циклом чекпоинтов
- Механизмы обеспечения согласованности данных при сбоях

Преимущества:
- Надежное восстановление обучения после прерывания
- Эффективное использование дискового пространства
- Возможность выбора лучшей модели на основе метрик
- Гибкая настройка через конфигурационные файлы
"""

import os
import time
import json
import glob
import logging
import shutil
from typing import Dict, List, Union, Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    path: str = "checkpoint.pth",
    modules: Optional[Dict[str, nn.Module]] = None,
    loss_balancer: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    **kwargs
) -> str:
    """
    Сохраняет состояние модели и других компонентов в чекпоинт.
    
    Args:
        model (nn.Module): Модель для сохранения
        optimizer (Optimizer, optional): Оптимизатор
        scheduler (_LRScheduler, optional): Планировщик скорости обучения
        epoch (int, optional): Текущая эпоха
        global_step (int, optional): Глобальный шаг обучения
        metrics (Dict[str, float], optional): Метрики производительности
        config (Dict, optional): Конфигурация модели и обучения
        path (str): Путь для сохранения чекпоинта
        modules (Dict[str, nn.Module], optional): Дополнительные модули для сохранения
        loss_balancer (Any, optional): Балансировщик лосса
        scaler (GradScaler, optional): Скейлер для смешанной точности
        **kwargs: Дополнительные данные для сохранения
        
    Returns:
        str: Путь к сохраненному чекпоинту
    """
    # Создаем директорию для чекпоинта, если она не существует
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    # Создаем словарь с состояниями компонентов
    checkpoint = {}
    
    # Сохраняем состояние модели
    checkpoint['model_state_dict'] = model.state_dict()
    
    # Сохраняем состояние оптимизатора, если есть
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
    # Сохраняем состояние планировщика, если есть
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    # Сохраняем состояния дополнительных модулей, если есть
    if modules is not None:
        modules_state_dict = {}
        for name, module in modules.items():
            if module is not None:
                modules_state_dict[name] = module.state_dict()
        checkpoint['modules_state_dict'] = modules_state_dict
        
    # Сохраняем состояние балансировщика лосса, если есть
    if loss_balancer is not None:
        if hasattr(loss_balancer, 'state_dict'):
            checkpoint['loss_balancer_state_dict'] = loss_balancer.state_dict()
        else:
            checkpoint['loss_balancer'] = loss_balancer
            
    # Сохраняем состояние скейлера смешанной точности, если есть
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
        
    # Сохраняем дополнительные данные
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if global_step is not None:
        checkpoint['global_step'] = global_step
    if metrics is not None:
        checkpoint['metrics'] = metrics
    if config is not None:
        checkpoint['config'] = config
        
    # Сохраняем время создания чекпоинта
    checkpoint['timestamp'] = time.time()
    
    # Сохраняем дополнительные аргументы
    for key, value in kwargs.items():
        checkpoint[key] = value
        
    # Создаем временный файл для предотвращения потери данных при сбоях
    temp_path = path + '.tmp'
    
    try:
        # Сохраняем чекпоинт во временный файл
        torch.save(checkpoint, temp_path)
        
        # Заменяем старый файл новым
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)
        
        return path
        
    except Exception as e:
        # В случае ошибки удаляем временный файл и прокидываем исключение
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    modules: Optional[Dict[str, nn.Module]] = None,
    loss_balancer: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Загружает состояние модели и других компонентов из чекпоинта.
    
    Args:
        path (str): Путь к чекпоинту
        device (torch.device, optional): Устройство для загрузки
        model (nn.Module, optional): Модель для загрузки состояния
        optimizer (Optimizer, optional): Оптимизатор для загрузки состояния
        scheduler (_LRScheduler, optional): Планировщик для загрузки состояния
        modules (Dict[str, nn.Module], optional): Дополнительные модули для загрузки состояния
        loss_balancer (Any, optional): Балансировщик лосса для загрузки состояния
        scaler (GradScaler, optional): Скейлер для загрузки состояния
        strict (bool): Строгая проверка соответствия ключей при загрузке состояния модели
        
    Returns:
        Dict[str, Any]: Загруженный чекпоинт
    """
    # Определяем устройство для загрузки
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Загружаем чекпоинт
    checkpoint = torch.load(path, map_location=device)
    
    # Загружаем состояние модели, если модель предоставлена
    if model is not None and 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except Exception as e:
            print(f"Ошибка загрузки состояния модели: {e}")
            
            if not strict:
                # Если не строгая загрузка, выводим информацию о несоответствии ключей
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(checkpoint['model_state_dict'].keys())
                
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                if missing_keys:
                    print(f"Отсутствующие ключи: {missing_keys}")
                if unexpected_keys:
                    print(f"Неожиданные ключи: {unexpected_keys}")
                    
                # Загружаем только соответствующие ключи
                matching_state_dict = {
                    k: v for k, v in checkpoint['model_state_dict'].items() if k in model_keys
                }
                model.load_state_dict(matching_state_dict, strict=False)
                print("Загружены только совпадающие ключи")
            else:
                raise e
    
    # Загружаем состояние оптимизатора, если предоставлен
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Ошибка загрузки состояния оптимизатора: {e}")
            
    # Загружаем состояние планировщика, если предоставлен
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Ошибка загрузки состояния планировщика: {e}")
            
    # Загружаем состояние дополнительных модулей, если предоставлены
    if modules is not None and 'modules_state_dict' in checkpoint:
        modules_state_dict = checkpoint['modules_state_dict']
        
        for name, module in modules.items():
            if name in modules_state_dict and module is not None:
                try:
                    module.load_state_dict(modules_state_dict[name])
                except Exception as e:
                    print(f"Ошибка загрузки состояния модуля {name}: {e}")
                    
    # Загружаем состояние балансировщика лосса, если предоставлен
    if loss_balancer is not None:
        if 'loss_balancer_state_dict' in checkpoint:
            try:
                if hasattr(loss_balancer, 'load_state_dict'):
                    loss_balancer.load_state_dict(checkpoint['loss_balancer_state_dict'])
            except Exception as e:
                print(f"Ошибка загрузки состояния балансировщика лосса: {e}")
        elif 'loss_balancer' in checkpoint:
            loss_balancer = checkpoint['loss_balancer']
                
    # Загружаем состояние скейлера, если предоставлен
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"Ошибка загрузки состояния скейлера: {e}")
            
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "checkpoint_epoch_*.pth") -> Optional[str]:
    """
    Находит последний чекпоинт в директории.
    
    Args:
        checkpoint_dir (str): Директория с чекпоинтами
        pattern (str): Шаблон имени файла чекпоинта
        
    Returns:
        Optional[str]: Путь к последнему чекпоинту или None, если чекпоинты не найдены
    """
    # Получаем список всех файлов, соответствующих шаблону
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
        
    # Сортируем файлы по времени модификации
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Возвращаем последний файл
    return checkpoint_files[-1]


def find_best_checkpoint(checkpoint_dir: str, metric: str = "val_loss", mode: str = "min") -> Optional[str]:
    """
    Находит лучший чекпоинт в директории на основе метрики.
    
    Args:
        checkpoint_dir (str): Директория с чекпоинтами
        metric (str): Имя метрики для сравнения
        mode (str): Режим сравнения ('min' - меньше лучше, 'max' - больше лучше)
        
    Returns:
        Optional[str]: Путь к лучшему чекпоинту или None, если чекпоинты не найдены
    """
    # Получаем список всех файлов чекпоинтов
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        return None
        
    best_checkpoint = None
    best_metric_value = float('inf') if mode == 'min' else float('-inf')
    
    # Перебираем все чекпоинты и находим лучший
    for file_path in checkpoint_files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # Проверяем, содержит ли чекпоинт метрики
            if 'metrics' not in checkpoint:
                continue
                
            # Извлекаем значение метрики
            metrics = checkpoint['metrics']
            
            if metric in metrics:
                metric_value = metrics[metric]
                
                if (mode == 'min' and metric_value < best_metric_value) or \
                   (mode == 'max' and metric_value > best_metric_value):
                    best_metric_value = metric_value
                    best_checkpoint = file_path
                    
        except Exception as e:
            print(f"Ошибка при загрузке чекпоинта {file_path}: {e}")
            
    return best_checkpoint


def clean_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_epoch_step: int = 10,
    keep_best: bool = True,
    best_metric: str = "val_loss",
    best_mode: str = "min"
) -> None:
    """
    Очищает директорию с чекпоинтами, удаляя старые и ненужные.
    
    Args:
        checkpoint_dir (str): Директория с чекпоинтами
        keep_last_n (int): Количество последних чекпоинтов для сохранения
        keep_epoch_step (int): Шаг эпох для сохранения чекпоинтов (например, каждая 10-я эпоха)
        keep_best (bool): Сохранять ли лучший чекпоинт по метрике
        best_metric (str): Имя метрики для определения лучшего чекпоинта
        best_mode (str): Режим сравнения для лучшего чекпоинта
    """
    # Получаем список всех файлов чекпоинтов
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        return
        
    # Сортируем чекпоинты по времени модификации
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Если количество чекпоинтов меньше или равно keep_last_n, ничего не делаем
    if len(checkpoint_files) <= keep_last_n:
        return
        
    # Находим лучший чекпоинт, если нужно
    best_checkpoint = None
    if keep_best:
        best_checkpoint = find_best_checkpoint(checkpoint_dir, best_metric, best_mode)
        
    # Словарь для хранения чекпоинтов с их эпохами
    checkpoint_epochs = {}
    
    # Извлекаем эпохи из имен файлов
    for file_path in checkpoint_files:
        try:
            # Извлекаем номер эпохи из имени файла
            file_name = os.path.basename(file_path)
            epoch_str = file_name.replace("checkpoint_epoch_", "").replace(".pth", "")
            epoch = int(epoch_str)
            
            checkpoint_epochs[file_path] = epoch
        except Exception:
            # Если не удается извлечь номер эпохи, пропускаем файл
            continue
            
    # Формируем список чекпоинтов для сохранения
    checkpoints_to_keep = set()
    
    # Добавляем последние N чекпоинтов
    checkpoints_to_keep.update(checkpoint_files[-keep_last_n:])
    
    # Добавляем чекпоинты с шагом keep_epoch_step
    for file_path, epoch in checkpoint_epochs.items():
        if epoch % keep_epoch_step == 0:
            checkpoints_to_keep.add(file_path)
            
    # Добавляем лучший чекпоинт, если есть
    if best_checkpoint:
        checkpoints_to_keep.add(best_checkpoint)
        
    # Удаляем остальные чекпоинты
    for file_path in checkpoint_files:
        if file_path not in checkpoints_to_keep:
            try:
                os.remove(file_path)
                print(f"Удален чекпоинт: {file_path}")
            except Exception as e:
                print(f"Ошибка при удалении чекпоинта {file_path}: {e}")


def extract_model_from_checkpoint(
    checkpoint_path: str,
    model_class: nn.Module,
    model_args: Dict = {},
    device: Optional[torch.device] = None,
    strict: bool = False,
    output_path: Optional[str] = None
) -> nn.Module:
    """
    Извлекает и создает модель из чекпоинта.
    
    Args:
        checkpoint_path (str): Путь к чекпоинту
        model_class (nn.Module): Класс модели
        model_args (Dict): Аргументы для конструктора модели
        device (torch.device, optional): Устройство для загрузки модели
        strict (bool): Строгая проверка соответствия ключей при загрузке состояния модели
        output_path (str, optional): Путь для сохранения извлеченной модели
        
    Returns:
        nn.Module: Извлеченная модель
    """
    # Определяем устройство для загрузки
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Извлекаем конфигурацию модели, если есть
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Объединяем конфигурацию с переданными аргументами
    model_args = {**model_config, **model_args}
    
    # Создаем экземпляр модели
    model = model_class(**model_args)
    
    # Загружаем состояние модели
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except Exception as e:
            print(f"Ошибка загрузки состояния модели: {e}")
            
            if not strict:
                # Если не строгая загрузка, выводим информацию о несоответствии ключей
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(checkpoint['model_state_dict'].keys())
                
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                if missing_keys:
                    print(f"Отсутствующие ключи: {missing_keys}")
                if unexpected_keys:
                    print(f"Неожиданные ключи: {unexpected_keys}")
                    
                # Загружаем только соответствующие ключи
                matching_state_dict = {
                    k: v for k, v in checkpoint['model_state_dict'].items() if k in model_keys
                }
                model.load_state_dict(matching_state_dict, strict=False)
                print("Загружены только совпадающие ключи")
            else:
                raise e
    else:
        raise ValueError(f"В чекпоинте {checkpoint_path} отсутствует состояние модели")
        
    # Переносим модель на устройство
    model.to(device)
    
    # Сохраняем извлеченную модель, если указан путь
    if output_path:
        torch.save({'model': model, 'config': model_config}, output_path)
        print(f"Модель сохранена в {output_path}")
        
    return model


def convert_checkpoint_to_model(
    checkpoint_path: str,
    output_path: str,
    include_config: bool = True,
    include_metrics: bool = True,
    include_timestamp: bool = True
) -> None:
    """
    Конвертирует чекпоинт в файл модели.
    
    Args:
        checkpoint_path (str): Путь к чекпоинту
        output_path (str): Путь для сохранения файла модели
        include_config (bool): Включать ли конфигурацию
        include_metrics (bool): Включать ли метрики
        include_timestamp (bool): Включать ли временную метку
    """
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Создаем словарь для файла модели
    model_dict = {}
    
    # Добавляем состояние модели
    if 'model_state_dict' in checkpoint:
        model_dict['state_dict'] = checkpoint['model_state_dict']
        
    # Добавляем конфигурацию, если нужно
    if include_config and 'config' in checkpoint:
        model_dict['config'] = checkpoint['config']
        
    # Добавляем метрики, если нужно
    if include_metrics and 'metrics' in checkpoint:
        model_dict['metrics'] = checkpoint['metrics']
        
    # Добавляем эпоху и глобальный шаг, если есть
    if 'epoch' in checkpoint:
        model_dict['epoch'] = checkpoint['epoch']
    if 'global_step' in checkpoint:
        model_dict['global_step'] = checkpoint['global_step']
        
    # Добавляем временную метку, если нужно
    if include_timestamp:
        if 'timestamp' in checkpoint:
            model_dict['timestamp'] = checkpoint['timestamp']
        else:
            model_dict['timestamp'] = time.time()
            
    # Сохраняем файл модели
    torch.save(model_dict, output_path)
    print(f"Файл модели сохранен в {output_path}")


def checkpoint_exists(path: str) -> bool:
    """
    Проверяет наличие чекпоинта по указанному пути.
    
    Args:
        path (str): Путь к чекпоинту
        
    Returns:
        bool: True, если чекпоинт существует, иначе False
    """
    return os.path.exists(path) and os.path.isfile(path)


def combine_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    model_keys: Optional[List[str]] = None,
    weights: Optional[List[float]] = None
) -> None:
    """
    Объединяет несколько чекпоинтов в один с усреднением весов.
    
    Args:
        checkpoint_paths (List[str]): Пути к чекпоинтам
        output_path (str): Путь для сохранения объединенного чекпоинта
        model_keys (List[str], optional): Ключи весов модели для объединения
        weights (List[float], optional): Веса для каждого чекпоинта
    """
    if len(checkpoint_paths) == 0:
        raise ValueError("Список путей чекпоинтов пуст")
        
    # Нормализуем веса, если они предоставлены
    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    else:
        if len(weights) != len(checkpoint_paths):
            raise ValueError("Количество весов должно совпадать с количеством чекпоинтов")
        
        # Нормализуем веса, чтобы их сумма была равна 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    # Загружаем первый чекпоинт
    combined_checkpoint = torch.load(checkpoint_paths[0], map_location='cpu')
    
    # Получаем состояние модели
    combined_state_dict = combined_checkpoint['model_state_dict']
    
    # Фильтруем ключи, если они указаны
    if model_keys is not None:
        combined_state_dict = {k: v for k, v in combined_state_dict.items() if k in model_keys}
        
    # Умножаем веса на первый вес
    for key in combined_state_dict:
        combined_state_dict[key] = combined_state_dict[key] * weights[0]
        
    # Добавляем остальные чекпоинты
    for i, path in enumerate(checkpoint_paths[1:], 1):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Фильтруем ключи, если они указаны
        if model_keys is not None:
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            
        # Добавляем веса с учетом весового коэффициента
        for key in combined_state_dict:
            if key in state_dict:
                combined_state_dict[key] += state_dict[key] * weights[i]
                
    # Обновляем состояние модели в объединенном чекпоинте
    combined_checkpoint['model_state_dict'] = combined_state_dict
    
    # Обновляем метаданные
    combined_checkpoint['checkpoint_type'] = 'combined'
    combined_checkpoint['source_checkpoints'] = checkpoint_paths
    combined_checkpoint['weights'] = weights
    combined_checkpoint['timestamp'] = time.time()
    
    # Сохраняем объединенный чекпоинт
    torch.save(combined_checkpoint, output_path)
    print(f"Объединенный чекпоинт сохранен в {output_path}")


class CheckpointManager:
    """
    Менеджер для управления чекпоинтами во время обучения.
    
    Args:
        checkpoint_dir (str): Директория для сохранения чекпоинтов
        max_checkpoints (int): Максимальное количество сохраняемых чекпоинтов
        save_best (bool): Сохранять ли отдельно лучший чекпоинт
        metric_name (str): Имя метрики для определения лучшего чекпоинта
        mode (str): Режим сравнения для метрики ('min' или 'max')
    """
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        
        # Создаем директорию для чекпоинтов
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Список сохраненных чекпоинтов
        self.checkpoints = []
        
        # Лучшее значение метрики
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Путь к лучшему чекпоинту
        self.best_checkpoint_path = None
        
        # Загружаем информацию о существующих чекпоинтах
        self._load_checkpoints_info()
        
    def _load_checkpoints_info(self) -> None:
        """Загружает информацию о существующих чекпоинтах."""
        # Получаем список файлов чекпоинтов
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth"))
        
        # Добавляем их в список чекпоинтов
        for file_path in checkpoint_files:
            self.checkpoints.append({
                'path': file_path,
                'timestamp': os.path.getmtime(file_path)
            })
            
        # Сортируем по времени создания
        self.checkpoints.sort(key=lambda x: x['timestamp'])
        
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Сохраняет чекпоинт и управляет количеством сохраненных чекпоинтов.
        
        Args:
            model (nn.Module): Модель для сохранения
            optimizer (Optimizer, optional): Оптимизатор
            scheduler (_LRScheduler, optional): Планировщик скорости обучения
            epoch (int, optional): Текущая эпоха
            global_step (int, optional): Глобальный шаг обучения
            metrics (Dict[str, float], optional): Метрики производительности
            config (Dict, optional): Конфигурация модели и обучения
            **kwargs: Дополнительные данные для сохранения
            
        Returns:
            str: Путь к сохраненному чекпоинту
        """
        # Формируем имя файла чекпоинта
        if epoch is not None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        elif global_step is not None:
            filename = f"checkpoint_step_{global_step}.pth"
        else:
            filename = f"checkpoint_{int(time.time())}.pth"
            
        # Полный путь к чекпоинту
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Сохраняем чекпоинт
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config=config,
            path=checkpoint_path,
            **kwargs
        )
        
        # Добавляем информацию о чекпоинте в список
        checkpoint_info = {
            'path': checkpoint_path,
            'timestamp': os.path.getmtime(checkpoint_path),
            'metrics': metrics
        }
        self.checkpoints.append(checkpoint_info)
        
        # Сортируем чекпоинты по времени создания
        self.checkpoints.sort(key=lambda x: x['timestamp'])
        
        # Проверяем, является ли текущий чекпоинт лучшим
        if self.save_best and metrics and self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            
            if ((self.mode == 'min' and current_metric < self.best_metric) or 
                (self.mode == 'max' and current_metric > self.best_metric)):
                # Обновляем лучшую метрику
                self.best_metric = current_metric
                
                # Копируем чекпоинт как лучший
                best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
                shutil.copy(checkpoint_path, best_path)
                self.best_checkpoint_path = best_path
                
                print(f"Сохранен новый лучший чекпоинт с {self.metric_name}={current_metric}")
        
        # Удаляем старые чекпоинты, если их слишком много
        self._clean_old_checkpoints()
        
        return checkpoint_path
        
    def _clean_old_checkpoints(self) -> None:
        """Удаляет старые чекпоинты, оставляя только max_checkpoints последних."""
        # Если количество чекпоинтов не превышает максимальное, ничего не делаем
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Определяем, сколько чекпоинтов нужно удалить
        num_to_delete = len(self.checkpoints) - self.max_checkpoints
        
        # Получаем пути к чекпоинтам для удаления (самые старые)
        checkpoints_to_delete = self.checkpoints[:num_to_delete]
        
        # Удаляем чекпоинты
        for checkpoint_info in checkpoints_to_delete:
            path = checkpoint_info['path']
            
            # Не удаляем лучший чекпоинт
            if self.best_checkpoint_path == path:
                continue
                
            try:
                os.remove(path)
                print(f"Удален старый чекпоинт: {path}")
            except Exception as e:
                print(f"Ошибка при удалении чекпоинта {path}: {e}")
                
        # Обновляем список чекпоинтов
        self.checkpoints = self.checkpoints[num_to_delete:]
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Возвращает путь к последнему сохраненному чекпоинту.
        
        Returns:
            Optional[str]: Путь к последнему чекпоинту или None, если чекпоинты не найдены
        """
        if not self.checkpoints:
            return None
            
        return self.checkpoints[-1]['path']
        
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Возвращает путь к лучшему чекпоинту.
        
        Returns:
            Optional[str]: Путь к лучшему чекпоинту или None, если лучший чекпоинт не найден
        """
        if not self.best_checkpoint_path:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            if os.path.exists(best_path):
                return best_path
            else:
                return find_best_checkpoint(self.checkpoint_dir, self.metric_name, self.mode)
                
        return self.best_checkpoint_path


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Загружает модель из чекпоинта.
    
    Args:
        checkpoint_path (str): Путь к чекпоинту
        config (Dict, optional): Дополнительная конфигурация для создания модели
        device (torch.device, optional): Устройство для загрузки модели
        
    Returns:
        nn.Module: Загруженная модель
    """
    # Определяем устройство для загрузки
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Проверяем, содержит ли чекпоинт модель напрямую
    if 'model' in checkpoint:
        model = checkpoint['model']
        model.to(device)
        return model
        
    # Если в чекпоинте нет модели, создаем ее
    model_config = {}
    
    # Извлекаем конфигурацию модели из чекпоинта
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        if isinstance(checkpoint_config, dict) and 'model' in checkpoint_config:
            model_config.update(checkpoint_config['model'])
            
    # Обновляем конфигурацию переданными параметрами
    if config:
        model_config.update(config)
        
    # Создаем модель на основе конфигурации
    from core.swin_unet import SwinUNet
    from core.vit_semantic import ViTSemantic
    from core.fpn_pyramid import FPNPyramid
    from core.cross_attention_bridge import CrossAttentionBridge
    from core.feature_fusion import MultiHeadFeatureFusion
    
    # Основная модель Swin-UNet
    swin_unet = SwinUNet(
        img_size=model_config.get('img_size', 256),
        patch_size=model_config.get('patch_size', 4),
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 2),
        embed_dim=model_config.get('embed_dim', 96),
        depths=model_config.get('depths', [2, 2, 6, 2]),
        num_heads=model_config.get('num_heads', [3, 6, 12, 24]),
        window_size=model_config.get('window_size', 8),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        attention_dropout_rate=model_config.get('attention_dropout_rate', 0.0),
        return_intermediate=True
    )
    
    # ViT для семантического понимания
    vit_semantic = ViTSemantic(
        img_size=model_config.get('img_size', 256),
        patch_size=model_config.get('vit_patch_size', 16),
        in_channels=model_config.get('in_channels', 1),
        embed_dim=model_config.get('vit_embed_dim', 768),
        depth=model_config.get('vit_depth', 12),
        num_heads=model_config.get('vit_num_heads', 12),
        mlp_ratio=model_config.get('vit_mlp_ratio', 4.0),
        dropout_rate=model_config.get('vit_dropout_rate', 0.0)
    )
    
    # FPN с пирамидальным пулингом
    fpn = FPNPyramid(
        in_channels_list=model_config.get('fpn_in_channels', [96, 192, 384, 768]),
        out_channels=model_config.get('fpn_out_channels', 256),
        use_pyramid_pooling=model_config.get('use_pyramid_pooling', True)
    )
    
    # Мост Cross-Attention
    cross_attention = CrossAttentionBridge(
        swin_dim=model_config.get('fpn_out_channels', 256),
        vit_dim=model_config.get('vit_embed_dim', 768),
        num_heads=model_config.get('cross_attention_heads', 8),
        dropout_rate=model_config.get('cross_attention_dropout', 0.0)
    )
    
    # Модуль слияния признаков
    feature_fusion = MultiHeadFeatureFusion(
        in_channels_list=[model_config.get('fpn_out_channels', 256), model_config.get('vit_embed_dim', 768)],
        out_channels=model_config.get('fusion_out_channels', 512),
        num_heads=model_config.get('fusion_num_heads', 8)
    )
    
    # Создаем полную модель
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.swin_unet = swin_unet
            self.vit_semantic = vit_semantic
            self.fpn = fpn
            self.cross_attention = cross_attention
            self.feature_fusion = feature_fusion
            
            # Финальный слой
            self.final = nn.Conv2d(model_config.get('fusion_out_channels', 512), 
                                  model_config.get('out_channels', 2), kernel_size=1)
            
        def forward(self, x):
            # Swin-UNet обработка
            swin_features = self.swin_unet(x)
            
            # ViT обработка
            vit_features = self.vit_semantic(x)
            
            # FPN обработка
            fpn_features = self.fpn(swin_features)
            
            # Cross-Attention между FPN и ViT
            attended_features = self.cross_attention(fpn_features, vit_features)
            
            # Слияние признаков
            fused_features = self.feature_fusion([attended_features, vit_features])
            
            # Финальное предсказание
            output = self.final(fused_features)
            
            # Возвращаем результат
            if model_config.get('out_channels', 2) == 2:
                # Возвращаем a и b каналы для пространства LAB
                return {'a': output[:, 0:1], 'b': output[:, 1:2]}
            else:
                # Возвращаем полное изображение
                return {'colorized': output}
    
    # Создаем модель
    model = FullModel()
    
    # Загружаем состояние модели
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        
    # Переносим модель на устройство
    model.to(device)
    model.eval()  # Переводим модель в режим оценки
    
    return model


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description="TintoraAI Checkpoint Utilities")
    parser.add_argument("--mode", type=str, required=True, choices=[
        "save", "load", "convert", "combine", "clean", "extract"
    ], help="Режим работы")
    parser.add_argument("--input", type=str, help="Путь к входному чекпоинту или директории")
    parser.add_argument("--output", type=str, help="Путь для сохранения результата")
    parser.add_argument("--config", type=str, help="Путь к конфигурации")
    parser.add_argument("--metric", type=str, default="val_loss", help="Метрика для поиска лучшего чекпоинта")
    parser.add_argument("--mode-metric", type=str, default="min", choices=["min", "max"], help="Режим сравнения для метрики")
    parser.add_argument("--keep-last", type=int, default=5, help="Количество последних чекпоинтов для сохранения")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "save":
            # Пример сохранения чекпоинта
            print("Для сохранения чекпоинта используйте функцию save_checkpoint из кода.")
            
        elif args.mode == "load":
            # Загрузка чекпоинта
            if not args.input:
                raise ValueError("Не указан путь к входному чекпоинту")
                
            checkpoint = load_checkpoint(args.input)
            
            print(f"Чекпоинт загружен из {args.input}")
            if 'epoch' in checkpoint:
                print(f"Эпоха: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"Глобальный шаг: {checkpoint['global_step']}")
            if 'metrics' in checkpoint:
                print(f"Метрики: {checkpoint['metrics']}")
                
        elif args.mode == "convert":
            # Конвертация чекпоинта в файл модели
            if not args.input or not args.output:
                raise ValueError("Не указаны пути к входному чекпоинту и выходному файлу модели")
                
            convert_checkpoint_to_model(args.input, args.output)
            
        elif args.mode == "combine":
            # Объединение чекпоинтов
            if not args.input or not args.output:
                raise ValueError("Не указаны пути к входным чекпоинтам и выходному чекпоинту")
                
            # Разделяем пути к входным чекпоинтам
            input_paths = args.input.split(',')
            
            combine_checkpoints(input_paths, args.output)
            
        elif args.mode == "clean":
            # Очистка директории с чекпоинтами
            if not args.input:
                raise ValueError("Не указана директория с чекпоинтами")
                
            clean_checkpoints(
                args.input,
                keep_last_n=args.keep_last,
                keep_epoch_step=10,
                keep_best=True,
                best_metric=args.metric,
                best_mode=args.mode_metric
            )
            
        elif args.mode == "extract":
            # Извлечение модели из чекпоинта
            if not args.input or not args.output:
                raise ValueError("Не указаны пути к входному чекпоинту и выходному файлу модели")
                
            # Загружаем модель из чекпоинта
            model = load_model_from_checkpoint(args.input)
            
            # Сохраняем только модель
            torch.save({'model': model}, args.output)
            
            print(f"Модель извлечена из чекпоинта {args.input} и сохранена в {args.output}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")