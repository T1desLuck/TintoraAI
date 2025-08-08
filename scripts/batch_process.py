#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Скрипт для пакетной обработки изображений

Данный скрипт предназначен для массовой колоризации большого количества
изображений с поддержкой многопроцессорной обработки, отслеживания прогресса,
и возможностью паузы/возобновления. Скрипт оптимизирован для эффективной 
обработки как небольших, так и очень крупных наборов изображений.

Возможности:
- Пакетная обработка целых директорий с изображениями
- Распределенная обработка на нескольких ядрах CPU или GPU
- Отслеживание прогресса с визуальным индикатором
- Поддержка паузы и возобновления процесса
- Автоматическое распределение нагрузки для оптимальной производительности
- Настраиваемые параметры колоризации для всей партии изображений
- Сохранение метаданных и статистики обработки
"""

import os
import sys
import time
import json
import argparse
import glob
import queue
import threading
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
import cv2
from PIL import Image
from tqdm import tqdm

# Добавляем корневую директорию проекта в путь поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из модулей проекта
from inference.predictor import ColorizationPredictor
from inference.postprocessor import ColorizationPostProcessor
from inference.batch_processor import BatchProcessor

from utils.config_parser import load_config
from utils.visualization import ColorizationVisualizer
from utils.metrics import MetricsCalculator

from modules.uncertainty_estimation import UncertaintyEstimation


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Пакетная обработка изображений",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Основные параметры
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Путь к файлу конфигурации инференса")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                        help="Путь к файлу конфигурации модели")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к файлу чекпоинта модели")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Директория с входными изображениями")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Директория для сохранения результатов")
    
    # Параметры обработки
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Размер батча для обработки (перезаписывает значение из конфига)")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Размер изображения для обработки (перезаписывает значение из конфига)")
    parser.add_argument("--color-space", type=str, default="lab",
                        choices=["lab", "rgb", "yuv"],
                        help="Цветовое пространство для колоризации")
    parser.add_argument("--recursive", action="store_true", default=False,
                        help="Рекурсивный поиск изображений в поддиректориях")
    parser.add_argument("--file-pattern", type=str, default="*",
                        help="Шаблон для фильтрации файлов (например: '*.jpg')")
    
    # Параметры параллельной обработки
    parser.add_argument("--workers", type=int, default=None,
                        help="Количество рабочих процессов (None = авто)")
    parser.add_argument("--gpu-batch-size", type=int, default=None,
                        help="Размер батча для GPU (если отличается от обычного)")
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Использовать распределенную обработку на нескольких GPU")
    parser.add_argument("--device-ids", type=str, default=None,
                        help="ID устройств для использования, разделенные запятыми (например: '0,1,2')")
    
    # Параметры для контроля обработки
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Возобновить обработку с последней контрольной точки")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                        help="Частота создания контрольных точек (в количестве файлов)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Максимальное количество файлов для обработки")
    parser.add_argument("--file-list", type=str, default=None,
                        help="Путь к файлу со списком файлов для обработки")
    
    # Параметры вывода и отслеживания
    parser.add_argument("--log-file", type=str, default=None,
                        help="Путь к файлу журнала")
    parser.add_argument("--save-comparison", action="store_true", default=False,
                        help="Сохранять сравнение до/после")
    parser.add_argument("--save-uncertainty", action="store_true", default=False,
                        help="Сохранять карту неопределенности")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                        help="Сохранять метаданные обработки")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Суффикс для имен выходных файлов")
    
    # Параметры постобработки
    parser.add_argument("--enhance", action="store_true", default=None,
                        help="Применить улучшение результатов")
    parser.add_argument("--saturation", type=float, default=None,
                        help="Коэффициент насыщенности (1.0 = оригинальная насыщенность)")
    parser.add_argument("--contrast", type=float, default=None,
                        help="Коэффициент контраста (1.0 = оригинальный контраст)")
    parser.add_argument("--style", type=str, default=None,
                        help="Стиль колоризации")
    parser.add_argument("--style-image", type=str, default=None,
                        help="Путь к изображению для переноса стиля")
    
    # Дополнительные параметры
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Выводить подробную информацию о процессе")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Режим тестирования без фактического выполнения колоризации")
    
    return parser.parse_args()


def setup_logging(args):
    """
    Настройка логирования.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger("TintoraAI")
    logger.setLevel(logging.INFO)
    
    # Форматирование логов
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Вывод в файл, если указан
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_environment(args, logger):
    """
    Настройка окружения для пакетной обработки.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        logger (logging.Logger): Логгер
        
    Returns:
        Tuple: (device, output_dirs) - устройство и директории для вывода
    """
    # Определяем устройства для вычислений
    if args.device_ids:
        # Пользовательский выбор устройств
        device_ids = [int(id.strip()) for id in args.device_ids.split(',')]
        
        # Проверяем доступность устройств
        available_devices = []
        for device_id in device_ids:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                available_devices.append(device_id)
            else:
                logger.warning(f"Устройство CUDA:{device_id} недоступно, будет пропущено")
                
        if available_devices:
            device = torch.device(f"cuda:{available_devices[0]}")
            logger.info(f"Используются GPU: {available_devices}")
        else:
            device = torch.device("cpu")
            logger.warning("Ни одно из указанных устройств не доступно, используется CPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU недоступен, используется CPU")
    
    # Создаем директории для вывода
    output_dirs = {
        'colorized': os.path.join(args.output_dir, 'colorized'),
        'comparisons': os.path.join(args.output_dir, 'comparisons'),
        'uncertainty_maps': os.path.join(args.output_dir, 'uncertainty_maps'),
        'metadata': os.path.join(args.output_dir, 'metadata'),
        'checkpoints': os.path.join(args.output_dir, 'process_checkpoints')
    }
    
    for dir_path in output_dirs.values():
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Создана директория: {dir_path}")
        except Exception as e:
            logger.error(f"Не удалось создать директорию {dir_path}: {str(e)}")
    
    return device, output_dirs


def load_model(args, device, logger):
    """
    Загрузка модели и компонентов для колоризации.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для вычислений
        logger (logging.Logger): Логгер
        
    Returns:
        Tuple: (model, predictor, postprocessor, uncertainty_module) - модель и компоненты для колоризации
    """
    # Загружаем конфигурации
    config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    if config is None:
        logger.warning("Не удалось загрузить конфигурацию инференса, используются значения по умолчанию")
        config = {}
    
    if model_config is None:
        logger.warning("Не удалось загрузить конфигурацию модели, используются значения по умолчанию")
        model_config = {}
    
    # Объединяем значения из конфигурации и аргументов командной строки
    batch_size = args.batch_size or config.get('batch_size', 8)
    gpu_batch_size = args.gpu_batch_size or batch_size
    color_space = args.color_space or config.get('color_space', 'lab')
    
    # Значения для постобработки
    enhance = args.enhance if args.enhance is not None else config.get('postprocessing', {}).get('enhance', False)
    saturation = args.saturation or config.get('postprocessing', {}).get('saturation', 1.0)
    contrast = args.contrast or config.get('postprocessing', {}).get('contrast', 1.0)
    
    # Загружаем модель из чекпоинта
    try:
        logger.info(f"Загрузка модели из чекпоинта: {args.checkpoint}")
        
        # Загрузка чекпоинта
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model' in checkpoint:
            model = checkpoint['model']
            model = model.to(device)
            logger.info("Модель успешно загружена из чекпоинта")
        elif 'model_state_dict' in checkpoint:
            # Импорт функции для загрузки модели
            from training.checkpoints import load_model_from_checkpoint
            model = load_model_from_checkpoint(args.checkpoint, model_config, device)
            logger.info("Модель успешно загружена из state_dict")
        else:
            raise ValueError(f"Некорректный формат чекпоинта: {args.checkpoint}")
        
        # Переводим модель в режим оценки
        model.eval()
        
        # Создаем постпроцессор
        postprocessor = ColorizationPostProcessor(
            color_space=color_space,
            apply_enhancement=enhance,
            saturation=saturation,
            contrast=contrast,
            device=device
        )
        
        # Инициализируем интеллектуальные модули
        intelligent_modules = {}
        
        # Создаем модуль оценки неопределенности, если нужно
        uncertainty_module = None
        if args.save_uncertainty and model_config.get('uncertainty', {}).get('enabled', True):
            uncertainty_config = model_config.get('uncertainty', {})
            uncertainty_module = UncertaintyEstimation(
                method=uncertainty_config.get('method', 'guided'),
                num_samples=uncertainty_config.get('num_samples', 5),
                dropout_rate=uncertainty_config.get('dropout_rate', 0.2),
                device=device
            ).to(device)
            intelligent_modules['uncertainty'] = uncertainty_module
            logger.info("Создан модуль оценки неопределенности")
        
        # Создаем предиктор
        predictor = ColorizationPredictor(
            model=model,
            device=device,
            color_space=color_space,
            intelligent_modules=intelligent_modules
        )
        
        logger.info("Предиктор и постпроцессор успешно созданы")
        
        return model, predictor, postprocessor, uncertainty_module
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise


def find_images(args, logger):
    """
    Находит изображения для обработки.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        logger (logging.Logger): Логгер
        
    Returns:
        List[str]: Список путей к изображениям для обработки
    """
    # Поддерживаемые форматы изображений
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    
    # Проверяем, указан ли файл со списком файлов
    if args.file_list:
        try:
            with open(args.file_list, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
                
            # Проверяем существование файлов
            files = [f for f in files if os.path.exists(f)]
            logger.info(f"Загружено {len(files)} файлов из списка")
            
            return files
        except Exception as e:
            logger.error(f"Не удалось загрузить список файлов: {str(e)}")
            return []
    
    # Проверяем, существует ли директория
    if not os.path.exists(args.input_dir):
        logger.error(f"Директория не существует: {args.input_dir}")
        return []
        
    # Находим все изображения по шаблону
    if args.recursive:
        # Рекурсивный поиск
        all_files = []
        
        # Разбиваем шаблон на части
        if '.' in args.file_pattern:
            pattern_parts = args.file_pattern.split('.')
            pattern_name = pattern_parts[0]
            pattern_ext = '.'.join(pattern_parts[1:])
        else:
            pattern_name = args.file_pattern
            pattern_ext = '*'
        
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_name, file_ext = os.path.splitext(file)
                
                # Проверяем соответствие шаблону и расширение
                if (pattern_name == '*' or file_name.startswith(pattern_name.replace('*', '')) or 
                    fnmatch.fnmatch(file_name, pattern_name)) and \
                   (pattern_ext == '*' or file_ext.lower() in supported_extensions or 
                    fnmatch.fnmatch(file_ext.lower()[1:], pattern_ext)):
                    all_files.append(file_path)
    else:
        # Поиск только в указанной директории
        pattern = args.file_pattern
        
        # Если шаблон не содержит расширения, добавляем все поддерживаемые
        if '.' not in pattern:
            all_files = []
            for ext in supported_extensions:
                all_files.extend(glob.glob(os.path.join(args.input_dir, f"{pattern}{ext}")))
                all_files.extend(glob.glob(os.path.join(args.input_dir, f"{pattern}{ext.upper()}")))
        else:
            all_files = glob.glob(os.path.join(args.input_dir, pattern))
    
    # Проверяем, что файлы действительно являются изображениями
    image_files = []
    for file_path in all_files:
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(file_path)
    
    # Ограничиваем количество файлов, если указано
    if args.max_files is not None and args.max_files > 0:
        image_files = image_files[:args.max_files]
    
    logger.info(f"Найдено {len(image_files)} изображений для обработки")
    
    return image_files


def load_checkpoint_state(checkpoint_dir, logger):
    """
    Загружает состояние контрольной точки для возобновления обработки.
    
    Args:
        checkpoint_dir (str): Директория с контрольными точками
        logger (logging.Logger): Логгер
        
    Returns:
        Tuple: (processed_files, last_checkpoint) - обработанные файлы и последняя точка
    """
    # Находим последний чекпоинт
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
    
    if not checkpoint_files:
        logger.info("Контрольные точки не найдены, начинаем с начала")
        return set(), None
        
    # Сортируем по времени создания
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    last_checkpoint = checkpoint_files[0]
    
    try:
        # Загружаем информацию о точке
        with open(last_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
            
        processed_files = set(checkpoint_data.get('processed_files', []))
        total_files = checkpoint_data.get('total_files', 0)
        timestamp = checkpoint_data.get('timestamp', 'неизвестно')
        
        logger.info(f"Загружена контрольная точка от {timestamp}")
        logger.info(f"Обработано {len(processed_files)}/{total_files} файлов")
        
        return processed_files, last_checkpoint
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке контрольной точки: {str(e)}")
        return set(), None


def save_checkpoint_state(checkpoint_dir, processed_files, total_files, logger):
    """
    Сохраняет состояние обработки в контрольную точку.
    
    Args:
        checkpoint_dir (str): Директория для сохранения контрольных точек
        processed_files (set): Множество обработанных файлов
        total_files (int): Общее количество файлов
        logger (logging.Logger): Логгер
        
    Returns:
        str: Путь к сохраненной контрольной точке
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.json")
    
    # Создаем данные для контрольной точки
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_files': list(processed_files),
        'total_files': total_files,
        'completed': len(processed_files) >= total_files
    }
    
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        logger.info(f"Сохранена контрольная точка: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении контрольной точки: {str(e)}")
        return None


class BatchProcessor:
    """
    Класс для пакетной обработки изображений.
    
    Attributes:
        args (argparse.Namespace): Аргументы командной строки
        logger (logging.Logger): Логгер
        device (torch.device): Устройство для вычислений
        output_dirs (dict): Директории для вывода
        predictor (ColorizationPredictor): Предиктор для колоризации
        postprocessor (ColorizationPostProcessor): Постпроцессор для обработки результатов
        uncertainty_module (UncertaintyEstimation): Модуль оценки неопределенности
        visualizer (ColorizationVisualizer): Визуализатор для создания сравнений
        batch_size (int): Размер батча для обработки
        files (list): Список файлов для обработки
        processed_files (set): Множество обработанных файлов
        paused (bool): Флаг паузы обработки
        stop_requested (bool): Флаг запроса на остановку
    """
    
    def __init__(self, args, logger, device, output_dirs, predictor, postprocessor, uncertainty_module):
        """
        Инициализация обработчика пакетов.
        
        Args:
            args (argparse.Namespace): Аргументы командной строки
            logger (logging.Logger): Логгер
            device (torch.device): Устройство для вычислений
            output_dirs (dict): Директории для вывода
            predictor (ColorizationPredictor): Предиктор для колоризации
            postprocessor (ColorizationPostProcessor): Постпроцессор для обработки результатов
            uncertainty_module (UncertaintyEstimation): Модуль оценки неопределенности
        """
        self.args = args
        self.logger = logger
        self.device = device
        self.output_dirs = output_dirs
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.uncertainty_module = uncertainty_module
        
        # Создаем визуализатор для сравнений
        self.visualizer = ColorizationVisualizer(output_dir=output_dirs['comparisons'])
        
        # Получаем размер батча
        self.batch_size = args.batch_size or 8
        
        # Находим файлы для обработки
        self.files = find_images(args, logger)
        
        # Инициализация состояния
        self.processed_files = set()
        self.paused = False
        self.stop_requested = False
        
        # Загружаем состояние, если нужно
        if args.resume:
            processed_files, _ = load_checkpoint_state(output_dirs['checkpoints'], logger)
            self.processed_files = processed_files
    
    def setup_signal_handlers(self):
        """Настройка обработчиков сигналов для паузы/остановки."""
        # Определяем обработчики сигналов
        def handle_pause(sig, frame):
            self.paused = not self.paused
            if self.paused:
                self.logger.info("Обработка приостановлена. Нажмите Ctrl+Z для возобновления.")
            else:
                self.logger.info("Обработка возобновлена.")
        
        def handle_stop(sig, frame):
            self.stop_requested = True
            self.logger.info("Получен сигнал остановки. Завершение после текущего батча...")
        
        # Регистрируем обработчики сигналов
        signal.signal(signal.SIGTSTP, handle_pause)  # Ctrl+Z
        signal.signal(signal.SIGINT, handle_stop)   # Ctrl+C
    
    def process_batch(self, batch_files):
        """
        Обрабатывает батч файлов.
        
        Args:
            batch_files (List[str]): Список файлов для обработки
            
        Returns:
            int: Количество успешно обработанных файлов
        """
        if not batch_files:
            return 0
            
        # Счетчик успешных обработок
        success_count = 0
        
        # Проверяем режим dry-run
        if self.args.dry_run:
            # В режиме dry-run только имитируем обработку
            time.sleep(0.1 * len(batch_files))  # Имитация времени обработки
            return len(batch_files)
        
        # Обрабатываем каждый файл в батче
        for file_path in batch_files:
            try:
                # Проверяем, что файл существует
                if not os.path.exists(file_path):
                    self.logger.warning(f"Файл не найден: {file_path}")
                    continue
                
                # Колоризируем изображение
                result = self.predictor.colorize_image(
                    image_path=file_path,
                    postprocessor=self.postprocessor,
                    batch_size=1  # Обрабатываем по одному внутри батча
                )
                
                if not result:
                    self.logger.warning(f"Не удалось колоризировать изображение: {file_path}")
                    continue
                
                # Получаем имя файла и пути для сохранения
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                if self.args.suffix:
                    base_name = f"{base_name}_{self.args.suffix}"
                    
                # Сохраняем колоризованное изображение
                colorized_path = os.path.join(self.output_dirs['colorized'], f"{base_name}.png")
                colorized_img = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
                colorized_img.save(colorized_path)
                
                # Сохраняем сравнение, если нужно
                if self.args.save_comparison:
                    comparison_path = os.path.join(self.output_dirs['comparisons'], f"{base_name}_comparison.png")
                    
                    # Загружаем исходное изображение для сравнения
                    grayscale_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Создаем сравнение
                    self.visualizer.create_comparison(
                        grayscale=grayscale_img,
                        colorized=result['colorized'],
                        filename=comparison_path
                    )
                
                # Сохраняем карту неопределенности, если нужно
                if self.args.save_uncertainty and self.uncertainty_module:
                    uncertainty_path = os.path.join(self.output_dirs['uncertainty_maps'], f"{base_name}_uncertainty.png")
                    
                    # Преобразуем результат в тензор для оценки неопределенности
                    colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # Оцениваем неопределенность
                    uncertainty_result = self.uncertainty_module(colorized_tensor)
                    uncertainty_map = uncertainty_result['uncertainty'][0, 0].cpu().numpy()
                    
                    # Преобразуем карту неопределенности для визуализации
                    uncertainty_vis = (uncertainty_map * 255).astype(np.uint8)
                    cv2.imwrite(uncertainty_path, uncertainty_vis)
                
                # Сохраняем метаданные, если нужно
                if self.args.save_metadata:
                    metadata_path = os.path.join(self.output_dirs['metadata'], f"{base_name}.json")
                    
                    # Собираем метаданные
                    metadata = {
                        'original_image': file_path,
                        'colorized_image': colorized_path,
                        'timestamp': datetime.now().isoformat(),
                        'model': os.path.basename(self.args.checkpoint),
                        'config': {
                            'color_space': self.args.color_space,
                            'enhancement': self.postprocessor.apply_enhancement,
                            'saturation': self.postprocessor.saturation,
                            'contrast': self.postprocessor.contrast
                        }
                    }
                    
                    # Добавляем пути к другим результатам, если они есть
                    if self.args.save_comparison:
                        metadata['comparison_image'] = comparison_path
                    if self.args.save_uncertainty and self.uncertainty_module:
                        metadata['uncertainty_image'] = uncertainty_path
                    
                    # Сохраняем метаданные в JSON
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Увеличиваем счетчик успешных операций
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
                if self.args.verbose:
                    import traceback
                    traceback.print_exc()
        
        return success_count
    
    def process_all(self):
        """
        Обрабатывает все файлы.
        
        Returns:
            Tuple[int, int]: (успешно_обработано, всего_файлов)
        """
        # Настраиваем обработчики сигналов
        self.setup_signal_handlers()
        
        # Отбираем файлы, которые не были обработаны
        files_to_process = [f for f in self.files if f not in self.processed_files]
        total_files = len(files_to_process)
        
        self.logger.info(f"Начинаем обработку {total_files} файлов")
        
        # Инициализируем прогресс-бар
        with tqdm(total=total_files, desc="Обработка изображений") as pbar:
            # Обновляем прогресс-бар с учетом уже обработанных файлов
            pbar.update(len(self.processed_files))
            
            # Обрабатываем файлы батчами
            for i in range(0, len(files_to_process), self.batch_size):
                # Проверяем запрос на остановку
                if self.stop_requested:
                    self.logger.info("Обработка остановлена пользователем")
                    break
                    
                # Проверяем режим паузы
                while self.paused and not self.stop_requested:
                    time.sleep(1)
                
                # Формируем батч
                batch = files_to_process[i:i+self.batch_size]
                
                # Обрабатываем батч
                processed = self.process_batch(batch)
                
                # Обновляем множество обработанных файлов
                self.processed_files.update(batch[:processed])
                
                # Обновляем прогресс-бар
                pbar.update(processed)
                
                # Сохраняем контрольную точку, если достигнут интервал
                if len(self.processed_files) % self.args.checkpoint_freq == 0 or i + self.batch_size >= len(files_to_process):
                    save_checkpoint_state(self.output_dirs['checkpoints'], self.processed_files, len(self.files), self.logger)
                    
        # Сохраняем итоговую статистику
        self.logger.info(f"Обработка завершена. Успешно обработано {len(self.processed_files)} из {len(self.files)} файлов")
        
        return len(self.processed_files), len(self.files)
        

def main():
    """Основная функция для запуска пакетной обработки."""
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Настройка логирования
    logger = setup_logging(args)
    
    try:
        logger.info("Запуск пакетной обработки изображений")
        
        # Настройка окружения
        device, output_dirs = setup_environment(args, logger)
        
        # Загрузка модели и компонентов
        model, predictor, postprocessor, uncertainty_module = load_model(args, device, logger)
        
        # Создаем обработчик пакетов
        batch_processor = BatchProcessor(
            args, logger, device, output_dirs, 
            predictor, postprocessor, uncertainty_module
        )
        
        # Выполняем обработку всех файлов
        processed_count, total_count = batch_processor.process_all()
        
        # Выводим итоги
        logger.info("=" * 40)
        logger.info("Итоги обработки:")
        logger.info(f"Всего файлов: {total_count}")
        logger.info(f"Успешно обработано: {processed_count}")
        logger.info(f"Процент успеха: {processed_count/total_count*100:.2f}%" if total_count > 0 else "Нет файлов для обработки")
        logger.info("=" * 40)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nОбработка прервана пользователем")
        return 1
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())