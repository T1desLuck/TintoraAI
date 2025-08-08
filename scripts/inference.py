#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Скрипт инференса модели колоризации изображений

Данный скрипт предоставляет возможность колоризации черно-белых изображений
с использованием обученной модели TintoraAI. Поддерживает работу в различных
режимах: единичная обработка изображения, пакетная обработка директории
и автоматическая обработка изображений из очереди.

Возможности:
- Колоризация одиночных изображений
- Пакетная обработка множества изображений
- Мониторинг и автоматическая обработка из очереди
- Применение различных стилей колоризации
- Управление параметрами через интерфейс командной строки и конфигурационные файлы
- Генерация карт неопределенности для оценки качества результата
- Сохранение метаданных и информации об обработке
- Визуализация сравнений "до/после"
"""

import os
import sys
import time
import yaml
import json
import argparse
import threading
import queue
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from PIL import Image
import cv2

# Добавляем корневую директорию проекта в путь поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из модулей проекта
from inference.predictor import ColorizationPredictor
from inference.postprocessor import ColorizationPostProcessor
from inference.batch_processor import BatchProcessor

from utils.config_parser import load_config
from utils.visualization import ColorizationVisualizer
from utils.user_interaction import UserInteractionModule
from utils.metrics import MetricsCalculator

from modules.guide_net import GuideNet
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.few_shot_adapter import AdaptableColorizer
from modules.style_transfer import StyleTransfer
from modules.memory_bank import MemoryBankModule

from training.checkpoints import load_checkpoint


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Инференс модели колоризации изображений",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Общие параметры
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Путь к файлу конфигурации инференса")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                        help="Путь к файлу конфигурации модели")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к файлу чекпоинта модели")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Директория для сохранения результатов")
    
    # Режимы работы (взаимоисключающие)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", type=str, 
                            help="Путь к одиночному изображению для колоризации")
    mode_group.add_argument("--batch", type=str,
                            help="Путь к директории с изображениями для пакетной обработки")
    mode_group.add_argument("--queue", type=str,
                            help="Путь к директории для автоматического мониторинга и обработки")
    
    # Параметры обработки
    parser.add_argument("--color-space", type=str, default="lab",
                        choices=["lab", "rgb", "yuv"],
                        help="Цветовое пространство для колоризации")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Размер изображения для обработки (перезаписывает значение из конфига)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Размер батча для обработки (перезаписывает значение из конфига)")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Использовать GPU для инференса")
    
    # Параметры постобработки
    parser.add_argument("--enhance", action="store_true", default=None,
                        help="Применить улучшение результатов")
    parser.add_argument("--saturation", type=float, default=None,
                        help="Коэффициент насыщенности (1.0 = оригинальная насыщенность)")
    parser.add_argument("--contrast", type=float, default=None,
                        help="Коэффициент контраста (1.0 = оригинальный контраст)")
    parser.add_argument("--style", type=str, default=None,
                        help="Стиль колоризации (vintage, modern, cinematic, и т.д.)")
    parser.add_argument("--style-image", type=str, default=None,
                        help="Путь к изображению для переноса стиля")
    
    # Параметры вывода
    parser.add_argument("--save-comparison", action="store_true", default=None,
                        help="Сохранять сравнение до/после")
    parser.add_argument("--save-uncertainty", action="store_true", default=None,
                        help="Сохранять карту неопределенности")
    parser.add_argument("--save-metadata", action="store_true", default=None,
                        help="Сохранять метаданные обработки")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Суффикс для имен выходных файлов")
    
    # Параметры режима очереди
    parser.add_argument("--queue-interval", type=float, default=None,
                        help="Интервал проверки очереди в секундах")
    parser.add_argument("--queue-max-files", type=int, default=None,
                        help="Максимальное количество файлов для обработки за один проход")
    parser.add_argument("--queue-timeout", type=int, default=None,
                        help="Время работы режима очереди в секундах (0 = бесконечно)")
    
    # Интерактивный режим
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Запустить в интерактивном режиме (принимать команды)")
    
    # Дополнительные параметры
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Выводить подробную информацию о процессе")
    
    return parser.parse_args()


def setup_environment(args, config):
    """
    Настройка окружения для инференса.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        
    Returns:
        torch.device: Устройство для инференса
    """
    # Определяем устройство
    use_gpu = config.get("use_gpu", False)
    
    # Параметр из командной строки имеет приоритет
    if args.gpu is not None:
        use_gpu = args.gpu
    
    # Проверяем доступность GPU
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if args.verbose:
            print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU недоступен, используется CPU")
        device = torch.device("cpu")
    
    return device


def load_configurations(args):
    """
    Загрузка конфигураций из файлов и объединение с аргументами командной строки.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        tuple: (inference_config, model_config) - словари с конфигурациями
    """
    # Загрузка конфигурации инференса
    inference_config = load_config(args.config)
    if inference_config is None:
        print(f"Не удалось загрузить конфигурацию инференса из {args.config}, используем значения по умолчанию")
        inference_config = {}
    
    # Загрузка конфигурации модели
    model_config = load_config(args.model_config)
    if model_config is None:
        print(f"Не удалось загрузить конфигурацию модели из {args.model_config}, используем значения по умолчанию")
        model_config = {}
    
    # Перезаписываем значения из аргументов командной строки, если они указаны
    if args.img_size is not None:
        inference_config['img_size'] = args.img_size
    
    if args.batch_size is not None:
        inference_config['batch_size'] = args.batch_size
    
    if args.color_space is not None:
        inference_config['color_space'] = args.color_space
    
    if args.enhance is not None:
        inference_config['postprocessing'] = inference_config.get('postprocessing', {})
        inference_config['postprocessing']['enhance'] = args.enhance
    
    if args.saturation is not None:
        inference_config['postprocessing'] = inference_config.get('postprocessing', {})
        inference_config['postprocessing']['saturation'] = args.saturation
    
    if args.contrast is not None:
        inference_config['postprocessing'] = inference_config.get('postprocessing', {})
        inference_config['postprocessing']['contrast'] = args.contrast
    
    if args.style is not None:
        inference_config['style'] = args.style
    
    if args.style_image is not None:
        inference_config['style_image'] = args.style_image
    
    if args.save_comparison is not None:
        inference_config['output'] = inference_config.get('output', {})
        inference_config['output']['save_comparison'] = args.save_comparison
    
    if args.save_uncertainty is not None:
        inference_config['output'] = inference_config.get('output', {})
        inference_config['output']['save_uncertainty'] = args.save_uncertainty
    
    if args.save_metadata is not None:
        inference_config['output'] = inference_config.get('output', {})
        inference_config['output']['save_metadata'] = args.save_metadata
    
    if args.suffix is not None:
        inference_config['output'] = inference_config.get('output', {})
        inference_config['output']['suffix'] = args.suffix
    
    if args.queue_interval is not None:
        inference_config['queue'] = inference_config.get('queue', {})
        inference_config['queue']['interval'] = args.queue_interval
    
    if args.queue_max_files is not None:
        inference_config['queue'] = inference_config.get('queue', {})
        inference_config['queue']['max_files'] = args.queue_max_files
    
    if args.queue_timeout is not None:
        inference_config['queue'] = inference_config.get('queue', {})
        inference_config['queue']['timeout'] = args.queue_timeout
    
    return inference_config, model_config


def setup_directories(args, config):
    """
    Создание директорий для сохранения результатов.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        
    Returns:
        dict: Словарь с путями к директориям для результатов
    """
    # Базовая директория для результатов
    base_dir = args.output_dir
    
    # Определяем структуру директорий на основе конфигурации
    output_config = config.get('output', {})
    save_comparison = output_config.get('save_comparison', False)
    save_uncertainty = output_config.get('save_uncertainty', False)
    save_metadata = output_config.get('save_metadata', False)
    
    # Создаем директории
    dirs = {
        'colorized': os.path.join(base_dir, 'colorized'),
    }
    
    os.makedirs(dirs['colorized'], exist_ok=True)
    
    if save_comparison:
        dirs['comparisons'] = os.path.join(base_dir, 'comparisons')
        os.makedirs(dirs['comparisons'], exist_ok=True)
    
    if save_uncertainty:
        dirs['uncertainty'] = os.path.join(base_dir, 'uncertainty_maps')
        os.makedirs(dirs['uncertainty'], exist_ok=True)
    
    if save_metadata:
        dirs['metadata'] = os.path.join(base_dir, 'metadata')
        os.makedirs(dirs['metadata'], exist_ok=True)
    
    return dirs


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Загрузка модели из чекпоинта.
    
    Args:
        checkpoint_path (str): Путь к чекпоинту
        device (torch.device): Устройство для размещения модели
        
    Returns:
        tuple: (model, intelligent_modules) - модель и словарь с интеллектуальными модулями
    """
    # Загрузка чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Если чекпоинт содержит уже готовую модель
    if 'model' in checkpoint:
        model = checkpoint['model']
        model.to(device)
        
        # Извлекаем интеллектуальные модули, если они есть
        intelligent_modules = {}
        if 'modules_state_dict' in checkpoint:
            # TODO: Здесь должна быть логика восстановления интеллектуальных модулей
            pass
        
        return model, intelligent_modules
    
    # Если чекпоинт содержит только состояния модели
    elif 'model_state_dict' in checkpoint:
        # Загружаем конфигурацию из чекпоинта или используем значения по умолчанию
        config = checkpoint.get('config', {}).get('model', {})
        
        # Создаем модель
        from training.checkpoints import load_model_from_checkpoint
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        
        # Извлекаем интеллектуальные модули, если они есть
        intelligent_modules = {}
        if 'modules_state_dict' in checkpoint:
            # TODO: Здесь должна быть логика восстановления интеллектуальных модулей
            pass
        
        return model, intelligent_modules
    else:
        raise ValueError(f"Некорректный формат чекпоинта: {checkpoint_path}")


def create_predictor_and_postprocessor(model, config, device, intelligent_modules=None):
    """
    Создание предиктора и постпроцессора для колоризации.
    
    Args:
        model (nn.Module): Модель колоризации
        config (dict): Конфигурация инференса
        device (torch.device): Устройство для инференса
        intelligent_modules (dict, optional): Словарь с интеллектуальными модулями
        
    Returns:
        tuple: (predictor, postprocessor) - объекты для предсказания и постобработки
    """
    # Параметры из конфигурации
    color_space = config.get('color_space', 'lab')
    img_size = config.get('img_size', 256)
    
    # Настройки постобработки
    postprocessing_config = config.get('postprocessing', {})
    apply_enhancement = postprocessing_config.get('enhance', False)
    saturation = postprocessing_config.get('saturation', 1.0)
    contrast = postprocessing_config.get('contrast', 1.0)
    
    # Создаем предиктор
    predictor = ColorizationPredictor(
        model=model,
        device=device,
        color_space=color_space,
        intelligent_modules=intelligent_modules
    )
    
    # Создаем постпроцессор
    postprocessor = ColorizationPostProcessor(
        color_space=color_space,
        apply_enhancement=apply_enhancement,
        saturation=saturation,
        contrast=contrast,
        device=device
    )
    
    return predictor, postprocessor


def process_single_image(
    args, config, image_path, predictor, postprocessor, 
    output_dirs, visualizer, uncertainty_module=None
):
    """
    Обработка одиночного изображения.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        image_path (str): Путь к изображению для обработки
        predictor (ColorizationPredictor): Предиктор
        postprocessor (ColorizationPostProcessor): Постпроцессор
        output_dirs (dict): Словарь с директориями для результатов
        visualizer (ColorizationVisualizer): Визуализатор
        uncertainty_module (UncertaintyEstimation, optional): Модуль оценки неопределенности
        
    Returns:
        dict: Результаты колоризации
    """
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл не существует - {image_path}")
        return None
    
    # Получаем имя файла для результатов
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    
    # Добавляем суффикс, если указан
    suffix = config.get('output', {}).get('suffix', '')
    if suffix:
        output_name = f"{name}_{suffix}"
    else:
        output_name = name
    
    if args.verbose:
        print(f"Обрабатывается: {image_path}")
        start_time = time.time()
    
    try:
        # Колоризация изображения
        result = predictor.colorize_image(
            image_path=image_path,
            postprocessor=postprocessor,
            batch_size=config.get('batch_size', 8)
        )
        
        if result is None:
            print(f"Ошибка обработки изображения: {image_path}")
            return None
        
        # Сохраняем колоризованное изображение
        colorized_path = os.path.join(output_dirs['colorized'], f"{output_name}.png")
        colorized_img = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
        colorized_img.save(colorized_path)
        
        # Сохраняем сравнение до/после, если нужно
        if 'comparisons' in output_dirs:
            comparison_path = os.path.join(output_dirs['comparisons'], f"{output_name}_comparison.png")
            
            # Загружаем исходное изображение для сравнения
            grayscale_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            grayscale_rgb = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
            
            # Создаем сравнение
            comparison = visualizer.create_comparison(
                grayscale=grayscale_img,
                colorized=result['colorized'],
                filename=comparison_path
            )
        
        # Оцениваем неопределенность, если нужно
        if uncertainty_module and 'uncertainty' in output_dirs:
            # Преобразуем результат в формат для оценки неопределенности
            colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(predictor.device)
            
            # Оцениваем неопределенность
            uncertainty_result = uncertainty_module(colorized_tensor)
            uncertainty_map = uncertainty_result['uncertainty'][0, 0].cpu().numpy()
            
            # Сохраняем карту неопределенности
            uncertainty_path = os.path.join(output_dirs['uncertainty'], f"{output_name}_uncertainty.png")
            
            # Преобразуем карту неопределенности для визуализации
            uncertainty_vis = (uncertainty_map * 255).astype(np.uint8)
            cv2.imwrite(uncertainty_path, uncertainty_vis)
            
            # Добавляем карту неопределенности к результатам
            result['uncertainty_map'] = uncertainty_map
        
        # Сохраняем метаданные, если нужно
        if 'metadata' in output_dirs:
            metadata_path = os.path.join(output_dirs['metadata'], f"{output_name}.json")
            
            # Собираем метаданные
            metadata = {
                'original_image': image_path,
                'colorized_image': colorized_path,
                'processing_time': time.time() - start_time if args.verbose else None,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'color_space': config.get('color_space'),
                    'enhancement': config.get('postprocessing', {}).get('enhance', False),
                    'saturation': config.get('postprocessing', {}).get('saturation', 1.0),
                    'contrast': config.get('postprocessing', {}).get('contrast', 1.0),
                    'style': config.get('style')
                }
            }
            
            # Добавляем пути к другим результатам, если они есть
            if 'comparisons' in output_dirs:
                metadata['comparison_image'] = comparison_path
            if 'uncertainty' in output_dirs and 'uncertainty_map' in result:
                metadata['uncertainty_image'] = uncertainty_path
            
            # Сохраняем метаданные в JSON
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        if args.verbose:
            end_time = time.time()
            print(f"Обработка завершена за {end_time - start_time:.2f} секунд: {colorized_path}")
        
        return result
        
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {str(e)}")
        return None


def process_batch(
    args, config, input_dir, predictor, postprocessor, 
    output_dirs, visualizer, uncertainty_module=None
):
    """
    Пакетная обработка изображений из директории.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        input_dir (str): Директория с изображениями для обработки
        predictor (ColorizationPredictor): Предиктор
        postprocessor (ColorizationPostProcessor): Постпроцессор
        output_dirs (dict): Словарь с директориями для результатов
        visualizer (ColorizationVisualizer): Визуализатор
        uncertainty_module (UncertaintyEstimation, optional): Модуль оценки неопределенности
        
    Returns:
        int: Количество успешно обработанных изображений
    """
    # Проверяем существование директории
    if not os.path.exists(input_dir):
        print(f"Ошибка: Директория не существует - {input_dir}")
        return 0
    
    # Находим все изображения в директории
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"В директории {input_dir} не найдено изображений")
        return 0
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Создаем батч-процессор
    batch_processor = BatchProcessor(
        predictor=predictor,
        postprocessor=postprocessor,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 2),
        output_dirs=output_dirs,
        visualizer=visualizer,
        uncertainty_module=uncertainty_module
    )
    
    # Запускаем пакетную обработку
    processed_count = batch_processor.process_files(
        file_list=image_files,
        save_comparison='comparisons' in output_dirs,
        save_uncertainty='uncertainty' in output_dirs and uncertainty_module is not None,
        save_metadata='metadata' in output_dirs,
        suffix=config.get('output', {}).get('suffix', ''),
        verbose=args.verbose
    )
    
    return processed_count


def process_queue(
    args, config, queue_dir, predictor, postprocessor, 
    output_dirs, visualizer, uncertainty_module=None
):
    """
    Режим очереди: мониторинг директории и обработка новых изображений.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        queue_dir (str): Директория для мониторинга
        predictor (ColorizationPredictor): Предиктор
        postprocessor (ColorizationPostProcessor): Постпроцессор
        output_dirs (dict): Словарь с директориями для результатов
        visualizer (ColorizationVisualizer): Визуализатор
        uncertainty_module (UncertaintyEstimation, optional): Модуль оценки неопределенности
    """
    # Проверяем существование директории
    if not os.path.exists(queue_dir):
        print(f"Ошибка: Директория не существует - {queue_dir}")
        return
    
    # Параметры режима очереди
    queue_config = config.get('queue', {})
    check_interval = queue_config.get('interval', 5.0)  # Интервал проверки в секундах
    max_files_per_iteration = queue_config.get('max_files', 10)  # Максимум файлов за раз
    timeout = queue_config.get('timeout', 0)  # Время работы (0 = бесконечно)
    
    # Создаем батч-процессор
    batch_processor = BatchProcessor(
        predictor=predictor,
        postprocessor=postprocessor,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 2),
        output_dirs=output_dirs,
        visualizer=visualizer,
        uncertainty_module=uncertainty_module
    )
    
    # Поддерживаемые расширения файлов
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Множество для отслеживания уже обработанных файлов
    processed_files = set()
    
    # Определяем время окончания работы, если задан timeout
    end_time = time.time() + timeout if timeout > 0 else None
    
    print(f"Запущен режим очереди, мониторинг директории: {queue_dir}")
    print(f"Интервал проверки: {check_interval} секунд")
    if end_time:
        print(f"Время работы: {timeout} секунд (до {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')})")
    else:
        print("Время работы: неограничено (Ctrl+C для завершения)")
    
    try:
        while True:
            # Проверяем время окончания
            if end_time and time.time() > end_time:
                print("Время работы истекло, завершение...")
                break
            
            # Находим новые файлы в директории
            new_files = []
            
            for ext in supported_extensions:
                files = glob.glob(os.path.join(queue_dir, f"*{ext}"))
                files.extend(glob.glob(os.path.join(queue_dir, f"*{ext.upper()}")))
                
                # Отбираем только новые файлы, которые еще не обрабатывались
                for file_path in files:
                    if file_path not in processed_files:
                        new_files.append(file_path)
            
            # Если есть новые файлы, обрабатываем их
            if new_files:
                print(f"Найдено {len(new_files)} новых файлов")
                
                # Ограничиваем количество файлов для текущей итерации
                if max_files_per_iteration > 0 and len(new_files) > max_files_per_iteration:
                    batch_files = new_files[:max_files_per_iteration]
                    print(f"Обрабатывается {len(batch_files)} файлов из {len(new_files)}")
                else:
                    batch_files = new_files
                
                # Обрабатываем файлы
                batch_processor.process_files(
                    file_list=batch_files,
                    save_comparison='comparisons' in output_dirs,
                    save_uncertainty='uncertainty' in output_dirs and uncertainty_module is not None,
                    save_metadata='metadata' in output_dirs,
                    suffix=config.get('output', {}).get('suffix', ''),
                    verbose=args.verbose
                )
                
                # Добавляем обработанные файлы в список
                processed_files.update(batch_files)
            
            # Ждем перед следующей проверкой
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("Обработка прервана пользователем")
    
    print(f"Всего обработано файлов: {len(processed_files)}")


def interactive_mode(
    args, config, predictor, postprocessor, 
    output_dirs, visualizer, uncertainty_module=None
):
    """
    Интерактивный режим для работы с колоризатором через командную строку.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        config (dict): Конфигурация инференса
        predictor (ColorizationPredictor): Предиктор
        postprocessor (ColorizationPostProcessor): Постпроцессор
        output_dirs (dict): Словарь с директориями для результатов
        visualizer (ColorizationVisualizer): Визуализатор
        uncertainty_module (UncertaintyEstimation, optional): Модуль оценки неопределенности
    """
    # Создаем модуль взаимодействия с пользователем
    user_interaction = UserInteractionModule()
    
    print("=== TintoraAI - Интерактивный режим ===")
    print("Введите 'помощь' для получения списка команд")
    print("Введите 'выход' для завершения работы")
    
    while True:
        try:
            # Получаем ввод пользователя
            user_input = input("\nTintoraAI> ")
            
            # Проверяем команду выхода
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("Завершение работы...")
                break
            
            # Проверяем команду помощи
            if user_input.lower() in ['помощь', 'help']:
                print("\n=== Доступные команды ===")
                print("колоризовать <путь_к_изображению> - Колоризация изображения")
                print("батч <путь_к_директории> - Пакетная обработка директории")
                print("стиль <название_стиля> - Установить стиль колоризации")
                print("насыщенность <значение> - Установить коэффициент насыщенности (0.0-2.0)")
                print("контраст <значение> - Установить коэффициент контраста (0.0-2.0)")
                print("улучшение <вкл/выкл> - Включить/выключить улучшение результатов")
                print("сравнение <вкл/выкл> - Включить/выключить сохранение сравнений")
                print("неопределенность <вкл/выкл> - Включить/выключить сохранение карт неопределенности")
                print("метаданные <вкл/выкл> - Включить/выключить сохранение метаданных")
                print("суффикс <текст> - Установить суффикс для выходных файлов")
                print("статус - Показать текущие настройки")
                print("выход - Завершение работы")
                continue
                
            # Проверяем статус
            if user_input.lower() == 'статус':
                print("\n=== Текущие настройки ===")
                print(f"Цветовое пространство: {config.get('color_space', 'lab')}")
                print(f"Размер изображения: {config.get('img_size', 256)}")
                print(f"Размер батча: {config.get('batch_size', 8)}")
                print(f"Стиль: {config.get('style', 'нет')}")
                print(f"Улучшение: {config.get('postprocessing', {}).get('enhance', False)}")
                print(f"Насыщенность: {config.get('postprocessing', {}).get('saturation', 1.0)}")
                print(f"Контраст: {config.get('postprocessing', {}).get('contrast', 1.0)}")
                print(f"Сохранение сравнений: {'comparisons' in output_dirs}")
                print(f"Сохранение карт неопределенности: {'uncertainty' in output_dirs}")
                print(f"Сохранение метаданных: {'metadata' in output_dirs}")
                print(f"Суффикс: {config.get('output', {}).get('suffix', '')}")
                continue
            
            # Парсим команду
            parsed = user_interaction.parse_command(user_input)
            
            # Если это не команда, проверяем другие варианты
            if not parsed['is_command']:
                # Проверяем другие команды
                parts = user_input.lower().split()
                
                if len(parts) >= 2:
                    cmd = parts[0]
                    
                    # Команда колоризации
                    if cmd == 'колоризовать':
                        image_path = ' '.join(parts[1:])
                        if os.path.exists(image_path):
                            result = process_single_image(
                                args, config, image_path, predictor, postprocessor,
                                output_dirs, visualizer, uncertainty_module
                            )
                            if result:
                                print(f"Изображение успешно колоризовано: {image_path}")
                        else:
                            print(f"Файл не существует: {image_path}")
                    
                    # Команда пакетной обработки
                    elif cmd == 'батч':
                        dir_path = ' '.join(parts[1:])
                        if os.path.exists(dir_path) and os.path.isdir(dir_path):
                            count = process_batch(
                                args, config, dir_path, predictor, postprocessor,
                                output_dirs, visualizer, uncertainty_module
                            )
                            print(f"Успешно обработано изображений: {count}")
                        else:
                            print(f"Директория не существует: {dir_path}")
                    
                    # Команда установки стиля
                    elif cmd == 'стиль':
                        style_name = ' '.join(parts[1:])
                        config['style'] = style_name
                        print(f"Установлен стиль: {style_name}")
                    
                    # Команда установки насыщенности
                    elif cmd == 'насыщенность':
                        try:
                            saturation = float(parts[1])
                            config['postprocessing'] = config.get('postprocessing', {})
                            config['postprocessing']['saturation'] = saturation
                            
                            # Обновляем постпроцессор
                            postprocessor.saturation = saturation
                            
                            print(f"Установлена насыщенность: {saturation}")
                        except ValueError:
                            print("Ошибка: значение насыщенности должно быть числом")
                    
                    # Команда установки контраста
                    elif cmd == 'контраст':
                        try:
                            contrast = float(parts[1])
                            config['postprocessing'] = config.get('postprocessing', {})
                            config['postprocessing']['contrast'] = contrast
                            
                            # Обновляем постпроцессор
                            postprocessor.contrast = contrast
                            
                            print(f"Установлен контраст: {contrast}")
                        except ValueError:
                            print("Ошибка: значение контраста должно быть числом")
                    
                    # Команда улучшения
                    elif cmd == 'улучшение':
                        value = parts[1].lower()
                        enhance = value in ['вкл', 'on', 'true', '1', 'да', 'yes']
                        config['postprocessing'] = config.get('postprocessing', {})
                        config['postprocessing']['enhance'] = enhance
                        
                        # Обновляем постпроцессор
                        postprocessor.apply_enhancement = enhance
                        
                        print(f"Улучшение {'включено' if enhance else 'выключено'}")
                    
                    # Команда сравнения
                    elif cmd == 'сравнение':
                        value = parts[1].lower()
                        save_comparison = value in ['вкл', 'on', 'true', '1', 'да', 'yes']
                        
                        if save_comparison and 'comparisons' not in output_dirs:
                            output_dirs['comparisons'] = os.path.join(args.output_dir, 'comparisons')
                            os.makedirs(output_dirs['comparisons'], exist_ok=True)
                            print("Сохранение сравнений включено")
                        elif not save_comparison and 'comparisons' in output_dirs:
                            del output_dirs['comparisons']
                            print("Сохранение сравнений выключено")
                    
                    # Команда неопределенности
                    elif cmd == 'неопределенность':
                        value = parts[1].lower()
                        save_uncertainty = value in ['вкл', 'on', 'true', '1', 'да', 'yes']
                        
                        if save_uncertainty:
                            if uncertainty_module is None:
                                print("Ошибка: модуль оценки неопределенности не доступен")
                            elif 'uncertainty' not in output_dirs:
                                output_dirs['uncertainty'] = os.path.join(args.output_dir, 'uncertainty_maps')
                                os.makedirs(output_dirs['uncertainty'], exist_ok=True)
                                print("Сохранение карт неопределенности включено")
                        elif 'uncertainty' in output_dirs:
                            del output_dirs['uncertainty']
                            print("Сохранение карт неопределенности выключено")
                    
                    # Команда метаданных
                    elif cmd == 'метаданные':
                        value = parts[1].lower()
                        save_metadata = value in ['вкл', 'on', 'true', '1', 'да', 'yes']
                        
                        if save_metadata and 'metadata' not in output_dirs:
                            output_dirs['metadata'] = os.path.join(args.output_dir, 'metadata')
                            os.makedirs(output_dirs['metadata'], exist_ok=True)
                            print("Сохранение метаданных включено")
                        elif not save_metadata and 'metadata' in output_dirs:
                            del output_dirs['metadata']
                            print("Сохранение метаданных выключено")
                    
                    # Команда суффикса
                    elif cmd == 'суффикс':
                        suffix = ' '.join(parts[1:])
                        config['output'] = config.get('output', {})
                        config['output']['suffix'] = suffix
                        print(f"Установлен суффикс: {suffix}")
                    
                    else:
                        print("Неизвестная команда. Введите 'помощь' для получения списка команд.")
                else:
                    print("Неверный формат команды. Введите 'помощь' для получения списка команд.")
            else:
                # Обработка структурированных команд
                cmd_type = parsed['command_type']
                params = parsed['params']
                
                # Команда ЦветСтиль
                if cmd_type == 'ЦветСтиль':
                    style_name = params
                    config['style'] = style_name
                    print(f"Установлен стиль: {style_name}")
                
                # Команда НасыщенностьЦвета
                elif cmd_type == 'НасыщенностьЦвета':
                    try:
                        saturation = float(params)
                        config['postprocessing'] = config.get('postprocessing', {})
                        config['postprocessing']['saturation'] = saturation
                        
                        # Обновляем постпроцессор
                        postprocessor.saturation = saturation
                        
                        print(f"Установлена насыщенность: {saturation}")
                    except ValueError:
                        print("Ошибка: значение насыщенности должно быть числом")
                
                # Команда ДеталиЛица
                elif cmd_type == 'ДеталиЛица':
                    level_map = {
                        'низкая': 0.5,
                        'средняя': 1.0,
                        'высокая': 1.5
                    }
                    
                    level = params.lower()
                    if level in level_map:
                        # В реальной реализации здесь должно быть управление детализацией лиц
                        # через интеллектуальные модули
                        print(f"Установлен уровень детализации лиц: {level} ({level_map[level]})")
                    else:
                        print(f"Неизвестный уровень детализации: {level}")
                        print("Доступные уровни: низкая, средняя, высокая")
                
                # Команда СоветыПоЦвету
                elif cmd_type == 'СоветыПоЦвету':
                    # В реальной реализации здесь должно быть управление советами по цвету
                    # через интеллектуальные модули (GuideNet)
                    print(f"Установлены советы по цвету: {params}")
                
                # Неизвестная команда
                else:
                    print(f"Неизвестная команда: {cmd_type}")
        
        except KeyboardInterrupt:
            print("\nРабота прервана пользователем")
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")


def main():
    """
    Основная функция запуска инференса.
    """
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Загрузка конфигураций
    inference_config, model_config = load_configurations(args)
    
    # Настройка окружения
    device = setup_environment(args, inference_config)
    
    # Создание директорий для результатов
    output_dirs = setup_directories(args, inference_config)
    
    try:
        # Загрузка модели из чекпоинта
        print(f"Загрузка модели из чекпоинта: {args.checkpoint}")
        model, intelligent_modules = load_model_from_checkpoint(args.checkpoint, device)
        print("Модель успешно загружена")
        
        # Создание предиктора и постпроцессора
        predictor, postprocessor = create_predictor_and_postprocessor(
            model, inference_config, device, intelligent_modules
        )
        
        # Создание визуализатора
        visualizer = ColorizationVisualizer(output_dir=args.output_dir)
        
        # Модуль оценки неопределенности (если есть в интеллектуальных модулях)
        uncertainty_module = intelligent_modules.get('uncertainty') if intelligent_modules else None
        
        # Выбор режима работы
        if args.interactive:
            print("Запуск в интерактивном режиме")
            interactive_mode(
                args, inference_config, predictor, postprocessor, 
                output_dirs, visualizer, uncertainty_module
            )
        elif args.single:
            print(f"Обработка одиночного изображения: {args.single}")
            result = process_single_image(
                args, inference_config, args.single, predictor, postprocessor,
                output_dirs, visualizer, uncertainty_module
            )
            if result:
                print("Обработка завершена успешно")
            else:
                print("Ошибка при обработке изображения")
        elif args.batch:
            print(f"Пакетная обработка директории: {args.batch}")
            count = process_batch(
                args, inference_config, args.batch, predictor, postprocessor,
                output_dirs, visualizer, uncertainty_module
            )
            print(f"Успешно обработано изображений: {count}")
        elif args.queue:
            print(f"Запуск в режиме очереди: {args.queue}")
            process_queue(
                args, inference_config, args.queue, predictor, postprocessor,
                output_dirs, visualizer, uncertainty_module
            )
            
    except KeyboardInterrupt:
        print("Работа прервана пользователем")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()