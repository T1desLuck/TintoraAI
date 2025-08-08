#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Скрипт оценки производительности модели колоризации

Данный скрипт предназначен для всесторонней оценки качества и производительности 
модели колоризации. Он выполняет колоризацию тестовых изображений, вычисляет 
различные метрики качества (PSNR, SSIM, LPIPS и др.), генерирует визуализации 
результатов и создает подробные отчеты о производительности.

Возможности:
- Количественная оценка качества колоризации с использованием множества метрик
- Тестирование на различных наборах данных для объективной оценки
- Визуализация результатов и создание сравнительных таблиц
- Анализ производительности по времени и потреблению ресурсов
- Детальный анализ по категориям изображений
- Экспорт результатов в различные форматы (CSV, JSON, HTML)
"""

import os
import sys
import time
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
from tabulate import tabulate

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

from training.checkpoints import load_checkpoint

from datasets.validation_dataset import create_validation_dataset, create_validation_dataloader


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Оценка качества модели колоризации",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Основные параметры
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Путь к файлу конфигурации инференса")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                        help="Путь к файлу конфигурации модели")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к файлу чекпоинта модели")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation",
                        help="Директория для сохранения результатов оценки")
    
    # Параметры входных данных
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--test-dir", type=str,
                            help="Путь к директории с тестовыми данными (парные изображения)")
    data_group.add_argument("--grayscale-dir", type=str,
                            help="Путь к директории с черно-белыми тестовыми изображениями")
    data_group.add_argument("--color-dir", type=str,
                            help="Путь к директории с цветными тестовыми изображениями")
    
    # Параметры оценки
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Размер батча для инференса")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Размер изображения для обработки")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Максимальное количество изображений для оценки (None = все)")
    parser.add_argument("--color-space", type=str, default="lab",
                        choices=["lab", "rgb", "yuv"],
                        help="Цветовое пространство для колоризации")
    
    # Параметры метрик
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,colorfulness,fid",
                        help="Метрики для оценки, разделенные запятыми")
    parser.add_argument("--fid-stats", type=str, default=None,
                        help="Путь к предварительно вычисленным статистикам для FID")
    
    # Параметры визуализации и отчета
    parser.add_argument("--save-images", action="store_true", default=False,
                        help="Сохранять колоризованные изображения")
    parser.add_argument("--save-comparison", action="store_true", default=True,
                        help="Сохранять сравнение до/после")
    parser.add_argument("--generate-html", action="store_true", default=True,
                        help="Генерировать HTML-отчет с результатами")
    parser.add_argument("--plot-metrics", action="store_true", default=True,
                        help="Создавать графики для метрик")
    
    # Параметры анализа производительности
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Выполнять профилирование производительности")
    parser.add_argument("--measure-memory", action="store_true", default=False,
                        help="Измерять использование памяти")
    parser.add_argument("--measure-latency", action="store_true", default=False,
                        help="Измерять задержку обработки")
    
    # Дополнительные параметры
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Использовать GPU для оценки")
    parser.add_argument("--categories", type=str, default=None,
                        help="Путь к JSON-файлу с категориями изображений")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Выводить подробную информацию о процессе")
    
    return parser.parse_args()


def setup_environment(args):
    """
    Настройка окружения для оценки.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        torch.device: Устройство для выполнения расчетов
    """
    # Проверяем доступность GPU
    use_gpu = args.gpu is not None and args.gpu
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if args.verbose:
            print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU недоступен, используется CPU")
        device = torch.device("cpu")
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Настраиваем подкаталоги для сохранения результатов
    results_dir = os.path.join(args.output_dir, 'results')
    plots_dir = os.path.join(args.output_dir, 'plots')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    reports_dir = os.path.join(args.output_dir, 'reports')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Возвращаем устройство и пути
    output_dirs = {
        'results': results_dir,
        'plots': plots_dir,
        'comparisons': comparisons_dir,
        'reports': reports_dir
    }
    
    return device, output_dirs


def load_model(args, device):
    """
    Загрузка модели из чекпоинта.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для размещения модели
        
    Returns:
        tuple: (model, predictor, postprocessor) - модель, предиктор и постпроцессор
    """
    # Загружаем конфигурации
    config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    if config is None:
        print("Не удалось загрузить конфигурацию инференса, используются значения по умолчанию")
        config = {}
    
    if model_config is None:
        print("Не удалось загрузить конфигурацию модели, используются значения по умолчанию")
        model_config = {}
    
    # Загружаем модель из чекпоинта
    print(f"Загрузка модели из чекпоинта: {args.checkpoint}")
    
    try:
        # Загрузка чекпоинта
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model' in checkpoint:
            # Чекпоинт содержит готовую модель
            model = checkpoint['model'].to(device)
            print("Модель загружена напрямую из чекпоинта")
        elif 'model_state_dict' in checkpoint:
            # Чекпоинт содержит состояние модели
            from training.checkpoints import load_model_from_checkpoint
            model = load_model_from_checkpoint(args.checkpoint, model_config, device)
            print("Модель загружена из state_dict")
        else:
            raise ValueError("Некорректный формат чекпоинта")
        
        # Извлечение интеллектуальных модулей из чекпоинта, если они есть
        intelligent_modules = {}
        if 'modules_state_dict' in checkpoint:
            # TODO: Логика восстановления интеллектуальных модулей
            # В реальной реализации здесь будет код для загрузки модулей
            pass
            
        # Создаем предиктор и постпроцессор
        predictor = ColorizationPredictor(
            model=model,
            device=device,
            color_space=args.color_space,
            intelligent_modules=intelligent_modules
        )
        
        postprocessor = ColorizationPostProcessor(
            color_space=args.color_space,
            apply_enhancement=False,  # Для оценки не применяем улучшение
            device=device
        )
        
        # Переводим модель в режим оценки
        model.eval()
        
        return model, predictor, postprocessor
    
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        raise


def load_test_data(args):
    """
    Загрузка тестовых данных для оценки.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        tuple: (dataloader, dataset) - загрузчик и датасет для тестовых данных
    """
    # Определяем пути к данным
    if args.test_dir:
        # Предполагаем, что в test_dir есть подпапки grayscale и color
        grayscale_dir = os.path.join(args.test_dir, 'grayscale')
        color_dir = os.path.join(args.test_dir, 'color')
        
        if not os.path.exists(grayscale_dir) or not os.path.exists(color_dir):
            # Если нет структуры с подпапками, используем основную папку как источник
            grayscale_dir = args.test_dir
            color_dir = args.test_dir
    else:
        grayscale_dir = args.grayscale_dir
        color_dir = args.color_dir
    
    # Проверяем, что директории существуют
    if not os.path.exists(grayscale_dir):
        raise FileNotFoundError(f"Директория с черно-белыми изображениями не существует: {grayscale_dir}")
    
    if color_dir and not os.path.exists(color_dir):
        raise FileNotFoundError(f"Директория с цветными изображениями не существует: {color_dir}")
    
    # Создаем датасет для тестирования
    dataset = create_validation_dataset(
        data_root=os.path.dirname(grayscale_dir),
        grayscale_dir=os.path.basename(grayscale_dir),
        color_dir=os.path.basename(color_dir) if color_dir else None,
        color_space=args.color_space,
        img_size=args.img_size,
        max_dataset_size=args.num_images
    )
    
    # Создаем загрузчик данных
    dataloader = create_validation_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Загружено тестовых изображений: {len(dataset)}")
    
    return dataloader, dataset


def create_metrics_calculator(args, device):
    """
    Создание калькулятора метрик для оценки качества колоризации.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для расчетов
        
    Returns:
        MetricsCalculator: Калькулятор метрик
    """
    # Парсим список метрик
    metrics_list = [m.strip().lower() for m in args.metrics.split(',')]
    
    # Создаем калькулятор метрик
    metrics_calc = MetricsCalculator(
        metrics=metrics_list,
        device=device,
        fid_stats_path=args.fid_stats
    )
    
    return metrics_calc


def evaluate_model(args, model, predictor, postprocessor, dataloader, metrics_calc, output_dirs, device):
    """
    Оценка качества модели на тестовом датасете.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        model (nn.Module): Модель для оценки
        predictor (ColorizationPredictor): Предиктор
        postprocessor (ColorizationPostProcessor): Постпроцессор
        dataloader (DataLoader): Загрузчик данных
        metrics_calc (MetricsCalculator): Калькулятор метрик
        output_dirs (dict): Директории для сохранения результатов
        device (torch.device): Устройство для расчетов
        
    Returns:
        tuple: (results_df, metrics_summary) - DataFrame с результатами и сводка метрик
    """
    # Создаем визуализатор
    visualizer = ColorizationVisualizer(output_dir=output_dirs['results'])
    
    # Готовимся к сбору результатов
    results = []
    all_metrics = {}
    
    # Засекаем время начала оценки
    evaluation_start_time = time.time()
    
    # Сбор метрик по категориям, если указан файл категорий
    categories_data = {}
    if args.categories:
        try:
            with open(args.categories, 'r') as f:
                categories_data = json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке файла категорий: {str(e)}")
            categories_data = {}
    
    # Итерация по тестовым данным с отображением прогресса
    print("\nОценка модели на тестовых данных:")
    
    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Обработка изображений")):
            # Загружаем изображения на устройство
            grayscale = batch['grayscale'].to(device)
            target_color = batch['color'].to(device) if 'color' in batch else None
            
            # Измеряем время инференса для этого батча
            batch_start_time = time.time()
            
            # Выполняем колоризацию
            outputs = model(grayscale)
            
            # Объединяем L-канал с предсказанными a и b каналами
            predicted_lab = torch.cat([grayscale, outputs['a'], outputs['b']], dim=1)
            
            # Применяем постобработку
            postprocessed_images = []
            for i in range(grayscale.size(0)):
                # Извлекаем изображение из батча
                img = predicted_lab[i].cpu().numpy().transpose(1, 2, 0)
                
                # Применяем постобработку
                img = postprocessor.process_image(img)
                
                postprocessed_images.append(img)
            
            # Измеряем время, затраченное на инференс
            batch_inference_time = time.time() - batch_start_time
            inference_time_per_image = batch_inference_time / grayscale.size(0)
            
            # Вычисляем метрики, если есть эталонные цветные изображения
            batch_metrics = {}
            if target_color is not None:
                batch_metrics = metrics_calc.calculate(predicted_lab, target_color)
                
                # Добавляем метрики в общий словарь
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].extend([value] * grayscale.size(0))
            
            # Обрабатываем каждое изображение в батче
            for i in range(grayscale.size(0)):
                # Получаем информацию об изображении
                image_id = batch['id'][i] if 'id' in batch else f"img_{batch_idx}_{i}"
                
                # Определяем категорию изображения, если доступно
                category = 'unknown'
                if categories_data and image_id in categories_data:
                    category = categories_data[image_id]
                
                # Результаты для этого изображения
                image_results = {
                    'id': image_id,
                    'category': category,
                    'inference_time': inference_time_per_image,
                }
                
                # Добавляем метрики, если они были вычислены
                if target_color is not None:
                    for key, value in batch_metrics.items():
                        if isinstance(value, torch.Tensor) and value.numel() > 1:
                            image_results[key] = value[i].item()
                        else:
                            image_results[key] = value
                
                # Сохраняем результаты
                results.append(image_results)
                
                # Сохраняем изображения и сравнения, если нужно
                if args.save_images:
                    # Сохраняем колоризованное изображение
                    colorized_img = postprocessed_images[i]
                    colorized_path = os.path.join(output_dirs['results'], f"{image_id}.png")
                    
                    # Преобразуем в формат PIL для сохранения
                    colorized_pil = Image.fromarray((colorized_img * 255).astype(np.uint8))
                    colorized_pil.save(colorized_path)
                
                if args.save_comparison and target_color is not None:
                    # Создаем сравнение до/после
                    comparison_path = os.path.join(output_dirs['comparisons'], f"{image_id}_comparison.png")
                    
                    # Извлекаем исходное и целевое изображения
                    grayscale_np = grayscale[i].cpu().numpy().transpose(1, 2, 0)
                    target_np = target_color[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Создаем и сохраняем сравнение
                    visualizer.create_comparison(
                        grayscale=grayscale_np,
                        colorized=postprocessed_images[i],
                        original=target_np,
                        filename=comparison_path
                    )
    
    # Общее время оценки
    total_evaluation_time = time.time() - evaluation_start_time
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Вычисляем сводную статистику по метрикам
    metrics_summary = {}
    
    for key in all_metrics:
        values = all_metrics[key]
        metrics_summary[key] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Добавляем информацию о времени обработки
    metrics_summary['time'] = {
        'total_evaluation_time': total_evaluation_time,
        'average_inference_time': np.mean(results_df['inference_time']),
        'median_inference_time': np.median(results_df['inference_time']),
        'images_per_second': len(results_df) / total_evaluation_time
    }
    
    return results_df, metrics_summary


def generate_category_report(results_df, metrics_summary, output_dir):
    """
    Генерация отчета по категориям изображений.
    
    Args:
        results_df (pd.DataFrame): DataFrame с результатами оценки
        metrics_summary (dict): Сводная информация по метрикам
        output_dir (str): Директория для сохранения отчета
        
    Returns:
        dict: Статистика по категориям
    """
    # Проверяем, что у нас есть колонка с категориями
    if 'category' not in results_df.columns:
        print("Информация о категориях не найдена, отчет по категориям не создан")
        return {}
    
    # Группируем данные по категориям
    category_groups = results_df.groupby('category')
    
    # Создаем статистику по категориям
    category_stats = {}
    
    for category, group in category_groups:
        # Собираем метрики для этой категории
        cat_metrics = {}
        
        # Перебираем все метрики, которые есть в DataFrame
        for col in group.columns:
            if col not in ['id', 'category', 'inference_time']:
                cat_metrics[col] = {
                    'mean': group[col].mean(),
                    'median': group[col].median(),
                    'std': group[col].std(),
                    'min': group[col].min(),
                    'max': group[col].max(),
                    'count': len(group)
                }
        
        # Добавляем статистику по времени
        cat_metrics['inference_time'] = {
            'mean': group['inference_time'].mean(),
            'median': group['inference_time'].median(),
            'std': group['inference_time'].std()
        }
        
        category_stats[category] = cat_metrics
    
    # Создаем таблицу сравнения категорий
    comparison_data = []
    metrics_to_compare = [col for col in results_df.columns if col not in ['id', 'category']]
    
    for category in category_stats:
        row = {'category': category, 'count': len(category_groups.get_group(category))}
        
        for metric in metrics_to_compare:
            if metric in category_stats[category]:
                row[f"{metric}_mean"] = category_stats[category][metric]['mean']
                row[f"{metric}_std"] = category_stats[category][metric]['std']
        
        comparison_data.append(row)
    
    # Создаем DataFrame с данными для сравнения
    comparison_df = pd.DataFrame(comparison_data)
    
    # Сохраняем в CSV
    comparison_df.to_csv(os.path.join(output_dir, 'category_comparison.csv'), index=False)
    
    return category_stats


def generate_plots(results_df, metrics_summary, category_stats, output_dirs):
    """
    Генерация графиков для визуализации результатов оценки.
    
    Args:
        results_df (pd.DataFrame): DataFrame с результатами оценки
        metrics_summary (dict): Сводная информация по метрикам
        category_stats (dict): Статистика по категориям
        output_dirs (dict): Директории для сохранения результатов
    """
    # Настраиваем внешний вид графиков
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Директория для сохранения графиков
    plots_dir = output_dirs['plots']
    
    # 1. Гистограммы метрик
    for metric in metrics_summary:
        if metric == 'time':
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Получаем данные для метрики
        metric_values = results_df[metric]
        
        # Строим гистограмму
        sns.histplot(metric_values, kde=True)
        plt.title(f'Распределение метрики {metric.upper()}', fontsize=14)
        plt.xlabel(metric.upper())
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        
        # Добавляем статистические показатели
        mean_val = metrics_summary[metric]['mean']
        median_val = metrics_summary[metric]['median']
        std_val = metrics_summary[metric]['std']
        
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Среднее: {mean_val:.3f}')
        plt.axvline(median_val, color='g', linestyle='-', label=f'Медиана: {median_val:.3f}')
        
        plt.legend()
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(os.path.join(plots_dir, f'{metric}_histogram.png'), dpi=150)
        plt.close()
    
    # 2. Сравнение категорий (если они есть)
    if category_stats:
        for metric in metrics_summary:
            if metric == 'time':
                continue
                
            # Собираем данные для всех категорий
            category_values = []
            category_names = []
            
            for category, stats in category_stats.items():
                if metric in stats:
                    category_values.append(stats[metric]['mean'])
                    category_names.append(category)
            
            if category_values:
                # Сортируем категории по среднему значению метрики
                sorted_indices = np.argsort(category_values)
                category_values = [category_values[i] for i in sorted_indices]
                category_names = [category_names[i] for i in sorted_indices]
                
                plt.figure(figsize=(12, 6))
                
                # Строим столбчатую диаграмму
                bars = plt.bar(category_names, category_values)
                
                # Подписываем значения над столбцами
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.title(f'Сравнение средних значений {metric.upper()} по категориям', fontsize=14)
                plt.ylabel(metric.upper())
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Сохраняем график
                plt.savefig(os.path.join(plots_dir, f'{metric}_by_category.png'), dpi=150)
                plt.close()
    
    # 3. График времени инференса
    plt.figure(figsize=(10, 6))
    
    sns.histplot(results_df['inference_time'], kde=True)
    plt.title('Распределение времени инференса', fontsize=14)
    plt.xlabel('Время (секунды)')
    plt.ylabel('Частота')
    
    mean_time = results_df['inference_time'].mean()
    median_time = results_df['inference_time'].median()
    
    plt.axvline(mean_time, color='r', linestyle='--', label=f'Среднее: {mean_time:.4f} сек')
    plt.axvline(median_time, color='g', linestyle='-', label=f'Медиана: {median_time:.4f} сек')
    
    plt.legend()
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig(os.path.join(plots_dir, 'inference_time_histogram.png'), dpi=150)
    plt.close()


def generate_html_report(results_df, metrics_summary, category_stats, output_dirs):
    """
    Генерация HTML-отчета с результатами оценки.
    
    Args:
        results_df (pd.DataFrame): DataFrame с результатами оценки
        metrics_summary (dict): Сводная информация по метрикам
        category_stats (dict): Статистика по категориям
        output_dirs (dict): Директории для сохранения результатов
    """
    # Директория для сохранения отчета
    reports_dir = output_dirs['reports']
    plots_dir = output_dirs['plots']
    comparisons_dir = output_dirs['comparisons']
    
    # Подготовка HTML-шаблона
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TintoraAI - Отчет об оценке модели</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .metric-card {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
            }
            .metric-title {
                font-size: 18px;
                margin-top: 0;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .gallery {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .gallery-item {
                width: calc(33.33% - 10px);
                margin-bottom: 15px;
            }
            .gallery-item img {
                width: 100%;
                border-radius: 4px;
            }
            .chart-container {
                margin-bottom: 30px;
            }
            .chart {
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
            }
            .tabs {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                margin-bottom: 20px;
            }
            .tab-button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }
            .tab-button:hover {
                background-color: #ddd;
            }
            .tab-button.active {
                background-color: #ccc;
            }
            .tab-content {
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TintoraAI - Отчет об оценке модели колоризации</h1>
            <p>Дата и время: {timestamp}</p>
            
            <div class="tabs">
                <button class="tab-button active" onclick="openTab(event, 'summary')">Сводка</button>
                <button class="tab-button" onclick="openTab(event, 'metrics')">Метрики</button>
                <button class="tab-button" onclick="openTab(event, 'categories')">Категории</button>
                <button class="tab-button" onclick="openTab(event, 'visualizations')">Визуализации</button>
                <button class="tab-button" onclick="openTab(event, 'performance')">Производительность</button>
            </div>
            
            <div id="summary" class="tab-content" style="display: block;">
                <h2>Общая сводка</h2>
                <div class="metric-cards">
                    {summary_cards}
                </div>
                <h3>Сводная таблица метрик</h3>
                {summary_table}
            </div>
            
            <div id="metrics" class="tab-content">
                <h2>Детальный анализ метрик</h2>
                <p>Распределение значений основных метрик качества колоризации.</p>
                <div class="chart-container">
                    {metrics_charts}
                </div>
            </div>
            
            <div id="categories" class="tab-content">
                <h2>Анализ по категориям</h2>
                {categories_content}
            </div>
            
            <div id="visualizations" class="tab-content">
                <h2>Визуализация результатов</h2>
                <p>Примеры колоризации изображений, сравнение с оригиналами.</p>
                <div class="gallery">
                    {gallery_content}
                </div>
            </div>
            
            <div id="performance" class="tab-content">
                <h2>Оценка производительности</h2>
                <h3>Время инференса</h3>
                <div class="chart">
                    <img src="../plots/inference_time_histogram.png" alt="Время инференса">
                </div>
                <h3>Статистика производительности</h3>
                {performance_table}
            </div>
            
            <footer>
                <p>TintoraAI - Система колоризации изображений</p>
            </footer>
        </div>
        
        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tab-button");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
        </script>
    </body>
    </html>
    """
    
    # Подготовка карточек для сводки
    summary_cards_html = ""
    metrics_to_show = [m for m in metrics_summary.keys() if m != 'time']
    
    for metric in metrics_to_show:
        mean_value = metrics_summary[metric]['mean']
        summary_cards_html += f"""
        <div class="metric-card">
            <h3 class="metric-title">{metric.upper()}</h3>
            <div class="metric-value">{mean_value:.4f}</div>
            <div>Медиана: {metrics_summary[metric]['median']:.4f}</div>
            <div>Стд. откл.: {metrics_summary[metric]['std']:.4f}</div>
        </div>
        """
    
    # Подготовка сводной таблицы метрик
    summary_table_html = """
    <table>
        <thead>
            <tr>
                <th>Метрика</th>
                <th>Среднее</th>
                <th>Медиана</th>
                <th>Стд. откл.</th>
                <th>Мин.</th>
                <th>Макс.</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for metric in metrics_to_show:
        stats = metrics_summary[metric]
        summary_table_html += f"""
            <tr>
                <td>{metric.upper()}</td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['median']:.4f}</td>
                <td>{stats['std']:.4f}</td>
                <td>{stats['min']:.4f}</td>
                <td>{stats['max']:.4f}</td>
            </tr>
        """
    
    summary_table_html += """
        </tbody>
    </table>
    """
    
    # Подготовка графиков метрик
    metrics_charts_html = ""
    
    for metric in metrics_to_show:
        metrics_charts_html += f"""
        <div class="chart">
            <img src="../plots/{metric}_histogram.png" alt="{metric} распределение">
        </div>
        """
    
    # Подготовка контента по категориям
    categories_content = "<p>Информация о категориях не доступна.</p>"
    
    if category_stats:
        categories_content = """
        <table>
            <thead>
                <tr>
                    <th>Категория</th>
                    <th>Кол-во</th>
        """
        
        # Добавляем заголовки для всех метрик
        for metric in metrics_to_show:
            categories_content += f"<th>{metric.upper()}</th>"
            
        categories_content += """
                </tr>
            </thead>
            <tbody>
        """
        
        for category, stats in category_stats.items():
            categories_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{stats[list(stats.keys())[0]]['count']}</td>
            """
            
            for metric in metrics_to_show:
                if metric in stats:
                    categories_content += f"<td>{stats[metric]['mean']:.4f}</td>"
                else:
                    categories_content += "<td>N/A</td>"
                    
            categories_content += "</tr>"
            
        categories_content += """
            </tbody>
        </table>
        
        <h3>Графики по категориям</h3>
        <div class="chart-container">
        """
        
        for metric in metrics_to_show:
            if os.path.exists(os.path.join(plots_dir, f'{metric}_by_category.png')):
                categories_content += f"""
                <div class="chart">
                    <img src="../plots/{metric}_by_category.png" alt="{metric} по категориям">
                </div>
                """
                
        categories_content += "</div>"
    
    # Подготовка галереи изображений
    gallery_content = ""
    
    # Ищем файлы с результатами сравнения
    comparison_files = glob.glob(os.path.join(comparisons_dir, "*_comparison.png"))
    comparison_files = comparison_files[:12]  # Ограничиваем количество для отображения
    
    for comparison_file in comparison_files:
        file_name = os.path.basename(comparison_file)
        gallery_content += f"""
        <div class="gallery-item">
            <img src="../comparisons/{file_name}" alt="{file_name}">
            <p>{file_name}</p>
        </div>
        """
    
    # Подготовка таблицы производительности
    time_stats = metrics_summary['time']
    performance_table_html = f"""
    <table>
        <tbody>
            <tr>
                <td>Общее время оценки</td>
                <td>{time_stats['total_evaluation_time']:.2f} сек</td>
            </tr>
            <tr>
                <td>Среднее время инференса</td>
                <td>{time_stats['average_inference_time']:.4f} сек/изображение</td>
            </tr>
            <tr>
                <td>Медианное время инференса</td>
                <td>{time_stats['median_inference_time']:.4f} сек/изображение</td>
            </tr>
            <tr>
                <td>Производительность</td>
                <td>{time_stats['images_per_second']:.2f} изображений/сек</td>
            </tr>
        </tbody>
    </table>
    """
    
    # Заполняем шаблон
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_cards=summary_cards_html,
        summary_table=summary_table_html,
        metrics_charts=metrics_charts_html,
        categories_content=categories_content,
        gallery_content=gallery_content,
        performance_table=performance_table_html
    )
    
    # Сохраняем HTML-отчет
    html_path = os.path.join(reports_dir, "evaluation_report.html")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML-отчет сохранен в {html_path}")


def measure_performance(args, model, dataloader, device):
    """
    Измерение производительности модели.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        model (nn.Module): Модель для оценки
        dataloader (DataLoader): Загрузчик данных
        device (torch.device): Устройство для расчетов
        
    Returns:
        dict: Результаты измерения производительности
    """
    print("\nИзмерение производительности модели...")
    
    performance_data = {
        'inference_time': [],
        'memory_usage': []
    }
    
    # Прогрев модели
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Обрабатываем только первые два батча для прогрева
                break
            
            grayscale = batch['grayscale'].to(device)
            _ = model(grayscale)
    
    # Сбор данных о производительности
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Измерение производительности")):
            grayscale = batch['grayscale'].to(device)
            
            # Измеряем время инференса
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            _ = model(grayscale)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            # Записываем время инференса (в миллисекундах)
            inference_time = (end_time - start_time) * 1000
            performance_data['inference_time'].append(inference_time)
            
            # Измеряем использование памяти, если нужно
            if args.measure_memory and device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # В МБ
                performance_data['memory_usage'].append(memory_used)
                
                # Сбрасываем статистику использования памяти
                torch.cuda.reset_peak_memory_stats()
    
    # Анализ результатов измерений
    if performance_data['inference_time']:
        avg_inference_time = np.mean(performance_data['inference_time'])
        std_inference_time = np.std(performance_data['inference_time'])
        
        print(f"Среднее время инференса: {avg_inference_time:.2f} мс (±{std_inference_time:.2f} мс)")
        print(f"Производительность: {1000 / avg_inference_time:.2f} изображений/сек")
    
    if performance_data['memory_usage']:
        avg_memory_usage = np.mean(performance_data['memory_usage'])
        max_memory_usage = np.max(performance_data['memory_usage'])
        
        print(f"Среднее использование памяти: {avg_memory_usage:.2f} МБ")
        print(f"Максимальное использование памяти: {max_memory_usage:.2f} МБ")
    
    return performance_data


def save_results(results_df, metrics_summary, category_stats, performance_data, output_dirs):
    """
    Сохранение результатов оценки в различных форматах.
    
    Args:
        results_df (pd.DataFrame): DataFrame с результатами оценки
        metrics_summary (dict): Сводная информация по метрикам
        category_stats (dict): Статистика по категориям
        performance_data (dict): Данные о производительности
        output_dirs (dict): Директории для сохранения результатов
    """
    # Сохранение детальных результатов в CSV
    results_df.to_csv(os.path.join(output_dirs['reports'], 'detailed_results.csv'), index=False)
    
    # Сохранение сводки метрик в JSON
    metrics_json = {}
    
    for metric, stats in metrics_summary.items():
        if isinstance(stats, dict):
            metrics_json[metric] = {k: float(v) for k, v in stats.items()}
    
    with open(os.path.join(output_dirs['reports'], 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Сохранение статистики по категориям в JSON
    if category_stats:
        # Преобразуем значения numpy в обычные Python-типы
        category_json = {}
        
        for category, stats in category_stats.items():
            category_json[category] = {}
            
            for metric, values in stats.items():
                category_json[category][metric] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                                   for k, v in values.items()}
        
        with open(os.path.join(output_dirs['reports'], 'category_stats.json'), 'w') as f:
            json.dump(category_json, f, indent=2)
    
    # Сохранение данных о производительности в JSON
    if performance_data:
        performance_json = {
            'inference_time_ms': {
                'data': [float(x) for x in performance_data['inference_time']],
                'mean': float(np.mean(performance_data['inference_time'])),
                'std': float(np.std(performance_data['inference_time'])),
                'min': float(np.min(performance_data['inference_time'])),
                'max': float(np.max(performance_data['inference_time']))
            }
        }
        
        if performance_data['memory_usage']:
            performance_json['memory_usage_mb'] = {
                'data': [float(x) for x in performance_data['memory_usage']],
                'mean': float(np.mean(performance_data['memory_usage'])),
                'max': float(np.max(performance_data['memory_usage']))
            }
            
        with open(os.path.join(output_dirs['reports'], 'performance_data.json'), 'w') as f:
            json.dump(performance_json, f, indent=2)
    
    # Создание README для директории с результатами
    readme_content = """# TintoraAI - Результаты оценки модели колоризации

## Содержимое директории

- `reports/` - Отчеты и анализ результатов
  - `evaluation_report.html` - HTML-отчет с результатами оценки
  - `detailed_results.csv` - Детальные результаты по каждому изображению
  - `metrics_summary.json` - Сводная статистика по метрикам
  - `category_stats.json` - Статистика по категориям изображений
  - `performance_data.json` - Данные о производительности модели

- `plots/` - Визуализация результатов
  - `*_histogram.png` - Гистограммы распределения метрик
  - `*_by_category.png` - Сравнение метрик по категориям

- `comparisons/` - Сравнения оригинальных и колоризованных изображений

- `results/` - Результаты колоризации тестовых изображений

## Метрики оценки

"""
    
    # Добавляем таблицу с метриками
    metrics_table = []
    headers = ["Метрика", "Среднее", "Медиана", "Стд. откл.", "Мин.", "Макс."]
    metrics_table.append(headers)
    
    for metric in metrics_summary:
        if metric == 'time':
            continue
        
        stats = metrics_summary[metric]
        row = [
            metric.upper(),
            f"{stats['mean']:.4f}",
            f"{stats['median']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}"
        ]
        metrics_table.append(row)
    
    readme_content += tabulate(metrics_table, headers="firstrow", tablefmt="pipe")
    
    # Добавляем информацию о производительности
    readme_content += "\n\n## Производительность\n\n"
    
    if 'time' in metrics_summary:
        time_stats = metrics_summary['time']
        
        readme_content += f"- Среднее время инференса: {time_stats['average_inference_time']:.4f} сек/изображение\n"
        readme_content += f"- Медианное время инференса: {time_stats['median_inference_time']:.4f} сек/изображение\n"
        readme_content += f"- Производительность: {time_stats['images_per_second']:.2f} изображений/сек\n"
    
    # Сохраняем README
    with open(os.path.join(output_dirs['reports'], 'README.md'), 'w') as f:
        f.write(readme_content)


def main():
    """Основная функция оценки."""
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Настройка окружения
    device, output_dirs = setup_environment(args)
    
    try:
        # Загрузка модели
        model, predictor, postprocessor = load_model(args, device)
        
        # Загрузка тестовых данных
        dataloader, dataset = load_test_data(args)
        
        # Создание калькулятора метрик
        metrics_calc = create_metrics_calculator(args, device)
        
        # Измерение производительности
        performance_data = {}
        if args.profile or args.measure_latency or args.measure_memory:
            performance_data = measure_performance(args, model, dataloader, device)
        
        # Оценка модели
        print("\nНачало оценки модели...")
        results_df, metrics_summary = evaluate_model(
            args, model, predictor, postprocessor, dataloader, 
            metrics_calc, output_dirs, device
        )
        
        # Генерация отчета по категориям
        category_stats = generate_category_report(results_df, metrics_summary, output_dirs['reports'])
        
        # Генерация графиков
        if args.plot_metrics:
            generate_plots(results_df, metrics_summary, category_stats, output_dirs)
        
        # Генерация HTML-отчета
        if args.generate_html:
            generate_html_report(results_df, metrics_summary, category_stats, output_dirs)
        
        # Сохранение результатов
        save_results(results_df, metrics_summary, category_stats, performance_data, output_dirs)
        
        # Вывод основных результатов
        print("\n=== Результаты оценки ===")
        for metric, stats in metrics_summary.items():
            if metric == 'time':
                continue
            print(f"{metric.upper()}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        print("\n=== Производительность ===")
        if 'time' in metrics_summary:
            time_stats = metrics_summary['time']
            print(f"Среднее время инференса: {time_stats['average_inference_time']:.4f} сек/изображение")
            print(f"Производительность: {time_stats['images_per_second']:.2f} изображений/сек")
            
        print(f"\nОтчеты и визуализации сохранены в: {args.output_dir}")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())