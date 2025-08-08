#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Интерактивное демо для колоризации изображений

Данный скрипт предоставляет интерактивное демо для демонстрации возможностей 
системы колоризации TintoraAI. Он позволяет выбирать и колоризировать изображения 
в режиме реального времени с возможностью интерактивной настройки параметров колоризации,
выбора различных стилей и применения эффектов.

Демо включает графический интерфейс (с помощью Streamlit) или интерактивный консольный 
режим для более гибкой работы и демонстрации.

Возможности:
- Интерактивная колоризация отдельных изображений
- Демонстрация различных стилей колоризации
- Визуальное сравнение результатов с разными настройками
- Отображение неопределенности колоризации
- Экспорт результатов и настроек
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import threading
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Добавляем корневую директорию проекта в путь поиска
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из модулей проекта
from inference.predictor import ColorizationPredictor
from inference.postprocessor import ColorizationPostProcessor
from utils.config_parser import load_config
from utils.visualization import ColorizationVisualizer
from utils.user_interaction import UserInteractionModule
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.style_transfer import StyleTransfer

# Проверяем наличие Streamlit для веб-интерфейса
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    pass


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Интерактивное демо колоризации изображений",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Основные параметры
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Путь к файлу конфигурации инференса")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                        help="Путь к файлу конфигурации модели")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Путь к файлу чекпоинта модели (обязателен для колоризации)")
    
    # Режимы работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--console", action="store_true", default=False,
                            help="Запустить в консольном интерактивном режиме")
    mode_group.add_argument("--web", action="store_true", default=False,
                            help="Запустить веб-интерфейс с помощью Streamlit")
    mode_group.add_argument("--gui", action="store_true", default=False,
                            help="Запустить графический интерфейс (требуется PyQt5/TkInter)")
    
    # Параметры для демо
    parser.add_argument("--samples-dir", type=str, default="data/samples",
                        help="Путь к директории с примерами изображений")
    parser.add_argument("--output-dir", type=str, default="output/demo",
                        help="Директория для сохранения результатов демо")
    parser.add_argument("--styles-dir", type=str, default="assets/style_presets",
                        help="Директория с пресетами стилей")
    
    # Параметры обработки
    parser.add_argument("--img-size", type=int, default=256,
                        help="Размер изображения для обработки")
    parser.add_argument("--color-space", type=str, default="lab",
                        choices=["lab", "rgb", "yuv"],
                        help="Цветовое пространство для колоризации")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Использовать GPU для инференса")
    
    # Дополнительные параметры
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Выводить подробную информацию о процессе")
    
    return parser.parse_args()


def setup_environment(args):
    """
    Настройка окружения для демо.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        
    Returns:
        Tuple[torch.device, Dict]: (device, output_dirs) - устройство и директории для результатов
    """
    # Определяем устройство для инференса
    use_gpu = args.gpu is not None and args.gpu and torch.cuda.is_available()
    
    if use_gpu:
        device = torch.device("cuda")
        if args.verbose:
            print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if args.verbose:
            print("Используется CPU")
    
    # Создаем директории для результатов
    output_dirs = {
        'colorized': os.path.join(args.output_dir, 'colorized'),
        'comparisons': os.path.join(args.output_dir, 'comparisons'),
        'uncertainty': os.path.join(args.output_dir, 'uncertainty_maps'),
        'metadata': os.path.join(args.output_dir, 'metadata')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return device, output_dirs


def load_model_and_components(args, device):
    """
    Загрузка модели и компонентов для инференса.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для размещения модели
        
    Returns:
        Tuple: (model, predictor, postprocessor, uncertainty_module) - модель и компоненты для инференса
    """
    # Проверяем, указан ли чекпоинт
    if args.checkpoint is None:
        print("Внимание: Чекпоинт модели не указан, будет запущено демо без колоризации")
        return None, None, None, None
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    if config is None:
        config = {}
    if model_config is None:
        model_config = {}
    
    # Загружаем модель
    try:
        print(f"Загрузка модели из чекпоинта: {args.checkpoint}")
        
        # Загрузка чекпоинта
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model' in checkpoint:
            model = checkpoint['model']
            model.to(device)
            print("Модель успешно загружена")
        elif 'model_state_dict' in checkpoint:
            # Импорт функции из модуля checkpoints
            from training.checkpoints import load_model_from_checkpoint
            model = load_model_from_checkpoint(args.checkpoint, model_config, device)
            print("Модель успешно загружена из state_dict")
        else:
            raise ValueError(f"Некорректный формат чекпоинта: {args.checkpoint}")
        
        # Получаем параметры колоризации
        color_space = args.color_space
        
        # Создаем постпроцессор
        postprocessor = ColorizationPostProcessor(
            color_space=color_space,
            apply_enhancement=config.get('postprocessing', {}).get('enhance', False),
            saturation=config.get('postprocessing', {}).get('saturation', 1.0),
            contrast=config.get('postprocessing', {}).get('contrast', 1.0),
            device=device
        )
        
        # Инициализируем интеллектуальные модули
        intelligent_modules = {}
        
        # Создаем модуль оценки неопределенности, если включен в конфигурации
        uncertainty_module = None
        if model_config.get('uncertainty', {}).get('enabled', True):
            uncertainty_config = model_config.get('uncertainty', {})
            uncertainty_module = UncertaintyEstimation(
                method=uncertainty_config.get('method', 'guided'),
                num_samples=uncertainty_config.get('num_samples', 5),
                dropout_rate=uncertainty_config.get('dropout_rate', 0.2),
                device=device
            ).to(device)
            intelligent_modules['uncertainty'] = uncertainty_module
        
        # Создаем предиктор
        predictor = ColorizationPredictor(
            model=model,
            device=device,
            color_space=color_space,
            intelligent_modules=intelligent_modules
        )
        
        return model, predictor, postprocessor, uncertainty_module
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None, None


def load_sample_images(samples_dir):
    """
    Загрузка примеров изображений для демо.
    
    Args:
        samples_dir (str): Путь к директории с примерами
        
    Returns:
        Dict[str, str]: Словарь с именами и путями к примерам изображений
    """
    sample_images = {}
    
    # Проверяем, что директория существует
    if not os.path.exists(samples_dir):
        print(f"Директория с примерами не найдена: {samples_dir}")
        print("Создаем директорию...")
        
        try:
            os.makedirs(samples_dir, exist_ok=True)
            print(f"Директория создана: {samples_dir}")
        except Exception as e:
            print(f"Не удалось создать директорию: {str(e)}")
            return sample_images
    
    # Поддерживаемые форматы изображений
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Ищем изображения в директории
    for file in os.listdir(samples_dir):
        file_path = os.path.join(samples_dir, file)
        
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
            # Получаем имя файла без расширения
            name = os.path.splitext(file)[0]
            sample_images[name] = file_path
    
    return sample_images


def load_style_presets(styles_dir):
    """
    Загрузка пресетов стилей для демо.
    
    Args:
        styles_dir (str): Путь к директории с пресетами стилей
        
    Returns:
        Dict[str, Dict]: Словарь с пресетами стилей
    """
    style_presets = {
        'default': {
            'name': 'Стандартный',
            'description': 'Естественная колоризация с нейтральными цветами',
            'saturation': 1.0,
            'contrast': 1.0,
            'style_image': None
        },
        'vintage': {
            'name': 'Винтаж',
            'description': 'Старинный вид с мягкими пастельными тонами',
            'saturation': 0.7,
            'contrast': 0.9,
            'temperature': 'warm'
        },
        'vivid': {
            'name': 'Яркий',
            'description': 'Насыщенные цвета для яркого впечатления',
            'saturation': 1.3,
            'contrast': 1.1,
            'vibrance': 1.2
        },
        'monochrome': {
            'name': 'Монохромный',
            'description': 'Одноцветная колоризация в выбранной гамме',
            'saturation': 0.5,
            'tint': 'blue',
            'contrast': 1.1
        },
        'cinematic': {
            'name': 'Кинематографический',
            'description': 'Стиль современной киносъемки',
            'saturation': 0.9,
            'contrast': 1.2,
            'color_grade': 'movie'
        }
    }
    
    # Проверяем, что директория существует
    if not os.path.exists(styles_dir):
        print(f"Директория с пресетами стилей не найдена: {styles_dir}")
        return style_presets
    
    # Ищем дополнительные пресеты в виде JSON файлов
    json_files = [f for f in os.listdir(styles_dir) if f.lower().endswith('.json')]
    
    for json_file in json_files:
        try:
            with open(os.path.join(styles_dir, json_file), 'r') as f:
                preset = json.load(f)
                
            if 'name' in preset and 'description' in preset:
                # Получаем имя пресета из имени файла
                preset_name = os.path.splitext(json_file)[0]
                style_presets[preset_name] = preset
        except Exception as e:
            print(f"Ошибка при загрузке пресета {json_file}: {str(e)}")
    
    # Ищем изображения для переноса стиля
    style_images = {}
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Поддиректория для изображений стилей
    styles_images_dir = os.path.join(styles_dir, 'images')
    if os.path.exists(styles_images_dir):
        for file in os.listdir(styles_images_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                # Получаем имя стиля из имени файла
                style_name = os.path.splitext(file)[0]
                style_images[style_name] = os.path.join(styles_images_dir, file)
    
    # Добавляем пути к изображениям стилей в пресеты
    for style_name, style_image_path in style_images.items():
        if style_name in style_presets:
            style_presets[style_name]['style_image'] = style_image_path
    
    return style_presets


class ConsoleDemo:
    """
    Класс для запуска демо в консольном режиме.
    
    Attributes:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для вычислений
        output_dirs (Dict[str, str]): Директории для сохранения результатов
        predictor (ColorizationPredictor): Предиктор для колоризации
        postprocessor (ColorizationPostProcessor): Постпроцессор для обработки результатов
        uncertainty_module (UncertaintyEstimation): Модуль оценки неопределенности
        visualizer (ColorizationVisualizer): Визуализатор результатов
        user_interaction (UserInteractionModule): Модуль взаимодействия с пользователем
        sample_images (Dict[str, str]): Примеры изображений
        style_presets (Dict[str, Dict]): Пресеты стилей
    """
    
    def __init__(self, args, device, output_dirs, predictor, postprocessor, uncertainty_module):
        """
        Инициализация консольного демо.
        
        Args:
            args (argparse.Namespace): Аргументы командной строки
            device (torch.device): Устройство для вычислений
            output_dirs (Dict[str, str]): Директории для сохранения результатов
            predictor (ColorizationPredictor): Предиктор для колоризации
            postprocessor (ColorizationPostProcessor): Постпроцессор для обработки результатов
            uncertainty_module (UncertaintyEstimation): Модуль оценки неопределенности
        """
        self.args = args
        self.device = device
        self.output_dirs = output_dirs
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.uncertainty_module = uncertainty_module
        
        # Загружаем примеры и пресеты стилей
        self.sample_images = load_sample_images(args.samples_dir)
        self.style_presets = load_style_presets(args.styles_dir)
        
        # Инициализируем визуализатор и модуль взаимодействия
        self.visualizer = ColorizationVisualizer(output_dir=output_dirs['comparisons'])
        self.user_interaction = UserInteractionModule()
        
        # Устанавливаем текущие параметры
        self.current_image_path = None
        self.current_style = 'default'
        self.current_saturation = self.style_presets['default']['saturation']
        self.current_contrast = self.style_presets['default']['contrast']
        self.show_uncertainty = False
        
        # Статус
        self.running = True
    
    def print_welcome(self):
        """Выводит приветственное сообщение и инструкции."""
        print("\n" + "=" * 60)
        print("TintoraAI - Интерактивное демо колоризации".center(60))
        print("=" * 60)
        print("\nКоманды:")
        print("  помощь - Показать список команд")
        print("  выход - Завершить демо")
        print("  образцы - Показать список доступных образцов")
        print("  стили - Показать список доступных стилей")
        print("  загрузить <путь> - Загрузить изображение по пути")
        print("  образец <имя> - Выбрать образец из списка")
        print("  стиль <имя> - Применить стиль из списка")
        print("  насыщенность <значение> - Установить насыщенность (0.0-2.0)")
        print("  контраст <значение> - Установить контраст (0.0-2.0)")
        print("  неопределенность <вкл/выкл> - Показывать карту неопределенности")
        print("  колоризовать - Колоризировать текущее изображение")
        print("  сохранить <имя> - Сохранить результат колоризации")
        print("  статус - Показать текущие настройки")
        print("\nНачните с выбора образца или загрузки изображения.")
        print("=" * 60 + "\n")
    
    def print_status(self):
        """Выводит текущие настройки."""
        print("\n=== Текущие настройки ===")
        print(f"Изображение: {os.path.basename(self.current_image_path) if self.current_image_path else 'не выбрано'}")
        print(f"Стиль: {self.style_presets[self.current_style]['name']}")
        print(f"Насыщенность: {self.current_saturation}")
        print(f"Контраст: {self.current_contrast}")
        print(f"Показывать неопределенность: {'Да' if self.show_uncertainty else 'Нет'}")
    
    def print_samples(self):
        """Выводит список доступных образцов."""
        if not self.sample_images:
            print("Образцы не найдены.")
            return
            
        print("\n=== Доступные образцы ===")
        for i, (name, path) in enumerate(self.sample_images.items(), 1):
            print(f"{i}. {name} ({os.path.basename(path)})")
    
    def print_styles(self):
        """Выводит список доступных стилей."""
        print("\n=== Доступные стили ===")
        for i, (style_id, style) in enumerate(self.style_presets.items(), 1):
            print(f"{i}. {style['name']} - {style['description']}")
    
    def colorize_image(self):
        """
        Колоризирует текущее изображение.
        
        Returns:
            Dict: Результат колоризации или None в случае ошибки
        """
        if not self.current_image_path:
            print("Ошибка: изображение не выбрано.")
            return None
            
        if not self.predictor:
            print("Ошибка: модель не загружена.")
            return None
            
        # Обновляем параметры постпроцессора
        self.postprocessor.saturation = self.current_saturation
        self.postprocessor.contrast = self.current_contrast
        
        print(f"Колоризация изображения: {os.path.basename(self.current_image_path)}")
        print("Применяемые настройки:")
        print(f"- Стиль: {self.style_presets[self.current_style]['name']}")
        print(f"- Насыщенность: {self.current_saturation}")
        print(f"- Контраст: {self.current_contrast}")
        
        try:
            # Колоризируем изображение
            result = self.predictor.colorize_image(
                image_path=self.current_image_path,
                postprocessor=self.postprocessor,
                batch_size=1
            )
            
            if not result:
                print("Ошибка: не удалось колоризировать изображение.")
                return None
                
            print("Колоризация успешно завершена!")
            
            # Создаем сравнение
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            comparison_path = os.path.join(
                self.output_dirs['comparisons'], 
                f"{base_name}_{self.current_style}.png"
            )
            
            # Загружаем исходное изображение
            grayscale_img = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
            
            # Создаем сравнение
            self.visualizer.create_comparison(
                grayscale=grayscale_img,
                colorized=result['colorized'],
                filename=comparison_path
            )
            
            print(f"Сравнение сохранено: {comparison_path}")
            
            # Если требуется отображение неопределенности
            if self.show_uncertainty and self.uncertainty_module:
                # Преобразуем изображение в тензор для модуля неопределенности
                colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Оцениваем неопределенность
                uncertainty_result = self.uncertainty_module(colorized_tensor)
                uncertainty_map = uncertainty_result['uncertainty'][0, 0].cpu().numpy()
                
                # Сохраняем карту неопределенности
                uncertainty_path = os.path.join(
                    self.output_dirs['uncertainty'], 
                    f"{base_name}_uncertainty.png"
                )
                
                # Преобразуем в визуальную форму
                plt.figure(figsize=(10, 8))
                plt.imshow(uncertainty_map, cmap='inferno')
                plt.colorbar(label='Неопределенность')
                plt.title(f"Карта неопределенности: {os.path.basename(self.current_image_path)}")
                plt.axis('off')
                plt.savefig(uncertainty_path, bbox_inches='tight')
                plt.close()
                
                print(f"Карта неопределенности сохранена: {uncertainty_path}")
                
                # Добавляем карту неопределенности к результату
                result['uncertainty_map'] = uncertainty_map
                
            return result
            
        except Exception as e:
            print(f"Ошибка при колоризации: {str(e)}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def save_result(self, result, name=None):
        """
        Сохраняет результат колоризации.
        
        Args:
            result (Dict): Результат колоризации
            name (str, optional): Имя для сохранения
        
        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        if not result or 'colorized' not in result:
            print("Ошибка: нет результата для сохранения.")
            return False
            
        # Определяем имя файла
        if not name:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            name = f"{base_name}_{self.current_style}"
            
        # Путь для сохранения
        output_path = os.path.join(self.output_dirs['colorized'], f"{name}.png")
        
        # Сохраняем как изображение
        try:
            colorized_img = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
            colorized_img.save(output_path)
            
            print(f"Результат колоризации сохранен: {output_path}")
            
            # Сохраняем метаданные
            metadata_path = os.path.join(self.output_dirs['metadata'], f"{name}.json")
            
            metadata = {
                'original_image': self.current_image_path,
                'style': self.current_style,
                'style_name': self.style_presets[self.current_style]['name'],
                'saturation': self.current_saturation,
                'contrast': self.current_contrast,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'output_path': output_path
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            print(f"Метаданные сохранены: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении результата: {str(e)}")
            return False
    
    def process_command(self, command):
        """
        Обрабатывает команду пользователя.
        
        Args:
            command (str): Команда пользователя
            
        Returns:
            bool: True, если продолжить выполнение, False для выхода
        """
        cmd = command.strip().lower()
        
        # Обработка базовых команд
        if cmd == 'выход' or cmd == 'exit' or cmd == 'quit':
            print("Завершение работы демо...")
            return False
            
        elif cmd == 'помощь' or cmd == 'help':
            self.print_welcome()
            
        elif cmd == 'статус':
            self.print_status()
            
        elif cmd == 'образцы':
            self.print_samples()
            
        elif cmd == 'стили':
            self.print_styles()
            
        elif cmd == 'колоризовать':
            result = self.colorize_image()
            if result:
                # Автоматически сохраняем результат
                self.save_result(result)
                
        # Обработка команд с параметрами
        elif cmd.startswith('загрузить '):
            path = command[9:].strip()
            if os.path.exists(path):
                self.current_image_path = path
                print(f"Загружено изображение: {os.path.basename(path)}")
            else:
                print(f"Ошибка: файл не найден - {path}")
                
        elif cmd.startswith('образец '):
            sample_name = command[8:].strip()
            
            # Проверяем, может быть это номер
            if sample_name.isdigit():
                sample_idx = int(sample_name) - 1
                if 0 <= sample_idx < len(self.sample_images):
                    sample_name = list(self.sample_images.keys())[sample_idx]
                else:
                    print(f"Ошибка: образец с номером {sample_name} не найден")
                    return True
                    
            # Проверяем, есть ли такой образец
            if sample_name in self.sample_images:
                self.current_image_path = self.sample_images[sample_name]
                print(f"Выбран образец: {sample_name} ({os.path.basename(self.current_image_path)})")
            else:
                print(f"Ошибка: образец '{sample_name}' не найден")
                
        elif cmd.startswith('стиль '):
            style_name = command[6:].strip()
            
            # Проверяем, может быть это номер
            if style_name.isdigit():
                style_idx = int(style_name) - 1
                if 0 <= style_idx < len(self.style_presets):
                    style_name = list(self.style_presets.keys())[style_idx]
                else:
                    print(f"Ошибка: стиль с номером {style_name} не найден")
                    return True
                    
            # Проверяем, есть ли такой стиль
            if style_name in self.style_presets:
                self.current_style = style_name
                style = self.style_presets[style_name]
                
                # Обновляем параметры в соответствии со стилем
                if 'saturation' in style:
                    self.current_saturation = style['saturation']
                if 'contrast' in style:
                    self.current_contrast = style['contrast']
                    
                print(f"Выбран стиль: {style['name']}")
                print(f"- Описание: {style['description']}")
                print(f"- Насыщенность: {self.current_saturation}")
                print(f"- Контраст: {self.current_contrast}")
            else:
                print(f"Ошибка: стиль '{style_name}' не найден")
                
        elif cmd.startswith('насыщенность '):
            try:
                value = float(command[13:].strip())
                if 0.0 <= value <= 2.0:
                    self.current_saturation = value
                    print(f"Насыщенность установлена: {value}")
                else:
                    print("Ошибка: значение должно быть в диапазоне от 0.0 до 2.0")
            except ValueError:
                print("Ошибка: невозможно преобразовать значение в число")
                
        elif cmd.startswith('контраст '):
            try:
                value = float(command[9:].strip())
                if 0.0 <= value <= 2.0:
                    self.current_contrast = value
                    print(f"Контраст установлен: {value}")
                else:
                    print("Ошибка: значение должно быть в диапазоне от 0.0 до 2.0")
            except ValueError:
                print("Ошибка: невозможно преобразовать значение в число")
                
        elif cmd.startswith('неопределенность '):
            value = command[16:].strip().lower()
            if value in ['вкл', 'on', 'true', '1', 'да', 'yes']:
                self.show_uncertainty = True
                print("Отображение неопределенности включено")
            elif value in ['выкл', 'off', 'false', '0', 'нет', 'no']:
                self.show_uncertainty = False
                print("Отображение неопределенности выключено")
            else:
                print("Ошибка: неверное значение. Используйте 'вкл' или 'выкл'")
                
        elif cmd.startswith('сохранить '):
            name = command[10:].strip()
            if name:
                # Получаем последний результат колоризации
                result = self.colorize_image()
                if result:
                    self.save_result(result, name)
            else:
                print("Ошибка: необходимо указать имя для сохранения")
                
        # Обработка неизвестных команд
        else:
            # Проверяем, может быть это команда для UserInteractionModule
            parsed = self.user_interaction.parse_command(command)
            
            if parsed['is_command']:
                cmd_type = parsed['command_type']
                params = parsed['params']
                
                # Обработка команд TintoraAI
                if cmd_type == 'ЦветСтиль':
                    # Ищем стиль по имени или префиксу
                    matching_styles = [s for s in self.style_presets.keys() 
                                       if s.lower().startswith(params.lower())]
                    
                    if matching_styles:
                        self.current_style = matching_styles[0]
                        style = self.style_presets[self.current_style]
                        
                        # Обновляем параметры
                        if 'saturation' in style:
                            self.current_saturation = style['saturation']
                        if 'contrast' in style:
                            self.current_contrast = style['contrast']
                            
                        print(f"Установлен стиль: {style['name']}")
                    else:
                        print(f"Ошибка: стиль '{params}' не найден")
                
                elif cmd_type == 'НасыщенностьЦвета':
                    try:
                        value = float(params)
                        if 0.0 <= value <= 2.0:
                            self.current_saturation = value
                            print(f"Насыщенность установлена: {value}")
                        else:
                            print("Ошибка: значение должно быть в диапазоне от 0.0 до 2.0")
                    except ValueError:
                        print("Ошибка: невозможно преобразовать значение в число")
                
                elif cmd_type == 'СоветыПоЦвету':
                    print(f"Применение цветовых советов: {params}")
                    print("Эта функция пока не реализована в демо-режиме")
                
                else:
                    print(f"Неизвестная команда: {cmd_type}")
            else:
                print("Неизвестная команда. Введите 'помощь' для получения списка команд.")
        
        return True
    
    def run(self):
        """Запускает интерактивное демо в консоли."""
        self.print_welcome()
        
        # Основной цикл
        while self.running:
            try:
                command = input("\nTintoraAI> ")
                self.running = self.process_command(command)
            except KeyboardInterrupt:
                print("\nРабота демо прервана пользователем.")
                self.running = False
            except Exception as e:
                print(f"Ошибка: {str(e)}")
                if self.args.verbose:
                    import traceback
                    traceback.print_exc()


def run_streamlit_app(args, device, output_dirs, predictor, postprocessor, uncertainty_module):
    """
    Запускает веб-приложение с использованием Streamlit.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
        device (torch.device): Устройство для вычислений
        output_dirs (Dict[str, str]): Директории для сохранения результатов
        predictor (ColorizationPredictor): Предиктор для колоризации
        postprocessor (ColorizationPostProcessor): Постпроцессор для обработки результатов
        uncertainty_module (UncertaintyEstimation): Модуль оценки неопределенности
    """
    # Эта функция будет реализована отдельным файлом при запуске через Streamlit
    # Здесь приводим только базовый код для проверки
    
    if not STREAMLIT_AVAILABLE:
        print("Ошибка: Streamlit не установлен. Установите его с помощью:")
        print("pip install streamlit")
        return
        
    # Для запуска через Python API (не рекомендуется)
    import streamlit as st
    
    # Базовая структура приложения
    st.title("TintoraAI - Колоризация изображений")
    st.sidebar.title("Настройки")
    
    # Загружаем примеры и пресеты стилей
    sample_images = load_sample_images(args.samples_dir)
    style_presets = load_style_presets(args.styles_dir)
    
    # Выбор изображения
    st.sidebar.header("Загрузка изображения")
    upload_method = st.sidebar.radio("Выберите источник изображения:", 
                                    ["Загрузить своё", "Выбрать из примеров"])
    
    # Загрузка изображения
    image_path = None
    if upload_method == "Загрузить своё":
        uploaded_file = st.sidebar.file_uploader("Выберите изображение", 
                                               type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file is not None:
            # Сохраняем временный файл
            temp_path = os.path.join(args.output_dir, "temp", "uploaded.jpg")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            image_path = temp_path
    else:
        sample_names = list(sample_images.keys())
        if sample_names:
            selected_sample = st.sidebar.selectbox("Выберите пример:", sample_names)
            image_path = sample_images[selected_sample]
    
    # Настройки стиля
    st.sidebar.header("Настройки стиля")
    style_names = [style['name'] for style in style_presets.values()]
    style_dict = {style['name']: style_id for style_id, style in style_presets.items()}
    
    selected_style_name = st.sidebar.selectbox("Выберите стиль:", style_names)
    selected_style_id = style_dict[selected_style_name]
    style = style_presets[selected_style_id]
    
    # Параметры постобработки
    st.sidebar.header("Постобработка")
    saturation = st.sidebar.slider("Насыщенность:", 0.0, 2.0, 
                                 style.get('saturation', 1.0), step=0.1)
    contrast = st.sidebar.slider("Контраст:", 0.0, 2.0, 
                               style.get('contrast', 1.0), step=0.1)
    
    # Отображение неопределенности
    show_uncertainty = st.sidebar.checkbox("Показать карту неопределенности", False)
    
    # Основной интерфейс
    if image_path:
        # Отображаем исходное изображение
        st.header("Исходное изображение")
        original_img = Image.open(image_path)
        st.image(original_img, caption="Исходное изображение", use_column_width=True)
        
        # Кнопка колоризации
        if st.button("Колоризировать"):
            if predictor:
                # Обновляем параметры постпроцессора
                postprocessor.saturation = saturation
                postprocessor.contrast = contrast
                
                # Выполняем колоризацию
                with st.spinner("Колоризация..."):
                    try:
                        result = predictor.colorize_image(
                            image_path=image_path,
                            postprocessor=postprocessor,
                            batch_size=1
                        )
                        
                        if result and 'colorized' in result:
                            # Отображаем результат
                            st.header("Результат колоризации")
                            colorized_img = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
                            st.image(colorized_img, caption="Колоризированное изображение", use_column_width=True)
                            
                            # Отображаем неопределенность, если нужно
                            if show_uncertainty and uncertainty_module:
                                st.header("Карта неопределенности")
                                
                                colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(device)
                                uncertainty_result = uncertainty_module(colorized_tensor)
                                uncertainty_map = uncertainty_result['uncertainty'][0, 0].cpu().numpy()
                                
                                fig, ax = plt.subplots()
                                im = ax.imshow(uncertainty_map, cmap='inferno')
                                ax.set_title("Карта неопределенности")
                                ax.axis('off')
                                fig.colorbar(im, ax=ax)
                                st.pyplot(fig)
                                
                            # Сохранение результата
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            output_path = os.path.join(output_dirs['colorized'], f"{base_name}_{selected_style_id}.png")
                            colorized_img.save(output_path)
                            
                            st.success(f"Результат сохранен в {output_path}")
                            
                    except Exception as e:
                        st.error(f"Ошибка при колоризации: {str(e)}")
            else:
                st.error("Модель не загружена. Укажите путь к чекпоинту при запуске.")
    else:
        st.info("Загрузите изображение или выберите пример из списка.")


def main():
    """Основная функция для запуска демо."""
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Настройка окружения
    device, output_dirs = setup_environment(args)
    
    # Загрузка модели и компонентов
    model, predictor, postprocessor, uncertainty_module = load_model_and_components(args, device)
    
    # Определяем режим запуска
    if args.web and STREAMLIT_AVAILABLE:
        # Запуск веб-интерфейса через Streamlit
        print("Запуск веб-интерфейса с Streamlit...")
        
        # Это не работает напрямую из скрипта, нужно запускать через командную строку
        print("Для запуска веб-интерфейса используйте:")
        print(f"streamlit run {__file__} -- --checkpoint {args.checkpoint} --samples-dir {args.samples_dir} --output-dir {args.output_dir}")
        
        if os.environ.get('STREAMLIT_RUN_MODE') == 'streamlit':
            # Код запускается через streamlit
            run_streamlit_app(args, device, output_dirs, predictor, postprocessor, uncertainty_module)
        else:
            # Автоматический запуск через subprocess (для удобства)
            try:
                import subprocess
                cmd = [
                    "streamlit", "run", __file__, "--",
                    "--checkpoint", args.checkpoint if args.checkpoint else "",
                    "--samples-dir", args.samples_dir,
                    "--output-dir", args.output_dir,
                    "--web"
                ]
                
                print("Запуск Streamlit...")
                subprocess.run(cmd)
            except Exception as e:
                print(f"Ошибка при запуске Streamlit: {str(e)}")
                print("Запускаем консольный режим вместо веб-интерфейса.")
                console_demo = ConsoleDemo(args, device, output_dirs, predictor, postprocessor, uncertainty_module)
                console_demo.run()
                
    elif args.gui:
        # Запуск графического интерфейса (не реализовано)
        print("Графический интерфейс не реализован в текущей версии.")
        print("Запускаем консольный режим вместо графического интерфейса.")
        
        console_demo = ConsoleDemo(args, device, output_dirs, predictor, postprocessor, uncertainty_module)
        console_demo.run()
        
    else:
        # Консольный режим по умолчанию
        console_demo = ConsoleDemo(args, device, output_dirs, predictor, postprocessor, uncertainty_module)
        console_demo.run()


if __name__ == "__main__":
    main()