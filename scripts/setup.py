#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - Скрипт автоматической настройки проекта

Данный скрипт предназначен для подготовки окружения и настройки проекта TintoraAI.
Он создает необходимую структуру директорий, проверяет и устанавливает зависимости,
проверяет наличие и работоспособность GPU, а также выполняет базовую конфигурацию проекта.

Возможности:
- Создание полной структуры директорий проекта
- Проверка и установка зависимостей
- Проверка доступности и производительности GPU
- Генерация конфигурационных файлов
- Настройка под различные сценарии использования (обучение, инференс)
- Проверка целостности существующих файлов проекта
"""

import os
import sys
import shutil
import json
import platform
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Проверяем, установлены ли необходимые пакеты, и импортируем их
try:
    import torch
    import numpy as np
    import yaml
    import tqdm
    import colorama
    from colorama import Fore, Style
    colorama.init()
except ImportError as e:
    print(f"Не удалось импортировать необходимый пакет: {str(e)}")
    print("Устанавливаем базовые зависимости...")
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "numpy", "pyyaml", "tqdm", "colorama"])
    
    # Повторяем импорт
    import torch
    import numpy as np
    import yaml
    import tqdm
    import colorama
    from colorama import Fore, Style
    colorama.init()


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="TintoraAI - Настройка проекта колоризации",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Основные параметры
    parser.add_argument("--project-root", type=str, default=".",
                        help="Корневая директория проекта (по умолчанию: текущая директория)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Принудительно пересоздать директории и файлы")
    parser.add_argument("--skip-deps", action="store_true", default=False,
                        help="Пропустить проверку и установку зависимостей")
    
    # Параметры создания структуры
    parser.add_argument("--dirs-only", action="store_true", default=False,
                        help="Создать только структуру директорий без генерации конфигов")
    parser.add_argument("--configs-only", action="store_true", default=False,
                        help="Создать только конфигурационные файлы")
    
    # Параметры настройки
    parser.add_argument("--mode", type=str, choices=["full", "train", "inference", "minimal"],
                        default="full", help="Режим настройки проекта")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Размер изображения для конфигурации")
    parser.add_argument("--gpu-check", action="store_true", default=True,
                        help="Выполнить проверку GPU")
    parser.add_argument("--skip-gpu-check", action="store_true", default=False,
                        help="Пропустить проверку GPU")
                        
    # Дополнительные параметры
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Выводить подробную информацию о процессе")
    
    return parser.parse_args()


def print_header(text: str):
    """
    Выводит отформатированный заголовок.
    
    Args:
        text (str): Текст заголовка
    """
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{text.center(60)}")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")


def print_success(text: str):
    """
    Выводит сообщение об успехе.
    
    Args:
        text (str): Текст сообщения
    """
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text: str):
    """
    Выводит предупреждение.
    
    Args:
        text (str): Текст предупреждения
    """
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_error(text: str):
    """
    Выводит сообщение об ошибке.
    
    Args:
        text (str): Текст сообщения об ошибке
    """
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def check_python_version():
    """
    Проверяет версию Python.
    
    Returns:
        bool: True, если версия Python удовлетворяет требованиям, иначе False
    """
    print("Проверка версии Python...")
    
    major, minor, _ = sys.version_info[:3]
    current_version = f"{major}.{minor}"
    
    if major < 3 or (major == 3 and minor < 7):
        print_error(f"Неподдерживаемая версия Python: {current_version}. Требуется Python 3.7 или выше.")
        return False
        
    print_success(f"Версия Python: {current_version}")
    return True


def check_gpu():
    """
    Проверяет наличие и доступность GPU.
    
    Returns:
        Tuple[bool, dict]: (gpu_available, gpu_info) - флаг доступности GPU и информация о GPU
    """
    gpu_info = {
        'available': False,
        'name': None,
        'memory': None,
        'capability': None,
        'cuda_version': None,
        'performance': None
    }
    
    print("Проверка наличия и доступности GPU...")
    
    # Проверяем доступность CUDA через PyTorch
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print_warning("GPU не обнаружен или CUDA недоступна. Будет использоваться CPU.")
        return False, gpu_info
    
    # Получаем информацию о GPU
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print_warning("Устройства CUDA обнаружены, но недоступны.")
        return False, gpu_info
        
    print_success(f"Обнаружено устройств CUDA: {device_count}")
    
    # Собираем информацию о первом GPU (основном)
    gpu_info['available'] = True
    gpu_info['name'] = torch.cuda.get_device_name(0)
    gpu_info['cuda_version'] = torch.version.cuda
    
    # Пытаемся получить информацию о памяти GPU
    try:
        gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # в ГБ
    except:
        gpu_info['memory'] = "Неизвестно"
    
    # Пытаемся получить информацию о вычислительной мощности
    try:
        capability = torch.cuda.get_device_capability(0)
        gpu_info['capability'] = f"{capability[0]}.{capability[1]}"
    except:
        gpu_info['capability'] = "Неизвестно"
    
    # Выводим информацию
    print(f"  - GPU: {gpu_info['name']}")
    print(f"  - Память: {gpu_info['memory']} ГБ" if isinstance(gpu_info['memory'], (int, float)) else f"  - Память: {gpu_info['memory']}")
    print(f"  - CUDA: {gpu_info['cuda_version']}")
    print(f"  - Compute Capability: {gpu_info['capability']}")
    
    # Проверяем производительность (опционально)
    print("\nПроверка производительности GPU...")
    
    try:
        # Создаем тензоры для теста
        size = 2000  # Размер матриц для умножения
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Прогреваем GPU
        for _ in range(5):
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        # Измеряем время
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        end_time = time.time()
        
        performance = (10 * size * size * size) / ((end_time - start_time) * 1e9)  # TFLOPS
        gpu_info['performance'] = performance
        
        print_success(f"Производительность: {performance:.2f} TFLOPS")
        
    except Exception as e:
        print_warning(f"Не удалось измерить производительность GPU: {str(e)}")
    
    return True, gpu_info


def check_and_install_dependencies(requirements_path: str = None, verbose: bool = False):
    """
    Проверяет и устанавливает зависимости проекта.
    
    Args:
        requirements_path (str, optional): Путь к файлу с зависимостями
        verbose (bool): Выводить подробную информацию
        
    Returns:
        bool: True, если все зависимости установлены успешно, иначе False
    """
    print_header("Проверка и установка зависимостей")
    
    # Основные зависимости проекта
    core_dependencies = [
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.4.0",
        "pyyaml>=5.3.0",
        "tqdm>=4.50.0",
        "pandas>=1.1.0"
    ]
    
    # Дополнительные зависимости для обучения
    training_dependencies = [
        "albumentations>=0.5.0",
        "tensorboard>=2.3.0",
        "lpips>=0.1.3",
        "scikit-image>=0.17.0",
        "scikit-learn>=0.23.0"
    ]
    
    # Дополнительные зависимости для интерфейса и визуализации
    ui_dependencies = [
        "streamlit>=1.0.0",
        "seaborn>=0.11.0",
        "plotly>=4.14.0",
        "colorama>=0.4.4"
    ]
    
    # Определяем список зависимостей для установки
    dependencies_to_install = []
    
    # Если указан файл requirements.txt, загружаем зависимости из него
    if requirements_path and os.path.exists(requirements_path):
        try:
            with open(requirements_path, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print_success(f"Загружены зависимости из {requirements_path}")
            dependencies_to_install = requirements
        except Exception as e:
            print_warning(f"Не удалось прочитать файл зависимостей {requirements_path}: {str(e)}")
            print("Будут установлены базовые зависимости")
            dependencies_to_install = core_dependencies + training_dependencies
    else:
        # Если файл не указан или не существует, используем базовые зависимости
        print("Файл requirements.txt не указан или не найден. Будут установлены базовые зависимости.")
        dependencies_to_install = core_dependencies + training_dependencies + ui_dependencies
    
    # Проверяем и устанавливаем зависимости
    failed_packages = []
    
    for package in dependencies_to_install:
        package_name = package.split('>=')[0].split('==')[0].strip()
        
        try:
            # Проверяем, установлен ли пакет
            if verbose:
                print(f"Проверка пакета {package_name}...")
                
            __import__(package_name)
            if verbose:
                print_success(f"Пакет {package_name} уже установлен")
        except ImportError:
            # Если пакет не установлен, пробуем его установить
            print(f"Установка пакета {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print_success(f"Пакет {package_name} успешно установлен")
            except subprocess.CalledProcessError as e:
                print_error(f"Не удалось установить пакет {package_name}: {str(e)}")
                failed_packages.append(package_name)
    
    # Генерация итогового requirements.txt, если его не было
    if not requirements_path or not os.path.exists(requirements_path):
        try:
            with open('requirements.txt', 'w') as f:
                for dep in dependencies_to_install:
                    f.write(f"{dep}\n")
            print_success("Создан файл requirements.txt")
        except Exception as e:
            print_warning(f"Не удалось создать файл requirements.txt: {str(e)}")
    
    if failed_packages:
        print_warning(f"Не удалось установить следующие пакеты: {', '.join(failed_packages)}")
        print("Вы можете попробовать установить их вручную с помощью pip")
        return False
    
    print_success("Все зависимости успешно установлены")
    return True


def create_directory_structure(project_root: str, mode: str = "full", force: bool = False, verbose: bool = False):
    """
    Создает структуру директорий проекта.
    
    Args:
        project_root (str): Корневая директория проекта
        mode (str): Режим настройки проекта ("full", "train", "inference", "minimal")
        force (bool): Принудительно пересоздать директории
        verbose (bool): Выводить подробную информацию
        
    Returns:
        bool: True, если структура создана успешно, иначе False
    """
    print_header("Создание структуры директорий проекта")
    
    # Определяем базовую структуру директорий
    base_dirs = [
        "configs",
        "core",
        "losses",
        "modules",
        "utils",
        "training",
        "inference",
        "datasets",
        "tests",
        "scripts",
        "experiments/logs",
        "experiments/checkpoints",
        "experiments/results",
        "experiments/configs",
    ]
    
    # Дополнительные директории в зависимости от режима
    if mode in ["full", "train"]:
        base_dirs.extend([
            "data/train/grayscale",
            "data/train/color",
            "data/val/grayscale",
            "data/val/color",
            "data/test/grayscale",
            "data/test/color",
            "data/reference/historical",
            "data/reference/artistic",
            "data/reference/natural"
        ])
        
    if mode in ["full", "inference"]:
        base_dirs.extend([
            "input/single",
            "input/batch",
            "input/queue",
            "output/colorized",
            "output/comparisons",
            "output/uncertainty_maps",
            "output/metadata",
            "assets/color_palettes",
            "assets/style_presets",
            "assets/masks",
            "assets/watermarks",
            "monitoring/performance",
            "monitoring/memory_usage",
            "monitoring/error_logs"
        ])
    
    # Создаем директории
    created_dirs = []
    skipped_dirs = []
    failed_dirs = []
    
    for directory in base_dirs:
        full_path = os.path.join(project_root, directory)
        
        try:
            if os.path.exists(full_path):
                if force:
                    shutil.rmtree(full_path)
                    os.makedirs(full_path)
                    created_dirs.append(directory)
                    if verbose:
                        print_success(f"Пересоздана директория: {directory}")
                else:
                    skipped_dirs.append(directory)
                    if verbose:
                        print_warning(f"Директория уже существует (пропущено): {directory}")
            else:
                os.makedirs(full_path)
                created_dirs.append(directory)
                if verbose:
                    print_success(f"Создана директория: {directory}")
                    
        except Exception as e:
            failed_dirs.append(directory)
            print_error(f"Не удалось создать директорию {directory}: {str(e)}")
    
    # Выводим итоговую статистику
    if not verbose:
        print(f"Создано директорий: {len(created_dirs)}")
        print(f"Пропущено существующих директорий: {len(skipped_dirs)}")
        
    if failed_dirs:
        print_error(f"Не удалось создать {len(failed_dirs)} директорий: {', '.join(failed_dirs)}")
        return False
    
    print_success("Структура директорий проекта создана успешно")
    return True


def generate_config_files(project_root: str, mode: str = "full", img_size: int = 256, force: bool = False):
    """
    Генерирует конфигурационные файлы проекта.
    
    Args:
        project_root (str): Корневая директория проекта
        mode (str): Режим настройки проекта
        img_size (int): Размер изображения для конфигурации
        force (bool): Принудительно пересоздать файлы
        
    Returns:
        bool: True, если файлы созданы успешно, иначе False
    """
    print_header("Создание конфигурационных файлов")
    
    configs_dir = os.path.join(project_root, "configs")
    
    # Убеждаемся, что директория существует
    if not os.path.exists(configs_dir):
        try:
            os.makedirs(configs_dir)
        except Exception as e:
            print_error(f"Не удалось создать директорию для конфигурационных файлов: {str(e)}")
            return False
    
    # Определяем основные конфигурации
    
    # 1. Основная конфигурация проекта
    main_config = {
        "project": {
            "name": "TintoraAI",
            "version": "1.0.0",
            "description": "Система колоризации изображений"
        },
        "paths": {
            "data_root": "data",
            "experiments": "experiments",
            "input": "input",
            "output": "output",
            "assets": "assets"
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_dir": "experiments/logs"
        },
        "gpu": {
            "use_gpu": True,
            "device_id": 0,
            "precision": "float32"
        }
    }
    
    # 2. Конфигурация модели
    model_config = {
        "img_size": img_size,
        "in_channels": 1,
        "out_channels": 2,
        "color_space": "lab",
        
        "swin_unet": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.2
        },
        
        "vit_semantic": {
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0
        },
        
        "fpn_pyramid": {
            "out_channels": 256,
            "use_pyramid_pooling": True
        },
        
        "cross_attention": {
            "num_heads": 8,
            "dropout_rate": 0.1
        },
        
        "feature_fusion": {
            "out_channels": 512,
            "num_heads": 8
        },
        
        "guide_net": {
            "enabled": True,
            "input_channels": 1,
            "advice_channels": 2,
            "feature_dim": 64,
            "num_layers": 6,
            "use_attention": True,
            "use_confidence": True
        },
        
        "discriminator": {
            "enabled": True,
            "input_nc": 3,
            "ndf": 64,
            "n_layers": 3,
            "use_spectral_norm": True,
            "reward_type": "adaptive"
        },
        
        "memory_bank": {
            "enabled": True,
            "feature_dim": 256,
            "max_items": 1000,
            "index_type": "flat",
            "use_fusion": True
        },
        
        "uncertainty": {
            "enabled": True,
            "method": "guided",
            "num_samples": 5,
            "dropout_rate": 0.2
        },
        
        "adaptable": {
            "enabled": True,
            "adapter_type": "standard",
            "bottleneck_dim": 64
        }
    }
    
    # 3. Конфигурация обучения
    training_config = {
        "data": {
            "color_space": "lab",
            "grayscale_dir": "train/grayscale",
            "color_dir": "train/color",
            "val_grayscale_dir": "val/grayscale",
            "val_color_dir": "val/color",
            "augmentation_level": "medium",
            "dynamic_batching": True,
            "max_train_samples": None,
            "max_val_samples": None,
            "reference_dataset_path": "data/reference"
        },
        "training": {
            "batch_size": 16,
            "epochs": 100,
            "val_freq": 1,
            "save_freq": 5,
            "log_freq": 10,
            "num_workers": 4,
            "seed": 42
        },
        "optimizer": {
            "type": "adam",
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "weight_decay": 0.0001
        },
        "scheduler": {
            "type": "cosine",
            "min_lr": 0.00001,
            "warmup_epochs": 5
        },
        "checkpoint": {
            "save_best": True,
            "max_to_keep": 5,
            "metric": "val_loss",
            "mode": "min"
        }
    }
    
    # 4. Конфигурация функций потерь
    loss_config = {
        "l1_loss": {
            "enabled": True,
            "weight": 10.0,
            "target": "color"
        },
        "mse_loss": {
            "enabled": False,
            "weight": 10.0,
            "target": "color"
        },
        "patch_nce": {
            "enabled": True,
            "weight": 1.0,
            "temperature": 0.07,
            "patch_size": 16,
            "n_patches": 256
        },
        "vgg_perceptual": {
            "enabled": True,
            "weight": 1.0,
            "layers": ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
            "layer_weights": [1.0, 1.0, 1.0, 1.0],
            "criterion": "l1",
            "resize": True,
            "normalize": True
        },
        "gan_loss": {
            "enabled": True,
            "weight": 0.1,
            "gan_mode": "lsgan",
            "real_label_val": 1.0,
            "fake_label_val": 0.0
        },
        "dynamic_balancer": {
            "enabled": True,
            "strategy": "adaptive",
            "target_metric": "lpips",
            "learning_rate": 0.01
        }
    }
    
    # 5. Конфигурация инференса
    inference_config = {
        "color_space": "lab",
        "img_size": img_size,
        "batch_size": 8,
        "num_workers": 4,
        "use_gpu": True,
        
        "postprocessing": {
            "enhance": False,
            "saturation": 1.0,
            "contrast": 1.0,
            "denoise": False,
            "sharpen": False
        },
        
        "style": None,
        "style_image": None,
        
        "output": {
            "save_comparison": True,
            "save_uncertainty": True,
            "save_metadata": True,
            "suffix": ""
        },
        
        "queue": {
            "interval": 5.0,
            "max_files": 10,
            "timeout": 0
        }
    }
    
    # Сохраняем конфигурации в файлы
    config_files = {
        "main_config.yaml": main_config,
        "model_config.yaml": model_config,
        "training_config.yaml": training_config,
        "loss_config.yaml": loss_config,
        "inference_config.yaml": inference_config
    }
    
    created_files = []
    skipped_files = []
    failed_files = []
    
    for filename, config_data in config_files.items():
        file_path = os.path.join(configs_dir, filename)
        
        try:
            if os.path.exists(file_path) and not force:
                skipped_files.append(filename)
                print_warning(f"Файл уже существует (пропущено): {filename}")
            else:
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                created_files.append(filename)
                print_success(f"Создан файл конфигурации: {filename}")
        except Exception as e:
            failed_files.append(filename)
            print_error(f"Не удалось создать файл {filename}: {str(e)}")
    
    if failed_files:
        print_error(f"Не удалось создать {len(failed_files)} файлов: {', '.join(failed_files)}")
        return False
    
    print_success("Конфигурационные файлы созданы успешно")
    return True


def check_environment_variables():
    """
    Проверяет и настраивает необходимые переменные окружения.
    
    Returns:
        dict: Настройки окружения
    """
    print_header("Проверка и настройка переменных окружения")
    
    env_vars = {}
    
    # Проверка и настройка PYTHONPATH
    python_path = os.environ.get('PYTHONPATH', '')
    current_dir = os.path.abspath(os.getcwd())
    
    if current_dir not in python_path.split(os.pathsep):
        if python_path:
            new_python_path = f"{current_dir}{os.pathsep}{python_path}"
        else:
            new_python_path = current_dir
            
        # Не изменяем глобальную переменную, просто информируем пользователя
        print_warning(f"Текущая директория не добавлена в PYTHONPATH.")
        print(f"Рекомендуется добавить её для правильной работы импортов:")
        print(f"export PYTHONPATH={new_python_path}")
        
        env_vars['PYTHONPATH'] = new_python_path
    else:
        print_success("PYTHONPATH настроен корректно")
    
    # Проверка переменных CUDA
    if torch.cuda.is_available():
        cuda_home = os.environ.get('CUDA_HOME')
        if not cuda_home:
            cuda_home = os.environ.get('CUDA_PATH')
            
        if cuda_home:
            print_success(f"CUDA_HOME настроен: {cuda_home}")
            env_vars['CUDA_HOME'] = cuda_home
        else:
            print_warning("CUDA_HOME не установлен, это может вызвать проблемы при компиляции некоторых расширений")
            
        # Проверка кэша компиляции PyTorch
        torch_cache = os.environ.get('TORCH_CUDA_ARCH_LIST')
        if torch_cache:
            print_success(f"TORCH_CUDA_ARCH_LIST настроен: {torch_cache}")
        else:
            print_warning("TORCH_CUDA_ARCH_LIST не установлен, может потребоваться для оптимизации под конкретные GPU")
    
    return env_vars


def create_simple_readme(project_root: str):
    """
    Создает простой README.md файл для проекта.
    
    Args:
        project_root (str): Корневая директория проекта
        
    Returns:
        bool: True, если файл создан успешно, иначе False
    """
    readme_path = os.path.join(project_root, "README.md")
    
    # Если файл уже существует, не перезаписываем его
    if os.path.exists(readme_path):
        print_warning("README.md уже существует, пропускаем создание")
        return True
        
    readme_content = """# TintoraAI - Система колоризации изображений"""

Современное решение для колоризации черно-белых изображений с использованием глубокого обучения и элементов искусственного интеллекта.

## Возможности

- Высококачественная колоризация черно-белых изображений
- Интеллектуальный анализ содержимого для правильного подбора цветов
- Поддержка различных стилей колоризации
- Адаптация к новым доменам с минимальным количеством примеров
- Интерактивное управление процессом колоризации

## Быстрый старт

### Установка

```bash
git clone https://github.com/yourusername/tintora-ai.git
cd tintora-ai
pip install -r requirements.txt
python scripts/setup.py
