#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TintoraAI - API для колоризации изображений

Данный модуль предоставляет API для взаимодействия с системой колоризации TintoraAI.
Включает RESTful API интерфейс для обработки запросов на колоризацию изображений,
а также SDK для программного взаимодействия с системой.

API поддерживает:
- Колоризацию отдельных изображений
- Пакетную обработку
- Выбор стилей и настроек колоризации
- Интеграцию с внешними системами
- Мониторинг статуса обработки
- Получение метаданных и статистики
"""

import os
import sys
import time
import json
import base64
import logging
import threading
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from io import BytesIO
from uuid import uuid4

import glob
import numpy as np
import torch
from PIL import Image
import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# Добавляем корневую директорию проекта в путь поиска
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Импорты из модулей проекта
from inference.predictor import ColorizationPredictor
from inference.postprocessor import ColorizationPostProcessor
from utils.config_parser import load_config
from utils.visualization import ColorizationVisualizer
from utils.user_interaction import UserInteractionModule
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.style_transfer import StyleTransfer
from modules.memory_bank import MemoryBankModule


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("monitoring/error_logs/api_logs.log")
    ]
)
logger = logging.getLogger("TintoraAI-API")


# Определение моделей данных для API
class ColorSpace(str, Enum):
    """Поддерживаемые цветовые пространства."""
    LAB = "lab"
    RGB = "rgb"
    YUV = "yuv"


class StylePreset(str, Enum):
    """Предустановленные стили колоризации."""
    DEFAULT = "default"
    VINTAGE = "vintage"
    VIVID = "vivid"
    MONOCHROME = "monochrome"
    CINEMATIC = "cinematic"
    HISTORICAL_1920S = "historical_1920s"
    HISTORICAL_1950S = "historical_1950s"
    ARTISTIC_IMPRESSIONISM = "artistic_impressionism"
    ARTISTIC_REALISM = "artistic_realism"
    NATURAL_SUNSET = "natural_sunset"
    NATURAL_FOREST = "natural_forest"
    NATURAL_PORTRAIT = "natural_portrait"


class ColorizationRequest(BaseModel):
    """Модель запроса на колоризацию."""
    # Основные параметры
    image_data: Optional[str] = Field(None, description="Изображение в формате base64")
    image_url: Optional[str] = Field(None, description="URL изображения для колоризации")
    style: Optional[StylePreset] = Field(StylePreset.DEFAULT, description="Стиль колоризации")
    
    # Дополнительные параметры
    color_space: ColorSpace = Field(ColorSpace.LAB, description="Цветовое пространство")
    saturation: float = Field(1.0, ge=0.0, le=2.0, description="Коэффициент насыщенности")
    contrast: float = Field(1.0, ge=0.0, le=2.0, description="Коэффициент контраста")
    enhance: bool = Field(False, description="Применить улучшение результатов")
    show_uncertainty: bool = Field(False, description="Включить карту неопределенности")
    
    # Расширенные параметры
    style_strength: float = Field(1.0, ge=0.0, le=1.0, description="Интенсивность применения стиля")
    custom_style_image: Optional[str] = Field(None, description="Пользовательское изображение стиля в base64")
    guidance: Optional[str] = Field(None, description="Подсказки для GuideNet в формате 'Команда:Текст'")
    
    # Валидация входных данных
    @validator("image_data", "image_url")
    def check_image_source(cls, v, values):
        """Проверяет, что указан хотя бы один источник изображения."""
        if "image_data" not in values and "image_url" not in values:
            if v is None:
                raise ValueError("Необходимо указать либо image_data, либо image_url")
        return v


class BatchColorizationRequest(BaseModel):
    """Модель запроса на пакетную колоризацию."""
    # Параметры пакетной обработки
    images: List[str] = Field(..., min_items=1, description="Список изображений в base64 или URLs")
    is_urls: bool = Field(False, description="Флаг, указывающий, что images содержит URLs")
    
    # Общие параметры колоризации для всех изображений
    style: Optional[StylePreset] = Field(StylePreset.DEFAULT, description="Стиль колоризации")
    color_space: ColorSpace = Field(ColorSpace.LAB, description="Цветовое пространство")
    saturation: float = Field(1.0, ge=0.0, le=2.0, description="Коэффициент насыщенности")
    contrast: float = Field(1.0, ge=0.0, le=2.0, description="Коэффициент контраста")
    enhance: bool = Field(False, description="Применить улучшение результатов")
    
    # Дополнительные параметры
    notify_url: Optional[str] = Field(None, description="URL для уведомления о завершении")
    callback_id: Optional[str] = Field(None, description="ID для callback")
    priority: int = Field(1, ge=1, le=10, description="Приоритет обработки (1-10)")


class ColorizationResponse(BaseModel):
    """Модель ответа на запрос колоризации."""
    # Основная информация
    status: str = Field(..., description="Статус обработки")
    message: str = Field(..., description="Сообщение о результате")
    request_id: str = Field(..., description="Уникальный идентификатор запроса")
    processing_time: float = Field(..., description="Время обработки в секундах")
    
    # Результаты колоризации
    colorized_image: Optional[str] = Field(None, description="Колоризованное изображение в base64")
    comparison_image: Optional[str] = Field(None, description="Сравнение до/после в base64")
    uncertainty_map: Optional[str] = Field(None, description="Карта неопределенности в base64")
    
    # Метаданные
    metadata: Dict[str, Any] = Field({}, description="Дополнительные метаданные")


class BatchProgressResponse(BaseModel):
    """Модель ответа с информацией о прогрессе пакетной обработки."""
    batch_id: str = Field(..., description="Идентификатор пакета")
    status: str = Field(..., description="Статус обработки")
    total_images: int = Field(..., description="Общее количество изображений")
    processed_images: int = Field(..., description="Количество обработанных изображений")
    progress_percent: float = Field(..., description="Процент завершения")
    estimated_time_left: Optional[float] = Field(None, description="Оценка оставшегося времени в секундах")
    errors: List[Dict[str, str]] = Field([], description="Список ошибок")


class TintoraAIModel:
    """
    Класс для управления моделью колоризации TintoraAI.
    Содержит логику загрузки модели, инференса и постобработки.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = "configs/inference_config.yaml"):
        """
        Инициализация модели колоризации.
        
        Args:
            checkpoint_path (str): Путь к чекпоинту модели
            config_path (str): Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка конфигурации
        self.config = load_config(config_path)
        if self.config is None:
            self.config = {}
            logger.warning("Не удалось загрузить конфигурацию, используются значения по умолчанию")
        
        # Состояние модели
        self.model = None
        self.predictor = None
        self.postprocessor = None
        self.uncertainty_module = None
        self.style_transfer = None
        self.memory_bank = None
        self.visualizer = ColorizationVisualizer(output_dir="output/api")
        self.user_interaction = UserInteractionModule()
        
        # Словарь стилей
        self.style_presets = self._load_style_presets()
        
        # Состояние загрузки
        self.is_loaded = False
        self.load_error = None
        
        # Запуск асинхронной загрузки модели
        self.load_model_async()
    
    def load_model_async(self):
        """Асинхронная загрузка модели в отдельном потоке."""
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()
    
    def _load_model(self):
        """Загрузка модели и всех компонентов."""
        try:
            logger.info(f"Загрузка модели из чекпоинта: {self.checkpoint_path}")
            
            # Загрузка чекпоинта
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'model' in checkpoint:
                self.model = checkpoint['model'].to(self.device)
                logger.info("Модель успешно загружена из чекпоинта")
            elif 'model_state_dict' in checkpoint:
                # Импорт функции для загрузки модели
                from training.checkpoints import load_model_from_checkpoint
                self.model = load_model_from_checkpoint(self.checkpoint_path, {}, self.device)
                logger.info("Модель успешно загружена из state_dict")
            else:
                raise ValueError(f"Некорректный формат чекпоинта: {self.checkpoint_path}")
            
            # Переводим модель в режим оценки
            self.model.eval()
            
            # Создаем постпроцессор
            self.postprocessor = ColorizationPostProcessor(
                color_space="lab",
                apply_enhancement=False,
                saturation=1.0,
                contrast=1.0,
                device=self.device
            )
            
            # Инициализируем интеллектуальные модули
            intelligent_modules = {}
            
            # Создаем модуль оценки неопределенности
            self.uncertainty_module = UncertaintyEstimation(
                method='guided',
                num_samples=5,
                dropout_rate=0.2,
                device=self.device
            ).to(self.device)
            intelligent_modules['uncertainty'] = self.uncertainty_module
            
            # Модуль переноса стиля
            self.style_transfer = StyleTransfer(
                content_weight=1.0,
                style_weight=100.0,
                device=self.device
            )
            intelligent_modules['style_transfer'] = self.style_transfer
            
            # Модуль банка памяти
            self.memory_bank = MemoryBankModule(
                feature_dim=256,
                max_items=1000,
                index_type='flat',
                device=self.device
            )
            intelligent_modules['memory_bank'] = self.memory_bank
            
            # Создаем предиктор
            self.predictor = ColorizationPredictor(
                model=self.model,
                device=self.device,
                color_space="lab",
                intelligent_modules=intelligent_modules
            )
            
            logger.info("Модель и все компоненты успешно загружены")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            self.load_error = str(e)
    
    def _load_style_presets(self):
        """Загрузка пресетов стилей."""
        styles_dir = "assets/style_presets"
        
        # Базовые стили
        style_presets = {
            'default': {
                'name': 'Стандартный',
                'description': 'Естественная колоризация с нейтральными цветами',
                'saturation': 1.0,
                'contrast': 1.0,
                'enhance': False
            },
            'vintage': {
                'name': 'Винтаж',
                'description': 'Старинный вид с мягкими пастельными тонами',
                'saturation': 0.8,
                'contrast': 0.9,
                'enhance': False,
                'temperature': 'warm'
            },
            'vivid': {
                'name': 'Яркий',
                'description': 'Насыщенные цвета для яркого впечатления',
                'saturation': 1.3,
                'contrast': 1.1,
                'enhance': True,
                'vibrance': 1.2
            }
        }
        
        # Добавляем другие стили из перечисления StylePreset
        for style in StylePreset:
            if style.value not in style_presets:
                style_presets[style.value] = {
                    'name': style.value.replace('_', ' ').title(),
                    'description': f'Стиль {style.value}',
                    'saturation': 1.0,
                    'contrast': 1.0,
                    'enhance': False
                }
        
        # Проверяем наличие директории с пресетами
        if os.path.exists(styles_dir):
            # Загружаем пресеты из JSON файлов
            json_files = [f for f in os.listdir(styles_dir) if f.endswith('.json')]
            
            for json_file in json_files:
                try:
                    with open(os.path.join(styles_dir, json_file), 'r') as f:
                        preset_data = json.load(f)
                    
                    preset_name = os.path.splitext(json_file)[0]
                    style_presets[preset_name] = preset_data
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке пресета {json_file}: {str(e)}")
        
        return style_presets
    
    def wait_for_model(self, timeout: int = 60) -> bool:
        """
        Ожидание загрузки модели.
        
        Args:
            timeout (int): Таймаут в секундах
            
        Returns:
            bool: True, если модель загружена, False в случае ошибки или таймаута
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_loaded:
                return True
            if self.load_error is not None:
                return False
            time.sleep(1)
            
        return False
    
    def _process_style_settings(self, style_name: str, saturation: float = None, contrast: float = None, 
                               enhance: bool = None) -> dict:
        """
        Получение настроек для указанного стиля с возможностью переопределения.
        
        Args:
            style_name (str): Имя стиля
            saturation (float, optional): Насыщенность (переопределение)
            contrast (float, optional): Контраст (переопределение)
            enhance (bool, optional): Улучшение (переопределение)
            
        Returns:
            dict: Настройки стиля
        """
        # Получаем базовые настройки стиля
        if style_name in self.style_presets:
            style_settings = dict(self.style_presets[style_name])
        else:
            style_settings = dict(self.style_presets['default'])
            logger.warning(f"Стиль {style_name} не найден, используется стиль по умолчанию")
        
        # Применяем переопределения
        if saturation is not None:
            style_settings['saturation'] = saturation
            
        if contrast is not None:
            style_settings['contrast'] = contrast
            
        if enhance is not None:
            style_settings['enhance'] = enhance
            
        return style_settings
    
    def colorize_image(self, image_data: np.ndarray, color_space: str = "lab", 
                       style: str = "default", saturation: float = None, 
                       contrast: float = None, enhance: bool = None,
                       style_strength: float = 1.0,
                       custom_style_image: np.ndarray = None,
                       guidance: str = None,
                       show_uncertainty: bool = False) -> dict:
        """
        Колоризация изображения.
        
        Args:
            image_data (np.ndarray): Изображение в виде numpy массива
            color_space (str): Цветовое пространство ("lab", "rgb", "yuv")
            style (str): Имя стиля
            saturation (float, optional): Насыщенность (переопределение)
            contrast (float, optional): Контраст (переопределение)
            enhance (bool, optional): Улучшение (переопределение)
            style_strength (float): Интенсивность применения стиля (0.0-1.0)
            custom_style_image (np.ndarray, optional): Пользовательское изображение стиля
            guidance (str, optional): Подсказки для GuideNet в формате "Команда:Текст"
            show_uncertainty (bool): Включить карту неопределенности
            
        Returns:
            dict: Результаты колоризации
        """
        # Проверяем, что модель загружена
        if not self.is_loaded:
            if not self.wait_for_model():
                raise RuntimeError(f"Модель не загружена: {self.load_error}")
        
        # Создаем временный файл для изображения
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            # Сохраняем изображение во временный файл
            cv2.imwrite(temp_path, image_data)
        
        try:
            # Получаем настройки стиля
            style_settings = self._process_style_settings(style, saturation, contrast, enhance)
            
            # Обновляем настройки постпроцессора
            self.postprocessor.color_space = color_space
            self.postprocessor.saturation = style_settings.get('saturation', 1.0)
            self.postprocessor.contrast = style_settings.get('contrast', 1.0)
            self.postprocessor.apply_enhancement = style_settings.get('enhance', False)
            
            # Обрабатываем команды и советы, если указаны
            if guidance:
                parsed_guidance = self.user_interaction.parse_command(guidance)
                if parsed_guidance['is_command']:
                    # TODO: Реализовать обработку команд
                    logger.info(f"Применена команда: {parsed_guidance['command_type']}")
            
            # Колоризируем изображение
            result = self.predictor.colorize_image(
                image_path=temp_path,
                postprocessor=self.postprocessor,
                batch_size=1
            )
            
            if not result:
                raise ValueError("Не удалось колоризировать изображение")
            
            # Применяем пользовательский стиль, если указан
            if custom_style_image is not None and self.style_transfer:
                # Преобразуем результат в тензор для переноса стиля
                colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                style_tensor = torch.from_numpy(custom_style_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Выполняем перенос стиля
                styled_result = self.style_transfer.transfer_style(
                    content_image=colorized_tensor,
                    style_image=style_tensor,
                    style_weight=style_strength * 100.0,  # Масштабируем вес стиля
                    num_steps=20  # Ограниченное число шагов для API
                )
                
                # Преобразуем обратно в numpy
                styled_result_np = styled_result[0].cpu().permute(1, 2, 0).numpy()
                result['colorized'] = styled_result_np
            
            # Оцениваем неопределенность, если требуется
            if show_uncertainty and self.uncertainty_module:
                # Преобразуем результат в тензор для оценки неопределенности
                colorized_tensor = torch.from_numpy(result['colorized']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Оцениваем неопределенность
                uncertainty_result = self.uncertainty_module(colorized_tensor)
                uncertainty_map = uncertainty_result['uncertainty'][0, 0].cpu().numpy()
                
                # Добавляем карту неопределенности к результату
                result['uncertainty_map'] = uncertainty_map
            
            return result
            
        finally:
            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Ошибка при удалении временного файла: {str(e)}")


class BatchProcessor:
    """
    Класс для обработки пакетных запросов на колоризацию.
    Управляет очередью, приоритетами и прогрессом обработки.
    """
    
    def __init__(self, tintora_model: TintoraAIModel):
        """
        Инициализация обработчика пакетных запросов.
        
        Args:
            tintora_model (TintoraAIModel): Модель колоризации
        """
        self.model = tintora_model
        self.batch_queue = []  # Очередь заданий
        self.active_batches = {}  # Активные пакеты: batch_id -> информация
        self.completed_batches = {}  # Завершенные пакеты
        self.lock = threading.Lock()  # Блокировка для многопоточного доступа
        
        # Запуск фонового потока для обработки очереди
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def add_batch(self, images: List[Union[np.ndarray, str]], is_urls: bool = False,
                  style: str = "default", color_space: str = "lab",
                  saturation: float = 1.0, contrast: float = 1.0, enhance: bool = False,
                  notify_url: str = None, callback_id: str = None, 
                  priority: int = 1) -> str:
        """
        Добавление пакета в очередь обработки.
        
        Args:
            images (List[Union[np.ndarray, str]]): Список изображений или URL
            is_urls (bool): Флаг, указывающий, что images содержит URL
            style (str): Имя стиля
            color_space (str): Цветовое пространство
            saturation (float): Насыщенность
            contrast (float): Контраст
            enhance (bool): Улучшение
            notify_url (str, optional): URL для уведомления о завершении
            callback_id (str, optional): ID для callback
            priority (int): Приоритет обработки (1-10)
            
        Returns:
            str: ID пакета
        """
        batch_id = str(uuid4())
        
        # Создаем запись пакета
        batch_info = {
            'id': batch_id,
            'images': images,
            'is_urls': is_urls,
            'style': style,
            'color_space': color_space,
            'saturation': saturation,
            'contrast': contrast,
            'enhance': enhance,
            'notify_url': notify_url,
            'callback_id': callback_id,
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'total_images': len(images),
            'processed_images': 0,
            'results': [],
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
        # Добавляем пакет в очередь с учетом приоритета
        with self.lock:
            # Находим позицию для вставки на основе приоритета
            insert_pos = 0
            for i, item in enumerate(self.batch_queue):
                if item['priority'] < priority:
                    insert_pos = i
                    break
                else:
                    insert_pos = i + 1
            
            # Вставляем пакет в нужную позицию
            self.batch_queue.insert(insert_pos, batch_info)
            self.active_batches[batch_id] = batch_info
            
            logger.info(f"Добавлен пакет {batch_id} с {len(images)} изображениями, приоритет: {priority}")
        
        return batch_id
    
    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о прогрессе обработки пакета.
        
        Args:
            batch_id (str): ID пакета
            
        Returns:
            Optional[Dict[str, Any]]: Информация о пакете или None, если пакет не найден
        """
        # Проверяем активные пакеты
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            
            # Вычисляем прогресс
            progress = (batch['processed_images'] / batch['total_images'] * 100) if batch['total_images'] > 0 else 0
            
            # Оцениваем оставшееся время
            estimated_time_left = None
            if batch['start_time'] and batch['processed_images'] > 0:
                elapsed_time = (datetime.now() - datetime.fromisoformat(batch['start_time'])).total_seconds()
                images_per_second = batch['processed_images'] / elapsed_time if elapsed_time > 0 else 0
                if images_per_second > 0:
                    remaining_images = batch['total_images'] - batch['processed_images']
                    estimated_time_left = remaining_images / images_per_second
            
            return {
                'batch_id': batch_id,
                'status': batch['status'],
                'total_images': batch['total_images'],
                'processed_images': batch['processed_images'],
                'progress_percent': progress,
                'estimated_time_left': estimated_time_left,
                'errors': batch['errors']
            }
            
        # Проверяем завершенные пакеты
        if batch_id in self.completed_batches:
            batch = self.completed_batches[batch_id]
            
            return {
                'batch_id': batch_id,
                'status': batch['status'],
                'total_images': batch['total_images'],
                'processed_images': batch['processed_images'],
                'progress_percent': 100.0,
                'estimated_time_left': 0,
                'errors': batch['errors']
            }
            
        return None
    
    def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение результатов обработки пакета.
        
        Args:
            batch_id (str): ID пакета
            
        Returns:
            Optional[Dict[str, Any]]: Результаты обработки или None, если пакет не найден или не завершен
        """
        # Проверяем завершенные пакеты
        if batch_id in self.completed_batches:
            batch = self.completed_batches[batch_id]
            
            if batch['status'] == 'completed':
                return {
                    'batch_id': batch_id,
                    'status': 'completed',
                    'total_images': batch['total_images'],
                    'processed_images': batch['processed_images'],
                    'results': batch['results'],
                    'errors': batch['errors'],
                    'processing_time': (datetime.fromisoformat(batch['end_time']) - 
                                      datetime.fromisoformat(batch['start_time'])).total_seconds()
                }
        
        return None
    
    def _process_queue(self):
        """Фоновая обработка очереди пакетов."""
        while self.running:
            # Берем пакет из очереди
            batch_info = None
            
            with self.lock:
                if self.batch_queue:
                    batch_info = self.batch_queue.pop(0)
            
            # Если есть пакет для обработки
            if batch_info:
                batch_id = batch_info['id']
                logger.info(f"Начинается обработка пакета {batch_id}")
                
                # Обновляем статус и время начала
                with self.lock:
                    batch_info['status'] = 'processing'
                    batch_info['start_time'] = datetime.now().isoformat()
                    self.active_batches[batch_id] = batch_info
                
                try:
                    # Обрабатываем каждое изображение в пакете
                    for i, image_data in enumerate(batch_info['images']):
                        try:
                            # Проверяем, не отменена ли задача
                            with self.lock:
                                if self.active_batches[batch_id]['status'] == 'cancelled':
                                    logger.info(f"Пакет {batch_id} был отменен")
                                    break
                            
                            # Загружаем изображение
                            if batch_info['is_urls']:
                                # Загружаем по URL (не реализовано)
                                image = None  # TODO: Реализовать загрузку по URL
                            else:
                                # Декодируем изображение из строки или base64
                                if isinstance(image_data, str):
                                    if image_data.startswith('data:image'):
                                        # Декодируем base64
                                        image_data = image_data.split(',')[1]
                                        image = np.array(Image.open(BytesIO(base64.b64decode(image_data))))
                                    else:
                                        # Обычный путь к файлу
                                        image = cv2.imread(image_data)
                                else:
                                    # Уже готовое изображение
                                    image = image_data
                            
                            if image is None:
                                raise ValueError("Не удалось загрузить изображение")
                            
                            # Колоризируем изображение
                            result = self.model.colorize_image(
                                image_data=image,
                                color_space=batch_info['color_space'],
                                style=batch_info['style'],
                                saturation=batch_info['saturation'],
                                contrast=batch_info['contrast'],
                                enhance=batch_info['enhance']
                            )
                            
                            # Кодируем результат в base64
                            colorized_pil = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
                            buffer = BytesIO()
                            colorized_pil.save(buffer, format="PNG")
                            colorized_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            # Добавляем результат
                            batch_info['results'].append({
                                'index': i,
                                'colorized_image': f"data:image/png;base64,{colorized_base64}",
                            })
                            
                            # Обновляем прогресс
                            with self.lock:
                                batch_info['processed_images'] += 1
                                self.active_batches[batch_id] = batch_info
                                
                        except Exception as e:
                            logger.error(f"Ошибка при обработке изображения {i} в пакете {batch_id}: {str(e)}")
                            batch_info['errors'].append({
                                'index': i,
                                'error': str(e)
                            })
                    
                    # Завершение обработки пакета
                    with self.lock:
                        batch_info['status'] = 'completed'
                        batch_info['end_time'] = datetime.now().isoformat()
                        self.completed_batches[batch_id] = batch_info
                        
                        if batch_id in self.active_batches:
                            del self.active_batches[batch_id]
                    
                    logger.info(f"Завершена обработка пакета {batch_id}: "
                               f"{batch_info['processed_images']}/{batch_info['total_images']} изображений, "
                               f"{len(batch_info['errors'])} ошибок")
                    
                    # Отправляем уведомление, если указан URL
                    if batch_info['notify_url']:
                        self._send_notification(batch_info)
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке пакета {batch_id}: {str(e)}")
                    
                    # Обновляем статус в случае ошибки
                    with self.lock:
                        batch_info['status'] = 'failed'
                        batch_info['end_time'] = datetime.now().isoformat()
                        self.completed_batches[batch_id] = batch_info
                        
                        if batch_id in self.active_batches:
                            del self.active_batches[batch_id]
            
            else:
                # Если очередь пуста, ждем немного перед следующей проверкой
                time.sleep(1)
    
    def _send_notification(self, batch_info: Dict[str, Any]):
        """
        Отправка уведомления о завершении обработки пакета.
        
        Args:
            batch_info (Dict[str, Any]): Информация о пакете
        """
        if not batch_info['notify_url']:
            return
        
        try:
            import requests
            
            # Формируем данные для уведомления
            notification_data = {
                'batch_id': batch_info['id'],
                'status': batch_info['status'],
                'total_images': batch_info['total_images'],
                'processed_images': batch_info['processed_images'],
                'errors_count': len(batch_info['errors']),
                'processing_time': (datetime.fromisoformat(batch_info['end_time']) - 
                                  datetime.fromisoformat(batch_info['start_time'])).total_seconds()
            }
            
            # Добавляем callback_id, если указан
            if batch_info['callback_id']:
                notification_data['callback_id'] = batch_info['callback_id']
            
            # Отправляем POST запрос
            response = requests.post(batch_info['notify_url'], json=notification_data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Отправлено уведомление для пакета {batch_info['id']}")
            else:
                logger.warning(f"Ошибка при отправке уведомления для пакета {batch_info['id']}: "
                              f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Не удалось отправить уведомление для пакета {batch_info['id']}: {str(e)}")
    
    def shutdown(self):
        """Завершение работы обработчика пакетов."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)


class TintoraAIAPI:
    """
    Класс для управления API колоризатора TintoraAI.
    Инициализирует FastAPI и определяет маршруты.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = "configs/inference_config.yaml"):
        """
        Инициализация API.
        
        Args:
            checkpoint_path (str): Путь к чекпоинту модели
            config_path (str): Путь к файлу конфигурации
        """
        self.app = FastAPI(
            title="TintoraAI API",
            description="API для колоризации изображений с использованием TintoraAI",
            version="1.0.0"
        )
        
        # Инициализируем модель
        self.model = TintoraAIModel(checkpoint_path, config_path)
        
        # Инициализируем обработчик пакетов
        self.batch_processor = BatchProcessor(self.model)
        
        # Настраиваем CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Определяем маршруты
        self._setup_routes()
    
    def _setup_routes(self):
        """Настройка маршрутов API."""
        
        @self.app.get("/")
        async def root():
            """Корневой маршрут API."""
            return {
                "name": "TintoraAI API",
                "version": "1.0.0",
                "status": "active",
                "model_loaded": self.model.is_loaded
            }
        
        @self.app.get("/health")
        async def health_check():
            """Проверка работоспособности API."""
            model_status = "loaded" if self.model.is_loaded else "loading"
            if self.model.load_error:
                model_status = "error"
                
            return {
                "status": "healthy",
                "model_status": model_status,
                "load_error": self.model.load_error,
                "active_batches": len(self.batch_processor.active_batches),
                "queue_length": len(self.batch_processor.batch_queue),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/colorize", response_model=ColorizationResponse)
        async def colorize(request: ColorizationRequest):
            """
            Колоризация одиночного изображения.
            
            Args:
                request (ColorizationRequest): Запрос на колоризацию
                
            Returns:
                ColorizationResponse: Результат колоризации
            """
            start_time = time.time()
            request_id = str(uuid4())
            
            try:
                # Проверяем, что указан хотя бы один источник изображения
                if not request.image_data and not request.image_url:
                    raise HTTPException(
                        status_code=400,
                        detail="Необходимо указать либо image_data, либо image_url"
                    )
                
                # Загружаем изображение
                if request.image_data:
                    # Из base64
                    try:
                        image_data = request.image_data
                        if image_data.startswith('data:image'):
                            image_data = image_data.split(',')[1]
                        
                        image = np.array(Image.open(BytesIO(base64.b64decode(image_data))))
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Не удалось декодировать изображение: {str(e)}"
                        )
                else:
                    # Из URL (не реализовано)
                    raise HTTPException(
                        status_code=501,
                        detail="Загрузка изображений по URL пока не реализована"
                    )
                
                # Колоризируем изображение
                result = self.model.colorize_image(
                    image_data=image,
                    color_space=request.color_space.value,
                    style=request.style.value if request.style else "default",
                    saturation=request.saturation,
                    contrast=request.contrast,
                    enhance=request.enhance,
                    style_strength=request.style_strength,
                    custom_style_image=None,  # TODO: Реализовать загрузку пользовательского стиля
                    guidance=request.guidance,
                    show_uncertainty=request.show_uncertainty
                )
                
                # Формируем результат
                processing_time = time.time() - start_time
                
                # Кодируем колоризованное изображение в base64
                colorized_pil = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
                buffer = BytesIO()
                colorized_pil.save(buffer, format="PNG")
                colorized_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Формируем ответ
                response = ColorizationResponse(
                    status="success",
                    message="Изображение успешно колоризовано",
                    request_id=request_id,
                    processing_time=processing_time,
                    colorized_image=f"data:image/png;base64,{colorized_base64}",
                    metadata={
                        "style": request.style.value if request.style else "default",
                        "color_space": request.color_space.value,
                        "saturation": request.saturation,
                        "contrast": request.contrast,
                        "enhance": request.enhance,
                        "device": str(self.model.device)
                    }
                )
                
                # Добавляем карту неопределенности, если запрошена
                if request.show_uncertainty and 'uncertainty_map' in result:
                    uncertainty_map = result['uncertainty_map']
                    # Преобразуем карту неопределенности в изображение
                    uncertainty_vis = (uncertainty_map * 255).astype(np.uint8)
                    uncertainty_pil = Image.fromarray(uncertainty_vis)
                    
                    buffer = BytesIO()
                    uncertainty_pil.save(buffer, format="PNG")
                    uncertainty_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    response.uncertainty_map = f"data:image/png;base64,{uncertainty_base64}"
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Ошибка при колоризации: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при колоризации: {str(e)}"
                )
        
        @self.app.post("/batch/colorize")
        async def batch_colorize(request: BatchColorizationRequest):
            """
            Пакетная колоризация изображений.
            
            Args:
                request (BatchColorizationRequest): Запрос на пакетную колоризацию
                
            Returns:
                dict: Информация о созданном пакете
            """
            try:
                # Добавляем пакет в очередь
                batch_id = self.batch_processor.add_batch(
                    images=request.images,
                    is_urls=request.is_urls,
                    style=request.style.value if request.style else "default",
                    color_space=request.color_space.value,
                    saturation=request.saturation,
                    contrast=request.contrast,
                    enhance=request.enhance,
                    notify_url=request.notify_url,
                    callback_id=request.callback_id,
                    priority=request.priority
                )
                
                return {
                    "status": "accepted",
                    "batch_id": batch_id,
                    "message": f"Пакет добавлен в очередь. Для проверки статуса используйте /batch/status/{batch_id}"
                }
                
            except Exception as e:
                logger.error(f"Ошибка при добавлении пакета: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при добавлении пакета: {str(e)}"
                )
        
        @self.app.get("/batch/status/{batch_id}", response_model=BatchProgressResponse)
        async def batch_status(batch_id: str):
            """
            Проверка статуса пакетной обработки.
            
            Args:
                batch_id (str): ID пакета
                
            Returns:
                BatchProgressResponse: Информация о прогрессе
            """
            progress = self.batch_processor.get_batch_progress(batch_id)
            
            if not progress:
                raise HTTPException(
                    status_code=404,
                    detail=f"Пакет с ID {batch_id} не найден"
                )
                
            return BatchProgressResponse(**progress)
        
        @self.app.get("/batch/results/{batch_id}")
        async def batch_results(batch_id: str):
            """
            Получение результатов пакетной обработки.
            
            Args:
                batch_id (str): ID пакета
                
            Returns:
                dict: Результаты обработки
            """
            results = self.batch_processor.get_batch_results(batch_id)
            
            if not results:
                # Проверяем, существует ли пакет
                if self.batch_processor.get_batch_progress(batch_id):
                    raise HTTPException(
                        status_code=202,
                        detail=f"Пакет с ID {batch_id} еще не завершен. "
                               f"Для проверки статуса используйте /batch/status/{batch_id}"
                    )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Пакет с ID {batch_id} не найден"
                    )
                
            return results
        
        @self.app.get("/styles")
        async def get_styles():
            """
            Получение списка доступных стилей.
            
            Returns:
                dict: Список стилей
            """
            return {
                "styles": [
                    {
                        "id": style_id,
                        "name": style_data.get('name', style_id),
                        "description": style_data.get('description', "")
                    }
                    for style_id, style_data in self.model.style_presets.items()
                ]
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """
        Запуск API сервера.
        
        Args:
            host (str): Хост для привязки
            port (int): Порт для привязки
            workers (int): Количество рабочих процессов
        """
        uvicorn.run(self.app, host=host, port=port, workers=workers)


# Точка входа для CLI
def main():
    """Основная функция для запуска API."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TintoraAI API")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Путь к чекпоинту модели")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                      help="Путь к файлу конфигурации")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Хост для привязки API сервера")
    parser.add_argument("--port", type=int, default=8000,
                      help="Порт для привязки API сервера")
    parser.add_argument("--workers", type=int, default=1,
                      help="Количество рабочих процессов")
    
    args = parser.parse_args()
    
    api = TintoraAIAPI(args.checkpoint, args.config)
    api.run(args.host, args.port, args.workers)


# Запуск через SDK
class TintoraAISDK:
    """
    SDK для программного взаимодействия с TintoraAI.
    Предоставляет простые методы для колоризации изображений без запуска API-сервера.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = "configs/inference_config.yaml"):
        """
        Инициализация SDK.
        
        Args:
            checkpoint_path (str): Путь к чекпоинту модели
            config_path (str): Путь к файлу конфигурации
        """
        self.model = TintoraAIModel(checkpoint_path, config_path)
    
    def colorize_image(self, image_path: str, output_path: str = None,
                       style: str = "default", saturation: float = 1.0,
                       contrast: float = 1.0, enhance: bool = False) -> dict:
        """
        Колоризация изображения.
        
        Args:
            image_path (str): Путь к изображению
            output_path (str, optional): Путь для сохранения результата
            style (str): Имя стиля
            saturation (float): Насыщенность
            contrast (float): Контраст
            enhance (bool): Улучшение
            
        Returns:
            dict: Результаты колоризации
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Колоризируем изображение
        result = self.model.colorize_image(
            image_data=image,
            style=style,
            saturation=saturation,
            contrast=contrast,
            enhance=enhance
        )
        
        # Сохраняем результат, если указан путь
        if output_path:
            colorized_img = Image.fromarray((result['colorized'] * 255).astype(np.uint8))
            colorized_img.save(output_path)
        
        return result
    
    def colorize_directory(self, input_dir: str, output_dir: str,
                           style: str = "default", saturation: float = 1.0,
                           contrast: float = 1.0, enhance: bool = False,
                           recursive: bool = False) -> dict:
        """
        Колоризация всех изображений в директории.
        
        Args:
            input_dir (str): Путь к директории с изображениями
            output_dir (str): Путь для сохранения результатов
            style (str): Имя стиля
            saturation (float): Насыщенность
            contrast (float): Контраст
            enhance (bool): Улучшение
            recursive (bool): Рекурсивный поиск изображений
            
        Returns:
            dict: Статистика обработки
        """
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Поддерживаемые форматы изображений
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Находим изображения
        image_files = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        image_files.append(file_path)
        else:
            for ext in supported_extensions:
                image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
        
        if not image_files:
            return {"status": "error", "message": "Изображения не найдены"}
        
        # Статистика обработки
        stats = {
            "total": len(image_files),
            "processed": 0,
            "success": 0,
            "errors": []
        }
        
        # Обрабатываем каждое изображение
        for image_path in image_files:
            try:
                # Формируем путь для сохранения результата
                rel_path = os.path.relpath(image_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Создаем директории, если нужно
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Колоризируем изображение
                self.colorize_image(
                    image_path=image_path,
                    output_path=output_path,
                    style=style,
                    saturation=saturation,
                    contrast=contrast,
                    enhance=enhance
                )
                
                stats["processed"] += 1
                stats["success"] += 1
                
            except Exception as e:
                stats["processed"] += 1
                stats["errors"].append({
                    "file": image_path,
                    "error": str(e)
                })
        
        return stats


if __name__ == "__main__":
    main()
