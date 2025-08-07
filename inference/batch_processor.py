"""
Batch Processor: Модуль для эффективной пакетной обработки изображений.

Данный модуль предоставляет функциональность для обработки больших наборов изображений
с использованием оптимизаций для эффективного использования ресурсов и мониторинга прогресса.
Он поддерживает различные режимы работы, включая параллельную обработку, обработку очередей
и автоматическую обработку новых файлов.

Ключевые особенности:
- Эффективная пакетная обработка с использованием батчей оптимального размера
- Параллельная обработка для максимального использования доступных ресурсов
- Мониторинг и отчетность о процессе обработки в реальном времени
- Обработка очередей с автоматическим подхватом новых файлов
- Возможность паузы и возобновления процесса обработки

Преимущества:
- Значительное ускорение обработки больших наборов изображений
- Оптимальное использование ресурсов GPU и CPU
- Гибкость настройки через конфигурационные параметры
- Встроенная система мониторинга и отчетности
"""

import os
import time
import json
import logging
import traceback
import threading
import queue
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image
import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .predictor import ColorizationPredictor, load_predictor, FallbackStrategy
from .postprocessor import ColorizationPostprocessor, create_postprocessor
from utils.config_parser import load_config, ConfigParser


class ProcessingMode(Enum):
    """Режимы обработки изображений."""
    SEQUENTIAL = "sequential"  # Последовательная обработка
    PARALLEL_THREAD = "parallel_thread"  # Параллельная обработка на уровне потоков
    PARALLEL_PROCESS = "parallel_process"  # Параллельная обработка на уровне процессов
    BATCH_SEQUENTIAL = "batch_sequential"  # Последовательная обработка батчами
    BATCH_PARALLEL = "batch_parallel"  # Параллельная обработка батчами
    QUEUE = "queue"  # Обработка очереди


class BatchProcessingStats:
    """
    Класс для отслеживания статистики пакетной обработки.
    """
    def __init__(self):
        self.total_images = 0
        self.processed_images = 0
        self.successful_images = 0
        self.failed_images = 0
        self.skipped_images = 0
        self.total_time = 0.0
        self.processing_times = []
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Отмечает начало обработки."""
        self.start_time = time.time()
        
    def finish(self):
        """Отмечает окончание обработки."""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        
    def update(self, status: str, processing_time: float):
        """
        Обновляет статистику на основе результата обработки изображения.
        
        Args:
            status (str): Статус обработки ('success', 'error', 'skipped')
            processing_time (float): Время обработки в секундах
        """
        self.processed_images += 1
        self.processing_times.append(processing_time)
        
        if status == 'success':
            self.successful_images += 1
        elif status == 'error':
            self.failed_images += 1
        elif status == 'skipped':
            self.skipped_images += 1
            
    def get_stats(self) -> Dict:
        """
        Возвращает текущую статистику обработки.
        
        Returns:
            Dict: Статистика обработки
        """
        stats = {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "skipped_images": self.skipped_images,
            "completion_percentage": (self.processed_images / max(1, self.total_images)) * 100
        }
        
        # Добавляем временные статистики
        if self.start_time is not None:
            stats["elapsed_time"] = time.time() - self.start_time
            
        if len(self.processing_times) > 0:
            stats["avg_processing_time"] = sum(self.processing_times) / len(self.processing_times)
            stats["min_processing_time"] = min(self.processing_times)
            stats["max_processing_time"] = max(self.processing_times)
            
        if self.end_time is not None:
            stats["total_time"] = self.total_time
            stats["images_per_second"] = self.processed_images / max(0.001, self.total_time)
            
        return stats


class BatchProcessor:
    """
    Процессор для пакетной обработки изображений.
    
    Args:
        predictor (ColorizationPredictor): Предиктор для колоризации
        postprocessor (Optional[ColorizationPostprocessor]): Постпроцессор для улучшения результатов
        batch_size (int): Размер батча для обработки
        mode (ProcessingMode): Режим обработки
        max_workers (int): Максимальное количество рабочих потоков/процессов
        config (Dict): Конфигурация обработки
    """
    def __init__(
        self,
        predictor: ColorizationPredictor,
        postprocessor: Optional[ColorizationPostprocessor] = None,
        batch_size: int = 4,
        mode: ProcessingMode = ProcessingMode.BATCH_SEQUENTIAL,
        max_workers: int = 4,
        config: Dict = None
    ):
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.batch_size = batch_size
        self.mode = mode
        self.max_workers = max_workers
        self.config = config or {}
        
        # Настройки пакетной обработки
        self.output_dir = config.get('output_dir', './output')
        self.save_comparison = config.get('save_comparison', True)
        self.save_uncertainty = config.get('save_uncertainty', False)
        self.postprocess_enabled = config.get('postprocess', True) and postprocessor is not None
        self.overwrite_existing = config.get('overwrite_existing', False)
        
        # Создаем директории для результатов
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "colorized"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
        # Настраиваем логирование
        self.logger = logging.getLogger("BatchProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Добавляем файловый обработчик
            os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.output_dir, "logs", "batch_processor.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Инициализирован BatchProcessor с режимом {mode.value}")
        
        # Статистика обработки
        self.stats = BatchProcessingStats()
        
        # Флаг паузы
        self.paused = False
        self.stop_requested = False
        
    def process_directory(
        self,
        input_dir: str,
        recursive: bool = False,
        extensions: List[str] = None,
        reference_dir: Optional[str] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Обрабатывает все изображения в указанной директории.
        
        Args:
            input_dir (str): Директория с изображениями для обработки
            recursive (bool): Искать ли изображения в поддиректориях
            extensions (List[str], optional): Список расширений файлов для обработки
            reference_dir (str, optional): Директория с референсными изображениями
            style_name (str, optional): Имя стиля для применения
            style_alpha (float, optional): Интенсивность применения стиля
            metadata (Dict, optional): Дополнительные метаданные для сохранения
            
        Returns:
            Dict: Результаты обработки
        """
        # Устанавливаем список расширений по умолчанию
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        else:
            # Добавляем точку в начало, если отсутствует
            extensions = [ext if ext.startswith('.') else f".{ext}" for ext in extensions]
            
        # Находим все файлы изображений
        image_paths = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for ext in extensions:
                image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
                image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
                
        # Сортируем пути для порядка обработки
        image_paths = sorted([str(path) for path in image_paths])
        
        if not image_paths:
            self.logger.warning(f"В директории {input_dir} не найдено изображений с расширениями {extensions}")
            return {"status": "warning", "message": "Не найдено изображений", "processed": 0}
            
        self.logger.info(f"Найдено {len(image_paths)} изображений для обработки")
        
        # Находим соответствующие референсные изображения, если указана директория
        reference_paths = None
        if reference_dir is not None:
            reference_paths = []
            for path in image_paths:
                # Извлекаем имя файла
                filename = os.path.basename(path)
                
                # Ищем соответствующее референсное изображение
                reference_path = os.path.join(reference_dir, filename)
                
                # Если не найдено, пробуем с другими расширениями
                if not os.path.exists(reference_path):
                    base_name = os.path.splitext(filename)[0]
                    found = False
                    for ext in extensions:
                        test_path = os.path.join(reference_dir, f"{base_name}{ext}")
                        if os.path.exists(test_path):
                            reference_path = test_path
                            found = True
                            break
                            
                    if not found:
                        reference_path = None
                        
                reference_paths.append(reference_path)
        
        # Инициализируем статистику
        self.stats = BatchProcessingStats()
        self.stats.total_images = len(image_paths)
        self.stats.start()
        
        # Выбираем режим обработки
        if self.mode == ProcessingMode.SEQUENTIAL:
            results = self._process_sequential(
                image_paths, reference_paths, style_name, style_alpha, metadata
            )
        elif self.mode == ProcessingMode.PARALLEL_THREAD:
            results = self._process_parallel_thread(
                image_paths, reference_paths, style_name, style_alpha, metadata
            )
        elif self.mode == ProcessingMode.PARALLEL_PROCESS:
            results = self._process_parallel_process(
                image_paths, reference_paths, style_name, style_alpha, metadata
            )
        elif self.mode == ProcessingMode.BATCH_SEQUENTIAL:
            results = self._process_batch_sequential(
                image_paths, reference_paths, style_name, style_alpha, metadata
            )
        elif self.mode == ProcessingMode.BATCH_PARALLEL:
            results = self._process_batch_parallel(
                image_paths, reference_paths, style_name, style_alpha, metadata
            )
        else:
            raise ValueError(f"Неподдерживаемый режим обработки: {self.mode}")
            
        # Завершаем статистику
        self.stats.finish()
        
        # Сохраняем общий отчет
        report = {
            "status": "success",
            "input_dir": input_dir,
            "output_dir": self.output_dir,
            "processed_images": self.stats.processed_images,
            "successful_images": self.stats.successful_images,
            "failed_images": self.stats.failed_images,
            "skipped_images": self.stats.skipped_images,
            "total_time": self.stats.total_time,
            "images_per_second": self.stats.processed_images / max(0.001, self.stats.total_time),
            "results": results
        }
        
        # Сохраняем отчет в файл
        report_path = os.path.join(self.output_dir, "metadata", f"batch_report_{time.strftime('%Y%m%d-%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Обработка завершена: {report['successful_images']} успешно, {report['failed_images']} с ошибками, {report['skipped_images']} пропущено")
        self.logger.info(f"Общее время: {report['total_time']:.2f} сек, скорость: {report['images_per_second']:.2f} изобр./сек")
        
        return report
        
    def pause(self):
        """Приостанавливает обработку."""
        self.paused = True
        self.logger.info("Обработка приостановлена")
        
    def resume(self):
        """Возобновляет обработку."""
        self.paused = False
        self.logger.info("Обработка возобновлена")
        
    def stop(self):
        """Останавливает обработку."""
        self.stop_requested = True
        self.logger.info("Запрошена остановка обработки")
        
    def _process_sequential(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]],
        style_name: Optional[str],
        style_alpha: Optional[float],
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Последовательно обрабатывает изображения.
        
        Args:
            image_paths (List[str]): Пути к изображениям
            reference_paths (List[str], optional): Пути к референсным изображениям
            style_name (str, optional): Имя стиля
            style_alpha (float, optional): Интенсивность стиля
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            List[Dict]: Результаты обработки
        """
        results = []
        
        # Создаем прогресс-бар
        with tqdm.tqdm(total=len(image_paths), desc="Обработка изображений") as pbar:
            # Обрабатываем каждое изображение
            for i, image_path in enumerate(image_paths):
                # Проверяем остановку
                if self.stop_requested:
                    self.logger.info("Обработка остановлена пользователем")
                    break
                    
                # Проверяем паузу
                while self.paused and not self.stop_requested:
                    time.sleep(0.1)
                    
                # Проверяем, существует ли уже выходной файл
                output_path = self._get_output_path(image_path)
                if not self.overwrite_existing and os.path.exists(output_path):
                    self.logger.debug(f"Пропуск существующего файла: {output_path}")
                    self.stats.update('skipped', 0.0)
                    pbar.update(1)
                    continue
                    
                # Получаем референсное изображение, если есть
                reference_image = None
                if reference_paths is not None and i < len(reference_paths) and reference_paths[i] is not None:
                    try:
                        reference_image = Image.open(reference_paths[i]).convert('RGB')
                    except Exception as e:
                        self.logger.warning(f"Не удалось загрузить референсное изображение {reference_paths[i]}: {e}")
                        
                # Выполняем колоризацию
                start_time = time.time()
                try:
                    # Загружаем изображение
                    image = Image.open(image_path).convert('RGB')
                    
                    # Создаем метаданные для изображения
                    image_metadata = {"source_path": image_path}
                    if metadata:
                        image_metadata.update(metadata)
                        
                    # Выполняем колоризацию
                    result = self._process_single_image(
                        image, reference_image, style_name, style_alpha, 
                        output_path, image_metadata
                    )
                    
                    processing_time = time.time() - start_time
                    self.stats.update('success', processing_time)
                    result['processing_time'] = processing_time
                    results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    self.logger.error(f"Ошибка при обработке {image_path}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    self.stats.update('error', processing_time)
                    results.append({
                        "status": "error",
                        "error_message": str(e),
                        "source_path": image_path,
                        "processing_time": processing_time
                    })
                    
                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_postfix({
                    'успешно': self.stats.successful_images,
                    'ошибки': self.stats.failed_images,
                    'скорость': f"{1.0/max(0.001, processing_time):.1f} изобр./сек"
                })
                
        return results
        
    def _process_parallel_thread(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]],
        style_name: Optional[str],
        style_alpha: Optional[float],
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Параллельно обрабатывает изображения, используя потоки.
        
        Args:
            image_paths (List[str]): Пути к изображениям
            reference_paths (List[str], optional): Пути к референсным изображениям
            style_name (str, optional): Имя стиля
            style_alpha (float, optional): Интенсивность стиля
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            List[Dict]: Результаты обработки
        """
        results = []
        futures = []
        
        # Определяем количество рабочих потоков
        max_workers = min(self.max_workers, len(image_paths))
        
        # Создаем пул потоков
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем прогресс-бар
            with tqdm.tqdm(total=len(image_paths), desc="Обработка изображений") as pbar:
                # Подготавливаем задачи для каждого изображения
                for i, image_path in enumerate(image_paths):
                    # Проверяем, существует ли уже выходной файл
                    output_path = self._get_output_path(image_path)
                    if not self.overwrite_existing and os.path.exists(output_path):
                        self.logger.debug(f"Пропуск существующего файла: {output_path}")
                        self.stats.update('skipped', 0.0)
                        pbar.update(1)
                        continue
                        
                    # Получаем референсное изображение, если есть
                    reference_path = None
                    if reference_paths is not None and i < len(reference_paths):
                        reference_path = reference_paths[i]
                        
                    # Создаем метаданные для изображения
                    image_metadata = {"source_path": image_path}
                    if metadata:
                        image_metadata.update(metadata)
                        
                    # Добавляем задачу в пул
                    future = executor.submit(
                        self._process_single_image_from_path,
                        image_path, reference_path, style_name, style_alpha, 
                        output_path, image_metadata
                    )
                    futures.append(future)
                    
                # Обрабатываем результаты по мере их готовности
                for future in as_completed(futures):
                    # Проверяем остановку
                    if self.stop_requested:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        self.logger.info("Обработка остановлена пользователем")
                        break
                        
                    # Получаем результат
                    try:
                        result = future.result()
                        if result.get('status') == 'success':
                            self.stats.update('success', result.get('processing_time', 0.0))
                        else:
                            self.stats.update('error', result.get('processing_time', 0.0))
                    except Exception as e:
                        self.logger.error(f"Ошибка при получении результата: {str(e)}")
                        self.stats.update('error', 0.0)
                        result = {
                            "status": "error",
                            "error_message": str(e),
                            "processing_time": 0.0
                        }
                        
                    # Добавляем результат
                    results.append(result)
                    
                    # Обновляем прогресс-бар
                    pbar.update(1)
                    pbar.set_postfix({
                        'успешно': self.stats.successful_images,
                        'ошибки': self.stats.failed_images,
                        'скорость': f"{len(results)/max(0.001, time.time() - self.stats.start_time):.1f} изобр./сек"
                    })
                    
        return results
        
    def _process_parallel_process(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]],
        style_name: Optional[str],
        style_alpha: Optional[float],
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Параллельно обрабатывает изображения, используя процессы.
        
        Args:
            image_paths (List[str]): Пути к изображениям
            reference_paths (List[str], optional): Пути к референсным изображениям
            style_name (str, optional): Имя стиля
            style_alpha (float, optional): Интенсивность стиля
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            List[Dict]: Результаты обработки
        """
        # Примечание: Для использования процессов мы не можем напрямую использовать self.predictor,
        # так как он содержит модели PyTorch, которые не могут быть переданы между процессами.
        # Вместо этого мы будем использовать внешнюю функцию, которая создаст новый экземпляр предиктора.
        
        # Создаем аргументы для каждого процесса
        tasks = []
        for i, image_path in enumerate(image_paths):
            # Проверяем, существует ли уже выходной файл
            output_path = self._get_output_path(image_path)
            if not self.overwrite_existing and os.path.exists(output_path):
                self.logger.debug(f"Пропуск существующего файла: {output_path}")
                self.stats.update('skipped', 0.0)
                continue
                
            # Получаем референсное изображение, если есть
            reference_path = None
            if reference_paths is not None and i < len(reference_paths):
                reference_path = reference_paths[i]
                
            # Создаем метаданные для изображения
            image_metadata = {"source_path": image_path}
            if metadata:
                image_metadata.update(metadata)
                
            # Добавляем задачу
            task = {
                'image_path': image_path,
                'reference_path': reference_path,
                'style_name': style_name,
                'style_alpha': style_alpha,
                'output_path': output_path,
                'metadata': image_metadata,
                'config': {
                    'model_path': self.config.get('model_path'),
                    'config_path': self.config.get('config_path'),
                    'modules_path': self.config.get('modules_path'),
                    'device': self.config.get('device', 'cuda'),
                    'postprocess': self.postprocess_enabled,
                    'save_comparison': self.save_comparison,
                    'save_uncertainty': self.save_uncertainty
                }
            }
            tasks.append(task)
            
        # Определяем количество рабочих процессов
        max_workers = min(self.max_workers, len(tasks), multiprocessing.cpu_count())
        
        # Создаем пул процессов
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Создаем прогресс-бар
            with tqdm.tqdm(total=len(tasks), desc="Обработка изображений") as pbar:
                # Запускаем задачи
                futures = [executor.submit(process_image_in_process, task) for task in tasks]
                
                # Обрабатываем результаты
                results = []
                for future in as_completed(futures):
                    # Получаем результат
                    try:
                        result = future.result()
                        if result.get('status') == 'success':
                            self.stats.update('success', result.get('processing_time', 0.0))
                        else:
                            self.stats.update('error', result.get('processing_time', 0.0))
                    except Exception as e:
                        self.logger.error(f"Ошибка при получении результата: {str(e)}")
                        self.stats.update('error', 0.0)
                        result = {
                            "status": "error",
                            "error_message": str(e),
                            "processing_time": 0.0
                        }
                        
                    # Добавляем результат
                    results.append(result)
                    
                    # Обновляем прогресс-бар
                    pbar.update(1)
                    pbar.set_postfix({
                        'успешно': self.stats.successful_images,
                        'ошибки': self.stats.failed_images,
                        'скорость': f"{len(results)/max(0.001, time.time() - self.stats.start_time):.1f} изобр./сек"
                    })
                    
        return results
        
    def _process_batch_sequential(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]],
        style_name: Optional[str],
        style_alpha: Optional[float],
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Последовательно обрабатывает изображения батчами.
        
        Args:
            image_paths (List[str]): Пути к изображениям
            reference_paths (List[str], optional): Пути к референсным изображениям
            style_name (str, optional): Имя стиля
            style_alpha (float, optional): Интенсивность стиля
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            List[Dict]: Результаты обработки
        """
        results = []
        
        # Создаем прогресс-бар
        with tqdm.tqdm(total=len(image_paths), desc="Обработка изображений") as pbar:
            # Обрабатываем изображения батчами
            for i in range(0, len(image_paths), self.batch_size):
                # Проверяем остановку
                if self.stop_requested:
                    self.logger.info("Обработка остановлена пользователем")
                    break
                    
                # Проверяем паузу
                while self.paused and not self.stop_requested:
                    time.sleep(0.1)
                    
                # Получаем батч изображений
                batch_indices = list(range(i, min(i + self.batch_size, len(image_paths))))
                batch_paths = [image_paths[j] for j in batch_indices]
                
                # Фильтруем существующие файлы
                if not self.overwrite_existing:
                    filtered_indices = []
                    filtered_paths = []
                    for j, path in zip(batch_indices, batch_paths):
                        output_path = self._get_output_path(path)
                        if os.path.exists(output_path):
                            self.logger.debug(f"Пропуск существующего файла: {output_path}")
                            self.stats.update('skipped', 0.0)
                            pbar.update(1)
                        else:
                            filtered_indices.append(j)
                            filtered_paths.append(path)
                    
                    batch_indices = filtered_indices
                    batch_paths = filtered_paths
                    
                # Пропускаем пустой батч
                if not batch_paths:
                    continue
                    
                # Получаем референсные изображения, если есть
                batch_references = None
                if reference_paths is not None:
                    batch_references = [
                        reference_paths[j] if j < len(reference_paths) else None
                        for j in batch_indices
                    ]
                    
                # Создаем метаданные для батча
                batch_metadata = None
                if metadata:
                    batch_metadata = [{"source_path": path, **metadata} for path in batch_paths]
                    
                # Выполняем колоризацию батча
                start_time = time.time()
                try:
                    batch_results = self._process_batch_images(
                        batch_paths, batch_references, style_name, style_alpha,
                        [self._get_output_path(path) for path in batch_paths], batch_metadata
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Обновляем статистику
                    for result in batch_results:
                        if result.get('status') == 'success':
                            self.stats.update('success', processing_time / len(batch_results))
                        else:
                            self.stats.update('error', processing_time / len(batch_results))
                            
                    # Добавляем результаты
                    results.extend(batch_results)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    self.logger.error(f"Ошибка при обработке батча: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    # Обновляем статистику для всех изображений в батче
                    for _ in batch_paths:
                        self.stats.update('error', processing_time / len(batch_paths))
                        
                    # Добавляем результаты с ошибкой
                    for path in batch_paths:
                        results.append({
                            "status": "error",
                            "error_message": str(e),
                            "source_path": path,
                            "processing_time": processing_time / len(batch_paths)
                        })
                        
                # Обновляем прогресс-бар
                pbar.update(len(batch_paths))
                pbar.set_postfix({
                    'успешно': self.stats.successful_images,
                    'ошибки': self.stats.failed_images,
                    'скорость': f"{len(batch_paths)/max(0.001, processing_time):.1f} изобр./сек"
                })
                
        return results
        
    def _process_batch_parallel(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]],
        style_name: Optional[str],
        style_alpha: Optional[float],
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Параллельно обрабатывает изображения батчами.
        
        Args:
            image_paths (List[str]): Пути к изображениям
            reference_paths (List[str], optional): Пути к референсным изображениям
            style_name (str, optional): Имя стиля
            style_alpha (float, optional): Интенсивность стиля
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            List[Dict]: Результаты обработки
        """
        results = []
        futures = []
        
        # Определяем количество рабочих потоков
        max_workers = min(self.max_workers, (len(image_paths) + self.batch_size - 1) // self.batch_size)
        
        # Создаем пул потоков
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем прогресс-бар
            with tqdm.tqdm(total=len(image_paths), desc="Обработка изображений") as pbar:
                # Обрабатываем изображения батчами
                for i in range(0, len(image_paths), self.batch_size):
                    # Получаем батч изображений
                    batch_indices = list(range(i, min(i + self.batch_size, len(image_paths))))
                    batch_paths = [image_paths[j] for j in batch_indices]
                    
                    # Фильтруем существующие файлы
                    if not self.overwrite_existing:
                        filtered_indices = []
                        filtered_paths = []
                        for j, path in zip(batch_indices, batch_paths):
                            output_path = self._get_output_path(path)
                            if os.path.exists(output_path):
                                self.logger.debug(f"Пропуск существующего файла: {output_path}")
                                self.stats.update('skipped', 0.0)
                                pbar.update(1)
                            else:
                                filtered_indices.append(j)
                                filtered_paths.append(path)
                        
                        batch_indices = filtered_indices
                        batch_paths = filtered_paths
                        
                    # Пропускаем пустой батч
                    if not batch_paths:
                        continue
                        
                    # Получаем референсные изображения, если есть
                    batch_references = None
                    if reference_paths is not None:
                        batch_references = [
                            reference_paths[j] if j < len(reference_paths) else None
                            for j in batch_indices
                        ]
                        
                    # Создаем метаданные для батча
                    batch_metadata = None
                    if metadata:
                        batch_metadata = [{"source_path": path, **metadata} for path in batch_paths]
                        
                    # Добавляем задачу в пул
                    future = executor.submit(
                        self._process_batch_images_from_paths,
                        batch_paths, batch_references, style_name, style_alpha,
                        [self._get_output_path(path) for path in batch_paths], batch_metadata
                    )
                    futures.append((future, len(batch_paths)))
                    
                # Обрабатываем результаты по мере их готовности
                for future, batch_size in futures:
                    # Проверяем остановку
                    if self.stop_requested:
                        for f, _ in futures:
                            if not f.done():
                                f.cancel()
                        self.logger.info("Обработка остановлена пользователем")
                        break
                        
                    # Получаем результат
                    try:
                        batch_results = future.result()
                        
                        # Обновляем статистику
                        for result in batch_results:
                            if result.get('status') == 'success':
                                self.stats.update('success', result.get('processing_time', 0.0))
                            else:
                                self.stats.update('error', result.get('processing_time', 0.0))
                                
                        # Добавляем результаты
                        results.extend(batch_results)
                        
                    except Exception as e:
                        self.logger.error(f"Ошибка при получении результата батча: {str(e)}")
                        
                        # Обновляем статистику для всех изображений в батче
                        for _ in range(batch_size):
                            self.stats.update('error', 0.0)
                            
                        # Добавляем результаты с ошибкой
                        for _ in range(batch_size):
                            results.append({
                                "status": "error",
                                "error_message": str(e),
                                "processing_time": 0.0
                            })
                            
                    # Обновляем прогресс-бар
                    pbar.update(batch_size)
                    pbar.set_postfix({
                        'успешно': self.stats.successful_images,
                        'ошибки': self.stats.failed_images,
                        'скорость': f"{len(results)/max(0.001, time.time() - self.stats.start_time):.1f} изобр./сек"
                    })
                    
        return results
        
    def _process_single_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        reference_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        output_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Обрабатывает одно изображение.
        
        Args:
            image: Входное изображение
            reference_image: Референсное изображение для стиля
            style_name: Имя стиля
            style_alpha: Интенсивность стиля
            output_path: Путь для сохранения результата
            metadata: Дополнительные метаданные
            
        Returns:
            Dict: Результат обработки
        """
        # Выполняем колоризацию
        result = self.predictor.colorize(
            image=image,
            reference_image=reference_image,
            style_name=style_name,
            style_alpha=style_alpha,
            output_path=output_path,
            save_comparison=self.save_comparison,
            save_uncertainty=self.save_uncertainty,
            metadata=metadata
        )
        
        # Если колоризация успешна и включена постобработка, применяем её
        if result.get('status') == 'success' and self.postprocess_enabled and self.postprocessor is not None:
            try:
                # Загружаем колоризованное изображение
                colorized_path = result.get('output_path')
                colorized_image = Image.open(colorized_path).convert('RGB')
                
                # Получаем оригинальное ЧБ изображение, если оно есть
                grayscale_image = image if isinstance(image, Image.Image) else None
                
                # Применяем постобработку
                postprocess_result = self.postprocessor.process_image(
                    image=colorized_image,
                    grayscale_image=grayscale_image,
                    output_path=colorized_path,  # Перезаписываем колоризованное изображение
                    save_comparison=self.save_comparison,
                    metadata=metadata
                )
                
                # Обновляем результат
                result['postprocessed'] = True
                result['postprocess_info'] = postprocess_result.get('applied_operations', {})
                
                # Если было создано сравнение, добавляем его в результат
                if 'comparison_path' in postprocess_result and postprocess_result['comparison_path']:
                    result['postprocess_comparison_path'] = postprocess_result['comparison_path']
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при постобработке: {str(e)}")
                result['postprocessed'] = False
                result['postprocess_error'] = str(e)
                
        return result
        
    def _process_single_image_from_path(
        self,
        image_path: str,
        reference_path: Optional[str] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        output_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Обрабатывает одно изображение по пути.
        
        Args:
            image_path: Путь к входному изображению
            reference_path: Путь к референсному изображению
            style_name: Имя стиля
            style_alpha: Интенсивность стиля
            output_path: Путь для сохранения результата
            metadata: Дополнительные метаданные
            
        Returns:
            Dict: Результат обработки
        """
        start_time = time.time()
        
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Загружаем референсное изображение, если есть
            reference_image = None
            if reference_path is not None:
                try:
                    reference_image = Image.open(reference_path).convert('RGB')
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить референсное изображение {reference_path}: {e}")
                    
            # Выполняем обработку
            result = self._process_single_image(
                image, reference_image, style_name, style_alpha, output_path, metadata
            )
            
            # Добавляем время обработки
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_path}: {str(e)}")
            
            return {
                "status": "error",
                "error_message": str(e),
                "source_path": image_path,
                "processing_time": time.time() - start_time
            }
            
    def _process_batch_images(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        reference_images: Optional[List[Union[np.ndarray, torch.Tensor, Image.Image]]] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        output_paths: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Обрабатывает пакет изображений.
        
        Args:
            images: Список входных изображений
            reference_images: Список референсных изображений
            style_name: Имя стиля
            style_alpha: Интенсивность стиля
            output_paths: Список путей для сохранения результатов
            metadata: Список метаданных для каждого изображения
            
        Returns:
            List[Dict]: Результаты обработки
        """
        # Выполняем колоризацию пакета
        results = self.predictor.colorize_batch(
            images=images,
            reference_images=reference_images,
            style_names=[style_name] * len(images) if style_name else None,
            style_alphas=[style_alpha] * len(images) if style_alpha else None,
            output_dir=self.output_dir,
            save_comparison=self.save_comparison,
            save_uncertainty=self.save_uncertainty,
            metadata=metadata
        )
        
        # Если включена постобработка, применяем её
        if self.postprocess_enabled and self.postprocessor is not None:
            for i, result in enumerate(results):
                if result.get('status') == 'success':
                    try:
                        # Загружаем колоризованное изображение
                        colorized_path = result.get('output_path')
                        colorized_image = Image.open(colorized_path).convert('RGB')
                        
                        # Получаем оригинальное ЧБ изображение, если оно есть
                        grayscale_image = images[i] if isinstance(images[i], Image.Image) else None
                        
                        # Применяем постобработку
                        postprocess_result = self.postprocessor.process_image(
                            image=colorized_image,
                            grayscale_image=grayscale_image,
                            output_path=colorized_path,  # Перезаписываем колоризованное изображение
                            save_comparison=self.save_comparison,
                            metadata=metadata[i] if metadata else None
                        )
                        
                        # Обновляем результат
                        result['postprocessed'] = True
                        result['postprocess_info'] = postprocess_result.get('applied_operations', {})
                        
                        # Если было создано сравнение, добавляем его в результат
                        if 'comparison_path' in postprocess_result and postprocess_result['comparison_path']:
                            result['postprocess_comparison_path'] = postprocess_result['comparison_path']
                            
                    except Exception as e:
                        self.logger.warning(f"Ошибка при постобработке: {str(e)}")
                        result['postprocessed'] = False
                        result['postprocess_error'] = str(e)
                        
        return results
        
    def _process_batch_images_from_paths(
        self,
        image_paths: List[str],
        reference_paths: Optional[List[str]] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        output_paths: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Обрабатывает пакет изображений по путям.
        
        Args:
            image_paths: Список путей к входным изображениям
            reference_paths: Список путей к референсным изображениям
            style_name: Имя стиля
            style_alpha: Интенсивность стиля
            output_paths: Список путей для сохранения результатов
            metadata: Список метаданных для каждого изображения
            
        Returns:
            List[Dict]: Результаты обработки
        """
        start_time = time.time()
        
        try:
            # Загружаем изображения
            images = []
            for path in image_paths:
                try:
                    images.append(Image.open(path).convert('RGB'))
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить изображение {path}: {e}")
                    images.append(None)
            
            # Загружаем референсные изображения, если есть
            reference_images = None
            if reference_paths is not None:
                reference_images = []
                for path in reference_paths:
                    if path is not None:
                        try:
                            reference_images.append(Image.open(path).convert('RGB'))
                        except Exception as e:
                            self.logger.warning(f"Не удалось загрузить референсное изображение {path}: {e}")
                            reference_images.append(None)
                    else:
                        reference_images.append(None)
            
            # Отфильтровываем неудачно загруженные изображения
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            valid_images = [images[i] for i in valid_indices]
            valid_reference_images = None if reference_images is None else [reference_images[i] for i in valid_indices]
            valid_output_paths = None if output_paths is None else [output_paths[i] for i in valid_indices]
            valid_metadata = None if metadata is None else [metadata[i] for i in valid_indices]
            
            # Добавляем результаты для неудачно загруженных изображений
            results = []
            for i in range(len(images)):
                if i not in valid_indices:
                    results.append({
                        "status": "error",
                        "error_message": "Не удалось загрузить изображение",
                        "source_path": image_paths[i],
                        "processing_time": 0.0
                    })
            
            # Выполняем обработку для валидных изображений
            if valid_images:
                batch_results = self._process_batch_images(
                    valid_images, valid_reference_images, style_name, style_alpha, 
                    valid_output_paths, valid_metadata
                )
                
                # Вычисляем время обработки для каждого изображения
                processing_time = (time.time() - start_time) / len(valid_images)
                
                # Добавляем время обработки к каждому результату
                for result in batch_results:
                    result['processing_time'] = processing_time
                    
                # Объединяем результаты
                for i, idx in enumerate(valid_indices):
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = batch_results[i]
            
            # Заполняем пропуски
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = {
                        "status": "error",
                        "error_message": "Результат обработки отсутствует",
                        "source_path": image_paths[i] if i < len(image_paths) else "unknown",
                        "processing_time": 0.0
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке батча: {str(e)}")
            
            # Возвращаем ошибку для всех изображений
            return [
                {
                    "status": "error",
                    "error_message": str(e),
                    "source_path": path,
                    "processing_time": (time.time() - start_time) / len(image_paths)
                }
                for path in image_paths
            ]
            
    def _get_output_path(self, input_path: str) -> str:
        """
        Генерирует путь для сохранения результата.
        
        Args:
            input_path (str): Путь к входному изображению
            
        Returns:
            str: Путь для сохранения результата
        """
        # Извлекаем имя файла
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        
        # Создаем путь для сохранения
        output_path = os.path.join(self.output_dir, "colorized", f"{base_name}_colorized.png")
        
        return output_path


class QueueEventHandler(FileSystemEventHandler):
    """
    Обработчик событий файловой системы для мониторинга новых файлов в очереди.
    
    Args:
        queue_processor (QueueProcessor): Процессор очереди
    """
    def __init__(self, queue_processor):
        self.queue_processor = queue_processor
        
    def on_created(self, event):
        """
        Обрабатывает событие создания файла.
        
        Args:
            event (FileCreatedEvent): Событие создания файла
        """
        # Проверяем, что это файл, а не директория
        if not event.is_directory:
            # Проверяем расширение файла
            file_path = event.src_path
            if any(file_path.lower().endswith(ext) for ext in self.queue_processor.extensions):
                # Добавляем файл в очередь
                self.queue_processor.add_to_queue(file_path)


class QueueProcessor:
    """
    Процессор для обработки очереди изображений.
    
    Args:
        batch_processor (BatchProcessor): Процессор пакетной обработки
        input_dir (str): Директория для входных изображений
        output_dir (str): Директория для результатов
        poll_interval (float): Интервал проверки очереди в секундах
        auto_start (bool): Автоматически запускать обработку при инициализации
        extensions (List[str], optional): Список расширений файлов для обработки
        reference_dir (str, optional): Директория с референсными изображениями
        style_name (str, optional): Имя стиля для применения
        style_alpha (float, optional): Интенсивность применения стиля
    """
    def __init__(
        self,
        batch_processor: BatchProcessor,
        input_dir: str,
        output_dir: str,
        poll_interval: float = 2.0,
        auto_start: bool = True,
        extensions: List[str] = None,
        reference_dir: Optional[str] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None
    ):
        self.batch_processor = batch_processor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.reference_dir = reference_dir
        self.style_name = style_name
        self.style_alpha = style_alpha
        
        # Добавляем точку в начало расширений, если отсутствует
        self.extensions = [ext if ext.startswith('.') else f".{ext}" for ext in self.extensions]
        
        # Создаем директории
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "colorized"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Создаем директорию для обработанных файлов
        self.processed_dir = os.path.join(input_dir, "_processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Создаем директорию для файлов с ошибками
        self.error_dir = os.path.join(input_dir, "_errors")
        os.makedirs(self.error_dir, exist_ok=True)
        
        # Настраиваем логирование
        self.logger = logging.getLogger("QueueProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Добавляем файловый обработчик
            os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(output_dir, "logs", "queue_processor.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Инициализирован QueueProcessor для директории {input_dir}")
        
        # Создаем наблюдатель за файловой системой
        self.observer = Observer()
        self.event_handler = QueueEventHandler(self)
        self.observer.schedule(self.event_handler, input_dir, recursive=False)
        
        # Флаги состояния
        self.running = False
        self.processing = False
        self.stop_requested = False
        
        # Потоки
        self.process_thread = None
        
        # Очередь файлов
        self.file_queue = queue.Queue()
        
        # Если автоматический запуск, запускаем обработку
        if auto_start:
            self.start()
            
    def start(self):
        """Запускает обработку очереди."""
        if self.running:
            self.logger.warning("Обработка очереди уже запущена")
            return
            
        self.running = True
        self.stop_requested = False
        
        # Запускаем наблюдатель
        self.observer.start()
        
        # Запускаем поток обработки
        self.process_thread = threading.Thread(target=self._process_queue_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Проверяем существующие файлы в директории
        self._check_existing_files()
        
        self.logger.info("Обработка очереди запущена")
        
    def stop(self):
        """Останавливает обработку очереди."""
        if not self.running:
            self.logger.warning("Обработка очереди не запущена")
            return
            
        self.running = False
        self.stop_requested = True
        
        # Останавливаем наблюдатель
        self.observer.stop()
        self.observer.join()
        
        # Ожидаем завершение потока обработки
        if self.process_thread:
            self.process_thread.join(timeout=5.0)
            
        self.logger.info("Обработка очереди остановлена")
        
    def add_to_queue(self, file_path: str):
        """
        Добавляет файл в очередь для обработки.
        
        Args:
            file_path (str): Путь к файлу
        """
        # Проверяем, что файл существует и имеет подходящее расширение
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in self.extensions):
            self.file_queue.put(file_path)
            self.logger.debug(f"Файл добавлен в очередь: {file_path}")
        
    def _check_existing_files(self):
        """Проверяет существующие файлы в директории и добавляет их в очередь."""
        # Находим все файлы изображений в директории
        for ext in self.extensions:
            for file_path in Path(self.input_dir).glob