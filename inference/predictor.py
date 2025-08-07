"""
Predictor: Основной класс для предсказания цветов изображений.

Данный модуль предоставляет функциональность для колоризации черно-белых изображений,
используя предварительно обученные модели и интеллектуальные модули. Он включает в себя
возможности для обработки одиночных изображений и пакетов, применение различных стилей
колоризации, оценку неопределенности и стратегии восстановления при проблемах.

Ключевые особенности:
- Колоризация изображений с использованием предобученных моделей
- Интеграция с различными интеллектуальными модулями (GuideNet, Memory Bank и др.)
- Оценка неопределенности и стратегии восстановления
- Поддержка переноса стиля и пользовательских настроек
- Система мониторинга и логирования процесса колоризации

Преимущества:
- Высокое качество колоризации благодаря продвинутой архитектуре
- Гибкость настройки через конфигурационные файлы
- Устойчивость к проблемам благодаря стратегиям восстановления
- Богатые возможности для пользовательского контроля над процессом
"""

import os
import time
import json
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.config_parser import ConfigParser, load_config
from utils.data_loader import ColorSpaceConverter, prepare_batch_for_colorization
from utils.visualization import save_image, normalize_image, tensor_to_numpy
from utils.metrics import MetricsCalculator
from modules.guide_net import GuideNet
from modules.memory_bank import MemoryBankModule
from modules.style_transfer import StyleTransferModule
from modules.uncertainty_estimation import UncertaintyEstimationModule
from modules.few_shot_adapter import AdaptableColorizer


class FallbackStrategy(Enum):
    """Стратегии восстановления при проблемах с колоризацией."""
    NONE = "none"
    DEFAULT = "default"  # Базовая стратегия - просто возвращаем результат основной модели
    MEMORY_BANK = "memory_bank"  # Использование банка памяти для замены проблемных областей
    GUIDE_NET = "guide_net"  # Использование советника по цветам
    ENSEMBLE = "ensemble"  # Комбинирование результатов разных стратегий


class ColorizationPredictor:
    """
    Основной класс для предсказания цветов изображений.
    
    Args:
        model (nn.Module): Модель колоризации
        config (Dict): Конфигурация для предсказания
        modules_manager (Optional): Менеджер интеллектуальных модулей
        device (torch.device): Устройство для вычислений
    """
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        modules_manager: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.modules_manager = modules_manager
        
        # Устанавливаем устройство
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Переводим модель в режим оценки
        self.model.eval()
        
        # Извлекаем параметры из конфигурации
        self.color_space = config.get('color_space', 'lab')
        self.input_size = config.get('input_size', 256)
        self.output_dir = config.get('output_dir', './output')
        self.save_comparisons = config.get('save_comparisons', True)
        self.save_uncertainty_maps = config.get('save_uncertainty_maps', False)
        self.fallback_strategy_name = config.get('fallback_strategy', 'default')
        
        # Устанавливаем стратегию восстановления
        try:
            self.fallback_strategy = FallbackStrategy(self.fallback_strategy_name)
        except ValueError:
            self.fallback_strategy = FallbackStrategy.DEFAULT
            logging.warning(f"Неизвестная стратегия восстановления: {self.fallback_strategy_name}, используется DEFAULT")
            
        # Настраиваем стиль переноса, если включен
        self.style_transfer_config = config.get('style_transfer', {})
        self.style_transfer_enabled = self.style_transfer_config.get('enabled', False)
        
        # Инициализируем интеллектуальные модули, если их нет
        if self.modules_manager is None:
            self.guide_net = None
            self.memory_bank = None
            self.style_transfer = None
            self.uncertainty_estimation = None
        else:
            self.guide_net = self.modules_manager.get_module('guide_net')
            self.memory_bank = self.modules_manager.get_module('memory_bank')
            self.style_transfer = self.modules_manager.get_module('style_transfer')
            self.uncertainty_estimation = self.modules_manager.get_module('uncertainty_estimation')
            
        # Создаем каталоги для результатов
        os.makedirs(os.path.join(self.output_dir, "colorized"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "uncertainty_maps"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
        # Настраиваем логирование
        self.logger = logging.getLogger("ColorizationPredictor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Добавляем файловый обработчик
            os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.output_dir, "logs", "predictor.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Инициализирован ColorizationPredictor с устройством {self.device}")
        
    def colorize(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        reference_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        output_path: Optional[str] = None,
        save_comparison: Optional[bool] = None,
        save_uncertainty: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Колоризует изображение.
        
        Args:
            image (Union[np.ndarray, torch.Tensor, Image.Image]): Изображение для колоризации
            reference_image (Union[np.ndarray, torch.Tensor, Image.Image], optional): Референсное изображение для стиля
            style_name (str, optional): Имя стиля для применения
            style_alpha (float, optional): Интенсивность применения стиля (0.0-1.0)
            output_path (str, optional): Путь для сохранения результата
            save_comparison (bool, optional): Сохранять ли сравнение до/после
            save_uncertainty (bool, optional): Сохранять ли карту неопределенности
            metadata (Dict, optional): Дополнительные метаданные для сохранения
            
        Returns:
            Dict: Результаты колоризации
        """
        start_time = time.time()
        
        try:
            # Подготавливаем изображение
            grayscale_tensor, original_size, original_image = self._prepare_image(image)
            
            # Подготавливаем референсное изображение, если есть
            reference_tensor = None
            if reference_image is not None:
                reference_tensor, _, _ = self._prepare_image(reference_image, is_reference=True)
                
            # Добавляем размерность батча, если нужно
            if len(grayscale_tensor.shape) == 3:
                grayscale_tensor = grayscale_tensor.unsqueeze(0)
            if reference_tensor is not None and len(reference_tensor.shape) == 3:
                reference_tensor = reference_tensor.unsqueeze(0)
                
            # Переносим данные на устройство
            grayscale_tensor = grayscale_tensor.to(self.device)
            if reference_tensor is not None:
                reference_tensor = reference_tensor.to(self.device)
                
            # Выполняем колоризацию
            colorized_tensor, uncertainty_map = self._perform_colorization(
                grayscale_tensor, reference_tensor, style_name, style_alpha
            )
            
            # Преобразуем результаты в numpy
            grayscale_np = tensor_to_numpy(grayscale_tensor[0].cpu())
            colorized_np = tensor_to_numpy(colorized_tensor[0].cpu())
            
            # Изменяем размер результата обратно к исходному
            if original_size != colorized_np.shape[:2]:
                from skimage.transform import resize
                colorized_np = resize(
                    colorized_np,
                    (original_size[0], original_size[1], colorized_np.shape[2]),
                    anti_aliasing=True
                )
                
                # Нормализуем результат, если нужно
                if colorized_np.max() > 1.0:
                    colorized_np = colorized_np / 255.0
                    
            # Преобразуем карту неопределенности, если есть
            uncertainty_np = None
            if uncertainty_map is not None:
                uncertainty_np = tensor_to_numpy(uncertainty_map[0].cpu())
                if uncertainty_np.ndim == 3 and uncertainty_np.shape[2] == 1:
                    uncertainty_np = uncertainty_np[:, :, 0]
                    
                # Изменяем размер карты неопределенности
                if original_size != uncertainty_np.shape[:2]:
                    from skimage.transform import resize
                    uncertainty_np = resize(
                        uncertainty_np,
                        (original_size[0], original_size[1]),
                        anti_aliasing=True
                    )
                    
            # Генерируем имя файла, если не указано
            if output_path is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_path = os.path.join(self.output_dir, "colorized", f"colorized_{timestamp}.png")
                
            # Создаем директорию, если нужно
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохраняем результат
            save_image(colorized_np, output_path)
            
            # Сохраняем сравнение, если нужно
            comparison_path = None
            if save_comparison or (save_comparison is None and self.save_comparisons):
                comparison_path = os.path.join(
                    self.output_dir, "comparisons",
                    f"comparison_{os.path.basename(output_path)}"
                )
                
                # Визуализируем сравнение
                from utils.visualization import ColorizationVisualizer
                visualizer = ColorizationVisualizer(self.output_dir)
                
                # Создаем сравнение
                comparison = visualizer.create_comparison(
                    grayscale=grayscale_np if grayscale_np.ndim == 2 else grayscale_np[:, :, 0],
                    colorized=colorized_np,
                    uncertainty=uncertainty_np,
                    filename=os.path.basename(comparison_path)
                )
                
            # Сохраняем карту неопределенности, если нужно
            uncertainty_path = None
            if uncertainty_np is not None and (save_uncertainty or (save_uncertainty is None and self.save_uncertainty_maps)):
                uncertainty_path = os.path.join(
                    self.output_dir, "uncertainty_maps",
                    f"uncertainty_{os.path.basename(output_path)}"
                )
                
                # Визуализируем карту неопределенности
                from utils.visualization import ColorizationVisualizer
                visualizer = ColorizationVisualizer(self.output_dir)
                
                # Создаем визуализацию карты неопределенности
                uncertainty_vis = visualizer.create_uncertainty_map(
                    uncertainty=uncertainty_np,
                    colorized=colorized_np,
                    filename=os.path.basename(uncertainty_path)
                )
                
            # Подготавливаем результаты
            result = {
                "status": "success",
                "output_path": output_path,
                "comparison_path": comparison_path,
                "uncertainty_path": uncertainty_path,
                "processing_time": time.time() - start_time,
                "input_size": original_size,
                "model_used": type(self.model).__name__,
                "color_space": self.color_space
            }
            
            # Добавляем метаданные, если предоставлены
            if metadata:
                result["metadata"] = metadata
                
            # Сохраняем метаданные
            metadata_path = os.path.join(
                self.output_dir, "metadata",
                f"metadata_{os.path.basename(output_path).split('.')[0]}.json"
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Колоризация успешно выполнена, результат сохранен: {output_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при колоризации: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time,
                "traceback": traceback.format_exc()
            }
            
    def colorize_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        reference_images: Optional[List[Union[np.ndarray, torch.Tensor, Image.Image]]] = None,
        style_names: Optional[List[str]] = None,
        style_alphas: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        save_comparison: Optional[bool] = None,
        save_uncertainty: Optional[bool] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Колоризует пакет изображений.
        
        Args:
            images (List[Union[np.ndarray, torch.Tensor, Image.Image]]): Изображения для колоризации
            reference_images (List[Union[np.ndarray, torch.Tensor, Image.Image]], optional): Референсные изображения
            style_names (List[str], optional): Имена стилей для каждого изображения
            style_alphas (List[float], optional): Интенсивности стилей для каждого изображения
            output_dir (str, optional): Директория для сохранения результатов
            save_comparison (bool, optional): Сохранять ли сравнение до/после
            save_uncertainty (bool, optional): Сохранять ли карту неопределенности
            metadata (List[Dict], optional): Дополнительные метаданные для каждого изображения
            
        Returns:
            List[Dict]: Результаты колоризации для каждого изображения
        """
        results = []
        
        # Устанавливаем директорию для результатов
        if output_dir is not None:
            original_output_dir = self.output_dir
            self.output_dir = output_dir
            
            # Создаем директории для результатов
            os.makedirs(os.path.join(self.output_dir, "colorized"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "uncertainty_maps"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
        # Обрабатываем каждое изображение
        for i, image in enumerate(images):
            # Подготавливаем аргументы для колоризации
            kwargs = {}
            
            # Добавляем референсное изображение, если есть
            if reference_images is not None and i < len(reference_images):
                kwargs['reference_image'] = reference_images[i]
                
            # Добавляем имя стиля, если есть
            if style_names is not None and i < len(style_names):
                kwargs['style_name'] = style_names[i]
                
            # Добавляем интенсивность стиля, если есть
            if style_alphas is not None and i < len(style_alphas):
                kwargs['style_alpha'] = style_alphas[i]
                
            # Генерируем имя файла
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(self.output_dir, "colorized", f"colorized_{timestamp}_{i:04d}.png")
            kwargs['output_path'] = output_path
            
            # Добавляем флаги сохранения
            if save_comparison is not None:
                kwargs['save_comparison'] = save_comparison
            if save_uncertainty is not None:
                kwargs['save_uncertainty'] = save_uncertainty
                
            # Добавляем метаданные, если есть
            if metadata is not None and i < len(metadata):
                kwargs['metadata'] = metadata[i]
                
            # Выполняем колоризацию
            result = self.colorize(image, **kwargs)
            results.append(result)
            
        # Восстанавливаем исходную директорию для результатов
        if output_dir is not None:
            self.output_dir = original_output_dir
            
        return results
        
    def colorize_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = False,
        extensions: List[str] = None,
        reference_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None,
        save_comparison: Optional[bool] = None,
        save_uncertainty: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Колоризует все изображения в указанной директории.
        
        Args:
            input_dir (str): Директория с изображениями для колоризации
            output_dir (str, optional): Директория для сохранения результатов
            recursive (bool): Искать ли изображения рекурсивно во вложенных директориях
            extensions (List[str], optional): Список расширений файлов для обработки
            reference_image (Union[np.ndarray, torch.Tensor, Image.Image], optional): Референсное изображение
            style_name (str, optional): Имя стиля для применения
            style_alpha (float, optional): Интенсивность применения стиля
            save_comparison (bool, optional): Сохранять ли сравнение до/после
            save_uncertainty (bool, optional): Сохранять ли карту неопределенности
            metadata (Dict, optional): Дополнительные метаданные для сохранения
            
        Returns:
            Dict: Результаты колоризации для директории
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
            
        self.logger.info(f"Найдено {len(image_paths)} изображений для колоризации")
        
        # Устанавливаем директорию для результатов
        if output_dir is not None:
            original_output_dir = self.output_dir
            self.output_dir = output_dir
            
            # Создаем директории для результатов
            os.makedirs(os.path.join(self.output_dir, "colorized"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "uncertainty_maps"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
            
        # Загружаем изображения
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                images.append(image)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки изображения {path}: {str(e)}")
                
        # Обрабатываем изображения
        results = []
        for i, (image, path) in enumerate(zip(images, image_paths)):
            # Создаем имя файла для результата
            filename = os.path.basename(path)
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(self.output_dir, "colorized", f"{base_name}_colorized.png")
            
            # Добавляем информацию о файле в метаданные
            current_metadata = {"source_path": path}
            if metadata:
                current_metadata.update(metadata)
                
            # Выполняем колоризацию
            result = self.colorize(
                image=image,
                reference_image=reference_image,
                style_name=style_name,
                style_alpha=style_alpha,
                output_path=output_path,
                save_comparison=save_comparison,
                save_uncertainty=save_uncertainty,
                metadata=current_metadata
            )
            
            results.append(result)
            
            # Выводим прогресс
            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                self.logger.info(f"Обработано {i + 1}/{len(images)} изображений")
                
        # Восстанавливаем исходную директорию для результатов
        if output_dir is not None:
            self.output_dir = original_output_dir
            
        # Формируем общий результат
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful
        
        return {
            "status": "success",
            "processed": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
            "input_dir": input_dir,
            "output_dir": output_dir or self.output_dir
        }
            
    def _prepare_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        is_reference: bool = False
    ) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Подготавливает изображение для колоризации.
        
        Args:
            image (Union[np.ndarray, torch.Tensor, Image.Image]): Исходное изображение
            is_reference (bool): Является ли изображение референсным
            
        Returns:
            Tuple[torch.Tensor, Tuple[int, int], np.ndarray]: Подготовленное изображение, исходный размер, исходное изображение
        """
        # Преобразуем в numpy, если нужно
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Пакет изображений [B, C, H, W]
                image = image.squeeze(0)  # Берем первое изображение [C, H, W]
                
            # Преобразуем в numpy [H, W, C]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Если значения в диапазоне [0, 1], преобразуем в [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
                
        elif isinstance(image, Image.Image):
            # Преобразуем PIL.Image в numpy
            image_np = np.array(image)
        else:
            # Предполагаем, что это уже numpy
            image_np = image.copy()
            
        # Сохраняем исходный размер
        original_size = (image_np.shape[0], image_np.shape[1])
        
        # Если это ЧБ изображение и это не референс, преобразуем в RGB
        if not is_reference and (len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1)):
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=2)
            else:
                image_np = np.concatenate([image_np] * 3, axis=2)
                
        # Если изображение уже RGB и это не референс, преобразуем в оттенки серого
        if not is_reference and len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # Используем только канал L (яркость) в пространстве Lab
            if self.color_space == 'lab':
                # Конвертируем в Lab
                from skimage.color import rgb2lab
                lab_image = rgb2lab(image_np / 255.0)
                
                # Берем только канал L
                l_channel = lab_image[:, :, 0:1]
                
                # Нормализуем в диапазон [0, 1]
                l_channel = l_channel / 100.0
                
                # Заменяем исходное изображение
                image_np_gray = l_channel
            else:
                # Используем взвешенное преобразование в оттенки серого
                image_np_gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
                image_np_gray = np.expand_dims(image_np_gray, axis=2)
        else:
            # Если изображение уже в оттенках серого или это референс, просто нормализуем
            image_np_gray = image_np.astype(np.float32) / 255.0
            
        # Приводим к нужному размеру для модели
        if image_np_gray.shape[0] != self.input_size or image_np_gray.shape[1] != self.input_size:
            from skimage.transform import resize
            
            if not is_reference:
                # Для входного изображения изменяем размер сохраняя один канал
                image_np_resized = resize(
                    image_np_gray,
                    (self.input_size, self.input_size),
                    anti_aliasing=True
                )
            else:
                # Для референсного изображения сохраняем все каналы
                image_np_resized = resize(
                    image_np_gray,
                    (self.input_size, self.input_size, image_np_gray.shape[2]),
                    anti_aliasing=True
                )
        else:
            image_np_resized = image_np_gray
            
        # Преобразуем в тензор [C, H, W]
        if not is_reference and len(image_np_resized.shape) == 3 and image_np_resized.shape[2] == 1:
            # Для ЧБ изображения
            image_tensor = torch.tensor(image_np_resized[:, :, 0], dtype=torch.float32).unsqueeze(0)
        else:
            # Для цветного изображения или референса
            image_tensor = torch.tensor(image_np_resized, dtype=torch.float32).permute(2, 0, 1)
            
        return image_tensor, original_size, image_np
            
    def _perform_colorization(
        self,
        grayscale_tensor: torch.Tensor,
        reference_tensor: Optional[torch.Tensor] = None,
        style_name: Optional[str] = None,
        style_alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Выполняет колоризацию изображения.
        
        Args:
            grayscale_tensor (torch.Tensor): Тензор ЧБ изображения [B, 1, H, W]
            reference_tensor (torch.Tensor, optional): Тензор референсного изображения [B, 3, H, W]
            style_name (str, optional): Имя стиля для применения
            style_alpha (float, optional): Интенсивность применения стиля
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Колоризованное изображение и карта неопределенности
        """
        with torch.no_grad():
            try:
                # Базовое предсказание
                if isinstance(self.model, nn.Sequential):
                    # Для последовательной модели
                    output = grayscale_tensor
                    for module in self.model:
                        output = module(output)
                    
                    colorized_tensor = output
                    uncertainty_map = None
                else:
                    # Для сложных моделей с разными выходами
                    output = self.model(grayscale_tensor)
                    
                    # Извлекаем колоризованное изображение и карту неопределенности
                    if isinstance(output, dict):
                        colorized_tensor = output.get('colorized', output.get('output', None))
                        uncertainty_map = output.get('uncertainty', None)
                    elif isinstance(output, tuple) and len(output) >= 2:
                        colorized_tensor = output[0]
                        uncertainty_map = output[1] if len(output) > 1 else None
                    else:
                        colorized_tensor = output
                        uncertainty_map = None
                
                # Проверка наличия колоризованного изображения
                if colorized_tensor is None:
                    raise ValueError("Модель не вернула колоризованное изображение")
                    
                # Если выход - словарь с каналами a и b, объединяем их с входным L
                if isinstance(output, dict) and 'a' in output and 'b' in output:
                    a_channel = output['a']
                    b_channel = output['b']
                    colorized_tensor = torch.cat([grayscale_tensor, a_channel, b_channel], dim=1)
                
                # Применяем стиль, если нужно
                if self.style_transfer_enabled and self.style_transfer is not None and (reference_tensor is not None or style_name is not None):
                    # Определяем интенсивность стиля
                    alpha = style_alpha if style_alpha is not None else self.style_transfer_config.get('alpha', 0.5)
                    
                    # Применяем перенос стиля
                    styled_output = self.style_transfer.apply_style_transfer(
                        grayscale_tensor,
                        colorized_tensor,
                        reference_image=reference_tensor,
                        style_name=style_name,
                        alpha=alpha
                    )
                    
                    # Обновляем результат
                    if isinstance(styled_output, dict) and 'stylized' in styled_output:
                        colorized_tensor = styled_output['stylized']
                        
                # Применяем стратегию восстановления, если нужно
                if uncertainty_map is not None and torch.any(uncertainty_map > 0.5) and self.fallback_strategy != FallbackStrategy.NONE:
                    colorized_tensor, uncertainty_map = self._apply_fallback_strategy(
                        grayscale_tensor, colorized_tensor, uncertainty_map
                    )
                    
                # Если модель выводит только ab каналы, добавляем L-канал
                if colorized_tensor.shape[1] == 2:
                    colorized_tensor = torch.cat([grayscale_tensor, colorized_tensor], dim=1)
                    
                # Преобразуем в RGB, если нужно
                if self.color_space == 'lab':
                    # Проверяем, что выход имеет 3 канала (Lab)
                    if colorized_tensor.shape[1] != 3:
                        raise ValueError(f"Ожидается 3 канала для Lab, получено: {colorized_tensor.shape[1]}")
                        
                    # Преобразуем Lab в RGB
                    colorized_rgb = self._lab_to_rgb(colorized_tensor)
                    return colorized_rgb, uncertainty_map
                else:
                    # Для других цветовых пространств просто возвращаем результат
                    return colorized_tensor, uncertainty_map
                    
            except Exception as e:
                self.logger.error(f"Ошибка в процессе колоризации: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # В случае ошибки возвращаем исходное ЧБ изображение, преобразованное в RGB
                if grayscale_tensor.shape[1] == 1:
                    rgb_tensor = torch.cat([grayscale_tensor] * 3, dim=1)
                    return rgb_tensor, None
                else:
                    return grayscale_tensor, None
                    
    def _lab_to_rgb(self, lab_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор из пространства Lab в RGB.
        
        Args:
            lab_tensor (torch.Tensor): Тензор в пространстве Lab [B, 3, H, W]
            
        Returns:
            torch.Tensor: Тензор в пространстве RGB [B, 3, H, W]
        """
        # Разделяем каналы
        l_channel = lab_tensor[:, 0:1, :, :]  # [B, 1, H, W]
        a_channel = lab_tensor[:, 1:2, :, :]  # [B, 1, H, W]
        b_channel = lab_tensor[:, 2:3, :, :]  # [B, 1, H, W]
        
        # Денормализуем значения
        # L: [0, 1] -> [0, 100]
        # a, b: [-1, 1] -> [-128, 127]
        l_channel = l_channel * 100.0
        a_channel = a_channel * 127.0
        b_channel = b_channel * 127.0
        
        # Объединяем каналы
        lab = torch.cat([l_channel, a_channel, b_channel], dim=1)  # [B, 3, H, W]
        
        # Преобразуем в RGB
        rgb_list = []
        
        for i in range(lab.shape[0]):
            # Извлекаем изображение Lab
            lab_img = lab[i].permute(1, 2, 0).cpu().numpy()
            
            # Преобразуем Lab в RGB
            rgb_img = ColorSpaceConverter.lab_to_rgb(lab_img)
            
            # Нормализуем в диапазон [0, 1]
            rgb_img = rgb_img.astype(np.float32) / 255.0
            
            # Преобразуем в тензор
            rgb_tensor = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)
            rgb_list.append(rgb_tensor)
            
        # Объединяем в пакет
        rgb_tensor = torch.stack(rgb_list, dim=0).to(lab_tensor.device)
        
        return rgb_tensor
            
    def _apply_fallback_strategy(
        self,
        grayscale_tensor: torch.Tensor,
        colorized_tensor: torch.Tensor,
        uncertainty_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Применяет стратегию восстановления для областей с высокой неопределенностью.
        
        Args:
            grayscale_tensor (torch.Tensor): Тензор ЧБ изображения [B, 1, H, W]
            colorized_tensor (torch.Tensor): Колоризованное изображение [B, C, H, W]
            uncertainty_map (torch.Tensor): Карта неопределенности [B, 1, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Улучшенное изображение и обновленная карта неопределенности
        """
        # Если карта неопределенности не предоставлена, возвращаем исходное изображение
        if uncertainty_map is None:
            return colorized_tensor, None
            
        # Создаем маску для областей с высокой неопределенностью
        uncertainty_threshold = 0.5
        high_uncertainty_mask = (uncertainty_map > uncertainty_threshold).float()
        
        # Стратегия по умолчанию - просто возвращаем исходное изображение
        if self.fallback_strategy == FallbackStrategy.DEFAULT:
            return colorized_tensor, uncertainty_map
            
        # Стратегия с использованием банка памяти
        elif self.fallback_strategy == FallbackStrategy.MEMORY_BANK and self.memory_bank is not None:
            try:
                # Запрашиваем банк памяти
                memory_result = self.memory_bank(grayscale_tensor)
                
                if 'colorized' in memory_result:
                    memory_colorized = memory_result['colorized']
                    
                    # Проверяем совместимость формы
                    if memory_colorized.shape == colorized_tensor.shape:
                        # Применяем смешивание на основе неопределенности
                        blend_weight = uncertainty_map.clone()
                        blend_weight = torch.clamp((blend_weight - uncertainty_threshold) / (1.0 - uncertainty_threshold), 0.0, 1.0)
                        
                        # Смешиваем результаты
                        improved_colorized = colorized_tensor * (1.0 - blend_weight) + memory_colorized * blend_weight
                        
                        # Обновляем карту неопределенности
                        updated_uncertainty = uncertainty_map * (1.0 - 0.8 * (memory_result.get('confidence', 0.5)))
                        
                        return improved_colorized, updated_uncertainty
                        
            except Exception as e:
                self.logger.warning(f"Ошибка при применении стратегии Memory Bank: {str(e)}")
                
        # Стратегия с использованием советника по цветам
        elif self.fallback_strategy == FallbackStrategy.GUIDE_NET and self.guide_net is not None:
            try:
                # Получаем советы по цветам
                guide_result = self.guide_net(grayscale_tensor)
                
                if 'color_advice' in guide_result:
                    color_advice = guide_result['color_advice']
                    
                    # Проверяем совместимость формы
                    if color_advice.shape[1] == colorized_tensor.shape[1] - 1:  # Советник возвращает только ab каналы
                        # Создаем полное изображение
                        guide_colorized = torch.cat([grayscale_tensor, color_advice], dim=1)
                        
                        # Применяем смешивание на основе неопределенности и уверенности советника
                        confidence = guide_result.get('confidence', torch.ones_like(uncertainty_map) * 0.8)
                        blend_weight = uncertainty_map.clone() * confidence
                        blend_weight = torch.clamp(blend_weight, 0.0, 1.0)
                        
                        # Смешиваем результаты
                        improved_colorized = colorized_tensor * (1.0 - blend_weight) + guide_colorized * blend_weight
                        
                        # Обновляем карту неопределенности
                        updated_uncertainty = uncertainty_map * (1.0 - 0.7 * confidence)
                        
                        return improved_colorized, updated_uncertainty
                        
            except Exception as e:
                self.logger.warning(f"Ошибка при применении стратегии Guide Net: {str(e)}")
                
        # Стратегия с использованием ансамбля методов
        elif self.fallback_strategy == FallbackStrategy.ENSEMBLE:
            try:
                results = []
                weights = []
                
                # Добавляем базовый результат
                results.append(colorized_tensor)
                weights.append(torch.ones_like(uncertainty_map) * 0.5)
                
                # Добавляем результат от банка памяти, если доступен
                if self.memory_bank is not None:
                    memory_result = self.memory_bank(grayscale_tensor)
                    
                    if 'colorized' in memory_result:
                        memory_colorized = memory_result['colorized']
                        
                        # Проверяем совместимость формы
                        if memory_colorized.shape == colorized_tensor.shape:
                            results.append(memory_colorized)
                            
                            # Определяем вес на основе уверенности
                            confidence = memory_result.get('confidence', torch.ones_like(uncertainty_map) * 0.7)
                            weights.append(confidence)
                
                # Добавляем результат от советника, если доступен
                if self.guide_net is not None:
                    guide_result = self.guide_net(grayscale_tensor)
                    
                    if 'color_advice' in guide_result:
                        color_advice = guide_result['color_advice']
                        
                        # Проверяем совместимость формы
                        if color_advice.shape[1] == colorized_tensor.shape[1] - 1:  # Советник возвращает только ab каналы
                            # Создаем полное изображение
                            guide_colorized = torch.cat([grayscale_tensor, color_advice], dim=1)
                            results.append(guide_colorized)
                            
                            # Определяем вес на основе уверенности
                            confidence = guide_result.get('confidence', torch.ones_like(uncertainty_map) * 0.6)
                            weights.append(confidence)
                
                # Если есть дополнительные результаты, объединяем их
                if len(results) > 1:
                    # Преобразуем веса в нормализованную форму
                    weights_tensor = torch.stack(weights, dim=0)  # [N, B, 1, H, W]
                    weights_sum = torch.sum(weights_tensor, dim=0, keepdim=True)  # [1, B, 1, H, W]
                    weights_normalized = weights_tensor / (weights_sum + 1e-8)  # [N, B, 1, H, W]
                    
                    # Взвешенная сумма результатов
                    ensemble_result = torch.zeros_like(colorized_tensor)
                    
                    for i, (result, weight) in enumerate(zip(results, weights)):
                        # Расширяем вес до формы результата
                        expanded_weight = weight.expand_as(result)
                        ensemble_result += result * expanded_weight
                        
                    # Обновляем карту неопределенности
                    updated_uncertainty = uncertainty_map * torch.clamp(1.0 - torch.mean(torch.stack(weights[1:], dim=0), dim=0) * 0.6, 0.3, 1.0)
                    
                    return ensemble_result, updated_uncertainty
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при применении стратегии Ensemble: {str(e)}")
                
        # Если не удалось применить стратегию, возвращаем исходное изображение
        return colorized_tensor, uncertainty_map
        
    def evaluate_metrics(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Вычисляет метрики качества колоризации.
        
        Args:
            pred_images (torch.Tensor): Предсказанные изображения [B, 3, H, W]
            target_images (torch.Tensor): Целевые изображения [B, 3, H, W]
            metrics (List[str], optional): Список метрик для вычисления
            
        Returns:
            Dict[str, float]: Значения метрик
        """
        # Устанавливаем метрики по умолчанию
        if metrics is None:
            metrics = ['psnr', 'ssim', 'lpips']
            
        try:
            # Создаем калькулятор метрик
            calculator = MetricsCalculator(metrics=metrics, device=self.device)
            
            # Вычисляем метрики
            metrics_values = calculator.calculate(pred_images, target_images)
            
            return metrics_values
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении метрик: {str(e)}")
            return {}


def load_predictor(
    model_path: str,
    config_path: Optional[str] = None,
    modules_path: Optional[str] = None,
    device: Optional[str] = None
) -> ColorizationPredictor:
    """
    Загружает предиктор колоризации из сохраненных файлов.
    
    Args:
        model_path (str): Путь к сохраненной модели
        config_path (str, optional): Путь к конфигурации
        modules_path (str, optional): Путь к сохраненным интеллектуальным модулям
        device (str, optional): Устройство для вычислений
        
    Returns:
        ColorizationPredictor: Загруженный предиктор
    """
    # Определяем устройство
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Загружаем конфигурацию
    if config_path is None:
        # Пытаемся найти конфигурацию рядом с моделью
        config_dir = os.path.dirname(model_path)
        potential_config_paths = [
            os.path.join(config_dir, "inference_config.yaml"),
            os.path.join(config_dir, "../configs/inference_config.yaml")
        ]
        
        for path in potential_config_paths:
            if os.path.exists(path):
                config_path = path
                break
                
    if config_path is not None:
        try:
            config = load_config(config_path, schema_type='inference')
        except Exception as e:
            logging.warning(f"Не удалось загрузить конфигурацию: {str(e)}. Используется конфигурация по умолчанию.")
            from utils.config_parser import create_default_config
            config = create_default_config('inference')
    else:
        from utils.config_parser import create_default_config
        config = create_default_config('inference')
        
    # Загружаем модель
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Загружаем модель из чекпоинта
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            from core.swin_unet import SwinUNet
            # Создаем модель и загружаем веса
            model_config = checkpoint.get('config', None) or config.get('model', {})
            model = SwinUNet(**model_config)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Предполагаем, что сам чекпоинт и есть модель
            model = checkpoint
            
        # Переводим модель в режим оценки
        model.eval()
        
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")
        
    # Загружаем интеллектуальные модули
    modules_manager = None
    if modules_path is not None:
        try:
            # Импортируем менеджер модулей
            from modules import ColorizationModulesManager
            
            # Загружаем конфигурацию модулей
            modules_config = torch.load(modules_path, map_location=device)
            
            # Создаем менеджер модулей
            modules_manager = ColorizationModulesManager(modules_config, model, device)
            
        except Exception as e:
            logging.warning(f"Не удалось загрузить интеллектуальные модули: {str(e)}")
    
    # Создаем предиктор
    predictor = ColorizationPredictor(model, config, modules_manager, device)
    
    return predictor


def get_available_fallback_strategies() -> List[str]:
    """
    Возвращает список доступных стратегий восстановления.
    
    Returns:
        List[str]: Список стратегий
    """
    return [strategy.value for strategy in FallbackStrategy]


if __name__ == "__main__":
    # Пример использования предиктора
    
    import argparse
    
    parser = argparse.ArgumentParser(description="TintoraAI Predictor")
    parser.add_argument("--model", type=str, required=True, help="Путь к модели")
    parser.add_argument("--config", type=str, help="Путь к конфигурации")
    parser.add_argument("--modules", type=str, help="Путь к интеллектуальным модулям")
    parser.add_argument("--input", type=str, required=True, help="Путь к изображению или директории")
    parser.add_argument("--output", type=str, help="Путь для сохранения результата")
    parser.add_argument("--recursive", action="store_true", help="Рекурсивный поиск в директориях")
    parser.add_argument("--style", type=str, help="Имя стиля для применения")
    parser.add_argument("--reference", type=str, help="Путь к референсному изображению")
    parser.add_argument("--alpha", type=float, default=0.5, help="Интенсивность стиля (0.0-1.0)")
    parser.add_argument("--device", type=str, help="Устройство (cuda, cpu)")
    parser.add_argument("--fallback", type=str, choices=get_available_fallback_strategies(),
                      help="Стратегия восстановления")
    
    args = parser.parse_args()
    
    try:
        # Загружаем предиктор
        predictor = load_predictor(
            model_path=args.model,
            config_path=args.config,
            modules_path=args.modules,
            device=args.device
        )
        
        # Устанавливаем стратегию восстановления, если указана
        if args.fallback:
            predictor.fallback_strategy = FallbackStrategy(args.fallback)
            
        # Загружаем референсное изображение, если указано
        reference_image = None
        if args.reference:
            reference_image = Image.open(args.reference).convert('RGB')
            
        # Проверяем, является ли вход файлом или директорией
        if os.path.isfile(args.input):
            # Колоризуем одно изображение
            result = predictor.colorize(
                image=Image.open(args.input).convert('RGB'),
                reference_image=reference_image,
                style_name=args.style,
                style_alpha=args.alpha,
                output_path=args.output
            )
            
            print(f"Результат колоризации: {result['status']}")
            if result['status'] == 'success':
                print(f"Сохранено: {result['output_path']}")
                if 'comparison_path' in result and result['comparison_path']:
                    print(f"Сравнение: {result['comparison_path']}")
                if 'uncertainty_path' in result and result['uncertainty_path']:
                    print(f"Карта неопределенности: {result['uncertainty_path']}")
                    
        elif os.path.isdir(args.input):
            # Колоризуем директорию
            results = predictor.colorize_directory(
                input_dir=args.input,
                output_dir=args.output,
                recursive=args.recursive,
                reference_image=reference_image,
                style_name=args.style,
                style_alpha=args.alpha
            )
            
            print(f"Обработано изображений: {results['processed']}")
            print(f"Успешно: {results['successful']}, Ошибок: {results['failed']}")
            print(f"Результаты сохранены: {results['output_dir']}")
            
        else:
            print(f"Указанный путь не существует: {args.input}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        traceback.print_exc()