"""
Postprocessor: Модуль для постобработки результатов колоризации.

Данный модуль предоставляет функции и классы для постобработки колоризованных изображений,
включая коррекцию цветов, улучшение насыщенности, применение фильтров и другие операции,
которые улучшают визуальное качество и естественность колоризации.

Ключевые особенности:
- Коррекция цветового баланса для более реалистичной колоризации
- Усиление или ослабление насыщенности отдельных цветовых диапазонов
- Улучшение контраста и детализации в колоризованных изображениях
- Удаление артефактов и шумов, возникающих в процессе колоризации
- Возможность применения художественных фильтров и стилей

Преимущества:
- Значительное повышение визуального качества результатов колоризации
- Возможность настройки параметров постобработки для конкретных задач
- Исправление типичных проблем алгоритмов колоризации
- Получение более реалистичных и эстетически приятных результатов
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
import cv2
from PIL import Image, ImageEnhance, ImageFilter

from utils.config_parser import load_config
from utils.visualization import normalize_image, save_image


class ColorBalanceMode(Enum):
    """Режимы коррекции цветового баланса."""
    NEUTRAL = "neutral"  # Нейтральный баланс
    WARM = "warm"  # Теплые тона
    COOL = "cool"  # Холодные тона
    VINTAGE = "vintage"  # Винтажный стиль
    FILM = "film"  # Кинематографический стиль
    NATURAL = "natural"  # Естественные цвета


class SaturationMode(Enum):
    """Режимы коррекции насыщенности."""
    NATURAL = "natural"  # Естественная насыщенность
    VIBRANT = "vibrant"  # Яркие, насыщенные цвета
    MUTED = "muted"  # Приглушенные цвета
    SELECTIVE = "selective"  # Выборочная насыщенность


class ColorizationPostprocessor:
    """
    Постпроцессор для улучшения результатов колоризации.
    
    Args:
        config (Dict): Конфигурация постпроцессора
        device (torch.device, optional): Устройство для вычислений
    """
    def __init__(
        self,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        self.config = config
        
        # Устанавливаем устройство
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Извлекаем параметры из конфигурации
        self.color_space = config.get('color_space', 'rgb')
        self.output_dir = config.get('output_dir', './output')
        self.save_comparison = config.get('save_comparison', False)
        
        # Параметры постобработки
        self.color_balance = config.get('color_balance', {})
        self.color_balance_mode = self.color_balance.get('mode', 'neutral')
        self.color_balance_strength = self.color_balance.get('strength', 0.5)
        
        self.saturation = config.get('saturation', {})
        self.saturation_mode = self.saturation.get('mode', 'natural')
        self.saturation_strength = self.saturation.get('strength', 0.5)
        
        self.sharpness = config.get('sharpness', {})
        self.sharpness_enabled = self.sharpness.get('enabled', False)
        self.sharpness_amount = self.sharpness.get('amount', 0.3)
        
        self.noise_reduction = config.get('noise_reduction', {})
        self.noise_reduction_enabled = self.noise_reduction.get('enabled', False)
        self.noise_reduction_strength = self.noise_reduction.get('strength', 0.5)
        
        self.artifact_removal = config.get('artifact_removal', {})
        self.artifact_removal_enabled = self.artifact_removal.get('enabled', False)
        self.artifact_removal_threshold = self.artifact_removal.get('threshold', 0.3)
        
        self.contrast_enhancement = config.get('contrast_enhancement', {})
        self.contrast_enabled = self.contrast_enhancement.get('enabled', False)
        self.contrast_amount = self.contrast_enhancement.get('amount', 0.2)
        self.contrast_preserve_brightness = self.contrast_enhancement.get('preserve_brightness', True)
        
        self.output_format = config.get('output_format', 'png')
        self.output_quality = config.get('output_quality', 95)
        
        # Создаем директорию для постобработанных изображений
        os.makedirs(os.path.join(self.output_dir, "postprocessed"), exist_ok=True)
        
        # Настраиваем логирование
        self.logger = logging.getLogger("ColorizationPostprocessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Добавляем файловый обработчик
            os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.output_dir, "logs", "postprocessor.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Инициализирован ColorizationPostprocessor")
    
    def process_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        grayscale_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
        output_path: Optional[str] = None,
        save_comparison: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Выполняет постобработку колоризованного изображения.
        
        Args:
            image (Union[np.ndarray, torch.Tensor, Image.Image]): Колоризованное изображение
            grayscale_image (Union[np.ndarray, torch.Tensor, Image.Image], optional): Оригинальное ЧБ изображение
            output_path (str, optional): Путь для сохранения результата
            save_comparison (bool, optional): Сохранять ли сравнение до/после постобработки
            metadata (Dict, optional): Дополнительные метаданные
            
        Returns:
            Dict: Результаты постобработки
        """
        start_time = time.time()
        
        try:
            # Преобразуем входное изображение в PIL.Image
            image_pil = self._to_pil_image(image)
            
            # Преобразуем ЧБ изображение в PIL.Image, если предоставлено
            grayscale_pil = self._to_pil_image(grayscale_image) if grayscale_image is not None else None
            
            # Сохраняем исходное колоризованное изображение для сравнения
            original_pil = image_pil.copy()
            
            # Применяем постобработку
            # 1. Коррекция цветового баланса
            image_pil = self._apply_color_balance(image_pil)
            
            # 2. Настройка насыщенности
            image_pil = self._apply_saturation(image_pil)
            
            # 3. Улучшение резкости, если включено
            if self.sharpness_enabled:
                image_pil = self._apply_sharpness(image_pil)
                
            # 4. Снижение шума, если включено
            if self.noise_reduction_enabled:
                image_pil = self._apply_noise_reduction(image_pil)
                
            # 5. Удаление артефактов, если включено
            if self.artifact_removal_enabled:
                image_pil = self._apply_artifact_removal(image_pil)
                
            # 6. Улучшение контраста, если включено
            if self.contrast_enabled:
                image_pil = self._apply_contrast_enhancement(image_pil)
            
            # Генерируем имя файла для результата, если не указано
            if output_path is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_path = os.path.join(self.output_dir, "postprocessed", f"postprocessed_{timestamp}.{self.output_format}")
            
            # Создаем директорию, если нужно
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохраняем постобработанное изображение
            image_pil.save(output_path, quality=self.output_quality)
            
            # Сохраняем сравнение, если требуется
            comparison_path = None
            if save_comparison or (save_comparison is None and self.save_comparison):
                comparison_path = os.path.join(
                    self.output_dir, "comparisons",
                    f"postprocess_comparison_{os.path.basename(output_path)}"
                )
                
                # Создаем директорию для сравнений, если нужно
                os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
                
                # Создаем изображение для сравнения
                comparison_image = self._create_comparison_image(
                    original_pil, image_pil, grayscale_pil
                )
                
                comparison_image.save(comparison_path, quality=self.output_quality)
            
            # Формируем результат
            processing_time = time.time() - start_time
            result = {
                "status": "success",
                "output_path": output_path,
                "comparison_path": comparison_path,
                "processing_time": processing_time,
                "applied_operations": {
                    "color_balance": self.color_balance_mode,
                    "saturation": self.saturation_mode,
                    "sharpness": self.sharpness_enabled,
                    "noise_reduction": self.noise_reduction_enabled,
                    "artifact_removal": self.artifact_removal_enabled,
                    "contrast_enhancement": self.contrast_enabled
                }
            }
            
            # Добавляем метаданные, если предоставлены
            if metadata:
                result["metadata"] = metadata
                
            # Сохраняем метаданные
            metadata_path = os.path.join(
                self.output_dir, "metadata",
                f"postprocess_metadata_{os.path.basename(output_path).split('.')[0]}.json"
            )
            
            # Создаем директорию для метаданных, если нужно
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Постобработка успешно выполнена, результат сохранен: {output_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при постобработке: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time,
                "traceback": traceback.format_exc()
            }
    
    def process_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        grayscale_images: Optional[List[Union[np.ndarray, torch.Tensor, Image.Image]]] = None,
        output_dir: Optional[str] = None,
        save_comparison: Optional[bool] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Выполняет постобработку пакета колоризованных изображений.
        
        Args:
            images (List[Union[np.ndarray, torch.Tensor, Image.Image]]): Список колоризованных изображений
            grayscale_images (List[Union[np.ndarray, torch.Tensor, Image.Image]], optional): Список ЧБ изображений
            output_dir (str, optional): Директория для сохранения результатов
            save_comparison (bool, optional): Сохранять ли сравнения до/после
            metadata (List[Dict], optional): Метаданные для каждого изображения
            
        Returns:
            List[Dict]: Результаты постобработки для каждого изображения
        """
        results = []
        
        # Устанавливаем директорию для результатов
        if output_dir is not None:
            original_output_dir = self.output_dir
            self.output_dir = output_dir
            
            # Создаем директории для результатов
            os.makedirs(os.path.join(self.output_dir, "postprocessed"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
        # Обрабатываем каждое изображение
        for i, image in enumerate(images):
            # Подготавливаем аргументы для постобработки
            kwargs = {}
            
            # Добавляем ЧБ изображение, если есть
            if grayscale_images is not None and i < len(grayscale_images):
                kwargs['grayscale_image'] = grayscale_images[i]
                
            # Генерируем имя файла
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(self.output_dir, "postprocessed", f"postprocessed_{timestamp}_{i:04d}.{self.output_format}")
            kwargs['output_path'] = output_path
            
            # Добавляем флаг сохранения сравнения
            if save_comparison is not None:
                kwargs['save_comparison'] = save_comparison
                
            # Добавляем метаданные, если есть
            if metadata is not None and i < len(metadata):
                kwargs['metadata'] = metadata[i]
                
            # Выполняем постобработку
            result = self.process_image(image, **kwargs)
            results.append(result)
            
            # Выводим прогресс
            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                self.logger.info(f"Обработано {i + 1}/{len(images)} изображений")
        
        # Восстанавливаем исходную директорию для результатов
        if output_dir is not None:
            self.output_dir = original_output_dir
            
        return results
    
    def _to_pil_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
        """
        Преобразует изображение различных типов в PIL.Image.
        
        Args:
            image (Union[np.ndarray, torch.Tensor, Image.Image]): Исходное изображение
            
        Returns:
            PIL.Image.Image: Изображение в формате PIL
        """
        if image is None:
            return None
            
        if isinstance(image, Image.Image):
            return image
            
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Пакет изображений [B, C, H, W]
                image = image.squeeze(0)  # Берем первое изображение [C, H, W]
                
            # Преобразуем в numpy [H, W, C]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Если значения в диапазоне [0, 1], преобразуем в [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
                
            return Image.fromarray(image_np)
            
        else:  # numpy array
            # Если изображение в оттенках серого
            if len(image.shape) == 2:
                return Image.fromarray(image, 'L')
                
            # Если одноканальное изображение с добавленной осью
            if len(image.shape) == 3 and image.shape[2] == 1:
                return Image.fromarray(image[:, :, 0], 'L')
                
            # Если значения в диапазоне [0, 1], преобразуем в [0, 255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            return Image.fromarray(image)
    
    def _apply_color_balance(self, image: Image.Image) -> Image.Image:
        """
        Применяет коррекцию цветового баланса.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Определяем режим коррекции
        try:
            mode = ColorBalanceMode(self.color_balance_mode)
        except ValueError:
            mode = ColorBalanceMode.NEUTRAL
            
        # Если режим нейтральный, возвращаем без изменений
        if mode == ColorBalanceMode.NEUTRAL:
            return image
            
        # Преобразуем в массив numpy для более гибкой обработки
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Применяем коррекцию в зависимости от режима
        if mode == ColorBalanceMode.WARM:
            # Усиливаем красные и желтые тона
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1.0 + 0.2 * self.color_balance_strength), 0.0, 1.0)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1.0 - 0.1 * self.color_balance_strength), 0.0, 1.0)  # Blue
            
        elif mode == ColorBalanceMode.COOL:
            # Усиливаем синие и голубые тона
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1.0 - 0.1 * self.color_balance_strength), 0.0, 1.0)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1.0 + 0.2 * self.color_balance_strength), 0.0, 1.0)  # Blue
            
        elif mode == ColorBalanceMode.VINTAGE:
            # Создаем винтажный вид (приглушенные цвета с усилением красных тонов)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1.0 + 0.15 * self.color_balance_strength), 0.0, 1.0)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1.0 - 0.1 * self.color_balance_strength), 0.0, 1.0)  # Green
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1.0 - 0.15 * self.color_balance_strength), 0.0, 1.0)  # Blue
            
            # Добавляем легкую сепию
            sepia_amount = 0.2 * self.color_balance_strength
            sepia_r = img_array[:, :, 0] * 0.393 + img_array[:, :, 1] * 0.769 + img_array[:, :, 2] * 0.189
            sepia_g = img_array[:, :, 0] * 0.349 + img_array[:, :, 1] * 0.686 + img_array[:, :, 2] * 0.168
            sepia_b = img_array[:, :, 0] * 0.272 + img_array[:, :, 1] * 0.534 + img_array[:, :, 2] * 0.131
            
            img_array[:, :, 0] = np.clip(sepia_r * sepia_amount + img_array[:, :, 0] * (1 - sepia_amount), 0.0, 1.0)
            img_array[:, :, 1] = np.clip(sepia_g * sepia_amount + img_array[:, :, 1] * (1 - sepia_amount), 0.0, 1.0)
            img_array[:, :, 2] = np.clip(sepia_b * sepia_amount + img_array[:, :, 2] * (1 - sepia_amount), 0.0, 1.0)
            
        elif mode == ColorBalanceMode.FILM:
            # Кинематографический вид (контрастные тени, теплые света)
            # Преобразуем в HSV для более точного контроля над тонами
            hsv_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Усиливаем контраст
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * (1.0 + 0.1 * self.color_balance_strength), 0, 255)
            
            # Смещаем оттенки в тенях в сторону холодных тонов, а в светах - в сторону теплых
            mask_shadows = (hsv_img[:, :, 2] < 128).astype(np.float32)
            mask_highlights = (hsv_img[:, :, 2] >= 128).astype(np.float32)
            
            # Смещаем оттенки в тенях к синему (180 в HSV)
            blue_hue_shift = 10 * self.color_balance_strength
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + blue_hue_shift * mask_shadows) % 180
            
            # Смещаем оттенки в светах к оранжевому (20 в HSV)
            orange_hue_shift = -10 * self.color_balance_strength
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + orange_hue_shift * mask_highlights) % 180
            
            # Конвертируем обратно в RGB
            img_array = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
        elif mode == ColorBalanceMode.NATURAL:
            # Естественные цвета (легкое усиление зеленых тонов)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1.0 + 0.1 * self.color_balance_strength), 0.0, 1.0)  # Green
            
            # Небольшая коррекция гаммы для более естественного вида
            gamma = 1.0 - 0.1 * self.color_balance_strength
            img_array = np.power(img_array, gamma)
            
        # Преобразуем обратно в PIL.Image
        processed_image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        return processed_image
    
    def _apply_saturation(self, image: Image.Image) -> Image.Image:
        """
        Применяет настройку насыщенности.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Определяем режим насыщенности
        try:
            mode = SaturationMode(self.saturation_mode)
        except ValueError:
            mode = SaturationMode.NATURAL
            
        # Если режим естественный и сила 0.5, возвращаем без изменений
        if mode == SaturationMode.NATURAL and abs(self.saturation_strength - 0.5) < 0.01:
            return image
            
        # Определяем коэффициент насыщенности
        saturation_factor = 1.0
        
        if mode == SaturationMode.NATURAL:
            # Естественная насыщенность (линейное масштабирование)
            saturation_factor = 0.5 + self.saturation_strength
            
        elif mode == SaturationMode.VIBRANT:
            # Яркие, насыщенные цвета
            saturation_factor = 1.0 + self.saturation_strength
            
        elif mode == SaturationMode.MUTED:
            # Приглушенные цвета
            saturation_factor = 0.5 * self.saturation_strength
            
        elif mode == SaturationMode.SELECTIVE:
            # Выборочная насыщенность (разные цвета масштабируются по-разному)
            # Для этого режима используем более сложную обработку
            
            # Преобразуем в массив numpy для более гибкой обработки
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Преобразуем в HSV
            hsv_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Создаем маски для разных диапазонов оттенков (HSV)
            # Красные тона (0-20 и 160-180)
            red_mask = np.logical_or(hsv_img[:, :, 0] < 20, hsv_img[:, :, 0] > 160).astype(np.float32)
            
            # Желтые и оранжевые тона (20-60)
            yellow_mask = np.logical_and(hsv_img[:, :, 0] >= 20, hsv_img[:, :, 0] < 60).astype(np.float32)
            
            # Зеленые тона (60-100)
            green_mask = np.logical_and(hsv_img[:, :, 0] >= 60, hsv_img[:, :, 0] < 100).astype(np.float32)
            
            # Голубые и синие тона (100-140)
            blue_mask = np.logical_and(hsv_img[:, :, 0] >= 100, hsv_img[:, :, 0] < 140).astype(np.float32)
            
            # Фиолетовые тона (140-160)
            purple_mask = np.logical_and(hsv_img[:, :, 0] >= 140, hsv_img[:, :, 0] < 160).astype(np.float32)
            
            # Применяем разные коэффициенты насыщенности к разным цветам
            # Красные и оранжевые тона усиливаем больше
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * (1.0 + 0.5 * self.saturation_strength) * red_mask + 
                                      hsv_img[:, :, 1] * (1.0 + 0.4 * self.saturation_strength) * yellow_mask + 
                                      hsv_img[:, :, 1] * (1.0 + 0.3 * self.saturation_strength) * green_mask + 
                                      hsv_img[:, :, 1] * (1.0 + 0.2 * self.saturation_strength) * blue_mask + 
                                      hsv_img[:, :, 1] * (1.0 + 0.3 * self.saturation_strength) * purple_mask, 0, 255)
            
            # Конвертируем обратно в RGB
            img_array = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
            # Преобразуем обратно в PIL.Image
            return Image.fromarray((img_array * 255).astype(np.uint8))
            
        # Для остальных режимов используем стандартное усиление насыщенности
        return ImageEnhance.Color(image).enhance(saturation_factor)
    
    def _apply_sharpness(self, image: Image.Image) -> Image.Image:
        """
        Применяет улучшение резкости.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Определяем коэффициент резкости
        sharpness_factor = 1.0 + self.sharpness_amount * 2.0
        
        # Применяем улучшение резкости
        enhanced_image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)
        
        return enhanced_image
    
    def _apply_noise_reduction(self, image: Image.Image) -> Image.Image:
        """
        Применяет снижение шума.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Преобразуем в numpy массив
        img_array = np.array(image)
        
        # Определяем силу шумоподавления
        strength = int(self.noise_reduction_strength * 10) + 1
        
        # Применяем билатеральный фильтр для сохранения краев при шумоподавлении
        denoised_array = cv2.bilateralFilter(
            img_array, d=strength, sigmaColor=strength * 3, sigmaSpace=strength
        )
        
        # Для более сильного шумоподавления можно использовать нелокальное усреднение (Non-local Means)
        if self.noise_reduction_strength > 0.7:
            denoised_array = cv2.fastNlMeansDenoisingColored(
                denoised_array, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
        
        # Преобразуем обратно в PIL.Image
        denoised_image = Image.fromarray(denoised_array)
        
        return denoised_image
    
    def _apply_artifact_removal(self, image: Image.Image) -> Image.Image:
        """
        Применяет удаление артефактов.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Преобразуем в numpy массив
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Преобразуем в YCrCb для лучшего выделения артефактов
        ycrcb_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        
        # Выделяем яркостной и хроматические каналы
        y_channel = ycrcb_img[:, :, 0]
        cr_channel = ycrcb_img[:, :, 1]
        cb_channel = ycrcb_img[:, :, 2]
        
        # Вычисляем градиенты для обнаружения резких переходов (потенциальные артефакты)
        grad_x = cv2.Sobel(cr_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cr_channel, cv2.CV_32F, 0, 1, ksize=3)
        cr_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_x = cv2.Sobel(cb_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cb_channel, cv2.CV_32F, 0, 1, ksize=3)
        cb_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Создаем маску артефактов (области с высоким градиентом в хроматических каналах)
        artifact_mask = (cr_grad_mag > self.artifact_removal_threshold) | (cb_grad_mag > self.artifact_removal_threshold)
        
        # Расширяем маску для захвата всей области артефакта
        kernel = np.ones((5, 5), np.uint8)
        artifact_mask = cv2.dilate(artifact_mask.astype(np.uint8), kernel, iterations=1)
        
        # Применяем медианный фильтр к областям с артефактами
        filtered_cr = cr_channel.copy()
        filtered_cb = cb_channel.copy()
        
        # Для областей с артефактами применяем медианный фильтр
        cr_filtered = cv2.medianBlur((cr_channel * 255).astype(np.uint8), 5) / 255.0
        cb_filtered = cv2.medianBlur((cb_channel * 255).astype(np.uint8), 5) / 255.0
        
        # Применяем фильтр только к областям с артефактами
        filtered_cr[artifact_mask == 1] = cr_filtered[artifact_mask == 1]
        filtered_cb[artifact_mask == 1] = cb_filtered[artifact_mask == 1]
        
        # Собираем изображение обратно
        ycrcb_filtered = np.stack([y_channel, filtered_cr, filtered_cb], axis=2)
        
        # Преобразуем обратно в RGB
        filtered_img = cv2.cvtColor((ycrcb_filtered * 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        
        # Преобразуем обратно в PIL.Image
        filtered_image = Image.fromarray(filtered_img)
        
        return filtered_image
    
    def _apply_contrast_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Применяет улучшение контраста с сохранением яркости.
        
        Args:
            image (Image.Image): Входное изображение
            
        Returns:
            Image.Image: Обработанное изображение
        """
        # Преобразуем в numpy массив
        img_array = np.array(image).astype(np.float32) / 255.0
        
        if self.contrast_preserve_brightness:
            # Улучшение контраста с сохранением средней яркости
            
            # Преобразуем в HSV для работы с яркостью отдельно
            hsv_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Запоминаем среднюю яркость
            mean_v = np.mean(hsv_img[:, :, 2])
            
            # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization) для улучшения контраста
            clahe = cv2.createCLAHE(clipLimit=2.0 + self.contrast_amount * 2.0, tileGridSize=(8, 8))
            hsv_img[:, :, 2] = clahe.apply((hsv_img[:, :, 2]).astype(np.uint8)).astype(np.float32)
            
            # Восстанавливаем среднюю яркость
            new_mean_v = np.mean(hsv_img[:, :, 2])
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * (mean_v / new_mean_v), 0, 255)
            
            # Конвертируем обратно в RGB
            enhanced_array = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        else:
            # Простое улучшение контраста
            contrast_factor = 1.0 + self.contrast_amount
            enhanced_array = np.clip((img_array - 0.5) * contrast_factor + 0.5, 0, 1) * 255
            enhanced_array = enhanced_array.astype(np.uint8)
            
        # Преобразуем обратно в PIL.Image
        enhanced_image = Image.fromarray(enhanced_array)
        
        return enhanced_image
    
    def _create_comparison_image(self, 
                              original: Image.Image, 
                              processed: Image.Image, 
                              grayscale: Optional[Image.Image] = None) -> Image.Image:
        """
        Создает изображение для сравнения результатов до и после постобработки.
        
        Args:
            original (Image.Image): Исходное колоризованное изображение
            processed (Image.Image): Изображение после постобработки
            grayscale (Image.Image, optional): Оригинальное ЧБ изображение
            
        Returns:
            Image.Image: Сравнительное изображение
        """
        # Определяем количество изображений для сравнения
        n_images = 2
        if grayscale is not None:
            n_images = 3
            
        # Приводим все изображения к одинаковому размеру
        width, height = original.size
        
        if processed.size != (width, height):
            processed = processed.resize((width, height), Image.LANCZOS)
            
        if grayscale is not None and grayscale.size != (width, height):
            grayscale = grayscale.resize((width, height), Image.LANCZOS)
            
        # Создаем новое изображение для сравнения
        comparison_image = Image.new('RGB', (width * n_images, height))
        
        # Добавляем ЧБ изображение, если предоставлено
        if grayscale is not None:
            # Если изображение в режиме 'L', преобразуем в RGB
            if grayscale.mode == 'L':
                grayscale = grayscale.convert('RGB')
            comparison_image.paste(grayscale, (0, 0))
            comparison_image.paste(original, (width, 0))
            comparison_image.paste(processed, (width * 2, 0))
        else:
            comparison_image.paste(original, (0, 0))
            comparison_image.paste(processed, (width, 0))
            
        # Добавляем подписи
        draw = ImageDraw.Draw(comparison_image)
        
        # Определяем размер и положение текста
        font_size = max(10, min(20, height // 30))
        
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = None
            
        # Добавляем подписи
        if grayscale is not None:
            draw.text((10, 10), "Оригинал ЧБ", fill="white", font=font)
            draw.text((width + 10, 10), "До постобработки", fill="white", font=font)
            draw.text((width * 2 + 10, 10), "После постобработки", fill="white", font=font)
        else:
            draw.text((10, 10), "До постобработки", fill="white", font=font)
            draw.text((width + 10, 10), "После постобработки", fill="white", font=font)
            
        return comparison_image


def create_postprocessor(config_path: Optional[str] = None, device: Optional[str] = None) -> ColorizationPostprocessor:
    """
    Создает постпроцессор на основе конфигурации.
    
    Args:
        config_path (str, optional): Путь к конфигурационному файлу
        device (str, optional): Устройство для вычислений
        
    Returns:
        ColorizationPostprocessor: Созданный постпроцессор
    """
    # Определяем устройство
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Загружаем конфигурацию
    if config_path is not None:
        try:
            config = load_config(config_path)
        except Exception as e:
            logging.warning(f"Не удалось загрузить конфигурацию: {str(e)}. Используется конфигурация по умолчанию.")
            config = {}
    else:
        config = {}
        
    # Создаем постпроцессор
    return ColorizationPostprocessor(config, device)


def process_directory(
    postprocessor: ColorizationPostprocessor,
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    extensions: List[str] = None,
    grayscale_dir: Optional[str] = None,
    save_comparison: bool = True
) -> Dict:
    """
    Обрабатывает все изображения в указанной директории.
    
    Args:
        postprocessor (ColorizationPostprocessor): Постпроцессор
        input_dir (str): Директория с изображениями для обработки
        output_dir (str, optional): Директория для сохранения результатов
        recursive (bool): Искать ли изображения рекурсивно
        extensions (List[str], optional): Список расширений файлов
        grayscale_dir (str, optional): Директория с оригинальными ЧБ изображениями
        save_comparison (bool): Сохранять ли сравнения
        
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
        logging.warning(f"В директории {input_dir} не найдено изображений с расширениями {extensions}")
        return {"status": "warning", "message": "Не найдено изображений", "processed": 0}
        
    logging.info(f"Найдено {len(image_paths)} изображений для постобработки")
    
    # Находим соответствующие ЧБ изображения, если указана директория
    grayscale_paths = None
    if grayscale_dir is not None:
        grayscale_paths = []
        for path in image_paths:
            # Извлекаем имя файла
            filename = os.path.basename(path)
            
            # Ищем соответствующее ЧБ изображение
            grayscale_path = os.path.join(grayscale_dir, filename)
            
            # Если не найдено, пробуем с другими расширениями
            if not os.path.exists(grayscale_path):
                base_name = os.path.splitext(filename)[0]
                found = False
                for ext in extensions:
                    test_path = os.path.join(grayscale_dir, f"{base_name}{ext}")
                    if os.path.exists(test_path):
                        grayscale_path = test_path
                        found = True
                        break
                        
                if not found:
                    grayscale_path = None
                    
            grayscale_paths.append(grayscale_path)
    
    # Загружаем изображения
    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            images.append(image)
        except Exception as e:
            logging.error(f"Ошибка загрузки изображения {path}: {str(e)}")
            
    # Загружаем ЧБ изображения, если есть
    grayscale_images = None
    if grayscale_paths is not None:
        grayscale_images = []
        for path in grayscale_paths:
            if path is None:
                grayscale_images.append(None)
            else:
                try:
                    image = Image.open(path).convert('RGB')
                    grayscale_images.append(image)
                except Exception as e:
                    logging.error(f"Ошибка загрузки ЧБ изображения {path}: {str(e)}")
                    grayscale_images.append(None)
                    
    # Устанавливаем директорию для результатов
    if output_dir is not None:
        original_output_dir = postprocessor.output_dir
        postprocessor.output_dir = output_dir
        
        # Создаем директории для результатов
        os.makedirs(os.path.join(output_dir, "postprocessed"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
                    
    # Обрабатываем изображения
    results = []
    for i, (image, path) in enumerate(zip(images, image_paths)):
        # Создаем имя файла для результата
        filename = os.path.basename(path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(
            postprocessor.output_dir, "postprocessed", 
            f"{base_name}_processed.{postprocessor.output_format}"
        )
        
        # Получаем ЧБ изображение, если есть
        grayscale_image = None
        if grayscale_images is not None and i < len(grayscale_images):
            grayscale_image = grayscale_images[i]
            
        # Выполняем постобработку
        result = postprocessor.process_image(
            image=image,
            grayscale_image=grayscale_image,
            output_path=output_path,
            save_comparison=save_comparison,
            metadata={"source_path": path}
        )
        
        results.append(result)
        
        # Выводим прогресс
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            logging.info(f"Обработано {i + 1}/{len(images)} изображений")
            
    # Восстанавливаем исходную директорию для результатов
    if output_dir is not None:
        postprocessor.output_dir = original_output_dir
        
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
        "output_dir": output_dir or postprocessor.output_dir
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TintoraAI Postprocessor")
    parser.add_argument("--config", type=str, help="Путь к конфигурации")
    parser.add_argument("--input", type=str, required=True, help="Путь к изображению или директории")
    parser.add_argument("--output", type=str, help="Путь для сохранения результата")
    parser.add_argument("--recursive", action="store_true", help="Рекурсивный поиск в директориях")
    parser.add_argument("--grayscale", type=str, help="Путь к оригинальному ЧБ изображению или директории")
    parser.add_argument("--color_balance", type=str, choices=[mode.value for mode in ColorBalanceMode], 
                      default="neutral", help="Режим цветового баланса")
    parser.add_argument("--saturation", type=str, choices=[mode.value for mode in SaturationMode], 
                      default="natural", help="Режим настройки насыщенности")
    parser.add_argument("--strength", type=float, default=0.5, help="Сила эффектов (0.0-1.0)")
    parser.add_argument("--sharpness", action="store_true", help="Включить улучшение резкости")
    parser.add_argument("--noise_reduction", action="store_true", help="Включить шумоподавление")
    parser.add_argument("--artifact_removal", action="store_true", help="Включить удаление артефактов")
    parser.add_argument("--contrast", action="store_true", help="Включить улучшение контраста")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"], 
                      help="Формат выходного файла")
    parser.add_argument("--quality", type=int, default=95, help="Качество изображения (для JPEG)")
    parser.add_argument("--device", type=str, help="Устройство (cuda, cpu)")
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        config = {}
        if args.config:
            config = load_config(args.config)
            
        # Обновляем конфигурацию параметрами командной строки
        config['color_balance'] = {
            'mode': args.color_balance,
            'strength': args.strength
        }
        
        config['saturation'] = {
            'mode': args.saturation,
            'strength': args.strength
        }
        
        config['sharpness'] = {
            'enabled': args.sharpness,
            'amount': 0.3
        }
        
        config['noise_reduction'] = {
            'enabled': args.noise_reduction,
            'strength': 0.5
        }
        
        config['artifact_removal'] = {
            'enabled': args.artifact_removal,
            'threshold': 0.3
        }
        
        config['contrast_enhancement'] = {
            'enabled': args.contrast,
            'amount': 0.2,
            'preserve_brightness': True
        }
        
        config['output_format'] = args.format
        config['output_quality'] = args.quality
        
        # Создаем постпроцессор
        postprocessor = ColorizationPostprocessor(config, device=args.device)
        
        # Проверяем, является ли вход файлом или директорией
        if os.path.isfile(args.input):
            # Обрабатываем одно изображение
            image = Image.open(args.input).convert('RGB')
            
            # Загружаем ЧБ изображение, если указано
            grayscale = None
            if args.grayscale and os.path.isfile(args.grayscale):
                grayscale = Image.open(args.grayscale).convert('RGB')
                
            # Выполняем постобработку
            result = postprocessor.process_image(
                image=image,
                grayscale_image=grayscale,
                output_path=args.output,
                save_comparison=True
            )
            
            print(f"Результат постобработки: {result['status']}")
            if result['status'] == 'success':
                print(f"Сохранено: {result['output_path']}")
                if 'comparison_path' in result and result['comparison_path']:
                    print(f"Сравнение: {result['comparison_path']}")
                    
        elif os.path.isdir(args.input):
            # Обрабатываем директорию
            results = process_directory(
                postprocessor=postprocessor,
                input_dir=args.input,
                output_dir=args.output,
                recursive=args.recursive,
                grayscale_dir=args.grayscale if args.grayscale and os.path.isdir(args.grayscale) else None,
                save_comparison=True
            )
            
            print(f"Обработано изображений: {results['processed']}")
            print(f"Успешно: {results['successful']}, Ошибок: {results['failed']}")
            print(f"Результаты сохранены: {results['output_dir']}")
            
        else:
            print(f"Указанный путь не существует: {args.input}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        traceback.print_exc()