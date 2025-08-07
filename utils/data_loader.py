"""
Data Loader: Модуль для загрузки и предобработки данных.

Данный модуль обеспечивает загрузку, предобработку и аугментацию изображений для
задачи колоризации. Включает в себя функции для работы с различными форматами изображений,
конвертации между цветовыми пространствами, нормализации данных и создания батчей для
обучения и инференса.

Ключевые особенности:
- Поддержка различных форматов изображений (JPEG, PNG, WEBP, TIFF и т.д.)
- Конвертация между различными цветовыми пространствами (RGB, Lab, YUV)
- Эффективная обработка данных с использованием многопоточности и кеширования
- Аугментация данных для улучшения обучения и предотвращения переобучения
- Создание пар изображений (черно-белое -> цветное) для обучения колоризации

Преимущества для колоризации:
- Специализированные преобразования для задачи колоризации
- Оптимизированная загрузка и предобработка изображений
- Гибкая настройка параметров обработки через конфигурацию
- Расширенная аугментация для улучшения генерализации модели
"""

import os
import sys
import glob
import random
import json
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile, ImageFilter, ImageOps
import cv2

# Настройка PIL для загрузки поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ColorSpaceConverter:
    """
    Конвертер между различными цветовыми пространствами.
    """
    @staticmethod
    def rgb_to_lab(rgb_img: np.ndarray) -> np.ndarray:
        """
        Конвертирует изображение из RGB в Lab.
        
        Args:
            rgb_img (np.ndarray): Изображение в формате RGB [H, W, 3]
            
        Returns:
            np.ndarray: Изображение в формате Lab [H, W, 3]
        """
        # OpenCV ожидает BGR формат
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        
        # Нормализуем значения
        # L: [0, 100] -> [0, 1]
        # a: [-127, 127] -> [-1, 1]
        # b: [-127, 127] -> [-1, 1]
        lab_img = lab_img.astype(np.float32)
        lab_img[:, :, 0] = lab_img[:, :, 0] / 100.0
        lab_img[:, :, 1:] = (lab_img[:, :, 1:] - 128) / 127.0
        
        return lab_img
        
    @staticmethod
    def lab_to_rgb(lab_img: np.ndarray) -> np.ndarray:
        """
        Конвертирует изображение из Lab в RGB.
        
        Args:
            lab_img (np.ndarray): Изображение в формате Lab [H, W, 3] или [3, H, W]
            
        Returns:
            np.ndarray: Изображение в формате RGB [H, W, 3]
        """
        # Если изображение в формате [3, H, W], преобразуем его в [H, W, 3]
        if lab_img.shape[0] == 3 and len(lab_img.shape) == 3:
            lab_img = np.transpose(lab_img, (1, 2, 0))
            
        # Копируем, чтобы не изменять входное изображение
        lab_img = lab_img.copy().astype(np.float32)
        
        # Денормализуем значения
        lab_img[:, :, 0] = lab_img[:, :, 0] * 100.0
        lab_img[:, :, 1:] = lab_img[:, :, 1:] * 127.0 + 128
        
        # Преобразуем значения к uint8
        lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
        
        # Конвертируем Lab в BGR
        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        
        # Конвертируем BGR в RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        return rgb_img
        
    @staticmethod
    def rgb_to_grayscale(rgb_img: np.ndarray) -> np.ndarray:
        """
        Конвертирует изображение из RGB в оттенки серого.
        
        Args:
            rgb_img (np.ndarray): Изображение в формате RGB [H, W, 3]
            
        Returns:
            np.ndarray: Изображение в оттенках серого [H, W, 1]
        """
        # Используем веса из стандарта ITU-R BT.601
        gray_img = np.dot(rgb_img[..., :3], [0.299, 0.587, 0.114])
        
        # Добавляем ось канала
        gray_img = gray_img[:, :, np.newaxis]
        
        return gray_img.astype(np.float32) / 255.0
        
    @staticmethod
    def lab_to_grayscale(lab_img: np.ndarray) -> np.ndarray:
        """
        Извлекает канал L из изображения Lab.
        
        Args:
            lab_img (np.ndarray): Изображение в формате Lab [H, W, 3] или [3, H, W]
            
        Returns:
            np.ndarray: Канал L [H, W, 1] или [1, H, W]
        """
        # Если изображение в формате [3, H, W], извлекаем только первый канал
        if lab_img.shape[0] == 3 and len(lab_img.shape) == 3:
            return lab_img[0:1]
            
        # Иначе извлекаем канал L и добавляем ось канала
        return lab_img[:, :, 0:1]


class ImageAugmentation:
    """
    Класс для аугментации изображений.
    
    Args:
        p_horizontal_flip (float): Вероятность горизонтального отражения
        p_vertical_flip (float): Вероятность вертикального отражения
        p_random_crop (float): Вероятность случайной обрезки
        p_color_jitter (float): Вероятность изменения цветовых параметров
        p_random_affine (float): Вероятность аффинных преобразований
        p_gaussian_blur (float): Вероятность размытия по Гауссу
        seed (int, optional): Зерно для генератора случайных чисел
    """
    def __init__(self, 
                 p_horizontal_flip: float = 0.5,
                 p_vertical_flip: float = 0.0,
                 p_random_crop: float = 0.5,
                 p_color_jitter: float = 0.3,
                 p_random_affine: float = 0.3,
                 p_gaussian_blur: float = 0.1,
                 seed: Optional[int] = None):
        self.p_horizontal_flip = p_horizontal_flip
        self.p_vertical_flip = p_vertical_flip
        self.p_random_crop = p_random_crop
        self.p_color_jitter = p_color_jitter
        self.p_random_affine = p_random_affine
        self.p_gaussian_blur = p_gaussian_blur
        
        # Инициализируем генератор случайных чисел
        self.rng = random.Random(seed)
        
        # Создаем преобразования
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Применяет аугментацию к изображению.
        
        Args:
            image (PIL.Image.Image): Входное изображение
            
        Returns:
            PIL.Image.Image: Аугментированное изображение
        """
        # Горизонтальное отражение
        if self.rng.random() < self.p_horizontal_flip:
            image = TF.hflip(image)
            
        # Вертикальное отражение
        if self.rng.random() < self.p_vertical_flip:
            image = TF.vflip(image)
            
        # Случайная обрезка с последующим изменением размера
        if self.rng.random() < self.p_random_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w, image.size)
            
        # Изменение цветовых параметров
        if self.rng.random() < self.p_color_jitter:
            image = self.color_jitter(image)
            
        # Аффинные преобразования
        if self.rng.random() < self.p_random_affine:
            angle = self.rng.uniform(-10, 10)
            translate = (self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.1, 0.1))
            scale = self.rng.uniform(0.9, 1.1)
            shear = self.rng.uniform(-5, 5)
            image = TF.affine(image, angle, translate, scale, shear)
            
        # Размытие по Гауссу
        if self.rng.random() < self.p_gaussian_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.1, 1.0)))
            
        return image
        
    def get_paired_transform(self):
        """
        Возвращает функцию для одинаковой аугментации пары изображений.
        
        Returns:
            Callable: Функция для аугментации пары изображений
        """
        # Определяем, какие преобразования будут применены к текущей паре
        apply_hflip = self.rng.random() < self.p_horizontal_flip
        apply_vflip = self.rng.random() < self.p_vertical_flip
        apply_crop = self.rng.random() < self.p_random_crop
        apply_affine = self.rng.random() < self.p_random_affine
        
        # Предварительно определяем параметры для случайных преобразований
        crop_params = None
        if apply_crop:
            # Будем использовать эти параметры для обоих изображений
            i, j, h, w = None, None, None, None  # Будут определены при первом использовании
            
        affine_params = None
        if apply_affine:
            angle = self.rng.uniform(-10, 10)
            translate = (self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.1, 0.1))
            scale = self.rng.uniform(0.9, 1.1)
            shear = self.rng.uniform(-5, 5)
            affine_params = (angle, translate, scale, shear)
            
        # Возвращаем функцию, которая будет применять одинаковую аугментацию к паре изображений
        def paired_transform(image1, image2):
            nonlocal crop_params
            
            # Горизонтальное отражение
            if apply_hflip:
                image1 = TF.hflip(image1)
                image2 = TF.hflip(image2)
                
            # Вертикальное отражение
            if apply_vflip:
                image1 = TF.vflip(image1)
                image2 = TF.vflip(image2)
                
            # Случайная обрезка с последующим изменением размера
            if apply_crop:
                if crop_params is None:
                    # Определяем параметры обрезки на основе первого изображения
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        image1, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                    )
                    crop_params = (i, j, h, w)
                else:
                    i, j, h, w = crop_params
                    
                image1 = TF.resized_crop(image1, i, j, h, w, image1.size)
                image2 = TF.resized_crop(image2, i, j, h, w, image2.size)
                
            # Аффинные преобразования
            if apply_affine and affine_params is not None:
                angle, translate, scale, shear = affine_params
                image1 = TF.affine(image1, angle, translate, scale, shear)
                image2 = TF.affine(image2, angle, translate, scale, shear)
                
            return image1, image2
            
        return paired_transform


class ColorizationDataset(Dataset):
    """
    Датасет для задачи колоризации изображений.
    
    Args:
        image_paths (List[str]): Список путей к изображениям
        color_space (str): Целевое цветовое пространство ('lab', 'rgb', 'yuv')
        image_size (int): Размер изображения
        split_ab (bool): Разделять ли каналы a и b в Lab пространстве
        augment (bool): Применять ли аугментацию
        cache_size (int): Размер кеша для изображений (0 - без кеша)
        transform (Callable, optional): Дополнительные преобразования
    """
    def __init__(self, 
                 image_paths: List[str], 
                 color_space: str = 'lab',
                 image_size: int = 256,
                 split_ab: bool = True,
                 augment: bool = True,
                 cache_size: int = 0,
                 transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.color_space = color_space.lower()
        self.image_size = image_size
        self.split_ab = split_ab
        self.augment = augment
        self.transform = transform
        self.cache_size = cache_size
        
        # Проверяем, что цветовое пространство поддерживается
        supported_color_spaces = ['lab', 'rgb', 'yuv']
        if self.color_space not in supported_color_spaces:
            raise ValueError(f"Неподдерживаемое цветовое пространство: {self.color_space}. "
                            f"Поддерживаемые пространства: {supported_color_spaces}")
        
        # Создаем кеш, если нужно
        self.image_cache = {}
        if cache_size > 0:
            self.image_cache = {}
        
        # Создаем аугментацию, если нужно
        self.augmentation = None
        if augment:
            self.augmentation = ImageAugmentation()
            
        # Создаем базовые преобразования
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self) -> int:
        """
        Возвращает количество изображений в датасете.
        
        Returns:
            int: Количество изображений
        """
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает элемент датасета по индексу.
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с изображением и метками
        """
        image_path = self.image_paths[idx]
        
        # Проверяем кеш
        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            # Загружаем изображение
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Ошибка загрузки изображения {image_path}: {e}")
                # Возвращаем пустое изображение в случае ошибки
                image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
            
            # Добавляем в кеш, если нужно
            if self.cache_size > 0 and len(self.image_cache) < self.cache_size:
                self.image_cache[image_path] = image
        
        # Применяем аугментацию, если нужно
        if self.augmentation is not None:
            image = self.augmentation(image)
            
        # Применяем базовые преобразования
        tensor_image = self.basic_transform(image)
        
        # Применяем дополнительные преобразования, если есть
        if self.transform is not None:
            tensor_image = self.transform(tensor_image)
            
        # Конвертируем в выбранное цветовое пространство
        if self.color_space == 'lab':
            # Конвертируем RGB -> Lab
            image_np = tensor_image.permute(1, 2, 0).numpy()
            lab_image = ColorSpaceConverter.rgb_to_lab(image_np)
            
            # Разделяем на L и ab каналы
            l_channel = torch.tensor(lab_image[:, :, 0], dtype=torch.float32).unsqueeze(0)
            ab_channels = torch.tensor(lab_image[:, :, 1:], dtype=torch.float32).permute(2, 0, 1)
            
            if self.split_ab:
                a_channel = ab_channels[0].unsqueeze(0)
                b_channel = ab_channels[1].unsqueeze(0)
                
                result = {
                    'l': l_channel,
                    'a': a_channel,
                    'b': b_channel,
                    'rgb': tensor_image,
                    'path': image_path
                }
            else:
                result = {
                    'l': l_channel,
                    'ab': ab_channels,
                    'rgb': tensor_image,
                    'path': image_path
                }
        else:
            # Конвертируем RGB -> Grayscale
            grayscale_image = ColorSpaceConverter.rgb_to_grayscale(
                tensor_image.permute(1, 2, 0).numpy()
            )
            grayscale_tensor = torch.tensor(grayscale_image, dtype=torch.float32).permute(2, 0, 1)
            
            result = {
                'grayscale': grayscale_tensor,
                'color': tensor_image,
                'path': image_path
            }
            
        return result


class ColorizationDatasetPaired(Dataset):
    """
    Датасет для задачи колоризации с парными изображениями (ЧБ и цветные).
    
    Args:
        grayscale_paths (List[str]): Список путей к ЧБ изображениям
        color_paths (List[str]): Список путей к цветным изображениям
        color_space (str): Целевое цветовое пространство ('lab', 'rgb', 'yuv')
        image_size (int): Размер изображения
        split_ab (bool): Разделять ли каналы a и b в Lab пространстве
        augment (bool): Применять ли аугментацию
        paired_transform (bool): Применять ли одинаковую аугментацию к парам
        cache_size (int): Размер кеша для изображений (0 - без кеша)
    """
    def __init__(self, 
                 grayscale_paths: List[str], 
                 color_paths: List[str],
                 color_space: str = 'lab',
                 image_size: int = 256,
                 split_ab: bool = True,
                 augment: bool = True,
                 paired_transform: bool = True,
                 cache_size: int = 0):
        # Проверяем, что количество ЧБ и цветных изображений совпадает
        if len(grayscale_paths) != len(color_paths):
            raise ValueError("Количество ЧБ и цветных изображений должно совпадать")
            
        self.grayscale_paths = grayscale_paths
        self.color_paths = color_paths
        self.color_space = color_space.lower()
        self.image_size = image_size
        self.split_ab = split_ab
        self.augment = augment
        self.paired_transform = paired_transform
        self.cache_size = cache_size
        
        # Проверяем, что цветовое пространство поддерживается
        supported_color_spaces = ['lab', 'rgb', 'yuv']
        if self.color_space not in supported_color_spaces:
            raise ValueError(f"Неподдерживаемое цветовое пространство: {self.color_space}. "
                            f"Поддерживаемые пространства: {supported_color_spaces}")
        
        # Создаем кеш, если нужно
        self.image_cache = {}
        if cache_size > 0:
            self.image_cache = {}
        
        # Создаем аугментацию, если нужно
        self.augmentation = None
        if augment:
            self.augmentation = ImageAugmentation()
            
        # Создаем базовые преобразования
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self) -> int:
        """
        Возвращает количество пар изображений в датасете.
        
        Returns:
            int: Количество пар изображений
        """
        return len(self.grayscale_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает пару изображений по индексу.
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с ЧБ и цветным изображениями
        """
        grayscale_path = self.grayscale_paths[idx]
        color_path = self.color_paths[idx]
        
        # Загружаем ЧБ изображение
        grayscale_image = self._load_image(grayscale_path)
        
        # Загружаем цветное изображение
        color_image = self._load_image(color_path)
        
        # Если нужно применить одинаковую аугментацию к паре изображений
        if self.augment and self.paired_transform:
            paired_transform = self.augmentation.get_paired_transform()
            grayscale_image, color_image = paired_transform(grayscale_image, color_image)
        else:
            # Иначе применяем аугментацию отдельно к каждому изображению
            if self.augment:
                grayscale_image = self.augmentation(grayscale_image)
                color_image = self.augmentation(color_image)
                
        # Применяем базовые преобразования
        grayscale_tensor = self.basic_transform(grayscale_image)
        color_tensor = self.basic_transform(color_image)
        
        # Конвертируем в выбранное цветовое пространство
        if self.color_space == 'lab':
            # Конвертируем RGB -> Lab для цветного изображения
            color_np = color_tensor.permute(1, 2, 0).numpy()
            lab_image = ColorSpaceConverter.rgb_to_lab(color_np)
            
            # Разделяем на L и ab каналы
            l_channel = torch.tensor(lab_image[:, :, 0], dtype=torch.float32).unsqueeze(0)
            ab_channels = torch.tensor(lab_image[:, :, 1:], dtype=torch.float32).permute(2, 0, 1)
            
            if self.split_ab:
                a_channel = ab_channels[0].unsqueeze(0)
                b_channel = ab_channels[1].unsqueeze(0)
                
                result = {
                    'grayscale': grayscale_tensor,
                    'l': l_channel,
                    'a': a_channel,
                    'b': b_channel,
                    'color': color_tensor,
                    'grayscale_path': grayscale_path,
                    'color_path': color_path
                }
            else:
                result = {
                    'grayscale': grayscale_tensor,
                    'l': l_channel,
                    'ab': ab_channels,
                    'color': color_tensor,
                    'grayscale_path': grayscale_path,
                    'color_path': color_path
                }
        else:
            result = {
                'grayscale': grayscale_tensor,
                'color': color_tensor,
                'grayscale_path': grayscale_path,
                'color_path': color_path
            }
            
        return result
        
    def _load_image(self, path: str) -> Image.Image:
        """
        Загружает изображение и кеширует его, если нужно.
        
        Args:
            path (str): Путь к изображению
            
        Returns:
            PIL.Image.Image: Загруженное изображение
        """
        # Проверяем кеш
        if path in self.image_cache:
            return self.image_cache[path]
            
        # Загружаем изображение
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения {path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        # Добавляем в кеш, если нужно
        if self.cache_size > 0 and len(self.image_cache) < self.cache_size:
            self.image_cache[path] = image
            
        return image


class DatasetFromFolder(Dataset):
    """
    Датасет для загрузки изображений из директории.
    
    Args:
        root_dir (str): Корневая директория с изображениями
        extensions (List[str], optional): Поддерживаемые расширения файлов
        recursive (bool): Рекурсивный поиск изображений в поддиректориях
    """
    def __init__(self, root_dir: str, extensions: List[str] = None, recursive: bool = True):
        self.root_dir = root_dir
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
        self.recursive = recursive
        
        # Находим все изображения
        self.image_paths = self._find_images()
        
    def _find_images(self) -> List[str]:
        """
        Находит все изображения в директории.
        
        Returns:
            List[str]: Список путей к изображениям
        """
        image_paths = []
        
        if self.recursive:
            # Рекурсивный поиск
            for ext in self.extensions:
                pattern = os.path.join(self.root_dir, f'**/*{ext}')
                image_paths.extend(glob.glob(pattern, recursive=True))
        else:
            # Поиск только в корневой директории
            for ext in self.extensions:
                pattern = os.path.join(self.root_dir, f'*{ext}')
                image_paths.extend(glob.glob(pattern))
                
        return sorted(image_paths)
        
    def __len__(self) -> int:
        """
        Возвращает количество изображений в датасете.
        
        Returns:
            int: Количество изображений
        """
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> str:
        """
        Возвращает путь к изображению по индексу.
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            str: Путь к изображению
        """
        return self.image_paths[idx]


class PairedDatasetFromFolders(Dataset):
    """
    Датасет для загрузки парных изображений из директорий.
    
    Args:
        grayscale_dir (str): Директория с ЧБ изображениями
        color_dir (str): Директория с цветными изображениями
        extensions (List[str], optional): Поддерживаемые расширения файлов
        recursive (bool): Рекурсивный поиск изображений в поддиректориях
    """
    def __init__(self, grayscale_dir: str, color_dir: str, extensions: List[str] = None, recursive: bool = True):
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
        self.recursive = recursive
        
        # Находим все изображения
        self.grayscale_paths, self.color_paths = self._find_paired_images()
        
    def _find_paired_images(self) -> Tuple[List[str], List[str]]:
        """
        Находит все парные изображения в директориях.
        
        Returns:
            Tuple[List[str], List[str]]: Пара списков путей к ЧБ и цветным изображениям
        """
        grayscale_paths = []
        color_paths = []
        
        # Находим все ЧБ изображения
        grayscale_dataset = DatasetFromFolder(
            root_dir=self.grayscale_dir, 
            extensions=self.extensions, 
            recursive=self.recursive
        )
        
        # Для каждого ЧБ изображения пытаемся найти соответствующее цветное
        for grayscale_path in grayscale_dataset:
            # Получаем относительный путь от корня ЧБ директории
            rel_path = os.path.relpath(grayscale_path, self.grayscale_dir)
            
            # Формируем путь к цветному изображению
            color_path = os.path.join(self.color_dir, rel_path)
            
            # Проверяем существование цветного изображения
            if os.path.isfile(color_path):
                grayscale_paths.append(grayscale_path)
                color_paths.append(color_path)
                
        return grayscale_paths, color_paths
        
    def __len__(self) -> int:
        """
        Возвращает количество пар изображений в датасете.
        
        Returns:
            int: Количество пар изображений
        """
        return len(self.grayscale_paths)
        
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Возвращает пару путей к изображениям по индексу.
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            Tuple[str, str]: Пара путей к ЧБ и цветному изображениям
        """
        return self.grayscale_paths[idx], self.color_paths[idx]


class ColorizationDataModule:
    """
    Модуль данных для задачи колоризации.
    
    Args:
        train_dir (str): Директория с обучающими изображениями
        val_dir (str, optional): Директория с валидационными изображениями
        test_dir (str, optional): Директория с тестовыми изображениями
        batch_size (int): Размер батча
        num_workers (int): Количество рабочих процессов для загрузки данных
        color_space (str): Целевое цветовое пространство ('lab', 'rgb', 'yuv')
        image_size (int): Размер изображения
        split_ab (bool): Разделять ли каналы a и b в Lab пространстве
        augment_train (bool): Применять ли аугментацию к обучающим данным
        val_split (float): Доля данных для валидации, если val_dir не указана
        cache_size (int): Размер кеша для изображений (0 - без кеша)
    """
    def __init__(self,
                 train_dir: str,
                 val_dir: Optional[str] = None,
                 test_dir: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 color_space: str = 'lab',
                 image_size: int = 256,
                 split_ab: bool = True,
                 augment_train: bool = True,
                 val_split: float = 0.1,
                 cache_size: int = 0):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.color_space = color_space
        self.image_size = image_size
        self.split_ab = split_ab
        self.augment_train = augment_train
        self.val_split = val_split
        self.cache_size = cache_size
        
        # Датасеты
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Создаем датасеты
        self._setup_datasets()
        
    def _setup_datasets(self):
        """
        Создает датасеты для обучения, валидации и тестирования.
        """
        # Находим все изображения в обучающей директории
        train_images = DatasetFromFolder(self.train_dir).image_paths
        
        # Создаем обучающий датасет
        full_train_dataset = ColorizationDataset(
            image_paths=train_images,
            color_space=self.color_space,
            image_size=self.image_size,
            split_ab=self.split_ab,
            augment=self.augment_train,
            cache_size=self.cache_size
        )
        
        # Если указана директория валидации, используем ее
        if self.val_dir:
            val_images = DatasetFromFolder(self.val_dir).image_paths
            self.val_dataset = ColorizationDataset(
                image_paths=val_images,
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                cache_size=self.cache_size
            )
            self.train_dataset = full_train_dataset
        else:
            # Иначе разделяем обучающий датасет на обучающий и валидационный
            val_size = int(len(full_train_dataset) * self.val_split)
            train_size = len(full_train_dataset) - val_size
            
            # Создаем датасеты с соответствующими индексами
            train_indices, val_indices = random_split(
                range(len(full_train_dataset)),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.train_dataset = Subset(full_train_dataset, train_indices)
            
            # Для валидационного датасета создаем новый без аугментации
            self.val_dataset = ColorizationDataset(
                image_paths=[full_train_dataset.image_paths[i] for i in val_indices],
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                cache_size=self.cache_size
            )
            
        # Если указана тестовая директория, создаем тестовый датасет
        if self.test_dir:
            test_images = DatasetFromFolder(self.test_dir).image_paths
            self.test_dataset = ColorizationDataset(
                image_paths=test_images,
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                cache_size=self.cache_size
            )
            
    def get_train_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для обучающего датасета.
        
        Returns:
            DataLoader: Даталоадер для обучения
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_val_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для валидационного датасета.
        
        Returns:
            DataLoader: Даталоадер для валидации
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_test_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для тестового датасета.
        
        Returns:
            DataLoader: Даталоадер для тестирования
        """
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class ColorizationPairedDataModule:
    """
    Модуль данных для задачи колоризации с парными изображениями.
    
    Args:
        train_grayscale_dir (str): Директория с обучающими ЧБ изображениями
        train_color_dir (str): Директория с обучающими цветными изображениями
        val_grayscale_dir (str, optional): Директория с валидационными ЧБ изображениями
        val_color_dir (str, optional): Директория с валидационными цветными изображениями
        test_grayscale_dir (str, optional): Директория с тестовыми ЧБ изображениями
        test_color_dir (str, optional): Директория с тестовыми цветными изображениями
        batch_size (int): Размер батча
        num_workers (int): Количество рабочих процессов для загрузки данных
        color_space (str): Целевое цветовое пространство ('lab', 'rgb', 'yuv')
        image_size (int): Размер изображения
        split_ab (bool): Разделять ли каналы a и b в Lab пространстве
        augment_train (bool): Применять ли аугментацию к обучающим данным
        paired_transform (bool): Применять ли одинаковую аугментацию к парам
        val_split (float): Доля данных для валидации, если директории валидации не указаны
        cache_size (int): Размер кеша для изображений (0 - без кеша)
    """
    def __init__(self,
                 train_grayscale_dir: str,
                 train_color_dir: str,
                 val_grayscale_dir: Optional[str] = None,
                 val_color_dir: Optional[str] = None,
                 test_grayscale_dir: Optional[str] = None,
                 test_color_dir: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 color_space: str = 'lab',
                 image_size: int = 256,
                 split_ab: bool = True,
                 augment_train: bool = True,
                 paired_transform: bool = True,
                 val_split: float = 0.1,
                 cache_size: int = 0):
        self.train_grayscale_dir = train_grayscale_dir
        self.train_color_dir = train_color_dir
        self.val_grayscale_dir = val_grayscale_dir
        self.val_color_dir = val_color_dir
        self.test_grayscale_dir = test_grayscale_dir
        self.test_color_dir = test_color_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.color_space = color_space
        self.image_size = image_size
        self.split_ab = split_ab
        self.augment_train = augment_train
        self.paired_transform = paired_transform
        self.val_split = val_split
        self.cache_size = cache_size
        
        # Датасеты
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Создаем датасеты
        self._setup_datasets()
        
    def _setup_datasets(self):
        """
        Создает датасеты для обучения, валидации и тестирования.
        """
        # Находим все парные изображения в обучающих директориях
        paired_dataset = PairedDatasetFromFolders(
            grayscale_dir=self.train_grayscale_dir,
            color_dir=self.train_color_dir
        )
        
        # Создаем обучающий датасет
        full_train_dataset = ColorizationDatasetPaired(
            grayscale_paths=paired_dataset.grayscale_paths,
            color_paths=paired_dataset.color_paths,
            color_space=self.color_space,
            image_size=self.image_size,
            split_ab=self.split_ab,
            augment=self.augment_train,
            paired_transform=self.paired_transform,
            cache_size=self.cache_size
        )
        
        # Если указаны директории валидации, используем их
        if self.val_grayscale_dir and self.val_color_dir:
            val_paired_dataset = PairedDatasetFromFolders(
                grayscale_dir=self.val_grayscale_dir,
                color_dir=self.val_color_dir
            )
            self.val_dataset = ColorizationDatasetPaired(
                grayscale_paths=val_paired_dataset.grayscale_paths,
                color_paths=val_paired_dataset.color_paths,
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                paired_transform=False,
                cache_size=self.cache_size
            )
            self.train_dataset = full_train_dataset
        else:
            # Иначе разделяем обучающий датасет на обучающий и валидационный
            val_size = int(len(full_train_dataset) * self.val_split)
            train_size = len(full_train_dataset) - val_size
            
            # Создаем датасеты с соответствующими индексами
            train_indices, val_indices = random_split(
                range(len(full_train_dataset)),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.train_dataset = Subset(full_train_dataset, train_indices)
            
            # Для валидационного датасета создаем новый без аугментации
            self.val_dataset = ColorizationDatasetPaired(
                grayscale_paths=[paired_dataset.grayscale_paths[i] for i in val_indices],
                color_paths=[paired_dataset.color_paths[i] for i in val_indices],
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                paired_transform=False,
                cache_size=self.cache_size
            )
            
        # Если указаны тестовые директории, создаем тестовый датасет
        if self.test_grayscale_dir and self.test_color_dir:
            test_paired_dataset = PairedDatasetFromFolders(
                grayscale_dir=self.test_grayscale_dir,
                color_dir=self.test_color_dir
            )
            self.test_dataset = ColorizationDatasetPaired(
                grayscale_paths=test_paired_dataset.grayscale_paths,
                color_paths=test_paired_dataset.color_paths,
                color_space=self.color_space,
                image_size=self.image_size,
                split_ab=self.split_ab,
                augment=False,
                paired_transform=False,
                cache_size=self.cache_size
            )
            
    def get_train_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для обучающего датасета.
        
        Returns:
            DataLoader: Даталоадер для обучения
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_val_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для валидационного датасета.
        
        Returns:
            DataLoader: Даталоадер для валидации
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_test_dataloader(self) -> DataLoader:
        """
        Возвращает даталоадер для тестового датасета.
        
        Returns:
            DataLoader: Даталоадер для тестирования
        """
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_datamodule(config: Dict) -> Union[ColorizationDataModule, ColorizationPairedDataModule]:
    """
    Создает модуль данных на основе конфигурации.
    
    Args:
        config (Dict): Конфигурация модуля данных
        
    Returns:
        Union[ColorizationDataModule, ColorizationPairedDataModule]: Модуль данных
    """
    # Определяем тип модуля данных
    paired = config.get('paired', False)
    
    # Общие параметры
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    color_space = config.get('color_space', 'lab')
    image_size = config.get('image_size', 256)
    split_ab = config.get('split_ab', True)
    augment_train = config.get('augment_train', True)
    val_split = config.get('val_split', 0.1)
    cache_size = config.get('cache_size', 0)
    
    if paired:
        # Параметры для парного модуля данных
        train_grayscale_dir = config.get('train_grayscale_dir')
        train_color_dir = config.get('train_color_dir')
        val_grayscale_dir = config.get('val_grayscale_dir')
        val_color_dir = config.get('val_color_dir')
        test_grayscale_dir = config.get('test_grayscale_dir')
        test_color_dir = config.get('test_color_dir')
        paired_transform = config.get('paired_transform', True)
        
        return ColorizationPairedDataModule(
            train_grayscale_dir=train_grayscale_dir,
            train_color_dir=train_color_dir,
            val_grayscale_dir=val_grayscale_dir,
            val_color_dir=val_color_dir,
            test_grayscale_dir=test_grayscale_dir,
            test_color_dir=test_color_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            color_space=color_space,
            image_size=image_size,
            split_ab=split_ab,
            augment_train=augment_train,
            paired_transform=paired_transform,
            val_split=val_split,
            cache_size=cache_size
        )
    else:
        # Параметры для обычного модуля данных
        train_dir = config.get('train_dir')
        val_dir = config.get('val_dir')
        test_dir = config.get('test_dir')
        
        return ColorizationDataModule(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            color_space=color_space,
            image_size=image_size,
            split_ab=split_ab,
            augment_train=augment_train,
            val_split=val_split,
            cache_size=cache_size
        )


def prepare_batch_for_colorization(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Подготавливает батч для колоризации.
    
    Args:
        batch (Dict[str, torch.Tensor]): Батч из даталоадера
        device (torch.device): Устройство для вычислений
        
    Returns:
        Dict[str, torch.Tensor]: Подготовленный батч
    """
    result = {}
    
    # Копируем все тензоры на устройство
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
            
    # Добавляем комбинированные данные, если необходимо
    if 'l' in result and 'ab' in result:
        # Объединяем L и ab каналы для полного Lab изображения
        result['lab'] = torch.cat([result['l'], result['ab']], dim=1)
    elif 'l' in result and 'a' in result and 'b' in result:
        # Объединяем L, a и b каналы для полного Lab изображения
        result['lab'] = torch.cat([result['l'], result['a'], result['b']], dim=1)
        # Объединяем a и b каналы
        result['ab'] = torch.cat([result['a'], result['b']], dim=1)
    
    return result


if __name__ == "__main__":
    # Пример использования модуля загрузки данных
    
    # Конфигурация для создания модуля данных
    config = {
        'paired': True,
        'train_grayscale_dir': './data/train/grayscale',
        'train_color_dir': './data/train/color',
        'val_grayscale_dir': './data/val/grayscale',
        'val_color_dir': './data/val/color',
        'batch_size': 8,
        'num_workers': 2,
        'color_space': 'lab',
        'image_size': 256,
        'split_ab': True,
        'augment_train': True,
        'paired_transform': True,
        'cache_size': 100
    }
    
    # Создаем модуль данных
    try:
        datamodule = create_datamodule(config)
        
        # Получаем даталоадеры
        train_loader = datamodule.get_train_dataloader()
        val_loader = datamodule.get_val_dataloader()
        
        # Пример использования даталоадера
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Прерываем после первого батча для примера
            break
    except Exception as e:
        print(f"Ошибка при создании или использовании модуля данных: {e}")