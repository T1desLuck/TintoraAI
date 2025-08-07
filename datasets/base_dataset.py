"""
Base Dataset: Базовый класс для датасетов колоризации.

Данный модуль предоставляет базовый класс для всех датасетов, используемых
при обучении и валидации моделей колоризации. Он обеспечивает общую функциональность
для загрузки, предобработки и преобразования изображений между различными
цветовыми пространствами.

Ключевые особенности:
- Гибкая структура для работы с различными организациями данных
- Поддержка различных цветовых пространств (RGB, Lab, YUV)
- Настраиваемые трансформации изображений с сохранением согласованности
- Эффективная обработка пар изображений (цветное/черно-белое)

Преимущества:
- Стандартизированный интерфейс для всех датасетов проекта
- Расширяемая архитектура для добавления новых функций
- Оптимизированная производительность с кэшированием данных
- Интеграция с экосистемой PyTorch для эффективного обучения
"""

import os
import glob
import random
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

# Поддерживаемые цветовые пространства
COLOR_SPACES = ['rgb', 'lab', 'yuv']


class BaseColorizationDataset(Dataset):
    """
    Базовый класс для датасетов колоризации.
    
    Args:
        data_root (str): Корневая директория с данными
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        extensions (List[str], optional): Список поддерживаемых расширений файлов
        color_space (str): Цветовое пространство для преобразования ('rgb', 'lab', 'yuv')
        img_size (int): Размер выходного изображения (изображения будут приведены к квадратному размеру)
        paired_data (bool): Режим работы с парными данными (когда цветное и ч/б хранятся в отдельных директориях)
        transform (Callable, optional): Трансформация, применяемая к обоим изображениям
        grayscale_transform (Callable, optional): Трансформация, применяемая только к ч/б изображениям
        color_transform (Callable, optional): Трансформация, применяемая только к цветным изображениям
        max_dataset_size (int, optional): Максимальный размер датасета (для ограничения количества изображений)
        use_cache (bool): Использовать кэширование для ускорения загрузки данных
    """
    def __init__(
        self,
        data_root: str,
        grayscale_dir: Optional[str] = None,
        color_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        color_space: str = 'lab',
        img_size: int = 256,
        paired_data: bool = True,
        transform: Optional[Callable] = None,
        grayscale_transform: Optional[Callable] = None,
        color_transform: Optional[Callable] = None,
        max_dataset_size: Optional[int] = None,
        use_cache: bool = False
    ):
        """Инициализирует датасет колоризации."""
        self.data_root = data_root
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.tiff', '.webp']
        self.color_space = color_space.lower()
        self.img_size = img_size
        self.paired_data = paired_data
        self.transform = transform
        self.grayscale_transform = grayscale_transform
        self.color_transform = color_transform
        self.max_dataset_size = max_dataset_size
        self.use_cache = use_cache
        
        # Проверка поддерживаемого цветового пространства
        if self.color_space not in COLOR_SPACES:
            raise ValueError(f"Неподдерживаемое цветовое пространство: {self.color_space}. "
                             f"Выберите одно из: {COLOR_SPACES}")
        
        # Инициализируем трансформации для цветовых пространств
        self.color_transforms = get_color_transforms(self.color_space)
        
        # Получаем пути к файлам
        self.image_paths = self._get_image_paths()
        
        # Ограничиваем размер датасета, если указано
        if self.max_dataset_size is not None:
            self.image_paths = self.image_paths[:min(self.max_dataset_size, len(self.image_paths))]
            
        # Инициализируем кэш
        self.cache = {} if self.use_cache else None
        
    def _get_image_paths(self) -> List[Dict[str, str]]:
        """
        Получает пути к изображениям в зависимости от структуры данных.
        
        Returns:
            List[Dict[str, str]]: Список словарей с путями к изображениям
        """
        image_paths = []
        
        if self.paired_data and self.grayscale_dir and self.color_dir:
            # Режим парных данных в отдельных директориях
            grayscale_root = os.path.join(self.data_root, self.grayscale_dir)
            color_root = os.path.join(self.data_root, self.color_dir)
            
            # Получаем список изображений в директории ч/б изображений
            grayscale_files = []
            for ext in self.extensions:
                grayscale_files.extend(glob.glob(os.path.join(grayscale_root, f"*{ext}")))
                grayscale_files.extend(glob.glob(os.path.join(grayscale_root, f"*{ext.upper()}")))
            
            # Для каждого ч/б изображения ищем соответствующее цветное
            for grayscale_path in grayscale_files:
                basename = os.path.basename(grayscale_path)
                base, ext = os.path.splitext(basename)
                
                # Ищем соответствующее цветное изображение
                color_path = None
                for color_ext in self.extensions:
                    candidate = os.path.join(color_root, f"{base}{color_ext}")
                    if os.path.exists(candidate):
                        color_path = candidate
                        break
                        
                    # Проверяем с расширением в верхнем регистре
                    candidate = os.path.join(color_root, f"{base}{color_ext.upper()}")
                    if os.path.exists(candidate):
                        color_path = candidate
                        break
                        
                # Если нашли соответствующее цветное изображение, добавляем пару в список
                if color_path:
                    image_paths.append({
                        'grayscale': grayscale_path,
                        'color': color_path,
                        'id': base
                    })
        else:
            # Режим непарных данных или когда пути не указаны
            # Просто ищем все изображения в корневой директории
            image_files = []
            for ext in self.extensions:
                image_files.extend(glob.glob(os.path.join(self.data_root, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(self.data_root, f"*{ext.upper()}")))
            
            for image_path in image_files:
                basename = os.path.basename(image_path)
                base, _ = os.path.splitext(basename)
                
                image_paths.append({
                    'color': image_path,
                    'id': base
                })
                
        return image_paths
    
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Получает пару изображений (цветное и ч/б) по индексу.
        
        Args:
            index (int): Индекс элемента
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с изображениями и метаданными
        """
        # Проверяем, есть ли элемент в кэше
        if self.use_cache and index in self.cache:
            return self.cache[index]
            
        # Получаем пути к изображениям
        paths = self.image_paths[index]
        
        # Загружаем цветное изображение
        color_path = paths['color']
        color_img = self._load_image(color_path)
        
        # Загружаем или создаем ч/б изображение
        if 'grayscale' in paths:
            grayscale_path = paths['grayscale']
            grayscale_img = self._load_image(grayscale_path)
        else:
            # Создаем ч/б изображение из цветного
            grayscale_img = self._create_grayscale(color_img)
            
        # Применяем общую трансформацию, если указана
        if self.transform:
            # Для общей трансформации передаем оба изображения
            grayscale_img, color_img = self.transform(grayscale_img, color_img)
            
        # Применяем отдельные трансформации, если указаны
        if self.grayscale_transform:
            grayscale_img = self.grayscale_transform(grayscale_img)
        if self.color_transform:
            color_img = self.color_transform(color_img)
            
        # Преобразуем изображения в тензоры, если они еще не являются тензорами
        if not isinstance(grayscale_img, torch.Tensor):
            grayscale_img = TF.to_tensor(grayscale_img)
        if not isinstance(color_img, torch.Tensor):
            color_img = TF.to_tensor(color_img)
            
        # Преобразуем в нужное цветовое пространство
        color_img = self._convert_to_target_colorspace(color_img)
        
        # Создаем результат
        result = {
            'grayscale': grayscale_img,
            'color': color_img,
            'id': paths['id'],
            'color_path': color_path
        }
        
        if 'grayscale' in paths:
            result['grayscale_path'] = paths['grayscale']
            
        # Кэшируем результат, если включено кэширование
        if self.use_cache:
            self.cache[index] = result
            
        return result
    
    def _load_image(self, path: str) -> Image.Image:
        """
        Загружает изображение по пути.
        
        Args:
            path (str): Путь к изображению
            
        Returns:
            Image.Image: Загруженное изображение
        """
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            raise IOError(f"Ошибка при загрузке изображения {path}: {str(e)}")
    
    def _create_grayscale(self, color_img: Image.Image) -> Image.Image:
        """
        Создает черно-белое изображение из цветного.
        
        Args:
            color_img (Image.Image): Цветное изображение
            
        Returns:
            Image.Image: Черно-белое изображение
        """
        return color_img.convert('L').convert('RGB')
    
    def _convert_to_target_colorspace(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор изображения в целевое цветовое пространство.
        
        Args:
            img_tensor (torch.Tensor): Тензор изображения в формате RGB
            
        Returns:
            torch.Tensor: Тензор в целевом цветовом пространстве
        """
        if self.color_space == 'rgb':
            # Уже в RGB, ничего не делаем
            return img_tensor
            
        # Получаем соответствующее преобразование
        return self.color_transforms['rgb_to_' + self.color_space](img_tensor)
    
    def get_color_conversion_function(self, source: str, target: str) -> Callable:
        """
        Возвращает функцию преобразования между цветовыми пространствами.
        
        Args:
            source (str): Исходное цветовое пространство
            target (str): Целевое цветовое пространство
            
        Returns:
            Callable: Функция преобразования
        """
        key = f"{source}_to_{target}"
        if key in self.color_transforms:
            return self.color_transforms[key]
        else:
            raise ValueError(f"Неподдерживаемое преобразование цветового пространства: {key}")


def create_transform(
    img_size: int = 256,
    crop_size: Optional[int] = None,
    resize_mode: str = 'resize',
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    apply_grayscale: bool = False
) -> Callable:
    """
    Создает трансформацию для одиночного изображения.
    
    Args:
        img_size (int): Размер изображения для изменения размера
        crop_size (int, optional): Размер для центрального кропа
        resize_mode (str): Режим изменения размера ('resize', 'resize_crop', 'crop')
        normalize (bool): Применять ли нормализацию
        mean (List[float], optional): Среднее значение для нормализации
        std (List[float], optional): Стандартное отклонение для нормализации
        apply_grayscale (bool): Преобразовывать ли изображение в оттенки серого
        
    Returns:
        Callable: Функция трансформации
    """
    transforms = []
    
    # Определяем преобразования размера в зависимости от режима
    if resize_mode == 'resize':
        transforms.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC))
    elif resize_mode == 'resize_crop':
        transforms.append(T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC))
        transforms.append(T.CenterCrop(img_size))
    elif resize_mode == 'crop':
        crop_size = crop_size or img_size
        transforms.append(T.CenterCrop(crop_size))
        if crop_size != img_size:
            transforms.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC))
    else:
        raise ValueError(f"Неизвестный режим изменения размера: {resize_mode}")
    
    # Добавляем преобразование в оттенки серого, если нужно
    if apply_grayscale:
        transforms.append(T.Grayscale(num_output_channels=1))
        
    # Добавляем преобразование в тензор
    transforms.append(T.ToTensor())
    
    # Добавляем нормализацию, если нужно
    if normalize:
        mean = mean or [0.5, 0.5, 0.5]
        std = std or [0.5, 0.5, 0.5]
        transforms.append(T.Normalize(mean, std))
        
    return T.Compose(transforms)


def create_paired_transform(
    img_size: int = 256,
    crop_size: Optional[int] = None,
    random_crop: bool = False,
    random_flip: bool = False,
    resize_mode: str = 'resize'
) -> Callable:
    """
    Создает трансформацию, которая применяется к паре изображений одновременно.
    
    Args:
        img_size (int): Размер изображения для изменения размера
        crop_size (int, optional): Размер для кропа
        random_crop (bool): Использовать ли случайный кроп
        random_flip (bool): Использовать ли случайное отражение по горизонтали
        resize_mode (str): Режим изменения размера ('resize', 'resize_crop', 'crop')
        
    Returns:
        Callable: Функция трансформации для пары изображений
    """
    def paired_transform(grayscale_img: Image.Image, color_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Проверка размеров
        if grayscale_img.size != color_img.size:
            # Если размеры не совпадают, изменяем размер цветного изображения под ч/б
            color_img = color_img.resize(grayscale_img.size, Image.BICUBIC)
            
        # Применяем изменение размера в зависимости от режима
        if resize_mode == 'resize':
            grayscale_img = TF.resize(grayscale_img, (img_size, img_size), interpolation=TF.InterpolationMode.BICUBIC)
            color_img = TF.resize(color_img, (img_size, img_size), interpolation=TF.InterpolationMode.BICUBIC)
        elif resize_mode == 'resize_crop':
            grayscale_img = TF.resize(grayscale_img, img_size, interpolation=TF.InterpolationMode.BICUBIC)
            color_img = TF.resize(color_img, img_size, interpolation=TF.InterpolationMode.BICUBIC)
            
            if random_crop:
                # Получаем параметры для случайного кропа
                i, j, h, w = T.RandomCrop.get_params(grayscale_img, output_size=(img_size, img_size))
                grayscale_img = TF.crop(grayscale_img, i, j, h, w)
                color_img = TF.crop(color_img, i, j, h, w)
            else:
                # Применяем центральный кроп
                grayscale_img = TF.center_crop(grayscale_img, img_size)
                color_img = TF.center_crop(color_img, img_size)
        elif resize_mode == 'crop':
            crop_size = crop_size or img_size
            
            if random_crop:
                # Получаем параметры для случайного кропа
                i, j, h, w = T.RandomCrop.get_params(grayscale_img, output_size=(crop_size, crop_size))
                grayscale_img = TF.crop(grayscale_img, i, j, h, w)
                color_img = TF.crop(color_img, i, j, h, w)
            else:
                # Применяем центральный кроп
                grayscale_img = TF.center_crop(grayscale_img, crop_size)
                color_img = TF.center_crop(color_img, crop_size)
                
            if crop_size != img_size:
                grayscale_img = TF.resize(grayscale_img, (img_size, img_size), interpolation=TF.InterpolationMode.BICUBIC)
                color_img = TF.resize(color_img, (img_size, img_size), interpolation=TF.InterpolationMode.BICUBIC)
                
        # Применяем случайное отражение, если нужно
        if random_flip and random.random() > 0.5:
            grayscale_img = TF.hflip(grayscale_img)
            color_img = TF.hflip(color_img)
            
        return grayscale_img, color_img
        
    return paired_transform


def get_color_transforms(color_space: str) -> Dict[str, Callable]:
    """
    Создает словарь с функциями преобразования между цветовыми пространствами.
    
    Args:
        color_space (str): Целевое цветовое пространство
        
    Returns:
        Dict[str, Callable]: Словарь с функциями преобразования
    """
    # Создаем словарь для функций преобразования
    transforms = {}
    
    # Преобразование RGB -> LAB
    def rgb_to_lab(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор изображения из RGB в LAB.
        
        Args:
            img_tensor (torch.Tensor): RGB-тензор изображения [C, H, W]
            
        Returns:
            torch.Tensor: LAB-тензор изображения [C, H, W]
        """
        # Преобразуем тензор в numpy-массив
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Масштабируем значения из [0, 1] в [0, 255]
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
            
        # Преобразуем в LAB
        lab_img = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Нормализуем: L в [0, 1], a и b в [-1, 1]
        lab_img[:, :, 0] /= 100.0  # L: [0, 100] -> [0, 1]
        lab_img[:, :, 1:] /= 127.0  # a, b: [-127, 127] -> [-1, 1]
        
        # Преобразуем обратно в тензор
        lab_tensor = torch.from_numpy(lab_img).permute(2, 0, 1).float()
        
        return lab_tensor
        
    # Преобразование LAB -> RGB
    def lab_to_rgb(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор изображения из LAB в RGB.
        
        Args:
            img_tensor (torch.Tensor): LAB-тензор изображения [C, H, W]
            
        Returns:
            torch.Tensor: RGB-тензор изображения [C, H, W]
        """
        # Преобразуем тензор в numpy-массив
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Денормализуем: L из [0, 1] в [0, 100], a и b из [-1, 1] в [-127, 127]
        img_np[:, :, 0] *= 100.0
        img_np[:, :, 1:] *= 127.0
        
        # Преобразуем в RGB
        rgb_img = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_LAB2RGB)
        
        # Масштабируем значения в [0, 1]
        rgb_img = np.clip(rgb_img, 0, 255) / 255.0
        
        # Преобразуем в тензор
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        
        return rgb_tensor
        
    # Преобразование RGB -> YUV
    def rgb_to_yuv(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор изображения из RGB в YUV.
        
        Args:
            img_tensor (torch.Tensor): RGB-тензор изображения [C, H, W]
            
        Returns:
            torch.Tensor: YUV-тензор изображения [C, H, W]
        """
        # Преобразуем тензор в numpy-массив
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Масштабируем значения из [0, 1] в [0, 255]
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
            
        # Преобразуем в YUV
        yuv_img = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2YUV).astype(np.float32)
        
        # Нормализуем: Y в [0, 1], U и V в [-0.5, 0.5]
        yuv_img[:, :, 0] /= 255.0  # Y: [0, 255] -> [0, 1]
        yuv_img[:, :, 1:] = (yuv_img[:, :, 1:] - 128.0) / 255.0  # U, V: [0, 255] -> [-0.5, 0.5]
        
        # Преобразуем в тензор
        yuv_tensor = torch.from_numpy(yuv_img).permute(2, 0, 1).float()
        
        return yuv_tensor
        
    # Преобразование YUV -> RGB
    def yuv_to_rgb(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует тензор изображения из YUV в RGB.
        
        Args:
            img_tensor (torch.Tensor): YUV-тензор изображения [C, H, W]
            
        Returns:
            torch.Tensor: RGB-тензор изображения [C, H, W]
        """
        # Преобразуем тензор в numpy-массив
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Денормализуем: Y из [0, 1] в [0, 255], U и V из [-0.5, 0.5] в [0, 255]
        img_np[:, :, 0] *= 255.0
        img_np[:, :, 1:] = img_np[:, :, 1:] * 255.0 + 128.0
        
        # Преобразуем в RGB
        rgb_img = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_YUV2RGB)
        
        # Масштабируем значения в [0, 1]
        rgb_img = np.clip(rgb_img, 0, 255) / 255.0
        
        # Преобразуем в тензор
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        
        return rgb_tensor
        
    # Добавляем функции преобразования в словарь
    transforms['rgb_to_lab'] = rgb_to_lab
    transforms['lab_to_rgb'] = lab_to_rgb
    transforms['rgb_to_yuv'] = rgb_to_yuv
    transforms['yuv_to_rgb'] = yuv_to_rgb
    
    return transforms


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    parser = argparse.ArgumentParser(description="TintoraAI Base Dataset Test")
    parser.add_argument("--data_root", type=str, required=True, help="Путь к директории с данными")
    parser.add_argument("--grayscale_dir", type=str, help="Поддиректория с черно-белыми изображениями")
    parser.add_argument("--color_dir", type=str, help="Поддиректория с цветными изображениями")
    parser.add_argument("--color_space", type=str, default="lab", choices=COLOR_SPACES, help="Цветовое пространство")
    parser.add_argument("--num_samples", type=int, default=4, help="Количество образцов для визуализации")
    parser.add_argument("--img_size", type=int, default=256, help="Размер изображения")
    
    args = parser.parse_args()
    
    try:
        # Создаем датасет
        dataset = BaseColorizationDataset(
            data_root=args.data_root,
            grayscale_dir=args.grayscale_dir,
            color_dir=args.color_dir,
            color_space=args.color_space,
            img_size=args.img_size,
            paired_data=args.grayscale_dir is not None and args.color_dir is not None,
            transform=create_paired_transform(img_size=args.img_size)
        )
        
        print(f"Создан датасет с {len(dataset)} изображениями")
        
        # Визуализируем несколько образцов
        fig, axes = plt.subplots(args.num_samples, 2, figsize=(10, 5 * args.num_samples))
        
        for i in range(args.num_samples):
            if i >= len(dataset):
                break
                
            sample = dataset[i]
            grayscale_img = sample['grayscale']
            color_img = sample['color']
            
            # Если изображение не в RGB, преобразуем для отображения
            if args.color_space != 'rgb':
                # Получаем функцию преобразования
                convert_fn = dataset.get_color_conversion_function(args.color_space, 'rgb')
                color_img_rgb = convert_fn(color_img)
            else:
                color_img_rgb = color_img
            
            # Отображаем изображения
            axes[i, 0].imshow(grayscale_img.permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title(f"Grayscale - {sample['id']}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(color_img_rgb.permute(1, 2, 0).cpu().numpy())
            axes[i, 1].set_title(f"Color ({args.color_space}) - {sample['id']}")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.savefig("dataset_samples.png")
        print(f"Образцы сохранены в dataset_samples.png")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")