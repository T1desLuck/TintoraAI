"""
Validation Dataset: Класс для валидационных данных колоризации.

Данный модуль расширяет базовый класс датасета для работы с валидационными данными.
Он обеспечивает стабильную, детерминированную загрузку изображений без аугментаций
для объективной оценки качества модели колоризации во время обучения.

Ключевые особенности:
- Отсутствие аугментаций для объективной оценки качества модели
- Сохранение метаданных для детального анализа производительности
- Поддержка различных метрик оценки качества колоризации
- Эффективная предобработка изображений для валидации

Преимущества:
- Надежная оценка обобщающей способности модели
- Отслеживание прогресса обучения на отложенном наборе данных
- Ранняя остановка обучения при отсутствии улучшений
- Возможность выбора лучшей модели на основе валидационных метрик
"""

import os
import random
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .base_dataset import BaseColorizationDataset, create_transform, create_paired_transform, get_color_transforms, COLOR_SPACES


class ValidationColorizationDataset(BaseColorizationDataset):
    """
    Класс для валидационных данных колоризации.
    
    Args:
        data_root (str): Корневая директория с данными
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        extensions (List[str], optional): Список поддерживаемых расширений файлов
        color_space (str): Цветовое пространство для преобразования ('rgb', 'lab', 'yuv')
        img_size (int): Размер выходного изображения
        center_crop (bool): Применять ли центральный кроп
        paired_data (bool): Режим работы с парными данными
        max_dataset_size (Optional[int]): Максимальный размер датасета
        use_cache (bool): Использовать кэширование для ускорения загрузки данных
        store_original_sizes (bool): Сохранять исходные размеры изображений для постобработки
        store_image_paths (bool): Сохранять пути к изображениям
    """
    def __init__(
        self,
        data_root: str,
        grayscale_dir: Optional[str] = None,
        color_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        color_space: str = 'lab',
        img_size: int = 256,
        center_crop: bool = True,
        paired_data: bool = True,
        max_dataset_size: Optional[int] = None,
        use_cache: bool = True,
        store_original_sizes: bool = True,
        store_image_paths: bool = True
    ):
        """Инициализирует валидационный датасет колоризации."""
        # Создаем базовую трансформацию для изменения размера и центрального кропа
        transform = create_paired_transform(
            img_size=img_size,
            random_crop=False,  # Для валидации не используем случайный кроп
            random_flip=False,   # Для валидации не используем случайное отражение
            resize_mode='resize_crop' if center_crop else 'resize'
        )
        
        # Инициализируем базовый класс
        super().__init__(
            data_root=data_root,
            grayscale_dir=grayscale_dir,
            color_dir=color_dir,
            extensions=extensions,
            color_space=color_space,
            img_size=img_size,
            paired_data=paired_data,
            transform=transform,
            grayscale_transform=None,  # Для валидации не используем дополнительные трансформации
            color_transform=None,      # Для валидации не используем дополнительные трансформации
            max_dataset_size=max_dataset_size,
            use_cache=use_cache
        )
        
        # Сохраняем дополнительные параметры
        self.store_original_sizes = store_original_sizes
        self.store_image_paths = store_image_paths
        
        # Если нужно хранить исходные размеры, загружаем их предварительно
        self.original_sizes = {}
        if self.store_original_sizes:
            for idx, path_info in enumerate(self.image_paths):
                if 'grayscale' in path_info:
                    img = Image.open(path_info['grayscale'])
                    self.original_sizes[idx] = img.size
                elif 'color' in path_info:
                    img = Image.open(path_info['color'])
                    self.original_sizes[idx] = img.size
                    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Получает элемент датасета по индексу.
        
        Args:
            index (int): Индекс элемента
            
        Returns:
            Dict[str, Any]: Словарь с изображениями и метаданными
        """
        # Получаем базовый элемент
        item = super().__getitem__(index)
        
        # Добавляем исходный размер, если нужно
        if self.store_original_sizes and index in self.original_sizes:
            item['original_size'] = self.original_sizes[index]
            
        # Добавляем пути к изображениям, если нужно
        if self.store_image_paths:
            path_info = self.image_paths[index]
            if 'grayscale' in path_info:
                item['grayscale_path'] = path_info['grayscale']
            if 'color' in path_info:
                item['color_path'] = path_info['color']
                
        return item


class SegmentedValidationDataset(ValidationColorizationDataset):
    """
    Класс для валидационных данных с сегментацией для отдельной оценки разных частей изображения.
    
    Args:
        data_root (str): Корневая директория с данными
        segmentation_dir (str): Директория с масками сегментации
        classes (List[str]): Список классов сегментации для оценки
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        extensions (List[str], optional): Список поддерживаемых расширений файлов
        color_space (str): Цветовое пространство для преобразования
        img_size (int): Размер выходного изображения
        center_crop (bool): Применять ли центральный кроп
        paired_data (bool): Режим работы с парными данными
        max_dataset_size (Optional[int]): Максимальный размер датасета
    """
    def __init__(
        self,
        data_root: str,
        segmentation_dir: str,
        classes: List[str],
        grayscale_dir: Optional[str] = None,
        color_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        color_space: str = 'lab',
        img_size: int = 256,
        center_crop: bool = True,
        paired_data: bool = True,
        max_dataset_size: Optional[int] = None
    ):
        """Инициализирует валидационный датасет с сегментацией."""
        # Инициализируем базовый класс
        super().__init__(
            data_root=data_root,
            grayscale_dir=grayscale_dir,
            color_dir=color_dir,
            extensions=extensions,
            color_space=color_space,
            img_size=img_size,
            center_crop=center_crop,
            paired_data=paired_data,
            max_dataset_size=max_dataset_size,
            use_cache=True,
            store_original_sizes=True,
            store_image_paths=True
        )
        
        # Сохраняем параметры сегментации
        self.segmentation_dir = os.path.join(data_root, segmentation_dir)
        self.classes = classes
        
        # Словарь для хранения масок сегментации
        self.segmentation_masks = {}
        
        # Загружаем маски сегментации
        self._load_segmentation_masks()
        
    def _load_segmentation_masks(self) -> None:
        """Загружает маски сегментации для изображений."""
        # Для каждого изображения в датасете
        for idx, path_info in enumerate(self.image_paths):
            image_id = path_info['id']
            
            # Для каждого класса сегментации
            masks = {}
            for class_name in self.classes:
                # Формируем путь к маске
                mask_path = os.path.join(self.segmentation_dir, class_name, f"{image_id}.png")
                
                # Если маска существует, загружаем ее
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path).convert('L')
                        
                        # Изменяем размер маски, чтобы соответствовал размеру изображения
                        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
                        
                        # Преобразуем в тензор
                        mask_tensor = TF.to_tensor(mask)
                        
                        # Бинаризуем маску
                        mask_tensor = (mask_tensor > 0.5).float()
                        
                        masks[class_name] = mask_tensor
                    except Exception as e:
                        print(f"Ошибка при загрузке маски {mask_path}: {str(e)}")
                        
            # Если есть маски, сохраняем их
            if masks:
                self.segmentation_masks[idx] = masks
                
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Получает элемент датасета с масками сегментации.
        
        Args:
            index (int): Индекс элемента
            
        Returns:
            Dict[str, Any]: Словарь с изображениями, масками и метаданными
        """
        # Получаем базовый элемент
        item = super().__getitem__(index)
        
        # Добавляем маски сегментации, если есть
        if index in self.segmentation_masks:
            item['segmentation_masks'] = self.segmentation_masks[index]
            
        return item


def create_validation_dataset(
    data_root: str,
    grayscale_dir: Optional[str] = None,
    color_dir: Optional[str] = None,
    color_space: str = 'lab',
    img_size: int = 256,
    center_crop: bool = True,
    max_dataset_size: Optional[int] = None,
    segmentation_dir: Optional[str] = None,
    segmentation_classes: Optional[List[str]] = None
) -> Union[ValidationColorizationDataset, SegmentedValidationDataset]:
    """
    Создает датасет для валидации модели колоризации.
    
    Args:
        data_root (str): Корневая директория с данными
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        color_space (str): Цветовое пространство для преобразования
        img_size (int): Размер выходного изображения
        center_crop (bool): Применять ли центральный кроп
        max_dataset_size (Optional[int]): Максимальный размер датасета
        segmentation_dir (Optional[str]): Директория с масками сегментации
        segmentation_classes (Optional[List[str]]): Список классов сегментации для оценки
        
    Returns:
        Union[ValidationColorizationDataset, SegmentedValidationDataset]: Созданный валидационный датасет
    """
    # Определяем, использовать ли датасет с сегментацией
    if segmentation_dir and segmentation_classes:
        return SegmentedValidationDataset(
            data_root=data_root,
            segmentation_dir=segmentation_dir,
            classes=segmentation_classes,
            grayscale_dir=grayscale_dir,
            color_dir=color_dir,
            color_space=color_space,
            img_size=img_size,
            center_crop=center_crop,
            paired_data=grayscale_dir is not None and color_dir is not None,
            max_dataset_size=max_dataset_size
        )
    else:
        return ValidationColorizationDataset(
            data_root=data_root,
            grayscale_dir=grayscale_dir,
            color_dir=color_dir,
            color_space=color_space,
            img_size=img_size,
            center_crop=center_crop,
            paired_data=grayscale_dir is not None and color_dir is not None,
            max_dataset_size=max_dataset_size,
            use_cache=True,
            store_original_sizes=True,
            store_image_paths=True
        )


def create_validation_dataloader(
    dataset: Union[ValidationColorizationDataset, SegmentedValidationDataset],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Создает загрузчик данных для валидации модели колоризации.
    
    Args:
        dataset (Union[ValidationColorizationDataset, SegmentedValidationDataset]): Валидационный датасет
        batch_size (int): Размер батча
        num_workers (int): Количество рабочих потоков для загрузки данных
        pin_memory (bool): Использовать ли закрепленную память для ускорения передачи на GPU
        
    Returns:
        DataLoader: Созданный загрузчик данных
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # Для валидации всегда используем последовательную выборку
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Для валидации не отбрасываем последний неполный батч
    )


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="TintoraAI Validation Dataset Test")
    parser.add_argument("--data_root", type=str, required=True, help="Путь к директории с данными")
    parser.add_argument("--grayscale_dir", type=str, help="Поддиректория с черно-белыми изображениями")
    parser.add_argument("--color_dir", type=str, help="Поддиректория с цветными изображениями")
    parser.add_argument("--color_space", type=str, default="lab", choices=COLOR_SPACES, help="Цветовое пространство")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--img_size", type=int, default=256, help="Размер изображения")
    parser.add_argument("--segmentation_dir", type=str, help="Директория с масками сегментации")
    parser.add_argument("--segmentation_classes", type=str, help="Список классов сегментации, разделенный запятыми")
    
    args = parser.parse_args()
    
    try:
        # Разбираем список классов сегментации, если указан
        segmentation_classes = None
        if args.segmentation_classes:
            segmentation_classes = args.segmentation_classes.split(',')
            
        # Создаем валидационный датасет
        dataset = create_validation_dataset(
            data_root=args.data_root,
            grayscale_dir=args.grayscale_dir,
            color_dir=args.color_dir,
            color_space=args.color_space,
            img_size=args.img_size,
            segmentation_dir=args.segmentation_dir,
            segmentation_classes=segmentation_classes
        )
        
        print(f"Создан валидационный датасет с {len(dataset)} изображениями")
        
        # Создаем загрузчик данных
        dataloader = create_validation_dataloader(
            dataset=dataset,
            batch_size=args.batch_size
        )
        
        print(f"Создан валидационный загрузчик данных с {len(dataloader)} батчами")
        
        # Получаем и визуализируем один батч
        batch = next(iter(dataloader))
        print(f"Размеры батча: grayscale={batch['grayscale'].shape}, color={batch['color'].shape}")
        
        # Визуализируем первые несколько изображений
        fig, axes = plt.subplots(min(args.batch_size, len(batch['grayscale'])), 2, figsize=(10, 5 * min(args.batch_size, len(batch['grayscale']))))
        
        # Получаем функцию преобразования цветового пространства, если нужно
        color_transforms = get_color_transforms(args.color_space)
        color_to_rgb = lambda x: x if args.color_space == 'rgb' else color_transforms[f'{args.color_space}_to_rgb'](x)
        
        for i in range(min(args.batch_size, len(batch['grayscale']))):
            # Преобразуем тензоры в изображения для отображения
            grayscale_img = batch['grayscale'][i].permute(1, 2, 0).cpu().numpy()
            color_img_tensor = color_to_rgb(batch['color'][i])
            color_img = color_img_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Отображаем изображения
            axes[i, 0].imshow(grayscale_img)
            axes[i, 0].set_title(f"Grayscale - {batch['id'][i]}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(color_img)
            axes[i, 1].set_title(f"Color ({args.color_space}) - {batch['id'][i]}")
            axes[i, 1].axis('off')
            
            # Если есть маски сегментации, отображаем их
            if 'segmentation_masks' in batch:
                for class_name, mask in batch['segmentation_masks'][i].items():
                    # Накладываем маску на цветное изображение для визуализации
                    mask_np = mask.squeeze().cpu().numpy()
                    overlay = color_img.copy()
                    overlay[mask_np > 0.5] = [1.0, 0.0, 0.0]  # Выделяем красным
                    
                    # Создаем дополнительные оси для отображения масок
                    fig_mask = plt.figure(figsize=(5, 5))
                    plt.imshow(overlay)
                    plt.title(f"Mask {class_name} - {batch['id'][i]}")
                    plt.axis('off')
                    plt.savefig(f"mask_{class_name}_{batch['id'][i]}.png")
                    plt.close(fig_mask)
        
        plt.tight_layout()
        plt.savefig("validation_samples.png")
        print(f"Образцы сохранены в validation_samples.png")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()