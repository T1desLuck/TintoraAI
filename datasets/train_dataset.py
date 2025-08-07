"""
Train Dataset: Класс для тренировочных данных колоризации.

Данный модуль расширяет базовый класс датасета, добавляя специфические функции
и трансформации, необходимые для обучения моделей колоризации. Он предоставляет
расширенные возможности аугментации данных, динамическое управление выборкой
и балансировку классов для улучшения процесса обучения.

Ключевые особенности:
- Расширенная аугментация данных с сохранением соответствия пар изображений
- Стратегии динамической выборки для улучшения сходимости модели
- Поддержка различных режимов сэмплирования и балансировки данных
- Оптимизации для эффективной загрузки и обработки больших наборов данных

Преимущества:
- Улучшенная обобщающая способность модели благодаря разнообразным аугментациям
- Гибкое управление сложностью данных в процессе обучения
- Контролируемое управление соотношением различных типов изображений
- Эффективное использование памяти и вычислительных ресурсов
"""

import os
import random
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler, DistributedSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
import cv2
import albumentations as A

from .base_dataset import BaseColorizationDataset, create_transform, create_paired_transform, get_color_transforms, COLOR_SPACES


class TrainColorizationDataset(BaseColorizationDataset):
    """
    Класс для тренировочных данных колоризации.
    
    Args:
        data_root (str): Корневая директория с данными
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        extensions (List[str], optional): Список поддерживаемых расширений файлов
        color_space (str): Цветовое пространство для преобразования ('rgb', 'lab', 'yuv')
        img_size (int): Размер выходного изображения
        augmentation_level (str): Уровень аугментации ('none', 'light', 'medium', 'heavy')
        random_crop (bool): Использовать ли случайный кроп
        random_flip (bool): Использовать ли случайное отражение по горизонтали
        paired_data (bool): Режим работы с парными данными
        max_dataset_size (int, optional): Максимальный размер датасета
        use_cache (bool): Использовать кэширование для ускорения загрузки данных
        sampling_strategy (str): Стратегия выборки данных ('random', 'balanced', 'weighted')
        difficult_samples_weight (float): Вес для сложных образцов при использовании взвешенной выборки
        dynamic_difficulty (bool): Использовать ли динамическую сложность выборки
        reference_dataset_path (str, optional): Путь к референсному датасету для стилизации
    """
    def __init__(
        self,
        data_root: str,
        grayscale_dir: Optional[str] = None,
        color_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        color_space: str = 'lab',
        img_size: int = 256,
        augmentation_level: str = 'medium',
        random_crop: bool = True,
        random_flip: bool = True,
        paired_data: bool = True,
        max_dataset_size: Optional[int] = None,
        use_cache: bool = False,
        sampling_strategy: str = 'random',
        difficult_samples_weight: float = 2.0,
        dynamic_difficulty: bool = False,
        reference_dataset_path: Optional[str] = None
    ):
        """Инициализирует тренировочный датасет колоризации."""
        # Создаем трансформации на основе уровня аугментации
        transform, grayscale_transform, color_transform = self._create_augmentations(
            augmentation_level, img_size, random_crop, random_flip
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
            grayscale_transform=grayscale_transform,
            color_transform=color_transform,
            max_dataset_size=max_dataset_size,
            use_cache=use_cache
        )
        
        # Параметры выборки
        self.sampling_strategy = sampling_strategy
        self.difficult_samples_weight = difficult_samples_weight
        self.dynamic_difficulty = dynamic_difficulty
        
        # Инициализируем веса сложности для каждого изображения
        self.difficulty_weights = np.ones(len(self.image_paths))
        
        # Загружаем референсный датасет для стилизации, если указан
        self.reference_dataset = None
        if reference_dataset_path and os.path.exists(reference_dataset_path):
            try:
                self.reference_dataset = BaseColorizationDataset(
                    data_root=reference_dataset_path,
                    color_space=color_space,
                    img_size=img_size,
                    transform=create_paired_transform(img_size=img_size),
                    paired_data=False
                )
                print(f"Загружен референсный датасет с {len(self.reference_dataset)} изображениями")
            except Exception as e:
                print(f"Ошибка при загрузке референсного датасета: {str(e)}")
                self.reference_dataset = None
    
    def _create_augmentations(
        self,
        level: str,
        img_size: int,
        random_crop: bool,
        random_flip: bool
    ) -> Tuple[Callable, Callable, Callable]:
        """
        Создает функции аугментации на основе уровня аугментации.
        
        Args:
            level (str): Уровень аугментации ('none', 'light', 'medium', 'heavy')
            img_size (int): Размер выходного изображения
            random_crop (bool): Использовать ли случайный кроп
            random_flip (bool): Использовать ли случайное отражение по горизонтали
            
        Returns:
            Tuple[Callable, Callable, Callable]: Функции трансформации для пары, ч/б и цветных изображений
        """
        # Создаем базовую парную трансформацию
        paired_transform = create_paired_transform(
            img_size=img_size, 
            random_crop=random_crop, 
            random_flip=random_flip
        )
        
        # Определяем аугментации для черно-белых изображений
        grayscale_aug = None
        if level != 'none':
            grayscale_transforms = []
            
            if level in ['light', 'medium', 'heavy']:
                # Базовые аугментации для всех уровней кроме 'none'
                grayscale_transforms.append(T.RandomAdjustSharpness(sharpness_factor=2, p=0.3))
                grayscale_transforms.append(T.RandomAutocontrast(p=0.3))
            
            if level in ['medium', 'heavy']:
                # Дополнительные аугментации для средних и высоких уровней
                grayscale_transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
                grayscale_transforms.append(T.RandomPosterize(bits=6, p=0.2))
            
            if level == 'heavy':
                # Сильные аугментации только для высокого уровня
                grayscale_transforms.append(T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5))
                grayscale_transforms.append(T.ElasticTransform(alpha=50.0, sigma=4.0, p=0.3))
                
            if grayscale_transforms:
                grayscale_aug = T.Compose(grayscale_transforms)
                
        # Определяем аугментации для цветных изображений
        color_aug = None
        if level != 'none':
            color_transforms = []
            
            if level in ['light', 'medium', 'heavy']:
                # Базовые аугментации для всех уровней кроме 'none'
                color_transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
            
            if level in ['medium', 'heavy']:
                # Дополнительные аугментации для средних и высоких уровней
                color_transforms.append(T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3))
                color_transforms.append(T.RandomAutocontrast(p=0.2))
            
            if level == 'heavy':
                # Сильные аугментации только для высокого уровня
                # Для цветных изображений мы более осторожны с аугментациями,
                # чтобы не исказить целевое цветовое пространство
                color_transforms.append(T.RandomEqualize(p=0.2))
                
            if color_transforms:
                color_aug = T.Compose(color_transforms)
        
        return paired_transform, grayscale_aug, color_aug
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Получает элемент датасета по индексу с применением тренировочных аугментаций.
        
        Args:
            index (int): Индекс элемента
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с изображениями и метаданными
        """
        # Получаем базовый элемент
        item = super().__getitem__(index)
        
        # Применяем референсное изображение для стилизации, если доступно
        if self.reference_dataset is not None and random.random() < 0.3:  # 30% шанс применения стиля
            # Выбираем случайное референсное изображение
            ref_idx = random.randint(0, len(self.reference_dataset) - 1)
            ref_item = self.reference_dataset[ref_idx]
            
            # Добавляем референсное изображение к элементу
            item['reference'] = ref_item['color']
            item['reference_id'] = ref_item['id']
            
        # Обновляем вес сложности, если используется динамическая сложность
        if self.dynamic_difficulty and random.random() < 0.01:  # Обновляем веса с 1% шансом для каждого элемента
            self._update_difficulty_weight(index, item)
            
        return item
    
    def _update_difficulty_weight(self, index: int, item: Dict[str, torch.Tensor]) -> None:
        """
        Обновляет вес сложности для элемента на основе его характеристик.
        
        Args:
            index (int): Индекс элемента
            item (Dict[str, torch.Tensor]): Данные элемента
        """
        # В реальном приложении здесь может быть более сложная логика определения сложности
        # На основе характеристик изображения, метрик качества и т.д.
        
        # Пример простой реализации: используем энтропию изображения как меру сложности
        grayscale_img = item['grayscale']
        color_img = item['color']
        
        # Преобразуем тензоры в numpy для вычисления энтропии
        grayscale_np = grayscale_img.cpu().numpy().mean(axis=0)  # Усредняем по каналам
        
        # Вычисляем энтропию (мера сложности текстуры)
        hist, _ = np.histogram(grayscale_np, bins=64, range=(0, 1), density=True)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Нормализуем энтропию в диапазон [0.5, 2.0]
        normalized_entropy = 0.5 + 1.5 * (entropy / 6.0)  # 6.0 - примерное максимальное значение энтропии
        
        # Обновляем вес сложности
        self.difficulty_weights[index] = normalized_entropy
    
    def get_weights(self) -> np.ndarray:
        """
        Возвращает веса для каждого элемента датасета, используемые для взвешенной выборки.
        
        Returns:
            np.ndarray: Массив весов
        """
        return self.difficulty_weights
    
    def update_sampling_strategy(self, strategy: str) -> None:
        """
        Обновляет стратегию выборки данных.
        
        Args:
            strategy (str): Новая стратегия выборки ('random', 'balanced', 'weighted')
        """
        self.sampling_strategy = strategy


class DynamicBatchSampler(Sampler):
    """
    Сэмплер, который динамически формирует батчи на основе сложности изображений.
    
    Args:
        dataset (TrainColorizationDataset): Датасет для выборки
        batch_size (int): Базовый размер батча
        num_iterations (int): Количество итераций (батчей) за эпоху
        drop_last (bool): Отбрасывать ли последний неполный батч
        strategy (str): Стратегия формирования батчей ('random', 'similar', 'diverse')
    """
    def __init__(
        self,
        dataset: TrainColorizationDataset,
        batch_size: int,
        num_iterations: int,
        drop_last: bool = True,
        strategy: str = 'random'
    ):
        """Инициализирует сэмплер."""
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.drop_last = drop_last
        self.strategy = strategy
        self.weights = dataset.get_weights() if hasattr(dataset, 'get_weights') else None
    
    def __iter__(self):
        """Итератор по батчам."""
        # Формируем батчи в зависимости от стратегии
        if self.strategy == 'random':
            # Случайная выборка
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            
            for i in range(self.num_iterations):
                batch_indices = indices[(i * self.batch_size) % len(indices):((i * self.batch_size) % len(indices)) + self.batch_size]
                # Если нужно дополнить батч
                if len(batch_indices) < self.batch_size and not self.drop_last:
                    additional_indices = indices[:self.batch_size - len(batch_indices)]
                    batch_indices.extend(additional_indices)
                
                yield batch_indices
                
        elif self.strategy == 'similar':
            # Группируем похожие по сложности изображения
            indices = list(range(len(self.dataset)))
            
            if self.weights is not None:
                # Сортируем индексы по весам сложности
                sorted_indices = sorted(indices, key=lambda idx: self.weights[idx])
                
                # Формируем батчи из изображений с похожей сложностью
                for i in range(self.num_iterations):
                    start_idx = (i * self.batch_size) % len(sorted_indices)
                    batch_indices = sorted_indices[start_idx:start_idx + self.batch_size]
                    
                    # Если не хватает элементов, дополняем с начала
                    if len(batch_indices) < self.batch_size and not self.drop_last:
                        additional_indices = sorted_indices[:self.batch_size - len(batch_indices)]
                        batch_indices.extend(additional_indices)
                    
                    yield batch_indices
            else:
                # Если веса не доступны, используем случайную выборку
                for i in range(self.num_iterations):
                    random.shuffle(indices)
                    batch_indices = indices[:self.batch_size]
                    yield batch_indices
                    
        elif self.strategy == 'diverse':
            # Формируем батчи с разнообразной сложностью
            indices = list(range(len(self.dataset)))
            
            for i in range(self.num_iterations):
                if self.weights is not None:
                    # Формируем батч, чередуя изображения разной сложности
                    sorted_indices = sorted(indices, key=lambda idx: self.weights[idx])
                    num_groups = min(self.batch_size, 5)  # Количество групп сложности
                    batch_indices = []
                    
                    for j in range(self.batch_size):
                        # Выбираем индекс из группы сложности
                        group_idx = j % num_groups
                        group_size = len(sorted_indices) // num_groups
                        idx_in_group = random.randint(0, group_size - 1)
                        selected_idx = sorted_indices[group_idx * group_size + idx_in_group]
                        batch_indices.append(selected_idx)
                else:
                    # Если веса не доступны, используем случайную выборку
                    random.shuffle(indices)
                    batch_indices = indices[:self.batch_size]
                    
                yield batch_indices
    
    def __len__(self) -> int:
        """Возвращает количество батчей за эпоху."""
        return self.num_iterations


def create_train_augmentations(
    augmentation_level: str = 'medium',
    img_size: int = 256,
    color_space: str = 'lab'
) -> Tuple[Callable, Callable, Callable]:
    """
    Создает расширенные функции аугментации для тренировочных данных.
    
    Args:
        augmentation_level (str): Уровень аугментации ('none', 'light', 'medium', 'heavy')
        img_size (int): Размер выходного изображения
        color_space (str): Целевое цветовое пространство
        
    Returns:
        Tuple[Callable, Callable, Callable]: Функции трансформации для пары, ч/б и цветных изображений
    """
    # Создаем базовую трансформацию изменения размера
    resize_transform = create_paired_transform(img_size=img_size)
    
    # Если уровень аугментации 'none', возвращаем только изменение размера
    if augmentation_level == 'none':
        return resize_transform, None, None
        
    # Создаем augmentations с использованием albumentations для более сложных преобразований
    albumentation_transforms = []
    
    if augmentation_level in ['light', 'medium', 'heavy']:
        # Базовые аугментации для всех уровней кроме 'none'
        albumentation_transforms.extend([
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            ], p=0.3)
        ])
    
    if augmentation_level in ['medium', 'heavy']:
        # Средние аугментации
        albumentation_transforms.extend([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1.0, shift_limit=0.5, p=0.5),
            ], p=0.3)
        ])
    
    if augmentation_level == 'heavy':
        # Сильные аугментации
        albumentation_transforms.extend([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.RandomShadow(p=0.3)
        ])
    
    # Создаем композицию albumentations трансформаций
    albumentations_compose = A.Compose(
        albumentation_transforms,
        additional_targets={'color': 'image'}
    )
    
    # Создаем функцию для применения albumentation трансформаций к паре изображений
    def paired_albumentation_transform(grayscale_img: Image.Image, color_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Сначала применяем базовое изменение размера
        grayscale_img, color_img = resize_transform(grayscale_img, color_img)
        
        # Преобразуем PIL Images в numpy arrays
        grayscale_np = np.array(grayscale_img)
        color_np = np.array(color_img)
        
        # Применяем albumentations трансформации
        transformed = albumentations_compose(image=grayscale_np, color=color_np)
        
        # Преобразуем обратно в PIL Images
        grayscale_transformed = Image.fromarray(transformed['image'])
        color_transformed = Image.fromarray(transformed['color'])
        
        return grayscale_transformed, color_transformed
    
    # Создаем дополнительные трансформации для каждого типа изображений
    grayscale_transforms = []
    color_transforms = []
    
    # Для черно-белых изображений
    if augmentation_level in ['medium', 'heavy']:
        grayscale_transforms.append(T.RandomAdjustSharpness(sharpness_factor=2, p=0.3))
        
    if augmentation_level == 'heavy':
        grayscale_transforms.append(T.RandomPosterize(bits=6, p=0.2))
        
    # Для цветных изображений (осторожно с аугментациями, чтобы не исказить цветовую информацию)
    if augmentation_level in ['light', 'medium']:
        if color_space == 'rgb':
            # Для RGB можем применять цветовые трансформации
            color_transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
    
    # Создаем функции для применения дополнительных трансформаций
    grayscale_aug = T.Compose(grayscale_transforms) if grayscale_transforms else None
    color_aug = T.Compose(color_transforms) if color_transforms else None
    
    return paired_albumentation_transform, grayscale_aug, color_aug


def create_train_dataset(
    data_root: str,
    grayscale_dir: Optional[str] = None,
    color_dir: Optional[str] = None,
    color_space: str = 'lab',
    img_size: int = 256,
    augmentation_level: str = 'medium',
    max_dataset_size: Optional[int] = None,
    reference_dataset_path: Optional[str] = None
) -> TrainColorizationDataset:
    """
    Создает датасет для тренировки модели колоризации.
    
    Args:
        data_root (str): Корневая директория с данными
        grayscale_dir (str, optional): Поддиректория с черно-белыми изображениями
        color_dir (str, optional): Поддиректория с цветными изображениями
        color_space (str): Цветовое пространство для преобразования
        img_size (int): Размер выходного изображения
        augmentation_level (str): Уровень аугментации
        max_dataset_size (int, optional): Максимальный размер датасета
        reference_dataset_path (str, optional): Путь к референсному датасету для стилизации
        
    Returns:
        TrainColorizationDataset: Созданный тренировочный датасет
    """
    return TrainColorizationDataset(
        data_root=data_root,
        grayscale_dir=grayscale_dir,
        color_dir=color_dir,
        color_space=color_space,
        img_size=img_size,
        augmentation_level=augmentation_level,
        random_crop=True,
        random_flip=True,
        paired_data=grayscale_dir is not None and color_dir is not None,
        max_dataset_size=max_dataset_size,
        use_cache=False,  # Для тренировочных данных лучше не использовать кэш из-за аугментаций
        sampling_strategy='random',
        dynamic_difficulty=True,
        reference_dataset_path=reference_dataset_path
    )


def create_train_dataloader(
    dataset: TrainColorizationDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    dynamic_batching: bool = False,
    dynamic_batch_strategy: str = 'random',
    num_iterations_per_epoch: Optional[int] = None
) -> DataLoader:
    """
    Создает загрузчик данных для тренировки модели колоризации.
    
    Args:
        dataset (TrainColorizationDataset): Тренировочный датасет
        batch_size (int): Размер батча
        shuffle (bool): Перемешивать ли данные
        num_workers (int): Количество рабочих потоков для загрузки данных
        pin_memory (bool): Использовать ли закрепленную память для ускорения передачи на GPU
        drop_last (bool): Отбрасывать ли последний неполный батч
        distributed (bool): Использовать ли распределенное обучение
        rank (int): Ранг текущего процесса (для распределенного обучения)
        world_size (int): Общее количество процессов (для распределенного обучения)
        dynamic_batching (bool): Использовать ли динамическое батчирование
        dynamic_batch_strategy (str): Стратегия для динамического батчирования
        num_iterations_per_epoch (int, optional): Количество итераций за эпоху
        
    Returns:
        DataLoader: Созданный загрузчик данных
    """
    # Определяем сэмплер в зависимости от конфигурации
    if dynamic_batching:
        # Если используется динамическое батчирование
        iterations = num_iterations_per_epoch or (len(dataset) + batch_size - 1) // batch_size
        sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_iterations=iterations,
            drop_last=drop_last,
            strategy=dynamic_batch_strategy
        )
        # При использовании кастомного сэмплера отключаем shuffle
        shuffle = False
    elif distributed:
        # Если используется распределенное обучение
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # При использовании DistributedSampler отключаем shuffle в DataLoader
        shuffle = False
    else:
        # В остальных случаях используем стандартные сэмплеры
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
    # Создаем загрузчик данных
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size if not dynamic_batching else 1,  # При динамическом батчинге размер батча задается сэмплером
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        batch_sampler=sampler if dynamic_batching else None  # Используем batch_sampler только при динамическом батчинге
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TintoraAI Train Dataset Test")
    parser.add_argument("--data_root", type=str, required=True, help="Путь к директории с данными")
    parser.add_argument("--grayscale_dir", type=str, help="Поддиректория с черно-белыми изображениями")
    parser.add_argument("--color_dir", type=str, help="Поддиректория с цветными изображениями")
    parser.add_argument("--color_space", type=str, default="lab", choices=COLOR_SPACES, help="Цветовое пространство")
    parser.add_argument("--augmentation", type=str, default="medium", choices=["none", "light", "medium", "heavy"], help="Уровень аугментации")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--img_size", type=int, default=256, help="Размер изображения")
    parser.add_argument("--dynamic_batching", action="store_true", help="Использовать динамическое батчирование")
    
    args = parser.parse_args()
    
    try:
        # Создаем тренировочный датасет
        dataset = create_train_dataset(
            data_root=args.data_root,
            grayscale_dir=args.grayscale_dir,
            color_dir=args.color_dir,
            color_space=args.color_space,
            img_size=args.img_size,
            augmentation_level=args.augmentation
        )
        
        print(f"Создан тренировочный датасет с {len(dataset)} изображениями")
        
        # Создаем загрузчик данных
        dataloader = create_train_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            dynamic_batching=args.dynamic_batching
        )
        
        print(f"Создан тренировочный загрузчик данных с {len(dataloader)} батчами")
        
        # Получаем и визуализируем один батч
        batch = next(iter(dataloader))
        print(f"Размеры батча: grayscale={batch['grayscale'].shape}, color={batch['color'].shape}")
        
        # Если есть визуализационные инструменты, можно добавить здесь отображение примеров
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")