"""
TintoraAI Datasets: Модули для работы с датасетами для обучения моделей колоризации.

Данный пакет предоставляет классы и функции для загрузки, предобработки и
управления датасетами, используемыми для обучения моделей колоризации.
Он включает базовый класс датасета с общей функциональностью, а также
специализированные классы для тренировочных и валидационных данных.

Основные компоненты:
- BaseColorizationDataset: Базовый класс для всех датасетов колоризации
- TrainColorizationDataset: Класс для тренировочных данных с аугментациями
- ValidationColorizationDataset: Класс для валидационных данных

Ключевые возможности:
- Загрузка пар изображений (цветное/черно-белое)
- Преобразование изображений между различными цветовыми пространствами
- Расширенные аугментации данных для улучшения обучения
- Поддержка различных форматов данных и организации директорий
- Стратегии динамической выборки данных и балансировки классов
"""

from .base_dataset import (
    BaseColorizationDataset,
    create_transform,
    create_paired_transform,
    get_color_transforms,
    COLOR_SPACES
)

from .train_dataset import (
    TrainColorizationDataset,
    create_train_dataset,
    create_train_dataloader
)

from .validation_dataset import (
    ValidationColorizationDataset,
    create_validation_dataset,
    create_validation_dataloader
)


# Экспортируемые классы и функции
__all__ = [
    # Из base_dataset.py
    'BaseColorizationDataset',
    'create_transform',
    'create_paired_transform',
    'get_color_transforms',
    'COLOR_SPACES',
    
    # Из train_dataset.py
    'TrainColorizationDataset',
    'create_train_dataset',
    'create_train_dataloader',
    
    # Из validation_dataset.py
    'ValidationColorizationDataset',
    'create_validation_dataset',
    'create_validation_dataloader'
]

# Версия пакета
__version__ = '1.0.0'