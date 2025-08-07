"""
TintoraAI Training: Компоненты для обучения моделей колоризации.

Данный пакет предоставляет компоненты и утилиты для обучения моделей колоризации,
включая настройку процесса обучения, планирование скорости обучения,
валидацию моделей и управление чекпоинтами.

Основные компоненты:
- Trainer: Основной класс для обучения модели колоризации
- Validator: Модуль валидации модели на тестовых данных
- Scheduler: Планировщики скорости обучения для оптимизации процесса обучения
- Checkpoints: Утилиты для сохранения и загрузки состояний модели

Ключевые возможности:
- Гибкое конфигурирование процесса обучения
- Система мотивации "награды и наказания" для улучшения качества колоризации
- Динамическое балансирование потерь для оптимальной сходимости
- Интеграция с различными интеллектуальными модулями
- Визуализация и мониторинг процесса обучения
"""

from .trainer import ColorizationTrainer, create_trainer, create_model_from_config, create_data_loaders_from_config
from .validator import ColorizationValidator, create_validator
from .scheduler import create_scheduler, GradualWarmupScheduler, CosineAnnealingWarmRestarts, AdaptiveLRScheduler, create_lr_lambda, create_lambda_scheduler
from .checkpoints import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint, find_best_checkpoint,
    clean_checkpoints, extract_model_from_checkpoint, convert_checkpoint_to_model,
    checkpoint_exists, combine_checkpoints, CheckpointManager, load_model_from_checkpoint
)


# Экспортируемые классы и функции
__all__ = [
    # Из trainer.py
    'ColorizationTrainer',
    'create_trainer',
    'create_model_from_config',
    'create_data_loaders_from_config',
    
    # Из validator.py
    'ColorizationValidator',
    'create_validator',
    
    # Из scheduler.py
    'create_scheduler',
    'GradualWarmupScheduler',
    'CosineAnnealingWarmRestarts',
    'AdaptiveLRScheduler',
    'create_lr_lambda',
    'create_lambda_scheduler',
    
    # Из checkpoints.py
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'find_best_checkpoint',
    'clean_checkpoints',
    'extract_model_from_checkpoint',
    'convert_checkpoint_to_model',
    'checkpoint_exists',
    'combine_checkpoints',
    'CheckpointManager',
    'load_model_from_checkpoint'
]

# Версия пакета
__version__ = '1.0.0'