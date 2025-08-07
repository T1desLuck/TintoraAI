"""
TintoraAI Inference: Модули для колоризации изображений на основе обученных моделей.

Данный пакет предоставляет компоненты для инференса моделей колоризации,
включая предсказание цветов, постобработку результатов и пакетную обработку.
Он позволяет использовать обученные модели для колоризации черно-белых изображений
с применением различных стилей, настроек и оптимизаций.

Основные компоненты:
- Predictor: Основной класс для предсказания цветов изображений
- Postprocessor: Модуль для улучшения результатов колоризации
- BatchProcessor: Компонент для эффективной пакетной обработки изображений
- FallbackStrategy: Стратегии восстановления при проблемах с колоризацией

Ключевые возможности:
- Колоризация изображений различных форматов и размеров
- Применение художественных стилей и цветовых палитр
- Оптимизированная пакетная обработка для больших наборов данных
- Постобработка для улучшения визуального качества результатов
- Система мониторинга и логирования процесса колоризации
"""

from .predictor import (
    ColorizationPredictor, FallbackStrategy,
    load_predictor, get_available_fallback_strategies
)

from .postprocessor import (
    ColorizationPostprocessor, ColorBalanceMode, SaturationMode,
    create_postprocessor, process_directory
)

from .batch_processor import (
    BatchProcessor, QueueProcessor, ProcessingMode,
    create_batch_processor, process_batch_from_config
)


# Экспортируемые классы и функции
__all__ = [
    # Из predictor.py
    'ColorizationPredictor',
    'FallbackStrategy',
    'load_predictor',
    'get_available_fallback_strategies',
    
    # Из postprocessor.py
    'ColorizationPostprocessor',
    'ColorBalanceMode',
    'SaturationMode',
    'create_postprocessor',
    'process_directory',
    
    # Из batch_processor.py
    'BatchProcessor',
    'QueueProcessor',
    'ProcessingMode',
    'create_batch_processor',
    'process_batch_from_config'
]

# Версия пакета
__version__ = '1.0.0'