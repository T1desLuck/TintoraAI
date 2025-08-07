"""
TintoraAI Utils: Набор вспомогательных функций и классов для проекта колоризации изображений.

Данный модуль объединяет различные утилиты, необходимые для работы системы колоризации,
включая загрузку и предобработку данных, визуализацию результатов, вычисление метрик
качества, интерактивное взаимодействие с пользователем и работу с конфигурациями.

Основные компоненты:
- DataLoader: Загрузка и предобработка изображений для обучения и инференса
- Visualization: Визуализация результатов колоризации и промежуточных представлений
- Metrics: Вычисление метрик качества колоризации (PSNR, SSIM, LPIPS и др.)
- UserInteraction: Интерактивное взаимодействие с пользователем через консольные команды
- ConfigParser: Парсинг и валидация конфигурационных файлов YAML
"""

from .data_loader import (
    ColorSpaceConverter, ImageAugmentation, ColorizationDataset, 
    ColorizationDatasetPaired, DatasetFromFolder, PairedDatasetFromFolders,
    ColorizationDataModule, ColorizationPairedDataModule,
    prepare_batch_for_colorization, create_datamodule
)

from .visualization import (
    ColorizationVisualizer, GridVisualizer, BatchVisualizer, 
    FeatureVisualizer, tensor_to_numpy, normalize_image,
    lab_to_rgb, save_image, create_visualizer
)

from .metrics import (
    PSNR, SSIM, LPIPS, Colorfulness, FID, LabColorAccuracy, 
    ColorConsistency, MetricsCalculator, MetricsLogger,
    create_metrics_calculator
)

from .user_interaction import (
    CommandType, CommandResult, Command, SetParameterCommand,
    ActionCommand, QueryCommand, HelpCommand, StyleCommand,
    CommandRegistry, CommandProcessor, InteractiveConsole,
    CommandScript, UserInteractionModule, create_user_interaction_module
)

from .config_parser import (
    ConfigValidator, ConfigParser, YAMLConfigParser,
    parse_config, load_config, save_config, create_default_config,
    validate_config, merge_configs
)


# Экспортируемые модули и функции
__all__ = [
    # Data Loader
    'ColorSpaceConverter', 'ImageAugmentation', 'ColorizationDataset', 
    'ColorizationDatasetPaired', 'DatasetFromFolder', 'PairedDatasetFromFolders',
    'ColorizationDataModule', 'ColorizationPairedDataModule',
    'prepare_batch_for_colorization', 'create_datamodule',
    
    # Visualization
    'ColorizationVisualizer', 'GridVisualizer', 'BatchVisualizer', 
    'FeatureVisualizer', 'tensor_to_numpy', 'normalize_image',
    'lab_to_rgb', 'save_image', 'create_visualizer',
    
    # Metrics
    'PSNR', 'SSIM', 'LPIPS', 'Colorfulness', 'FID', 'LabColorAccuracy', 
    'ColorConsistency', 'MetricsCalculator', 'MetricsLogger',
    'create_metrics_calculator',
    
    # User Interaction
    'CommandType', 'CommandResult', 'Command', 'SetParameterCommand',
    'ActionCommand', 'QueryCommand', 'HelpCommand', 'StyleCommand',
    'CommandRegistry', 'CommandProcessor', 'InteractiveConsole',
    'CommandScript', 'UserInteractionModule', 'create_user_interaction_module',
    
    # Config Parser
    'ConfigValidator', 'ConfigParser', 'YAMLConfigParser',
    'parse_config', 'load_config', 'save_config', 'create_default_config',
    'validate_config', 'merge_configs',
]


# Версия модуля
__version__ = '1.0.0'