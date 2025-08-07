"""
Config Parser: Модуль для работы с конфигурационными файлами.

Данный модуль предоставляет функции и классы для загрузки, парсинга, валидации и сохранения
конфигурационных файлов в формате YAML. Он обеспечивает удобный и гибкий способ настройки
параметров системы колоризации без необходимости изменения исходного кода.

Ключевые особенности:
- Загрузка конфигураций из YAML файлов и их преобразование в удобные объекты Python
- Валидация конфигураций с проверкой обязательных параметров и типов данных
- Слияние нескольких конфигураций с различными приоритетами
- Создание дефолтных конфигураций для всех компонентов системы
- Возможность задания пользовательских схем валидации для конкретных модулей

Преимущества для колоризации:
- Гибкая настройка всех параметров системы без изменения кода
- Возможность быстрого экспериментирования с различными конфигурациями
- Предотвращение ошибок конфигурации благодаря встроенной валидации
- Удобное хранение и восстановление параметров для воспроизведения результатов
"""

import os
import json
import copy
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import yaml
import re
import pydoc
import inspect


class ConfigValidator:
    """
    Валидатор конфигураций с поддержкой проверки типов и обязательных полей.
    
    Args:
        schema (Dict): Схема валидации
    """
    def __init__(self, schema: Dict):
        self.schema = schema
    
    def validate(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Валидирует конфигурацию согласно схеме.
        
        Args:
            config (Dict): Конфигурация для валидации
            
        Returns:
            Tuple[bool, List[str]]: Результат валидации (успех/неуспех, список ошибок)
        """
        errors = []
        self._validate_dict(config, self.schema, "", errors)
        return len(errors) == 0, errors
    
    def _validate_dict(self, config: Dict, schema: Dict, path: str, errors: List[str]):
        """
        Валидирует словарь конфигурации рекурсивно.
        
        Args:
            config (Dict): Конфигурация для валидации
            schema (Dict): Схема валидации для текущего уровня
            path (str): Текущий путь в конфигурации
            errors (List[str]): Список для накопления ошибок
        """
        # Проверяем обязательные поля
        for key, value in schema.items():
            if "required" in value and value["required"] and key not in config:
                errors.append(f"Отсутствует обязательное поле '{path}.{key}'")
                
        # Проверяем все поля в конфигурации
        for key, value in config.items():
            # Формируем текущий путь
            current_path = f"{path}.{key}" if path else key
            
            # Если поле не определено в схеме, пропускаем его
            if key not in schema:
                if "strict" in schema.get("__meta__", {}) and schema["__meta__"]["strict"]:
                    errors.append(f"Неизвестное поле '{current_path}'")
                continue
                
            # Получаем схему для текущего поля
            field_schema = schema[key]
            
            # Проверяем тип
            if "type" in field_schema:
                expected_type = field_schema["type"]
                if expected_type == "dict":
                    if not isinstance(value, dict):
                        errors.append(f"Поле '{current_path}' должно быть словарем")
                    elif "schema" in field_schema:
                        # Рекурсивно проверяем вложенный словарь
                        self._validate_dict(value, field_schema["schema"], current_path, errors)
                elif expected_type == "list":
                    if not isinstance(value, list):
                        errors.append(f"Поле '{current_path}' должно быть списком")
                    elif "items" in field_schema:
                        # Проверяем элементы списка
                        self._validate_list_items(value, field_schema["items"], current_path, errors)
                elif expected_type == "int":
                    if not isinstance(value, int):
                        errors.append(f"Поле '{current_path}' должно быть целым числом")
                elif expected_type == "float":
                    if not isinstance(value, (int, float)):
                        errors.append(f"Поле '{current_path}' должно быть числом")
                elif expected_type == "str":
                    if not isinstance(value, str):
                        errors.append(f"Поле '{current_path}' должно быть строкой")
                elif expected_type == "bool":
                    if not isinstance(value, bool):
                        errors.append(f"Поле '{current_path}' должно быть логическим значением")
                        
            # Проверяем ограничения
            if "min" in field_schema and value < field_schema["min"]:
                errors.append(f"Поле '{current_path}' должно быть не меньше {field_schema['min']}")
                
            if "max" in field_schema and value > field_schema["max"]:
                errors.append(f"Поле '{current_path}' должно быть не больше {field_schema['max']}")
                
            if "allowed_values" in field_schema and value not in field_schema["allowed_values"]:
                errors.append(f"Поле '{current_path}' должно быть одним из {field_schema['allowed_values']}")
                
            # Проверяем пользовательский валидатор
            if "validator" in field_schema and callable(field_schema["validator"]):
                try:
                    if not field_schema["validator"](value):
                        errors.append(f"Поле '{current_path}' не прошло пользовательскую валидацию")
                except Exception as e:
                    errors.append(f"Ошибка при валидации поля '{current_path}': {str(e)}")
    
    def _validate_list_items(self, items: List, schema: Dict, path: str, errors: List[str]):
        """
        Валидирует элементы списка.
        
        Args:
            items (List): Список для валидации
            schema (Dict): Схема валидации для элементов списка
            path (str): Текущий путь в конфигурации
            errors (List[str]): Список для накопления ошибок
        """
        for i, item in enumerate(items):
            current_path = f"{path}[{i}]"
            
            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "dict" and not isinstance(item, dict):
                    errors.append(f"Элемент '{current_path}' должен быть словарем")
                elif expected_type == "dict" and "schema" in schema:
                    # Рекурсивно проверяем вложенный словарь
                    self._validate_dict(item, schema["schema"], current_path, errors)
                elif expected_type == "list" and not isinstance(item, list):
                    errors.append(f"Элемент '{current_path}' должен быть списком")
                elif expected_type == "int" and not isinstance(item, int):
                    errors.append(f"Элемент '{current_path}' должен быть целым числом")
                elif expected_type == "float" and not isinstance(item, (int, float)):
                    errors.append(f"Элемент '{current_path}' должен быть числом")
                elif expected_type == "str" and not isinstance(item, str):
                    errors.append(f"Элемент '{current_path}' должен быть строкой")
                elif expected_type == "bool" and not isinstance(item, bool):
                    errors.append(f"Элемент '{current_path}' должен быть логическим значением")


class ConfigParser:
    """
    Базовый класс для парсеров конфигураций.
    """
    def parse(self, file_path: str) -> Dict:
        """
        Парсит конфигурационный файл.
        
        Args:
            file_path (str): Путь к конфигурационному файлу
            
        Returns:
            Dict: Распарсенная конфигурация
        """
        raise NotImplementedError("Метод parse должен быть реализован в подклассах")
    
    def to_string(self, config: Dict) -> str:
        """
        Преобразует конфигурацию в строку для сохранения.
        
        Args:
            config (Dict): Конфигурация для преобразования
            
        Returns:
            str: Строковое представление конфигурации
        """
        raise NotImplementedError("Метод to_string должен быть реализован в подклассах")
    
    def save(self, config: Dict, file_path: str):
        """
        Сохраняет конфигурацию в файл.
        
        Args:
            config (Dict): Конфигурация для сохранения
            file_path (str): Путь для сохранения
        """
        # Создаем директорию, если нужно
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Преобразуем конфигурацию в строку
        config_str = self.to_string(config)
        
        # Сохраняем в файл
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(config_str)


class YAMLConfigParser(ConfigParser):
    """
    Парсер конфигураций в формате YAML.
    """
    def parse(self, file_path: str) -> Dict:
        """
        Парсит YAML-файл конфигурации.
        
        Args:
            file_path (str): Путь к YAML-файлу
            
        Returns:
            Dict: Распарсенная конфигурация
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                return config if config is not None else {}
            except yaml.YAMLError as e:
                raise ValueError(f"Ошибка парсинга YAML: {str(e)}")
    
    def to_string(self, config: Dict) -> str:
        """
        Преобразует конфигурацию в строку YAML.
        
        Args:
            config (Dict): Конфигурация для преобразования
            
        Returns:
            str: Строковое представление конфигурации в формате YAML
        """
        return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)


# Схемы валидации для различных компонентов системы

MODEL_CONFIG_SCHEMA = {
    "swin_unet": {
        "type": "dict",
        "required": True,
        "schema": {
            "img_size": {"type": "int", "required": True, "min": 32},
            "patch_size": {"type": "int", "required": True, "min": 1},
            "in_channels": {"type": "int", "required": True, "min": 1},
            "out_channels": {"type": "int", "required": True, "min": 1},
            "embed_dim": {"type": "int", "required": True, "min": 1},
            "depths": {"type": "list", "required": True},
            "num_heads": {"type": "list", "required": True},
            "window_size": {"type": "int", "required": True, "min": 1},
            "mlp_ratio": {"type": "float", "required": True, "min": 0.1},
            "dropout_rate": {"type": "float", "required": False, "min": 0.0, "max": 1.0},
            "attention_dropout_rate": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
        }
    },
    "vit_semantic": {
        "type": "dict",
        "required": True,
        "schema": {
            "img_size": {"type": "int", "required": True, "min": 32},
            "patch_size": {"type": "int", "required": True, "min": 1},
            "in_channels": {"type": "int", "required": True, "min": 1},
            "embed_dim": {"type": "int", "required": True, "min": 1},
            "depth": {"type": "int", "required": True, "min": 1},
            "num_heads": {"type": "int", "required": True, "min": 1},
            "mlp_ratio": {"type": "float", "required": True, "min": 0.1},
            "dropout_rate": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
        }
    },
    "fpn_pyramid": {
        "type": "dict",
        "required": True,
        "schema": {
            "in_channels_list": {"type": "list", "required": True},
            "out_channels": {"type": "int", "required": True, "min": 1},
            "use_pyramid_pooling": {"type": "bool", "required": False}
        }
    },
    "cross_attention": {
        "type": "dict",
        "required": True,
        "schema": {
            "swin_dim": {"type": "int", "required": True, "min": 1},
            "vit_dim": {"type": "int", "required": True, "min": 1},
            "num_heads": {"type": "int", "required": True, "min": 1},
            "dropout_rate": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
        }
    },
    "feature_fusion": {
        "type": "dict",
        "required": True,
        "schema": {
            "in_channels_list": {"type": "list", "required": True},
            "out_channels": {"type": "int", "required": True, "min": 1},
            "num_heads": {"type": "int", "required": True, "min": 1}
        }
    }
}

LOSS_CONFIG_SCHEMA = {
    "patch_nce": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "weight": {"type": "float", "required": True, "min": 0.0},
            "temperature": {"type": "float", "required": False, "min": 0.01},
            "num_patches": {"type": "int", "required": False, "min": 1},
            "patch_size": {"type": "int", "required": False, "min": 1}
        }
    },
    "vgg_perceptual": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "weight": {"type": "float", "required": True, "min": 0.0},
            "layers": {"type": "list", "required": False},
            "consistency_weight": {"type": "float", "required": False, "min": 0.0}
        }
    },
    "gan_loss": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "weight": {"type": "float", "required": True, "min": 0.0},
            "type": {"type": "str", "required": False, "allowed_values": ["vanilla", "lsgan", "wgan", "hinge"]},
            "reward_weight": {"type": "float", "required": False, "min": 0.0}
        }
    },
    "dynamic_balancing": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "interval": {"type": "int", "required": False, "min": 1},
            "strategy": {"type": "str", "required": False, "allowed_values": ["fixed", "adaptive", "uncertainty"]}
        }
    },
    "l1_loss": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "weight": {"type": "float", "required": True, "min": 0.0}
        }
    }
}

TRAINING_CONFIG_SCHEMA = {
    "optimizer": {
        "type": "dict",
        "required": True,
        "schema": {
            "type": {"type": "str", "required": True, "allowed_values": ["adam", "sgd", "adamw"]},
            "lr": {"type": "float", "required": True, "min": 0.0},
            "betas": {"type": "list", "required": False},
            "weight_decay": {"type": "float", "required": False, "min": 0.0}
        }
    },
    "scheduler": {
        "type": "dict",
        "required": True,
        "schema": {
            "type": {"type": "str", "required": True, 
                    "allowed_values": ["step", "cosine", "plateau", "warmup_cosine", "none"]},
            "step_size": {"type": "int", "required": False, "min": 1},
            "gamma": {"type": "float", "required": False, "min": 0.0, "max": 1.0},
            "patience": {"type": "int", "required": False, "min": 1},
            "warmup_epochs": {"type": "int", "required": False, "min": 0}
        }
    },
    "batch_size": {"type": "int", "required": True, "min": 1},
    "epochs": {"type": "int", "required": True, "min": 1},
    "num_workers": {"type": "int", "required": True, "min": 0},
    "device": {"type": "str", "required": True},
    "grad_clip": {"type": "float", "required": False, "min": 0.0},
    "checkpoint_interval": {"type": "int", "required": True, "min": 1},
    "validation_interval": {"type": "int", "required": True, "min": 1},
    "mixed_precision": {"type": "bool", "required": False},
    "resume_from": {"type": "str", "required": False}
}

INFERENCE_CONFIG_SCHEMA = {
    "checkpoint_path": {"type": "str", "required": True},
    "device": {"type": "str", "required": True},
    "batch_size": {"type": "int", "required": True, "min": 1},
    "num_workers": {"type": "int", "required": True, "min": 0},
    "color_space": {"type": "str", "required": True, "allowed_values": ["lab", "rgb", "yuv"]},
    "input_size": {"type": "int", "required": True, "min": 32},
    "output_dir": {"type": "str", "required": True},
    "save_comparisons": {"type": "bool", "required": False},
    "save_uncertainty_maps": {"type": "bool", "required": False},
    "fallback_strategy": {"type": "str", "required": False, 
                         "allowed_values": ["memory_bank", "guide_net", "default", "none"]},
    "style_transfer": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "reference_path": {"type": "str", "required": False},
            "alpha": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
        }
    }
}

DATA_CONFIG_SCHEMA = {
    "train_dir": {"type": "str", "required": False},
    "val_dir": {"type": "str", "required": False},
    "test_dir": {"type": "str", "required": False},
    "train_grayscale_dir": {"type": "str", "required": False},
    "train_color_dir": {"type": "str", "required": False},
    "val_grayscale_dir": {"type": "str", "required": False},
    "val_color_dir": {"type": "str", "required": False},
    "test_grayscale_dir": {"type": "str", "required": False},
    "test_color_dir": {"type": "str", "required": False},
    "paired": {"type": "bool", "required": True},
    "color_space": {"type": "str", "required": True, "allowed_values": ["lab", "rgb", "yuv"]},
    "image_size": {"type": "int", "required": True, "min": 32},
    "split_ab": {"type": "bool", "required": False},
    "augment_train": {"type": "bool", "required": False},
    "cache_size": {"type": "int", "required": False, "min": 0}
}

MODULE_CONFIG_SCHEMA = {
    "enabled_modules": {"type": "list", "required": False},
    "guide_net": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "feature_dim": {"type": "int", "required": False, "min": 1},
            "num_layers": {"type": "int", "required": False, "min": 1},
            "advice_channels": {"type": "int", "required": False, "min": 1},
            "use_attention": {"type": "bool", "required": False}
        }
    },
    "discriminator": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "ndf": {"type": "int", "required": False, "min": 1},
            "n_layers": {"type": "int", "required": False, "min": 1},
            "use_spectral_norm": {"type": "bool", "required": False},
            "reward_type": {"type": "str", "required": False, 
                           "allowed_values": ["gan", "wgan", "hinge", "adaptive"]}
        }
    },
    "style_transfer": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "content_weight": {"type": "float", "required": False, "min": 0.0},
            "style_weight": {"type": "float", "required": False, "min": 0.0},
            "content_layers": {"type": "list", "required": False},
            "style_layers": {"type": "list", "required": False}
        }
    },
    "memory_bank": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "feature_dim": {"type": "int", "required": False, "min": 1},
            "max_items": {"type": "int", "required": False, "min": 1},
            "index_type": {"type": "str", "required": False, "allowed_values": ["flat", "ivf", "hnsw"]},
            "use_fusion": {"type": "bool", "required": False}
        }
    },
    "uncertainty_estimation": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "method": {"type": "str", "required": False, 
                      "allowed_values": ["mc_dropout", "probabilistic", "guided"]},
            "num_samples": {"type": "int", "required": False, "min": 1},
            "dropout_rate": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
        }
    },
    "few_shot_adapter": {
        "type": "dict",
        "required": False,
        "schema": {
            "enabled": {"type": "bool", "required": True},
            "adapter_type": {"type": "str", "required": False, "allowed_values": ["standard", "meta"]},
            "bottleneck_dim": {"type": "int", "required": False, "min": 1}
        }
    }
}

MAIN_CONFIG_SCHEMA = {
    "name": {"type": "str", "required": True},
    "version": {"type": "str", "required": True},
    "description": {"type": "str", "required": False},
    "seed": {"type": "int", "required": False},
    "experiment_dir": {"type": "str", "required": True},
    "data": {"type": "dict", "required": True, "schema": DATA_CONFIG_SCHEMA},
    "model": {"type": "dict", "required": True, "schema": MODEL_CONFIG_SCHEMA},
    "modules": {"type": "dict", "required": True, "schema": MODULE_CONFIG_SCHEMA},
    "training": {"type": "dict", "required": True, "schema": TRAINING_CONFIG_SCHEMA},
    "losses": {"type": "dict", "required": True, "schema": LOSS_CONFIG_SCHEMA},
    "inference": {"type": "dict", "required": True, "schema": INFERENCE_CONFIG_SCHEMA},
    "logging": {
        "type": "dict",
        "required": False,
        "schema": {
            "log_dir": {"type": "str", "required": True},
            "log_interval": {"type": "int", "required": True, "min": 1},
            "metrics": {"type": "list", "required": False},
            "save_images": {"type": "bool", "required": False}
        }
    }
}


def parse_config(file_path: str, schema: Optional[Dict] = None) -> Dict:
    """
    Загружает и валидирует конфигурацию из файла.
    
    Args:
        file_path (str): Путь к файлу конфигурации
        schema (Dict, optional): Схема для валидации
        
    Returns:
        Dict: Загруженная конфигурация
    """
    # Определяем тип файла по расширению
    _, ext = os.path.splitext(file_path)
    
    # Выбираем парсер
    if ext.lower() in ['.yml', '.yaml']:
        parser = YAMLConfigParser()
    else:
        raise ValueError(f"Неподдерживаемый формат файла конфигурации: {ext}")
        
    # Парсим конфигурацию
    config = parser.parse(file_path)
    
    # Валидируем, если указана схема
    if schema is not None:
        validator = ConfigValidator(schema)
        is_valid, errors = validator.validate(config)
        
        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Конфигурация не прошла валидацию:\n{error_msg}")
            
    return config


def load_config(config_path: str, schema_type: Optional[str] = None) -> Dict:
    """
    Загружает конфигурацию с учетом типа схемы.
    
    Args:
        config_path (str): Путь к файлу конфигурации
        schema_type (str, optional): Тип схемы ('main', 'model', 'training', 'loss', 'inference')
        
    Returns:
        Dict: Загруженная конфигурация
    """
    # Определяем схему в зависимости от типа
    schema = None
    
    if schema_type == 'main':
        schema = MAIN_CONFIG_SCHEMA
    elif schema_type == 'model':
        schema = MODEL_CONFIG_SCHEMA
    elif schema_type == 'training':
        schema = TRAINING_CONFIG_SCHEMA
    elif schema_type == 'loss':
        schema = LOSS_CONFIG_SCHEMA
    elif schema_type == 'inference':
        schema = INFERENCE_CONFIG_SCHEMA
    elif schema_type == 'data':
        schema = DATA_CONFIG_SCHEMA
    elif schema_type == 'module':
        schema = MODULE_CONFIG_SCHEMA
        
    # Загружаем и валидируем конфигурацию
    return parse_config(config_path, schema)


def save_config(config: Dict, file_path: str):
    """
    Сохраняет конфигурацию в файл.
    
    Args:
        config (Dict): Конфигурация для сохранения
        file_path (str): Путь для сохранения
    """
    # Определяем тип файла по расширению
    _, ext = os.path.splitext(file_path)
    
    # Выбираем парсер
    if ext.lower() in ['.yml', '.yaml']:
        parser = YAMLConfigParser()
    else:
        raise ValueError(f"Неподдерживаемый формат файла конфигурации: {ext}")
        
    # Сохраняем конфигурацию
    parser.save(config, file_path)


def create_default_config(config_type: str) -> Dict:
    """
    Создает дефолтную конфигурацию указанного типа.
    
    Args:
        config_type (str): Тип конфигурации ('main', 'model', 'training', 'loss', 'inference', 'data', 'module')
        
    Returns:
        Dict: Дефолтная конфигурация
    """
    if config_type == 'main':
        return {
            "name": "TintoraAI",
            "version": "1.0.0",
            "description": "Современная система колоризации изображений",
            "seed": 42,
            "experiment_dir": "./experiments",
            "data": create_default_config('data'),
            "model": create_default_config('model'),
            "modules": create_default_config('module'),
            "training": create_default_config('training'),
            "losses": create_default_config('loss'),
            "inference": create_default_config('inference'),
            "logging": {
                "log_dir": "./experiments/logs",
                "log_interval": 100,
                "metrics": ["psnr", "ssim", "lpips", "colorfulness"],
                "save_images": True
            }
        }
    elif config_type == 'model':
        return {
            "swin_unet": {
                "img_size": 256,
                "patch_size": 4,
                "in_channels": 1,
                "out_channels": 2,
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 8,
                "mlp_ratio": 4.0,
                "dropout_rate": 0.0,
                "attention_dropout_rate": 0.0
            },
            "vit_semantic": {
                "img_size": 256,
                "patch_size": 16,
                "in_channels": 1,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "dropout_rate": 0.0
            },
            "fpn_pyramid": {
                "in_channels_list": [96, 192, 384, 768],
                "out_channels": 256,
                "use_pyramid_pooling": True
            },
            "cross_attention": {
                "swin_dim": 256,
                "vit_dim": 768,
                "num_heads": 8,
                "dropout_rate": 0.0
            },
            "feature_fusion": {
                "in_channels_list": [256, 768],
                "out_channels": 512,
                "num_heads": 8
            }
        }
    elif config_type == 'training':
        return {
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 1e-4
            },
            "scheduler": {
                "type": "warmup_cosine",
                "warmup_epochs": 5
            },
            "batch_size": 16,
            "epochs": 100,
            "num_workers": 4,
            "device": "cuda",
            "grad_clip": 1.0,
            "checkpoint_interval": 5,
            "validation_interval": 1,
            "mixed_precision": True
        }
    elif config_type == 'loss':
        return {
            "patch_nce": {
                "enabled": True,
                "weight": 1.0,
                "temperature": 0.07,
                "num_patches": 256,
                "patch_size": 16
            },
            "vgg_perceptual": {
                "enabled": True,
                "weight": 1.0,
                "layers": ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
                "consistency_weight": 0.5
            },
            "gan_loss": {
                "enabled": True,
                "weight": 0.5,
                "type": "hinge",
                "reward_weight": 0.1
            },
            "dynamic_balancing": {
                "enabled": True,
                "interval": 100,
                "strategy": "adaptive"
            },
            "l1_loss": {
                "enabled": True,
                "weight": 10.0
            }
        }
    elif config_type == 'inference':
        return {
            "checkpoint_path": "./experiments/checkpoints/latest.pth",
            "device": "cuda",
            "batch_size": 4,
            "num_workers": 2,
            "color_space": "lab",
            "input_size": 256,
            "output_dir": "./output",
            "save_comparisons": True,
            "save_uncertainty_maps": True,
            "fallback_strategy": "memory_bank",
            "style_transfer": {
                "enabled": False,
                "alpha": 0.5
            }
        }
    elif config_type == 'data':
        return {
            "paired": True,
            "train_grayscale_dir": "./data/train/grayscale",
            "train_color_dir": "./data/train/color",
            "val_grayscale_dir": "./data/val/grayscale",
            "val_color_dir": "./data/val/color",
            "test_grayscale_dir": "./data/test/grayscale",
            "test_color_dir": "./data/test/color",
            "color_space": "lab",
            "image_size": 256,
            "split_ab": True,
            "augment_train": True,
            "cache_size": 1000
        }
    elif config_type == 'module':
        return {
            "enabled_modules": [
                "guide_net",
                "discriminator",
                "style_transfer",
                "memory_bank",
                "uncertainty_estimation",
                "few_shot_adapter"
            ],
            "guide_net": {
                "enabled": True,
                "feature_dim": 512,
                "num_layers": 4,
                "advice_channels": 2,
                "use_attention": True
            },
            "discriminator": {
                "enabled": True,
                "ndf": 64,
                "n_layers": 3,
                "use_spectral_norm": True,
                "reward_type": "adaptive"
            },
            "style_transfer": {
                "enabled": True,
                "content_weight": 1.0,
                "style_weight": 100.0,
                "content_layers": ["conv4_2"],
                "style_layers": ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
            },
            "memory_bank": {
                "enabled": True,
                "feature_dim": 512,
                "max_items": 100000,
                "index_type": "flat",
                "use_fusion": True
            },
            "uncertainty_estimation": {
                "enabled": True,
                "method": "guided",
                "num_samples": 10,
                "dropout_rate": 0.1
            },
            "few_shot_adapter": {
                "enabled": True,
                "adapter_type": "standard",
                "bottleneck_dim": 64
            }
        }
    else:
        raise ValueError(f"Неизвестный тип конфигурации: {config_type}")


def validate_config(config: Dict, schema_type: str) -> Tuple[bool, List[str]]:
    """
    Валидирует конфигурацию по указанному типу схемы.
    
    Args:
        config (Dict): Конфигурация для валидации
        schema_type (str): Тип схемы ('main', 'model', 'training', 'loss', 'inference', 'data', 'module')
        
    Returns:
        Tuple[bool, List[str]]: Результат валидации (успех/неуспех, список ошибок)
    """
    # Определяем схему в зависимости от типа
    schema = None
    
    if schema_type == 'main':
        schema = MAIN_CONFIG_SCHEMA
    elif schema_type == 'model':
        schema = MODEL_CONFIG_SCHEMA
    elif schema_type == 'training':
        schema = TRAINING_CONFIG_SCHEMA
    elif schema_type == 'loss':
        schema = LOSS_CONFIG_SCHEMA
    elif schema_type == 'inference':
        schema = INFERENCE_CONFIG_SCHEMA
    elif schema_type == 'data':
        schema = DATA_CONFIG_SCHEMA
    elif schema_type == 'module':
        schema = MODULE_CONFIG_SCHEMA
    else:
        raise ValueError(f"Неизвестный тип схемы: {schema_type}")
        
    # Валидируем конфигурацию
    validator = ConfigValidator(schema)
    return validator.validate(config)


def merge_configs(*configs, priority: str = 'last') -> Dict:
    """
    Объединяет несколько конфигураций с учетом приоритета.
    
    Args:
        *configs: Конфигурации для объединения
        priority (str): Приоритет при конфликтах ('first' или 'last')
        
    Returns:
        Dict: Объединенная конфигурация
    """
    if not configs:
        return {}
        
    if len(configs) == 1:
        return copy.deepcopy(configs[0])
        
    # Выбираем порядок конфигураций в зависимости от приоритета
    configs_to_merge = list(configs)
    if priority == 'first':
        configs_to_merge = configs_to_merge[::-1]
        
    # Начинаем с пустой конфигурации
    result = {}
    
    # Объединяем конфигурации
    for config in configs_to_merge:
        _merge_dicts(result, config)
        
    return result


def _merge_dicts(target: Dict, source: Dict):
    """
    Рекурсивно объединяет словари.
    
    Args:
        target (Dict): Целевой словарь
        source (Dict): Исходный словарь
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # Рекурсивно объединяем вложенные словари
            _merge_dicts(target[key], value)
        else:
            # Копируем значение
            target[key] = copy.deepcopy(value)


class ConfigManager:
    """
    Менеджер конфигураций для удобного доступа к параметрам.
    
    Args:
        config (Dict): Конфигурация
    """
    def __init__(self, config: Dict):
        self._config = config
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Получает значение по пути в конфигурации.
        
        Args:
            path (str): Путь в формате 'section.subsection.param'
            default (Any): Значение по умолчанию, если путь не найден
            
        Returns:
            Any: Значение параметра
        """
        parts = path.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, path: str, value: Any):
        """
        Устанавливает значение по пути в конфигурации.
        
        Args:
            path (str): Путь в формате 'section.subsection.param'
            value (Any): Значение для установки
        """
        parts = path.split('.')
        config = self._config
        
        # Проходим по пути до последней части
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                # Если текущее значение не словарь, заменяем его словарем
                config[part] = {}
                
            config = config[part]
            
        # Устанавливаем значение
        config[parts[-1]] = value
    
    def to_dict(self) -> Dict:
        """
        Преобразует конфигурацию в словарь.
        
        Returns:
            Dict: Словарь конфигурации
        """
        return copy.deepcopy(self._config)


def create_config_from_args(args_dict: Dict, config_type: str) -> Dict:
    """
    Создает конфигурацию из словаря аргументов командной строки.
    
    Args:
        args_dict (Dict): Словарь аргументов
        config_type (str): Тип конфигурации
        
    Returns:
        Dict: Созданная конфигурация
    """
    # Получаем дефолтную конфигурацию
    config = create_default_config(config_type)
    
    # Обновляем конфигурацию значениями из аргументов
    for arg_name, arg_value in args_dict.items():
        # Пропускаем None-значения
        if arg_value is None:
            continue
            
        # Преобразуем имя аргумента в путь конфигурации
        config_path = arg_name.replace('_', '.')
        
        # Устанавливаем значение
        manager = ConfigManager(config)
        manager.set(config_path, arg_value)
        
        # Обновляем конфигурацию
        config = manager.to_dict()
        
    return config


if __name__ == "__main__":
    # Пример использования модуля конфигурации
    
    # Создаем дефолтную конфигурацию
    default_config = create_default_config('main')
    
    # Сохраняем конфигурацию в файл
    try:
        save_config(default_config, 'example_config.yaml')
        print("Конфигурация успешно сохранена в example_config.yaml")
        
        # Загружаем конфигурацию из файла
        loaded_config = load_config('example_config.yaml', 'main')
        print("Конфигурация успешно загружена из файла")
        
        # Получаем значение из конфигурации
        manager = ConfigManager(loaded_config)
        batch_size = manager.get('training.batch_size')
        print(f"Размер батча: {batch_size}")
        
        # Устанавливаем новое значение
        manager.set('training.batch_size', 32)
        print(f"Новый размер батча: {manager.get('training.batch_size')}")
        
    except Exception as e:
        print(f"Ошибка при работе с конфигурацией: {e}")