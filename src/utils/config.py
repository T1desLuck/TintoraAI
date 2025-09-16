from pathlib import Path
from typing import Any, Dict, Mapping
import yaml  # type: ignore[import-untyped]


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Простейшая схема допустимых ключей и типов (минимально достаточная для валидации)
_SCHEMA: Dict[str, Any] = {
    # Разрешённые верхнеуровневые ключи
    "project_name": str,
    "seed": int,
    "paths": Mapping,
    "runtime": Mapping,
    "checkpointing": Mapping,
    "optimizer": Mapping,
    "scheduler": Mapping,
    "training": Mapping,
    "loss": Mapping,
    "ssl": Mapping,
    "omm": Mapping,
    "logging": Mapping,
    # Новые секции для модульного слияния весов
    "merging": Mapping,
    "adapters": Mapping,
    # Модель: разрешаем произвольные ключи, но ниже отдельно проверим некоторые типы
    "model": {
        "c1": int,
        "c2": int,
        "c3": int,
        "film_dim": int,
        "use_guidenet": bool,
        "guide_feature_dim": (int, type(None)),
        "guide_out_dim": (int, type(None)),
        "use_saturation_head": bool,
        "omm": Mapping,
    },
    "gan": Mapping,
    "validation": Mapping,
    "inference": {
        "pad_divisor": int,
        "tta": Mapping,
    },
}


def _validate_section(name: str, section: Any, schema: Any) -> None:
    if isinstance(schema, dict):
        if section is None:
            return
        if not isinstance(section, dict):
            raise TypeError(
                f"Секция '{name}' должна быть словарём, получено: {type(section).__name__}"
            )
        # проверка только известных ключей; неизвестные допускаем для гибкости
        for k, subschema in schema.items():
            if k in section:
                _validate_section(f"{name}.{k}", section[k], subschema)
    else:
        # schema указывает ожидаемый тип
        expected = schema
        if section is None:
            return
        if isinstance(expected, tuple):
            if not isinstance(section, expected):
                exp_names = ", ".join(t.__name__ for t in expected)
                raise TypeError(
                    f"'{name}' имеет тип {type(section).__name__}, ожидалось один из: {exp_names}"
                )
        elif expected is Mapping:
            if not isinstance(section, Mapping):
                raise TypeError(
                    f"'{name}' должен быть словарём (Mapping), получено: {type(section).__name__}"
                )
        else:
            if not isinstance(section, expected):
                raise TypeError(
                    f"'{name}' имеет тип {type(section).__name__}, ожидалось: {expected.__name__}"
                )


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Базовая валидация структуры YAML: неизвестные ключи и грубая проверка типов.
    Ключи, отсутствующие в конфиге, игнорируются (используются значения по умолчанию в коде).
    """
    if not isinstance(cfg, dict):
        raise TypeError("Конфиг должен быть словарём")
    # неизвестные ключи верхнего уровня
    for k in cfg.keys():
        if k not in _SCHEMA:
            raise ValueError(f"Неизвестная секция верхнего уровня '{k}' в конфиге")
    # валидируем известные секции
    for name, subschema in _SCHEMA.items():
        if name in cfg:
            _validate_section(name, cfg[name], subschema)
    # Дополнительные точечные проверки типов для часто используемых полей
    model = cfg.get("model", {})
    if isinstance(model, dict):
        for key, t in ("c1", int), ("c2", int), ("c3", int), ("film_dim", int):
            if key in model and not isinstance(model[key], t):
                raise TypeError(f"'model.{key}' должен быть типа {t.__name__}")
        if "use_guidenet" in model and not isinstance(model["use_guidenet"], bool):
            raise TypeError("'model.use_guidenet' должен быть типа bool")
        if "use_saturation_head" in model and not isinstance(
            model["use_saturation_head"], bool
        ):
            raise TypeError("'model.use_saturation_head' должен быть типа bool")
        if (
            "guide_feature_dim" in model
            and model["guide_feature_dim"] is not None
            and not isinstance(model["guide_feature_dim"], int)
        ):
            raise TypeError("'model.guide_feature_dim' должен быть int или null")
        if (
            "guide_out_dim" in model
            and model["guide_out_dim"] is not None
            and not isinstance(model["guide_out_dim"], int)
        ):
            raise TypeError("'model.guide_out_dim' должен быть int или null")
    inf = cfg.get("inference", {})
    if (
        isinstance(inf, dict)
        and "pad_divisor" in inf
        and not isinstance(inf["pad_divisor"], int)
    ):
        raise TypeError("'inference.pad_divisor' должен быть типа int")
