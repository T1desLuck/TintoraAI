import copy
import pytest

from src.utils.config import load_config, validate_config


def test_default_config_validates():
    cfg = load_config("configs/default.yaml")
    # Should not raise
    validate_config(cfg)


def test_unknown_top_level_key_raises():
    cfg = load_config("configs/default.yaml")
    bad = copy.deepcopy(cfg)
    bad["unknown_section"] = {"x": 1}
    with pytest.raises(ValueError):
        validate_config(bad)


def test_type_mismatch_raises():
    cfg = load_config("configs/default.yaml")
    bad = copy.deepcopy(cfg)
    bad.setdefault("model", {})["c1"] = "ninety-six"
    with pytest.raises(TypeError):
        validate_config(bad)
