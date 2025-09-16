# Auto-loaded Python hook to enable non-invasive Adapter/LoRA merging at inference time.
# This file DOES NOT alter core project code. It only attaches lightweight hooks.
from __future__ import annotations
from typing import Any, Dict

import types
import torch

# We defer heavy imports to first use to avoid slowing down training.
_last_cfg: Dict[str, Any] | None = None


def _wrap_load_config():
    try:
        import src.utils.config as cfgmod
        from src.models import merge_all
    except Exception:
        return

    if getattr(cfgmod, "_tintorai_cfg_wrapped", False):
        return

    orig_load_config = cfgmod.load_config

    def wrapped_load_config(path: str | None = None, overrides: Dict[str, Any] | None = None):
        cfg = orig_load_config(path, overrides)
        # cache cfg for later merging
        try:
            merge_all.set_config(cfg)
        except Exception:
            pass
        global _last_cfg
        _last_cfg = cfg
        return cfg

    cfgmod.load_config = wrapped_load_config  # type: ignore
    cfgmod._tintorai_cfg_wrapped = True  # type: ignore


def _wrap_module_load_state_dict():
    try:
        from src.models import merge_all
    except Exception:
        return

    if getattr(torch.nn.Module, "_tintorai_ld_wrapped", False):
        return

    orig_load_state_dict = torch.nn.Module.load_state_dict

    def wrapped_load_state_dict(self, state_dict, strict: bool = True):
        out = orig_load_state_dict(self, state_dict, strict)
        # After model weights are loaded (e.g., in inference), attempt a merge if config & checkpoints exist.
        try:
            cfg = merge_all.MERGE_CFG
            if cfg is not None:
                merge_all.scan_and_merge(self, cfg)
        except Exception:
            # Never break user flow on merge issues
            pass
        return out

    torch.nn.Module.load_state_dict = wrapped_load_state_dict  # type: ignore
    torch.nn.Module._tintorai_ld_wrapped = True  # type: ignore


# Activate wrappers
_wrap_load_config()
_wrap_module_load_state_dict()
