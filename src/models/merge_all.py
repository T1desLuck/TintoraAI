from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn

from .adapter import AdapterCheckpoint, apply_adapter_inplace
from .lora import LoRACheckpoint, apply_lora_inplace

# Last provided config (set by sitecustomize hook when load_config is called)
MERGE_CFG: Dict[str, Any] | None = None


def set_config(cfg: Dict[str, Any] | None) -> None:
    global MERGE_CFG
    MERGE_CFG = cfg


def _get_ckpt_root(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    ck = paths.get("checkpoints", "checkpoints")
    return Path(ck)


def _load_adapter_file(p: Path) -> AdapterCheckpoint | None:
    try:
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict):
            if state.get("type") == "adapter" or "deltas" in state:
                return AdapterCheckpoint.from_state_dict(state)
    except Exception:
        return None
    return None


def _load_lora_file(p: Path) -> LoRACheckpoint | None:
    try:
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict):
            if state.get("type") == "lora" or "factors" in state:
                return LoRACheckpoint.from_state_dict(state)
    except Exception:
        return None
    return None


def _weight_for_lora(name: str, cfg: Dict[str, Any]) -> float:
    w = cfg.get("merging", {}).get("weights", {}).get("lora", {})
    if not isinstance(w, dict):
        return 0.5
    if name in w:
        return float(w[name])
    return float(w.get("default", 0.5))


def scan_and_merge(model: nn.Module, cfg: Dict[str, Any]) -> None:
    """
    Scan checkpoints/adapters and checkpoints/lora.
    Merge in order: base (implicit) -> adapter (single or many, combined weight) -> lora (multiple with per-name weights).
    Safe no-op if folders/files are missing.
    """
    root = _get_ckpt_root(cfg)
    # Adapters
    adapters_dir = root / "adapters"
    if adapters_dir.exists():
        weight_adapter = float(cfg.get("merging", {}).get("weights", {}).get("adapter", 0.8))
        adapter_files = sorted(adapters_dir.glob("*.pth"))
        if len(adapter_files) > 1:
            try:
                print(
                    f"[ВНИМАНИЕ] Найдено несколько Adapter чекпоинтов ({len(adapter_files)} шт.) в {adapters_dir}.\n"
                    f"Все они будут суммироваться с одинаковым весом adapter={weight_adapter}.\n"
                    "Рекомендуется держать активным только один adapter.pth, чтобы избежать непреднамеренного усиления."
                )
            except Exception:
                pass
        for p in adapter_files:
            ck = _load_adapter_file(p)
            if ck is not None and weight_adapter > 0:
                apply_adapter_inplace(model, ck, weight_adapter)
    # LoRA
    lora_dir = root / "lora"
    if lora_dir.exists():
        files: List[Path] = sorted(lora_dir.glob("*.pth"))
        if len(files) > 5:
            try:
                print(
                    f"[ИНФО] Обнаружено много LoRA чекпоинтов ({len(files)} шт.) в {lora_dir}.\n"
                    "Если замечаете конфликты между тематиками (перекраски), рассмотрите TIES‑Merging\n"
                    "(Trim‑Elect‑Sum) или снизьте веса отдельных LoRA в конфиге merging.weights.lora.*."
                )
            except Exception:
                pass
        for p in files:
            # infer logical name: lora_<name>_*.pth or <name>.pth
            stem = p.stem
            name = stem
            if stem.startswith("lora_"):
                name = stem[len("lora_") :].split("_")[0]
            weight = _weight_for_lora(name, cfg)
            ck = _load_lora_file(p)
            if ck is not None and weight > 0:
                apply_lora_inplace(model, ck, weight)

