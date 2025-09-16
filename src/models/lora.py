from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


@dataclass
class LoRACheckpoint:
    """
    Low-Rank Adaptation weights stored as pairs (A, B) per targeted parameter.
    For a base weight W (out, in, ...), we add: W <- W + scale * A @ B (properly reshaped).
    Shapes recorded in meta to safely reshape.
    """
    factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # name -> (A, B)
    meta: Dict[str, Any]

    @staticmethod
    def from_state_dict(state: Dict[str, Any]) -> "LoRACheckpoint":
        f = {}
        raw = state.get("factors", {})
        for k, pair in raw.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                f[k] = (pair[0], pair[1])
        return LoRACheckpoint(factors=f, meta=dict(state.get("meta", {})))

    def to_state(self) -> Dict[str, Any]:
        return {"type": "lora", "factors": self.factors, "meta": self.meta}


def _flat_weight(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    shape = tuple(t.shape)
    if t.ndim == 2:
        return t, shape
    # conv weight: (out, in, kh, kw) -> flatten to (out, in * kh * kw)
    if t.ndim == 4:
        out, in_c, kh, kw = t.shape
        flat = t.reshape(out, in_c * kh * kw)
        return flat, shape
    # fallback keep as 2D by collapsing all but first dim
    out = t.shape[0]
    flat = t.reshape(out, -1)
    return flat, shape


def _unflat_weight(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return t.reshape(*shape)


@torch.no_grad()
def apply_lora_inplace(model: nn.Module, lora: LoRACheckpoint, weight: float = 0.5) -> None:
    sd = model.state_dict()
    for name, (A, B) in lora.factors.items():
        if name not in sd:
            continue
        base = sd[name]
        if not torch.is_floating_point(base):
            continue
        flat, shape = _flat_weight(base)
        # Ensure dtypes/devices match
        A = A.to(flat.device, dtype=flat.dtype)
        B = B.to(flat.device, dtype=flat.dtype)
        delta = A @ B  # (out, r) @ (r, in_flat) -> (out, in_flat)
        flat.add_(delta * float(weight))
        sd[name] = _unflat_weight(flat, shape)
    model.load_state_dict(sd, strict=False)
