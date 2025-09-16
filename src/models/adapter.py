from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable
import torch
import torch.nn as nn


@dataclass
class AdapterCheckpoint:
    """
    Lightweight container for adapter deltas.
    Stores parameter-wise additive deltas keyed by full parameter names.
    """
    deltas: Dict[str, torch.Tensor]
    meta: Dict[str, Any]

    @staticmethod
    def from_state_dict(state: Dict[str, Any]) -> "AdapterCheckpoint":
        deltas = {k: v for k, v in state.get("deltas", {}).items()}
        meta = dict(state.get("meta", {}))
        return AdapterCheckpoint(deltas=deltas, meta=meta)

    def to_state(self) -> Dict[str, Any]:
        # Tensors kept as-is for torch.save
        return {"type": "adapter", "deltas": self.deltas, "meta": self.meta}


def select_default_adapter_targets(model: nn.Module) -> Iterable[str]:
    """
    Heuristic: target decoder and CRB params only, to avoid disturbing backbone.
    """
    for name, p in model.named_parameters():
        n = name.lower()
        if ("decoder" in n or "crb" in n) and ("weight" in n):
            yield name


def snapshot_base(model: nn.Module, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    base: Dict[str, torch.Tensor] = {}
    sd = model.state_dict()
    for n in names:
        if n in sd:
            base[n] = sd[n].detach().clone()
    return base


def build_zero_deltas(base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in base.items()}


class TrainableAdapter(nn.Module):
    """
    Trainable container of additive deltas ΔW for a subset of model weights.
    During training, you can temporarily apply deltas to the frozen base model.
    """

    def __init__(self, base_weights: Dict[str, torch.Tensor]):
        super().__init__()
        for k, v in base_weights.items():
            self.register_parameter(k.replace(".", "__"), nn.Parameter(torch.zeros_like(v)))

    def named_deltas(self) -> Iterable[tuple[str, nn.Parameter]]:
        for n, p in self.named_parameters():
            yield n.replace("__", "."), p

    def export_checkpoint(self, meta: Dict[str, Any] | None = None) -> AdapterCheckpoint:
        deltas: Dict[str, torch.Tensor] = {n: p.detach().cpu().clone() for n, p in self.named_deltas()}
        return AdapterCheckpoint(deltas=deltas, meta=meta or {})


@torch.no_grad()
def apply_adapter_inplace(model: nn.Module, adapter: AdapterCheckpoint, weight: float = 1.0) -> None:
    """
    In-place merge: W <- W + weight * ΔW for keys present both in model and adapter.
    Safe no-op if keys are missing.
    """
    msd = model.state_dict()
    for k, delta in adapter.deltas.items():
        if k in msd and torch.is_floating_point(msd[k]):
            msd[k].add_(delta.to(msd[k].device, dtype=msd[k].dtype) * float(weight))
    model.load_state_dict(msd, strict=False)


@torch.no_grad()
def revert_to_base(model: nn.Module, base_snapshot: Dict[str, torch.Tensor]) -> None:
    sd = model.state_dict()
    for k, v in base_snapshot.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))
    model.load_state_dict(sd, strict=False)
