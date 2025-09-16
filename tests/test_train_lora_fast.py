import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from src.models.lora import LoRACheckpoint
from src.losses import L1LabLoss


class LoRAParam(nn.Module):
    def __init__(self, base: torch.Tensor, rank: int = 4):
        super().__init__()
        out = base.shape[0]
        in_flat = int(torch.prod(torch.tensor(base.shape[1:])).item()) if base.ndim > 2 else base.shape[1]
        self.A = nn.Parameter(torch.zeros(out, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_flat))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        orig_shape = W.shape
        if W.ndim == 2:
            flat = W
        else:
            out, *rest = W.shape
            flat = W.reshape(out, -1)
        flat = flat + self.A @ self.B
        return flat.reshape(*orig_shape)


class MockColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.crb = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
        )
        self.head_a = nn.Conv2d(8, 1, 1)
        self.head_b = nn.Conv2d(8, 1, 1)

    def forward(self, L: torch.Tensor, gt_ab: torch.Tensor | None = None, omm_read_only: bool = True):
        x = self.decoder(L)
        x = self.crb(x)
        a = self.head_a(x)
        b = self.head_b(x)
        return {"a": a, "b": b}


def test_lora_micro_training_and_checkpoint(tmp_path: Path):
    device = torch.device("cpu")
    model = MockColorizer().to(device)
    model.train()

    # Select a handful of Linear/Conv2d
    targets = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            targets.append(f"{n}.weight")
        if len(targets) >= 6:
            break

    # Map param name to owning module
    name_to_module: Dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        for p_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{p_name}" if module_name else p_name
            name_to_module[full] = module

    # Register LoRA parametrizations
    registered: Dict[str, LoRAParam] = {}
    for full in targets:
        mod = name_to_module.get(full)
        if mod is None:
            continue
        base = getattr(mod, full.split(".")[-1])
        lp = LoRAParam(base.data, rank=4)
        parametrize.register_parametrization(mod, full.split(".")[-1], lp)
        registered[full] = lp

    # One micro step with synthetic data
    B, H, W = 1, 128, 128
    L = torch.randn(B, 1, H, W, device=device)
    ab_gt = torch.randn(B, 2, H, W, device=device)

    loss_fn = L1LabLoss().to(device)
    opt = torch.optim.AdamW((p for lp in registered.values() for p in lp.parameters()), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    out = model(L, gt_ab=ab_gt, omm_read_only=True)
    loss = loss_fn(out["a"], out["b"], ab_gt)
    loss.backward()
    opt.step()

    # Export and save LoRA checkpoint
    out_dir = tmp_path / "checkpoints" / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for full, lp in registered.items():
        factors[full] = (lp.A.detach().cpu().clone(), lp.B.detach().cpu().clone())
    state = LoRACheckpoint(factors=factors, meta={"rank": 4}).to_state()
    out_path = out_dir / "lora_test.pth"
    torch.save(state, out_path)

    # Validate saved file can be read back
    loaded = torch.load(out_path, map_location="cpu")
    ck = LoRACheckpoint.from_state_dict(loaded)
    assert isinstance(ck, LoRACheckpoint)
    assert ck.factors, "LoRA checkpoint must contain factors"
    for name, (A, B) in ck.factors.items():
        assert name in name_to_module, f"Unknown param name in LoRA checkpoint: {name}"
        mod = name_to_module[name]
        base = getattr(mod, name.split(".")[-1]).detach().cpu()
        out_dim = base.shape[0]
        in_flat = int(torch.prod(torch.tensor(base.shape[1:])).item()) if base.ndim > 2 else base.shape[1]
        assert tuple(A.shape)[0] == out_dim
        assert tuple(B.shape)[1] == in_flat
