import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from src.models.adapter import AdapterCheckpoint
from src.losses import L1LabLoss


class AdditiveDelta(nn.Module):
    def __init__(self, base: torch.Tensor):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros_like(base))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.delta


class MockColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal subset mimicking decoder/CRB params
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


def test_adapter_micro_training_and_checkpoint(tmp_path: Path):
    # Build lightweight model on CPU
    device = torch.device("cpu")
    model = MockColorizer().to(device)
    model.train()

    # Register additive deltas on a small subset of parameters (pretend decoder/CRB)
    targets = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            for p_name, _ in m.named_parameters(recurse=False):
                if p_name == "weight":
                    targets.append(f"{n}.weight")
        if len(targets) >= 6:
            break

    # Map full param name -> module
    name_to_module: dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        for p_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{p_name}" if module_name else p_name
            name_to_module[full] = module

    registered: dict[str, AdditiveDelta] = {}
    for full in targets:
        mod = name_to_module.get(full)
        if mod is None:
            continue
        base = getattr(mod, full.split(".")[-1])
        pd = AdditiveDelta(base.data)
        parametrize.register_parametrization(mod, full.split(".")[-1], pd)
        registered[full] = pd

    # One micro step with synthetic data
    B, H, W = 1, 128, 128
    L = torch.randn(B, 1, H, W, device=device)
    ab_gt = torch.randn(B, 2, H, W, device=device)

    loss_fn = L1LabLoss().to(device)
    opt = torch.optim.AdamW((p for pd in registered.values() for p in pd.parameters()), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    out = model(L, gt_ab=ab_gt, omm_read_only=True)
    loss = loss_fn(out["a"], out["b"], ab_gt)
    loss.backward()
    opt.step()

    # Export and save adapter checkpoint
    checkpoints_root = tmp_path / "checkpoints" / "adapters"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    deltas: dict[str, torch.Tensor] = {full: pd.delta.detach().cpu().clone() for full, pd in registered.items()}
    state = AdapterCheckpoint(deltas=deltas, meta={"targets": list(registered.keys())}).to_state()
    out_path = checkpoints_root / "adapter.pth"
    torch.save(state, out_path)

    # Validate saved file can be read back
    loaded = torch.load(out_path, map_location="cpu")
    ck = AdapterCheckpoint.from_state_dict(loaded)
    assert isinstance(ck, AdapterCheckpoint)
    assert ck.deltas, "Adapter checkpoint must contain deltas"
    # Basic sanity: shapes match model params for the subset
    for name, delta in ck.deltas.items():
        assert name in name_to_module, f"Unknown param name in checkpoint: {name}"
        mod = name_to_module[name]
        base = getattr(mod, name.split(".")[-1]).detach().cpu()
        assert tuple(delta.shape) == tuple(base.shape)
