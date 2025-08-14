import torch
import pytest
from src.models import TintoraAI
from src.inference import colorize_single, colorize_tiled


def test_colorize_single_tensor():
    device = torch.device("cpu")
    model = TintoraAI().to(device).eval()
    L = torch.randn(1, 1, 224, 224, device=device)
    with torch.no_grad():
        rgb = colorize_single(model, L, omm_read_only=True)
    assert rgb.shape == (1, 3, 224, 224)
    assert torch.isfinite(rgb).all()
    assert (rgb >= 0.0 - 1e-3).all() and (rgb <= 1.0 + 1e-3).all()


def test_colorize_tiled_tensor():
    device = torch.device("cpu")
    model = TintoraAI().to(device).eval()
    L = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        rgb = colorize_tiled(model, L, tile=128, overlap=16, omm_read_only=True)
    assert rgb.shape == (1, 3, 256, 256)
    assert torch.isfinite(rgb).all()
    assert (rgb >= 0.0 - 1e-3).all() and (rgb <= 1.0 + 1e-3).all()
