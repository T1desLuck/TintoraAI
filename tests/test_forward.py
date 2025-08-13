import torch
import pytest

from models import TintoraAI


def test_model_forward_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TintoraAI().to(device).eval()
    # входной размер кратен 8
    L = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        out = model(L, omm_read_only=True)
    assert isinstance(out, dict)
    assert "a" in out and "b" in out
    a, b = out["a"], out["b"]
    assert a.shape == (2, 1, 64, 64)
    assert b.shape == (2, 1, 64, 64)
