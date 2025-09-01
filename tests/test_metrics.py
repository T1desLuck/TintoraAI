import math
import pytest
import torch

from src.utils.metrics import ssim, try_dists, ciede2000


def test_ssim_identity_and_noise():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = x.clone()
    v_id = ssim(x, y)
    assert isinstance(v_id, torch.Tensor)
    # SSIM of identical images ~ 1
    assert float(v_id) > 0.99

    # With noise, SSIM should drop
    y_noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    v_noisy = ssim(x, y_noisy)
    assert float(v_noisy) < float(v_id)


def test_ssim_dtype_float64_support():
    x = torch.rand(1, 3, 16, 16, dtype=torch.float64)
    y = x.clone()
    v = ssim(x, y)
    assert float(v) > 0.99


def test_ciede2000_identity_and_symmetry():
    # Lab tensors in absolute ranges: L in [0,100], a/b ~ [-128,127]
    lab1 = (
        torch.tensor([[[[50.0]], [[10.0]], [[-20.0]]]]).expand(1, 3, 8, 8).contiguous()
    )
    lab2 = lab1.clone()
    d_same = ciede2000(lab1, lab2)
    assert isinstance(d_same, torch.Tensor)
    assert math.isfinite(float(d_same))
    assert float(d_same) < 1e-6

    # Change only L
    lab3 = lab1.clone()
    lab3[:, 0].add_(10.0)
    d13 = ciede2000(lab1, lab3)
    d31 = ciede2000(lab3, lab1)
    assert float(d13) > 0
    # Symmetry
    assert torch.allclose(d13, d31, atol=1e-6)

    # Change a/b
    lab4 = lab1.clone()
    lab4[:, 1].add_(30.0)
    lab4[:, 2].sub_(15.0)
    d14 = ciede2000(lab1, lab4)
    assert float(d14) > 0


def test_dists_wrapper_identity_and_ordering():
    metric = try_dists()
    # If DISTS is unavailable (no package or weights), skip gracefully
    if not getattr(metric, "enabled", False) or getattr(metric, "metric", None) is None:
        pytest.skip("DISTS not available")

    x = torch.rand(1, 3, 32, 32)
    y = x.clone()
    z = torch.rand(1, 3, 32, 32)

    v_xx = metric(x, y)
    v_xz = metric(x, z)

    assert isinstance(v_xx, torch.Tensor)
    assert isinstance(v_xz, torch.Tensor)

    # DISTS distance: identical images should be ~0 and <= different pairs
    assert float(v_xx) <= 1e-6
    assert float(v_xz) >= float(v_xx)
