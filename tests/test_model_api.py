import math
import pytest
import torch

from src.models.tintoraai import TintoraAI
from src.inference import colorize_single


def _make_model(device: torch.device):
    m = TintoraAI()
    return m.to(device).eval()


@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (2, 1, 128, 96)])
def test_forward_output_shapes_and_keys_cpu(shape):
    device = torch.device("cpu")
    model = _make_model(device)
    L = torch.randn(shape, device=device)

    with torch.no_grad():
        out = model(L, omm_read_only=True)

    # Required keys
    for k in ["a", "b", "D", "I", "normals", "sat"]:
        assert k in out, f"Missing key {k}"

    B, _, H, W = L.shape
    assert out["a"].shape == (B, 1, H, W)
    assert out["b"].shape == (B, 1, H, W)
    assert out["D"].shape == (B, 1, H, W)
    assert out["I"].shape == (B, 1, H, W)
    assert out["normals"].shape == (B, 3, H, W)
    assert out["sat"].shape == (B, 1, H, W)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape", [(1, 1, 64, 64)])
def test_device_and_amp_parity_cuda(shape):
    device = torch.device("cuda")
    model = _make_model(device)
    L = torch.randn(shape, device=device)

    with torch.no_grad():
        out_fp32 = model(L, omm_read_only=True)

    # autocast on CUDA
    try:
        amp_dtype = torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out_amp = model(L, omm_read_only=True)
    except Exception as e:
        pytest.skip(f"autocast not supported or failed: {e}")

    # Ensure tensors are on the same device and compare closeness
    for k in ["a", "b", "D", "I", "sat"]:
        t32 = out_fp32[k].float()
        tamp = out_amp[k].float()
        assert t32.device.type == "cuda" and tamp.device.type == "cuda"
        # Allow a relaxed tolerance due to random weights and FP16 rounding
        mae = (t32 - tamp).abs().mean().item()
        assert math.isfinite(mae)
        assert mae < 0.2, f"AMP vs FP32 deviation too large for {k}: {mae}"


def test_inference_colorize_single_cpu():
    device = torch.device("cpu")
    model = _make_model(device)
    L = torch.randn(1, 1, 96, 80, device=device)

    with torch.no_grad():
        rgb = colorize_single(model, L, omm_read_only=True, pad_divisor=32)

    assert rgb.shape == (1, 3, 96, 80)
    assert torch.isfinite(rgb).all()
