import torch
from src.models import TintoraAI
from src.inference import colorize_single, colorize_tiled


def test_tiling_equivalence_with_tolerance():
    device = torch.device("cpu")
    model = TintoraAI().to(device).eval()
    # Use a moderately sized tensor to exercise tiling path
    H, W = 192, 208
    L = torch.randn(1, 1, H, W, device=device)

    with torch.no_grad():
        rgb_single = colorize_single(model, L, omm_read_only=True, pad_divisor=32)
        rgb_tiled = colorize_tiled(
            model, L, tile=96, overlap=16, omm_read_only=True, pad_divisor=32
        )

    assert rgb_single.shape == (1, 3, H, W)
    assert rgb_tiled.shape == (1, 3, H, W)
    assert torch.isfinite(rgb_single).all()
    assert torch.isfinite(rgb_tiled).all()

    # Allow some tolerance because the model is nonlinear and tiling blends with Hann windows
    mae = (rgb_single - rgb_tiled).abs().mean().item()
    assert (
        mae < 0.15
    ), f"Mean absolute error too high between single and tiled: {mae:.4f}"
