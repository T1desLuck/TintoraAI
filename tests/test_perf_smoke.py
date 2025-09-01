import time
import torch
import pytest

from src.models import TintoraAI


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for VRAM smoke test"
)
def test_vram_no_leak_and_throughput_cuda():
    device = torch.device("cuda")
    model = (
        TintoraAI(
            c1=96,
            c2=192,
            c3=384,
            film_dim=256,
            use_guidenet=False,
            guide_feature_dim=None,
            guide_out_dim=None,
            omm_config={},
            use_saturation_head=False,
        )
        .to(device)
        .eval()
    )

    H, W = 256, 256  # small size to keep fast
    x = torch.randn(1, 1, H, W, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    iters = 6
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            out = model(x, omm_read_only=True)
            # use outputs to prevent DCE
            _ = out["a"].shape, out["b"].shape
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak = torch.cuda.max_memory_allocated(device)
    # Run a few more iters and ensure peak doesn't grow significantly (tolerance 5 MB)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(iters):
            out = model(x, omm_read_only=True)
            _ = out["a"].shape
    torch.cuda.synchronize()
    peak2 = torch.cuda.max_memory_allocated(device)

    # Assertions (very lenient):
    assert peak2 <= peak + 5 * 1024 * 1024
    # Throughput sanity: at least ~2 iters/sec on typical GPU for this size
    assert elapsed < 5.0
