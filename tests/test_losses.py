import torch
import pytest

from losses import (
    PhotometricSmoothnessLoss,
    DepthSmoothnessLoss,
    ColorConsistencyPyrLoss,
    EntropyLoss,
    OMMClusterLoss,
)


def test_advanced_losses_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 2, 64, 64
    # L в диапазоне [-1, 1]
    L = torch.rand(B, 1, H, W, device=device) * 2.0 - 1.0
    # a, b — произвольные вещественные значения
    a = torch.randn(B, 1, H, W, device=device)
    b = torch.randn(B, 1, H, W, device=device)
    ab = torch.cat([a, b], dim=1)

    # sat в диапазоне [0, 1]
    sat = torch.sigmoid(torch.randn(B, 1, H, W, device=device))

    photo = PhotometricSmoothnessLoss()
    ds = DepthSmoothnessLoss()
    cc = ColorConsistencyPyrLoss(levels=3)
    ent = EntropyLoss()
    clu = OMMClusterLoss()

    lp = photo(L, a, b)
    assert torch.isfinite(lp).all()

    ld = ds(L, torch.randn(B, 1, H, W, device=device))
    assert torch.isfinite(ld).all()

    lcc = cc(ab, ab)
    assert lcc >= 0

    le = ent(sat)
    assert torch.isfinite(le).all()

    D = 32
    F2 = torch.randn(B, D, H // 8, W // 8, device=device)
    mem_map = torch.randn(B, D, H, W, device=device)
    lc = clu(F2, mem_map)
    assert torch.isfinite(lc).all()
