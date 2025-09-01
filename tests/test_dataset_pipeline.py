import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.simple_dataset import SimpleColorizationDataset


def _make_imgs(dir_path: Path, sizes=((37, 55), (128, 90), (16, 16))):
    from PIL import Image

    dir_path.mkdir(parents=True, exist_ok=True)
    for i, (w, h) in enumerate(sizes):
        arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(dir_path / f"im_{i:02d}.png")


@pytest.mark.parametrize("geom", ["center_crop", "random_crop", "random_resized_crop"])
@pytest.mark.parametrize(
    "filt", ["lanczos", "bicubic", "bilinear", "nearest", "unknown_fallsback"]
)
def test_loading_shapes_and_filters(tmp_path, geom, filt):
    train_dir = tmp_path / "train"
    _make_imgs(train_dir)

    ds = SimpleColorizationDataset(
        str(train_dir), image_size=64, geom_mode=geom, resize_filter=filt
    )
    assert len(ds) == 3

    L, ab, path = ds[0]
    assert isinstance(path, str)
    assert L.shape == (1, 64, 64)
    assert ab.shape == (2, 64, 64)
    assert torch.isfinite(L).all()
    assert torch.isfinite(ab).all()


def test_edge_case_small_and_non_square(tmp_path):
    # Very small and non-square images are included in helper
    train_dir = tmp_path / "train"
    _make_imgs(train_dir, sizes=((9, 9), (5, 17), (300, 40)))
    ds = SimpleColorizationDataset(
        str(train_dir), image_size=32, geom_mode="center_crop", resize_filter="lanczos"
    )
    for i in range(len(ds)):
        L, ab, _ = ds[i]
        assert L.shape == (1, 32, 32)
        assert ab.shape == (2, 32, 32)


def test_deterministic_with_seed(tmp_path):
    # Use random_crop which has randomness
    train_dir = tmp_path / "train"
    _make_imgs(train_dir, sizes=((73, 45),))
    ds = SimpleColorizationDataset(
        str(train_dir), image_size=48, geom_mode="random_crop", resize_filter="bicubic"
    )

    # Set seeds and fetch
    random.seed(123)
    np.random.seed(123)
    L1, ab1, _ = ds[0]

    # Reset seeds and fetch again; should match
    random.seed(123)
    np.random.seed(123)
    L2, ab2, _ = ds[0]

    assert torch.allclose(L1, L2)
    assert torch.allclose(ab1, ab2)
