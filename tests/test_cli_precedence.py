import sys
from pathlib import Path

import numpy as np
from PIL import Image

import src.inference as inf


def _make_temp_image(tmpdir: Path, name: str = "img.png", size=(101, 77)) -> Path:
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8)
    p = tmpdir / name
    Image.fromarray(arr).save(p)
    return p


def _write_cfg(tmpdir: Path) -> Path:
    yml = tmpdir / "cfg.yaml"
    yml.write_text(
        """
paths:
  experiments: experiments
inference:
  pad_divisor: 64
  tta:
    enabled: false
        """.strip()
    )
    return yml


def test_cli_pad_div_precedence(monkeypatch, tmp_path):
    # Arrange temp files
    cfg = _write_cfg(tmp_path)
    img = _make_temp_image(tmp_path)
    out_dir = tmp_path / "out"

    # Stub colorize_single to capture pad_divisor from inference.main
    captured = {}

    def stub_colorize_single(model, L, omm_read_only=True, pad_divisor=32):
        captured["pad_divisor"] = pad_divisor
        # Return a dummy RGB tensor with correct spatial size
        import torch

        H, W = L.shape[-2:]
        return torch.zeros(1, 3, H, W)

    monkeypatch.setattr(inf, "colorize_single", stub_colorize_single)

    # Ensure TTA disabled and tiling disabled (defaults), force CPU
    argv = [
        "python",
        "--config",
        str(cfg),
        "--input",
        str(img),
        "--output",
        str(out_dir),
        "--pad-div",
        "16",
        "--cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Act
    inf.main()

    # Assert: CLI value 16 must override YAML value 64
    assert (
        captured.get("pad_divisor") == 16
    ), f"Expected pad_divisor=16, got {captured.get('pad_divisor')}"
    assert (out_dir / img.name).exists()
