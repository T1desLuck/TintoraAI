import sys
from pathlib import Path

import numpy as np
from PIL import Image

import src.train as tr


def _make_imgs(dir_path: Path, n=3, size=(64, 64)):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(dir_path / f"im_{i:02d}.png")


def _write_cfg(tmp: Path, train_dir: Path) -> Path:
    yml = tmp / "micro.yaml"
    yml.write_text(
        """
seed: 123
paths:
  data_root: .
  train_dir: {train_dir}
  checkpoints: {ckpt}
  logs: {logs}
runtime:
  amp: false
  num_workers: 0
  pin_memory: false
training:
  dataset: simple
  image_size: 64
  batch_size: 1
  epochs: 1
  ema:
    enabled: false
  dlb:
    enabled: false
validation:
  enabled: false
checkpointing:
  save_best: false
  save_latest: false
logging:
  tensorboard: false
model:
  use_guidenet: false
        """.strip().format(
            train_dir=str(train_dir).replace("\\", "/"),
            ckpt=str((tmp / "ckpt").as_posix()),
            logs=str((tmp / "logs").as_posix()),
        ),
        encoding="utf-8",
    )
    return yml


def test_training_micro_cpu(tmp_path, monkeypatch):
    # Arrange: tiny dataset
    train_dir = tmp_path / "train"
    _make_imgs(train_dir, n=3, size=(64, 64))
    cfg = _write_cfg(tmp_path, train_dir)

    argv = [
        "python",
        "--config",
        str(cfg),
        "--no-resume",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Act/Assert: should complete without raising
    tr.main()
