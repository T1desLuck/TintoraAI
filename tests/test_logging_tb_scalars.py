import os
import sys
import types
import tempfile
from pathlib import Path

import importlib


def test_tb_scalars_phase_and_omm_flag(monkeypatch):
    # Dummy SummaryWriter to capture scalars
    class DummyWriter:
        last_instance = None
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
            self.scalars = []
            DummyWriter.last_instance = self
        def add_scalar(self, tag, scalar_value, global_step=None):
            self.scalars.append((tag, float(scalar_value), int(global_step)))
        def close(self):
            pass
        def add_images(self, *args, **kwargs):
            # no-op for tests
            pass
        def flush(self):
            pass

    # Monkeypatch SummaryWriter before importing train
    import torch.utils.tensorboard as tb
    monkeypatch.setattr(tb, "SummaryWriter", DummyWriter, raising=True)

    # Create temp config pointing data dirs to existing assets and minimal training
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = tmp / "tb_test.yaml"
        logs_dir = tmp / "logs"
        ckpt_dir = tmp / "ckpts"
        # assets folder from repo
        repo_root = Path(__file__).resolve().parents[1]
        assets_dir = repo_root / "assets"
        assert assets_dir.exists(), "assets/ must exist for this test"

        cfg_text = f"""
seed: 123
paths:
  logs: "{logs_dir.as_posix()}"
  checkpoints: "{ckpt_dir.as_posix()}"
  data_root: "{assets_dir.as_posix()}"
  train_dir: "{assets_dir.as_posix()}"
  val_dir: "{assets_dir.as_posix()}"
logging:
  tensorboard: true
training:
  epochs: 1
  batch_size: 1
  image_size: 128
  geometry:
    train_mode: center_crop
    val_mode: center_crop
  ema:
    enabled: false
validation:
  enabled: false
runtime:
  amp: false
  num_workers: 0
"""
        cfg_path.write_text(cfg_text, encoding="utf-8")

        # Prepare argv and import/run train.main
        argv_backup = sys.argv[:]
        try:
            sys.argv = ["train", "--config", str(cfg_path), "--no-resume"]
            # Import fresh module each time to ensure monkeypatch applies
            if "src.train" in sys.modules:
                importlib.reload(sys.modules["src.train"])
            else:
                import src.train  # noqa: F401
            # Call main
            import src.train as train_mod
            train_mod.main()
        finally:
            sys.argv = argv_backup

        # Validate that our DummyWriter captured expected scalar tags
        w = DummyWriter.last_instance
        assert w is not None, "SummaryWriter should be instantiated"
        tags = {t for (t, _, _) in w.scalars}
        assert "train/phase" in tags, "train/phase scalar should be logged"
        assert "train/omm_read_only" in tags, "train/omm_read_only scalar should be logged"
