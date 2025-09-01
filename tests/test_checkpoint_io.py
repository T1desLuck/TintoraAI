from pathlib import Path
import copy
import torch

from src.models import TintoraAI


def make_model():
    # Small default model is fine; previous tests instantiate it successfully
    return TintoraAI(
        c1=96,
        c2=192,
        c3=384,
        film_dim=256,
        use_guidenet=False,
        guide_feature_dim=None,
        guide_out_dim=None,
        omm_config={},
        use_saturation_head=False,
    ).eval()


def test_checkpoint_latest_policy_and_resume_parity(tmp_path: Path):
    # Config-like naming
    latest_names = {"model": "latest.pth", "ema": "latest_ema.pth"}

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Set a known tensor value in the first parameter
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
            p.add_(1.2345)
            break

    # Save latest model and EMA
    latest_model_path = ckpt_dir / latest_names["model"]
    latest_ema_path = ckpt_dir / latest_names["ema"]

    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": 5,
        "global_step": 123,
        "best_score": 0.9,
    }
    torch.save(state, latest_model_path)
    torch.save({"model": copy.deepcopy(state["model"])}, latest_ema_path)

    # Corrupt model then load back mimicking train.py resume branch
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    loaded = torch.load(latest_model_path, map_location="cpu")
    sd = loaded.get("model", loaded)
    model.load_state_dict(sd, strict=False)
    # optimizer
    opt.load_state_dict(loaded["opt"])

    # Verify first param restored to 1.2345
    first = next(model.parameters()).detach().cpu()
    assert torch.allclose(first, torch.full_like(first, 1.2345), atol=0, rtol=0)
    # Files exist
    assert latest_model_path.exists() and latest_ema_path.exists()


def test_checkpoint_best_policy(tmp_path: Path):
    best_names = {"model": "best_ssim.pth", "ema": "best_ssim_ema.pth"}
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    sd = model.state_dict()
    best_model_path = ckpt_dir / best_names["model"]
    best_ema_path = ckpt_dir / best_names["ema"]

    torch.save({"model": sd, "score": 0.92}, best_model_path)
    torch.save({"model": sd}, best_ema_path)

    assert best_model_path.exists()
    assert best_ema_path.exists()

    loaded = torch.load(best_model_path, map_location="cpu")
    assert "model" in loaded and "score" in loaded
