import os
import sys
from pathlib import Path
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.models import TintoraAI


def _ddp_worker(rank: int, world_size: int, init_file: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    # Use file init method to avoid port conflicts/flaky env on CI
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=30),
    )

    # Intentionally different seeds per rank to verify DDP broadcast
    torch.manual_seed(rank + 1)
    model = TintoraAI(
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
    model.train()

    ddp = torch.nn.parallel.DistributedDataParallel(model)  # CPU DDP
    # Use momentum to create optimizer state for parity checks
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01, momentum=0.9)

    x = torch.randn(2, 1, 32, 32)
    out = ddp(x, omm_read_only=True)
    loss = sum(
        v.abs().mean()
        for k, v in out.items()
        if isinstance(v, torch.Tensor) and v.ndim >= 1
    )
    loss.backward()

    # Gradient parity across ranks: compare to all-reduced mean
    with torch.no_grad():
        p0 = next(ddp.parameters())
        g = p0.grad.detach().clone()
        g_mean = g.clone()
        dist.all_reduce(g_mean, op=dist.ReduceOp.SUM)
        g_mean /= world_size
        assert torch.allclose(g, g_mean, atol=0, rtol=0)
    opt.step()

    # Optimizer state parity (momentum buffer) after step
    with torch.no_grad():
        state = opt.state[p0]
        if "momentum_buffer" in state:
            mb = state["momentum_buffer"].detach().clone()
            mb_mean = mb.clone()
            dist.all_reduce(mb_mean, op=dist.ReduceOp.SUM)
            mb_mean /= world_size
            assert torch.allclose(mb, mb_mean, atol=0, rtol=0)

    # Check a reference tensor value equality across ranks
    with torch.no_grad():
        ref = next(ddp.parameters()).clone().detach()
        # All-reduce to ensure identical across ranks by comparing to mean
        t = ref.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        # After DDP sync and same update, local params should match world mean exactly
        assert torch.allclose(ref, t, atol=0, rtol=0)

    dist.barrier()
    dist.destroy_process_group()


def test_ddp_cpu_spawn_state_sync():
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)
    world_size = 2
    with tempfile.TemporaryDirectory() as td:
        init_file = str(Path(td) / "ddp_init")
        ctx = mp.get_context("spawn")
        procs = []
        for r in range(world_size):
            p = ctx.Process(target=_ddp_worker, args=(r, world_size, init_file))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(45)
        for p in procs:
            assert p.exitcode == 0


def test_windows_mixed_path_checkpoint(tmp_path: Path):
    # Mixed separators path
    mixed = tmp_path / Path("runs/exp1").joinpath("checkpoints\\latest.pth")
    mixed.parent.mkdir(parents=True, exist_ok=True)

    model = TintoraAI(
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
    sd = {"model": model.state_dict(), "epoch": 1}
    torch.save(sd, mixed)

    # Normalize and load
    assert mixed.exists()
    loaded = torch.load(str(mixed))
    assert "model" in loaded and loaded["epoch"] == 1
