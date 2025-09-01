import sys
import torch

sys.path.append("c:/TintoraAI")
from src import inference as I


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, L, omm_read_only=True):
        B, C, H, W = L.shape
        a = torch.zeros((B, 1, H, W), device=L.device)
        b = torch.zeros((B, 1, H, W), device=L.device)
        return {"a": a, "b": b}


def main():
    model = DummyModel().to("cpu").eval()
    L = torch.rand(1, 1, 300, 450)

    rgb_single = I.colorize_single(model, L, omm_read_only=True, pad_divisor=32)
    print(
        "single",
        tuple(rgb_single.shape),
        float(rgb_single.min()),
        float(rgb_single.max()),
        flush=True,
    )

    rgb_tiled = I.colorize_tiled(
        model, L, tile=128, overlap=32, omm_read_only=True, pad_divisor=32
    )
    print(
        "tiled",
        tuple(rgb_tiled.shape),
        float(rgb_tiled.min()),
        float(rgb_tiled.max()),
        flush=True,
    )

    cfg = {"enabled": True, "flip": True, "scales": [1.0, 0.75, 1.25]}
    rgb_tta = I.tta_colorize(
        model, L, tile=128, overlap=32, pad_divisor=32, tta_cfg=cfg
    )
    print(
        "tta",
        tuple(rgb_tta.shape),
        float(rgb_tta.min()),
        float(rgb_tta.max()),
        flush=True,
    )


if __name__ == "__main__":
    main()
