import sys
import torch

sys.path.append("c:/TintoraAI")

# 1) Metrics: CIEDE2000
from src.utils.metrics import ciede2000

# 2) DLB
from src.utils.dlb import DynamicLossBalancer

# 3) GANLoss
from src.losses.gan import GANLoss

# 4) Inference paths
from src import inference as I


def test_ciede2000():
    # two simple RGB images (H,W,3) in [0,1]
    a = torch.zeros(1, 3, 8, 8)
    b = torch.ones(1, 3, 8, 8)
    de = ciede2000(a, b)
    print("ciede2000", float(de))


def test_dlb():
    base = {"l1": 1.0, "perc": 0.1, "adv": 0.01}
    hist = {"l1": 0.5, "perc": 0.7, "adv": 0.3}

    ema = DynamicLossBalancer(strategy="ema")
    ema.update(hist)
    w1 = ema.compute_weights(base)
    print("dlb_ema", w1)

    ent = DynamicLossBalancer(strategy="entropy_aware")
    ent.update(hist)
    w2 = ent.compute_weights(base, context={"entropy": 0.8})
    print("dlb_entropy", w2)


def test_ganloss_and_r1():
    # BCE with label smoothing
    gan = GANLoss(loss_type="bce", real_label=0.95, fake_label=0.05)
    pred_real = torch.randn(4, 1)
    pred_fake = torch.randn(4, 1)
    loss_d_real = gan(pred_real, True)
    loss_d_fake = gan(pred_fake, False)
    print("gan_bce_ls", float(loss_d_real + loss_d_fake))

    # Dummy R1 gradient penalty on real images
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    # pretend D(x) = x.mean over spatial (scalar per-sample)
    y = x.mean(dim=(1, 2, 3))
    r1_grad = torch.autograd.grad(outputs=y.sum(), inputs=x, create_graph=True)[0]
    r1_pen = (r1_grad.pow(2).view(x.size(0), -1).sum(dim=1)).mean()
    print("r1_pen_dummy", float(r1_pen))


def test_inference_paths():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def eval(self):
            return self

        def to(self, device):
            return self

        def forward(self, L, omm_read_only=True):
            B, C, H, W = L.shape
            a = torch.zeros((B, 1, H, W), device=L.device)
            b = torch.zeros((B, 1, H, W), device=L.device)
            return {"a": a, "b": b}

    model = DummyModel().to("cpu").eval()
    L = torch.rand(1, 1, 300, 450)

    rgb_single = I.colorize_single(model, L, omm_read_only=True, pad_divisor=32)
    print(
        "single",
        tuple(rgb_single.shape),
        float(rgb_single.min()),
        float(rgb_single.max()),
    )

    rgb_tiled = I.colorize_tiled(
        model, L, tile=128, overlap=32, omm_read_only=True, pad_divisor=32
    )
    print(
        "tiled", tuple(rgb_tiled.shape), float(rgb_tiled.min()), float(rgb_tiled.max())
    )

    cfg = {"enabled": True, "flip": True, "scales": [1.0, 0.75, 1.25]}
    rgb_tta = I.tta_colorize(
        model, L, tile=128, overlap=32, pad_divisor=32, tta_cfg=cfg
    )
    print("tta", tuple(rgb_tta.shape), float(rgb_tta.min()), float(rgb_tta.max()))


if __name__ == "__main__":
    test_ciede2000()
    test_dlb()
    test_ganloss_and_r1()
    test_inference_paths()
