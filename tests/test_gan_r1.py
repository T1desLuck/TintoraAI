import torch
from src.losses.gan import GANLoss


def test_gan_bce_with_label_smoothing_and_dummy_r1():
    # GAN loss with BCE + label smoothing should run without error
    gan = GANLoss(loss_type="bce", real_label=0.95, fake_label=0.05)
    pred_real = torch.randn(4, 1)
    pred_fake = torch.randn(4, 1)
    loss_d_real = gan(pred_real, True)
    loss_d_fake = gan(pred_fake, False)
    total = loss_d_real + loss_d_fake
    assert torch.isfinite(total)

    # Dummy R1 penalty (gradient penalty on real images) computes without error
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = x.mean(dim=(1, 2, 3))  # pretend discriminator outputs per-sample scalars
    grad = torch.autograd.grad(outputs=y.sum(), inputs=x, create_graph=True)[0]
    r1_pen = (grad.pow(2).view(x.size(0), -1).sum(dim=1)).mean()
    assert torch.isfinite(r1_pen)
