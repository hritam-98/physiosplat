"""Tests for the composite objective (Sec. 2.5, Eq. 8)."""

import torch

from physiosplat.losses import mask_bce_loss, psnr, ssim, ssim_loss, total_loss
from physiosplat.rasterizer import RenderOutput


def test_ssim_identical_images_is_one():
    img = torch.rand(16, 16, 3)
    assert torch.allclose(ssim(img, img), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(ssim_loss(img, img), torch.tensor(0.0), atol=1e-4)


def test_psnr_higher_for_closer_prediction():
    target = torch.rand(16, 16, 3)
    close = target + 0.01 * torch.randn_like(target)
    far = target + 0.3 * torch.randn_like(target)
    assert float(psnr(close, target)) > float(psnr(far, target))


def test_mask_bce_low_when_semantic_matches():
    # tool mask: left half tool (1), right half tissue (0).
    tool = torch.zeros(8, 8)
    tool[:, :4] = 1.0
    # rendered tissue prob = 1 - tool (perfect).
    sem = 1.0 - tool
    loss_good = mask_bce_loss(sem.clamp(1e-4, 1 - 1e-4), tool)
    loss_bad = mask_bce_loss(tool.clamp(1e-4, 1 - 1e-4), tool)  # inverted
    assert float(loss_good) < float(loss_bad)


def test_total_loss_assembles_and_backprops():
    H = W = 8
    color = torch.rand(H, W, 3, requires_grad=True)
    sem = torch.rand(H, W, requires_grad=True).clamp(1e-3, 1 - 1e-3)
    render = RenderOutput(color=color, depth=torch.rand(H, W),
                          alpha=torch.rand(H, W), semantic=sem)
    gt = torch.rand(H, W, 3)
    mask = (torch.rand(H, W) > 0.5).float()
    biomech = torch.tensor(0.5, requires_grad=True)

    total, parts = total_loss(render, gt, mask, biomech)
    assert set(parts) == {"L1", "L_SSIM", "L_mask", "L_biomech", "L_total"}
    total.backward()
    assert color.grad is not None and color.grad.abs().sum() > 0
