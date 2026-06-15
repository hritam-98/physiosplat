"""Composite optimisation objective (Sec. 2.5).

    L_total = (1 - lambda_ssim) L1 + lambda_ssim L_SSIM
            + lambda_mask L_mask + lambda_biomech L_biomech              (Eq. 8)

* ``L1`` / ``L_SSIM`` are the standard photometric losses between the rendered
  specular-aware colour ``C(u, t)`` and the (tool-free) ground-truth frame.
* ``L_mask`` is the pixel-wise binary cross-entropy between the ground-truth
  tool masks and the rendered 2D semantic probability map.
* ``L_biomech`` is the DBR energy (Eq. 3), supplied by the model.

Photometric losses are evaluated only over the *valid* (non-tool) region, since
the paper computes metrics "excluding surgical tools denoted by masks".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    """Masked L1 loss.  ``mask`` is a per-pixel weight in ``[0, 1]``."""
    diff = (pred - target).abs()
    if mask is not None:
        m = mask.unsqueeze(-1) if mask.dim() == diff.dim() - 1 else mask
        return (diff * m).sum() / (m.sum() * diff.shape[-1] + 1e-8)
    return diff.mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11):
    """Structural Similarity Index (mean over the image), channels-last input.

    Args:
        pred, target: ``(H, W, 3)`` in ``[0, 1]``.

    Returns:
        scalar SSIM in ``[-1, 1]`` (1 == identical).
    """
    # To (1, C, H, W).
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    win1d = _gaussian_window(window_size, 1.5, x.device, x.dtype)
    win = (win1d[:, None] @ win1d[None, :]).expand(C, 1, window_size, window_size)
    pad = window_size // 2

    def filt(z):
        return F.conv2d(z, win, padding=pad, groups=C)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filt(x * x) - mu_x2
    sigma_y = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def ssim_loss(pred, target, window_size: int = 11):
    """SSIM loss = ``1 - SSIM`` (so lower is better)."""
    return 1.0 - ssim(pred, target, window_size)


def mask_bce_loss(rendered_semantic: torch.Tensor, gt_tool_mask: torch.Tensor):
    """Pixel-wise BCE between rendered tissue prob and GT tool mask.

    The rendered semantic map is the accumulated *tissue* probability, so the
    target is the tissue label ``1 - tool_mask``.

    Args:
        rendered_semantic: ``(H, W)`` accumulated tissue probability in ``[0, 1]``.
        gt_tool_mask: ``(H, W)`` ground-truth tool mask (1 = tool).
    """
    tissue_target = (1.0 - gt_tool_mask).clamp(0.0, 1.0)
    pred = rendered_semantic.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(pred, tissue_target)


def total_loss(
    render,
    gt_color: torch.Tensor,
    gt_tool_mask: torch.Tensor,
    biomech: torch.Tensor,
    *,
    lambda_ssim: float = 0.2,
    lambda_mask: float = 0.1,
    lambda_biomech: float = 0.01,
):
    """Full ``L_total`` (Eq. 8).

    Args:
        render: a :class:`~physiosplat.rasterizer.RenderOutput`.
        gt_color: ``(H, W, 3)`` tool-free ground-truth frame.
        gt_tool_mask: ``(H, W)`` ground-truth tool mask (1 = tool).
        biomech: scalar ``L_biomech`` (Eq. 3).

    Returns:
        ``(total, parts_dict)``.
    """
    # Photometric region = non-tool pixels.
    valid = (1.0 - gt_tool_mask).clamp(0.0, 1.0)

    l1 = l1_loss(render.color, gt_color, mask=valid)
    l_ssim = ssim_loss(render.color, gt_color)
    l_mask = mask_bce_loss(render.semantic, gt_tool_mask)

    photometric = (1.0 - lambda_ssim) * l1 + lambda_ssim * l_ssim
    total = photometric + lambda_mask * l_mask + lambda_biomech * biomech

    parts = {
        "L1": l1.detach(),
        "L_SSIM": l_ssim.detach(),
        "L_mask": l_mask.detach(),
        "L_biomech": biomech.detach(),
        "L_total": total.detach(),
    }
    return total, parts


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    """Peak signal-to-noise ratio (dB), optionally over a valid mask."""
    if mask is not None:
        m = mask.unsqueeze(-1)
        mse = ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1] + 1e-8)
    else:
        mse = ((pred - target) ** 2).mean()
    return -10.0 * torch.log10(mse + 1e-12)
