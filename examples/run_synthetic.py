#!/usr/bin/env python3
"""End-to-end demo on the tiny synthetic surgical scene (CPU-friendly).

Runs the full PhysioSplat pipeline -- PSGD + DBR + SAAM + the differentiable
rasterizer -- and reports that the reconstruction PSNR improves with training
and that Physio-Inpainting (Eq. 2) removes the tool.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiosplat.config import PhysioSplatConfig
from physiosplat.losses import psnr
from physiosplat.synthetic import make_synthetic_scene
from physiosplat.trainer import build_model_from_dataset, train


def main():
    torch.manual_seed(0)
    device = "cpu"

    dataset, _ = make_synthetic_scene(num_frames=6, H=48, W=48)
    cfg = PhysioSplatConfig()
    cfg.iterations = 300
    cfg.densify_until_iter = 0        # keep the toy scene fixed size
    cfg.log_interval = 50
    # Keep the seed cloud small so the pure-Python demo rasterizer stays fast.
    cfg.init_stride = 2
    cfg.init_max_points = 2000

    model = build_model_from_dataset(dataset, cfg, device=device)
    print(f"Initialised {model.gaussians.num_points} Gaussians, "
          f"{len(dataset)} frames.\n")

    frame0 = dataset.to(device)[0]
    valid = 1.0 - frame0.tool_mask

    with torch.no_grad():
        before = psnr(model.render(frame0.camera, 0).color, frame0.image, valid)
    print(f"PSNR before training: {float(before):.2f} dB")

    train(model, dataset, cfg, device=device)

    with torch.no_grad():
        after = psnr(model.render(frame0.camera, 0).color, frame0.image, valid)
        full = model.render(frame0.camera, 0)
        inpaint = model.render(frame0.camera, 0, inpaint=True)

    print(f"\nPSNR after training:  {float(after):.2f} dB  "
          f"(+{float(after - before):.2f} dB)")

    # Tool-removal check: inside the tool region, the rendered tissue-only
    # (inpainted) view should match the tool-free reference better than the full
    # render -- which there still shows the bright occluding tool.
    tool_region = frame0.tool_mask > 0.5
    clean = frame0.clean_image  # tool-free reference (synthetic only)
    if tool_region.any() and clean is not None:
        gt = clean[tool_region]
        full_err = (full.color[tool_region] - gt).abs().mean()
        inpaint_err = (inpaint.color[tool_region] - gt).abs().mean()
        print(f"\nTool-region L1 vs tool-free GT (full render):    {float(full_err):.4f}")
        print(f"Tool-region L1 vs tool-free GT (inpaint render): {float(inpaint_err):.4f}")
        print("Physio-Inpainting recovers occluded anatomy: "
              f"{'YES' if inpaint_err < full_err else 'no improvement'}")


if __name__ == "__main__":
    main()
