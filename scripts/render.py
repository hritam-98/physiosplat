#!/usr/bin/env python3
"""Render frames from a trained PhysioSplat model.

Renders both the full reconstruction and the Physio-Inpainted (tool-removed)
tissue view (Eq. 2), writing PNGs side by side.

Usage:
    python scripts/render.py --ckpt output/physiosplat.pt --data /path/to/scene
    python scripts/render.py --ckpt output/physiosplat.pt --synthetic
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiosplat.config import PhysioSplatConfig
from physiosplat.dataset import EndoscopicDataset
from physiosplat.trainer import build_model_from_dataset


def _save(path, img):
    import imageio.v2 as imageio

    arr = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, arr)


def main():
    ap = argparse.ArgumentParser(description="Render PhysioSplat")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="renders")
    args = ap.parse_args()

    cfg = PhysioSplatConfig()
    if args.synthetic:
        from physiosplat.synthetic import make_synthetic_scene

        dataset, _ = make_synthetic_scene()
        cfg.densify_until_iter = 0
    else:
        dataset = EndoscopicDataset.from_directory(args.data)

    model = build_model_from_dataset(dataset, cfg, device=args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        for f in dataset.to(args.device).frames:
            full = model.render(f.camera, f.t_index)
            inpaint = model.render(f.camera, f.t_index, inpaint=True)
            _save(os.path.join(args.out, f"{f.t_index:04d}_full.png"), full.color)
            _save(os.path.join(args.out, f"{f.t_index:04d}_inpaint.png"), inpaint.color)
    print(f"Wrote renders to {args.out}/")


if __name__ == "__main__":
    main()
