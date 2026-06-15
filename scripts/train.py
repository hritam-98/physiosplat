#!/usr/bin/env python3
"""Train PhysioSplat on an endoscopic scene.

Usage:
    python scripts/train.py --data /path/to/scene --iterations 30000 --device cuda
    python scripts/train.py --synthetic            # tiny CPU smoke run

The scene directory must follow the EndoNeRF-style layout described in
``physiosplat.dataset.EndoscopicDataset.from_directory``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiosplat.config import PhysioSplatConfig
from physiosplat.dataset import EndoscopicDataset
from physiosplat.trainer import build_model_from_dataset, train


def main():
    ap = argparse.ArgumentParser(description="Train PhysioSplat")
    ap.add_argument("--data", type=str, default=None, help="scene directory")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic scene")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="output")
    args = ap.parse_args()

    cfg = PhysioSplatConfig()
    if args.iterations is not None:
        cfg.iterations = args.iterations

    if args.synthetic:
        from physiosplat.synthetic import make_synthetic_scene

        dataset, _ = make_synthetic_scene()
        if args.iterations is None:
            cfg.iterations = 300
            cfg.densify_until_iter = 0  # keep the tiny scene fixed-size
    elif args.data:
        dataset = EndoscopicDataset.from_directory(args.data)
    else:
        ap.error("provide --data <scene> or --synthetic")

    model = build_model_from_dataset(dataset, cfg, device=args.device)
    print(f"Initialised {model.gaussians.num_points} Gaussians "
          f"on {len(dataset)} frames ({args.device}).")

    train(model, dataset, cfg, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "physiosplat.pt")
    torch.save({"state_dict": model.state_dict(),
                "num_points": model.gaussians.num_points,
                "num_frames": model.num_frames}, ckpt)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
