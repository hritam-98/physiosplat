"""Tiny synthetic surgical scene for tests and the runnable example.

Generates a short monocular sequence of a deforming "tissue" surface (a smooth
coloured height field) partially occluded by a moving rigid "tool" (a bright
bar that translates across the frame).  For every frame we emit:

* ``I_t``  -- the **tool-free** ground-truth RGB (what PhysioSplat must recover),
* ``M_t``  -- the tool mask (where the tool occludes the view),
* ``D_t``  -- a depth map of the visible surface.

This mirrors the data contract ``V = {I_t, M_t, D_t}`` from Sec. 2 and lets the
full pipeline (PSGD + DBR + SAAM + rasterizer + training) run end-to-end on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from physiosplat.cameras import Camera
from physiosplat.dataset import EndoscopicDataset


def make_synthetic_scene(
    num_frames: int = 6,
    H: int = 48,
    W: int = 48,
    seed: int = 0,
):
    """Build a small synthetic :class:`EndoscopicDataset` plus its camera.

    Returns ``(dataset, camera)``.
    """
    rng = np.random.default_rng(seed)

    fx = fy = 0.9 * W
    cx, cy = W / 2.0, H / 2.0
    # Camera at the origin looking down +z (identity world->cam rotation).
    camera = Camera(
        R=torch.eye(3, dtype=torch.float32),
        t=torch.zeros(3, dtype=torch.float32),
        fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, znear=1e-3,
    )

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = (xs - cx) / fx
    v = (ys - cy) / fy

    # A fixed base depth (tissue ~6 cm away) with a static bump; the tissue then
    # deforms gently over time (breathing-like motion).
    base_depth = 0.06 + 0.004 * np.exp(-((u) ** 2 + (v) ** 2) / 0.05)

    # A smooth, fixed diffuse texture for the tissue.
    tex = np.stack(
        [
            0.55 + 0.25 * np.sin(6 * xs / W + 1.0),
            0.45 + 0.25 * np.cos(5 * ys / H + 0.5),
            0.50 + 0.20 * np.sin(4 * (xs + ys) / (H + W)),
        ],
        axis=-1,
    ).clip(0, 1).astype(np.float32)

    # A distinctive stationary anatomical landmark on the tissue (a darker
    # spot).  Because the tool sweeps across the frame, this landmark is
    # occluded in some frames -- the test of Physio-Inpainting is to recover it.
    spot = np.exp(-(((xs - 0.5 * W) ** 2 + (ys - 0.5 * H) ** 2) / (0.02 * W * H)))
    tissue_tex = (tex - 0.35 * spot[..., None]).clip(0, 1).astype(np.float32)

    images, clean_images, masks, depths = [], [], [], []
    for t in range(num_frames):
        phase = t / max(num_frames - 1, 1)

        # Tissue deformation: a travelling sinusoidal displacement in depth.
        deform = 0.003 * np.sin(2 * np.pi * (xs / W) + 2 * np.pi * phase)
        depth = (base_depth + deform).astype(np.float32)

        # Tool: a vertical *bright, specular* bar translating left->right.
        tool_center = int(0.2 * W + 0.6 * W * phase)
        half_w = max(2, W // 12)
        tool_mask = np.zeros((H, W), dtype=np.float32)
        tool_mask[:, max(0, tool_center - half_w): tool_center + half_w] = 1.0

        # Clean (tool-free) ground truth = the tissue texture only.
        clean = (tissue_tex + rng.normal(0, 0.01, tissue_tex.shape)).clip(0, 1)
        clean = clean.astype(np.float32)

        # Observed frame = tissue with a bright metallic tool composited on top.
        tool_color = np.array([0.92, 0.92, 0.85], dtype=np.float32)  # bright tool
        observed = clean.copy()
        observed[tool_mask > 0.5] = tool_color
        observed = observed.clip(0, 1).astype(np.float32)

        # Depth under the tool reflects the (nearer) tool surface.
        depth_obs = depth.copy()
        depth_obs[tool_mask > 0.5] = 0.045

        images.append(observed)
        clean_images.append(clean)
        masks.append(tool_mask)
        depths.append(depth_obs)

    dataset = EndoscopicDataset.from_arrays(
        np.stack(images),
        np.stack(masks),
        np.stack(depths),
        camera,
        clean_images=np.stack(clean_images),
    )
    return dataset, camera
