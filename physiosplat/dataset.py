"""Endoscopic video dataset (V = {I_t, M_t, D_t : t in [0, T]}, Sec. 2).

Each frame provides:

* ``I_t`` -- RGB frame (tool-free ground truth for the photometric loss).
* ``M_t`` -- semantic tool mask (1 = tool pixel).
* ``D_t`` -- monocular / stereo-derived depth map (used to initialise the
  canonical Gaussians by back-projection).

The loader supports the common EndoNeRF-style on-disk layout::

    scene/
      images/      0000.png 0001.png ...
      masks/       0000.png ...        (white = tool)
      depth/       0000.npy ...        (or .png)
      cameras.json                     (intrinsics + per-frame extrinsics)

and an in-memory constructor used by the tests / synthetic example.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from physiosplat.cameras import Camera


@dataclass
class Frame:
    image: torch.Tensor       # (H, W, 3) in [0, 1] -- the *observed* frame
    tool_mask: torch.Tensor   # (H, W) in {0, 1}, 1 = tool
    depth: torch.Tensor       # (H, W) metric depth (0 = invalid)
    camera: Camera
    t_index: int
    # Optional tool-free reference used only for *evaluating* inpainting (Eq. 2).
    # Real endoscopic datasets do not provide this (the tool genuinely occludes
    # the anatomy); it is available for synthetic scenes.
    clean_image: Optional[torch.Tensor] = None


class EndoscopicDataset:
    """A sequence of :class:`Frame` objects sharing intrinsics."""

    def __init__(self, frames: list[Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i) -> Frame:
        return self.frames[i]

    def to(self, device) -> "EndoscopicDataset":
        out = []
        for f in self.frames:
            out.append(
                Frame(
                    image=f.image.to(device),
                    tool_mask=f.tool_mask.to(device),
                    depth=f.depth.to(device),
                    camera=f.camera.to(device),
                    t_index=f.t_index,
                    clean_image=None
                    if f.clean_image is None
                    else f.clean_image.to(device),
                )
            )
        return EndoscopicDataset(out)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        images: np.ndarray,
        masks: np.ndarray,
        depths: np.ndarray,
        camera: Camera,
        clean_images: np.ndarray | None = None,
    ) -> "EndoscopicDataset":
        """Build from stacked arrays ``(T, H, W, 3)`` / ``(T, H, W)``."""
        frames = []
        for t in range(len(images)):
            frames.append(
                Frame(
                    image=torch.as_tensor(images[t], dtype=torch.float32),
                    tool_mask=torch.as_tensor(masks[t], dtype=torch.float32),
                    depth=torch.as_tensor(depths[t], dtype=torch.float32),
                    camera=camera,
                    t_index=t,
                    clean_image=None
                    if clean_images is None
                    else torch.as_tensor(clean_images[t], dtype=torch.float32),
                )
            )
        return cls(frames)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_directory(cls, root: str) -> "EndoscopicDataset":
        """Load an EndoNeRF-style scene directory (see module docstring)."""
        import imageio.v2 as imageio

        with open(os.path.join(root, "cameras.json")) as fh:
            meta = json.load(fh)

        intr = meta["intrinsics"]
        extrinsics = meta["frames"]  # list of {file, R(3x3), t(3)}

        img_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")
        depth_dir = os.path.join(root, "depth")
        names = sorted(os.listdir(img_dir))

        frames = []
        for t, name in enumerate(names):
            stem = os.path.splitext(name)[0]
            image = imageio.imread(os.path.join(img_dir, name)).astype(np.float32) / 255.0
            image = image[..., :3]

            mask_path = os.path.join(mask_dir, stem + ".png")
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path).astype(np.float32)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = (mask / max(mask.max(), 1.0) > 0.5).astype(np.float32)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.float32)

            depth_npy = os.path.join(depth_dir, stem + ".npy")
            if os.path.exists(depth_npy):
                depth = np.load(depth_npy).astype(np.float32)
            else:
                depth = imageio.imread(
                    os.path.join(depth_dir, stem + ".png")
                ).astype(np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]

            ext = extrinsics[t]
            cam = Camera(
                R=torch.as_tensor(ext["R"], dtype=torch.float32),
                t=torch.as_tensor(ext["t"], dtype=torch.float32),
                fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                width=image.shape[1], height=image.shape[0],
            )
            frames.append(
                Frame(
                    image=torch.as_tensor(image),
                    tool_mask=torch.as_tensor(mask),
                    depth=torch.as_tensor(depth),
                    camera=cam,
                    t_index=t,
                )
            )
        return cls(frames)


def backproject_pointcloud(frame: Frame, stride: int = 1, max_points: int = 100000):
    """Back-project a frame's depth map into a world-space point cloud + colour.

    Used to initialise the canonical Gaussians.  Tool pixels and invalid depths
    are skipped so initial primitives lie on the tissue surface.

    Returns ``(points (P,3), colors (P,3), is_tool (P,))``.
    """
    cam = frame.camera
    H, W = frame.depth.shape
    device = frame.depth.device

    ys, xs = torch.meshgrid(
        torch.arange(0, H, stride, device=device),
        torch.arange(0, W, stride, device=device),
        indexing="ij",
    )
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)
    z = frame.depth[ys, xs]
    valid = z > 1e-6
    ys, xs, z = ys[valid], xs[valid], z[valid]

    x_cam = (xs.float() - cam.cx) / cam.fx * z
    y_cam = (ys.float() - cam.cy) / cam.fy * z
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)         # (P,3)
    # camera -> world: x_world = R^T (x_cam - t).
    pts_world = (pts_cam - cam.t.to(device)) @ cam.R.to(device)

    colors = frame.image[ys, xs]
    is_tool = frame.tool_mask[ys, xs] > 0.5

    if pts_world.shape[0] > max_points:
        sel = torch.randperm(pts_world.shape[0], device=device)[:max_points]
        pts_world, colors, is_tool = pts_world[sel], colors[sel], is_tool[sel]

    return pts_world, colors, is_tool


def backproject_multiframe(
    dataset: "EndoscopicDataset",
    stride: int = 1,
    max_points: int = 100000,
):
    """Initialise canonical points by back-projecting **all** frames.

    Surgical 4DGS seeds the tissue field from every frame so that anatomy
    occluded by the tool in one view is still represented (it is visible in
    other views as the tool moves).  A Gaussian is labelled *tool* only if it is
    initialised from a tool pixel; tissue points get a tissue label.

    Returns ``(points (P,3), colors (P,3), is_tool (P,))`` aggregated and
    sub-sampled to ``max_points``.
    """
    all_pts, all_col, all_tool = [], [], []
    for frame in dataset.frames:
        p, c, tool = backproject_pointcloud(frame, stride=stride, max_points=max_points)
        all_pts.append(p)
        all_col.append(c)
        all_tool.append(tool)
    points = torch.cat(all_pts, dim=0)
    colors = torch.cat(all_col, dim=0)
    is_tool = torch.cat(all_tool, dim=0)

    if points.shape[0] > max_points:
        sel = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points, colors, is_tool = points[sel], colors[sel], is_tool[sel]
    return points, colors, is_tool
