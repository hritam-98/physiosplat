"""Differentiable Gaussian rasterizer (EWA splatting + alpha-blending).

This is a pure-PyTorch implementation of the differentiable ``alpha``-blending
of Sec. 2.1:

    C(u) = sum_i c_i a_i prod_{j<i} (1 - a_j),
    a_i  = exp(-1/2 (u - mu_i^2D)^T (Sigma_i^2D)^-1 (u - mu_i^2D))

It projects each 3D Gaussian to a 2D mean ``mu^2D`` and 2D covariance
``Sigma^2D`` (via the projection Jacobian -- the EWA affine approximation),
depth-sorts the primitives, and front-to-back composites colour, depth, and the
semantic probability map.

It is intentionally backend-agnostic and CUDA-free so the framework runs and is
testable on any device.  A production deployment can swap in the tiled CUDA
rasterizer of 3DGS; the physics modules (PSGD / DBR / SAAM) are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physiosplat.cameras import Camera
from physiosplat.transforms import build_covariance


@dataclass
class RenderOutput:
    color: torch.Tensor      # (H, W, 3)
    depth: torch.Tensor      # (H, W) expected depth
    alpha: torch.Tensor      # (H, W) accumulated opacity
    semantic: torch.Tensor   # (H, W) accumulated tissue probability


def _project_covariance_2d(
    cov3d: torch.Tensor, J: torch.Tensor, W: torch.Tensor, blur: float = 0.3
) -> torch.Tensor:
    """Project 3D covariance to 2D: ``Sigma^2D = J W Sigma W^T J^T``.

    ``W`` is the world-to-camera rotation.  A small isotropic ``blur`` is added
    to the diagonal (the standard low-pass / anti-aliasing term) to keep the 2D
    covariance invertible for sub-pixel Gaussians.
    """
    M = J @ W                                 # (N,2,3)
    cov2d = M @ cov3d @ M.transpose(-1, -2)   # (N,2,2)
    eye = torch.eye(2, device=cov2d.device, dtype=cov2d.dtype)
    cov2d = cov2d + blur * eye
    return cov2d


def rasterize(
    camera: Camera,
    xyz: torch.Tensor,
    cov3d: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    semantic_prob: torch.Tensor,
    bg_color: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
) -> RenderOutput:
    """Render a set of (already deformed) Gaussians from ``camera``.

    Args:
        camera: the :class:`~physiosplat.cameras.Camera`.
        xyz: ``(N, 3)`` world positions.
        cov3d: ``(N, 3, 3)`` world covariances.
        colors: ``(N, 3)`` per-Gaussian RGB (diffuse + specular already summed).
        opacity: ``(N, 1)`` or ``(N,)`` opacities.
        semantic_prob: ``(N, 1)`` or ``(N,)`` tissue probabilities (for the
            rendered semantic map used by the mask BCE loss).
        bg_color: optional ``(3,)`` background colour (default black).
        visibility: optional ``(N,)`` multiplicative gate in ``[0, 1]`` used for
            tissue-only inpainting (Eq. 2) -- a Gaussian with visibility 0 is
            removed from compositing.

    Returns:
        :class:`RenderOutput`.
    """
    device = xyz.device
    H, W = camera.height, camera.width
    opacity = opacity.reshape(-1)
    semantic_prob = semantic_prob.reshape(-1)

    if bg_color is None:
        bg_color = torch.zeros(3, device=device, dtype=xyz.dtype)

    uv, depth, J = camera.project(xyz)              # (N,2), (N,), (N,2,3)
    cov2d = _project_covariance_2d(cov3d, J, camera.R)
    # Invert 2D covariance (conic).
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = det.clamp_min(1e-12)
    inv = torch.empty_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1] / det
    inv[:, 1, 1] = cov2d[:, 0, 0] / det
    inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv[:, 1, 0] = -cov2d[:, 1, 0] / det

    # Cull Gaussians behind the camera.
    valid = depth > camera.znear
    if visibility is not None:
        eff_opacity = opacity * visibility.reshape(-1)
    else:
        eff_opacity = opacity

    # Pixel grid (H, W, 2) in (x, y) = (col, row) convention.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=xyz.dtype),
        torch.arange(W, device=device, dtype=xyz.dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys], dim=-1).reshape(-1, 2)   # (P, 2)

    # Depth-sort (front to back).
    order = torch.argsort(depth)
    color_acc = torch.zeros(H * W, 3, device=device, dtype=xyz.dtype)
    depth_acc = torch.zeros(H * W, device=device, dtype=xyz.dtype)
    sem_acc = torch.zeros(H * W, device=device, dtype=xyz.dtype)
    T = torch.ones(H * W, device=device, dtype=xyz.dtype)  # transmittance

    for i in order.tolist():
        if not bool(valid[i]):
            continue
        oi = eff_opacity[i]
        if float(oi.detach()) < 1e-4:
            continue
        d = pix - uv[i]                             # (P,2)
        # Gaussian exponent: -1/2 d^T inv d.
        e = (
            inv[i, 0, 0] * d[:, 0] * d[:, 0]
            + 2 * inv[i, 0, 1] * d[:, 0] * d[:, 1]
            + inv[i, 1, 1] * d[:, 1] * d[:, 1]
        )
        g = torch.exp(-0.5 * e)
        a = (oi * g).clamp(max=0.999)               # per-pixel alpha
        # Skip negligible contributions for speed (keeps gradient where it matters).
        contrib = a * T
        color_acc = color_acc + contrib.unsqueeze(-1) * colors[i]
        depth_acc = depth_acc + contrib * depth[i]
        sem_acc = sem_acc + contrib * semantic_prob[i]
        T = T * (1.0 - a)

    # Composite background onto remaining transmittance.
    color_acc = color_acc + T.unsqueeze(-1) * bg_color
    alpha = 1.0 - T

    # Expected depth = alpha-weighted depth normalised by accumulated opacity, so
    # a partially-covered pixel still reports the true surface depth (not biased
    # towards zero by low coverage).  This is the depth consumed by SAAM's
    # geometry-derived normals (n_i ~ grad D_rend).
    depth_expected = depth_acc / alpha.clamp_min(1e-6)

    return RenderOutput(
        color=color_acc.reshape(H, W, 3),
        depth=depth_expected.reshape(H, W),
        alpha=alpha.reshape(H, W),
        semantic=sem_acc.reshape(H, W),
    )


def normals_from_depth(depth: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Estimate a per-pixel surface normal map ``n ~ grad D_rend`` (Sec. 2.4).

    The paper derives Gaussian normals on-the-fly from the spatial gradient of
    the accumulated rendered depth map.  We back-project the depth map to a
    camera-space point map and take the normalised cross product of its spatial
    finite differences.

    Returns ``(H, W, 3)`` unit normals in *world* coordinates.
    """
    H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    # Back-project pixels to camera-space points using the pinhole model.
    z = depth.clamp_min(1e-6)
    x = (xs - camera.cx) / camera.fx * z
    y = (ys - camera.cy) / camera.fy * z
    pts = torch.stack([x, y, z], dim=-1)            # (H,W,3) camera space

    # Spatial finite differences.
    dpdx = torch.zeros_like(pts)
    dpdy = torch.zeros_like(pts)
    dpdx[:, 1:-1, :] = (pts[:, 2:, :] - pts[:, :-2, :]) * 0.5
    dpdy[1:-1, :, :] = (pts[2:, :, :] - pts[:-2, :, :]) * 0.5

    n_cam = torch.cross(dpdx, dpdy, dim=-1)
    n_cam = n_cam / (n_cam.norm(dim=-1, keepdim=True) + 1e-8)
    # Orient towards the camera (camera looks along +z in camera space).
    flip = (n_cam[..., 2] > 0).unsqueeze(-1)
    n_cam = torch.where(flip, -n_cam, n_cam)
    # Camera -> world: n_world = R^T n_cam.
    n_world = n_cam.reshape(-1, 3) @ camera.R       # (P,3) since (R^T n)^T = n^T R
    return n_world.reshape(H, W, 3)
