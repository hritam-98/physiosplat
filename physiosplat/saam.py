r"""Specular-Aware Appearance Modeling -- optical physics (Sec. 2.4).

Endoscopy uses a light source co-located with the camera, so the illumination
half-vector aligns with the view direction ``v``.  PhysioSplat decomposes each
Gaussian's radiance into a view-independent diffuse term ``c_{d,i}`` (the SH0
coefficient) and a view-dependent specular lobe:

    C = sum_i a_i ( c_{d,i} + beta_i exp( (n_i . v - 1) / gamma_i^2 ) )      (Eq. 7)
                                \_________ specular c_{s,i} _________/

* ``beta_i``  -- learnable specular albedo.
* ``gamma_i`` -- learnable roughness (width of the specular lobe).
* ``n_i``     -- surface normal, **not** a free parameter: it is derived
  on-the-fly from the spatial gradient of the accumulated rendered depth map
  (``n_i ~ grad D_rend``), forcing specularity to stay consistent with geometry.

This module computes the per-Gaussian colour ``c_i = c_{d,i} + c_{s,i}`` that is
fed to the rasterizer, given per-Gaussian normals and view directions.
"""

from __future__ import annotations

import torch


def specular_color(
    diffuse: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    normals: torch.Tensor,
    view_dirs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq. (7)'s per-Gaussian radiance ``c = c_d + c_s``.

    Args:
        diffuse: ``(N, 3)`` diffuse colour ``c_d`` in ``[0, 1]``.
        beta: ``(N, 1)`` specular albedo.
        gamma: ``(N, 1)`` roughness (> 0).
        normals: ``(N, 3)`` unit surface normals ``n_i``.
        view_dirs: ``(N, 3)`` unit view directions ``v`` (Gaussian -> camera).

    Returns:
        ``(N, 3)`` combined radiance, clamped to ``[0, 1]``.
    """
    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    v = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

    # n . v in [-1, 1]; the lobe peaks (=1) when the normal faces the viewer.
    ndotv = (n * v).sum(dim=-1, keepdim=True)            # (N,1)
    lobe = torch.exp((ndotv - 1.0) / (gamma ** 2 + 1e-8))  # (N,1) in (0,1]
    c_s = beta * lobe                                     # specular (N,1) -> broadcast
    c = diffuse + c_s                                     # (N,3)
    return torch.clamp(c, 0.0, 1.0)


def view_directions(xyz: torch.Tensor, camera_center: torch.Tensor) -> torch.Tensor:
    """Unit view direction from each Gaussian towards the camera centre."""
    v = camera_center.unsqueeze(0) - xyz
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)
