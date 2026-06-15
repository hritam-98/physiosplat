"""Spherical-harmonics helpers.

PhysioSplat decomposes Gaussian radiance into a *view-independent diffuse*
term and a *view-dependent specular* term (Sec. 2.4).  Following the paper,
"the diffuse component ``c_{d,i}`` is represented purely by the degree-0 SH
coefficient (SH0)".  Degree-0 SH is a single constant per colour channel, so
the diffuse colour is obtained by the usual ``SH0 * C0 + 0.5`` mapping used by
3D Gaussian Splatting.
"""

import torch

# 0-th band real SH constant: Y_0^0 = 1 / (2 sqrt(pi)).
C0 = 0.28209479177387814


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients to an RGB colour in ``[0, 1]``.

    Args:
        sh0: ``(N, 3)`` degree-0 SH coefficients.

    Returns:
        ``(N, 3)`` diffuse RGB, clamped to ``[0, 1]``.
    """
    return torch.clamp(sh0 * C0 + 0.5, 0.0, 1.0)


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`sh0_to_rgb` (used for initialisation)."""
    return (rgb - 0.5) / C0
