"""Multi-resolution hex-plane deformation field ``Phi_tissue`` (Sec. 2.1-2.2).

The paper models temporal dynamics with a deformation field

    F_theta : (g_i, t) -> (d_mu, d_s, d_q, d_alpha, d_c)

"parameterised by a multi-resolution hex-plane encoder (feature dim D = 64)".

A hex-plane decomposes the 4D space-time volume ``(x, y, z, t)`` into the six
2D coordinate planes ``{xy, xz, yz, xt, yt, zt}``.  Each plane is sampled
(bilinearly) at multiple spatial resolutions; the per-plane features are
multiplied across the six planes (HexPlane fusion) and concatenated across
resolutions, then decoded by a small MLP into the per-Gaussian deltas.

For the deformable **tissue** field we use the positional delta ``d_mu`` to
drive Eq. (1) (``mu_i(t) = mu_i(t-1) + d_mu_i(t)``); the scale / rotation /
opacity / colour deltas refine the canonical primitive over time.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

# The six coordinate planes of a 4D (x, y, z, t) hex-plane, as index pairs into
# the normalised coordinate vector [x, y, z, t].
_PLANES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]


class HexPlaneEncoder(nn.Module):
    """Multi-resolution hex-plane feature encoder for 4D ``(x, y, z, t)``."""

    def __init__(
        self,
        feature_dim: int = 64,
        resolutions: tuple[int, ...] = (32, 64, 128),
        bounds: float = 1.5,
    ) -> None:
        """Args:
        feature_dim: per-plane feature channels ``D`` (paper uses 64).
        resolutions: spatial grid resolution at each multi-resolution level.
        bounds: half-extent the normalised scene coordinates are mapped into
            ``[-1, 1]`` (coordinates are divided by ``bounds`` then clamped).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.resolutions = resolutions
        self.bounds = bounds

        self.planes = nn.ModuleList()
        for res in resolutions:
            level = nn.ParameterList()
            for _ in _PLANES:
                # Grid of shape (1, D, res, res) for grid_sample.
                grid = nn.Parameter(torch.empty(1, feature_dim, res, res))
                # Small init so the field starts near "no deformation".
                nn.init.uniform_(grid, -1e-2, 1e-2)
                level.append(grid)
            self.planes.append(level)

    @property
    def output_dim(self) -> int:
        return self.feature_dim * len(self.resolutions)

    def forward(self, xyz: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Sample hex-plane features for points ``(N, 3)`` at time ``t``.

        ``t`` is a scalar (or ``(N,)``) time in ``[0, 1]``.
        """
        N = xyz.shape[0]
        device = xyz.device

        if not torch.is_tensor(t):
            t = torch.full((N,), float(t), device=device, dtype=xyz.dtype)
        else:
            t = t.to(device=device, dtype=xyz.dtype)
            if t.dim() == 0:
                t = t.expand(N)

        # Normalise (x, y, z, t) into [-1, 1].  Spatial coords are scaled by the
        # scene bound; time is already in [0, 1] -> map to [-1, 1].
        coords = torch.empty(N, 4, device=device, dtype=xyz.dtype)
        coords[:, :3] = torch.clamp(xyz / self.bounds, -1.0, 1.0)
        coords[:, 3] = torch.clamp(t * 2.0 - 1.0, -1.0, 1.0)

        level_features = []
        for level in self.planes:
            # Multiply the six plane features together (HexPlane fusion).
            prod = None
            for grid, (a, b) in zip(level, _PLANES):
                # grid_sample expects coords as (1, N, 1, 2) in [-1, 1].
                ab = torch.stack([coords[:, a], coords[:, b]], dim=-1)
                ab = ab.view(1, N, 1, 2)
                feat = F.grid_sample(
                    grid, ab, mode="bilinear", align_corners=True
                )  # (1, D, N, 1)
                feat = feat.squeeze(0).squeeze(-1).transpose(0, 1)  # (N, D)
                prod = feat if prod is None else prod * feat
            level_features.append(prod)
        return torch.cat(level_features, dim=-1)  # (N, D * num_levels)


class HexPlaneDeformationField(nn.Module):
    """Deformation field ``F_theta``: maps canonical Gaussians + time to deltas.

    Returns the tuple ``(d_mu, d_scale, d_quat, d_opacity, d_diffuse)``.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        resolutions: tuple[int, ...] = (32, 64, 128),
        hidden_dim: int = 128,
        bounds: float = 1.5,
    ) -> None:
        super().__init__()
        self.encoder = HexPlaneEncoder(feature_dim, resolutions, bounds)

        in_dim = self.encoder.output_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Separate heads for each deformable quantity.
        self.head_xyz = nn.Linear(hidden_dim, 3)
        self.head_scale = nn.Linear(hidden_dim, 3)
        self.head_quat = nn.Linear(hidden_dim, 4)
        self.head_opacity = nn.Linear(hidden_dim, 1)
        self.head_diffuse = nn.Linear(hidden_dim, 3)

        # Zero-init the output heads so the field starts as the identity
        # deformation (canonical == deformed at iteration 0).
        for head in (
            self.head_xyz,
            self.head_scale,
            self.head_quat,
            self.head_opacity,
            self.head_diffuse,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, xyz: torch.Tensor, t: float | torch.Tensor):
        """Compute deformation deltas for canonical points ``(N, 3)`` at ``t``."""
        feat = self.encoder(xyz, t)
        h = self.backbone(feat)
        return {
            "d_xyz": self.head_xyz(h),
            "d_scaling": self.head_scale(h),
            "d_rotation": self.head_quat(h),
            "d_opacity": self.head_opacity(h),
            "d_diffuse": self.head_diffuse(h),
        }

    def deformation_jacobian(
        self, xyz: torch.Tensor, t: float | torch.Tensor, eps: float = 1e-3
    ) -> torch.Tensor:
        """Jacobian ``d(deformed_mu)/d(canonical_mu)`` for each point.

        Required by the volume-preservation energy ``E_vol`` (Eq. 6), which
        penalises ``(det(grad Phi_tissue) - 1)^2``.  The deformed position is
        ``Phi(mu) = mu + d_xyz(mu, t)`` so the Jacobian is
        ``I + d(d_xyz)/d(mu)``.

        We estimate the spatial Jacobian by **central finite differences** of
        the deformation in canonical space.  This keeps ``E_vol`` first-order
        differentiable w.r.t. the field parameters (the only quantities being
        optimised) while avoiding second-order backprop through ``grid_sample``
        (whose double-backward is unsupported in PyTorch).

        Returns ``(N, 3, 3)``.
        """
        base = xyz.detach()
        N = base.shape[0]
        jac = torch.zeros(N, 3, 3, device=base.device, dtype=base.dtype)
        for j in range(3):
            offset = torch.zeros(3, device=base.device, dtype=base.dtype)
            offset[j] = eps
            d_plus = self.forward(base + offset, t)["d_xyz"]
            d_minus = self.forward(base - offset, t)["d_xyz"]
            # Column j of d(d_xyz)/d(mu): central difference.
            jac[:, :, j] = (d_plus - d_minus) / (2 * eps)
        # Add identity (Phi = mu + d_xyz => J = I + d(d_xyz)/d(mu)).
        jac = jac + torch.eye(3, device=base.device, dtype=base.dtype)
        return jac
