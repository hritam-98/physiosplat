"""Physio-Semantic Gaussian Disentanglement -- kinematic physics (Sec. 2.2).

The global Gaussian set ``G`` is partitioned into two *disjoint* semantic
subsets governed by the learnable per-Gaussian tissue probability
``p_i = sigmoid(s_i)`` and a threshold ``tau``:

* the **Tool Field** ``G_tool`` (``p_i < tau``) moves rigidly:

      mu_i(t) = R_tool(t) mu_i(t-1) + T_tool(t)                       (Eq. 1, rigid)

* the **Tissue Field** ``G_tissue`` (``p_i >= tau``) deforms non-rigidly via
  the hex-plane field ``Phi_tissue``:

      mu_i(t) = mu_i(t-1) + d_mu_i(t)                          (Eq. 1, deformable)

At inference we can *erase* the tool (Physio-Inpainting, Eq. 2) by rendering
only the tissue field.

This module owns the per-frame **rigid tool pose** ``(R_tool(t), T_tool(t))``,
parameterised as a learnable axis-angle + translation per timestep, and the
soft/hard kinematic blending that produces deformed Gaussian states.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physiosplat.transforms import (
    normalize_quaternion,
    quaternion_to_rotation_matrix,
)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: axis-angle ``(..., 3)`` -> rotation ``(..., 3, 3)``."""
    theta = axis_angle.norm(dim=-1, keepdim=True)              # (...,1)
    small = theta < 1e-8
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(K.shape)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    R = eye + s * K + (1 - c) * (K @ K)
    # For ~zero rotation, fall back to identity to avoid 0/0.
    R = torch.where(small.unsqueeze(-1), eye, R)
    return R


class RigidToolMotion(nn.Module):
    """Per-timestep rigid pose ``(R_tool(t), T_tool(t)) in SE(3)`` for the tool.

    Poses are *relative* (frame ``t-1 -> t``), matching the incremental form of
    Eq. (1), and are zero-initialised so the tool starts static.
    """

    def __init__(self, num_frames: int) -> None:
        super().__init__()
        self.num_frames = num_frames
        # Relative axis-angle rotation and translation for each frame.
        self.axis_angle = nn.Parameter(torch.zeros(num_frames, 3))
        self.translation = nn.Parameter(torch.zeros(num_frames, 3))

    def relative_pose(self, t_index: int):
        """Return ``(R, T)`` mapping frame ``t-1`` to frame ``t``."""
        R = _axis_angle_to_matrix(self.axis_angle[t_index])  # (3,3)
        T = self.translation[t_index]                        # (3,)
        return R, T


class PhysioSemanticDisentanglement(nn.Module):
    """Combine rigid tool motion and deformable tissue motion (Eq. 1).

    Produces, for a given timestep, the deformed Gaussian attributes used by the
    rasterizer.  The kinematic split is implemented with a **soft** blend during
    training (so semantic logits receive gradients) that converges to the hard
    threshold ``tau`` of Eq. (1); a hard split is exposed for inference / Eq. 2.
    """

    def __init__(self, tau: float = 0.5, soft_temperature: float = 20.0) -> None:
        super().__init__()
        self.tau = tau
        # Sharpness of the soft tissue/tool gate around tau.
        self.soft_temperature = soft_temperature

    def kinematic_gate(self, tissue_prob: torch.Tensor, hard: bool) -> torch.Tensor:
        """Return per-Gaussian tissue gate ``g_i in [0, 1]`` (N, 1).

        ``g_i = 1`` -> deformable tissue motion, ``g_i = 0`` -> rigid tool motion.
        """
        if hard:
            return (tissue_prob >= self.tau).to(tissue_prob.dtype)
        return torch.sigmoid(self.soft_temperature * (tissue_prob - self.tau))

    def deform(
        self,
        gaussians,
        deform_field,
        rigid_motion: RigidToolMotion,
        t_index: int,
        t_norm: float,
        prev_xyz: torch.Tensor | None = None,
        hard: bool = False,
    ):
        """Compute deformed Gaussian state at frame ``t_index``.

        Args:
            gaussians: the :class:`~physiosplat.gaussians.GaussianModel`.
            deform_field: the hex-plane :class:`HexPlaneDeformationField`.
            rigid_motion: per-frame rigid tool poses.
            t_index: integer frame index.
            t_norm: time normalised to ``[0, 1]`` (input to the hex-plane).
            prev_xyz: ``mu(t-1)`` to apply the incremental Eq. (1).  If ``None``
                the canonical positions are used (i.e. absolute deformation from
                canonical space, as in standard 4DGS).
            hard: use the hard threshold split (inference) vs. soft (training).

        Returns:
            dict with deformed ``xyz, scaling, rotation, opacity, diffuse_color``
            plus ``tissue_prob`` and the kinematic ``gate``.
        """
        canonical_xyz = gaussians.xyz
        base_xyz = canonical_xyz if prev_xyz is None else prev_xyz

        tissue_prob = gaussians.tissue_prob          # (N,1)
        gate = self.kinematic_gate(tissue_prob, hard)  # (N,1)

        # --- Deformable tissue branch: mu + d_mu (and attribute refinement) ---
        deltas = deform_field(canonical_xyz, t_norm)
        tissue_xyz = base_xyz + deltas["d_xyz"]

        # --- Rigid tool branch: R_tool mu + T_tool -------------------------
        R, T = rigid_motion.relative_pose(t_index)
        tool_xyz = base_xyz @ R.T + T

        xyz = gate * tissue_xyz + (1.0 - gate) * tool_xyz

        # Attribute deltas only apply to the (deformable) tissue field.
        scaling = gaussians.scaling * torch.exp(gate * deltas["d_scaling"])
        rotation = normalize_quaternion(
            gaussians.rotation + gate * deltas["d_rotation"]
        )
        opacity = torch.sigmoid(
            gaussians._opacity + gate * deltas["d_opacity"]
        )
        diffuse = torch.clamp(
            gaussians.diffuse_color + gate * deltas["d_diffuse"], 0.0, 1.0
        )

        return {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "diffuse_color": diffuse,
            "tissue_prob": tissue_prob,
            "gate": gate,
            "tool_R": R,
            "tool_T": T,
        }
