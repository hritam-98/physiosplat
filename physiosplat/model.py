"""The full PhysioSplat model (Fig. 1 workflow).

Glues together the three physical models:

    (A) Physio-Semantic Gaussian Disentanglement (PSGD, Sec. 2.2)
    (B) Differentiable Biomechanical Regularization (DBR, Sec. 2.3)
    (C) Specular-Aware Appearance Modeling (SAAM, Sec. 2.4)

A render proceeds in two passes, as required by SAAM's geometry-derived normals:

    1. A depth pass produces the accumulated rendered depth ``D_rend``.
       Per-Gaussian normals are sampled from ``grad D_rend`` at each Gaussian's
       projected pixel (``n_i ~ grad D_rend``).
    2. A colour pass evaluates Eq. (7) using those normals and the co-located
       view direction, then alpha-composites colour / depth / semantics.

``render(..., inpaint=True)`` performs Physio-Inpainting (Eq. 2): only the
tissue field is composited, "erasing" the tool to recover occluded anatomy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physiosplat.cameras import Camera
from physiosplat.dbr import biomechanical_loss, build_arap_neighborhood
from physiosplat.deformation import HexPlaneDeformationField
from physiosplat.gaussians import GaussianModel
from physiosplat.psgd import PhysioSemanticDisentanglement, RigidToolMotion
from physiosplat.rasterizer import RenderOutput, normals_from_depth, rasterize
from physiosplat.saam import specular_color, view_directions
from physiosplat.transforms import build_covariance


class PhysioSplat(nn.Module):
    """Physics-informed dynamic Gaussian Splatting for surgical reconstruction."""

    def __init__(
        self,
        gaussians: GaussianModel,
        num_frames: int,
        *,
        tau: float = 0.5,
        hexplane_feature_dim: int = 64,
        hexplane_resolutions: tuple[int, ...] = (32, 64, 128),
        scene_bounds: float = 1.5,
        arap_k: int = 16,
        collision_epsilon: float = 2e-3,
        lambda_strain: float = 0.1,
        lambda_col: float = 1.0,
        lambda_vol: float = 0.05,
    ) -> None:
        super().__init__()
        self.gaussians = gaussians
        self.deform_field = HexPlaneDeformationField(
            feature_dim=hexplane_feature_dim,
            resolutions=hexplane_resolutions,
            bounds=scene_bounds,
        )
        self.rigid_motion = RigidToolMotion(num_frames)
        self.psgd = PhysioSemanticDisentanglement(tau=tau)

        self.tau = tau
        self.num_frames = num_frames
        self.arap_k = arap_k
        self.collision_epsilon = collision_epsilon
        self.lambda_strain = lambda_strain
        self.lambda_col = lambda_col
        self.lambda_vol = lambda_vol

        # ARAP neighbourhood is precomputed on the (fixed) canonical frame.
        self._neighborhood = None

    # ------------------------------------------------------------------ #
    def neighborhood(self):
        """Lazily (re)build the ARAP neighbourhood structure."""
        if self._neighborhood is None:
            with torch.no_grad():
                self._neighborhood = build_arap_neighborhood(
                    self.gaussians.xyz.detach(), k=self.arap_k
                )
        return self._neighborhood

    def reset_neighborhood(self):
        """Invalidate cached neighbourhood (after pruning / densification)."""
        self._neighborhood = None

    def time_norm(self, t_index: int) -> float:
        return t_index / max(self.num_frames - 1, 1)

    # ------------------------------------------------------------------ #
    def deformed_state(self, t_index: int, hard: bool):
        """Return the PSGD-deformed Gaussian state at frame ``t_index``."""
        return self.psgd.deform(
            self.gaussians,
            self.deform_field,
            self.rigid_motion,
            t_index=t_index,
            t_norm=self.time_norm(t_index),
            prev_xyz=None,  # absolute deformation from canonical (stable 4DGS).
            hard=hard,
        )

    def render(
        self,
        camera: Camera,
        t_index: int,
        *,
        inpaint: bool = False,
        hard_split: bool | None = None,
        bg_color: torch.Tensor | None = None,
    ) -> RenderOutput:
        """Render frame ``t_index`` from ``camera``.

        Args:
            inpaint: if True, render only the tissue field (Eq. 2, tool removal).
            hard_split: force hard/soft kinematic gate.  Defaults to hard at
                inference (no grad) and soft during training.
        """
        if hard_split is None:
            hard_split = (not self.training) or inpaint

        state = self.deformed_state(t_index, hard=hard_split)
        xyz = state["xyz"]
        cov3d = build_covariance(state["scaling"], state["rotation"])

        # --- Pass 1: depth -> per-Gaussian normals (SAAM, n_i ~ grad D_rend) ---
        diffuse = state["diffuse_color"]
        depth_pass = rasterize(
            camera,
            xyz,
            cov3d,
            colors=diffuse,
            opacity=state["opacity"],
            semantic_prob=state["tissue_prob"],
            bg_color=bg_color,
        )
        normal_map = normals_from_depth(depth_pass.depth, camera)  # (H,W,3)

        # Sample the normal map at each Gaussian's projected pixel.
        uv, _, _ = camera.project(xyz)
        H, W = camera.height, camera.width
        px = uv[:, 0].round().long().clamp(0, W - 1)
        py = uv[:, 1].round().long().clamp(0, H - 1)
        gauss_normals = normal_map[py, px]                         # (N,3)

        # --- Pass 2: specular colour (Eq. 7) + composite ---
        v = view_directions(xyz, camera.camera_center())
        colors = specular_color(
            diffuse,
            self.gaussians.specular_albedo,
            self.gaussians.roughness,
            gauss_normals,
            v,
        )

        visibility = None
        if inpaint:
            # Physio-Inpainting: keep only tissue Gaussians (Eq. 2).
            visibility = (state["tissue_prob"].reshape(-1) >= self.tau).to(xyz.dtype)

        out = rasterize(
            camera,
            xyz,
            cov3d,
            colors=colors,
            opacity=state["opacity"],
            semantic_prob=state["tissue_prob"],
            bg_color=bg_color,
            visibility=visibility,
        )
        return out

    # ------------------------------------------------------------------ #
    def biomech_loss(self, t_index: int):
        """Compute ``L_biomech`` (Eq. 3) at frame ``t_index`` (soft split)."""
        state = self.deformed_state(t_index, hard=False)
        tissue_mask = self.gaussians.tissue_mask(self.tau)
        tool_mask = ~tissue_mask
        tool_xyz = state["xyz"][tool_mask].detach()  # tool is a fixed occluder here
        return biomechanical_loss(
            deformed_xyz=state["xyz"],
            canonical_xyz=self.gaussians.xyz,
            neighborhood=self.neighborhood(),
            tissue_mask=tissue_mask,
            tool_xyz=tool_xyz,
            deform_field=self.deform_field,
            t_norm=self.time_norm(t_index),
            lambda_strain=self.lambda_strain,
            lambda_col=self.lambda_col,
            lambda_vol=self.lambda_vol,
            epsilon=self.collision_epsilon,
        )
