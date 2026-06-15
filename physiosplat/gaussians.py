"""Canonical Gaussian primitives (Sec. 2.1).

Each Gaussian ``g_i = {mu_i, s_i, q_i, alpha_i, c_i}`` is augmented by
PhysioSplat with:

* a learnable **semantic logit** ``s_i in R`` used by Physio-Semantic
  Disentanglement (Sec. 2.2). ``p_i = sigmoid(s_i)`` is the probability that
  the primitive belongs to the *tissue* class.
* the **specular** parameters of the Specular-Aware Appearance Model
  (Sec. 2.4): a learnable specular albedo ``beta_i`` and roughness
  ``gamma_i``.  The diffuse colour is the degree-0 SH coefficient ``c_{d,i}``.

Activations follow standard 3DGS conventions so the raw optimisation variables
are unconstrained:

* ``scaling``  is stored in log-space, activated with ``exp``.
* ``opacity``  is stored as a logit, activated with ``sigmoid``.
* ``rotation`` is stored as a raw quaternion, normalised on use.
* ``beta`` (specular albedo) is stored as a logit, activated with ``sigmoid``
  so it lives in ``[0, 1]``.
* ``gamma`` (roughness) is stored in log-space, activated with ``exp`` so it is
  strictly positive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physiosplat.sh_utils import rgb_to_sh0
from physiosplat.transforms import build_covariance, normalize_quaternion


class GaussianModel(nn.Module):
    """A differentiable collection of ``N`` canonical 3D Gaussians."""

    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor | None = None,
        init_scale: float = 0.02,
        init_opacity: float = 0.1,
        init_specular_albedo: float = 0.1,
        init_roughness: float = 0.3,
        init_semantic_logit: float = 0.0,
    ) -> None:
        """Create Gaussians from initial point positions.

        Args:
            means: ``(N, 3)`` initial canonical positions (e.g. back-projected
                points from the first frame's depth map).
            colors: optional ``(N, 3)`` RGB in ``[0, 1]`` used to initialise the
                diffuse SH0 coefficient.  Defaults to mid-grey.
            init_scale: isotropic initial scale (metres).
            init_opacity: initial opacity in ``(0, 1)``.
            init_specular_albedo: initial ``beta`` (paper uses 0.1).
            init_roughness: initial ``gamma`` (roughness of the specular lobe).
            init_semantic_logit: initial semantic logit (0 -> p = 0.5).
        """
        super().__init__()
        N = means.shape[0]
        device = means.device

        if colors is None:
            colors = torch.full((N, 3), 0.5, device=device)

        # Geometry ----------------------------------------------------------
        self._xyz = nn.Parameter(means.clone())
        log_scale = torch.log(torch.full((N, 3), float(init_scale), device=device))
        self._scaling = nn.Parameter(log_scale)
        quat = torch.zeros(N, 4, device=device)
        quat[:, 0] = 1.0  # identity rotation (w=1)
        self._rotation = nn.Parameter(quat)
        opa_logit = torch.logit(torch.full((N, 1), float(init_opacity), device=device))
        self._opacity = nn.Parameter(opa_logit)

        # Appearance --------------------------------------------------------
        # Diffuse colour = degree-0 SH (SH0).
        self._diffuse_sh0 = nn.Parameter(rgb_to_sh0(colors.clone()))
        # Specular albedo beta in [0,1] (stored as logit) and roughness gamma>0.
        beta_logit = torch.logit(
            torch.full((N, 1), float(init_specular_albedo), device=device)
        )
        self._specular_albedo = nn.Parameter(beta_logit)
        log_gamma = torch.log(torch.full((N, 1), float(init_roughness), device=device))
        self._roughness = nn.Parameter(log_gamma)

        # Semantics ---------------------------------------------------------
        self._semantic_logit = nn.Parameter(
            torch.full((N, 1), float(init_semantic_logit), device=device)
        )

    # ------------------------------------------------------------------ #
    # Activated accessors                                                #
    # ------------------------------------------------------------------ #
    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def scaling(self) -> torch.Tensor:
        """Positive scale vector ``s`` (N, 3)."""
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        """Unit quaternion ``q`` (N, 4)."""
        return normalize_quaternion(self._rotation)

    @property
    def opacity(self) -> torch.Tensor:
        """Opacity ``alpha`` in ``(0, 1)`` (N, 1)."""
        return torch.sigmoid(self._opacity)

    @property
    def diffuse_color(self) -> torch.Tensor:
        """Diffuse RGB ``c_d`` in ``[0, 1]`` (N, 3)."""
        from physiosplat.sh_utils import sh0_to_rgb

        return sh0_to_rgb(self._diffuse_sh0)

    @property
    def specular_albedo(self) -> torch.Tensor:
        """Specular albedo ``beta`` in ``[0, 1]`` (N, 1)."""
        return torch.sigmoid(self._specular_albedo)

    @property
    def roughness(self) -> torch.Tensor:
        """Roughness ``gamma > 0`` of the specular lobe (N, 1)."""
        return torch.exp(self._roughness)

    @property
    def semantic_logit(self) -> torch.Tensor:
        return self._semantic_logit

    @property
    def tissue_prob(self) -> torch.Tensor:
        """``p_i = sigmoid(s_i)`` -- probability of belonging to tissue (N, 1)."""
        return torch.sigmoid(self._semantic_logit)

    def covariance(self) -> torch.Tensor:
        """Canonical 3D covariance ``Sigma = R S S^T R^T`` (N, 3, 3)."""
        return build_covariance(self.scaling, self.rotation)

    def tissue_mask(self, tau: float = 0.5) -> torch.Tensor:
        """Boolean mask ``p_i >= tau`` selecting the tissue field (N,)."""
        return (self.tissue_prob.squeeze(-1) >= tau)

    def tool_mask(self, tau: float = 0.5) -> torch.Tensor:
        """Boolean mask ``p_i < tau`` selecting the tool field (N,)."""
        return ~self.tissue_mask(tau)

    # ------------------------------------------------------------------ #
    # Densification / pruning bookkeeping                                #
    # ------------------------------------------------------------------ #
    def replace_points(self, keep_mask: torch.Tensor) -> None:
        """Prune Gaussians, keeping those where ``keep_mask`` is True.

        Re-creates parameters in place (used by the trainer's pruning step).
        """
        with torch.no_grad():
            self._xyz = nn.Parameter(self._xyz[keep_mask].detach())
            self._scaling = nn.Parameter(self._scaling[keep_mask].detach())
            self._rotation = nn.Parameter(self._rotation[keep_mask].detach())
            self._opacity = nn.Parameter(self._opacity[keep_mask].detach())
            self._diffuse_sh0 = nn.Parameter(self._diffuse_sh0[keep_mask].detach())
            self._specular_albedo = nn.Parameter(
                self._specular_albedo[keep_mask].detach()
            )
            self._roughness = nn.Parameter(self._roughness[keep_mask].detach())
            self._semantic_logit = nn.Parameter(
                self._semantic_logit[keep_mask].detach()
            )
