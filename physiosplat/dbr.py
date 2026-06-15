"""Differentiable Biomechanical Regularization -- biomechanical physics (Sec. 2.3).

The deformation field ``Phi_tissue`` is ill-posed, so PhysioSplat restricts the
solution space to biologically plausible deformations with the energy

    L_biomech = lambda_strain * E_strain
              + lambda_col    * E_col
              + lambda_vol    * E_vol                                  (Eq. 3)

* **E_strain** -- As-Rigid-As-Possible local elastic strain energy (Eq. 4),
  with stiffness weights ``w_ij = exp(-|mu_i(0) - mu_j(0)|^2 / zeta^2)`` derived
  from canonical proximity.
* **E_col** -- tool-tissue collision repulsion (Eq. 5), where the tool volume
  is approximated by a proxy SDF ``Psi_tool`` built from the convex hull of
  ``G_tool``, and ``epsilon`` is a contact-thickness safety margin.
* **E_vol** -- volume preservation (Eq. 6), penalising deviation of the local
  deformation gradient's determinant from unity (quasi-incompressibility).

``zeta`` is "computed dynamically for each primitive as the average distance to
its K = 16 nearest neighbours in the canonical space".
"""

from __future__ import annotations

import torch


def knn(points: torch.Tensor, k: int):
    """Brute-force k-NN (excluding self).

    Args:
        points: ``(N, 3)`` canonical positions.
        k: number of neighbours.

    Returns:
        idx: ``(N, k)`` neighbour indices.
        dist2: ``(N, k)`` squared distances to those neighbours.
    """
    N = points.shape[0]
    k = min(k, max(N - 1, 1))
    d2 = torch.cdist(points, points) ** 2          # (N, N)
    # Exclude self by setting the diagonal to +inf.
    d2 = d2 + torch.diag(
        torch.full((N,), float("inf"), device=points.device, dtype=points.dtype)
    )
    dist2, idx = torch.topk(d2, k, dim=-1, largest=False)
    return idx, dist2


def build_arap_neighborhood(canonical_xyz: torch.Tensor, k: int = 16):
    """Precompute ARAP neighbourhood structure on the canonical frame.

    Returns a dict with ``idx`` (N,k), the stiffness weights ``w_ij`` (N,k) and
    the rest squared distances ``rest_d2`` (N,k).  ``zeta`` is per-primitive,
    set to the mean neighbour distance (Sec. 3, implementation details).
    """
    idx, dist2 = knn(canonical_xyz, k)              # (N,k)
    rest_dist = torch.sqrt(torch.clamp(dist2, min=0.0))
    # zeta_i = average distance to the K nearest neighbours.
    zeta = rest_dist.mean(dim=-1, keepdim=True).clamp_min(1e-6)  # (N,1)
    w = torch.exp(-dist2 / (zeta ** 2))             # w_ij = exp(-d2 / zeta^2)
    return {"idx": idx, "w": w, "rest_d2": dist2}


def strain_energy(
    deformed_xyz: torch.Tensor,
    canonical_xyz: torch.Tensor,
    neighborhood: dict,
    tissue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local elastic (ARAP) strain energy ``E_strain`` (Eq. 4) for one frame.

    E_strain = sum_i sum_{j in N(i)} w_ij ( |mu_i - mu_j|^2 - |mu_i^0 - mu_j^0|^2 )^2
    """
    idx = neighborhood["idx"]                       # (N,k)
    w = neighborhood["w"]                           # (N,k)
    rest_d2 = neighborhood["rest_d2"]               # (N,k)

    # Current squared distances between i and its canonical neighbours.
    mu_i = deformed_xyz.unsqueeze(1)                # (N,1,3)
    mu_j = deformed_xyz[idx]                        # (N,k,3)
    cur_d2 = ((mu_i - mu_j) ** 2).sum(dim=-1)       # (N,k)

    residual = (cur_d2 - rest_d2) ** 2              # (N,k)
    per_pair = w * residual                         # (N,k)
    per_point = per_pair.sum(dim=-1)                # (N,)

    if tissue_mask is not None:
        per_point = per_point * tissue_mask.to(per_point.dtype)
    return per_point.sum()


def tool_sdf_proxy(query: torch.Tensor, tool_points: torch.Tensor) -> torch.Tensor:
    """Proxy signed distance to the tool volume ``Psi_tool`` (Eq. 5 support).

    The paper approximates the tool volume by the convex hull of ``G_tool``.
    A differentiable, dependency-free surrogate that behaves like a signed
    distance to that solid is the (smooth) distance to the tool point set with
    an *inside* test against the tool's axis-aligned extent: points inside the
    tool's spatial support get a negative distance, points outside get the
    positive distance to the nearest tool Gaussian.

    Args:
        query: ``(M, 3)`` tissue positions to test.
        tool_points: ``(P, 3)`` tool Gaussian centres (the hull samples).

    Returns:
        ``(M,)`` signed distance (negative inside the tool volume).
    """
    if tool_points.numel() == 0:
        # No tool -> everything is "far outside"; large positive SDF.
        return torch.full(
            (query.shape[0],), 1e3, device=query.device, dtype=query.dtype
        )

    # Unsigned distance to the nearest tool sample.
    d = torch.cdist(query, tool_points)             # (M, P)
    nearest = d.min(dim=-1).values                  # (M,)

    # Inside test against the tool's axis-aligned bounding box (a cheap, smooth
    # convex-hull surrogate).  Inside -> sign = -1.
    lo = tool_points.min(dim=0).values
    hi = tool_points.max(dim=0).values
    inside = ((query >= lo) & (query <= hi)).all(dim=-1)
    sign = torch.where(inside, -1.0, 1.0)
    return sign * nearest


def collision_energy(
    tissue_xyz: torch.Tensor,
    tool_xyz: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Tool-tissue collision repulsion ``E_col`` (Eq. 5) for one frame.

    E_col = sum_i ReLU( epsilon - Psi_tool(mu_i) )^2
    """
    if tissue_xyz.numel() == 0 or tool_xyz.numel() == 0:
        return tissue_xyz.new_zeros(())
    psi = tool_sdf_proxy(tissue_xyz, tool_xyz)      # (M,)
    violation = torch.relu(epsilon - psi)           # >0 when inside the margin
    return (violation ** 2).sum()


def volume_energy(
    deform_field,
    canonical_xyz: torch.Tensor,
    t_norm: float,
    tissue_mask: torch.Tensor | None = None,
    max_points: int = 2048,
) -> torch.Tensor:
    """Volume-preservation energy ``E_vol`` (Eq. 6) for one frame.

    E_vol = sum_i ( det( grad Phi_tissue(mu_i, t) ) - 1 )^2

    The Jacobian is evaluated by the deformation field.  For efficiency on large
    scenes a random subset of tissue Gaussians (``max_points``) is used, matching
    the stochastic estimation common in 4DGS regularisers.
    """
    pts = canonical_xyz
    if tissue_mask is not None:
        pts = canonical_xyz[tissue_mask]
    if pts.shape[0] == 0:
        return canonical_xyz.new_zeros(())

    if pts.shape[0] > max_points:
        sel = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts = pts[sel]

    jac = deform_field.deformation_jacobian(pts, t_norm)  # (M,3,3)
    det = torch.linalg.det(jac)                            # (M,)
    return ((det - 1.0) ** 2).sum()


def biomechanical_loss(
    deformed_xyz: torch.Tensor,
    canonical_xyz: torch.Tensor,
    neighborhood: dict,
    tissue_mask: torch.Tensor,
    tool_xyz: torch.Tensor,
    deform_field,
    t_norm: float,
    lambda_strain: float = 0.1,
    lambda_col: float = 1.0,
    lambda_vol: float = 0.05,
    epsilon: float = 2e-3,
):
    """Full ``L_biomech`` (Eq. 3) for a single frame.

    ``epsilon`` is the collision safety margin (paper: 2 mm; expressed here in
    the scene's metric units, default 2e-3 = 2 mm if the scene is in metres).

    Returns ``(total, parts_dict)``.
    """
    tissue_deformed = deformed_xyz[tissue_mask]

    e_strain = strain_energy(
        deformed_xyz, canonical_xyz, neighborhood, tissue_mask
    )
    e_col = collision_energy(tissue_deformed, tool_xyz, epsilon)
    e_vol = volume_energy(deform_field, canonical_xyz, t_norm, tissue_mask)

    total = lambda_strain * e_strain + lambda_col * e_col + lambda_vol * e_vol
    parts = {
        "E_strain": e_strain.detach(),
        "E_col": e_col.detach(),
        "E_vol": e_vol.detach(),
    }
    return total, parts
