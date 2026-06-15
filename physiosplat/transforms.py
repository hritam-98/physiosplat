"""Rotation / covariance utilities.

Each Gaussian stores a scale vector ``s`` and a unit quaternion ``q``.  Its 3D
covariance is reconstructed as ``Sigma = R S S^T R^T`` (Sec. 2.1), which is
positive semi-definite by construction.
"""

import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise quaternions ``(..., 4)`` to unit norm."""
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions ``(N, 4)`` (w, x, y, z) to rotations ``(N, 3, 3)``."""
    q = normalize_quaternion(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.empty(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_covariance(scaling: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """Build per-Gaussian 3D covariance ``Sigma = R S S^T R^T``.

    Args:
        scaling: ``(N, 3)`` *positive* scale (already activated, e.g. exp).
        quaternion: ``(N, 4)`` quaternion.

    Returns:
        ``(N, 3, 3)`` covariance matrices.
    """
    R = quaternion_to_rotation_matrix(quaternion)  # (N,3,3)
    S = torch.diag_embed(scaling)                   # (N,3,3)
    M = R @ S                                        # R S
    return M @ M.transpose(-1, -2)                   # (R S)(R S)^T = R S S^T R^T
