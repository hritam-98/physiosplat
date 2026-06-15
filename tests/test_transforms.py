"""Tests for rotation / covariance utilities (Sec. 2.1)."""

import torch

from physiosplat.transforms import (
    build_covariance,
    quaternion_to_rotation_matrix,
    normalize_quaternion,
)


def test_identity_quaternion_is_identity_rotation():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    R = quaternion_to_rotation_matrix(q)[0]
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_rotation_is_orthonormal():
    q = torch.randn(10, 4)
    R = quaternion_to_rotation_matrix(q)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    # det = +1 (proper rotation).
    assert torch.allclose(torch.linalg.det(R), torch.ones(10), atol=1e-5)


def test_covariance_is_symmetric_psd():
    scaling = torch.rand(8, 3) + 0.1
    quat = normalize_quaternion(torch.randn(8, 4))
    cov = build_covariance(scaling, quat)
    # Symmetric.
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    # PSD: all eigenvalues >= 0.
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > -1e-5).all()
