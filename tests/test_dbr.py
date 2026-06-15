"""Tests for Differentiable Biomechanical Regularization (Sec. 2.3)."""

import torch

from physiosplat.dbr import (
    build_arap_neighborhood,
    collision_energy,
    strain_energy,
    tool_sdf_proxy,
    volume_energy,
)
from physiosplat.deformation import HexPlaneDeformationField


def test_strain_energy_zero_for_rigid_motion():
    # A rigid translation preserves all pairwise distances -> E_strain == 0.
    xyz = torch.randn(30, 3)
    nb = build_arap_neighborhood(xyz, k=5)
    deformed = xyz + torch.tensor([0.1, -0.2, 0.05])  # pure translation
    mask = torch.ones(30, dtype=torch.bool)
    e = strain_energy(deformed, xyz, nb, mask)
    assert torch.allclose(e, torch.tensor(0.0), atol=1e-4)


def test_strain_energy_positive_for_stretch():
    xyz = torch.randn(30, 3)
    nb = build_arap_neighborhood(xyz, k=5)
    deformed = xyz * 1.5  # non-rigid stretch
    mask = torch.ones(30, dtype=torch.bool)
    e = strain_energy(deformed, xyz, nb, mask)
    assert float(e) > 0.0


def test_collision_energy_penalises_interpenetration():
    # A tissue point inside the tool's bounding box violates the margin.
    tool = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])
    inside = torch.tensor([[0.5, 0.5, 0.5]])
    outside = torch.tensor([[5.0, 5.0, 5.0]])
    e_in = collision_energy(inside, tool, epsilon=0.1)
    e_out = collision_energy(outside, tool, epsilon=0.1)
    assert float(e_in) > 0.0
    assert float(e_out) == 0.0


def test_tool_sdf_sign():
    tool = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    inside = torch.tensor([[0.5, 0.5, 0.5]])
    outside = torch.tensor([[2.0, 2.0, 2.0]])
    assert float(tool_sdf_proxy(inside, tool)[0]) < 0       # inside -> negative
    assert float(tool_sdf_proxy(outside, tool)[0]) > 0      # outside -> positive


def test_volume_energy_zero_for_identity_field():
    # A freshly-initialised (zero-output) deformation field is the identity,
    # so the Jacobian is I and det == 1 -> E_vol == 0.
    field = HexPlaneDeformationField(feature_dim=8, resolutions=(8, 16))
    xyz = torch.randn(40, 3) * 0.1
    mask = torch.ones(40, dtype=torch.bool)
    e = volume_energy(field, xyz, t_norm=0.0, tissue_mask=mask)
    assert torch.allclose(e, torch.tensor(0.0), atol=1e-5)


def test_strain_energy_has_gradient():
    xyz = torch.randn(20, 3)
    nb = build_arap_neighborhood(xyz, k=4)
    deformed = (xyz * 1.2).requires_grad_(True)
    mask = torch.ones(20, dtype=torch.bool)
    e = strain_energy(deformed, xyz, nb, mask)
    e.backward()
    assert deformed.grad.abs().sum() > 0
