"""Tests for Physio-Semantic Gaussian Disentanglement (Sec. 2.2)."""

import torch

from physiosplat.deformation import HexPlaneDeformationField
from physiosplat.gaussians import GaussianModel
from physiosplat.psgd import (
    PhysioSemanticDisentanglement,
    RigidToolMotion,
    _axis_angle_to_matrix,
)


def _toy_model(n=20):
    means = torch.randn(n, 3) * 0.1
    g = GaussianModel(means)
    # Make first half tool (logit -3 -> p ~ 0), second half tissue (+3 -> p ~ 1).
    with torch.no_grad():
        g._semantic_logit[: n // 2] = -3.0
        g._semantic_logit[n // 2:] = 3.0
    deform = HexPlaneDeformationField(feature_dim=8, resolutions=(8, 16))
    rigid = RigidToolMotion(num_frames=4)
    return g, deform, rigid


def test_axis_angle_zero_is_identity():
    R = _axis_angle_to_matrix(torch.zeros(3))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_tissue_tool_masks_are_disjoint_and_complete():
    g, _, _ = _toy_model()
    tissue = g.tissue_mask(0.5)
    tool = g.tool_mask(0.5)
    assert not (tissue & tool).any()              # disjoint
    assert (tissue | tool).all()                  # complete partition


def test_rigid_branch_applies_pure_rotation_translation():
    g, deform, rigid = _toy_model()
    # Set a known rigid pose at frame 1.
    with torch.no_grad():
        rigid.axis_angle[1] = torch.tensor([0.0, 0.0, 0.3])
        rigid.translation[1] = torch.tensor([0.01, -0.02, 0.0])
    psgd = PhysioSemanticDisentanglement(tau=0.5)
    state = psgd.deform(g, deform, rigid, t_index=1, t_norm=0.33, hard=True)

    tool_mask = g.tool_mask(0.5)
    R, T = rigid.relative_pose(1)
    expected = g.xyz[tool_mask] @ R.T + T
    assert torch.allclose(state["xyz"][tool_mask], expected, atol=1e-5)


def test_inpaint_gate_selects_only_tissue():
    g, deform, rigid = _toy_model()
    psgd = PhysioSemanticDisentanglement(tau=0.5)
    state = psgd.deform(g, deform, rigid, t_index=0, t_norm=0.0, hard=True)
    gate = state["gate"].reshape(-1)
    # Hard gate must be exactly 0 (tool) or 1 (tissue).
    assert set(gate.unique().tolist()).issubset({0.0, 1.0})
    assert torch.equal(gate.bool(), g.tissue_mask(0.5))


def test_soft_gate_is_differentiable_wrt_semantic_logit():
    g, deform, rigid = _toy_model()
    # Give the tool a non-trivial rigid pose so the rigid and deformable motion
    # branches differ -- only then does the (soft) kinematic gate, and hence the
    # semantic logit, influence the output positions.
    with torch.no_grad():
        rigid.axis_angle[1] = torch.tensor([0.0, 0.0, 0.4])
        rigid.translation[1] = torch.tensor([0.05, 0.05, 0.0])
    psgd = PhysioSemanticDisentanglement(tau=0.5)
    state = psgd.deform(g, deform, rigid, t_index=1, t_norm=0.33, hard=False)
    loss = state["xyz"].sum()
    loss.backward()
    assert g._semantic_logit.grad is not None
    assert g._semantic_logit.grad.abs().sum() > 0
