"""Tests for Specular-Aware Appearance Modeling (Sec. 2.4)."""

import torch

from physiosplat.saam import specular_color, view_directions


def test_specular_peaks_when_normal_faces_viewer():
    diffuse = torch.zeros(1, 3)
    beta = torch.tensor([[0.8]])
    gamma = torch.tensor([[0.2]])
    v = torch.tensor([[0.0, 0.0, 1.0]])

    # Normal aligned with view -> n.v = 1 -> lobe = exp(0) = 1 -> full beta.
    n_aligned = torch.tensor([[0.0, 0.0, 1.0]])
    c_aligned = specular_color(diffuse, beta, gamma, n_aligned, v)
    assert torch.allclose(c_aligned, torch.full((1, 3), 0.8), atol=1e-4)

    # Normal orthogonal -> n.v = 0 -> lobe = exp(-1/gamma^2) ~ 0.
    n_ortho = torch.tensor([[1.0, 0.0, 0.0]])
    c_ortho = specular_color(diffuse, beta, gamma, n_ortho, v)
    assert (c_ortho < c_aligned).all()


def test_diffuse_only_when_albedo_zero():
    diffuse = torch.rand(5, 3)
    beta = torch.zeros(5, 1)
    gamma = torch.full((5, 1), 0.3)
    n = torch.randn(5, 3)
    v = torch.randn(5, 3)
    c = specular_color(diffuse, beta, gamma, n, v)
    assert torch.allclose(c, diffuse, atol=1e-6)


def test_roughness_widens_lobe():
    diffuse = torch.zeros(1, 3)
    beta = torch.tensor([[1.0]])
    v = torch.tensor([[0.0, 0.0, 1.0]])
    n = torch.tensor([[0.3, 0.0, 0.954]])  # slightly off-axis
    n = n / n.norm()
    sharp = specular_color(diffuse, beta, torch.tensor([[0.1]]), n, v)
    broad = specular_color(diffuse, beta, torch.tensor([[0.6]]), n, v)
    # Larger roughness -> wider lobe -> more specular at off-axis angle.
    assert (broad >= sharp).all()


def test_view_directions_point_to_camera():
    xyz = torch.tensor([[0.0, 0.0, 0.0]])
    center = torch.tensor([0.0, 0.0, 1.0])
    v = view_directions(xyz, center)
    assert torch.allclose(v[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_specular_is_differentiable():
    diffuse = torch.rand(4, 3, requires_grad=True)
    beta = torch.rand(4, 1, requires_grad=True)
    gamma = (torch.rand(4, 1) + 0.1).requires_grad_(True)
    n = torch.randn(4, 3)
    v = torch.randn(4, 3)
    specular_color(diffuse, beta, gamma, n, v).sum().backward()
    assert beta.grad is not None and beta.grad.abs().sum() > 0
