"""Tests for the differentiable rasterizer (Sec. 2.1 alpha-blending)."""

import torch

from physiosplat.cameras import Camera
from physiosplat.rasterizer import normals_from_depth, rasterize
from physiosplat.transforms import build_covariance


def _camera(H=16, W=16):
    return Camera(
        R=torch.eye(3), t=torch.zeros(3),
        fx=0.9 * W, fy=0.9 * W, cx=W / 2, cy=H / 2, width=W, height=H,
    )


def test_single_gaussian_renders_near_projected_center():
    cam = _camera()
    xyz = torch.tensor([[0.0, 0.0, 0.06]])           # on the optical axis
    cov = build_covariance(torch.tensor([[0.01, 0.01, 0.01]]),
                           torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    color = torch.tensor([[1.0, 0.0, 0.0]])
    opacity = torch.tensor([[0.99]])
    sem = torch.tensor([[1.0]])
    out = rasterize(cam, xyz, cov, color, opacity, sem)

    # Brightest red pixel should be at the image centre (cx, cy).
    red = out.color[..., 0]
    idx = torch.argmax(red)
    py, px = idx // cam.width, idx % cam.width
    assert abs(int(px) - cam.width // 2) <= 1
    assert abs(int(py) - cam.height // 2) <= 1


def test_alpha_and_depth_are_consistent():
    cam = _camera()
    xyz = torch.tensor([[0.0, 0.0, 0.05]])
    cov = build_covariance(torch.tensor([[0.02, 0.02, 0.02]]),
                           torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    out = rasterize(cam, xyz, cov, torch.tensor([[0.5, 0.5, 0.5]]),
                    torch.tensor([[0.99]]), torch.tensor([[1.0]]))
    # Where opacity accumulated, expected depth must be ~ the gaussian depth.
    covered = out.alpha > 0.5
    assert covered.any()
    assert torch.allclose(
        out.depth[covered].mean(), torch.tensor(0.05), atol=0.01
    )


def test_visibility_gate_erases_gaussian():
    # Two Gaussians; gating the front one to visibility 0 should reveal the back.
    cam = _camera()
    xyz = torch.tensor([[0.0, 0.0, 0.04], [0.0, 0.0, 0.06]])
    cov = build_covariance(
        torch.tensor([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # front red, back green
    opacity = torch.tensor([[0.99], [0.99]])
    sem = torch.tensor([[0.0], [1.0]])

    full = rasterize(cam, xyz, cov, colors, opacity, sem)
    cy, cx = cam.height // 2, cam.width // 2
    assert full.color[cy, cx, 0] > full.color[cy, cx, 1]   # red dominates

    vis = torch.tensor([0.0, 1.0])                          # erase front gaussian
    gated = rasterize(cam, xyz, cov, colors, opacity, sem, visibility=vis)
    assert gated.color[cy, cx, 1] > gated.color[cy, cx, 0]  # green now visible


def test_render_is_differentiable():
    cam = _camera()
    xyz = torch.tensor([[0.0, 0.0, 0.06]], requires_grad=True)
    cov = build_covariance(torch.tensor([[0.02, 0.02, 0.02]]),
                           torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    color = torch.tensor([[1.0, 0.5, 0.2]], requires_grad=True)
    out = rasterize(cam, xyz, cov, color, torch.tensor([[0.9]]),
                    torch.tensor([[1.0]]))
    out.color.sum().backward()
    assert color.grad.abs().sum() > 0


def test_normals_from_flat_depth_face_camera():
    cam = _camera()
    depth = torch.full((cam.height, cam.width), 0.06)
    n = normals_from_depth(depth, cam)
    # A fronto-parallel plane has normals along the camera -z axis (in world,
    # camera is identity) -> z-component dominant, pointing back at camera.
    interior = n[2:-2, 2:-2]
    assert interior[..., 2].abs().mean() > 0.9
