"""Pinhole camera model used for projecting and rendering Gaussians.

The endoscopic datasets the paper evaluates on (EndoNeRF, StereoMIS, SCARED)
provide a single monocular camera per frame.  We keep the camera intrinsics /
extrinsics here together with the projection maths required by the rasterizer.
"""

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """A monocular pinhole camera.

    Attributes:
        R: ``(3, 3)`` world-to-camera rotation.
        t: ``(3,)`` world-to-camera translation, so ``x_cam = R x_world + t``.
        fx, fy: focal lengths in pixels.
        cx, cy: principal point in pixels.
        width, height: image resolution.
        znear: near clipping distance (Gaussians closer are culled).
    """

    R: torch.Tensor
    t: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    znear: float = 1e-3

    def to(self, device) -> "Camera":
        return Camera(
            R=self.R.to(device),
            t=self.t.to(device),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            znear=self.znear,
        )

    @property
    def device(self):
        return self.R.device

    def world_to_camera(self, xyz: torch.Tensor) -> torch.Tensor:
        """Transform world points ``(N, 3)`` into camera space ``(N, 3)``."""
        return xyz @ self.R.T + self.t

    def project(self, xyz: torch.Tensor):
        """Project world points to image plane.

        Returns:
            uv:     ``(N, 2)`` pixel coordinates.
            depth:  ``(N,)`` camera-space z (depth).
            J:      ``(N, 2, 3)`` Jacobian of the projection w.r.t. camera-space
                    points, used to propagate the 3D covariance to 2D (the EWA
                    splatting affine approximation).
        """
        cam = self.world_to_camera(xyz)             # (N,3)
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        z_safe = torch.clamp(z, min=self.znear)

        u = self.fx * x / z_safe + self.cx
        v = self.fy * y / z_safe + self.cy
        uv = torch.stack([u, v], dim=-1)

        # Jacobian of (u, v) wrt (x, y, z) of the camera-space point.
        N = xyz.shape[0]
        J = torch.zeros(N, 2, 3, device=xyz.device, dtype=xyz.dtype)
        inv_z = 1.0 / z_safe
        J[:, 0, 0] = self.fx * inv_z
        J[:, 0, 2] = -self.fx * x * inv_z * inv_z
        J[:, 1, 1] = self.fy * inv_z
        J[:, 1, 2] = -self.fy * y * inv_z * inv_z
        return uv, z, J

    def camera_center(self) -> torch.Tensor:
        """Return the camera centre in world coordinates ``C = -R^T t``."""
        return -self.R.T @ self.t
