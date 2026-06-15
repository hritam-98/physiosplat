"""PhysioSplat: Physics-Informed Dynamic Gaussian Splatting for Surgical
Scene Reconstruction.

Reference implementation of the framework described in:

    H. Basak and Z. Yin, "PhysioSplat: Physics-Informed Dynamic Gaussian
    Splatting for Surgical Scene Reconstruction".

The package is organised around the paper's three physical models:

* ``physiosplat.gaussians``  -- canonical Gaussian primitives (Sec. 2.1).
* ``physiosplat.deformation`` -- multi-resolution hex-plane deformation field.
* ``physiosplat.psgd``      -- Physio-Semantic Gaussian Disentanglement
  (kinematic physics, Sec. 2.2).
* ``physiosplat.dbr``       -- Differentiable Biomechanical Regularization
  (biomechanical physics, Sec. 2.3).
* ``physiosplat.saam``      -- Specular-Aware Appearance Modeling
  (optical physics, Sec. 2.4).
* ``physiosplat.rasterizer`` -- differentiable EWA splatting / alpha-blending.
* ``physiosplat.model``     -- the full ``PhysioSplat`` module gluing the
  three physical models together.
* ``physiosplat.losses``    -- the composite optimisation objective (Sec. 2.5).
* ``physiosplat.trainer``   -- the optimisation loop (Sec. 3, implementation
  details).
"""

from physiosplat.cameras import Camera
from physiosplat.gaussians import GaussianModel
from physiosplat.deformation import HexPlaneDeformationField
from physiosplat.model import PhysioSplat
from physiosplat.losses import total_loss

__all__ = [
    "Camera",
    "GaussianModel",
    "HexPlaneDeformationField",
    "PhysioSplat",
    "total_loss",
]

__version__ = "1.0.0"
