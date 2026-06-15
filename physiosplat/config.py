"""Training / model configuration (defaults follow Sec. 3 implementation details)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysioSplatConfig:
    # --- Optimisation (Sec. 3) ---
    iterations: int = 30000
    position_lr: float = 1.6e-3          # initial position LR with cosine decay
    feature_lr: float = 2.5e-3           # diffuse / specular / opacity LR
    deform_lr: float = 1.6e-3            # hex-plane + MLP LR
    rigid_lr: float = 1e-3               # rigid tool pose LR
    semantic_lr: float = 5e-3            # semantic logit LR

    # --- PSGD (Sec. 2.2) ---
    tau: float = 0.5                     # physio-semantic threshold

    # --- Deformation field (Sec. 3) ---
    hexplane_feature_dim: int = 64       # feature dim D = 64
    hexplane_resolutions: tuple = (32, 64, 128)
    scene_bounds: float = 1.5

    # --- DBR (Sec. 2.3 / Sec. 3) ---
    arap_k: int = 16                     # K = 16 nearest neighbours for zeta
    collision_epsilon: float = 2e-3      # 2 mm safety margin (metres)
    lambda_strain: float = 0.1
    lambda_col: float = 1.0
    lambda_vol: float = 0.05

    # --- Objective weights (Eq. 8 / Sec. 3) ---
    lambda_ssim: float = 0.2
    lambda_mask: float = 0.1
    lambda_biomech: float = 0.01

    # --- SAAM init (Sec. 3) ---
    init_specular_albedo: float = 0.1    # beta initialised to 0.1
    init_roughness: float = 0.3

    # --- Densification / pruning (standard 3DGS) ---
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_interval: int = 100
    opacity_prune_threshold: float = 0.005
    # densification is disabled where Psi_tool < epsilon (Sec. 3).

    # --- Init ---
    init_scale: float = 0.02
    init_stride: int = 1            # pixel stride when back-projecting depth
    init_max_points: int = 200000   # cap on the number of seed Gaussians

    # --- Logging ---
    log_interval: int = 100
    seed: int = 0
