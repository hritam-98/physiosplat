"""Optimisation loop for PhysioSplat (Sec. 3, implementation details).

The framework is optimised via Adam for 30,000 iterations with an initial
position learning rate ``eta = 1.6e-3`` and **cosine decay**.  The composite
objective ``L_total`` (Eq. 8) jointly optimises geometry, diffuse colour,
specular parameters, semantic logits, the hex-plane deformation field, and the
per-frame rigid tool poses.

Standard 3DGS opacity pruning is applied; densification is disabled inside the
tool volume (``Psi_tool < epsilon``) to prevent artifact generation.
"""

from __future__ import annotations

import math

import torch

from physiosplat.config import PhysioSplatConfig
from physiosplat.dataset import EndoscopicDataset, backproject_multiframe
from physiosplat.gaussians import GaussianModel
from physiosplat.losses import psnr, total_loss
from physiosplat.model import PhysioSplat


def build_model_from_dataset(
    dataset: EndoscopicDataset, cfg: PhysioSplatConfig, device="cpu"
) -> PhysioSplat:
    """Initialise canonical Gaussians from the depth back-projection of all frames.

    Tool-pixel points are kept (the tool field is an explicit occluder, Sec. 2.2)
    and seeded with a tool-leaning semantic logit; tissue points get a
    tissue-leaning logit.  Aggregating across frames lets the tissue field
    represent anatomy that is occluded by the tool in any single view.
    """
    dataset = dataset.to(device)
    points, colors, is_tool = backproject_multiframe(
        dataset, stride=cfg.init_stride, max_points=cfg.init_max_points
    )

    if points.shape[0] == 0:
        raise ValueError("First frame produced no valid depth points for init.")

    gaussians = GaussianModel(
        means=points,
        colors=colors,
        init_scale=cfg.init_scale,
        init_specular_albedo=cfg.init_specular_albedo,
        init_roughness=cfg.init_roughness,
    ).to(device)

    # Seed semantic logits from the initial tool/tissue labels:
    #   p_i = sigmoid(s_i); tissue -> p high (logit +), tool -> p low (logit -).
    with torch.no_grad():
        logit = torch.where(
            is_tool.unsqueeze(-1),
            torch.full_like(gaussians._semantic_logit, -2.0),
            torch.full_like(gaussians._semantic_logit, +2.0),
        )
        gaussians._semantic_logit.copy_(logit)

    model = PhysioSplat(
        gaussians,
        num_frames=len(dataset),
        tau=cfg.tau,
        hexplane_feature_dim=cfg.hexplane_feature_dim,
        hexplane_resolutions=tuple(cfg.hexplane_resolutions),
        scene_bounds=cfg.scene_bounds,
        arap_k=cfg.arap_k,
        collision_epsilon=cfg.collision_epsilon,
        lambda_strain=cfg.lambda_strain,
        lambda_col=cfg.lambda_col,
        lambda_vol=cfg.lambda_vol,
    ).to(device)
    return model


def build_optimizer(model: PhysioSplat, cfg: PhysioSplatConfig) -> torch.optim.Adam:
    """Per-parameter-group Adam, mirroring 3DGS's grouped learning rates."""
    g = model.gaussians
    groups = [
        {"params": [g._xyz], "lr": cfg.position_lr, "name": "xyz"},
        {"params": [g._scaling, g._rotation], "lr": cfg.feature_lr, "name": "geom"},
        {
            "params": [g._opacity, g._diffuse_sh0, g._specular_albedo, g._roughness],
            "lr": cfg.feature_lr,
            "name": "appearance",
        },
        {"params": [g._semantic_logit], "lr": cfg.semantic_lr, "name": "semantic"},
        {"params": list(model.deform_field.parameters()), "lr": cfg.deform_lr,
         "name": "deform"},
        {"params": list(model.rigid_motion.parameters()), "lr": cfg.rigid_lr,
         "name": "rigid"},
    ]
    return torch.optim.Adam(groups, eps=1e-15)


def cosine_lr(step: int, total: int, base_lr: float, final_factor: float = 0.01):
    """Cosine-decayed learning rate from ``base_lr`` to ``base_lr*final_factor``."""
    if total <= 1:
        return base_lr
    progress = min(step / total, 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (final_factor + (1 - final_factor) * cos)


def train(
    model: PhysioSplat,
    dataset: EndoscopicDataset,
    cfg: PhysioSplatConfig,
    device="cpu",
    log_fn=print,
):
    """Run the optimisation loop. Returns a list of per-log-step metric dicts."""
    torch.manual_seed(cfg.seed)
    dataset = dataset.to(device)
    optimizer = build_optimizer(model, cfg)
    model.train()

    num_frames = len(dataset)
    history = []

    for it in range(1, cfg.iterations + 1):
        # Cosine decay only on the position LR (Sec. 3 specifies position LR
        # decay; others held constant, as in 3DGS).
        for group in optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = cosine_lr(it, cfg.iterations, cfg.position_lr)

        # Sample a random frame each iteration.
        t_index = int(torch.randint(0, num_frames, (1,)).item())
        frame = dataset[t_index]

        render = model.render(frame.camera, t_index)
        biomech, biomech_parts = model.biomech_loss(t_index)
        loss, parts = total_loss(
            render,
            gt_color=frame.image,
            gt_tool_mask=frame.tool_mask,
            biomech=biomech,
            lambda_ssim=cfg.lambda_ssim,
            lambda_mask=cfg.lambda_mask,
            lambda_biomech=cfg.lambda_biomech,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # --- Standard 3DGS opacity pruning (densification disabled in tool) ---
        if (
            cfg.densify_from_iter <= it <= cfg.densify_until_iter
            and it % cfg.densify_interval == 0
        ):
            if _prune_low_opacity(model, cfg):
                # Gaussian parameters were re-created; rebuild the optimizer so
                # its param groups reference the live tensors.
                optimizer = build_optimizer(model, cfg)

        if it % cfg.log_interval == 0 or it == 1:
            valid = 1.0 - frame.tool_mask
            p = psnr(render.color.detach(), frame.image, mask=valid)
            rec = {"iter": it, "psnr": float(p), **{k: float(v) for k, v in parts.items()},
                   **{k: float(v) for k, v in biomech_parts.items()}}
            history.append(rec)
            if log_fn is not None:
                log_fn(
                    f"[{it:>6}/{cfg.iterations}] "
                    f"PSNR={float(p):5.2f}  L_total={parts['L_total']:.4f}  "
                    f"L1={parts['L1']:.4f}  SSIM_loss={parts['L_SSIM']:.4f}  "
                    f"mask={parts['L_mask']:.4f}  biomech={parts['L_biomech']:.4f}"
                )

    model.eval()
    return history


@torch.no_grad()
def _prune_low_opacity(model: PhysioSplat, cfg: PhysioSplatConfig) -> bool:
    """Remove near-transparent Gaussians and rebuild ARAP neighbourhood.

    Returns ``True`` if any Gaussian was pruned (so the caller rebuilds the
    optimizer to point at the freshly created parameters).
    """
    keep = model.gaussians.opacity.reshape(-1) > cfg.opacity_prune_threshold
    if bool(keep.all()):
        return False
    model.gaussians.replace_points(keep)
    model.reset_neighborhood()
    return True
