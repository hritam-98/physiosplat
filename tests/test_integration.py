"""End-to-end integration test on the synthetic surgical scene.

Verifies the full PhysioSplat pipeline trains (PSNR improves) and that
Physio-Inpainting (Eq. 2) erases the tool to recover occluded anatomy.
"""

import torch

from physiosplat.config import PhysioSplatConfig
from physiosplat.losses import psnr
from physiosplat.model import PhysioSplat
from physiosplat.synthetic import make_synthetic_scene
from physiosplat.trainer import build_model_from_dataset, train


def _tiny_cfg():
    cfg = PhysioSplatConfig()
    cfg.iterations = 120
    cfg.densify_until_iter = 0      # keep the toy scene fixed-size
    cfg.log_interval = 1000
    cfg.hexplane_resolutions = (8, 16)
    cfg.hexplane_feature_dim = 16
    cfg.arap_k = 6
    # Keep the seed cloud small so the pure-Python rasterizer stays fast.
    cfg.init_stride = 2
    cfg.init_max_points = 1200
    return cfg


def test_model_builds_and_renders():
    dataset, _ = make_synthetic_scene(num_frames=4, H=32, W=32)
    cfg = _tiny_cfg()
    model = build_model_from_dataset(dataset, cfg, device="cpu")
    assert isinstance(model, PhysioSplat)
    assert model.gaussians.num_points > 0
    out = model.render(dataset[0].camera, 0)
    assert out.color.shape == (32, 32, 3)
    assert torch.isfinite(out.color).all()


def test_training_improves_psnr():
    torch.manual_seed(0)
    dataset, _ = make_synthetic_scene(num_frames=4, H=32, W=32)
    cfg = _tiny_cfg()
    model = build_model_from_dataset(dataset, cfg, device="cpu")

    frame0 = dataset[0]
    valid = 1.0 - frame0.tool_mask
    with torch.no_grad():
        before = float(psnr(model.render(frame0.camera, 0).color, frame0.image, valid))

    train(model, dataset, cfg, device="cpu", log_fn=None)

    with torch.no_grad():
        after = float(psnr(model.render(frame0.camera, 0).color, frame0.image, valid))

    assert after > before, f"PSNR did not improve: {before:.2f} -> {after:.2f}"


def test_inpainting_recovers_occluded_tissue():
    torch.manual_seed(0)
    dataset, _ = make_synthetic_scene(num_frames=4, H=32, W=32)
    cfg = _tiny_cfg()
    cfg.iterations = 150
    model = build_model_from_dataset(dataset, cfg, device="cpu")
    train(model, dataset, cfg, device="cpu", log_fn=None)

    frame0 = dataset.to("cpu")[0]
    tool_region = frame0.tool_mask > 0.5
    assert tool_region.any()
    assert frame0.clean_image is not None

    with torch.no_grad():
        full = model.render(frame0.camera, 0)
        inpaint = model.render(frame0.camera, 0, inpaint=True)

    # Physio-Inpainting (Eq. 2) renders only the tissue field, so the bright,
    # specular tool that dominates the tool region in the full render must be
    # *erased*: the inpainted colour there moves away from the tool's appearance
    # and towards the tissue (tool-free) reference.
    tool_color = frame0.image[tool_region]        # observed (bright tool) colour
    tissue_gt = frame0.clean_image[tool_region]   # tool-free reference

    full_to_tool = float((full.color[tool_region] - tool_color).abs().mean())
    inpaint_to_tool = float((inpaint.color[tool_region] - tool_color).abs().mean())
    # The full render reproduces the tool; inpainting departs from it.
    assert inpaint_to_tool > full_to_tool

    # And inpainting is closer to the tool-free tissue than to the tool.
    inpaint_to_tissue = float((inpaint.color[tool_region] - tissue_gt).abs().mean())
    assert inpaint_to_tissue < inpaint_to_tool
