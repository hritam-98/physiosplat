# PhysioSplat

**Physics-Informed Dynamic Gaussian Splatting for Surgical Scene Reconstruction**
— H. Basak and Z. Yin.

🌐 **Project page:** https://hritam-98.github.io/physiosplat/

PhysioSplat reconstructs dynamic endoscopic scenes by grounding 4D Gaussian
Splatting in three physical models:

- **PSGD** (kinematic) — splits the scene into a rigid *tool* field and a
  deformable *tissue* field, enabling tool removal and recovery of occluded
  anatomy.
- **DBR** (biomechanical) — ARAP strain, tool–tissue collision repulsion, and
  volume preservation for biologically plausible tissue dynamics.
- **SAAM** (optical) — diffuse + view-dependent specular decomposition for the
  co-located camera–light of endoscopy, stabilizing highlights on wet surfaces.

Runs on CPU or GPU with a pure-PyTorch differentiable rasterizer (no custom CUDA
kernel required).

---

## Install

```bash
git clone https://github.com/hritam-98/physiosplat.git
cd physiosplat
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Try it (no data needed)

Train on a built-in synthetic scene and see the tool removed via inpainting:

```bash
python examples/run_synthetic.py
```

Expected: PSNR rises sharply during training and Physio-Inpainting recovers the
tool-occluded tissue.

## Run the tests

```bash
pytest tests/ -q
```

## Train on your own scene

```bash
python scripts/train.py --data /path/to/scene --iterations 30000 --device cuda
python scripts/render.py --ckpt output/physiosplat.pt --data /path/to/scene
```

`render.py` writes both the full reconstruction and the tool-removed (inpainted)
view per frame.

### Dataset layout

```
scene/
  images/      0000.png 0001.png ...   # RGB frames
  masks/       0000.png ...            # tool masks (white = tool)
  depth/       0000.npy ...            # depth maps (.npy or .png)
  cameras.json                         # intrinsics + per-frame extrinsics
```

```json
{
  "intrinsics": {"fx": 1035.0, "fy": 1035.0, "cx": 320.0, "cy": 256.0},
  "frames": [
    {"file": "0000.png", "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}
  ]
}
```

Evaluated in the paper on EndoNeRF, StereoMIS, and SCARED.

## Configuration

Defaults follow the paper (Sec. 3) and live in
[`configs/endonerf.yaml`](configs/endonerf.yaml) / `physiosplat/config.py`:
30k Adam iterations, position LR `1.6e-3` (cosine decay), hex-plane dim `D=64`,
`τ=0.5`, `λ_strain=0.1`, `λ_col=1.0`, `λ_vol=0.05`, collision margin `ε=2mm`,
`K=16`, `λ_ssim=0.2`, `λ_mask=0.1`, `λ_biomech=0.01`.

## Layout

```
physiosplat/
  gaussians.py    deformation.py   psgd.py        dbr.py
  saam.py         rasterizer.py    model.py       losses.py
  cameras.py      dataset.py       trainer.py     config.py    synthetic.py
scripts/   train.py, render.py
examples/  run_synthetic.py
tests/     unit + integration tests
```

## License

MIT — see [`LICENSE`](LICENSE).
