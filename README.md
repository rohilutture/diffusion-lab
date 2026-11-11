# Diffusion lab: image, video, and 3D point-cloud generation

Status: **32 CPU tests pass. On one RTX 5090: two models trained on procedural data (point clouds and toy video, 60,000
steps each) and a CIFAR-10 flow-matching vs VP-diffusion comparison (DiT-S/2, 100,000 steps each, one seed) scored with
FID at 10,000 samples. The FID numbers come from my own pipeline (torch-fidelity Inception, 10k generated samples) and are
NOT comparable to published CIFAR-10 FIDs.**

## CIFAR-10 FID vs number of network evaluations (`results/fid_flow.csv`, `results/fid_vp.csv`; unguided, EMA weights)

Same network, data, optimizer and 100,000 steps (batch 128, lr 2e-4) for both probability paths. n=10,000 samples, one
training seed, one sampling run per cell (lower is better).

| NFE | Flow + Euler | Flow + Heun | VP + DDIM | VP + DPM-Solver++(2M) |
|---|---|---|---|---|
| 10 | 22.26 | 19.73 (9 NFE) | 22.22 | 18.80 |
| 20 | 17.26 | 16.87 (19) | 18.35 | 17.03 |
| 50 | 14.14 | 14.32 (49) | 15.65 | 15.09 |
| 100 | 13.31 | 13.31 (99) | 14.46 | 14.19 |

- Flow matching reached lower FID than VP diffusion at 50 and 100 NFE (13.3 vs 14.2-14.5 at 100), by about 1 FID.
  The higher-order solvers help mainly at very low NFE (about 3 FID at 10 NFE) and give no gain by 50-100.
- With one seed and one sampling run per cell, differences under about 1 FID should not be read as real; the 100-NFE
  flow vs VP gap is near that size. The ordering is suggestive, not established.
- Absolute values are well above published DiT/flow numbers: the model is small and 10k-sample FID is biased upward
  versus 50k. Compare cells with each other only.

## Results (`scripts/eval_trained.py`, `results/eval_trained.json`; EMA weights, one training seed)

Training: point-cloud DiT (9.8M params) 60,000 steps in about 32 minutes at batch 64; toy video model 60,000 steps.

**Point clouds** (5 procedural solids x 512 points; 256 generated vs 256 held-out reference clouds from a different seed):

| Sampler | Evals | CFG | MMD-CD (lower better) | COV-CD (higher better) | 1-NNA (0.5 = indistinguishable) |
|---|---|---|---|---|---|
| Euler | 10 | 1.5 | 0.0226 | 0.06 | 0.98 |
| Euler | 50 | 1.5 | 0.0080 | 0.47 | **0.61** |
| Heun | 9 | 1.5 | 0.0158 | 0.22 | 0.88 |
| Heun | 19 | 1.5 | 0.0083 | 0.51 | 0.78 |
| Heun | 19 | 1.0 | 0.0082 | 0.52 | 0.78 |
| Heun | 19 | 3.0 | 0.0098 | 0.43 | 0.85 |

For scale, two disjoint sets of real reference clouds score 1-NNA 0.457 and MMD 0.0079 against each other (128 each, so only
a rough floor). Reading: with enough steps the generated clouds reach the reference's MMD but a nearest-neighbour test can
still tell them apart (best 0.61, not 0.5); at 10 steps Euler is clearly poor. At matched cost Heun beat Euler at about
10 evaluations (0.88 vs 0.98) but **Euler with 50 steps had the best 1-NNA**, so "higher-order sampler is better" is not shown
beyond the low-step regime. Stronger classifier-free guidance (3.0) hurt; 1.0 and 1.5 were indistinguishable. Single sample set
per row, no confidence intervals.

**Toy video** (one bouncing shape, 16x32x32; 128 generated vs 128 real clips): all samples finite and almost never blank
(blank-frame fraction 0 to 0.1%); object area is stable over time (area CV 0.002 vs 0.0 for real). But the generated centroid
motion is **jerkier than real**: median centroid acceleration 0.86 px/frame^2 (Heun, 19 evals) vs 0.5 for real clips
(Euler 10 evals: 0.96). So the model captures appearance but not smooth constant-velocity motion; this is a limitation, not a win.

One small codebase that trains the same style of transformer denoiser (DiT with adaLN-Zero) on three modalities
under either of two probability paths, and compares samplers on a controlled footing.

| Modality | Model | Data | Metrics |
|---|---|---|---|
| Image 32x32, class-conditional | `DiT` (S/2, ~33M params) | CIFAR-10 (you download it) | FID vs NFE |
| Video 16x32x32, class-conditional | `VideoDiT`, factorized space-time attention | `MovingShapes`, procedural bouncing shapes, no download | physics stats (area stability, centroid acceleration, blank frames) |
| 3D point clouds, class-conditional | `PointDiT`, permutation-equivariant | `PointShapes`, procedural solids, no download; `ShapeNetPoints` loader for clouds you supply | MMD-CD, COV-CD, 1-NNA-CD |

| Path | Training target | Samplers (ODE, deterministic) |
|---|---|---|
| `flow` (rectified flow / flow matching) | `v = eps - x0`, logit-normal t | Euler, Heun |
| `vp` (cosine VP diffusion, v-prediction) | `v = a*eps - s*x0` | DDIM, DPM-Solver++(2M) |

Classifier-free guidance is built in (label dropout in training, `cfg_scale` at sampling). An EMA copy is sampled.

## What is verified (CPU, `pytest`)

- **Sampler correctness against closed-form oracles.** For Gaussian data the optimal network output and the
  exact noise-to-data map (`x0 = m + sd*z`) are known analytically, so each sampler's discretization error is
  measured directly with no training: error shrinks as steps grow for all four samplers, Heun beats Euler and
  DPM-Solver++(2M) beats DDIM at equal NFE, NFE accounting is exact, and the oracle itself is checked against a
  Monte Carlo estimate of `E[target | x_t]` for both paths.
- Model shapes, adaLN-Zero zero-init (untrained output is exactly 0), permutation equivariance of `PointDiT`,
  and that `VideoDiT` really mixes information across frames.
- Datasets are deterministic per index, and the procedural geometry is checked against the closed-form surface
  equations (sphere radius, cube, torus, ellipsoid).
- Metrics: Chamfer is zero for identical sets and symmetric; MMD/COV/1-NNA separate matching from mismatched
  distributions; the video physics stat is sensitive to destroyed temporal order.
- Tiny end-to-end runs: loss falls, checkpoint reloads, sampling (with guidance) stays finite, resume continues
  the step counter.

## Experiments (video, point-cloud and CIFAR-10 flow-vs-VP FID done above; guidance sweep not run)

1. **Flow vs diffusion, same network, data and budget** (`configs/cifar_flow.yaml`, `configs/cifar_vp.yaml`):
   FID vs NFE for Euler/Heun (flow) and DDIM/DPM-Solver++ (VP) at NFE 10/20/50/100 (`eval_fid.py`). The
   question is how few network evaluations each family needs; the analytic tests give the expected ordering,
   the trained-model result may differ.
2. **Guidance sweep**: FID and diversity vs `cfg_scale`.
3. **Video** (`configs/video_flow.yaml`): does the generated object keep its area and move at near-constant
   velocity? Compare generated stats to real, per class.
4. **Points** (`configs/points_flow.yaml`): MMD/COV/1-NNA vs real, then swap in real ShapeNet clouds.

## Run

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m pytest -q                                     # CPU, ~12 s

# when the GPU is free (procedural data needs no download):
.venv\Scripts\python.exe -m diffusionlab.train --config configs/points_flow.yaml
.venv\Scripts\python.exe -m diffusionlab.train --config configs/video_flow.yaml
.venv\Scripts\python.exe -m diffusionlab.sample --ckpt runs/points_flow/ckpt.pt --sampler heun --nfe 20 --cfg-scale 2 --out out/pts

# CIFAR-10 + FID need downloads you must allow: put CIFAR-10 under data/, install torchmetrics + torch-fidelity
.venv\Scripts\python.exe -m diffusionlab.train --config configs/cifar_flow.yaml
.venv\Scripts\python.exe -m diffusionlab.eval_fid --ckpt runs/cifar_flow/ckpt.pt --n 10000 --nfes 10,20,50,100
```

## Honest scope limits

- **Video here is a toy**: one bouncing shape on a 32x32 canvas. It tests temporal modeling mechanics and has a
  checkable physics metric, but it is not a video benchmark; do not call it video-generation SOTA or compare to
  FVD numbers from real-video papers. Real video (UCF-101 etc.) and FVD are future work.
- **3D here means point clouds**, not meshes, NeRFs, or Gaussian splats. The procedural solids are easy by
  design; ShapeNet results would need real data (license required, not included).
- FID at 10k samples is biased upward vs the usual 50k and the DiT-S/2 budget is small; report the sample count
  and compare runs against each other, not against published numbers.
- Everything here is single-seed unless more seeds are run.
- Not implemented: latent diffusion/VAE, text conditioning, SDE/stochastic samplers, distillation, FVD.

## Phrasing rules (once there are results)

- Name the data (procedural vs real), the sample count for FID/COV, the NFE, the sampler, the guidance scale, and
  "single run".
- Claim "implemented and compared flow matching and VP diffusion with Euler/Heun/DDIM/DPM-Solver++ samplers, with
  samplers validated against closed-form Gaussian oracles", not "state-of-the-art" anything.
- Frame video and 3D as small-scale generative models on procedural data until real-data results exist.
