"""Evaluate the trained procedural point-cloud and video models; writes results/eval_trained.json.

    python scripts/eval_trained.py

Points: sampler x NFE sweep, scored against held-out reference clouds (a different seed than training) with MMD-CD, COV-CD
and 1-NNA. The reference-vs-reference 1-NNA (an honest "indistinguishable" level) is reported next to it.
Video: motion statistics of generated vs real clips (see metrics.video_stats).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusionlab import metrics  # noqa: E402
from diffusionlab.data import MovingShapes, PointShapes  # noqa: E402
from diffusionlab.sample import load, sample_checkpoint  # noqa: E402

dev = "cuda"
out = {}
N_GEN = 256


def stack(ds, n):
    return torch.stack([ds[i][0] for i in range(n)])


# ---------------------------------------------------------------- points
model, cfg = load("runs/points_flow/ckpt.pt", dev)
ref = stack(PointShapes(n=2000, num_points=512, seed=123), 256).to(dev)         # held out: seed differs from training (0)
ref_a, ref_b = ref[:128], ref[128:]
out["points"] = {"ref_vs_ref_1nna": metrics.one_nna_cd(ref_a, ref_b), "ref_vs_ref_mmd": metrics.mmd_cd(ref_a, ref_b),
                 "n_gen": N_GEN, "n_ref": len(ref), "runs": []}
for sampler, nfe, cfg_scale in [("euler", 10, 1.5), ("euler", 50, 1.5), ("heun", 10, 1.5), ("heun", 20, 1.5),
                                ("heun", 20, 1.0), ("heun", 20, 3.0)]:
    t = time.time()
    with torch.no_grad():
        gen, used = sample_checkpoint(model, cfg, (512, 3), N_GEN, sampler, nfe, cfg_scale, 0, dev)
    torch.cuda.synchronize()
    gen = gen.float()
    row = {"sampler": sampler, "nfe": nfe, "nfe_used": used, "cfg": cfg_scale, "finite": bool(torch.isfinite(gen).all()),
           "mmd_cd": metrics.mmd_cd(gen, ref), "cov_cd": metrics.cov_cd(gen, ref), "one_nna": metrics.one_nna_cd(gen, ref),
           "sample_s": round(time.time() - t, 2)}
    out["points"]["runs"].append(row)
    print(row, flush=True)

# ---------------------------------------------------------------- video
model, cfg = load("runs/video_flow/ckpt.pt", dev)
real = torch.stack([MovingShapes(n=500, frames=16, size=32, seed=123)[i][0] for i in range(128)]).to(dev)
out["video"] = {"real": metrics.video_stats(real), "runs": []}
for sampler, nfe, cfg_scale in [("euler", 10, 1.5), ("heun", 20, 1.5), ("heun", 20, 3.0)]:
    with torch.no_grad():
        gen, used = sample_checkpoint(model, cfg, (1, 16, 32, 32), 128, sampler, nfe, cfg_scale, 0, dev)
    gen = gen.float()
    row = {"sampler": sampler, "nfe": nfe, "nfe_used": used, "cfg": cfg_scale, "finite": bool(torch.isfinite(gen).all()),
           "range": [float(gen.min()), float(gen.max())], **metrics.video_stats(gen.clamp(-1, 1))}
    out["video"]["runs"].append(row)
    print(row, flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/eval_trained.json").write_text(json.dumps(out, indent=1))
print("real video:", out["video"]["real"], "ref-vs-ref 1nna:", out["points"]["ref_vs_ref_1nna"])
