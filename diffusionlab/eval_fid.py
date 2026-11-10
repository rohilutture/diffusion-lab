"""FID vs NFE for every sampler valid for the checkpoint's path (CIFAR-10).

    python -m diffusionlab.eval_fid --ckpt runs/cifar_flow/ckpt.pt --n 10000 --nfes 10,20,50,100 --cfg-scale 1.0

Needs `torchmetrics` + `torch-fidelity` and downloads Inception weights (and CIFAR-10 if absent) -- run it
only when that is acceptable. Writes results/fid_<path>.csv. FID with 10k samples is biased upward relative to
the usual 50k; compare runs against each other, and say the sample count when reporting.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .data import cifar10
from .metrics import fid
from .sample import load, sample_checkpoint
from .sampling import SAMPLERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--nfes", default="10,20,50,100")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--download", action="store_true", help="allow downloading CIFAR-10")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load(a.ckpt, device)
    train = cifar10(a.data_root, train=True, download=a.download, flip=False)
    real = torch.stack([train[i][0] for i in range(a.n)])
    rows = []
    for sampler in SAMPLERS[cfg["path"]]:
        for nfe in [int(v) for v in a.nfes.split(",")]:
            fakes, used = [], 0
            for s in range(0, a.n, a.batch):
                x, used = sample_checkpoint(model, cfg, (3, 32, 32), min(a.batch, a.n - s), sampler, nfe, a.cfg_scale,
                                            seed=s, device=device)
                fakes.append(x.cpu())
            score = fid(real, torch.cat(fakes), device)
            rows.append([cfg["path"], sampler, nfe, used, a.n, a.cfg_scale, f"{score:.3f}"])
            print(rows[-1])
    Path("results").mkdir(exist_ok=True)
    with open(f"results/fid_{cfg['path']}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "sampler", "nfe_requested", "nfe_used", "n_samples", "cfg_scale", "fid"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
