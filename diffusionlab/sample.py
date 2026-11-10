"""Sample from a trained checkpoint (EMA weights).

    python -m diffusionlab.sample --ckpt runs/points_flow/ckpt.pt --sampler heun --nfe 20 --cfg-scale 2.0 --n 16 --out out/pts
"""
from __future__ import annotations

import argparse

import torch

from .io_utils import modality_of, save_samples
from .models import build_model
from .paths import PATHS
from .sampling import SAMPLERS, cfg_model_fn, sample


def load(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location=device)
    cfg = ck["cfg"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["ema"])
    return model.eval(), cfg


def sample_checkpoint(model, cfg, data_shape, n, sampler, nfe, cfg_scale, seed, device, labels=None, shift=1.0):
    path = PATHS[cfg["path"]]()
    y = labels if labels is not None else torch.arange(n, device=device) % model.num_classes
    g = torch.Generator(device).manual_seed(seed)
    return sample(cfg_model_fn(model, y, model.num_classes, cfg_scale), path, (n, *data_shape), sampler, nfe, device,
                  generator=g, shift=shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sampler")
    ap.add_argument("--nfe", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/samples")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load(a.ckpt, device)
    from .data import build_dataset

    shape = tuple(build_dataset(cfg["data"])[0][0].shape)
    sampler = a.sampler or SAMPLERS[cfg["path"]][-1]
    x, nfe = sample_checkpoint(model, cfg, shape, a.n, sampler, a.nfe, a.cfg_scale, a.seed, device)
    save_samples(modality_of(cfg["model"]["kind"]), x, a.out)
    print(f"{a.n} samples, {sampler}, {nfe} NFE, cfg {a.cfg_scale} -> {a.out}")


if __name__ == "__main__":
    main()
