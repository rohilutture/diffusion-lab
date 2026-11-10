"""Train a denoiser under flow matching or VP diffusion.

    python -m diffusionlab.train --config configs/points_flow.yaml [--resume runs/x/ckpt.pt] [--max-steps N]

Per step: x0 ~ data, eps ~ N(0,I), t ~ path.sample_t, x_t = path.noisy, loss = MSE(model(x_t, t, y), path.target).
Labels are replaced by the null label with probability `cfg_drop` so one network serves guided and unguided
sampling. An EMA copy is what gets sampled and evaluated.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import build_dataset
from .io_utils import modality_of, save_samples
from .models import build_model
from .paths import PATHS
from .sampling import SAMPLERS, cfg_model_fn, sample


def infinite(loader):
    while True:
        yield from loader


def data_shape(model_cfg: dict, ds) -> tuple:
    return tuple(ds[0][0].shape)


def run(cfg: dict, resume: str | None = None, device: str | None = None, max_steps: int | None = None,
        log=print) -> list[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["train"]
    torch.manual_seed(tc.get("seed", 0))
    out = Path(cfg.get("out_dir", "runs/default"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    ds = build_dataset(cfg["data"])
    loader = DataLoader(ds, batch_size=tc["batch"], shuffle=True, drop_last=True, num_workers=tc.get("workers", 0),
                        persistent_workers=tc.get("workers", 0) > 0)
    model = build_model(cfg["model"]).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    path = PATHS[cfg["path"]]()
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("wd", 0.0))
    null = model.num_classes
    modality = modality_of(cfg["model"]["kind"])
    use_amp = device == "cuda" and tc.get("amp", True)
    step = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"{modality} / {cfg['path']} / {cfg['model']['kind']}: {n_params:.1f}M params, device {device}, amp {use_amp}")

    total = max_steps if max_steps is not None else tc["steps"]
    losses, it, t0 = [], infinite(loader), time.time()
    logf = (out / "log.jsonl").open("a")
    while step < total:
        x0, y = next(it)
        x0, y = x0.to(device, non_blocking=True), y.to(device)
        y = torch.where(torch.rand(len(y), device=device) < tc.get("cfg_drop", 0.1), torch.full_like(y, null), y)
        eps = torch.randn_like(x0)
        t = path.sample_t(len(x0), device)
        xt, target = path.noisy(x0, eps, t), path.target(x0, eps, t)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(xt, t, y)
        loss = (pred.float() - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip", 1.0))
        opt.step()
        decay = min(tc.get("ema", 0.9999), (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        losses.append(loss.item())
        if step % tc.get("log_every", 50) == 0 or step == total:
            rec = {"step": step, "loss": sum(losses[-50:]) / len(losses[-50:]), "grad_norm": float(gn),
                   "sec": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n"); logf.flush(); log(rec)
        if tc.get("sample_every") and step % tc["sample_every"] == 0:
            generate_preview(ema, path, cfg, ds, device, out / f"samples_step{step}")
        if tc.get("ckpt_every") and step % tc["ckpt_every"] == 0 or step == total:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg}, out / "ckpt.pt")
    return losses


@torch.no_grad()
def generate_preview(ema, path, cfg, ds, device, dest, n=16, nfe=20):
    modality = modality_of(cfg["model"]["kind"])
    shape = (n, *data_shape(cfg["model"], ds))
    y = torch.arange(n, device=device) % ema.num_classes
    sampler = "heun" if cfg["path"] == "flow" else "dpmpp2m"
    x, _ = sample(cfg_model_fn(ema, y, ema.num_classes, cfg.get("cfg_scale", 1.0)), path, shape, sampler, nfe, device,
                  generator=torch.Generator(device).manual_seed(0))
    save_samples(modality, x, dest, nrow=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    run(yaml.safe_load(Path(a.config).read_text()), a.resume, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
