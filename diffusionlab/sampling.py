"""Samplers and classifier-free guidance.

model_fn(x, t) -> v is the (already guided) network output for a batch at per-sample times t in [0,1].
All samplers are deterministic ODE solvers going from t=1 (noise) to t=0 (data):

  flow path:  euler (1 NFE/step), heun (2 NFE/step except the last: 2*steps-1)
  vp path:    ddim (1 NFE/step), dpmpp2m = DPM-Solver++(2M), data prediction, multistep (1 NFE/step)

`sample(..., nfe=N)` picks steps so the number of network evaluations (NFE) is N (heun: N must be odd-ish; it
uses (N+1)//2 steps and reports the true NFE).
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from .paths import FlowPath, VPPath

ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SAMPLERS = {"flow": ("euler", "heun"), "vp": ("ddim", "dpmpp2m")}


def cfg_model_fn(model, y: torch.Tensor, null_label: int, scale: float) -> ModelFn:
    """Classifier-free guidance: v = v_u + scale * (v_c - v_u). scale == 1 is plain conditional sampling."""

    def fn(x, t):
        if scale == 1.0:
            return model(x, t, y)
        v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, null_label)]))
        vc, vu = v.chunk(2)
        return vu + scale * (vc - vu)

    return fn


def time_grid(steps: int, device, shift: float = 1.0) -> torch.Tensor:
    """Descending grid 1 -> 0 with `steps` intervals. shift > 1 spends more steps at high noise (SD3-style)."""
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float64)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


class _Counter:
    def __init__(self, fn: ModelFn):
        self.fn, self.n = fn, 0

    def __call__(self, x, t):
        self.n += 1
        return self.fn(x, t)


def _tvec(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)


@torch.no_grad()
def euler(f, x, ts):
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        x = x + float(tn - t) * f(x, _tvec(t, x))
    return x


@torch.no_grad()
def heun(f, x, ts):
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        dt = float(tn - t)
        v1 = f(x, _tvec(t, x))
        xe = x + dt * v1
        if float(tn) == 0.0:  # last step: Euler (skips an evaluation at t=0)
            x = xe
        else:
            x = x + 0.5 * dt * (v1 + f(xe, _tvec(tn, x)))
    return x


@torch.no_grad()
def ddim(f, x, ts, path: VPPath):
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        v = f(x, _tvec(t, x))
        x0, eps = path.x0_eps(x, _tvec(t, x), v)
        an, sn = math.cos(0.5 * math.pi * float(tn)), math.sin(0.5 * math.pi * float(tn))
        x = an * x0 + sn * eps
    return x


@torch.no_grad()
def dpmpp2m(f, x, ts, path: VPPath):
    x0_prev, h_prev = None, None
    for i in range(len(ts) - 1):
        t, tn = float(ts[i]), float(ts[i + 1])
        a, s = math.cos(0.5 * math.pi * t), math.sin(0.5 * math.pi * t)
        an, sn = math.cos(0.5 * math.pi * tn), math.sin(0.5 * math.pi * tn)
        v = f(x, _tvec(ts[i], x))
        x0 = a * x - s * v
        if sn == 0.0:  # final step to t=0: the data prediction is the answer
            return x0
        h = math.log(an / sn) - math.log(a / s)
        if x0_prev is None:
            d = x0
        else:
            r = h_prev / h
            d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0_prev
        x = (sn / s) * x - an * math.expm1(-h) * d
        x0_prev, h_prev = x0, h
    return x


def sample(model_fn: ModelFn, path, shape, sampler: str, nfe: int, device, generator=None, dtype=torch.float32,
           shift: float = 1.0, x_init: torch.Tensor | None = None):
    """Returns (samples, actual_nfe)."""
    if sampler not in SAMPLERS[path.name]:
        raise ValueError(f"{sampler} is not valid for the {path.name} path (use {SAMPLERS[path.name]})")
    steps = (nfe + 1) // 2 if sampler == "heun" else nfe
    ts = time_grid(steps, device, shift)
    x = x_init if x_init is not None else torch.randn(shape, device=device, dtype=dtype, generator=generator)
    f = _Counter(model_fn)
    if sampler == "euler":
        out = euler(f, x, ts)
    elif sampler == "heun":
        out = heun(f, x, ts)
    elif sampler == "ddim":
        out = ddim(f, x, ts, path)
    else:
        out = dpmpp2m(f, x, ts, path)
    return out, f.n
