"""Probability paths. Both predict a velocity-like target "v" so the same network trains under either.

FlowPath (rectified flow / flow matching):  x_t = (1-t) x0 + t eps,   v = eps - x0,   t: 1 = noise, 0 = data
VPPath   (cosine VP diffusion, v-pred):     x_t = a x0 + s eps,       v = a eps - s x0,  a=cos(pi t/2), s=sin(pi t/2)

For VP, x0 = a x_t - s v and eps = s x_t + a v (no division by a, so t=1 where a~0 is fine).
"""
from __future__ import annotations

import math

import torch


def _bc(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (x.dim() - 1)))


class FlowPath:
    name = "flow"

    def sample_t(self, n: int, device) -> torch.Tensor:
        # logit-normal: concentrates training on mid-noise levels (Esser et al. 2024)
        return torch.sigmoid(torch.randn(n, device=device))

    def noisy(self, x0, eps, t):
        tt = _bc(t, x0)
        return (1 - tt) * x0 + tt * eps

    def target(self, x0, eps, t):
        return eps - x0

    def x0_eps(self, x, t, v):
        tt = _bc(t, x)
        return x - tt * v, x + (1 - tt) * v


class VPPath:
    name = "vp"

    @staticmethod
    def alpha(t):
        return torch.cos(0.5 * math.pi * t)

    @staticmethod
    def sigma(t):
        return torch.sin(0.5 * math.pi * t)

    def sample_t(self, n: int, device) -> torch.Tensor:
        return torch.rand(n, device=device)

    def noisy(self, x0, eps, t):
        return _bc(self.alpha(t), x0) * x0 + _bc(self.sigma(t), x0) * eps

    def target(self, x0, eps, t):
        return _bc(self.alpha(t), x0) * eps - _bc(self.sigma(t), x0) * x0

    def x0_eps(self, x, t, v):
        a, s = _bc(self.alpha(t), x), _bc(self.sigma(t), x)
        return a * x - s * v, s * x + a * v


PATHS = {"flow": FlowPath, "vp": VPPath}
