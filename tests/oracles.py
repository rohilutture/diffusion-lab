"""Closed-form optimal 'networks' for Gaussian data N(m, sd^2 I), used to test samplers without training.

For such data the probability-flow ODE is linear, so the exact noise->data map is x0 = m + sd * z
(z = the starting noise at t=1). Any deviation of a sampler's output from that is pure discretization error.
"""
import math

import torch


class FlowOracle:
    def __init__(self, m, sd):
        self.m, self.sd = torch.as_tensor(m, dtype=torch.float32), float(sd)

    def __call__(self, x, t):
        t = t.view(-1, 1)
        V = (1 - t) ** 2 * self.sd**2 + t**2
        xm = x - (1 - t) * self.m
        return (t - (1 - t) * self.sd**2) / V * xm - self.m


class VPOracle:
    def __init__(self, m, sd):
        self.m, self.sd = torch.as_tensor(m, dtype=torch.float32), float(sd)

    def __call__(self, x, t):
        t = t.view(-1, 1)
        a, s = torch.cos(0.5 * math.pi * t), torch.sin(0.5 * math.pi * t)
        V = a**2 * self.sd**2 + s**2
        xm = x - a * self.m
        eps_hat = s / V * xm
        x0_hat = self.m + a * self.sd**2 / V * xm
        return a * eps_hat - s * x0_hat

    def exact(self, z):
        return self.m + self.sd * z


def exact_flow(m, sd, z):
    return torch.as_tensor(m, dtype=torch.float32) + sd * z
