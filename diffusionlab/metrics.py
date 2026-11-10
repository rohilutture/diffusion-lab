"""Evaluation metrics.

Points : Chamfer distance (squared, symmetric mean), and the standard set-level generation metrics from
         PointFlow/PVD: MMD-CD (lower is better), COV-CD (higher is better), 1-NNA-CD (ideal 0.5; 1.0 = trivially separable).
Video  : physics-based stats for MovingShapes (no pretrained network needed): does the object keep its area and
         move at ~constant velocity between bounces? Compare generated vs real, never to an absolute target.
Image  : FID wrapper (needs torchmetrics + torch-fidelity and downloads Inception weights on first use).
"""
from __future__ import annotations

import torch


# ------------------------------------------------------------------ points
def pairwise_chamfer(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A [n,N,3], B [m,N,3] -> [n,m] Chamfer distance (mean squared nearest-neighbour distance, both directions)."""
    out = torch.empty(A.shape[0], B.shape[0], device=A.device)
    for i in range(A.shape[0]):
        d = torch.cdist(A[i : i + 1].expand(B.shape[0], -1, -1), B) ** 2  # [m,N,N]
        out[i] = d.min(2).values.mean(1) + d.min(1).values.mean(1)
    return out


def mmd_cd(gen: torch.Tensor, ref: torch.Tensor) -> float:
    return pairwise_chamfer(ref, gen).min(1).values.mean().item()


def cov_cd(gen: torch.Tensor, ref: torch.Tensor) -> float:
    nearest_ref = pairwise_chamfer(gen, ref).argmin(1)
    return nearest_ref.unique().numel() / ref.shape[0]


def one_nna_cd(gen: torch.Tensor, ref: torch.Tensor) -> float:
    """Leave-one-out 1-nearest-neighbour accuracy at telling gen from ref. 0.5 means indistinguishable."""
    both = torch.cat([gen, ref])
    labels = torch.cat([torch.ones(len(gen)), torch.zeros(len(ref))]).to(both.device)
    D = pairwise_chamfer(both, both)
    D.fill_diagonal_(float("inf"))
    return (labels[D.argmin(1)] == labels).float().mean().item()


# ------------------------------------------------------------------ video
def video_stats(videos: torch.Tensor) -> dict:
    """videos [B,1,T,H,W] in [-1,1]. Returns medians over the batch of:
    area_cv   : std/mean of the object's pixel mass over time (a rigid shape keeps constant area)
    accel     : mean |second difference| of the intensity-weighted centroid in px/frame^2 (bounces add spikes,
                so compare against real data rather than expecting ~0)
    blank_frac: fraction of frames with almost no object (generation failure)"""
    v = ((videos[:, 0] + 1) / 2).clamp(0, 1)  # [B,T,H,W]
    B, T, H, W = v.shape
    mass = v.sum((2, 3))
    ys = torch.arange(H, dtype=v.dtype, device=v.device).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=v.dtype, device=v.device).view(1, 1, 1, W)
    cy, cx = (v * ys).sum((2, 3)) / mass.clamp(min=1e-6), (v * xs).sum((2, 3)) / mass.clamp(min=1e-6)
    c = torch.stack([cy, cx], -1)  # [B,T,2]
    accel = (c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]).norm(dim=-1).mean(1)
    return {
        "area_cv": (mass.std(1) / mass.mean(1).clamp(min=1e-6)).median().item(),
        "accel": accel.median().item(),
        "blank_frac": (mass < 5).float().mean().item(),
    }


# ------------------------------------------------------------------ images
def fid(real: torch.Tensor, fake: torch.Tensor, device="cuda", batch=500) -> float:
    """real, fake: [N,3,H,W] in [-1,1]. Requires `pip install torchmetrics torch-fidelity`; downloads Inception weights."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    m = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for src, is_real in ((real, True), (fake, False)):
        for i in range(0, len(src), batch):
            m.update(((src[i : i + batch].to(device) + 1) / 2).clamp(0, 1), real=is_real)
    return m.compute().item()
