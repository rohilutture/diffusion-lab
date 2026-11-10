"""Transformer denoisers with adaLN-Zero conditioning (DiT, Peebles & Xie 2023): image DiT, factorized
space-time video DiT, and a point-cloud DiT. All share forward(x, t, y) -> v with t in [0,1] and class labels y
(label == num_classes is the 'null' label used for classifier-free guidance).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim // 2))
    out = pos.double()[:, None] * omega[None]
    return torch.cat([out.sin(), out.cos()], dim=1).float()


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    assert dim % 4 == 0
    gy, gx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, gy.flatten()), sincos_1d(dim // 2, gx.flatten())], dim=1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Transformer block whose shift/scale/gate come from the conditioning vector, zero-initialized
    so every block starts as the identity."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), sh1, sc1))
        return x + g2.unsqueeze(1) * self.mlp(modulate(self.n2(x), sh2, sc2))


class Cond(nn.Module):
    """Timestep + class embedding -> conditioning vector c."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim, self.num_classes = dim, num_classes
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_emb = nn.Embedding(num_classes + 1, dim)  # last index = null label

    def forward(self, t, y):
        return self.t_mlp(timestep_embedding(t, 256)) + self.y_emb(y)


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, out_dim)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), sh, sc))


def _unpatchify(h, g, p, C):
    B = h.shape[0]
    return h.view(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)


class DiT(nn.Module):
    """Image DiT. x: [B, C, H, W]."""

    def __init__(self, img_size=32, patch=2, in_ch=3, dim=384, depth=12, heads=6, num_classes=10):
        super().__init__()
        self.p, self.C, self.g = patch, in_ch, img_size // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.register_buffer("pos", sincos_2d(dim, self.g)[None], persistent=False)
        self.cond = Cond(dim, num_classes)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final = FinalLayer(dim, patch * patch * in_ch)
        self.num_classes = num_classes

    def forward(self, x, t, y):
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.cond(t, y)
        for b in self.blocks:
            h = b(h, c)
        return _unpatchify(self.final(h, c), self.g, self.p, self.C)


class VideoDiT(nn.Module):
    """Factorized space-time DiT: each layer is a spatial block (attention within a frame) followed by a
    temporal block (attention across frames at each spatial position). x: [B, C, T, H, W]."""

    def __init__(self, frames=16, img_size=32, patch=4, in_ch=1, dim=256, depth=6, heads=4, num_classes=3):
        super().__init__()
        self.T, self.p, self.C, self.g = frames, patch, in_ch, img_size // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.register_buffer("pos_s", sincos_2d(dim, self.g)[None], persistent=False)
        self.register_buffer("pos_t", sincos_1d(dim, torch.arange(frames))[None, :, None], persistent=False)
        self.cond = Cond(dim, num_classes)
        self.spatial = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.temporal = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final = FinalLayer(dim, patch * patch * in_ch)
        self.num_classes = num_classes

    def forward(self, x, t, y):
        B, C, T, H, W = x.shape
        L = self.g * self.g
        h = self.embed(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))  # [B*T, D, g, g]
        D = h.shape[1]
        h = h.flatten(2).transpose(1, 2).reshape(B, T, L, D) + self.pos_s.unsqueeze(1) + self.pos_t
        c = self.cond(t, y)
        c_s, c_t = c.repeat_interleave(T, 0), c.repeat_interleave(L, 0)
        for sb, tb in zip(self.spatial, self.temporal):
            h = sb(h.reshape(B * T, L, D), c_s).view(B, T, L, D)
            h = tb(h.permute(0, 2, 1, 3).reshape(B * L, T, D), c_t).view(B, L, T, D).permute(0, 2, 1, 3)
        out = self.final(h.reshape(B * T, L, D), c_s)  # per-frame tokens
        out = _unpatchify(out, self.g, self.p, self.C).view(B, T, self.C, H, W)
        return out.permute(0, 2, 1, 3, 4)


class PointDiT(nn.Module):
    """Point-cloud DiT: every point is a token; attention over all points makes it permutation-equivariant.
    x: [B, N, 3]."""

    def __init__(self, dim=256, depth=8, heads=4, num_classes=5):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond = Cond(dim, num_classes)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final = FinalLayer(dim, 3)
        self.num_classes = num_classes

    def forward(self, x, t, y):
        h = self.embed(x)
        c = self.cond(t, y)
        for b in self.blocks:
            h = b(h, c)
        return self.final(h, c)


def build_model(cfg: dict) -> nn.Module:
    kind = cfg["kind"]
    args = {k: v for k, v in cfg.items() if k != "kind"}
    return {"dit": DiT, "video_dit": VideoDiT, "point_dit": PointDiT}[kind](**args)
