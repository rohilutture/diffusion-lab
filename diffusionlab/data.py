"""Datasets.

Image : CIFAR-10 via torchvision (you must allow the download; nothing is fetched at import time).
Video : MovingShapes, procedural bouncing shapes (no download). Class = shape. Physics is known, so
        generated videos can be checked against it (see metrics.video_stats).
Points: PointShapes, procedural point clouds of 5 solids (no download), and ShapeNetPoints, a loader for
        point clouds you supply yourself (ShapeNet needs a license; this repo does not ship or fetch it).

All items are float32 in [-1, 1] (points: roughly unit sphere) and deterministic given (seed, index).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------ images
def cifar10(root: str = "data", train: bool = True, download: bool = False, flip: bool = True):
    import torchvision
    import torchvision.transforms as T

    tf = [T.RandomHorizontalFlip()] if flip else []
    tf += [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return torchvision.datasets.CIFAR10(root, train=train, download=download, transform=T.Compose(tf))


# ------------------------------------------------------------------ video
SHAPE_NAMES = ("square", "circle", "cross")


class MovingShapes(Dataset):
    """One shape bouncing inside a size x size frame at constant speed. Returns (video [1,T,S,S], class)."""

    def __init__(self, n=10000, frames=16, size=32, seed=0, speed=(0.6, 1.6)):
        self.n, self.T, self.S, self.seed, self.speed = n, frames, size, seed, speed
        ax = np.arange(size, dtype=np.float32)
        self.yy, self.xx = np.meshgrid(ax, ax, indexing="ij")

    def __len__(self):
        return self.n

    def _mask(self, cls, px, py):
        dx, dy = np.abs(self.xx - px), np.abs(self.yy - py)
        if cls == 0:
            return (dx <= 3.5) & (dy <= 3.5)
        if cls == 1:
            return dx**2 + dy**2 <= 4.2**2
        return ((dx <= 1.0) & (dy <= 4.0)) | ((dy <= 1.0) & (dx <= 4.0))

    def trajectory(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        cls = int(rng.integers(0, 3))
        margin = 5.0
        pos = rng.uniform(margin, self.S - 1 - margin, size=2)
        ang, spd = rng.uniform(0, 2 * math.pi), rng.uniform(*self.speed)
        vel = spd * np.array([math.cos(ang), math.sin(ang)])
        traj = []
        for _ in range(self.T):
            traj.append(pos.copy())
            pos = pos + vel
            for d in range(2):  # reflect at the walls
                if pos[d] < margin:
                    pos[d], vel[d] = 2 * margin - pos[d], -vel[d]
                elif pos[d] > self.S - 1 - margin:
                    pos[d], vel[d] = 2 * (self.S - 1 - margin) - pos[d], -vel[d]
        return cls, np.array(traj)

    def __getitem__(self, idx):
        cls, traj = self.trajectory(idx)
        v = np.stack([self._mask(cls, p[0], p[1]) for p in traj]).astype(np.float32)
        return torch.from_numpy(v * 2 - 1).unsqueeze(0), cls


# ------------------------------------------------------------------ point clouds
POINT_CLASSES = ("sphere", "cube", "torus", "cylinder", "ellipsoid")


def _unit_dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample_solid(cls: int, n: int, rng) -> np.ndarray:
    if cls == 0:  # sphere r=0.9
        return 0.9 * _unit_dirs(rng, n)
    if cls == 1:  # cube surface, half-size 0.55
        face = rng.integers(0, 6, size=n)
        p = rng.uniform(-0.55, 0.55, size=(n, 3))
        axis, sign = face // 2, np.where(face % 2 == 0, -0.55, 0.55)
        p[np.arange(n), axis] = sign
        return p
    if cls == 2:  # torus R=0.65, r=0.25
        u, v = rng.uniform(0, 2 * math.pi, size=(2, n))
        R, r = 0.65, 0.25
        return np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
    if cls == 3:  # cylinder r=0.45, height 1.2 (area-weighted sides vs caps)
        r, hgt = 0.45, 1.2
        a_side, a_cap = 2 * math.pi * r * hgt, 2 * math.pi * r * r
        side = rng.uniform(size=n) < a_side / (a_side + a_cap)
        th = rng.uniform(0, 2 * math.pi, size=n)
        rad = np.where(side, r, r * np.sqrt(rng.uniform(size=n)))
        z = np.where(side, rng.uniform(-hgt / 2, hgt / 2, size=n), np.where(rng.uniform(size=n) < 0.5, -hgt / 2, hgt / 2))
        return np.stack([rad * np.cos(th), rad * np.sin(th), z], 1)
    return _unit_dirs(rng, n) * np.array([0.9, 0.55, 0.35])  # ellipsoid


class PointShapes(Dataset):
    """Returns (points [N,3] float32, class). Small per-sample scale jitter keeps it from being trivially one point set."""

    def __init__(self, n=10000, num_points=512, seed=0, jitter=0.15):
        self.n, self.N, self.seed, self.jitter = n, num_points, seed, jitter

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        cls = int(rng.integers(0, len(POINT_CLASSES)))
        pts = sample_solid(cls, self.N, rng) * rng.uniform(1 - self.jitter, 1 + self.jitter)
        return torch.from_numpy(pts.astype(np.float32)), cls


class ShapeNetPoints(Dataset):
    """Point clouds you provide: a directory of .npy files, each [M, 3] (M >= num_points). Every cloud is
    centered, scaled to fit the unit sphere, and randomly subsampled to num_points. Single class (label 0)."""

    def __init__(self, root: str, num_points=2048, seed=0):
        self.files = sorted(Path(root).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"no .npy point clouds under {root}")
        self.N, self.seed = num_points, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = np.load(self.files[idx]).astype(np.float32)
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        pts = pts[rng.choice(len(pts), self.N, replace=len(pts) < self.N)]
        pts = pts - pts.mean(0, keepdims=True)
        pts = pts / np.linalg.norm(pts, axis=1).max()
        return torch.from_numpy(pts), 0


def build_dataset(cfg: dict):
    kind = cfg["kind"]
    args = {k: v for k, v in cfg.items() if k != "kind"}
    return {"cifar10": cifar10, "moving_shapes": MovingShapes, "point_shapes": PointShapes,
            "shapenet_points": ShapeNetPoints}[kind](**args)
