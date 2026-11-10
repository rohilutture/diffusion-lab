"""Saving samples: image grids (PNG), videos (GIF), point clouds (.npy + .ply)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def modality_of(model_kind: str) -> str:
    return {"dit": "image", "video_dit": "video", "point_dit": "points"}[model_kind]


def save_samples(modality: str, x: torch.Tensor, path: str | Path, nrow: int = 8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if modality == "image":
        import torchvision.utils as vu

        vu.save_image(((x + 1) / 2).clamp(0, 1), path.with_suffix(".png"), nrow=nrow)
    elif modality == "video":
        import imageio.v2 as imageio

        v = ((x.clamp(-1, 1) + 1) / 2 * 255).byte()[:, 0]  # [B,T,H,W]
        n = min(len(v), nrow)
        mosaic = torch.cat(list(v[:n]), dim=-1)  # side by side -> [T,H,n*W]
        imageio.mimsave(path.with_suffix(".gif"), [f.numpy() for f in mosaic], duration=0.12, loop=0)
    elif modality == "points":
        np.save(path.with_suffix(".npy"), x.numpy())
        for i in range(min(len(x), 4)):  # a few as .ply for MeshLab / Open3D
            write_ply(path.with_name(f"{path.stem}_{i}.ply"), x[i].numpy())
    else:
        raise ValueError(modality)


def write_ply(path, pts: np.ndarray):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
