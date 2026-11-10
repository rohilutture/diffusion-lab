"""End-to-end smoke on CPU with tiny models: train -> loss drops -> checkpoint -> reload -> sample -> metrics."""
import numpy as np
import pytest
import torch

from diffusionlab.metrics import video_stats
from diffusionlab.sample import load, sample_checkpoint
from diffusionlab.train import run


def cfg(tmp_path, kind, path, model, data, steps=80):
    return {"out_dir": str(tmp_path / f"{kind}_{path}"), "path": path, "cfg_scale": 1.0, "model": model, "data": data,
            "train": {"steps": steps, "batch": 16, "lr": 3e-3, "ema": 0.9, "cfg_drop": 0.2, "amp": False, "log_every": 40,
                      "ckpt_every": 1000, "workers": 0, "seed": 0}}


POINTS = ({"kind": "point_dit", "dim": 64, "depth": 2, "heads": 2, "num_classes": 5},
          {"kind": "point_shapes", "n": 256, "num_points": 64, "seed": 0})
VIDEO = ({"kind": "video_dit", "frames": 4, "img_size": 16, "patch": 4, "in_ch": 1, "dim": 64, "depth": 1, "heads": 2,
          "num_classes": 3}, {"kind": "moving_shapes", "n": 128, "frames": 4, "size": 16, "seed": 0})


@pytest.mark.parametrize("path", ["flow", "vp"])
def test_points_train_reload_sample(tmp_path, path):
    c = cfg(tmp_path, "pts", path, *POINTS, steps=120)
    losses = run(c, device="cpu", log=lambda *_: None)
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-20:]) < np.mean(losses[:20]) * 0.9, (np.mean(losses[:20]), np.mean(losses[-20:]))
    model, c2 = load(str(tmp_path / f"pts_{path}" / "ckpt.pt"), "cpu")
    sampler = "heun" if path == "flow" else "dpmpp2m"
    x, nfe = sample_checkpoint(model, c2, (64, 3), 6, sampler, 6, 2.0, seed=0, device="cpu")  # with CFG (2 branches)
    assert x.shape == (6, 64, 3) and torch.isfinite(x).all()
    assert x.abs().max() < 20  # a trained tiny model should not diverge


def test_video_train_and_sample_shapes(tmp_path):
    c = cfg(tmp_path, "vid", "flow", *VIDEO, steps=40)
    losses = run(c, device="cpu", log=lambda *_: None)
    assert np.all(np.isfinite(losses))
    model, c2 = load(str(tmp_path / "vid_flow" / "ckpt.pt"), "cpu")
    x, _ = sample_checkpoint(model, c2, (1, 4, 16, 16), 3, "euler", 4, 1.0, seed=0, device="cpu")
    assert x.shape == (3, 1, 4, 16, 16)
    assert set(video_stats(x.clamp(-1, 1))) == {"area_cv", "accel", "blank_frac"}


def test_resume_continues_step_counter(tmp_path):
    c = cfg(tmp_path, "pts", "flow", *POINTS, steps=20)
    run(c, device="cpu", log=lambda *_: None)
    ck = str(tmp_path / "pts_flow" / "ckpt.pt")
    assert torch.load(ck)["step"] == 20
    more = run(c, resume=ck, device="cpu", max_steps=30, log=lambda *_: None)
    assert len(more) == 10 and torch.load(ck)["step"] == 30
