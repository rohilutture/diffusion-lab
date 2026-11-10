import torch

from diffusionlab.data import POINT_CLASSES, MovingShapes, PointShapes
from diffusionlab.metrics import cov_cd, mmd_cd, one_nna_cd, pairwise_chamfer, video_stats


# ------------------------------------------------------------------ data
def test_moving_shapes_deterministic_range_and_shape():
    ds = MovingShapes(n=20, frames=8, size=32, seed=3)
    v, c = ds[5]
    v2, c2 = ds[5]
    assert v.shape == (1, 8, 32, 32) and v.dtype == torch.float32 and c == c2 and torch.equal(v, v2)
    assert v.min() == -1 and v.max() == 1
    assert {ds[i][1] for i in range(20)} == {0, 1, 2}  # all classes occur
    assert not torch.equal(ds[1][0], ds[2][0])


def test_moving_shapes_physics_is_clean():
    ds = MovingShapes(n=60, frames=16, size=32, seed=0)
    vids = torch.stack([ds[i][0] for i in range(60)])
    s = video_stats(vids)
    assert s["area_cv"] < 0.12, s       # rigid shape: area stays ~constant
    assert s["blank_frac"] == 0.0
    # Pixel quantization alone gives ~0.7 px/frame^2 of centroid second-difference noise on perfectly rigid,
    # constant-velocity motion, so an absolute bound is meaningless; the metric must instead be *sensitive*:
    # destroying temporal order must raise it a lot.
    g = torch.Generator().manual_seed(0)
    shuffled = torch.stack([v[:, torch.randperm(v.shape[1], generator=g)] for v in vids])
    assert s["accel"] < 1.5, s
    assert video_stats(shuffled)["accel"] > 4 * s["accel"], (s, video_stats(shuffled))


def test_video_stats_flag_broken_video():
    ds = MovingShapes(n=8, frames=16, size=32, seed=0)
    good = torch.stack([ds[i][0] for i in range(8)])
    bad = good.clone()
    bad[:, :, 5:9] = -1                  # object vanishes for 4 frames
    assert video_stats(bad)["blank_frac"] > 0.2 and video_stats(good)["blank_frac"] == 0
    noisy = (good + 0.8 * torch.randn_like(good)).clamp(-1, 1)
    assert video_stats(noisy)["area_cv"] > video_stats(good)["area_cv"]


def test_point_shapes_geometry_per_class():
    ds = PointShapes(n=200, num_points=400, seed=0, jitter=0.0)
    seen = {}
    for i in range(200):
        p, c = ds[i]
        seen[c] = p
    assert set(seen) == set(range(len(POINT_CLASSES)))
    assert torch.allclose(seen[0].norm(dim=1), torch.full((400,), 0.9), atol=1e-5)        # sphere radius
    assert (seen[1].abs().max(dim=1).values - 0.55).abs().max() < 1e-5                    # cube surface (inf-norm = 0.55)
    t = seen[2]                                                                           # torus: (sqrt(x²+y²) - R)² + z² = r²
    assert ((torch.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2) - 0.65) ** 2 + t[:, 2] ** 2 - 0.25**2).abs().max() < 1e-5
    e = seen[4]                                                                           # ellipsoid equation
    assert ((e / torch.tensor([0.9, 0.55, 0.35])).pow(2).sum(1) - 1).abs().max() < 1e-5


def test_point_shapes_deterministic_and_jittered():
    a, b = PointShapes(n=10, seed=1), PointShapes(n=10, seed=1)
    assert torch.equal(a[3][0], b[3][0])
    assert not torch.equal(PointShapes(n=10, seed=2)[3][0], a[3][0])


# ------------------------------------------------------------------ metrics
def clouds(cls_ids, n_each, seed):
    ds = PointShapes(n=2000, num_points=128, seed=seed)
    out = []
    for i in range(2000):
        p, c = ds[i]
        if c in cls_ids and sum(1 for _ in out) < n_each * len(cls_ids):
            out.append(p)
    return torch.stack(out)


def test_chamfer_zero_for_identical_and_symmetric():
    a, b = clouds({0}, 4, 0), clouds({1}, 4, 1)
    d = pairwise_chamfer(a, a)
    assert torch.allclose(d.diag(), torch.zeros(len(a)), atol=1e-6)
    torch.testing.assert_close(pairwise_chamfer(a, b), pairwise_chamfer(b, a).T, atol=1e-5, rtol=1e-5)
    assert pairwise_chamfer(a, b).min() > pairwise_chamfer(a, a)[~torch.eye(len(a), dtype=bool)].max()  # different shapes are farther


def test_set_metrics_separate_matching_from_mismatched_distributions():
    ref = clouds({0, 1, 2}, 16, 10)
    same = clouds({0, 1, 2}, 16, 11)            # same distribution, different samples
    other = clouds({3, 4}, 24, 12)              # different shapes entirely
    assert mmd_cd(same, ref) < mmd_cd(other, ref) / 2
    assert cov_cd(same, ref) > cov_cd(other, ref)
    assert one_nna_cd(same, ref) < 0.75         # hard to tell apart (ideal 0.5)
    assert one_nna_cd(other, ref) > 0.95        # trivially separable
