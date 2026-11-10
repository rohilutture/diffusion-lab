import pytest
import torch

from diffusionlab.models import DiT, PointDiT, VideoDiT, build_model, sincos_2d


def tiny_models():
    return {
        "dit": (DiT(img_size=8, patch=2, in_ch=3, dim=32, depth=2, heads=2, num_classes=4), (2, 3, 8, 8)),
        "video": (VideoDiT(frames=4, img_size=8, patch=4, in_ch=1, dim=32, depth=2, heads=2, num_classes=3), (2, 1, 4, 8, 8)),
        "points": (PointDiT(dim=32, depth=2, heads=2, num_classes=5), (2, 24, 3)),
    }


@pytest.mark.parametrize("name", ["dit", "video", "points"])
def test_output_shape_and_zero_init(name):
    m, shape = tiny_models()[name]
    x = torch.randn(shape)
    out = m(x, torch.rand(2), torch.zeros(2, dtype=torch.long))
    assert out.shape == x.shape
    # adaLN-Zero: every block and the output layer start at zero, so the untrained model outputs exactly 0
    assert out.abs().max() == 0


@pytest.mark.parametrize("name", ["dit", "video", "points"])
def test_one_step_makes_params_move_and_output_nonzero(name):
    m, shape = tiny_models()[name]
    x, y, t = torch.randn(shape), torch.zeros(2, dtype=torch.long), torch.rand(2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(3):  # zero-init needs a couple of steps before gradients reach the early blocks
        loss = m(x, t, y).pow(2).mean() + (m(x, t, y) - 1).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert m(x, t, y).abs().max() > 0
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_point_model_is_permutation_equivariant():
    m = PointDiT(dim=32, depth=2, heads=2, num_classes=5)
    for p in m.parameters():  # break the zero init so the test is not vacuous
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.02)
    x, t, y = torch.randn(2, 24, 3), torch.rand(2), torch.tensor([1, 2])
    perm = torch.randperm(24)
    out, out_p = m(x, t, y), m(x[:, perm], t, y)
    torch.testing.assert_close(out[:, perm], out_p, atol=1e-4, rtol=1e-4)


def test_video_model_uses_time_axis():
    """Changing one frame must change other frames' outputs (temporal attention actually mixes frames)."""
    m = VideoDiT(frames=4, img_size=8, patch=4, in_ch=1, dim=32, depth=2, heads=2, num_classes=3)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.02)
    x, t, y = torch.randn(1, 1, 4, 8, 8), torch.rand(1), torch.zeros(1, dtype=torch.long)
    base = m(x, t, y)
    x2 = x.clone(); x2[:, :, 3] += 1.0
    delta = (m(x2, t, y) - base).abs()
    assert delta[:, :, 0].max() > 1e-5  # frame 0's output depends on frame 3's input


def test_label_changes_output_after_training_signal():
    m = DiT(img_size=8, patch=2, in_ch=3, dim=32, depth=2, heads=2, num_classes=4)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.02)
    x, t = torch.randn(1, 3, 8, 8), torch.rand(1)
    a, b = m(x, t, torch.tensor([0])), m(x, t, torch.tensor([4]))  # class 0 vs null
    assert (a - b).abs().max() > 1e-6


def test_sincos_2d_shape_and_uniqueness():
    e = sincos_2d(32, 4)
    assert e.shape == (16, 32)
    assert torch.cdist(e, e).fill_diagonal_(1).min() > 1e-3  # all positions distinguishable


def test_build_model_from_config():
    m = build_model({"kind": "point_dit", "dim": 32, "depth": 1, "heads": 2, "num_classes": 5})
    assert isinstance(m, PointDiT)
