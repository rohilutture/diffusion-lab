"""Sampler correctness against closed-form oracles: error must shrink with more steps, at the right order,
and the higher-order methods must beat first-order ones at equal NFE."""
import pytest
import torch

from diffusionlab.paths import FlowPath, VPPath
from diffusionlab.sampling import cfg_model_fn, sample

from oracles import FlowOracle, VPOracle, exact_flow

M = [1.0, -2.0, 0.5, 3.0]
SD = 0.5


def z0(n=128, seed=0):
    return torch.randn(n, 4, generator=torch.Generator().manual_seed(seed))


def err(sampler, nfe, path, oracle, exact_fn):
    z = z0()
    out, used = sample(oracle, path, z.shape, sampler, nfe, "cpu", x_init=z.clone())
    return (out - exact_fn(z)).abs().max().item(), used


FLOW = (FlowPath(), FlowOracle(M, SD), lambda z: exact_flow(M, SD, z))
VP = (VPPath(), VPOracle(M, SD), lambda z: VPOracle(M, SD).exact(z))


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_flow_error_decreases_with_steps(sampler):
    errs = [err(sampler, n, *FLOW)[0] for n in (5, 11, 41, 161)]
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 2e-2, errs


@pytest.mark.parametrize("sampler", ["ddim", "dpmpp2m"])
def test_vp_error_decreases_with_steps(sampler):
    errs = [err(sampler, n, *VP)[0] for n in (5, 10, 40, 160)]
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 2e-2, errs


def test_heun_beats_euler_at_equal_nfe():
    e_euler, n1 = err("euler", 19, *FLOW)
    e_heun, n2 = err("heun", 19, *FLOW)
    assert n1 == n2 == 19  # same compute
    assert e_heun < e_euler / 3, (e_euler, e_heun)


def test_dpmpp2m_beats_ddim_at_equal_nfe():
    e_ddim, n1 = err("ddim", 10, *VP)
    e_dpm, n2 = err("dpmpp2m", 10, *VP)
    assert n1 == n2 == 10
    assert e_dpm < e_ddim / 3, (e_ddim, e_dpm)


def test_nfe_accounting():
    assert err("euler", 8, *FLOW)[1] == 8
    assert err("heun", 8, *FLOW)[1] == 7   # (8+1)//2 = 4 steps -> 2*4-1 evals
    assert err("ddim", 8, *VP)[1] == 8
    assert err("dpmpp2m", 8, *VP)[1] == 8


def test_wrong_sampler_for_path_rejected():
    with pytest.raises(ValueError):
        sample(FlowOracle(M, SD), FlowPath(), (2, 4), "ddim", 5, "cpu")


def test_oracle_matches_path_definition():
    """The oracle's velocity must equal E[target | x_t] under each path (checked by Monte Carlo)."""
    torch.manual_seed(0)
    n = 400000
    for path, oracle in ((FlowPath(), FlowOracle(M, SD)), (VPPath(), VPOracle(M, SD))):
        t = torch.full((n,), 0.4)
        x0 = torch.tensor(M) + SD * torch.randn(n, 4)
        eps = torch.randn(n, 4)
        xt = path.noisy(x0, eps, t)
        # regress the target on x_t in a narrow slab around a fixed point: E[v | x_t ~ x*]
        x_star = xt.mean(0, keepdim=True)  # high-density point, so the slab is well populated
        close = (xt - x_star).norm(dim=1) < 0.35
        mc = path.target(x0, eps, t)[close].mean(0)
        pred = oracle(x_star, t[:1])[0]
        assert close.sum() > 300
        assert (mc - pred).abs().max() < 0.2, (path.name, mc, pred)


def test_cfg_scale_one_is_conditional_and_equal_branches_collapse():
    class M_:
        def __call__(self, x, t, y):
            return x * 0 + y.view(-1, 1).float()

    y = torch.tensor([3, 5])
    x, t = torch.zeros(2, 4), torch.zeros(2)
    assert torch.equal(cfg_model_fn(M_(), y, 10, 1.0)(x, t), M_()(x, t, y))
    # guided = v_u + w (v_c - v_u): here v_c = y, v_u = 10
    out = cfg_model_fn(M_(), y, 10, 2.0)(x, t)
    assert torch.allclose(out[:, 0], torch.tensor([10 + 2 * (3 - 10), 10 + 2 * (5 - 10)]).float())
