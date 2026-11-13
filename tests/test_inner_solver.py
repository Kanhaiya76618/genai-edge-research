"""Propositions 1 and 2 (inner problem) against the generic convex solver CVXPY."""
import numpy as np
import pytest

from edgegen.inner_solver import bandwidth_shares, gpu_shares, node_cost, node_cost_cvxpy

B, F, FMAX, KAPPA = 20e6, 1500.0, 200.0, 8.9e-6


def _instance(seed, n=None):
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(3, 16))
    return rng.uniform(1e4, 2e6, n), rng.uniform(20, 600, n), rng.choice([1.0, 2.0, 5.0], n)


@pytest.mark.parametrize("seed", range(6))
def test_proposition1_closed_form_matches_cvxpy(seed):
    a, C, w = _instance(seed)
    b, value = bandwidth_shares(a, w, B)
    _, f_none = node_cost(a, C, w, np.zeros_like(a), B, F, 1e12)[1:]
    ref, bc, fc = node_cost_cvxpy(a, C, w, B, F, 1e12)                  # ceiling inactive
    assert np.max(np.abs(b - bc)) / B < 1e-4                            # Prop. 1 allocation
    assert np.max(np.abs(f_none - fc)) / F < 1e-4                       # Prop. 2 with lambda = 0
    assert node_cost(a, C, w, np.zeros_like(a), B, F, 1e12)[0] == pytest.approx(ref, rel=1e-4)
    assert value == pytest.approx(np.sum(np.sqrt(w * a)) ** 2 / B)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("lam", [0.0, 0.05, 1.0])
def test_proposition2_with_ceiling_and_energy_matches_cvxpy(seed, lam):
    a, C, w = _instance(100 + seed)
    cost, b, f = node_cost(a, C, w, np.zeros_like(a), B, F, FMAX, lam, KAPPA)
    ref, _, fc = node_cost_cvxpy(a, C, w, B, F, FMAX, lam, KAPPA)
    assert cost == pytest.approx(ref, rel=1e-4)
    assert np.max(np.abs(f - fc)) / F < 1e-3
    assert f.sum() <= F * (1 + 1e-9) and np.all(f <= FMAX * (1 + 1e-12)) and np.all(f > 0)
    assert b.sum() == pytest.approx(B)


def test_waterfilling_caps_the_largest_requests():
    a, C, w = _instance(7, n=12)
    f = gpu_shares(C, w, F, FMAX)
    assert f.sum() == pytest.approx(F)
    capped = f >= FMAX * (1 - 1e-12)
    v = np.sqrt(w * C)
    if capped.any() and (~capped).any():
        assert v[capped].min() >= v[~capped].max()                       # Prop. 2 proof: largest capped
    unc = ~capped
    assert np.allclose(f[unc] / v[unc], f[unc][0] / v[unc][0])           # square-root rule elsewhere


def test_light_load_runs_everyone_at_the_ceiling():
    _, C, w = _instance(3, n=5)                                           # 5 x 200 < 1500 dte/s
    assert np.allclose(gpu_shares(C, w, F, FMAX), FMAX)
