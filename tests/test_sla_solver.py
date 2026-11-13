"""Propositions 3 and 4 (SLA-constrained inner problem) against the generic convex solver CVXPY."""
import numpy as np
import pytest

from edgegen.sla_solver import DELTA, box_waterfill, sla_min_bandwidth, sla_node, sla_node_cvxpy

B, F, FMAX, KAPPA = 20e6, 1500.0, 200.0, 8.9e-6


def _instance(seed, n):
    r = np.random.default_rng(seed)
    s = np.clip(np.round(200 * r.lognormal(0, .8, n)), 4, 2048)
    o = np.clip(np.round(150 * r.lognormal(0, .8, n)), 4, 2048)
    A = np.where(r.random(n) < .4, r.uniform(50, 500, n) * 8000, 0.0)
    eta = r.uniform(0.5, 7, n)
    aup = (s * 32 + A) / eta
    a = aup + o * 32 / eta
    P, cls = 0.125 * s, r.choice(3, n, p=[.5, .3, .2])
    C = P + o
    w = np.array([1., 2., 5.])[cls]
    tau = np.array([1., .5, .5])[cls]
    ell = 1 / np.array([.02, .01, .01])[cls]
    return a, aup, C, P, w, tau, ell


def test_box_waterfill_is_exact():
    r = np.random.default_rng(0)
    v, lo = r.uniform(1, 10, 12), r.uniform(0, 1, 12)
    hi = lo + r.uniform(0.5, 3, 12)
    x = box_waterfill(v, 10.0, lo, hi)
    assert x.sum() == pytest.approx(10.0) and np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
    free = (x > lo + 1e-9) & (x < hi - 1e-9)
    assert np.allclose(x[free] / v[free], (x[free] / v[free])[0])            # one water level


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("lam", [0.0, 0.05])
def test_proposition4_matches_cvxpy_and_meets_every_target(seed, lam):
    n = int(np.random.default_rng(1000 + seed).integers(3, 18))
    a, aup, C, P, w, tau, ell = _instance(seed, n)
    r = sla_node(a, aup, C, P, w, tau, ell, np.zeros(n), B, F, FMAX, lam, KAPPA)
    assert r.b.sum() <= B * (1 + 1e-9) and r.f.sum() <= F * (1 + 1e-9) and np.all(r.f <= FMAX * (1 + 1e-12))
    g = r.guaranteed
    assert np.all(aup[g] / r.b[g] + P[g] / r.f[g] <= tau[g]) and np.all(r.f[g] >= ell[g])
    status, ref, _, _ = sla_node_cvxpy(a, aup, C, P, w, tau * (1 - DELTA), ell / (1 - DELTA), B, F, FMAX, lam, KAPPA)
    if r.feasible:
        assert status == "optimal" and r.cost == pytest.approx(ref, rel=3e-3)
    else:
        assert status.startswith("infeasible")                               # Prop. 3 agrees with CVXPY


def test_proposition3_minimum_bandwidth_is_tight():
    a, aup, C, P, w, tau, ell = _instance(5, 10)
    bw, f = sla_min_bandwidth(aup, P, tau, ell, F, FMAX)
    assert np.isfinite(bw) and f.sum() <= F * (1 + 1e-9)
    status, *_ = sla_node_cvxpy(a, aup, C, P, w, tau, ell, bw * 1.001, F, FMAX)
    assert status == "optimal"
    status, *_ = sla_node_cvxpy(a, aup, C, P, w, tau, ell, bw * 0.99, F, FMAX)
    assert status.startswith("infeasible")
