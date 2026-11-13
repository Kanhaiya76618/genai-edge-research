"""Constraints of P1, the SLA guarantee (Theorems 2-3), exhaustive optimum, true-length evaluation."""
import numpy as np
import pytest

from edgegen.config import load_config, with_overrides
from edgegen.exhaustive import exhaustive, exhaustive_lex
from edgegen.experiments import METHODS, small_cfg
from edgegen.metrics import evaluate
from edgegen.online import run_online
from edgegen.outer_solver import NodeEvaluator, jcc_edge, lower_bound, p1_objective
from edgegen.scenario import generate_scenario

CFG = load_config()
SETTINGS = [{}, {"layout.n_users": 80}, {"layout.hotspot_share": 0.0}, {"channel.bandwidth_mhz": 5.0},
            {"gpu.throughput_dte_s": 750.0}, {"energy.lambda": 0.3}, {"gpu.kv_capacity_tokens": 4096}]


@pytest.mark.parametrize("over", SETTINGS)
def test_every_method_satisfies_capacities(over):
    sc = generate_scenario(with_overrides(CFG, over), 1)
    for name, fn in METHODS.items():
        sol = fn(sc)
        assert sol.node.shape == (sc.M,) and np.all((sol.node >= 0) & (sol.node < sc.N)), name
        for n in range(sc.N):
            on = sol.node == n
            assert sol.b[on].sum() <= sc.B[n] * (1 + 1e-9), name
            assert sol.f[on].sum() <= sc.F[n] * (1 + 1e-9), name
        assert np.all(sol.b > 0) and np.all(sol.f > 0) and np.all(sol.f <= sc.f_max * (1 + 1e-12)), name
        assert np.isfinite(evaluate(sc, sol)[1]["wmean_latency"])


@pytest.mark.parametrize("over", SETTINGS)
@pytest.mark.parametrize("k", range(3))
def test_guaranteed_requests_meet_ttft_and_tpot(over, k):
    """Theorem 2: every request that Algorithm 1 marks as guaranteed meets both targets with its TRUE
    output length (TTFT and TPOT do not depend on o_i)."""
    sc = generate_scenario(with_overrides(CFG, over), k)
    for name in ("JCC-Edge", "JCC-Greedy", "Load-aware-SLA", "Decoupled-SLA"):
        sol = METHODS[name](sc)
        _, agg = evaluate(sc, sol)
        assert agg["guarantee_broken"] == 0, name
        i = np.arange(sc.M)
        ttft = sc.up_bits / (sol.b * sc.eta[i, sol.node]) + sc.pre_work / sol.f
        g = sol.guaranteed
        assert np.all(ttft[g] <= sc.tau_ttft[g]) and np.all(1.0 / sol.f[g] <= sc.tau_tok[g]), name


def test_feasible_nodes_guarantee_everyone():
    """If no node needs to demote anyone, every request is guaranteed."""
    sc = generate_scenario(with_overrides(CFG, {"layout.n_users": 20}), 0)
    sol = jcc_edge(sc)
    if sol.info["feasible_nodes"] == sol.info["used_nodes"]:
        assert sol.guaranteed.all()


@pytest.mark.parametrize("k", range(3))
def test_kv_constraint_respected_with_predicted_lengths(k):
    """When the predicted footprint fits into the pooled KV capacity with some slack, no node overflows."""
    sc = generate_scenario(with_overrides(CFG, {"layout.n_users": 60, "gpu.kv_capacity_tokens": 12288}), k)
    if sc.tok_hat.sum() > 0.8 * sc.K.sum():
        pytest.skip("pooled KV capacity too tight for this instance")
    sol = jcc_edge(sc)
    for n in range(sc.N):
        assert sc.tok_hat[sol.node == n].sum() <= sc.K[n] + 1e-9


@pytest.mark.parametrize("M", [4, 6, 7])
@pytest.mark.parametrize("k", range(3))
def test_exhaustive_is_a_lower_bound(M, k):
    sc = generate_scenario(small_cfg(CFG, M), k)
    opt = exhaustive(sc).objective
    for name in ("Greedy (Phi)", "Greedy + LS (Phi)", "Decoupled", "Comm-only"):
        assert opt <= METHODS[name](sc).objective * (1 + 1e-9), name
    v_opt, phi_opt = exhaustive_lex(sc)
    ev = NodeEvaluator(sc)
    node = jcc_edge(sc).node
    v = sum(ev.shortfall(n, tuple(np.flatnonzero(node == n).tolist())) for n in range(sc.N))
    assert v_opt <= v + 1e-9


def test_local_search_never_worsens_greedy():
    for k in range(4):
        sc = generate_scenario(CFG, k)
        assert METHODS["Greedy + LS (Phi)"](sc).objective <= METHODS["Greedy (Phi)"](sc).objective * (1 + 1e-12)


def test_sla_local_search_never_increases_demoted_weight():
    for k in range(3):
        sc = generate_scenario(with_overrides(CFG, {"layout.n_users": 80}), k)
        ev = NodeEvaluator(sc)
        V = lambda node: sum(ev.shortfall(n, tuple(np.flatnonzero(node == n).tolist())) for n in range(sc.N))
        assert V(jcc_edge(sc).node) <= V(jcc_edge(sc, sla_assoc=False).node) + 1e-9


def test_lower_bound_is_below_every_scheme():
    for k in range(3):
        sc = generate_scenario(with_overrides(CFG, {"layout.n_users": 60}), k)
        lb = lower_bound(sc)
        for name in ("Decoupled", "Load-aware", "JCC-Edge", "Classical-JCC"):
            sol = METHODS[name](sc)
            i = np.arange(sc.M)
            T = sc.D_true / (sol.b * sc.eta[i, sol.node]) + sc.C_true / sol.f
            assert lb <= np.sum(sc.w * T) * (1 + 1e-9), name


def test_evaluation_uses_true_lengths():
    sc = generate_scenario(CFG, 2)
    assert not np.allclose(sc.o_true, sc.o_hat)
    sol = jcc_edge(sc)
    users, _ = evaluate(sc, sol)
    assert np.allclose(users.decode_s, sc.o_true / sol.f)
    assert p1_objective(sc, sol.node, sol.b, sol.f) == pytest.approx(sol.objective)


def test_hotspot_and_reproducibility():
    a, b = generate_scenario(CFG, 5), generate_scenario(CFG, 5)
    assert np.array_equal(a.user_xy, b.user_xy) and np.array_equal(a.o_true, b.o_true)
    assert a.in_hotspot.sum() == round(CFG["layout"]["hotspot_share"] * a.M)
    d = np.linalg.norm(a.user_xy[a.in_hotspot] - a.node_xy[0], axis=1)
    assert np.median(d) < 3 * CFG["layout"]["hotspot_sigma_m"]
    c = generate_scenario(with_overrides(CFG, {"channel.bandwidth_mhz": 5.0, "gpu.throughput_dte_s": 750.0}), 5)
    assert np.array_equal(a.eta, c.eta)                        # sweeps keep users and channels fixed


@pytest.mark.parametrize("policy", ["Load-aware-SLA", "JCC-Edge"])
def test_online_guarantee_holds(policy):
    """Theorem 3: in the dynamic simulation no admitted (enforced) request misses TTFT or TPOT."""
    cfg = with_overrides(CFG, {"experiments.online.horizon_s": 40.0, "experiments.online.warmup_s": 5.0,
                               "experiments.online.cooldown_s": 5.0})
    r = run_online(cfg, 20.0, policy, 0)
    assert r["guarantee_broken"] == 0 and r["enforced_pct"] > 50.0
