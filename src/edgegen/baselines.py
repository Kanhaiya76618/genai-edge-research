"""Benchmarks of the paper (Sec. V-A).

SLA-unaware schemes
  Nearest-ES     : nearest node, equal bandwidth and GPU shares
  Comm-only      : strongest-SINR node, optimal bandwidth (Prop. 1), equal GPU shares
  Compute-only   : load-aware association (below), optimal GPU shares (Prop. 2), equal bandwidth
  Load-aware     : requests in decreasing predicted work C_i, each to the less loaded (assigned work /
                   F_n) of its two nodes with the largest eta_{i,n}; optimal bandwidth and GPU shares
  Decoupled      : strongest-SINR node, optimal bandwidth and optimal GPU shares, i.e. each resource is
                   optimised in its own silo
  Classical-JCC  : joint association and optimal shares on a classical MEC task model (outer_solver)
The same associations with the SLA-enforcing allocation of Algorithm 1 (step 3)
  Decoupled-SLA, Load-aware-SLA
Equal GPU shares respect the single-stream ceiling: f_i = min(F_n / |S_n|, f_max).
"""
from __future__ import annotations

import time

import numpy as np

from .inner_solver import bandwidth_shares, gpu_shares
from .outer_solver import Solution, classical_jcc, p1_objective, with_sla_allocation
from .scenario import Scenario


def _allocate(sc: Scenario, node: np.ndarray, opt_b: bool, opt_f: bool):
    b, f = np.zeros(sc.M), np.zeros(sc.M)
    for n in range(sc.N):
        u = np.flatnonzero(node == n)
        if u.size == 0:
            continue
        b[u] = bandwidth_shares(sc.a_hat[u, n], sc.w[u], sc.B[n])[0] if opt_b else sc.B[n] / u.size
        f[u] = (gpu_shares(sc.C_hat[u], sc.w[u], sc.F[n], sc.f_max, sc.lam, sc.kappa) if opt_f
                else min(sc.F[n] / u.size, sc.f_max))
    return b, f


def _run(sc: Scenario, name: str, node_fn, opt_b: bool, opt_f: bool) -> Solution:
    t0 = time.perf_counter()
    node = node_fn(sc)
    b, f = _allocate(sc, node, opt_b, opt_f)
    return Solution(name, node, b, f, p1_objective(sc, node, b, f), time.perf_counter() - t0)


def least_loaded_of_two_strongest(sc: Scenario) -> np.ndarray:
    """Load-aware association: heaviest predicted work first, to the lighter of its 2 best cells."""
    best2 = np.argsort(-sc.eta, axis=1)[:, :2]
    load = np.zeros(sc.N)
    node = np.zeros(sc.M, dtype=int)
    for i in np.argsort(-sc.C_hat, kind="stable"):
        n = best2[i][np.argmin(load[best2[i]] / sc.F[best2[i]])]
        node[i] = n
        load[n] += sc.C_hat[i]
    return node


def nearest_es(sc):
    return _run(sc, "Nearest-ES", Scenario.nearest, False, False)


def comm_only(sc):
    return _run(sc, "Comm-only", Scenario.strongest, True, False)


def compute_only(sc):
    return _run(sc, "Compute-only", least_loaded_of_two_strongest, False, True)


def load_aware(sc):
    return _run(sc, "Load-aware", least_loaded_of_two_strongest, True, True)


def decoupled(sc):
    return _run(sc, "Decoupled", Scenario.strongest, True, True)


def decoupled_sla(sc):
    t0 = time.perf_counter()
    return with_sla_allocation(sc, sc.strongest(), "Decoupled-SLA", t0)


def load_aware_sla(sc):
    t0 = time.perf_counter()
    return with_sla_allocation(sc, least_loaded_of_two_strongest(sc), "Load-aware-SLA", t0)


UNAWARE = {"Nearest-ES": nearest_es, "Comm-only": comm_only, "Compute-only": compute_only,
           "Load-aware": load_aware, "Decoupled": decoupled, "Classical-JCC": classical_jcc}
BASELINES = {**UNAWARE, "Decoupled-SLA": decoupled_sla, "Load-aware-SLA": load_aware_sla}
