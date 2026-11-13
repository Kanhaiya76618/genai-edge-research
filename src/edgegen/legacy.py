"""Deadline handling of the previous drafts, kept only for comparison (ablation and Fig. 9).

``dual_ascent_phasewise`` relaxes the TTFT and TPOT constraints with multipliers that are updated by a
constant-step subgradient method while the association is re-optimised by local search, and returns
the best iterate. Because the association is discrete the iterates oscillate, and no request is
guaranteed to meet its targets. Algorithm 1 of the paper replaces this loop by an exact per-node
feasibility test and an exact SLA-constrained allocation (``sla_solver.py``).
"""
from __future__ import annotations

import time

import numpy as np

from .outer_solver import NodeEvaluator, Solution, greedy, local_search, p1_objective
from .scenario import Scenario


def dual_ascent_phasewise(sc: Scenario, *, name: str = "Dual ascent (prev. draft)") -> Solution:
    """Subgradient dual ascent on TTFT_i <= tau_ttft (nu_i) and o_i/f_i <= o_i tau_tok (xi_i):
    qb_i = (w_i + nu_i) a_up + w_i a_dn,  qf_i = (w_i + nu_i) rho s_i + (w_i + xi_i) o_hat_i,
    nu_i <- [nu_i + eps w_i (TTFT_i/tau_ttft - 1)]^+,  xi_i <- [xi_i + eps w_i (TPOT_i/tau_tok - 1)]^+."""
    t0 = time.perf_counter()
    alg = sc.cfg["algorithm"]
    K, eps = int(alg["dual_steps"]), float(alg["dual_step_size"])
    ev = NodeEvaluator(sc)
    a_up, a_dn = sc.a_up, sc.a_hat - sc.a_up
    nu, xi = np.zeros(sc.M), np.zeros(sc.M)

    def coefs():
        return ((sc.w + nu)[:, None] * a_up + sc.w[:, None] * a_dn,
                (sc.w + nu) * sc.pre_work + (sc.w + xi) * (sc.C_hat - sc.pre_work))

    ev.set_coefficients(*coefs())
    node = greedy(ev)
    node, p = local_search(ev, node)
    passes = [p]
    best, best_key, best_count, trace = None, None, 0, []
    i = np.arange(sc.M)
    for k in range(K):
        b, f = ev.allocate(node)
        g1 = (a_up[i, node] / b + sc.pre_work / f) / sc.tau_ttft - 1.0
        g2 = 1.0 / (f * sc.tau_tok) - 1.0
        viol = (g1 > 0) | (g2 > 0)
        key = (float(np.sum(sc.w * viol)), p1_objective(sc, node, b, f))
        if best_key is None or key < best_key:
            best, best_key, best_count = (node.copy(), b, f), key, int(viol.sum())
        trace.append(dict(iteration=k, violations=int(viol.sum()), weighted_violations=key[0], objective=key[1],
                          best_violations=best_count, max_violation=float(max(g1.max(), g2.max()))))
        if not viol.any() or k == K - 1:
            break
        nu = np.maximum(0.0, nu + eps * sc.w * g1)
        xi = np.maximum(0.0, xi + eps * sc.w * g2)
        ev.set_coefficients(*coefs())
        node, p = local_search(ev, node)
        passes.append(p)
    node, b, f = best
    return Solution(name, node, b, f, best_key[1], time.perf_counter() - t0, trace, passes)
