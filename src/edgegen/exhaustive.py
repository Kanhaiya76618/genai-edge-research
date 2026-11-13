"""Exhaustive search: the global optimum of P1 without deadlines (Fig. 5), for small M and N.

Every node cost Phi_n(S; w) is computed once per user subset S (2^M * N inner solves); the cost
of each of the N^M associations is then a sum of cached subset costs, evaluated in one pass.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .outer_solver import NodeEvaluator, Solution
from .scenario import Scenario


def exhaustive(sc: Scenario) -> Solution:
    t0 = time.perf_counter()
    M, N = sc.M, sc.N
    ev = NodeEvaluator(sc)
    ev.set_weights(sc.w)
    subset = np.zeros((N, 1 << M))
    for mask in range(1, 1 << M):
        users = tuple(i for i in range(M) if mask >> i & 1)
        for n in range(N):
            subset[n, mask] = ev.solve(n, users)[0]
    assign = np.array(list(itertools.product(range(N), repeat=M)), dtype=np.int64)
    bits = 1 << np.arange(M, dtype=np.int64)
    total = sum(subset[n, ((assign == n) * bits).sum(axis=1)] for n in range(N))
    k = int(np.argmin(total))
    node = assign[k].astype(int)
    b, f = ev.allocate(node)
    return Solution("Exhaustive", node, b, f, float(total[k]), time.perf_counter() - t0)


def exhaustive_lex(sc: Scenario) -> tuple[float, float]:
    """Lexicographic optimum over all N^M associations: the least total demoted priority weight
    V(x) = sum_n V_n(S_n) (admission rule of Algorithm 1, step 4), and among those the least sum_n Phi_n.
    Associations that violate the predicted KV-cache limit are excluded."""
    M, N = sc.M, sc.N
    ev = NodeEvaluator(sc)
    phi, v, kv_ok = np.zeros((N, 1 << M)), np.zeros((N, 1 << M)), np.ones((N, 1 << M), bool)
    for mask in range(1, 1 << M):
        users = tuple(i for i in range(M) if mask >> i & 1)
        for n in range(N):
            phi[n, mask] = ev.solve(n, users)[0]
            v[n, mask] = ev.shortfall(n, users)
            kv_ok[n, mask] = ev.kv_ok(n, users)
    assign = np.array(list(itertools.product(range(N), repeat=M)), dtype=np.int64)
    bits = 1 << np.arange(M, dtype=np.int64)
    masks = [((assign == n) * bits).sum(axis=1) for n in range(N)]
    tv = sum(v[n, masks[n]] for n in range(N))
    tp = sum(phi[n, masks[n]] for n in range(N))
    ok = np.all([kv_ok[n, masks[n]] for n in range(N)], axis=0)
    tv = np.where(ok, tv, np.inf)
    vmin = tv.min()
    return float(vmin), float(tp[tv <= vmin + 1e-9].min())
