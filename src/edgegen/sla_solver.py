"""SLA-constrained inner problem at one node (Sec. IV-B and IV-C of the paper).

A request i at a node needs
    TTFT_i = a_up_i / b_i + P_i / f_i <= tau_i          (uplink transfer + prefill; P_i = rho s_i)
    TPOT_i = 1 / f_i <= tau_tok_i  <=>  f_i >= ell_i     (ell_i = 1 / tau_tok_i, a rate floor)
and the node has  sum_i b_i <= B,  sum_i f_i <= F,  f_i <= f_max.

Three exact building blocks, all for a FIXED set of requests:

  sla_min_bandwidth   Proposition 3: the smallest total bandwidth with which all requests meet both
                      targets given GPU capacity F (one-dimensional bisection). The node is
                      SLA-feasible iff the GPU floors fit and this bandwidth is <= B.
  sla_allocate        Proposition 4: the allocation minimising the weighted latency (+ energy) subject
                      to the SLA constraints. The problem is convex, so the dual over the TTFT
                      multipliers nu_i has no gap; the TPOT targets are plain rate floors.
  sla_node            Feasibility test, demotion of the cheapest requests to best effort if the node
                      is overloaded, exact allocation, and a primal repair that makes the returned
                      shares satisfy every enforced target exactly (not only up to solver tolerance).

Neither constraint depends on the output length, so SLA feasibility does not depend on its prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from .inner_solver import _cubic_root

DELTA = 1e-3      # targets are tightened by this relative margin inside the solver
ZETA = 0.5        # a demoted (best-effort) request keeps at least ZETA of the equal share


# ----------------------------------------------------------------------------------------------
# Box-constrained water-filling (Propositions 1 and 2 with floors and ceilings)
# ----------------------------------------------------------------------------------------------
def box_waterfill(v, total: float, lo, hi) -> np.ndarray:
    """x_i = clip(v_i t, lo_i, hi_i) with sum_i x_i = total (t >= 0); x = hi if sum(hi) <= total.

    This minimises sum_i v_i^2 / x_i subject to sum x <= total, lo <= x <= hi (exact, by sorting the
    breakpoints of the piecewise-linear map t -> sum_i clip(v_i t, lo_i, hi_i)). If the floors do not
    fit, the floors scaled down to the capacity are returned; callers test feasibility beforehand."""
    v, lo, hi = (np.asarray(x, float) for x in (v, lo, hi))
    if np.sum(hi) <= total:
        return hi.copy()
    s_lo = float(lo.sum())
    if s_lo >= total:
        return lo * (total / s_lo)
    fin = np.isfinite(hi)
    bp = np.concatenate([lo / v, hi[fin] / v[fin]])
    bp = np.unique(bp[bp > 0])
    if bp.size == 0:                                   # no floors, no ceilings: x = v t
        return v * (total / v.sum())
    g = np.clip(v[None, :] * bp[:, None], lo[None, :], hi[None, :]).sum(axis=1)
    k = int(np.searchsorted(g, total, side="left"))
    if k == len(bp):                                   # beyond the last breakpoint (some hi = inf)
        t = bp[-1] + (total - g[-1]) / v[~fin].sum()
    else:
        t0, g0 = (0.0, s_lo) if k == 0 else (bp[k - 1], g[k - 1])
        t = t0 + (total - g0) * (bp[k] - t0) / (g[k] - g0)
    return np.clip(v * t, lo, hi)


def gpu_shares_box(q, C, F: float, lo, hi, lam: float = 0.0, kappa: float = 0.0) -> np.ndarray:
    """min sum_i q_i / f_i + lam kappa C_i f_i^2  s.t.  sum f <= F,  lo_i <= f_i <= hi_i.

    Proposition 2 with rate floors: f_i = clip(phi_i(mu), lo_i, hi_i), phi_i the positive root of
    2 lam kappa C_i phi^3 + mu phi^2 = q_i. For lam = 0 this is the exact box water-filling."""
    q, C, lo = np.asarray(q, float), np.asarray(C, float), np.asarray(lo, float)
    hi = np.broadcast_to(np.asarray(hi, float), q.shape)
    if lam <= 0:
        return box_waterfill(np.sqrt(q), F, lo, hi)
    e = lam * kappa * C
    phi = lambda mu: np.clip(_cubic_root(q, e, mu), lo, hi)
    f0 = phi(0.0)
    if f0.sum() <= F:
        return f0
    if lo.sum() >= F:
        return lo * (F / lo.sum())
    l_hi = np.log((np.sqrt(q).sum() / F) ** 2)
    while phi(np.exp(l_hi)).sum() > F:
        l_hi += 5.0
    l_lo = l_hi - 30.0
    while phi(np.exp(l_lo)).sum() <= F:
        l_lo -= 10.0
    return phi(np.exp(brentq(lambda lm: phi(np.exp(lm)).sum() - F, l_lo, l_hi, xtol=1e-13, rtol=1e-13)))


# ----------------------------------------------------------------------------------------------
# Proposition 3: exact SLA-feasibility test
# ----------------------------------------------------------------------------------------------
def sla_min_bandwidth(aup, P, tau, ell, F: float, f_max: float):
    """Smallest total bandwidth [Hz] that meets TTFT <= tau and f >= ell for every request when the
    node's GPU capacity is F. Returns (bandwidth, f); bandwidth = inf if the GPU floors do not fit.

    The bandwidth a request needs, beta_i(f) = a_up_i / (tau_i - P_i / f), is convex and decreasing in
    its GPU share, so the minimum of sum_i beta_i(f_i) over sum_i f_i <= F has the KKT form
    f_i(g) = clip((P_i + sqrt(a_up_i P_i / g)) / tau_i, ell_i, f_max) with one multiplier g >= 0."""
    aup, P, tau, ell = (np.asarray(x, float) for x in (aup, P, tau, ell))
    if aup.size == 0:
        return 0.0, np.zeros(0)
    base = np.clip(P / tau, ell, f_max)                 # share that suffices when a_up P = 0
    prod = aup * P
    act = prod > 0

    def shares(g):
        f = base.copy()
        f[act] = np.clip((P[act] + np.sqrt(prod[act] / g)) / tau[act], ell[act], f_max)
        return f

    f = np.where(act, f_max, base)                      # g -> 0: as much GPU as allowed
    if f.sum() > F:
        if np.maximum(ell, P / tau).sum() >= F:
            return np.inf, None
        lg = np.log(np.median(prod[act]) / (np.median(tau[act]) * f_max) ** 2)
        lo, hi = lg - 5.0, lg + 5.0
        while shares(np.exp(lo)).sum() <= F:
            lo -= 5.0
        while shares(np.exp(hi)).sum() > F:
            hi += 5.0
        f = shares(np.exp(brentq(lambda x: shares(np.exp(x)).sum() - F, lo, hi, xtol=1e-12, rtol=1e-12)))
    slack = tau - P / f
    if np.any(slack[aup > 0] <= 0):
        return np.inf, None
    return float(np.sum(aup[aup > 0] / slack[aup > 0])), f


# ----------------------------------------------------------------------------------------------
# Proposition 4: exact SLA-constrained allocation (dual over the TTFT multipliers)
# ----------------------------------------------------------------------------------------------
def sla_allocate(a, aup, C, P, w, tau, f_lo, b_lo, enforced, B, F, f_max, lam=0.0, kappa=0.0,
                 max_iter: int = 400):
    """Minimise sum_i w_i (a_i/b_i + C_i/f_i) + lam kappa C_i f_i^2 subject to the capacities,
    f_lo <= f <= f_max, b >= b_lo and TTFT_i <= tau_i for the enforced requests.

    For multipliers nu >= 0 the Lagrangian separates into a bandwidth water-filling with coefficients
    w_i a_i + nu_i a_up_i and a GPU water-filling with coefficients w_i C_i + nu_i P_i; the dual is
    concave and differentiable with gradient TTFT_i - tau_i and is maximised by L-BFGS-B.
    Returns (b, f, nu, evaluations)."""
    a, aup, C, P, w, tau, f_lo, b_lo = (np.asarray(x, float) for x in (a, aup, C, P, w, tau, f_lo, b_lo))
    idx = np.flatnonzero(enforced)
    inf = np.full(a.size, np.inf)

    def primal(nu_e):
        nu = np.zeros(a.size)
        nu[idx] = nu_e
        b = box_waterfill(np.sqrt(w * a + nu * aup), B, b_lo, inf)
        f = gpu_shares_box(w * C + nu * P, C, F, f_lo, f_max, lam, kappa)
        return nu, b, f

    nu, b, f = primal(np.zeros(idx.size))
    if idx.size == 0 or np.all(aup[idx] / b[idx] + P[idx] / f[idx] <= tau[idx]):
        return b, f, nu, 1                              # the unconstrained optimum already meets the SLAs

    scale = w[idx]                                      # optimise nu_i / w_i: well-conditioned

    def neg_dual(z):
        nu, b, f = primal(z * scale)
        val = np.sum((w * a + nu * aup) / b) + np.sum((w * C + nu * P) / f) - np.sum(nu[idx] * tau[idx])
        if lam > 0:
            val += lam * kappa * np.sum(C * f**2)
        grad = (aup[idx] / b[idx] + P[idx] / f[idx] - tau[idx]) * scale
        return -val, -grad

    res = minimize(neg_dual, np.zeros(idx.size), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * idx.size, options=dict(maxiter=max_iter, ftol=1e-15, gtol=1e-10))
    nu, b, f = primal(res.x * scale)
    return b, f, nu, int(res.nfev) + 1


# ----------------------------------------------------------------------------------------------
# One node: feasibility, demotion, allocation, repair
# ----------------------------------------------------------------------------------------------
@dataclass
class NodeAllocation:
    b: np.ndarray             # bandwidth [Hz]
    f: np.ndarray             # GPU share [dte/s]
    cost: float               # sum_i w_i (a_i/b_i + C_i/f_i) + lam (kappa C_i f_i^2 + e_up_i)
    guaranteed: np.ndarray    # True: TTFT and TPOT targets are met by construction
    nu: np.ndarray            # TTFT multipliers (0 where the target is slack or not enforced)
    evals: int                # dual evaluations (1 = closed form sufficed)
    feasible: bool            # True: no request had to be demoted


def solo_requirements(aup, P, tau, ell, B, F, f_max):
    """Per-request SLA floors when served alone: GPU floor, bandwidth need at f_max, and whether the
    request can meet its targets at this node at all (even with the whole node to itself)."""
    aup, P, tau, ell = (np.asarray(x, float) for x in (aup, P, tau, ell))
    f_floor = np.maximum(ell, P / tau)
    slack = tau - P / f_max
    with np.errstate(divide="ignore", invalid="ignore"):
        b_need = np.where(aup > 0, aup / np.where(slack > 0, slack, np.nan), 0.0)
    ok = (ell <= f_max) & ((slack > 0) | ((slack >= 0) & (aup == 0))) & (np.nan_to_num(b_need, nan=np.inf) <= B)
    return f_floor, np.nan_to_num(b_need, nan=np.inf), ok


def sla_admit(aup, P, w, tau, ell, B, F, f_max, enforce=None, delta: float = DELTA, zeta: float = ZETA,
              need_f=None, need_b=None):
    """Which requests of a node get their SLA enforced (steps 1-2 of ``sla_node``).

    Returns (q, f_be, b_be, tau_t, ell_t, feasible): q is the enforced set, f_be / b_be the best-effort
    reservations of the others, and feasible is True iff no request had to be demoted."""
    aup, P, w, tau, ell = (np.asarray(x, float) for x in (aup, P, w, tau, ell))
    n = aup.size
    tau_t, ell_t = tau * (1.0 - delta), ell / (1.0 - delta)
    f_floor, b_need, ok = solo_requirements(aup, P, tau_t, ell_t, B, F, f_max)
    q = ok.copy() if enforce is None else (ok & np.asarray(enforce, bool))
    f_be = np.minimum(zeta * F / n, np.where(ok, f_floor, np.inf))       # best-effort reservations
    b_be = np.minimum(zeta * B / n, np.where(ok, b_need, np.inf))
    if need_f is not None:                                               # nothing left to compute / send
        f_be = np.where(need_f, f_be, 1e-9)
    if need_b is not None:
        b_be = np.where(need_b, b_be, 1e-9)

    def feasible(mask) -> bool:
        if not mask.any():
            return True
        bw, _ = sla_min_bandwidth(aup[mask], P[mask], tau_t[mask], ell_t[mask], F - f_be[~mask].sum(), f_max)
        return bw <= B - b_be[~mask].sum()

    all_ok = feasible(q)
    if not all_ok:
        order = np.flatnonzero(q)
        value = w[order] / (f_floor[order] / F + b_need[order] / B)
        order = order[np.argsort(value, kind="stable")]                 # cheapest to give up first
        lo_k, hi_k = 0, order.size                                       # smallest feasible prefix removal
        while lo_k < hi_k:
            mid = (lo_k + hi_k) // 2
            m = q.copy()
            m[order[:mid]] = False
            lo_k, hi_k = (lo_k, mid) if feasible(m) else (mid + 1, hi_k)
        q[order[:lo_k]] = False
        for i in order[:lo_k][::-1]:                                     # promote back what still fits
            q[i] = True
            if not feasible(q):
                q[i] = False
    return q, f_be, b_be, tau_t, ell_t, bool(all_ok and q.all())


def sla_node(a, aup, C, P, w, tau, ell, e_up, B, F, f_max, lam=0.0, kappa=0.0, enforce=None,
             delta: float = DELTA, zeta: float = ZETA) -> NodeAllocation:
    """SLA-enforcing allocation at one node.

    1. Requests that cannot meet their targets even alone are served best effort.
    2. If the remaining set is infeasible (Prop. 3), requests are demoted in increasing order of
       priority per unit of required resource until it is feasible; requests that fit again are
       promoted back. A demoted request keeps at least `zeta` of the equal share (or its own SLA
       requirement if that is smaller), so it is never starved.
    3. The enforced set gets the optimal SLA-constrained shares (Prop. 4) followed by a bandwidth
       repair, so that every enforced request meets TTFT <= tau and TPOT <= tau_tok exactly."""
    a, aup, C, P, w, tau, ell, e_up = (np.asarray(x, float) for x in (a, aup, C, P, w, tau, ell, e_up))
    n = a.size
    if n == 0:
        z = np.zeros(0)
        return NodeAllocation(z, z, 0.0, np.zeros(0, bool), z, 0, True)
    q, f_be, b_be, tau_t, ell_t, was_feasible = sla_admit(aup, P, w, tau, ell, B, F, f_max, enforce, delta, zeta,
                                                          C > 1e-3, a > 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pre_floor = np.where(np.isfinite(tau_t) & (tau_t > 0), P / tau_t, 0.0)
    f_lo = np.where(q, np.maximum(ell_t, pre_floor), f_be)              # TPOT floor and prefill-only TTFT
    b_lo = np.where(q, 0.0, b_be)
    tau_e = np.where(q, tau_t, np.inf)
    evals = 0
    while True:
        b, f, nu, ev = sla_allocate(a, aup, C, P, w, tau_e, f_lo, b_lo, q, B, F, f_max, lam, kappa)
        evals += ev
        # primal repair: with f fixed, the TTFT targets are bandwidth floors (exact water-filling)
        slack = tau * (1.0 - 0.5 * delta) - P / f
        floors = b_lo.copy()
        good = q & (slack > 0)
        floors[good] = np.where(aup[good] > 0, aup[good] / slack[good], 0.0)
        bad = q & ~good & ((aup > 0) | (P > 0))
        if not bad.any() and floors.sum() <= B:
            b = box_waterfill(np.sqrt(w * a), B, floors, np.full(n, np.inf))
            break
        worst = np.flatnonzero(q)[np.argmax(nu[q])] if not bad.any() else np.flatnonzero(bad)[0]
        q[worst] = False                                                 # numerically marginal: demote one
        f_lo[worst], b_lo[worst], tau_e[worst] = f_be[worst], b_be[worst], np.inf
    cost = float(np.sum(w * (a / b + C / f)))
    if lam > 0:
        cost += lam * float(np.sum(kappa * C * f**2) + np.sum(e_up))
    return NodeAllocation(b, f, cost, q, nu, evals, was_feasible and bool(q.all()))


def sla_node_cvxpy(a, aup, C, P, w, tau, ell, B, F, f_max, lam=0.0, kappa=0.0):
    """Reference solution of the SLA-constrained problem with CVXPY (used only by the tests)."""
    import cvxpy as cp

    n = len(a)
    x, y = cp.Variable(n, pos=True), cp.Variable(n, pos=True)            # fractions of B and F
    obj = cp.sum(cp.multiply(np.asarray(w) * np.asarray(a) / B, cp.inv_pos(x))) \
        + cp.sum(cp.multiply(np.asarray(w) * np.asarray(C) / F, cp.inv_pos(y)))
    if lam > 0:
        obj += lam * kappa * F**2 * cp.sum(cp.multiply(np.asarray(C), cp.square(y)))
    cons = [cp.sum(x) <= 1, cp.sum(y) <= 1, y <= f_max / F, y >= np.asarray(ell) / F,
            cp.multiply(np.asarray(aup) / B, cp.inv_pos(x)) + cp.multiply(np.asarray(P) / F, cp.inv_pos(y))
            <= np.asarray(tau)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.status, (float(prob.value) if prob.value is not None else np.inf), x.value, y.value
