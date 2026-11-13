"""Inner problem (Sec. IV-B): optimal bandwidth and GPU shares at one node for a fixed user set S.

Proposition 1 (bandwidth):  b_i = B sqrt(w_i a_i) / sum_j sqrt(w_j a_j),  value (sum_j sqrt(w_j a_j))^2 / B
Proposition 2 (GPU):        f_i = min{f_max, phi_i(mu)},  phi_i(mu) the positive root of
                            2 lam kappa C_i phi^3 + mu phi^2 = w_i C_i;  mu = 0 if the capped phi_i(0) fit
                            into F, else the unique mu with sum_i min{f_max, phi_i(mu)} = F.
                            lam = 0: phi_i = sqrt(w_i C_i / mu)  -> capped water-filling by sorting.
Node cost Phi_n(S; w) = bandwidth term + GPU term (incl. lam kappa C f^2) + lam * uplink energy.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def bandwidth_shares(a, w, B: float):
    """Proposition 1: optimal bandwidth shares and the optimal value of sum_i w_i a_i / b_i."""
    v = np.sqrt(np.asarray(w, float) * np.asarray(a, float))
    return B * v / v.sum(), float(v.sum() ** 2 / B)


def _waterfill(v, F: float, f_max: float) -> np.ndarray:
    """lam = 0 case of Proposition 2: f_i = min(f_max, v_i / s) with sum f = F (v_i = sqrt(w_i C_i))."""
    n = v.size
    if n * f_max <= F:                                   # mu* = 0: every request runs at the ceiling
        return np.full(n, f_max)
    vs = np.sort(v)[::-1]                                # the largest requests are capped first
    tail = np.cumsum(vs[::-1])[::-1]                     # sum_{j >= k} vs_j
    rem = F - np.arange(n) * f_max                       # capacity left after capping the k largest
    with np.errstate(divide="ignore", invalid="ignore"):
        s_k = tail / rem                                 # = sqrt(mu*) if exactly k requests are capped
        ok = (rem > 0) & (vs <= f_max * s_k * (1 + 1e-12))
    return np.minimum(f_max, v / s_k[int(np.argmax(ok))])


def _cubic_root(q, e, mu: float) -> np.ndarray:
    """Positive root phi of 2 e phi^3 + mu phi^2 = q (Newton from an upper bound, monotone)."""
    u = np.cbrt(q / (2.0 * e))
    if mu > 0:
        u = np.minimum(u, np.sqrt(q / mu))
    for _ in range(12):
        u = u - (2 * e * u**3 + mu * u**2 - q) / (6 * e * u**2 + 2 * mu * u)
    return u


def gpu_shares(C, w, F: float, f_max: float, lam: float = 0.0, kappa: float = 0.0) -> np.ndarray:
    """Proposition 2: optimal GPU shares minimising sum_i (w_i C_i / f_i + lam kappa C_i f_i^2)."""
    C, w = np.asarray(C, float), np.asarray(w, float)
    if lam <= 0:
        return _waterfill(np.sqrt(w * C), F, f_max)
    q, e = w * C, lam * kappa * C
    phi = lambda mu: np.minimum(f_max, _cubic_root(q, e, mu))
    if phi(0.0).sum() <= F:
        return phi(0.0)
    hi = np.log((np.sqrt(q).sum() / F) ** 2)             # at this mu, sum sqrt(q/mu) = F >= sum phi
    lo = hi - 30.0
    while phi(np.exp(lo)).sum() <= F:
        lo -= 10.0
    mu = np.exp(brentq(lambda lm: phi(np.exp(lm)).sum() - F, lo, hi, xtol=1e-13, rtol=1e-13))
    return phi(mu)


def node_cost(a, C, w, e_up, B: float, F: float, f_max: float, lam: float = 0.0, kappa: float = 0.0):
    """Phi_n(S; w): optimal inner cost of one node and its allocation (b, f)."""
    if np.size(a) == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    b, comm = bandwidth_shares(a, w, B)
    f = gpu_shares(C, w, F, f_max, lam, kappa)
    cost = comm + float(np.sum(w * C / f))
    if lam > 0:
        cost += lam * float(np.sum(kappa * C * f**2) + np.sum(e_up))
    return cost, b, f


def node_cost_cvxpy(a, C, w, B, F, f_max, lam=0.0, kappa=0.0):
    """Reference solution of the same convex problem with CVXPY (used only by the tests)."""
    import cvxpy as cp

    n = len(a)
    beta, phi = cp.Variable(n, pos=True), cp.Variable(n, pos=True)       # shares as fractions of B and F
    obj = cp.sum(cp.multiply(np.asarray(w) * np.asarray(a) / B, cp.inv_pos(beta))) \
        + cp.sum(cp.multiply(np.asarray(w) * np.asarray(C) / F, cp.inv_pos(phi)))
    if lam > 0:
        obj += lam * kappa * F**2 * cp.sum(cp.multiply(np.asarray(C), cp.square(phi)))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(beta) <= 1, cp.sum(phi) <= 1, phi <= f_max / F])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), beta.value * B, phi.value * F
