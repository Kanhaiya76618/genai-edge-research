"""Outer problem and Algorithm 1 of the paper (JCC-Edge, Sec. IV).

Algorithm 1 (JCC-Edge)
    1  x <- GREEDY            users in decreasing w_i C_i; each joins the KV-feasible node whose optimal
                              cost Phi_n rises least (any node if none has KV room)
    2  x <- LOCALSEARCH(x)    single-user moves between KV-feasible nodes that decrease sum_n Phi_n
    3  x <- SLA-LOCALSEARCH   single-user moves that decrease, lexicographically, the priority weight
                              of requests that must be demoted, V(x) = sum_n V_n(S_n), and then sum_n Phi_n
    4  for every node n       exact SLA-feasibility test (Prop. 3); demote the cheapest requests to best
                              effort if needed; exact SLA-constrained shares (Prop. 4)
    5  return (x, b, f) and the set of requests whose TTFT and TPOT targets are guaranteed

Phi_n(S) is the optimal weighted latency (+ lambda energy) of node n without SLA constraints
(Propositions 1 and 2, closed form). V_n(S) is the total priority weight that the admission rule of
step 4 must demote at node n (zero iff every request of S can meet both targets there). Step 4 is
exact for the association found in steps 1-3: every request it does not demote meets both targets,
whatever the output length turns out to be (Theorem 2).

Variants (ablation): sla_assoc=False skips step 3; sla_alloc=False keeps the unconstrained shares of
Propositions 1-2; local_search_on=False skips steps 2-3 (GREEDY only, then step 4).
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from .inner_solver import bandwidth_shares, gpu_shares
from .scenario import Scenario
from .sla_solver import sla_admit, sla_node


@dataclass
class Solution:
    """A decision (x, b, f) for every user, plus bookkeeping."""

    method: str
    node: np.ndarray          # serving node of each user (x_{i,n} = 1)
    b: np.ndarray             # bandwidth [Hz]
    f: np.ndarray             # GPU share [dte/s]
    objective: float          # P1 objective (5a) with predicted lengths and the nominal weights w
    runtime_s: float
    trace: list = field(default_factory=list)       # per-iteration records (legacy dual loop, Fig. 9)
    ls_passes: list = field(default_factory=list)   # passes of every LOCALSEARCH call
    guaranteed: np.ndarray | None = None            # True: TTFT and TPOT are met by construction
    info: dict = field(default_factory=dict)


def p1_objective(sc: Scenario, node, b, f) -> float:
    """Objective (5a): sum_i w_i (a_i/b_i + C_i/f_i) + lam E_i, with predicted lengths."""
    i = np.arange(sc.M)
    T = sc.a_hat[i, node] / b + sc.C_hat / f
    obj = float(np.sum(sc.w * T))
    if sc.lam > 0:
        obj += sc.lam * float(np.sum(sc.e_up[i, node]) + np.sum(sc.kappa * sc.C_hat * f**2))
    return obj


def coef_node_cost(qb, qf, C, e_up, B, F, f_max, lam=0.0, kappa=0.0):
    """Inner cost of one node for  sum_i qb_i / b_i + qf_i / f_i (+ lam kappa C_i f_i^2 + lam e_up_i).

    With qb = w a and qf = w C this is Phi_n(S) of the paper (Propositions 1 and 2)."""
    if np.size(qb) == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    b, comm = bandwidth_shares(qb, np.ones_like(qb), B)
    f = gpu_shares(C, qf / C, F, f_max, lam, kappa)
    cost = comm + float(np.sum(qf / f))
    if lam > 0:
        cost += lam * float(np.sum(kappa * C * f**2) + np.sum(e_up))
    return cost, b, f


class NodeEvaluator:
    """Phi_n(S) and the demoted priority weight V_n(S), cached per (node, user set)."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.cache: dict = {}
        self.vcache: dict = {}
        self.qb = self.qf = None
        self.set_weights(sc.w)

    def set_weights(self, w):
        """qb_{i,n} = w_i a_{i,n}, qf_i = w_i C_i."""
        w = np.asarray(w, float)
        self.set_coefficients(w[:, None] * self.sc.a_hat, w * self.sc.C_hat)

    def set_coefficients(self, qb, qf):
        self.qb, self.qf = np.asarray(qb, float), np.asarray(qf, float)
        self.cache.clear()

    def solve(self, n: int, users: tuple):
        key = (n, users)
        if key not in self.cache:
            sc, idx = self.sc, np.fromiter(users, dtype=int, count=len(users))
            self.cache[key] = coef_node_cost(self.qb[idx, n], self.qf[idx], sc.C_hat[idx], sc.e_up[idx, n],
                                             float(sc.B[n]), float(sc.F[n]), sc.f_max, sc.lam, sc.kappa)
        return self.cache[key]

    def cost(self, n: int, users) -> float:
        return self.solve(n, tuple(sorted(users)))[0]

    def shortfall(self, n: int, users) -> float:
        """V_n(S): priority weight that the admission rule (Prop. 3 + demotion) must demote at node n."""
        key = (n, tuple(sorted(users)))
        if key not in self.vcache:
            v = 0.0
            if key[1]:
                sc, idx = self.sc, np.fromiter(key[1], dtype=int, count=len(key[1]))
                q, *_ = sla_admit(sc.a_up[idx, n], sc.pre_work[idx], sc.w[idx], sc.tau_ttft[idx],
                                  1.0 / sc.tau_tok[idx], float(sc.B[n]), float(sc.F[n]), sc.f_max)
                v = float(sc.w[idx][~q].sum())
            self.vcache[key] = v
        return self.vcache[key]

    def kv_ok(self, n: int, users) -> bool:
        """KV-cache constraint sum_{i in S} (s_i + o_hat_i) <= K_n (predicted footprint)."""
        return float(self.sc.tok_hat[list(users)].sum()) <= float(self.sc.K[n]) if users else True

    def allocate(self, node: np.ndarray):
        """Unconstrained optimal shares (Propositions 1-2) for an association."""
        b, f = np.zeros(self.sc.M), np.zeros(self.sc.M)
        for n in range(self.sc.N):
            users = np.flatnonzero(node == n)
            if users.size:
                _, b[users], f[users] = self.solve(n, tuple(users.tolist()))
        return b, f


V_TOL = 1e-9


def greedy(ev: NodeEvaluator) -> np.ndarray:
    """GREEDY: users in decreasing q^f_i (= w_i C_i) join the KV-feasible node whose Phi_n rises least
    (if no node has KV room, the node whose Phi_n rises least: overflow is then unavoidable)."""
    sc = ev.sc
    sets = [set() for _ in range(sc.N)]
    cost, kv = np.zeros(sc.N), np.zeros(sc.N)
    node = np.zeros(sc.M, dtype=int)
    for i in np.argsort(-ev.qf, kind="stable"):
        i = int(i)
        d_phi = np.array([ev.cost(n, sets[n] | {i}) - cost[n] for n in range(sc.N)])
        cov = sc.cover[i]
        d_phi = np.where(cov, d_phi, np.inf)
        fits = (kv + sc.tok_hat[i] <= sc.K) & cov
        n = int(np.flatnonzero(fits)[np.argmin(d_phi[fits])]) if fits.any() else int(np.argmin(d_phi))
        sets[n].add(i)
        cost[n] += d_phi[n]
        kv[n] += sc.tok_hat[i]
        node[i] = n
    return node


def local_search(ev: NodeEvaluator, node: np.ndarray, sla: bool = False, rel_tol: float = 1e-9):
    """LOCALSEARCH (sla=False): passes over users 1..M; each user moves at once to the KV-feasible node
    that most decreases sum_n Phi_n. SLA-LOCALSEARCH (sla=True): moves must first decrease the demoted
    weight V, or keep it and decrease sum_n Phi_n. Returns the association and the number of passes."""
    sc = ev.sc
    node = node.copy()
    sets = [set(np.flatnonzero(node == n).tolist()) for n in range(sc.N)]
    cost = np.array([ev.cost(n, sets[n]) for n in range(sc.N)])
    kv = np.array([float(sc.tok_hat[list(sets[n])].sum()) for n in range(sc.N)])
    short = np.array([ev.shortfall(n, sets[n]) for n in range(sc.N)]) if sla else np.zeros(sc.N)
    improved, passes = True, 0
    while improved:
        improved, passes = False, passes + 1
        for i in range(sc.M):
            src = int(node[i])
            rest = sets[src] - {i}
            c_wo = ev.cost(src, rest)
            # KV: the destination must have room, or (when overflow is unavoidable) end up relatively
            # less loaded than the source is now, so that moves never worsen the largest overflow
            fits = (kv + sc.tok_hat[i] <= sc.K) | ((kv[src] > sc.K[src]) & ((kv + sc.tok_hat[i]) / sc.K < kv[src] / sc.K[src]))
            fits &= sc.cover[i]
            fits[src] = True
            g_phi = np.array([(cost[src] + cost[n]) - (c_wo + ev.cost(n, sets[n] | {i}))
                              if (n != src and fits[n]) else -np.inf for n in range(sc.N)])
            if sla:
                v_wo = ev.shortfall(src, rest)
                g_v = np.array([(short[src] + short[n]) - (v_wo + ev.shortfall(n, sets[n] | {i}))
                                if (n != src and fits[n]) else -np.inf for n in range(sc.N)])
                if g_v.max() > V_TOL:                        # a move that reduces the demoted weight
                    cand = g_v >= g_v.max() - V_TOL
                else:                                        # otherwise: latency, without worsening V
                    cand = (g_v >= -V_TOL) & (g_phi > rel_tol * cost.sum())
                if not cand.any():
                    continue
                n = int(np.flatnonzero(cand)[np.argmax(g_phi[cand])])
            else:
                n = int(np.argmax(g_phi))
                if g_phi[n] <= rel_tol * cost.sum():
                    continue
            if n == src:
                continue
            sets[src].discard(i)
            sets[n].add(i)
            cost[src], cost[n] = c_wo, ev.cost(n, sets[n])
            kv[src], kv[n] = kv[src] - sc.tok_hat[i], kv[n] + sc.tok_hat[i]
            if sla:
                short[src], short[n] = v_wo, ev.shortfall(n, sets[n])
            node[i] = n
            improved = True
    return node, passes


def allocate_sla(sc: Scenario, node: np.ndarray):
    """Step 3 of Algorithm 1 at every node: exact SLA test, demotion if needed, exact shares."""
    b, f = np.zeros(sc.M), np.zeros(sc.M)
    guaranteed = np.zeros(sc.M, bool)
    evals, feasible_nodes, used = 0, 0, 0
    for n in range(sc.N):
        u = np.flatnonzero(node == n)
        if u.size == 0:
            continue
        r = sla_node(sc.a_hat[u, n], sc.a_up[u, n], sc.C_hat[u], sc.pre_work[u], sc.w[u], sc.tau_ttft[u],
                     1.0 / sc.tau_tok[u], sc.e_up[u, n], float(sc.B[n]), float(sc.F[n]), sc.f_max, sc.lam,
                     sc.kappa)
        b[u], f[u], guaranteed[u] = r.b, r.f, r.guaranteed
        evals, feasible_nodes, used = evals + r.evals, feasible_nodes + int(r.feasible), used + 1
    return b, f, guaranteed, dict(dual_evals=evals, feasible_nodes=feasible_nodes, used_nodes=used)


def jcc_edge(sc: Scenario, *, local_search_on: bool = True, sla_assoc: bool = True, sla_alloc: bool = True,
             name: str = "JCC-Edge") -> Solution:
    """Algorithm 1 (see the module docstring)."""
    t0 = time.perf_counter()
    ev = NodeEvaluator(sc)
    node, passes = greedy(ev), []
    if local_search_on:
        node, p = local_search(ev, node)
        passes.append(p)
        if sla_assoc:
            node, p = local_search(ev, node, sla=True)
            passes.append(p)
    if sla_alloc:
        b, f, guaranteed, info = allocate_sla(sc, node)
    else:
        (b, f), guaranteed, info = ev.allocate(node), None, {}
    return Solution(name, node, b, f, p1_objective(sc, node, b, f), time.perf_counter() - t0, [], passes,
                    guaranteed, info)


def with_sla_allocation(sc: Scenario, node: np.ndarray, name: str, t0: float | None = None) -> Solution:
    """A given association followed by step 3 of Algorithm 1 (used for the '-SLA' baselines)."""
    t0 = time.perf_counter() if t0 is None else t0
    b, f, guaranteed, info = allocate_sla(sc, node)
    return Solution(name, node, b, f, p1_objective(sc, node, b, f), time.perf_counter() - t0, [], [],
                    guaranteed, info)


# ----------------------------------------------------------------------------------------------
# Joint baselines that plan on a phase-unaware model of the requests
# ----------------------------------------------------------------------------------------------
def _belief(sc: Scenario, C_hat, D_hat) -> Scenario:
    return dataclasses.replace(sc, C_hat=np.asarray(C_hat, float), D_hat=np.asarray(D_hat, float))


def classical_jcc(sc: Scenario) -> Solution:
    """Classical joint MEC [9]-[11]: a task is a fixed amount of work proportional to its input
    (c_bar dte per prompt token, calibrated so that the median request has the median LLM work) and only
    the input is transmitted. Joint association by GREEDY + LOCALSEARCH on Phi and optimal shares
    (Propositions 1-2) for that model: no prefill/decode split, no output-length prediction, no
    token-level SLA."""
    t0 = time.perf_counter()
    wl, rho = sc.cfg["workload"], float(sc.cfg["gpu"]["prefill_cost"])
    s_med, o_med = float(wl["prompt_median_tokens"]), float(wl["output_median_tokens"])
    belief = _belief(sc, (rho * s_med + o_med) / s_med * sc.s, sc.up_bits)
    sol = jcc_edge(belief, sla_assoc=False, sla_alloc=False, name="Classical-JCC")
    sol.objective, sol.runtime_s = p1_objective(sc, sol.node, sol.b, sol.f), time.perf_counter() - t0
    return sol


def uniform_cost_jcc(sc: Scenario) -> Solution:
    """Algorithm 1 with a phase-unaware latency objective: a prompt token is weighted like a decode
    token (rho = 1 in C_i). The SLA constraints keep the true prefill cost."""
    t0 = time.perf_counter()
    sol = jcc_edge(_belief(sc, sc.s + (sc.C_hat - sc.pre_work), sc.D_hat), name="JCC (rho=1)")
    sol.objective, sol.runtime_s = p1_objective(sc, sol.node, sol.b, sol.f), time.perf_counter() - t0
    return sol


def lower_bound(sc: Scenario, true_lengths: bool = True) -> float:
    """Pooled lower bound on sum_i w_i T_i of ANY scheme (no SLA, lambda = 0): every request uses its
    best channel and all nodes' bandwidth and GPU capacity are pooled (Cauchy-Schwarz over nodes).
    With true_lengths the bound holds for the realised latencies."""
    D = sc.D_true if true_lengths else sc.D_hat
    C = sc.C_true if true_lengths else sc.C_hat
    a = D / sc.eta.max(axis=1)
    comm = float(np.sum(np.sqrt(sc.w * a)) ** 2 / sc.B.sum())
    f = gpu_shares(C, sc.w, float(sc.F.sum()), sc.f_max)
    return comm + float(np.sum(sc.w * C / f))
