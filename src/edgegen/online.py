"""Dynamic arrivals (Sec. V-F of the paper): an event-driven simulation of the multi-cell edge.

Requests arrive as a Poisson process; their positions, channels, lengths and classes follow the same
model as the snapshot scenarios. A request moves through four phases:
    UL (uplink of s_i beta + A_i bits)  ->  PRE (prefill, rho s_i dte; first token at its end)
    ->  DEC (o_i decode steps)  ->  DL (o_i beta bits)
Each active request holds a bandwidth slice b_i and a GPU slice f_i (processor sharing, as in P1);
the slices of a node are re-optimised at every event at that node (arrival, start, phase change,
departure) from the REMAINING work and deadlines. Each node holds at most K_n tokens of KV cache;
a request is started only if its predicted footprint s_i + o_hat_i fits, otherwise it waits in the
node's FIFO queue (as in vLLM-style engines). The true KV occupancy sum_i (s_i + generated tokens) is
tracked over time.

Policies
  Decoupled        strongest-SINR node; unconstrained optimal shares (Props. 1-2)
  Load-aware       lighter of the two strongest nodes (remaining work / F_n); Props. 1-2
  Load-aware-SLA   same node; SLA admission (Prop. 3) and SLA-constrained shares (Prop. 4)
  JCC-Edge         online Algorithm 1: among nodes with KV room, the SLA-feasible node with the
                   smallest increase of Phi_n (best effort at the Load-aware node if none is feasible);
                   SLA admission and SLA-constrained shares at every event

A request admitted with an enforced SLA keeps it: at every later event the previous slices remain
feasible for the remaining work (Theorem 3), so the exact re-optimisation never has to demote it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import with_overrides
from .inner_solver import bandwidth_shares, gpu_shares
from .scenario import generate_scenario
from .sla_solver import sla_admit, sla_node

UL, PRE, DEC, DL, DONE, WAIT = 0, 1, 2, 3, 4, -1
EPS = 1e-12


class OnlineSim:
    def __init__(self, cfg: dict, rate: float, horizon: float, warmup: float, policy: str, seed: int,
                 drain: float = 60.0):
        rng = np.random.default_rng([int(cfg["seed"]), 991, seed])
        n_req = int(rng.poisson(rate * horizon))
        sc = generate_scenario(with_overrides(cfg, {"layout.n_users": max(n_req, 1)}), 10_000 + seed)
        perm = rng.permutation(sc.M)                               # hotspot users are not all first
        self.t_arr = np.sort(rng.uniform(0.0, horizon, sc.M))
        self.sc, self.perm, self.policy = sc, perm, policy
        self.rate, self.horizon, self.warmup, self.t_stop = rate, horizon, warmup, horizon + drain
        g = lambda x: np.asarray(x)[perm]
        self.eta, self.s, self.o, self.oh = g(sc.eta), g(sc.s), g(sc.o_true), g(sc.o_hat)
        self.w, self.cls = g(sc.w), g(sc.cls)
        self.tau, self.tok = g(sc.tau_ttft), g(sc.tau_tok)
        self.upb, self.beta = g(sc.up_bits), float(cfg["workload"]["bits_per_token"])
        self.rho = float(cfg["gpu"]["prefill_cost"])
        self.B, self.F, self.K, self.fmax = sc.B, sc.F, sc.K, sc.f_max
        M = sc.M
        self.node = np.full(M, -1)
        self.phase = np.full(M, WAIT)
        self.rem = np.zeros(M)                                     # remaining bits (UL/DL) or dte (PRE/DEC)
        self.gen = np.zeros(M)                                     # generated tokens
        self.b, self.f = np.zeros(M), np.zeros(M)                 # reserved slices (policy output)
        self.br, self.fr = np.zeros(M), np.zeros(M)               # executed rates (work-conserving)
        self.enf = np.zeros(M, bool)                               # SLA enforced (guaranteed)
        self.t_start, self.t_ft = np.full(M, np.nan), np.full(M, np.nan)
        self.t_dec0, self.t_dec1, self.t_end = (np.full(M, np.nan) for _ in range(3))
        self.kv_over = np.zeros(M, bool)
        self.active = [set() for _ in range(sc.N)]
        self.queue = [[] for _ in range(sc.N)]
        self.kv_plan = np.zeros(sc.N)                              # predicted footprint of active requests
        self.kv_trace = []                                         # (t, node, true occupancy)
        self.t = 0.0

    # ------------------------------------------------------------------ helpers
    def rem_quantities(self, idx, n):
        """Remaining predicted communication a, uplink a_up, work C, prefill P, TTFT budget, TPOT floor."""
        eta = self.eta[idx, n]
        ph, rem = self.phase[idx], self.rem[idx]
        dl_bits = np.maximum(self.oh[idx] - self.gen[idx], 1.0) * self.beta
        up = np.where(ph == UL, rem, 0.0)
        a = (up + np.where(ph == DL, rem, dl_bits)) / eta
        P = np.where(ph == UL, self.rho * self.s[idx], np.where(ph == PRE, rem, 0.0))
        dec = np.where(ph <= PRE, self.oh[idx], np.where(ph == DEC, np.maximum(self.oh[idx] - self.gen[idx], 1.0), 0.0))
        C = np.maximum(P + dec, 1e-6)
        before_ft = ph <= PRE
        budget = np.where(before_ft, self.tau[idx] - (self.t - self.t_arr[idx]), np.inf)
        floor = np.where(ph <= DEC, 1.0 / self.tok[idx], 0.0)
        return a, up / eta, C, P, budget, floor

    def phi(self, idx, n) -> float:
        """Unconstrained optimal weighted cost of a set at node n (Props. 1-2) with remaining work."""
        if len(idx) == 0:
            return 0.0
        idx = np.asarray(idx)
        a, _, C, _, _, _ = self.rem_quantities(idx, n)
        _, comm = bandwidth_shares(a, self.w[idx], self.B[n])
        f = gpu_shares(C, self.w[idx], self.F[n], self.fmax)
        return comm + float(np.sum(self.w[idx] * C / f))

    def sla_feasible(self, idx, n, enforce) -> bool:
        idx = np.asarray(idx)
        a, aup, C, P, budget, floor = self.rem_quantities(idx, n)
        if np.any(enforce & (budget <= 0)):
            return False
        q, *_, feas = sla_admit(aup, P, self.w[idx], np.where(enforce, budget, 1e9),
                                np.maximum(np.where(enforce, floor, 0.0), 1e-9), self.B[n], self.F[n], self.fmax, enforce)
        return bool(np.all(q[enforce]))

    def reallocate(self, n):
        idx = np.array(sorted(self.active[n]), dtype=int)
        if idx.size == 0:
            return
        a, aup, C, P, budget, floor = self.rem_quantities(idx, n)
        if self.policy in ("Decoupled", "Load-aware"):
            self.b[idx] = bandwidth_shares(a, self.w[idx], self.B[n])[0]
            self.f[idx] = gpu_shares(C, self.w[idx], self.F[n], self.fmax)
            self.execute(n, idx)
            return
        enf = self.enf[idx]
        tau = np.where(enf, np.where(np.isfinite(budget), budget, 1e9), 1e9)
        ell = np.where(enf, floor, 0.0)
        r = sla_node(a, aup, C, P, self.w[idx], tau, np.maximum(ell, 1e-9), np.zeros(idx.size), self.B[n],
                     self.F[n], self.fmax, enforce=enf)
        lost = enf & ~r.guaranteed
        self.enf[idx[lost]] = False                                # should not happen (Theorem 3)
        self.b[idx], self.f[idx] = r.b, r.f
        self.execute(n, idx)

    def execute(self, n, idx):
        """Work-conserving execution of the reserved slices: the GPU slice of a request that is not
        computing (UL/DL phase) and the bandwidth slice of a request that is not transferring (PRE/DEC
        phase) are lent to the requests of the node that can use them, in proportion to their own slices
        (GPU rates capped at f_max). Every request progresses at least at its reserved rate."""
        ph = self.phase[idx]
        comp, comm = (ph == PRE) | (ph == DEC), (ph == UL) | (ph == DL)
        f = np.where(comp, self.f[idx], 0.0)
        spare = self.F[n] - f.sum()
        for _ in range(8):                                   # proportional sharing with the f_max cap
            room = comp & (f < self.fmax * (1 - 1e-12))
            if spare <= 1e-9 or not room.any():
                break
            add = spare * self.f[idx] * room / max(float((self.f[idx] * room).sum()), EPS)
            new = np.minimum(f + add, self.fmax)
            spare -= float((new - f).sum())
            f = new
        b = np.where(comm, self.b[idx], 0.0)
        if comm.any():
            b = b + (self.B[n] - b.sum()) * np.where(comm, self.b[idx], 0.0) / max(float(b.sum()), EPS)
        self.fr[idx], self.br[idx] = np.where(comp, f, self.f[idx]), np.where(comm, b, self.b[idx])

    def choose_node(self, i) -> int:
        """Association at arrival; also decides whether the SLA is enforced."""
        sc = self.sc
        if self.policy.startswith("Decoupled"):
            return int(np.argmax(self.eta[i]))
        if self.policy.startswith("Load-aware"):
            return self.load_aware_node(i)
        # JCC-Edge: an SLA-feasible node with KV room and the least increase of Phi_n; if the request cannot
        # be guaranteed anywhere, it is served best effort at the load-aware node
        room = self.kv_plan + self.s[i] + self.oh[i] <= self.K
        room &= np.array([len(q) == 0 for q in self.queue])
        best, best_key = None, None
        eta_min = float(sc.cfg["channel"].get("min_spectral_eff", 0.0) or 0.0)
        cover = (self.eta[i] >= eta_min) | (self.eta[i] >= self.eta[i].max())
        for n in np.flatnonzero(cover):
            idx = sorted(self.active[n] | {i})
            if room[n]:
                enf = np.array([self.enf[j] or j == i for j in idx])
                tmp = self.node[i], self.phase[i], self.rem[i]
                self.node[i], self.phase[i], self.rem[i] = n, UL, self.upb[i]
                ok = self.sla_feasible(idx, n, enf)
                d_phi = self.phi(idx, n) - self.phi(sorted(self.active[n]), n)
                self.node[i], self.phase[i], self.rem[i] = tmp
                key = (0 if ok else 1, d_phi)
            else:
                key = (2, len(self.queue[n]))
            if best_key is None or key < best_key:
                best, best_key = n, key
        return int(best) if best_key[0] == 0 else self.load_aware_node(i)

    def load_aware_node(self, i) -> int:
        """Lighter (remaining predicted work / F_n, queues included) of the two strongest nodes."""
        best2 = np.argsort(-self.eta[i])[:2]
        load = [sum(self.rho * self.s[j] + self.oh[j] for j in self.active[n] | set(self.queue[n])) / self.F[n]
                for n in best2]
        return int(best2[int(np.argmin(load))])

    def start(self, i, n):
        self.node[i], self.phase[i], self.rem[i] = n, UL, self.upb[i]
        self.t_start[i] = self.t
        self.kv_plan[n] += self.s[i] + self.oh[i]
        if self.policy.endswith("SLA") or self.policy == "JCC-Edge":
            idx = sorted(self.active[n] | {i})
            enf = np.array([self.enf[j] or j == i for j in idx])
            self.enf[i] = self.sla_feasible(idx, n, enf)
        self.active[n].add(i)

    def try_start(self, n):
        while self.queue[n]:
            i = self.queue[n][0]
            if self.kv_plan[n] + self.s[i] + self.oh[i] > self.K[n] and self.active[n]:
                break
            self.queue[n].pop(0)
            self.start(i, n)

    def kv_true(self, n) -> float:
        return float(sum(self.s[j] + self.gen[j] for j in self.active[n]))

    # ------------------------------------------------------------------ main loop
    def run(self) -> pd.DataFrame:
        M, nxt = self.sc.M, 0
        while True:
            act = [j for n in range(self.sc.N) for j in self.active[n]]
            dt_c, j_c = np.inf, -1
            for j in act:
                n = self.node[j]
                rate = self.br[j] * self.eta[j, n] if self.phase[j] in (UL, DL) else self.fr[j]
                d = self.rem[j] / rate
                if d < dt_c:
                    dt_c, j_c = d, j
            t_a = self.t_arr[nxt] if nxt < M else np.inf
            if not np.isfinite(dt_c) and not np.isfinite(t_a):
                break
            if min(self.t + dt_c, t_a) > self.t_stop:              # stop: unfinished requests are censored
                break
            completion = self.t + dt_c <= t_a
            t_new = self.t + dt_c if completion else t_a
            dt = dt_c if completion else t_a - self.t
            for j in act:                                          # advance every active request
                n = self.node[j]
                if self.phase[j] in (UL, DL):
                    self.rem[j] -= self.br[j] * self.eta[j, n] * dt
                else:
                    self.rem[j] -= self.fr[j] * dt
                    if self.phase[j] == DEC:
                        self.gen[j] = min(self.o[j], self.o[j] - self.rem[j])
            self.t = t_new
            if completion:                                         # exact: avoids steps below the time resolution
                self.rem[j_c] = 0.0
            touched = set()
            if not completion:                                     # arrival
                i = nxt
                nxt += 1
                n = self.choose_node(i)
                self.node[i] = n
                self.queue[n].append(i)
                self.try_start(n)
                touched.add(n)
            for j in act:                                          # phase completions
                if self.rem[j] > 1e-9 * max(1.0, self.upb[j]):
                    continue
                n = self.node[j]
                if self.phase[j] == UL:
                    self.phase[j], self.rem[j] = PRE, self.rho * self.s[j]
                elif self.phase[j] == PRE:
                    self.phase[j], self.rem[j] = DEC, float(self.o[j])
                    self.t_ft[j] = self.t_dec0[j] = self.t
                elif self.phase[j] == DEC:
                    self.gen[j] = self.o[j]
                    self.phase[j], self.rem[j] = DL, self.o[j] * self.beta
                    self.t_dec1[j] = self.t
                elif self.phase[j] == DL:
                    self.phase[j] = DONE
                    self.t_end[j] = self.t
                    self.active[n].discard(j)
                    self.kv_plan[n] -= self.s[j] + self.oh[j]
                    self.try_start(n)
                touched.add(n)
            for n in touched:
                occ = self.kv_true(n)
                if occ > self.K[n]:
                    for j in self.active[n]:
                        self.kv_over[j] = True
                self.kv_trace.append((self.t, n, occ))
                self.reallocate(n)
        return self.results()

    def results(self) -> pd.DataFrame:
        done = np.isfinite(self.t_end)
        cens = lambda x: np.where(np.isfinite(x), x, self.t_stop)  # unfinished: censored at the stop time
        ttft = cens(self.t_ft) - self.t_arr
        tpot = np.where(np.isfinite(self.t_dec1), (self.t_dec1 - self.t_dec0) / np.maximum(self.o, 1), np.inf)
        lat = cens(self.t_end) - self.t_arr
        met = done & (ttft <= self.tau * (1 + 1e-9)) & (tpot <= self.tok * (1 + 1e-9)) & ~self.kv_over
        return pd.DataFrame(dict(t_arr=self.t_arr, node=self.node, w=self.w, cls=self.cls, ttft=ttft, tpot=tpot,
                                 latency=lat, wait=cens(self.t_start) - self.t_arr, met=met, enforced=self.enf,
                                 done=done, kv_over=self.kv_over, tau=self.tau, tok=self.tok))


def run_online(cfg: dict, rate: float, policy: str, seed: int) -> dict:
    oc = cfg["experiments"]["online"]
    sim = OnlineSim(cfg, rate, float(oc["horizon_s"]), float(oc["warmup_s"]), policy, seed,
                    float(oc.get("drain_s", 60.0)))
    df = sim.run()
    d = df[(df.t_arr >= sim.warmup) & (df.t_arr <= sim.horizon - float(oc["cooldown_s"]))]
    kv = pd.DataFrame(sim.kv_trace, columns=["t", "node", "occ"])
    kv = kv[kv.t >= sim.warmup]
    kv_util = []
    for n, g in kv.groupby("node"):                                 # time-average occupancy per node
        tt = np.append(g.t.to_numpy(), sim.horizon)
        kv_util.append(float(np.sum(g.occ.to_numpy() * np.diff(tt)) / max(tt[-1] - tt[0], EPS) / sim.K[int(n)]))
    broken = d.enforced & d.done & ~((d.ttft <= d.tau * (1 + 1e-9)) & (d.tpot <= d.tok * (1 + 1e-9)))
    broken |= d.enforced & ~d.done & (d.ttft > d.tau)
    return dict(policy=policy, rate=rate, seed=seed, requests=len(d),
                wmean_latency=float(np.sum(d.w * d.latency) / d.w.sum()), mean_latency=float(d.latency.mean()),
                p95_latency=float(np.percentile(d.latency, 95)), mean_ttft=float(d.ttft.mean()),
                p95_ttft=float(np.percentile(d.ttft, 95)),
                p95_tpot_ms=1e3 * float(np.percentile(d.tpot[np.isfinite(d.tpot)], 95)) if d.done.any() else np.nan,
                unfinished_pct=100.0 * float((~d.done).mean()),
                mean_wait=float(d.wait.mean()), sla_pct=100.0 * float(d.met.mean()),
                sla_w_pct=100.0 * float(np.sum(d.w * d.met) / d.w.sum()),
                enforced_pct=100.0 * float(d.enforced.mean()), guarantee_broken=int(broken.sum()),
                kv_over_pct=100.0 * float(d.kv_over.mean()),
                kv_mean_pct=100.0 * float(np.mean(kv_util)) if kv_util else 0.0,
                kv_peak_pct=100.0 * float((kv.occ / sim.K[kv.node.to_numpy()]).max()) if len(kv) else 0.0)
