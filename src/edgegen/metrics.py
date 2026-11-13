"""Evaluation with the TRUE output lengths (Sec. V-A) and 95% confidence intervals.

The controller decided (x, b, f) from predicted lengths; every reported metric is recomputed with the
true o_i. A request meets its token-level SLA iff

    TTFT_i = (s_i beta + A_i) / (b_i eta_i) + rho s_i / f_i <= tau_ttft_i   and
    TPOT_i = 1 / f_i <= tau_tok_i                                            (Eq. 4)

and its node can hold it in the KV cache: if the true footprint sum_i (s_i + o_i) of a node exceeds
K_n, the requests with the lowest priority per token are counted as not served within their SLA
until the rest fits (the most favourable eviction for the scheme under test).
The completion bound T_i <= tau_ttft + o_i tau_tok is reported as a secondary metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .energy import compute_energy
from .latency import comm_time, comp_time
from .outer_solver import Solution
from .scenario import Scenario


def jain(x) -> float:
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (x.size * np.sum(x**2)))


def kv_evicted(sc: Scenario, node: np.ndarray) -> np.ndarray:
    """Requests that do not fit into their node's KV cache (lowest priority per token first)."""
    out = np.zeros(sc.M, bool)
    tok = sc.tok_true
    for n in range(sc.N):
        u = np.flatnonzero(node == n)
        excess = tok[u].sum() - sc.K[n]
        if excess <= 0:
            continue
        for i in u[np.argsort(sc.w[u] / tok[u], kind="stable")]:
            out[i] = True
            excess -= tok[i]
            if excess <= 0:
                break
    return out


def evaluate(sc: Scenario, sol: Solution) -> tuple[pd.DataFrame, dict]:
    i, n = np.arange(sc.M), sol.node
    eta = sc.eta[i, n]
    comm = comm_time(sc.D_true, sol.b, eta)
    prefill = comp_time(sc.pre_work, sol.f)
    decode = comp_time(sc.o_true, sol.f)
    T = comm + prefill + decode
    ttft = comm_time(sc.up_bits, sol.b, eta) + prefill
    tpot = 1.0 / sol.f
    energy = sc.e_up[i, n] + compute_energy(sc.kappa, sc.C_true, sol.f)
    # post-hoc static GPU energy: P_static over each node's busy period max_{i in S_n} T^comp_{i,n}
    comp = prefill + decode
    p_static = float(sc.cfg["gpu"].get("static_power_w", 0.0))
    busy = sum(float(comp[n == m].max()) for m in range(sc.N) if np.any(n == m))
    static_j = p_static * busy / sc.M
    # UE transmit power p_bar * b_i against a 23 dBm (0.2 W) cap
    p_bar_w_hz = 10.0 ** (sc.cfg["channel"]["ue_psd_dbm_hz"] / 10.0) / 1000.0
    over_cap = p_bar_w_hz * sol.b > float(sc.cfg["channel"].get("ue_pmax_w", np.inf))
    # token-level SLA (Eq. 4): TTFT and TPOT together, and the request must fit into the KV cache
    met_ttft = ttft <= sc.tau_ttft * (1 + 1e-9)
    met_tpot = tpot <= sc.tau_tok * (1 + 1e-9)
    evicted = kv_evicted(sc, n)
    met = met_ttft & met_tpot & ~evicted
    met_e2e = (T <= sc.Tmax_true) & ~evicted                     # secondary: completion bound
    guaranteed = sol.guaranteed if sol.guaranteed is not None else np.zeros(sc.M, bool)
    broken = guaranteed & ~(met_ttft & met_tpot)                 # must be empty: the guarantee of Thm. 2
    # requests that cannot meet their SLA even alone at their best node (all of B_n, f = f_max)
    alone_ttft = np.min(sc.up_bits[:, None] / (sc.B[None, :] * sc.eta), axis=1) + sc.pre_work / sc.f_max
    solo_inf = (alone_ttft > sc.tau_ttft) | (1.0 / sc.f_max > sc.tau_tok)
    kv_load = np.array([sc.tok_true[n == m].sum() / sc.K[m] for m in range(sc.N)])
    users = pd.DataFrame(dict(user=i, node=n, cls=np.array(sc.class_names)[sc.cls], weight=sc.w, latency_s=T,
                              comm_s=comm, prefill_s=prefill, decode_s=decode, ttft_s=ttft, tpot_s=tpot,
                              sla_met=met, guaranteed=guaranteed, energy_j=energy))
    agg = dict(
        wmean_latency=float(np.sum(sc.w * T) / sc.w.sum()), mean_latency=float(T.mean()),
        p95_latency=float(np.percentile(T, 95)), mean_ttft=float(ttft.mean()), p95_ttft=float(np.percentile(ttft, 95)),
        energy_total_j=float(energy.mean()) + static_j, ue_over_cap_pct=100.0 * float(over_cap.mean()),
        sla_pct=100.0 * float(met.mean()), sla_w_pct=100.0 * float(np.sum(sc.w * met) / sc.w.sum()),
        ttft_ok_pct=100.0 * float(met_ttft.mean()), tpot_ok_pct=100.0 * float(met_tpot.mean()),
        sla_e2e_pct=100.0 * float(met_e2e.mean()),
        guaranteed_pct=100.0 * float(guaranteed.mean()) if sol.guaranteed is not None else np.nan,
        guarantee_broken=int(broken.sum()),
        kv_load_pct=100.0 * float(kv_load.max()), kv_evicted_pct=100.0 * float(evicted.mean()),
        infeasible_pct=100.0 * float(solo_inf.mean()),
        energy_j=float(energy.mean()), jain=jain(sc.Tmax_true / T),
        comm_s=float(comm.mean()), prefill_s=float(prefill.mean()), decode_s=float(decode.mean()),
        objective=float(sol.objective), runtime_s=float(sol.runtime_s),
        ls_passes=float(np.mean(sol.ls_passes)) if sol.ls_passes else np.nan,
        ls_passes_max=float(np.max(sol.ls_passes)) if sol.ls_passes else np.nan,
        dual_evals=float(sol.info.get("dual_evals", np.nan)),
        all_feasible=float(sol.info["feasible_nodes"] == sol.info["used_nodes"]) if "used_nodes" in sol.info
        else np.nan,
    )
    for k, name in enumerate(sc.class_names):
        agg[f"sla_pct_{name}"] = 100.0 * float(met[sc.cls == k].mean()) if np.any(sc.cls == k) else np.nan
    return users, agg


def mean_ci(values, conf: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval over scenarios."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan), 0.0
    return float(v.mean()), float(stats.t.ppf(0.5 + conf / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))


def summarise(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and 95% CI half-width (column suffix _ci) of each metric per group."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for m in metrics:
            row[m], row[f"{m}_ci"] = mean_ci(g[m])
        row["scenarios"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)
