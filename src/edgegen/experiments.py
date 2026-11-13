"""Experiment runner: one experiment per figure / table of the paper.

A snapshot setting is (M, hotspot share, B_n, F_n, lambda, predictor, trace, KV capacity, rho). Every
(setting, scenario k) pair is solved once with the union of the methods any experiment needs, and all
methods see the same scenario. Dynamic-arrival runs (online.py) and run times are separate tasks.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .config import with_overrides
from .exhaustive import exhaustive, exhaustive_lex
from .legacy import dual_ascent_phasewise
from .metrics import evaluate, mean_ci, summarise
from .online import run_online
from .outer_solver import jcc_edge, lower_bound, uniform_cost_jcc
from .scenario import generate_scenario

OURS = "JCC-Edge"
METHODS = {
    **BASELINES,
    "JCC-Greedy": lambda sc: jcc_edge(sc, local_search_on=False, name="JCC-Greedy"),
    "JCC-Edge (no SLA-LS)": lambda sc: jcc_edge(sc, sla_assoc=False, name="JCC-Edge (no SLA-LS)"),
    OURS: jcc_edge,
    "JCC-Edge (no SLA alloc)": lambda sc: jcc_edge(sc, sla_assoc=False, sla_alloc=False,
                                                   name="JCC-Edge (no SLA alloc)"),
    "JCC (rho=1)": uniform_cost_jcc,
    "Dual ascent (prev. draft)": dual_ascent_phasewise,
    "Greedy (Phi)": lambda sc: jcc_edge(sc, local_search_on=False, sla_alloc=False, name="Greedy (Phi)"),
    "Greedy + LS (Phi)": lambda sc: jcc_edge(sc, sla_assoc=False, sla_alloc=False, name="Greedy + LS (Phi)"),
}
METRICS = ["wmean_latency", "mean_latency", "p95_latency", "mean_ttft", "p95_ttft", "sla_pct", "sla_w_pct",
           "ttft_ok_pct", "tpot_ok_pct", "sla_e2e_pct", "guaranteed_pct", "guarantee_broken", "kv_load_pct",
           "kv_evicted_pct", "infeasible_pct", "energy_j", "energy_total_j", "jain", "comm_s", "prefill_s",
           "decode_s", "runtime_s", "ls_passes", "dual_evals", "all_feasible", "lb_gap_pct"]
KEYS = ["M", "hotspot", "B_mhz", "F_dte", "lam", "predictor", "trace", "K_tok", "rho"]
STR_KEYS = ("predictor", "trace")
ABLATION = {"Load-aware-SLA": "Load-aware association + SLA allocation",
            "JCC-Greedy": "GREEDY + SLA allocation (JCC-Greedy)",
            "JCC-Edge (no SLA-LS)": "+ LOCALSEARCH on Phi",
            OURS: "+ SLA-LOCALSEARCH (JCC-Edge)",
            "JCC-Edge (no SLA alloc)": "JCC-Edge association, unconstrained shares",
            "Dual ascent (prev. draft)": "Dual ascent on TTFT/TPOT (previous draft)",
            "Classical-JCC": "Classical task model (Classical-JCC)",
            "JCC (rho=1)": "Phase-unaware latency weights (rho = 1)"}


def default_setting(cfg) -> dict:
    wl = cfg["workload"]
    return dict(M=int(cfg["layout"]["n_users"]), hotspot=float(cfg["layout"]["hotspot_share"]),
                B_mhz=float(cfg["channel"]["bandwidth_mhz"]), F_dte=float(cfg["gpu"]["throughput_dte_s"]),
                lam=float(cfg["energy"]["lambda"]), predictor=(wl["predictor"], float(wl["predictor_sigma"])),
                trace="", K_tok=float(cfg["gpu"]["kv_capacity_tokens"]), rho=float(cfg["gpu"]["prefill_cost"]))


def setting_cfg(cfg, s: dict) -> dict:
    return with_overrides(cfg, {"layout.n_users": s["M"], "layout.hotspot_share": s["hotspot"],
                                "channel.bandwidth_mhz": s["B_mhz"], "gpu.throughput_dte_s": s["F_dte"],
                                "energy.lambda": s["lam"], "workload.predictor": s["predictor"][0],
                                "workload.predictor_sigma": s["predictor"][1], "gpu.kv_capacity_tokens": s["K_tok"],
                                "gpu.prefill_cost": s["rho"]}
                         | (trace_overrides(s["trace"], s["rho"], cfg) if s.get("trace") else {}))


def trace_overrides(name: str, rho: float, cfg) -> dict:
    """Workload drawn from data/AzureLLMInferenceTrace_<name>.csv; the population medians used by the
    constant predictor and by Classical-JCC's calibration are the trace's own medians."""
    from .scenario import trace_medians

    base, _, opt = name.partition("+")
    f = f"data/AzureLLMInferenceTrace_{base}.csv"
    s_med, o_med = trace_medians(f)
    out = {"workload.source": "trace", "workload.trace_file": f, "workload.prompt_median_tokens": s_med,
           "workload.output_median_tokens": o_med}
    if opt == "ps":      # prompt-scaled TTFT budget: tau_ttft + rho s_i / f_max (isolated prefill time)
        out["workload.ttft_per_prompt_token"] = rho / float(cfg["gpu"]["f_max_dte_s"])
    return out


def plan(cfg) -> dict:
    """{experiment: (list of settings, methods, number of scenarios)}"""
    ex, sc_n, d = cfg["experiments"], cfg["scenarios"], default_setting(cfg)
    var = lambda **kw: {**d, **kw}
    sw = list(ex["sweep_methods"])
    preds = [(p["predictor"], float(p.get("predictor_sigma", d["predictor"][1]))) for p in ex["predictors"]]
    return {
        "table3": ([d], list(ex["table_methods"]), sc_n["table3"]),
        "load": ([var(M=int(m)) for m in ex["users_sweep"]], sw, sc_n["default"]),
        "hotspot": ([var(hotspot=float(h)) for h in ex["hotspot_sweep"]], sw, sc_n["default"]),
        "bandwidth": ([var(B_mhz=float(b)) for b in ex["bandwidth_sweep_mhz"]], sw, sc_n["default"]),
        "gpu": ([var(F_dte=float(f)) for f in ex["gpu_sweep_dte_s"]], sw, sc_n["default"]),
        "heatmap": ([var(B_mhz=float(b), F_dte=float(f), M=int(ex["heatmap_users"]))
                     for b in ex["heatmap_bandwidth_mhz"] for f in ex["heatmap_gpu_dte_s"]],
                    list(ex["heatmap_methods"]), sc_n["heatmap"]),
        "lambda": ([var(lam=float(l)) for l in ex["lambda_sweep"]], list(ex["lambda_methods"]), sc_n["lambda"]),
        "predictor": ([var(predictor=p, M=int(m)) for p in preds for m in ex["predictor_users"]],
                      list(ex["predictor_methods"]), sc_n["default"]),
        "ablation": ([var(M=int(m)) for m in ex["ablation_users"]], list(ABLATION), sc_n["default"]),
        "kv": ([var(M=int(ex["kv_users"]), K_tok=float(k)) for k in ex["kv_sweep_tokens"]], list(ex["kv_methods"]),
               sc_n["default"]),
        "rho": ([var(rho=float(r), M=int(m)) for r in ex["rho_sweep"] for m in ex["rho_users"]],
                list(ex["rho_methods"]), sc_n["default"]),
        "trace": ([var(M=int(m), trace=str(t)) for t in ex["traces"] for m in ex["trace_users"]],
                  list(ex["trace_methods"]), sc_n["default"]),
    }


def _freeze(s: dict) -> tuple:
    return tuple(s[k] for k in KEYS)


def build_tasks(cfg) -> list[dict]:
    merged: dict = {}
    for name, (settings, methods, n) in plan(cfg).items():
        for s in settings:
            for k in range(int(n)):
                t = merged.setdefault((_freeze(s), k), dict(kind="run", setting=s, k=k, methods=[], details=False))
                t["methods"] += [m for m in methods if m not in t["methods"]]
                t["details"] |= name == "table3"
    tasks = list(merged.values())
    si = cfg["experiments"]["small_instance"]
    tasks += [dict(kind="small", M=int(m), k=k, load=float(lf)) for lf in si["load_factors"] for m in si["users"]
              for k in range(cfg["scenarios"]["small"])]
    oc = cfg["experiments"]["online"]
    tasks += [dict(kind="online", rate=float(r), policy=p, seed=s) for r in oc["rates"] for p in oc["policies"]
              for s in range(int(oc["seeds"]))]
    for t in tasks:
        t["cfg"] = cfg
    cost = lambda t: (t["setting"]["M"] ** 2 * len(t["methods"]) if t["kind"] == "run" else
                      3 ** t["M"] / 20 if t["kind"] == "small" else 4000 * t["rate"])
    return sorted(tasks, key=lambda t: -cost(t))


def small_cfg(cfg, M: int, load: float = 1.0) -> dict:
    """N = 3; capacities scaled so that each node carries `load` times the default load per node."""
    n = int(cfg["experiments"]["small_instance"]["n_nodes"])
    scale = (M / n) / (cfg["layout"]["n_users"] / cfg["layout"]["n_nodes"]) / load
    return with_overrides(cfg, {"layout.n_users": M, "layout.n_nodes": n, "energy.lambda": 0.0,
                                "channel.bandwidth_mhz": cfg["channel"]["bandwidth_mhz"] * scale,
                                "gpu.throughput_dte_s": cfg["gpu"]["throughput_dte_s"] * scale,
                                "gpu.kv_capacity_tokens": cfg["gpu"]["kv_capacity_tokens"] * scale})


def run_task(t: dict) -> dict:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = t["cfg"]
    if t["kind"] == "online":
        return dict(kind="online", row=run_online(cfg, t["rate"], t["policy"], t["seed"]))
    if t["kind"] == "small":
        sc = generate_scenario(small_cfg(cfg, t["M"], t["load"]), t["k"])
        opt = exhaustive(sc)
        v_opt, phi_opt = exhaustive_lex(sc)
        rows = [dict(M=t["M"], k=t["k"], load=t["load"], method="Exhaustive", objective=opt.objective, gap_pct=0.0,
                     runtime_s=opt.runtime_s, demoted_w=v_opt, demoted_gap=0.0)]
        for label in cfg["experiments"]["small_instance"]["methods"]:
            sol = METHODS[label](sc)
            from .outer_solver import NodeEvaluator
            ev = NodeEvaluator(sc)
            v = sum(ev.shortfall(n, tuple(np.flatnonzero(sol.node == n).tolist())) for n in range(sc.N))
            rows.append(dict(M=t["M"], k=t["k"], load=t["load"], method=label, objective=sol.objective,
                             gap_pct=100.0 * (sol.objective - opt.objective) / opt.objective, runtime_s=sol.runtime_s,
                             demoted_w=v, demoted_gap=v - v_opt))
        return dict(kind="small", rows=rows)
    s = t["setting"]
    sc = generate_scenario(setting_cfg(cfg, s), t["k"])
    lb = lower_bound(sc) / sc.w.sum()
    rows, users = [], []
    for m in t["methods"]:
        u, agg = evaluate(sc, METHODS[m](sc))
        agg["lb_gap_pct"] = 100.0 * (agg["wmean_latency"] - lb) / lb
        rows.append({**{k: (str(s[k]) if k in STR_KEYS else s[k]) for k in KEYS}, "k": t["k"], "method": m, **agg})
        if t["details"]:
            users.append(u.assign(method=m, k=t["k"], tau=sc.tau_ttft, tok=sc.tau_tok))
    return dict(kind="run", rows=rows, users=pd.concat(users) if users else None)


def run_timing(cfg) -> pd.DataFrame:
    """Run time of JCC-Greedy / JCC-Edge vs M and of exhaustive search (N = 3), one process."""
    d, rows = default_setting(cfg), []
    for M in cfg["experiments"]["runtime_users"]:
        for k in range(cfg["scenarios"]["runtime"]):
            sc = generate_scenario(setting_cfg(cfg, {**d, "M": int(M)}), k)
            for m in ("JCC-Greedy", OURS):
                sol = METHODS[m](sc)
                rows.append(dict(method=m, M=int(M), k=k, runtime_ms=1e3 * sol.runtime_s,
                                 ls_passes=float(np.sum(sol.ls_passes)) if sol.ls_passes else 0.0))
    for M in cfg["experiments"]["small_instance"]["users"]:
        for k in range(cfg["scenarios"]["runtime"]):
            rows.append(dict(method="Exhaustive (N=3)", M=int(M), k=k,
                             runtime_ms=1e3 * exhaustive(generate_scenario(small_cfg(cfg, int(M)), k)).runtime_s))
    return pd.DataFrame(rows)


def run_all(cfg, n_jobs: int = 0, verbose: bool = True) -> dict:
    tasks = build_tasks(cfg)
    n_jobs = n_jobs or int(cfg.get("n_jobs", 0)) or (os.cpu_count() or 1)
    t0, out = time.perf_counter(), []
    if verbose:
        print(f"[run] {len(tasks)} tasks on {n_jobs} worker(s)", flush=True)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if verbose and (i == len(tasks) or i % max(1, len(tasks) // 10) == 0):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)} done, {el:.0f} s elapsed, ~{el / i * (len(tasks) - i):.0f} s left", flush=True)
    if verbose:
        print("[run] timing pass (single process) ...", flush=True)
    return dict(runs=pd.DataFrame([r for o in out if o["kind"] == "run" for r in o["rows"]]),
                users=pd.concat([o["users"] for o in out if o["kind"] == "run" and o["users"] is not None]),
                small=pd.DataFrame([r for o in out if o["kind"] == "small" for r in o["rows"]]),
                online=pd.DataFrame([o["row"] for o in out if o["kind"] == "online"]),
                timing=run_timing(cfg))


# ----------------------------------------------------------------------------------------------
# Tables (one CSV per figure / table of the paper)
# ----------------------------------------------------------------------------------------------
def _select(runs: pd.DataFrame, s: dict, n: int | None = None, **free) -> pd.DataFrame:
    m = np.ones(len(runs), bool)
    for key in KEYS:
        if key in free:
            continue
        val = str(s[key]) if key in STR_KEYS else s[key]
        if key in STR_KEYS:
            m &= (runs[key].fillna("").astype(str) == val).to_numpy()
        else:
            m &= np.isclose(runs[key].astype(float), float(val))
    d = runs[m]
    return d[d.k < n] if n is not None else d


def paired(runs, s: dict, n: int, a: str, b: str, col: str = "wmean_latency", rel: bool = True) -> dict:
    """Paired comparison over scenarios (all methods run on the same scenarios): mean of the per-scenario
    reduction of a relative to b (%, rel=True) or difference b - a (rel=False), with a 95% t-interval."""
    g = _select(runs, s, n)
    pa = g[g.method == a].set_index("k")[col]
    pb = g[g.method == b].set_index("k")[col]
    k = pa.index.intersection(pb.index)
    d = 100.0 * (pb[k] - pa[k]) / pb[k] if rel else (pa[k] - pb[k])
    m, h = mean_ci(d)
    return dict(a=a, b=b, metric=col, n=len(k), mean=m, ci=h, wins=int((d > 0).sum()), ties=int((d == 0).sum()))


def make_tables(cfg, raw: dict) -> dict:
    runs, users = raw["runs"], raw["users"]
    d, ex, n = default_setting(cfg), cfg["experiments"], cfg["scenarios"]
    order = {m: i for i, m in enumerate(list(METHODS))}
    srt = lambda df: df.sort_values("method", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    T = {}
    t3 = _select(runs, d, n["table3"])
    t3 = t3[t3.method.isin(ex["table_methods"])]
    T["table3_default"] = srt(summarise(t3, ["method"], METRICS + [f"sla_pct_{c['name']}" for c in cfg["classes"]]))
    T["fig02a_sla_per_class"] = T["table3_default"][["method"] + [c for c in T["table3_default"].columns
                                                                  if c.startswith("sla_pct_")]]
    T["fig02b_latency_breakdown"] = T["table3_default"][["method", "comm_s", "prefill_s", "decode_s", "mean_latency"]]
    u = users[users.k < n["table3"]]
    T["fig09_normalised"] = u.assign(ttft_n=u.ttft_s / u.tau, tpot_n=u.tpot_s / u.tok)[
        ["method", "k", "user", "cls", "weight", "guaranteed", "ttft_n", "tpot_n", "latency_s"]].reset_index(drop=True)
    sm = ["wmean_latency", "sla_pct", "sla_w_pct", "p95_latency", "guaranteed_pct", "kv_load_pct", "kv_evicted_pct",
          "lb_gap_pct"]
    for name, col, values in (("sweep_load", "M", ex["users_sweep"]), ("sweep_hotspot", "hotspot", ex["hotspot_sweep"]),
                              ("sweep_bandwidth", "B_mhz", ex["bandwidth_sweep_mhz"]),
                              ("sweep_gpu", "F_dte", ex["gpu_sweep_dte_s"])):
        sel = _select(runs, d, n["default"], **{col: True})
        sel = sel[sel.method.isin(ex["sweep_methods"]) & np.isclose(sel[col].to_numpy(float)[:, None],
                                                                     np.asarray(values, float)[None]).any(axis=1)]
        T[name] = srt(summarise(sel, ["method", col], sm)).sort_values(
            ["method", col], key=lambda c: c.map(order) if c.name == "method" else c, kind="stable")
    hm = []
    for B in ex["heatmap_bandwidth_mhz"]:
        for F in ex["heatmap_gpu_dte_s"]:
            s = {**d, "B_mhz": float(B), "F_dte": float(F), "M": int(ex["heatmap_users"])}
            g = _select(runs, s, n["heatmap"]).groupby("method")
            sla, lat = g.sla_w_pct.mean(), g.wmean_latency.mean()
            base = [m for m in ex["heatmap_methods"] if m != OURS]
            best_sla = sla[base].idxmax()
            pr = paired(runs, s, n["heatmap"], OURS, best_sla, "sla_w_pct", rel=False)
            hm.append(dict(B_mhz=B, F_dte=F, best_baseline=best_sla, best_sla=sla[best_sla], jcc_sla=sla[OURS],
                           sla_gain_pp=sla[OURS] - sla[best_sla], sla_gain_ci=pr["ci"],
                           lat_change_pct=100.0 * (lat[OURS] - lat[best_sla]) / lat[best_sla]))
    T["fig07_heatmap"] = pd.DataFrame(hm)
    lam = _select(runs, d, n["lambda"], lam=True)
    T["fig08_energy_latency"] = srt(summarise(lam[lam.method.isin(ex["lambda_methods"])], ["method", "lam"],
                                              ["wmean_latency", "energy_j", "energy_total_j", "sla_pct"]))
    pr = []
    for p in ex["predictors"]:
        key = (p["predictor"], float(p.get("predictor_sigma", d["predictor"][1])))
        for M in ex["predictor_users"]:
            g = _select(runs, {**d, "predictor": key, "M": int(M)}, n["default"])
            for m, gg in g.groupby("method"):
                pr.append(dict(predictor=p["label"], M=M, method=m, wmean_latency=gg.wmean_latency.mean(),
                               sla_pct=gg.sla_pct.mean(), sla_w_pct=gg.sla_w_pct.mean(),
                               kv_evicted_pct=gg.kv_evicted_pct.mean(), broken=int(gg.guarantee_broken.sum())))
    T["table_predictor"] = pd.DataFrame(pr)
    ab = _select(runs, d, n["default"], M=True)
    ab = ab[ab.M.isin(ex["ablation_users"]) & ab.method.isin(list(ABLATION))]
    T["table_ablation"] = srt(summarise(ab, ["method", "M"], ["wmean_latency", "sla_pct", "sla_w_pct", "runtime_s",
                                                              "guaranteed_pct", "lb_gap_pct"]))
    kv = _select(runs, {**d, "M": int(ex["kv_users"])}, n["default"], K_tok=True)
    kv = kv[kv.method.isin(ex["kv_methods"])]
    T["table_kv"] = srt(summarise(kv, ["method", "K_tok"], ["wmean_latency", "sla_pct", "sla_w_pct", "kv_load_pct",
                                                           "kv_evicted_pct"]))
    rh = _select(runs, d, n["default"], rho=True, M=True)
    rh = rh[rh.M.isin(ex["rho_users"]) & rh.method.isin(ex["rho_methods"])]
    T["table_rho"] = srt(summarise(rh, ["method", "rho", "M"], ["wmean_latency", "sla_pct", "sla_w_pct"]))
    tr = []
    for t in ex["traces"]:
        g = _select(runs, {**d, "trace": str(t)}, n["default"], M=True)
        g = g[g.M.isin(ex["trace_users"]) & g.method.isin(ex["trace_methods"])]
        if len(g):
            tr.append(srt(summarise(g, ["method", "M"], ["wmean_latency", "sla_pct", "sla_w_pct", "p95_ttft",
                                                         "infeasible_pct", "guaranteed_pct"])).assign(trace=t))
    T["table_trace"] = pd.concat(tr, ignore_index=True)
    rows = []
    for b in ("Load-aware-SLA", "Decoupled-SLA", "Classical-JCC", "Load-aware", "JCC-Greedy"):
        for col, rel in (("wmean_latency", True), ("sla_w_pct", False), ("sla_pct", False)):
            rows.append({"setting": "default", **paired(runs, d, n["table3"], OURS, b, col, rel)})
    for M in ex["users_sweep"]:
        for b in ("Load-aware-SLA", "Classical-JCC"):
            for col, rel in (("wmean_latency", True), ("sla_w_pct", False)):
                rows.append({"setting": f"load M={M}", **paired(runs, {**d, "M": int(M)}, n["default"], OURS, b,
                                                                col, rel)})
    for t in ex["traces"]:
        for M in ex["trace_users"]:
            for b in ("Load-aware-SLA", "Classical-JCC"):
                for col, rel in (("wmean_latency", True), ("sla_w_pct", False)):
                    rows.append({"setting": f"trace {t} M={M}",
                                 **paired(runs, {**d, "trace": str(t), "M": int(M)}, n["default"], OURS, b, col, rel)})
    T["paired_ci"] = pd.DataFrame(rows)
    sm_ = raw["small"]
    T["fig05a_optimality_gap"] = summarise(sm_[sm_.method != "Exhaustive"], ["method", "load", "M"],
                                           ["gap_pct", "demoted_gap"]).assign(
        max_gap_pct=lambda x: [sm_[(sm_.method == a) & (sm_.M == b) & (sm_.load == c)].gap_pct.max()
                               for a, b, c in zip(x.method, x.M, x.load)])
    tm = raw["timing"]
    T["fig05b_runtime"] = tm.groupby(["method", "M"], sort=False).runtime_ms.describe(
        percentiles=[0.25, 0.5, 0.75])[["25%", "50%", "75%"]].rename(
        columns={"25%": "q25_ms", "50%": "median_ms", "75%": "q75_ms"}).reset_index()
    on = raw["online"]
    T["online"] = summarise(on, ["policy", "rate"], ["wmean_latency", "mean_latency", "p95_latency", "mean_ttft",
                                                     "p95_ttft", "p95_tpot_ms", "mean_wait", "sla_pct", "sla_w_pct",
                                                     "enforced_pct", "guarantee_broken", "kv_over_pct", "unfinished_pct",
                                                     "kv_mean_pct", "kv_peak_pct", "requests"])
    return T
