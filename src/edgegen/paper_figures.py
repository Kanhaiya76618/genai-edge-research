"""Figures of the paper (figures/paper/*.pdf), drawn from the CSV files in results/.

Every scheme keeps one colour, marker and line style in every figure (Okabe-Ito based palette,
checked for colour-blind separation); markers and line styles are a second encoding.
Error bars and bands are 95% Student-t confidence intervals of the mean over independent scenarios
(snapshot) or independent arrival processes (dynamic): mean +/- t_{0.975, n-1} s / sqrt(n).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

W, W2 = 3.5, 7.16      # IEEE column / page width [in]
INK, MUTED, GRID = "#222222", "#666666", "#E3E3E3"
STYLE = {   # colour, marker, line style, label
    "Nearest-ES":     ("#CC79A7", "D", "--", "Nearest-ES"),
    "Comm-only":      ("#009E73", "^", "-.", "Comm-only"),
    "Compute-only":   ("#E69F00", "v", ":", "Compute-only"),
    "Load-aware":     ("#B07800", "<", "--", "Load-aware"),
    "Decoupled":      ("#D55E00", "s", "--", "Decoupled"),
    "Classical-JCC":  ("#555555", "P", "-.", "Classical-JCC"),
    "Decoupled-SLA":  ("#F08A4B", "X", ":", "Decoupled-SLA"),
    "Load-aware-SLA": ("#7A5195", ">", "-", "Load-aware-SLA"),
    "JCC-Greedy":     ("#56B4E9", "x", ":", "JCC-Greedy"),
    "JCC-Edge":       ("#0072B2", "o", "-", "JCC-Edge (proposed)"),
    "Greedy (Phi)":   ("#56B4E9", "x", ":", "Greedy"),
    "Greedy + LS (Phi)": ("#0072B2", "o", "--", "Greedy + LS"),
    "Exhaustive (N=3)": ("#444444", "*", "-", "Exhaustive (N=3)"),
}
COMPONENTS = [("comm_s", "Uplink + downlink", "#A6CEE3"), ("prefill_s", "Prefill", "#3A87C8"),
              ("decode_s", "Decode", "#1B3A6B")]


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.1,
        "lines.markersize": 4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def kw(m):
    c, mk, ls, label = STYLE[m]
    ours = m == "JCC-Edge"
    return dict(color=c, marker=mk, ls=ls, label=label, lw=1.6 if ours else 1.0, ms=4.2 if ours else 3.4,
                zorder=10 if ours else 3, mew=1.0 if mk in "x+" else 0.5, mec="white" if mk not in "x*+" else c)


def _lines(ax, d, x, y, methods, xscale=1.0, key="method"):
    for m in methods:
        g = d[d[key] == m].sort_values(x)
        if g.empty:
            continue
        xs = g[x] * xscale
        ax.plot(xs, g[y], **kw(m))
        if f"{y}_ci" in g:
            ax.fill_between(xs, g[y] - g[f"{y}_ci"], g[y] + g[f"{y}_ci"], color=STYLE[m][0], alpha=0.12, lw=0)


def _legend(fig, axes, ncol, y=1.0):
    h, lab = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=2.4,
               columnspacing=1.0)


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=8)


FIG2 = ["Comm-only", "Decoupled", "Load-aware", "Classical-JCC", "Decoupled-SLA", "Load-aware-SLA", "JCC-Edge"]


def fig2_default(T, cfg, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.22))
    d = T["fig02a_sla_per_class"].set_index("method")
    ms, cls = [m for m in FIG2 if m in d.index], cfg["classes"]
    bw = 0.84 / len(ms)
    for j, m in enumerate(ms):
        x = np.arange(len(cls)) - 0.42 + bw * (j + 0.5)
        v = np.array([d.loc[m, f"sla_pct_{c['name']}"] for c in cls])
        ci = np.array([d.loc[m, f"sla_pct_{c['name']}_ci"] for c in cls])
        a.bar(x, v, bw, color=STYLE[m][0], edgecolor="white", lw=0.4, label=STYLE[m][3], zorder=3)
        a.errorbar(x, v, yerr=[np.minimum(ci, v), np.minimum(ci, 100 - v)], fmt="none", ecolor=INK,
                   elinewidth=0.5, capsize=1, zorder=4)
    a.set_xticks(range(len(cls)))
    a.set_xticklabels([f"{c['name']} ($w$={c['weight']:g})" for c in cls])
    a.set_ylabel("SLA satisfaction (%)\nTTFT and TPOT met")
    a.set_ylim(0, 105)
    a.grid(axis="x", visible=False)
    _title(a, "(a) SLA per priority class")
    _legend(fig, [a], ncol=7, y=0.98)
    d2 = T["fig02b_latency_breakdown"].set_index("method")
    x, bottom = np.arange(len(ms)), np.zeros(len(ms))
    for col, label, c in COMPONENTS:
        v = d2.loc[ms, col].to_numpy()
        b.bar(x, v, 0.62, bottom=bottom, color=c, edgecolor="white", lw=0.5, label=label, zorder=3)
        bottom += v
    for xi, tot in zip(x, bottom):
        b.text(xi, tot + 0.02 * bottom.max(), f"{tot:.2f}", ha="center", va="bottom", fontsize=6)
    b.set_xticks(x)
    b.set_xticklabels([STYLE[m][3].replace(" (proposed)", "") for m in ms], rotation=30, ha="right", fontsize=6.5)
    b.set_ylabel("Mean latency (s)")
    b.set_ylim(0, bottom.max() * 1.42)
    b.grid(axis="x", visible=False)
    b.legend(loc="upper center", fontsize=6, ncol=3, handlelength=1.2, columnspacing=0.8)
    _title(b, "(b) Where the time goes")
    save(fig, "pfig02_default", out)


def fig3_online(T, cfg, out):
    d = T["online"]
    pol = list(cfg["experiments"]["online"]["policies"])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.32))
    _lines(ax[0], d, "rate", "wmean_latency", pol, key="policy")
    ax[0].set_yscale("log")
    yt = [v for v in (1, 2, 5, 10, 20, 50, 100) if v <= 1.5 * d.wmean_latency.max()]
    ax[0].set_yticks(yt)
    ax[0].set_yticklabels([f"{v:g}" for v in yt])
    ax[0].yaxis.set_minor_locator(plt.NullLocator())
    ax[0].set_ylabel("Weighted mean latency (s)")
    _title(ax[0], "(a) Latency")
    _lines(ax[1], d, "rate", "sla_w_pct", pol, key="policy")
    ax[1].set_ylabel("Weighted SLA satisfaction (%)")
    ax[1].set_ylim(0, 102)
    _title(ax[1], "(b) SLA (TTFT and TPOT)")
    _lines(ax[2], d, "rate", "kv_peak_pct", pol, key="policy")
    ax[2].axhline(100, color=MUTED, lw=0.8, ls=":")
    ax[2].set_ylabel("Peak KV-cache occupancy (% of $K_n$)")
    _title(ax[2], "(c) KV cache")
    for a in ax:
        a.set_xlabel("Arrival rate $\\Lambda$ (requests/s)")
    _legend(fig, ax, ncol=4, y=0.97)
    save(fig, "pfig03_online", out)


def fig4_sweeps(T, cfg, out):
    ms = list(cfg["experiments"]["sweep_methods"])
    hs = int(round(100 * cfg["layout"]["hotspot_share"]))
    M = cfg["layout"]["n_users"]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.1), gridspec_kw=dict(hspace=0.45, wspace=0.22))
    spec = [("sweep_load", "M", "sla_w_pct", 1, f"Number of users $M$ ({hs}% hotspot)", "Weighted SLA (%)",
             "(a) SLA vs. load"),
            ("sweep_load", "M", "wmean_latency", 1, f"Number of users $M$ ({hs}% hotspot)",
             "Weighted mean latency (s)", "(b) Latency vs. load"),
            ("sweep_hotspot", "hotspot", "sla_w_pct", 100, f"Users inside the hotspot (%), $M$ = {M}",
             "Weighted SLA (%)", "(c) SLA vs. spatial imbalance"),
            ("sweep_gpu", "F_dte", "sla_w_pct", 1, "GPU capacity per node $F_n$ (dte/s)", "Weighted SLA (%)",
             "(d) SLA vs. compute capacity")]
    for a, (name, x, y, sc, xl, yl, tt) in zip(ax.ravel(), spec):
        _lines(a, T[name], x, y, ms, sc)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        _title(a, tt)
        if y.startswith("sla"):
            a.set_ylim(0, 102)
    _legend(fig, ax, ncol=6, y=0.965)
    save(fig, "pfig04_sweeps", out)


def fig5_normalised(T, cfg, out):
    d = T["fig09_normalised"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25), gridspec_kw=dict(wspace=0.28))
    for a, m in zip(axes[:2], ("Decoupled", "JCC-Edge")):
        g = d[d.method == m]
        lim = 4.0
        x, y = np.minimum(g.ttft_n, lim), np.minimum(g.tpot_n, lim)
        gu = g.guaranteed.astype(bool).to_numpy()
        a.scatter(x[~gu], y[~gu], s=5, c="#D55E00", alpha=0.45, lw=0,
                  label="not guaranteed" if m == "JCC-Edge" else "SLA-unaware")
        if gu.any():
            a.scatter(x[gu], y[gu], s=5, c="#0072B2", alpha=0.45, lw=0, label="guaranteed")
        a.axvline(1, color=INK, lw=0.7, ls="--")
        a.axhline(1, color=INK, lw=0.7, ls="--")
        a.fill_between([0, 1], 0, 1, color="#0072B2", alpha=0.06, lw=0)
        a.set_xlim(0, lim * 1.02)
        a.set_ylim(0, lim * 1.02)
        a.set_xlabel("TTFT / $\\tau^{\\mathrm{ttft}}$")
        a.set_ylabel("TPOT / $\\tau^{\\mathrm{tok}}$")
        inside = 100.0 * float(((g.ttft_n <= 1) & (g.tpot_n <= 1)).mean())
        _title(a, f"({'a' if m == 'Decoupled' else 'b'}) {STYLE[m][3].replace(' (proposed)', '')}: "
                  f"{inside:.1f}% inside")
        a.legend(loc="upper right", fontsize=6, markerscale=2)
    a = axes[2]
    for m in FIG2:
        g = d[d.method == m]
        v = np.sort(np.maximum(g.ttft_n, g.tpot_n).to_numpy())
        k = kw(m)
        k.pop("marker")
        a.plot(v, np.arange(1, v.size + 1) / v.size, **k)
    a.axvline(1, color=INK, lw=0.7, ls="--")
    a.set_xscale("log")
    a.set_xlim(0.2, 8)
    a.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    a.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    a.xaxis.set_minor_locator(plt.NullLocator())
    a.set_xlabel("max(TTFT/$\\tau^{\\mathrm{ttft}}$, TPOT/$\\tau^{\\mathrm{tok}}$)")
    a.set_ylabel("CDF over requests")
    _title(a, "(c) Normalised SLA margin")
    a.legend(loc="lower right", fontsize=5.5, handlelength=2.0)
    save(fig, "pfig05_normalised", out)


def fig6_heatmap(T, cfg, out):
    d = T["fig07_heatmap"]
    Bs, Fs = sorted(d.B_mhz.unique()), sorted(d.F_dte.unique())
    Z = np.array([[d[(d.B_mhz == b) & (d.F_dte == f)].sla_gain_pp.iloc[0] for f in Fs] for b in Bs])
    fig, ax = plt.subplots(figsize=(W, 2.7))
    from matplotlib.colors import TwoSlopeNorm
    lim = max(np.abs(Z).max(), 1e-6)
    im = ax.imshow(Z, origin="lower", cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(Bs)):
        for j in range(len(Fs)):
            r, g, b, _ = im.cmap(im.norm(Z[i, j]))
            ax.text(j, i, f"{Z[i, j]:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0"), ha="center",
                    va="center", fontsize=6.5, color="white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK)
    ax.set_xticks(range(len(Fs)))
    ax.set_xticklabels([f"{f:g}" for f in Fs])
    ax.set_yticks(range(len(Bs)))
    ax.set_yticklabels([f"{b:g}" for b in Bs])
    ax.set_xlabel("GPU capacity per edge node $F_n$ (dte/s)")
    ax.set_ylabel("Bandwidth $B_n$ (MHz)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Weighted SLA gain vs. best baseline (pp)")
    cb.outline.set_linewidth(0.4)
    save(fig, "pfig06_heatmap", out)


def fig7_energy(T, cfg, out):
    d = T["fig08_energy_latency"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for m in cfg["experiments"]["lambda_methods"]:
        g = d[d.method == m].sort_values("lam")
        ax.plot(g.energy_j, g.wmean_latency, **kw(m))
        for r in g.itertuples():
            if m == "JCC-Edge":
                ax.annotate(f"$\\lambda$={r.lam:g}", (r.energy_j, r.wmean_latency), textcoords="offset points",
                            xytext=(3, 4), fontsize=5.5, color=MUTED)
    ax.set_xlabel("Mean dynamic energy per request (J)")
    ax.set_ylabel("Weighted mean latency (s)")
    ax.legend(loc="upper right", fontsize=6)
    save(fig, "pfig07_energy", out)


def fig8_opt_runtime(T, cfg, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.22))
    d = T["fig05a_optimality_gap"]
    d = d[d.load == 1.0]
    labels = [m for m in cfg["experiments"]["small_instance"]["methods"] if m in set(d.method)]
    Ms = sorted(d.M.unique())
    bw = 0.8 / len(labels)
    for j, m in enumerate(labels):
        g = d[d.method == m].set_index("M").loc[Ms]
        x = np.arange(len(Ms)) - 0.4 + bw * (j + 0.5)
        a.bar(x, g.gap_pct, bw, color=STYLE[m][0], edgecolor="white", lw=0.4, label=STYLE[m][3], zorder=3)
        a.errorbar(x, g.gap_pct, yerr=[np.minimum(g.gap_pct_ci, g.gap_pct), g.gap_pct_ci], fmt="none",
                   ecolor=INK, elinewidth=0.5, capsize=1, zorder=4)
    a.set_xticks(range(len(Ms)))
    a.set_xticklabels([str(m) for m in Ms])
    a.set_xlabel(f"Number of users $M$ ($N$ = {cfg['experiments']['small_instance']['n_nodes']})")
    a.set_ylabel("Gap to optimum of $\\sum_n\\Phi_n$ (%)")
    a.set_yscale("symlog", linthresh=1.0)
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=6, ncol=2)
    _title(a, "(a) Optimality gap (latency objective)")
    d = T["fig05b_runtime"]
    for m in ("JCC-Greedy", "JCC-Edge", "Exhaustive (N=3)"):
        g = d[d.method == m].sort_values("M")
        k = kw(m)
        k["label"] += ", median" if m != "Exhaustive (N=3)" else ""
        b.plot(g.M, g.median_ms, **k)
        b.fill_between(g.M, g.q25_ms, g.q75_ms, color=STYLE[m][0], alpha=0.15, lw=0)
    b.set_yscale("log")
    b.set_xlabel("Number of users $M$")
    b.set_ylabel("Run time (ms), one CPU core")
    b.legend(loc="lower right", fontsize=6)
    _title(b, "(b) Computation time")
    save(fig, "pfig08_opt_runtime", out)


def fig9_llm(cfg, root: Path, out):
    """Measured Mistral-7B serving profile (llama.cpp, Apple M5) against the fluid model."""
    p = root / "llm_measure" / "profile.json"
    if not p.exists():
        return
    prof = json.loads(p.read_text())
    tg = pd.DataFrame(prof["batched"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw=dict(wspace=0.28))
    for ctx, g in tg.groupby("npp"):
        g = g.sort_values("B")
        a.plot(g.B, g.S_TG, marker="o", ms=3.5, lw=1.0, label=f"measured, prompt {int(ctx)} tokens")
    Bs = np.array(sorted(tg.B.unique()), float)
    a.plot(Bs, np.minimum(Bs * prof["f_max"], prof["F"]), color=INK, ls="--", lw=1.0,
           label="fluid model min($n f^{\\max}$, $F$)")
    a.set_xscale("log", base=2)
    a.set_xticks(Bs)
    a.set_xticklabels([f"{int(v)}" for v in Bs])
    a.set_xlabel("Concurrent sequences $n$")
    a.set_ylabel("Aggregate decode rate (tokens/s)")
    a.legend(loc="upper left", fontsize=5.5)
    _title(a, "(a) Decode throughput vs. concurrency")
    pp = pd.DataFrame(prof["prefill"])
    b.plot(pp.n_prompt, pp.tokens_per_s, marker="s", ms=3.5, lw=1.0, color="#3A87C8", label="prefill (prompt tokens/s)")
    b.axhline(prof["f_max"], color="#1B3A6B", ls="--", lw=1.0, label="single-stream decode (tokens/s)")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xlabel("Prompt length (tokens)")
    b.set_ylabel("Tokens per second")
    b.legend(loc="center right", fontsize=5.5)
    _title(b, "(b) Prefill vs. decode speed")
    save(fig, "pfig09_llm", out)


def make_paper_figures(cfg, results_dir: Path, figures_dir: Path, root: Path | None = None) -> list[str]:
    setup()
    T = {p.stem: pd.read_csv(p) for p in Path(results_dir).glob("*.csv")}
    out, errors = Path(figures_dir) / "paper", []
    for fn in (fig2_default, fig3_online, fig4_sweeps, fig5_normalised, fig6_heatmap, fig7_energy, fig8_opt_runtime):
        try:
            fn(T, cfg, out)
        except Exception as exc:              # one broken figure must not hide the others
            errors.append(f"{fn.__name__}: {exc!r}")
            plt.close("all")
    if root is not None:
        try:
            fig9_llm(cfg, root, out)
        except Exception as exc:
            errors.append(f"fig9_llm: {exc!r}")
            plt.close("all")
    return errors
