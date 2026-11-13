# genai-edge-research

Simulator for the paper **"Integrated Networking and Computing for GenAI Serving at the Wireless Edge"**
(K. Mehta et al., IIIT Dharwad). One command regenerates **all 14 graphs (Figs. 2–9)** and **Tables III, IV
and V** of the paper. Every number in `results/` and `figures/` comes from running this code; nothing is
typed in by hand.

### Revision 3 (Dr. Jagadeesha R. B.'s comments, 13 Nov 2026) -- current version

- SLA = TTFT <= tau_ttft AND TPOT <= tau_tok (plus KV-cache fit); the old completion deadline is a reference metric.
- `sla_solver.py`: exact SLA-feasibility test (Prop. 3), exact SLA-constrained shares by a zero-gap dual (Prop. 4),
  repair step; guaranteed requests meet both targets for any output length (Theorem 2).
- `outer_solver.py`: Algorithm 1 = GREEDY + LOCALSEARCH (latency) + SLA-LOCALSEARCH (demoted weight), KV-cache
  constraint, coverage rule eta >= 0.1; the old dual-ascent loop is kept only in `legacy.py` for comparison.
- `online.py`: event-driven dynamic-arrival simulator (Poisson arrivals, KV queue, work-conserving execution,
  online guarantee = Theorem 3).
- `llm_measure/`: Mistral-7B measurement with llama.cpp (`measure.sh`, `process.py` -> `profile.json`). The model
  file (4.4 GB) is not shipped; download it as described in `measure.sh`.
- Paper source: `../paper_ieee/` (IEEE conference template); `python make_tables.py` there rebuilds the tables.
- `python -m pytest`: 114 tests, including CVXPY checks of Props. 3-4 and the guarantees.

### Revision 2 (second review, Nov 2026)

- Dual-multiplier safeguards: methods `JCC-Edge (cap)` (nu_i <= 5 w_i, `experiments.nu_cap`) and
  `JCC-Edge (freeze)` (no dual update for requests infeasible even when served alone); `results/table8_safeguard.csv`.
- Prompt-scaled TTFT budget `workload.ttft_per_prompt_token` (traces `conv+ps`, `code+ps` use rho/f_max).
- Post-hoc total energy with static GPU power (`gpu.static_power_w`, metric `energy_total_j`), share of users above
  a 23 dBm UE power cap (`ue_over_cap_pct`), and a no-duals curve in Fig. 8.
- Paired 95% CIs of latency reductions: `results/paired_ci.csv`.

### Revision 1 (response to review, Nov 2026)

- New benchmarks: **Compute-only+** (Compute-only association with optimal bandwidth), **Classical-JCC**
  (Algorithm 1 on a classical MEC task model: input-proportional work, static deadlines), **JCC (rho=1)**
  (phase-unaware token cost) and **JCC-Edge-PW** (phase-wise SLA, two multipliers per request).
- New metrics: phase-wise SLA (TTFT <= tau_ttft and TPOT <= tau_tok), TTFT/TPOT pass rates, "masked" share,
  number of LOCALSEARCH passes.
- JCC-Greedy now re-runs GREEDY(w~) at every dual step (Line 9 of Algorithm 1).
- New outputs: `results/table6_phasewise.csv`, `results/extra_ls_passes.csv`, Table IV at M = 40/60/80, and the
  composite manuscript figures in `figures/paper/` (built by `src/edgegen/paper_figures.py`).
- Trace-driven workload: `workload.source: trace` samples (prompt, output) pairs from
  `data/AzureLLMInferenceTrace_{conv,code}.csv` (Azure LLM inference traces 2023, downloaded from
  github.com/Azure/AzurePublicDataset; see that repository for its license). Results: `results/table7_trace.csv`.
- The manuscript source is in `../paper_rev1/`; `python make_tables.py` there rebuilds its table bodies from
  `results/`.

---

## 1. Quick start

You need Python 3.10 or newer. Every package installs from pip on Windows 11, macOS and Linux.

**macOS / Linux**

```bash
cd genai-edge-research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest                # 59 tests, a few seconds
python run_all.py --quick       # about 15 s: checks that everything works (results/quick, figures/quick)
python run_all.py               # full run of the paper (results/, figures/, report/dashboard.html)
```

**Windows 11 (PowerShell or Command Prompt)**

```bat
cd genai-edge-research
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
python run_all.py --quick
python run_all.py
```

If PowerShell refuses to run `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or
skip activation and use `.venv\Scripts\python.exe run_all.py`.

| Command | What it does |
|---|---|
| `python run_all.py` | Full run: every graph, table, LaTeX table, summary and the dashboard |
| `python run_all.py --quick` | Same pipeline with few scenarios, written to `results/quick`, `figures/quick` |
| `python run_all.py --plots-only` | Redraws graphs and tables from `results/raw/` without re-simulating |
| `python run_all.py --jobs 4` | Limits the number of CPU cores used (default: all) |

The full run takes **about one minute on a 10-core laptop** (a few minutes on a 4-core one). The simulation is fully seeded,
so a re-run gives identical numbers; only the measured run times in Fig. 5(b) vary a little.

---

## 2. What you get

| Output | Paper item | What it shows |
|---|---|---|
| `figures/fig02a_sla_per_class` | Fig. 2(a) | SLA satisfaction per priority class (Basic, Premium, Critical) |
| `figures/fig02b_latency_breakdown` | Fig. 2(b) | Mean latency split into radio transfer, prefill and decode |
| `figures/fig03a_latency_vs_load` | Fig. 3(a) | Weighted mean latency vs number of users (40% hotspot) |
| `figures/fig03b_sla_vs_load` | Fig. 3(b) | SLA satisfaction vs number of users |
| `figures/fig03c_latency_vs_hotspot` | Fig. 3(c) | Weighted mean latency vs share of users in the hotspot |
| `figures/fig03d_sla_vs_hotspot` | Fig. 3(d) | SLA satisfaction vs share of users in the hotspot |
| `figures/fig04a_bandwidth` | Fig. 4(a) | Weighted mean latency vs bandwidth per node |
| `figures/fig04b_gpu` | Fig. 4(b) | Weighted mean latency vs GPU throughput per node |
| `figures/fig05a_optimality_gap` | Fig. 5(a) | Gap to the exhaustive-search optimum (N = 3, M = 4–8) |
| `figures/fig05b_runtime` | Fig. 5(b) | Run time on one CPU core vs number of users |
| `figures/fig06_latency_cdf` | Fig. 6 | CDF of per-request latency, default setting |
| `figures/fig07_heatmap` | Fig. 7 | Latency reduction over the best baseline across bandwidth × GPU |
| `figures/fig08_energy_latency` | Fig. 8 | Energy–latency trade-off (sweep of λ) |
| `figures/fig09_dual_ascent` | Fig. 9 | The dual loop of Algorithm 1 on one scenario with M = 60 |
| `results/table3_default.csv` | Table III | All metrics at the default setting, mean ± 95% CI |
| `results/table4_predictor.csv` | Table IV | Effect of the output-length predictor |
| `results/table5_ablation.csv` | Table V | Ablation: greedy, duals, local search |
| `results/extra_load_uniform.csv` | Sec. V-C text | Load sweep with uniformly placed users (no figure in the paper) |
| `results/latex/table3.tex …` | Tables III–V | Ready-to-paste LaTeX versions |
| `results/results_summary.md` | all | Tables, every setting where JCC-Edge does **not** win, and **"Numbers for the paper text"**: the value for each claim in Secs. V-B to V-G |
| `report/dashboard.html` | all | One self-contained page with the 14 graphs and Table III; open it in any browser |

Each graph is saved as PNG (300 dpi) and PDF at IEEE single-column width (3.5 in), with its data in
`results/<same name>.csv`. `results/raw/` holds the per-scenario data behind everything.

---

## 3. How it works

### 3.1 Pipeline

```
config/default.yaml ──► scenario.py ──► baselines.py / outer_solver.py ──► metrics.py ──► experiments.py ──► plots.py
   (Table II)          one random        (x, b, f) decided with          delays, SLA,     mean ± 95% CI      14 graphs
                       scenario          PREDICTED output lengths        energy with the  per figure/table   + report.py
                                                                         TRUE lengths
```

1. **Scenario** (`scenario.py`). The four edge nodes sit on a 2×2 grid. 40% of the users are drawn from a
   Gaussian hotspot (σ = 120 m) around node 1; the rest are uniform over the area. For every (user, node)
   pair the spectral efficiency η = log₂(1 + γ) uses 3GPP path loss and 8 dB shadowing. Prompt and output
   lengths are log-normal; 40% of requests carry an image; each user belongs to one of three classes; the
   controller sees a predicted output length ô = o·LogN(0, σp²).
2. **Decide** (`outer_solver.py`, `baselines.py`). Each method chooses the node x, the bandwidth b and the
   GPU share f of every request, using only predicted lengths.
3. **Evaluate** (`metrics.py`). Delays, TTFT, SLA (T ≤ τ_ttft + o·τ_tok), energy and Jain's index are
   recomputed with the **true** output lengths.
4. **Repeat and aggregate** (`experiments.py`). 20 random scenarios per point (30 for Table III and
   Fig. 6, 10 for Figs. 7–8). Every method runs on the same scenarios, and results are reported as mean ±
   95% Student-t confidence interval.

### 3.2 Model ↔ code

| Paper | Code |
|---|---|
| Eq. (1) T^comm = D/(b·η), D = (s + o)β + A | `latency.data_bits`, `latency.comm_time`, `scenario.spectral_efficiency` |
| Eq. (2) T^comp = C/f, C = ρs + o | `latency.work`, `latency.comp_time` |
| TTFT = (sβ + A)/(bη) + ρs/f | `metrics.evaluate` |
| Eq. (3) E = p̄(sβ + A)/η + κ C f² | `energy.uplink_energy`, `energy.compute_energy` |
| Eq. (4) T^max = τ_ttft + o·τ_tok | `latency.deadline` |
| P1 objective (5a) | `outer_solver.p1_objective` |
| Proposition 1 (optimal bandwidth) | `inner_solver.bandwidth_shares` |
| Proposition 2 (optimal GPU shares: capped water-filling; cubic + bisection when λ > 0) | `inner_solver.gpu_shares` |
| Φ_n(S; w̃) = bandwidth term + GPU term + λ × uplink energy | `inner_solver.node_cost` |
| Algorithm 1 (JCC-Edge) | `outer_solver.jcc_edge`, `greedy`, `local_search` |

### 3.3 Algorithm 1 in the code

`jcc_edge()` follows the pseudocode line by line:

1. ν = 0 and w̃ = w (`NodeEvaluator.set_weights`).
2. **Greedy**: users in decreasing w̃ᵢCᵢ each join the node whose optimal cost Φ_n rises least. Then
   **local search**: passes over the users, moving each one to the node that most lowers Σₙ Φₙ, until no
   move helps.
3. For k = 0 … K−1 (K = 25): compute (b, f) at every node with Propositions 1–2, and compute
   gᵢ = Tᵢ/Tᵢ^max − 1 with predicted lengths.
4. Keep the **incumbent** with the smallest (Σᵢ wᵢ·1[gᵢ > 0], P1 objective). Stop if no deadline is
   predicted to be missed.
5. Dual ascent: νᵢ ← max(0, νᵢ + ε·wᵢ·gᵢ) with ε = 8, then w̃ = w + ν, and local search again from the
   current association (warm start).
6. Return the incumbent, not the last iterate, because the iterates can oscillate (Fig. 9).

Node costs are cached per user set and recomputed when w̃ changes. The variants used in Table V and
Fig. 5 are switches of the same function:
- `local_search_on=False` gives **JCC-Greedy** (greedy + duals).
- `duals=False` gives Greedy only / Greedy + LS (= "JCC-Edge (no duals)").

### 3.4 Benchmarks (`baselines.py`, `exhaustive.py`)

| Name | Association | Bandwidth | GPU |
|---|---|---|---|
| Nearest-ES | nearest node | equal | equal (capped at f_max) |
| Comm-only | strongest SINR | Proposition 1 | equal |
| Compute-only | heaviest request first → least-loaded of its two strongest nodes | equal | Proposition 2 |
| Decoupled | strongest SINR | Proposition 1 | Proposition 2 |
| JCC-Greedy | Algorithm 1 without local search | Prop. 1 | Prop. 2 |
| Exhaustive | all Nᴹ associations (N = 3, M ≤ 8), P1 without deadlines | Prop. 1 | Prop. 2 |

"Best baseline" in Figs. 3, 4, 7 and the summary means the best of Nearest-ES, Comm-only, Compute-only and
Decoupled at that point.

---

## 4. Project layout

```
genai-edge-research/
  run_all.py             one command for everything
  config/default.yaml    all parameters (Table II), each with its source
  config/quick.yaml      same model, fewer scenarios (pipeline check)
  src/edgegen/
    config.py            YAML loading
    scenario.py          layout + hotspot, channel (Eq. 1), requests, predictor
    latency.py           Eqs. (1), (2), (4)
    energy.py            Eq. (3)
    inner_solver.py      Propositions 1 and 2, node cost, CVXPY reference (tests only)
    outer_solver.py      Algorithm 1 (greedy, local search, dual ascent)
    baselines.py         Nearest-ES, Comm-only, Compute-only, Decoupled
    exhaustive.py        global optimum for small instances
    metrics.py           evaluation with true lengths, 95% CIs
    experiments.py       one experiment per figure/table, parallel runner
    plots.py             the 14 graphs
    report.py            summary, LaTeX tables, dashboard
  tests/                 pytest suite (59 tests)
  results/  figures/  report/    outputs
```

---

## 5. Parameters

All defaults equal **Table II** of the paper (marked `[Table II]` in `config/default.yaml`; a test checks
this). The code also needs the values below, which **the paper does not state yet**. Add them to Table II
or Section V-A:

| Parameter | Value used | Where |
|---|---|---|
| Log-normal shape of prompt and output length | σ = 0.8 for both (medians 200 / 150 as in the paper) | `workload.prompt_sigma`, `output_sigma` |
| Length clipping | 4 to 2048 tokens | `workload.min_tokens`, `max_tokens` |
| Hotspot centre | exactly at edge node 1 | `layout.hotspot_node` |
| Minimum UE–node distance | 10 m | `layout.min_distance_m` |
| Spectral-efficiency cap | none (pure log₂(1+γ) as in Eq. 1) | `channel.max_spectral_eff` |
| Node positions for N = 3 (Fig. 5) | ring of radius 300 m around the centre | `scenario.node_positions` |
| Capacity scaling for N = 3 (Fig. 5) | B_n and F_n × (M/N)/(40/4), i.e. same users per node as the default | `experiments.small_cfg` |
| Compute-only "load" | sum of predicted work C assigned to the node / F_n | `baselines.least_loaded_of_two_strongest` |
| P95 latency / TTFT | 95th percentile within each scenario, then averaged over scenarios | `metrics.evaluate` |
| Incumbent ranking (Alg. 1 line 6) | (Σ wᵢ 1[gᵢ > 0], P1 objective with the true weights w) | `outer_solver.jcc_edge` |
| Random seeds | `SeedSequence([2026, k])` for scenario k | `seed` |

To change a parameter, edit `config/default.yaml` and rerun `python run_all.py`. To add a sweep, add its
values under `experiments:` and a table in `experiments.make_tables`.

---

## 6. Tests

`python -m pytest` runs 59 tests:

- the delay, SLA and energy equations against a hand-calculated example;
- Propositions 1–2 against CVXPY to a relative error below 10⁻⁴, with and without the ceiling and the
  energy term;
- every method satisfies the constraints of P1 (one node per user, Σb ≤ B_n, Σf ≤ F_n, 0 < f ≤ f_max);
- exhaustive search is never worse than any heuristic;
- local search never worsens the greedy;
- Algorithm 1 returns the best iterate;
- metrics use the true lengths;
- the defaults equal Table II.

---

## 7. Model limitations (as discussed in the paper, Sec. V-H)

- Snapshot model: one request per user, no arrivals or queueing.
- Batching abstracted as processor sharing of an aggregate GPU throughput.
- No KV-cache limit, chunked prefill or prefill/decode disaggregation.
- DVFS energy model rather than a measured power curve.
- GPU parameters are representative assumptions (in dte), not measurements.
- The downlink answer travels at the uplink rate.
- Fast fading is averaged out.
