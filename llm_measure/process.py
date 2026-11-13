"""Turn the llama.cpp measurements into the serving profile used in Sec. V-B (profile.json).

Mapping to the model of the paper (work in decode-token equivalents, dte):
  f_max  single-stream decode rate (llama-bench tg128, tokens/s)                     [dte/s]
  F      peak aggregate decode rate over the number of concurrent sequences n        [dte/s]
  rho    cost of one prompt token in dte when the GPU is saturated = F / PP,
         PP = prefill rate (tokens/s, llama-bench pp512)
  KV     bytes of KV cache per token (llama.cpp allocation log, else the model architecture)
The fluid model predicts an aggregate decode rate min(n f_max, F) for n concurrent sequences; the
relative error of that prediction is reported per n.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def batched(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0].isdigit():
            pp, tg, B, nkv, tpp, spp, ttg, stg, t, s = cells
            rows.append(dict(npp=int(pp), ntg=int(tg), B=int(B), S_PP=float(spp), S_TG=float(stg)))
    return rows


def main():
    pp, tg = [], None
    with open(HERE / "llama_bench.csv") as fh:
        for r in csv.DictReader(fh):
            if int(r["n_gen"]) == 0:
                pp.append(dict(n_prompt=int(r["n_prompt"]), tokens_per_s=float(r["avg_ts"]), sd=float(r["stddev_ts"])))
            elif int(r["n_prompt"]) == 0:
                tg = dict(tokens_per_s=float(r["avg_ts"]), sd=float(r["stddev_ts"]))
            gpu, size, backend = r.get("gpu_info", ""), int(r["model_size"]), r.get("backends", "")
    runs = [batched(p) for p in sorted(HERE.glob("batched_bench_*.txt"))] or [batched(HERE / "batched_bench.txt")]
    rows = {}
    for run in runs:                                   # best of the repetitions (throughput measurement)
        for r in run:
            key = (r["npp"], r["B"])
            if key not in rows or r["S_TG"] > rows[key]["S_TG"]:
                rows[key] = r
    rows = sorted(rows.values(), key=lambda r: (r["npp"], r["B"]))
    f_max = tg["tokens_per_s"]
    F = max(r["S_TG"] for r in rows if r["npp"] == min(x["npp"] for x in rows))
    pp512 = next(p["tokens_per_s"] for p in pp if p["n_prompt"] == 512)
    kv_bytes, kv_src = None, "architecture"
    log = HERE / "kv_probe.log"
    if log.exists():
        m = re.search(r"size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells", log.read_text())
        if m:
            kv_bytes, kv_src = float(m.group(1)) * 2**20 / int(m.group(2)), "llama.cpp log"
    if kv_bytes is None:                               # Mistral-7B: 32 layers, 8 KV heads, head dim 128, fp16
        kv_bytes = 2 * 32 * 8 * 128 * 2
    # saturating fit S(n) = F_inf n / (n + n0) by least squares on the short-prompt rows (grid search on n0)
    short = [r for r in rows if r["npp"] == min(x["npp"] for x in rows)]
    nB, sT = np.array([r["B"] for r in short], float), np.array([r["S_TG"] for r in short], float)
    best = None
    for n0 in np.linspace(0.01, 50, 5000):
        x = nB / (nB + n0)
        finf = float(x @ sT / (x @ x))
        err = float(np.sum((finf * x - sT) ** 2))
        if best is None or err < best[0]:
            best = (err, n0, finf)
    _, n0, finf = best
    for r in rows:
        r["sat_model"] = finf * r["B"] / (r["B"] + n0)
        r["sat_err_pct"] = 100.0 * (r["sat_model"] - r["S_TG"]) / r["S_TG"]
    for r in rows:
        model = min(r["B"] * f_max, F)
        r["model"] = model
        r["rel_err_pct"] = 100.0 * (model - r["S_TG"]) / r["S_TG"]
    prof = dict(model="Mistral-7B-Instruct-v0.2, Q4_K_M (4.37 GB)", device=gpu, backend=backend,
                f_max=f_max, f_max_sd=tg["sd"], F=F, prefill_tps_512=pp512, rho=F / pp512,
                prefill_to_decode_speed=pp512 / f_max, kv_bytes_per_token=kv_bytes, kv_source=kv_src,
                kv_bytes_per_token_llama2_7b_fp16=2 * 32 * 32 * 128 * 2,
                prefill=pp, batched=rows,
                fluid_model_mae_pct=float(np.mean([abs(r["rel_err_pct"]) for r in rows])),
                sat_F_inf=finf, sat_n0=float(n0),
                sat_model_mae_pct=float(np.mean([abs(r["sat_err_pct"]) for r in rows])))
    (HERE / "profile.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps({k: v for k, v in prof.items() if k not in ("prefill", "batched")}, indent=2))
    for r in rows:
        print(f"npp={r['npp']:5d} n={r['B']:3d}  measured {r['S_TG']:7.2f} tok/s   model {r['model']:7.2f}  "
              f"err {r['rel_err_pct']:+6.1f}%   saturating {r['sat_model']:7.2f} ({r['sat_err_pct']:+5.1f}%)")


if __name__ == "__main__":
    main()
