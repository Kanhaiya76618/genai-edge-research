"""Regenerate every table and figure of the paper with one command.

    python run_all.py               full run  -> results/, figures/paper/, results/results_summary.md
    python run_all.py --quick       ~2-minute pipeline check -> results/quick/, figures/quick/
    python run_all.py --plots-only  redraw figures and tables from results/raw/ without re-simulating
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker; we parallelise over scenarios

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from edgegen.config import load_config  # noqa: E402
from edgegen.experiments import make_tables, run_all  # noqa: E402
from edgegen.paper_figures import make_paper_figures  # noqa: E402
from edgegen.report import write_summary  # noqa: E402

RAW = ("runs", "users", "small", "online", "timing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small pipeline check (config/quick.yaml)")
    ap.add_argument("--plots-only", action="store_true", help="redraw from results/raw without simulating")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all CPU cores)")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config" / ("quick.yaml" if args.quick else "default.yaml"))
    sub = "quick" if args.quick else ""
    res, figs = ROOT / "results" / sub, ROOT / "figures" / sub
    raw_dir = res / "raw"
    t0 = time.perf_counter()
    if args.plots_only:
        raw = {k: pd.read_csv(raw_dir / f"{k}.csv") for k in RAW}
    else:
        raw = run_all(cfg, n_jobs=args.jobs)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for k in RAW:
            raw[k].to_csv(raw_dir / f"{k}.csv", index=False)
        raw = {k: pd.read_csv(raw_dir / f"{k}.csv") for k in RAW}     # identical path for --plots-only

    tables = make_tables(cfg, raw)
    for name, df in tables.items():
        df.to_csv(res / f"{name}.csv", index=False)
    errors = make_paper_figures(cfg, res, figs, ROOT)
    minutes = None if args.plots_only else (time.perf_counter() - t0) / 60
    summary = write_summary(cfg, tables, res, "quick" if args.quick else "full", minutes)
    print(f"\nSummary: {summary}\nFigures: {figs / 'paper'}")
    if minutes:
        print(f"Total time: {minutes:.1f} min")
    if errors:
        print("Figure errors:\n  " + "\n  ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":      # required on Windows (multiprocessing uses 'spawn')
    raise SystemExit(main())
