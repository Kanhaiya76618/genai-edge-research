"""One random scenario of P1 (Sec. III-A, V-A): layout with a crowd hotspot, channel, requests.

Scenario k of a data point uses ``numpy.random.SeedSequence([seed, k])``, split into independent
streams (layout, channel, requests, predictor). Sweeps over B_n, F_n, lambda and the predictor
therefore see exactly the same users and channels, and every method is run on the same scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .energy import uplink_energy
from .latency import data_bits, deadline, uplink_bits, work


def db_to_lin(x_db):
    return np.power(10.0, np.asarray(x_db, dtype=float) / 10.0)


def node_positions(n: int, area: float) -> np.ndarray:
    """k x k grid at cell centres when N = k^2 (N = 4: (250,250),(750,250),(250,750),(750,750) m);
    otherwise N points on a ring of radius 0.3 * area around the centre (used for N = 3)."""
    k = int(round(np.sqrt(n)))
    if k * k == n:
        c = (np.arange(k) + 0.5) * area / k
        xx, yy = np.meshgrid(c, c)
        return np.column_stack([xx.ravel(), yy.ravel()])
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    return area / 2 + 0.3 * area * np.column_stack([np.cos(ang), np.sin(ang)])


def user_positions(rng, M: int, area: float, share: float, sigma: float, centre: np.ndarray) -> tuple:
    """A share of the users in a Gaussian crowd hotspot around ``centre``, the rest uniform."""
    n_hot = int(round(share * M))
    hot = np.clip(rng.normal(centre, sigma, size=(n_hot, 2)), 0.0, area)
    uni = rng.uniform(0.0, area, size=(M - n_hot, 2))
    return np.vstack([hot, uni]), np.arange(M) < n_hot


def spectral_efficiency(dist_m, ch: dict, rng) -> np.ndarray:
    """eta_{i,n} = log2(1 + p_bar g_{i,n} / (N0 NF + I0))  (Eq. 1; PSD form, independent of b)."""
    pl_db = ch["pathloss_a_db"] + ch["pathloss_b_db"] * np.log10(dist_m / 1000.0)
    gain_db = -(pl_db + rng.normal(0.0, ch["shadowing_std_db"], size=dist_m.shape))
    snr_db = ch["ue_psd_dbm_hz"] + gain_db - ch["noise_psd_dbm_hz"] - ch["noise_figure_db"] \
        - ch["interference_margin_db"]
    eta = np.log2(1.0 + db_to_lin(snr_db))
    return np.minimum(eta, ch["max_spectral_eff"]) if ch.get("max_spectral_eff") else eta


def lognormal_lengths(rng, median, sigma, size, lo, hi):
    return np.clip(np.round(median * rng.lognormal(0.0, sigma, size=size)), lo, hi)


@lru_cache(maxsize=8)
def load_trace(path: str) -> np.ndarray:
    """(prompt tokens, output tokens) pairs of a public LLM serving trace, e.g. the Azure LLM inference
    trace 2023 (columns TIMESTAMP, ContextTokens, GeneratedTokens) [Patel et al., Splitwise]."""
    import pandas as pd

    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    d = pd.read_csv(p)
    return d[["ContextTokens", "GeneratedTokens"]].to_numpy(dtype=float)


def trace_medians(path: str) -> tuple[float, float]:
    t = load_trace(path)
    return float(np.median(t[:, 0])), float(np.median(t[:, 1]))


def predict_lengths(rng, o_true, wl: dict) -> np.ndarray:
    """Predicted output length o_hat used by the controller (Sec. III-A, Table IV)."""
    mode = wl["predictor"]
    if mode == "oracle":
        return o_true.astype(float)
    if mode == "median":
        return np.full(o_true.shape, float(wl["output_median_tokens"]))
    if mode == "noisy":
        o = o_true * rng.lognormal(0.0, wl["predictor_sigma"], size=o_true.shape)
        return np.clip(np.round(o), wl["min_tokens"], wl["max_tokens"])
    raise ValueError(f"unknown predictor '{mode}'")


@dataclass
class Scenario:
    """All data of one scenario; arrays indexed by user i (M) and/or node n (N)."""

    cfg: dict
    M: int
    N: int
    user_xy: np.ndarray
    node_xy: np.ndarray
    in_hotspot: np.ndarray
    dist_m: np.ndarray        # (M, N)
    eta: np.ndarray           # (M, N) spectral efficiency [bit/s/Hz]
    B: np.ndarray             # (N,) bandwidth [Hz]
    F: np.ndarray             # (N,) GPU throughput [dte/s]
    K: np.ndarray             # (N,) KV-cache capacity [tokens]
    f_max: float              # single-stream ceiling [dte/s]
    s: np.ndarray             # prompt tokens
    o_true: np.ndarray        # true output tokens (evaluation only)
    o_hat: np.ndarray         # predicted output tokens (controller only)
    attach_bits: np.ndarray
    w: np.ndarray             # priority weight
    cls: np.ndarray           # class index
    class_names: list
    up_bits: np.ndarray       # s beta + A
    D_hat: np.ndarray         # Eq. (1) with o_hat
    D_true: np.ndarray        # Eq. (1) with o
    C_hat: np.ndarray         # Eq. (2) with o_hat [dte]
    C_true: np.ndarray        # Eq. (2) with o [dte]
    pre_work: np.ndarray      # rho s [dte]
    Tmax_hat: np.ndarray      # Eq. (4) with o_hat
    Tmax_true: np.ndarray     # Eq. (4) with o
    tau_ttft: np.ndarray      # TTFT allowance of the user's class [s]
    tau_tok: np.ndarray       # per-output-token allowance of the user's class [s]
    e_up: np.ndarray          # (M, N) uplink energy [J], Eq. (3)
    kappa: float
    lam: float

    @property
    def a_hat(self) -> np.ndarray:
        """a_{i,n} = D_i / eta_{i,n} with predicted lengths, so that T^comm = a / b."""
        return self.D_hat[:, None] / self.eta

    @property
    def a_up(self) -> np.ndarray:
        """(s_i beta + A_i) / eta_{i,n}: the part of a_{i,n} sent before the first token (TTFT)."""
        return self.up_bits[:, None] / self.eta

    @property
    def tok_hat(self) -> np.ndarray:
        """Predicted KV-cache footprint s_i + o_hat_i [tokens] (controller)."""
        return self.s + self.o_hat

    @property
    def tok_true(self) -> np.ndarray:
        """True KV-cache footprint s_i + o_i [tokens] at the end of generation (evaluation)."""
        return self.s + self.o_true

    @property
    def cover(self) -> np.ndarray:
        """(M, N) coverage: node n may serve user i if eta_{i,n} >= eta_min or n is i's strongest node."""
        eta_min = float(self.cfg["channel"].get("min_spectral_eff", 0.0) or 0.0)
        return (self.eta >= eta_min) | (self.eta >= self.eta.max(axis=1, keepdims=True))

    def nearest(self) -> np.ndarray:
        return np.argmin(self.dist_m, axis=1)

    def strongest(self) -> np.ndarray:
        return np.argmax(self.eta, axis=1)


def generate_scenario(cfg: dict, k: int) -> Scenario:
    """Draw scenario ``k`` of the configuration ``cfg``."""
    r_lay, r_ch, r_req, r_pred = (np.random.default_rng(s) for s in
                                  np.random.SeedSequence([int(cfg["seed"]), int(k)]).spawn(4))
    lay, ch, wl, gpu = cfg["layout"], cfg["channel"], cfg["workload"], cfg["gpu"]
    M, N, area = int(lay["n_users"]), int(lay["n_nodes"]), float(lay["area_m"])
    node_xy = node_positions(N, area)
    user_xy, hot = user_positions(r_lay, M, area, float(lay["hotspot_share"]), float(lay["hotspot_sigma_m"]),
                                  node_xy[int(lay["hotspot_node"]) % N])
    dist = np.maximum(np.linalg.norm(user_xy[:, None] - node_xy[None], axis=-1), lay["min_distance_m"])
    eta = spectral_efficiency(dist, ch, r_ch)

    if wl.get("source", "synthetic") == "trace":            # (s_i, o_i) pairs drawn jointly from a real trace
        pairs = load_trace(str(wl["trace_file"]))[r_req.integers(0, len(load_trace(str(wl["trace_file"]))), M)]
        s, o = np.maximum(pairs[:, 0], 1.0), np.maximum(pairs[:, 1], 1.0)
    else:
        s = lognormal_lengths(r_req, wl["prompt_median_tokens"], wl["prompt_sigma"], M, wl["min_tokens"],
                              wl["max_tokens"])
        o = lognormal_lengths(r_req, wl["output_median_tokens"], wl["output_sigma"], M, wl["min_tokens"],
                              wl["max_tokens"])
    has_img = r_req.random(M) < wl["attach_prob"]
    attach = np.where(has_img, r_req.uniform(wl["attach_kb_min"], wl["attach_kb_max"], M) * 8000.0, 0.0)
    classes = cfg["classes"]
    share = np.array([c["share"] for c in classes], float)
    cls = r_req.choice(len(classes), size=M, p=share / share.sum())
    o_hat = predict_lengths(r_pred, o, wl)

    beta, rho = float(wl["bits_per_token"]), float(gpu["prefill_cost"])
    ttft = np.array([c["ttft_s"] for c in classes])[cls] + float(wl.get("ttft_per_prompt_token", 0.0)) * s
    tok = np.array([c["tok_s"] for c in classes])[cls]
    up = uplink_bits(s, attach, beta)
    psd_w_hz = db_to_lin(ch["ue_psd_dbm_hz"]) / 1000.0
    return Scenario(
        cfg=cfg, M=M, N=N, user_xy=user_xy, node_xy=node_xy, in_hotspot=hot, dist_m=dist, eta=eta,
        B=np.full(N, float(ch["bandwidth_mhz"]) * 1e6), F=np.full(N, float(gpu["throughput_dte_s"])),
        K=np.full(N, float(gpu.get("kv_capacity_tokens", np.inf))),
        f_max=float(gpu["f_max_dte_s"]), s=s, o_true=o, o_hat=o_hat, attach_bits=attach,
        w=np.array([c["weight"] for c in classes], float)[cls], cls=cls, class_names=[c["name"] for c in classes],
        up_bits=up, D_hat=data_bits(s, o_hat, attach, beta), D_true=data_bits(s, o, attach, beta),
        C_hat=work(s, o_hat, rho), C_true=work(s, o, rho), pre_work=rho * s,
        Tmax_hat=deadline(o_hat, ttft, tok), Tmax_true=deadline(o, ttft, tok), tau_ttft=ttft, tau_tok=tok,
        e_up=uplink_energy(psd_w_hz, up[:, None], eta), kappa=float(cfg["energy"]["kappa"]),
        lam=float(cfg["energy"]["lambda"]),
    )
