"""Delay model of the paper (Sec. III-B and III-C).

    Eq. (1)  T^comm_{i,n} = D_i / (b_{i,n} eta_{i,n}),   D_i = (s_i + o_i) beta + A_i
    Eq. (2)  T^comp_{i,n} = C_i / f_{i,n},               C_i = rho s_i + o_i   [dte]
    TTFT_i = (s_i beta + A_i) / (b_{i,n} eta_{i,n}) + rho s_i / f_{i,n}
    Eq. (4)  T_i^max = tau_ttft + o_i tau_tok
"""
from __future__ import annotations

import numpy as np


def data_bits(prompt_tokens, out_tokens, attach_bits, bits_per_token):
    """D_i = (s_i + o_i) beta + A_i [bit] (Eq. 1)."""
    return (np.asarray(prompt_tokens, float) + np.asarray(out_tokens, float)) * bits_per_token + attach_bits


def uplink_bits(prompt_tokens, attach_bits, bits_per_token):
    """Bits sent before the first token can be produced: s_i beta + A_i."""
    return np.asarray(prompt_tokens, float) * bits_per_token + attach_bits


def work(prompt_tokens, out_tokens, rho):
    """C_i = L_pre(s_i) + L_dec o_i = rho s_i + o_i [dte] (Eq. 2)."""
    return rho * np.asarray(prompt_tokens, float) + np.asarray(out_tokens, float)


def comm_time(bits, b_hz, eta):
    """T^comm = D / (b eta) [s] (Eq. 1)."""
    return np.asarray(bits, float) / (np.asarray(b_hz, float) * np.asarray(eta, float))


def comp_time(work_dte, f_dte_s):
    """T^comp = C / f [s] (Eq. 2)."""
    return np.asarray(work_dte, float) / np.asarray(f_dte_s, float)


def deadline(out_tokens, ttft_s, tok_s):
    """Token-level SLA T^max = tau_ttft + o tau_tok [s] (Eq. 4)."""
    return np.asarray(ttft_s, float) + np.asarray(out_tokens, float) * np.asarray(tok_s, float)
