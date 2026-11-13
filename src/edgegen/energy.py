"""Energy model of the paper (Eq. 3):

    E_{i,n} = p_bar (s_i beta + A_i) / eta_{i,n}  +  kappa C_i f_{i,n}^2

The uplink energy does not depend on b because the UE transmits at a fixed power spectral
density p_bar [W/Hz]; the second term is the DVFS compute-energy model of MEC [9].
"""
from __future__ import annotations

import numpy as np


def uplink_energy(psd_w_hz, up_bits, eta):
    """p_bar (s beta + A) / eta [J] (Eq. 3, first term)."""
    return psd_w_hz * np.asarray(up_bits, float) / np.asarray(eta, float)


def compute_energy(kappa, work_dte, f_dte_s):
    """kappa C f^2 [J] (Eq. 3, second term)."""
    return kappa * np.asarray(work_dte, float) * np.asarray(f_dte_s, float) ** 2
