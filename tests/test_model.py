"""Delay, SLA, energy and channel formulas (Eqs. 1-4) against a hand-calculated example."""
import math

import numpy as np
import pytest

from edgegen.energy import compute_energy, uplink_energy
from edgegen.latency import comm_time, comp_time, data_bits, deadline, uplink_bits, work
from edgegen.scenario import spectral_efficiency

# s = 100 prompt tokens, o = 50 output tokens, beta = 32 bit, A = 1 Mbit image,
# b = 1 MHz, eta = 2 bit/s/Hz, rho = 0.125 dte/token, f = 125 dte/s.


def test_eq1_comm_delay():
    D = data_bits(100, 50, 1e6, 32)                    # (100 + 50) * 32 + 1e6
    assert D == pytest.approx(1_004_800)
    assert comm_time(D, 1e6, 2.0) == pytest.approx(0.5024)


def test_eq2_comp_delay_and_ttft():
    C = work(100, 50, 0.125)                           # 0.125 * 100 + 50 = 62.5 dte
    assert C == pytest.approx(62.5)
    assert comp_time(C, 125.0) == pytest.approx(0.5)
    ttft = comm_time(uplink_bits(100, 1e6, 32), 1e6, 2.0) + comp_time(0.125 * 100, 125.0)
    assert ttft == pytest.approx(1_003_200 / 2e6 + 0.1)


def test_eq4_token_level_sla():
    assert deadline(50, 0.5, 0.010) == pytest.approx(1.0)          # Premium: 0.5 s + 50 x 10 ms


def test_eq3_energy():
    psd = 10 ** (-47 / 10) / 1000                                    # -47 dBm/Hz in W/Hz
    assert uplink_energy(psd, 1_003_200, 2.0) == pytest.approx(psd * 501_600)
    assert compute_energy(8.9e-6, 62.5, 125.0) == pytest.approx(8.9e-6 * 62.5 * 125**2)


def test_eq1_spectral_efficiency_at_1km():
    ch = dict(pathloss_a_db=128.1, pathloss_b_db=37.6, shadowing_std_db=0.0, ue_psd_dbm_hz=-47.0,
              noise_psd_dbm_hz=-174.0, noise_figure_db=9.0, interference_margin_db=3.0, max_spectral_eff=None)
    eta = spectral_efficiency(np.array([[1000.0]]), ch, np.random.default_rng(0))
    snr_db = -47 - 128.1 + 174 - 9 - 3                               # = -13.1 dB
    assert eta[0, 0] == pytest.approx(math.log2(1 + 10 ** (snr_db / 10)))
