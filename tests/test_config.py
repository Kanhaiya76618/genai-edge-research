"""The default configuration equals Table II of the paper and every parameter is documented."""
import re

from edgegen.config import CONFIG_DIR, load_config


def test_defaults_equal_table_ii():
    c = load_config()
    assert (c["layout"]["n_nodes"], c["layout"]["n_users"], c["layout"]["area_m"]) == (4, 40, 1000.0)
    assert (c["layout"]["hotspot_share"], c["layout"]["hotspot_sigma_m"]) == (0.4, 120.0)
    ch = c["channel"]
    assert (ch["pathloss_a_db"], ch["pathloss_b_db"], ch["shadowing_std_db"]) == (128.1, 37.6, 8.0)
    assert (ch["ue_psd_dbm_hz"], ch["noise_psd_dbm_hz"], ch["noise_figure_db"], ch["interference_margin_db"]) \
        == (-47.0, -174.0, 9.0, 3.0)
    assert ch["bandwidth_mhz"] == 20.0
    wl = c["workload"]
    assert (wl["bits_per_token"], wl["attach_prob"], wl["attach_kb_min"], wl["attach_kb_max"]) == (32, 0.4, 50.0, 500.0)
    assert (wl["prompt_median_tokens"], wl["output_median_tokens"], wl["predictor_sigma"]) == (200.0, 150.0, 0.5)
    assert (c["gpu"]["throughput_dte_s"], c["gpu"]["f_max_dte_s"], c["gpu"]["prefill_cost"]) == (1500.0, 200.0, 0.125)
    assert (c["energy"]["kappa"], c["energy"]["lambda"]) == (8.9e-6, 0.0)
    assert (c["algorithm"]["dual_steps"], c["algorithm"]["dual_step_size"]) == (25, 8.0)
    assert [(k["name"], k["share"], k["weight"], k["ttft_s"], k["tok_s"]) for k in c["classes"]] == [
        ("Basic", 0.5, 1.0, 1.0, 0.020), ("Premium", 0.3, 2.0, 0.5, 0.010), ("Critical", 0.2, 5.0, 0.5, 0.010)]
    assert c["experiments"]["table_methods"][-1] == "JCC-Edge"
    assert c["gpu"]["kv_capacity_tokens"] == 16384


def test_quick_inherits_the_same_model():
    d, q = load_config(), load_config(CONFIG_DIR / "quick.yaml")
    for key in ("layout", "channel", "workload", "classes", "gpu", "energy", "algorithm"):
        assert d[key] == q[key]
    assert q["experiments"]["small_instance"]["n_nodes"] == 3


def test_every_parameter_has_a_comment():
    missing = [ln.strip() for ln in (CONFIG_DIR / "default.yaml").read_text(encoding="utf-8").splitlines()
               if re.match(r"^\s*\w+:\s*[-+0-9.eE]+\s*$|^\s*\w+:\s*(true|false|null|noisy)\s*$", ln)]
    assert not missing, missing
