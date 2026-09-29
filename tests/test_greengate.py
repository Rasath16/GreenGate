"""Unit tests for the GPU-free core logic (run: pytest tests/)."""

import math

import torch

from greengate.entropy import shannon_entropy
from greengate.calibration import fit_temperature, ece, calibrated_entropy
from greengate.vqa import vqa_match, normalize
from greengate.budget import SlidingWindowBudget
from greengate.core import _load_temperature
from eval_mmlu import simulate_policy


def test_entropy_uniform_is_log2_vocab():
    logits = torch.zeros(1024)
    assert abs(shannon_entropy(logits) - 10.0) < 1e-4  # log2(1024)


def test_entropy_peaked_is_near_zero():
    logits = torch.full((1024,), -100.0)
    logits[7] = 100.0
    assert shannon_entropy(logits) < 1e-3


def test_temperature_softens_entropy():
    logits = torch.tensor([5.0, 1.0, 0.5, 0.1])
    assert calibrated_entropy(logits, 4.0) > calibrated_entropy(logits, 1.0)


def test_fit_temperature_detects_overconfidence():
    # ground truth is random w.r.t. hugely confident logits -> best T is large
    torch.manual_seed(0)
    logits = torch.randn(200, 4) * 10       # overconfident
    labels = torch.randint(0, 4, (200,))    # uncorrelated truth
    T = fit_temperature(logits, labels)
    assert T > 2.0
    assert ece(logits, labels, T) < ece(logits, labels, 1.0)


def test_vqa_match_normalisation():
    assert vqa_match("It's a Cat.", "cat")
    assert vqa_match("the red one", "red")
    assert not vqa_match("category", "cat")       # no substring false-positive
    assert normalize("The  Red-Car!") == "redcar" or normalize("The Red Car!") == "red car"


def test_budget_blocks_then_recovers():
    b = SlidingWindowBudget(budget_g=1.0, window_s=100)
    b.record(now=0.0, carbon_g=0.9)
    assert not b.allows(now=10.0, escalation_cost_g=0.5)   # would exceed
    assert b.allows(now=150.0, escalation_cost_g=0.5)      # old spend expired


def test_full_accounting_charges_wasted_small_run():
    records = [{"small_correct": 0, "large_correct": 1,
                "small_carbon": 1.0, "large_carbon": 3.0,
                "small_energy": 0.0, "large_energy": 0.0}]
    cascade = simulate_policy(records, [True], runs_small_first=True)
    direct = simulate_policy(records, [True], runs_small_first=False)
    assert abs(cascade["carbon_g"] - 4.0) < 1e-9   # small wasted + large
    assert abs(direct["carbon_g"] - 3.0) < 1e-9


def test_preset_registry_has_mistral():
    T, source = _load_temperature("mistralai/Mistral-7B-Instruct-v0.2")
    assert source == "preset" and T > 1.0


def test_unknown_model_falls_back_uncalibrated():
    T, source = _load_temperature("no-such/model")
    assert (T, source) == (1.0, "uncalibrated")


# --------------------------------------------------------------------------
# Measurement-only API (T10-T14): profiling inference GreenGate does not run
# --------------------------------------------------------------------------

def test_measure_records_energy_and_per_query_figures():
    import greengate
    with greengate.measure(n_queries=4, label="unit") as m:
        sum(i * i for i in range(50_000))
    assert m.duration_s > 0
    assert m.energy_joules > 0
    assert m.n_queries == 4
    assert abs(m.energy_per_query_j - m.energy_joules / 4) < 1e-9
    assert "unit" in m.report()


def test_measure_reports_gpu_exclusivity():
    """A measurement must say whether it had the device to itself, because
    device-level power cannot be split between concurrent processes."""
    import greengate
    with greengate.measure() as m:
        pass
    assert isinstance(m.exclusive, bool)
    assert m.exclusive == (not m.shared_with)


def test_should_cascade_matches_the_break_even_condition():
    from greengate import should_cascade
    r = should_cascade(carbon_small=0.05, carbon_large=0.20, escalation_rate=0.30)
    assert abs(r["cost_ratio_small_over_large"] - 0.25) < 1e-9
    assert abs(r["break_even_escalation_rate"] - 0.75) < 1e-9
    assert abs(r["predicted_saving"] - 0.45) < 1e-9        # 1 - 0.25 - 0.30
    assert r["headroom"] > 0


def test_should_cascade_rejects_a_small_tier_that_is_not_cheaper():
    """The measured SmolVLM-2B vision pair: C_s/C_l > 1, so no escalation
    rate can save energy."""
    from greengate import should_cascade
    r = should_cascade(carbon_small=2.5830, carbon_large=2.4620, escalation_rate=0.05)
    assert r["cost_ratio_small_over_large"] > 1.0
    assert r["predicted_saving"] < 0
    assert "not cheaper" in r["verdict"]


def test_service_ledger_aggregates_batched_windows():
    """Under batching, per-query attribution is impossible; the ledger works
    from totals over windows of work instead."""
    from greengate import ServiceLedger
    led = ServiceLedger()
    led.add(tier="small", queries=100, carbon_grams=5.0)    # 0.05 g per query
    led.add(tier="large", queries=30, carbon_grams=6.0)     # 0.20 g per query
    rep = led.report()
    assert rep["small_tier_queries"] == 100
    assert abs(rep["escalation_rate"] - 0.30) < 1e-9
    assert abs(rep["cost_ratio_small_over_large"] - 0.25) < 1e-9
    assert abs(rep["predicted_saving"] - 0.45) < 1e-9
    assert abs(rep["total_carbon_grams"] - 11.0) < 1e-9
    assert abs(rep["wasted_carbon_grams"] - 1.5) < 1e-9     # 30 discarded small runs
