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


# --------------------------------------------------------------------- #
# profile() reporting the break-even verdict, and audit() deciding
# whether to cascade before anything is deployed. Both are exercised
# without loading a model.
# --------------------------------------------------------------------- #

from dataclasses import dataclass as _dataclass

from greengate.core import GreenGate, _Session, MIN_LARGE_CALLS_FOR_VERDICT


def _bare_gate(session: _Session) -> GreenGate:
    """A GreenGate with a session but no models, for testing reporting."""
    gw = object.__new__(GreenGate)
    gw._session = session
    gw._fixed_threshold = 0.5
    gw._entropy_history = []
    gw.mode = "balanced"
    gw.small_name, gw.large_name = "small", "large"
    gw.signal = "entropy"
    return gw


def test_profile_reports_verdict_once_the_large_tier_is_measured():
    s = _Session(queries=100, escalated=30, carbon_g=10.0,
                 small_carbon_g=5.0, large_carbon_g=30.0, large_calls=30)
    p = _bare_gate(s).profile()
    # mean small 0.05, mean large 1.0 -> ratio 0.05, break-even 95%
    assert p["mean_small_carbon_g"] == 0.05
    assert p["mean_large_carbon_g"] == 1.0
    assert p["cost_ratio_small_over_large"] == 0.05
    assert p["break_even_escalation_rate"] == 0.95
    assert "saves" in p["verdict"]


def test_profile_says_what_is_missing_before_it_can_decide():
    s = _Session(queries=4, escalated=0, small_carbon_g=0.2)
    p = _bare_gate(s).profile()
    assert p["verdict"].startswith("no verdict yet")
    assert "more queries" in p["verdict"]
    assert f"{MIN_LARGE_CALLS_FOR_VERDICT} more escalations" in p["verdict"]
    assert "cost_ratio_small_over_large" not in p


def test_profile_verdict_turns_negative_past_break_even():
    # small costs half of large, so break-even is 50%; escalating 80% must
    # be reported as a loss rather than a saving
    s = _Session(queries=10, escalated=8, small_carbon_g=5.0,
                 large_carbon_g=8.0, large_calls=8)
    p = _bare_gate(s).profile()
    assert p["cost_ratio_small_over_large"] == 0.5
    assert p["break_even_escalation_rate"] == 0.5
    assert p["predicted_saving"] < 0
    assert "costs" in p["verdict"]


@_dataclass
class _FakeGen:
    carbon_grams: float
    entropy_calibrated: float
    response: str = ""


class _FakeTier:
    """Cost is fixed; entropy rises with query length, so ordering is known."""

    def __init__(self, carbon):
        self.carbon = carbon

    def generate(self, q):
        return _FakeGen(carbon_grams=self.carbon, entropy_calibrated=float(len(q)))


def _audit_gate(small_cost, large_cost):
    gw = object.__new__(GreenGate)
    gw._small = _FakeTier(small_cost)
    gw._large = _FakeTier(large_cost)
    gw._large_is_api = False
    return gw


def test_audit_measures_the_cost_ratio_and_break_even():
    qs = ["a" * i for i in range(1, 21)]
    r = _audit_gate(0.25, 1.0).audit(qs, verbose=False)
    assert r["queries_audited"] == 20
    assert r["cost_ratio_small_over_large"] == 0.25
    assert r["break_even_escalation_rate"] == 0.75


def test_audit_thresholds_realise_the_requested_escalation_rates():
    qs = ["a" * i for i in range(1, 21)]
    r = _audit_gate(0.25, 1.0).audit(qs, rates=(0.0, 0.25, 0.5), verbose=False)
    by_target = {row["target_rate"]: row for row in r["rows"]}
    assert by_target[0.0]["realised_rate"] == 0.0
    assert abs(by_target[0.25]["realised_rate"] - 0.25) <= 0.05
    assert abs(by_target[0.5]["realised_rate"] - 0.5) <= 0.05
    # saving falls as escalation rises
    savings = [row["predicted_saving"] for row in r["rows"]]
    assert savings == sorted(savings, reverse=True)


def test_audit_refuses_a_small_tier_that_is_not_cheaper():
    qs = ["a" * i for i in range(1, 11)]
    r = _audit_gate(1.2, 1.0).audit(qs, verbose=False)
    assert r["cost_ratio_small_over_large"] > 1.0
    assert "do not cascade" in r["recommendation"]


def test_audit_rejects_an_empty_sample():
    import pytest
    with pytest.raises(ValueError):
        _audit_gate(0.5, 1.0).audit([], verbose=False)


# --------------------------------------------------------------------- #
# A persisted gate: an auto-tuned threshold needs traffic, and a process
# that restarts without it never leaves warm-up.
# --------------------------------------------------------------------- #

from pathlib import Path as _Path

from greengate.core import MAX_HISTORY, SAVE_EVERY


def _gate_with_state(tmp_path, history=None):
    gw = object.__new__(GreenGate)
    gw._session = _Session()
    gw._fixed_threshold = None
    gw._entropy_history = list(history or [])
    gw.mode = "balanced"
    gw._rate = 0.40                 # balanced, as a fraction of traffic
    gw.small_name, gw.large_name = "small/m", "large/m"
    gw.signal = "entropy"
    gw._state_path = _Path(tmp_path) / "gate.json"
    gw._since_save = 0
    return gw


def test_saved_history_makes_a_restarted_gate_warm(tmp_path):
    seen = [float(i) for i in range(40)]
    _gate_with_state(tmp_path, seen).save_state()

    restarted = _gate_with_state(tmp_path)
    assert restarted._threshold() is None          # nothing loaded yet
    restarted._load_state()
    assert len(restarted._entropy_history) == 40
    # balanced is the 60th percentile of 0..39
    assert restarted._threshold() == seen[int(0.60 * 39)]


def test_a_cold_gate_without_state_still_refuses_to_guess(tmp_path):
    gw = _gate_with_state(tmp_path, history=[1.0, 2.0, 3.0])
    assert gw._threshold() is None                 # under WARMUP_QUERIES


def test_state_from_a_different_pairing_is_ignored(tmp_path):
    _gate_with_state(tmp_path, [float(i) for i in range(40)]).save_state()

    other = _gate_with_state(tmp_path)
    other.small_name = "a/different-small"
    other._load_state()
    assert other._entropy_history == []            # not this pairing's traffic


def test_unreadable_state_costs_a_warmup_not_a_crash(tmp_path):
    bad = _Path(tmp_path) / "gate.json"
    bad.write_text("{ this is not json")
    gw = _gate_with_state(tmp_path)
    gw._load_state()                               # must not raise
    assert gw._entropy_history == []


def test_history_is_capped_so_the_file_cannot_grow_without_bound(tmp_path):
    gw = _gate_with_state(tmp_path, [float(i) for i in range(MAX_HISTORY + 500)])
    gw.save_state()
    restarted = _gate_with_state(tmp_path)
    restarted._load_state()
    assert len(restarted._entropy_history) == MAX_HISTORY
    # the most recent observations are the ones kept
    assert restarted._entropy_history[-1] == float(MAX_HISTORY + 499)


def test_save_state_is_a_no_op_when_persistence_is_off(tmp_path):
    gw = _gate_with_state(tmp_path, [1.0])
    gw._state_path = None
    assert gw.save_state() is None


# --------------------------------------------------------------------- #
# Setting the escalation rate directly. It is the term the break-even
# condition is written in, so it is the thing a developer has decided.
# --------------------------------------------------------------------- #

import pytest as _pytest

from greengate.core import AUTO_THRESHOLD_PERCENTILE


def _rate_gate(rate=None, mode="balanced", history=None):
    gw = object.__new__(GreenGate)
    gw._session = _Session()
    gw._fixed_threshold = None
    gw._entropy_history = list(history if history is not None
                               else [float(i) for i in range(100)])
    gw.mode = mode
    gw._rate = rate if rate is not None else 1.0 - AUTO_THRESHOLD_PERCENTILE[mode] / 100.0
    gw.small_name, gw.large_name = "small/m", "large/m"
    gw.signal = "entropy"
    gw._state_path = None
    gw._since_save = 0
    return gw


def _escalated_fraction(gw):
    thr = gw._threshold()
    return sum(1 for s in gw._entropy_history if s > thr) / len(gw._entropy_history)


def test_requested_escalation_rate_is_what_actually_escalates():
    for rate in (0.10, 0.25, 0.40, 0.75):
        gw = _rate_gate(rate=rate)
        assert abs(_escalated_fraction(gw) - rate) <= 0.02


def test_named_modes_are_positions_on_the_same_dial():
    for mode, pct in AUTO_THRESHOLD_PERCENTILE.items():
        gw = _rate_gate(mode=mode)
        assert abs(_escalated_fraction(gw) - (1 - pct / 100)) <= 0.02


def test_rate_of_zero_escalates_nothing_and_one_escalates_everything():
    assert _escalated_fraction(_rate_gate(rate=0.0)) == 0.0
    assert _escalated_fraction(_rate_gate(rate=1.0)) == 1.0


def test_config_accepts_a_rate_and_clears_a_pinned_threshold():
    gw = _rate_gate(rate=0.40)
    gw.config(threshold=12.0)
    assert gw._threshold() == 12.0            # pinned value wins
    gw.config(escalation_rate=0.20)
    assert gw._fixed_threshold is None        # asking for a rate un-pins it
    assert abs(_escalated_fraction(gw) - 0.20) <= 0.02


def test_config_mode_also_un_pins_and_moves_the_dial():
    gw = _rate_gate(rate=0.40)
    gw.config(threshold=12.0)
    gw.config(mode="green")
    assert gw._fixed_threshold is None
    assert abs(_escalated_fraction(gw) - 0.20) <= 0.02


def test_a_rate_outside_zero_to_one_is_refused():
    for bad in (-0.1, 1.5):
        with _pytest.raises(ValueError):
            _rate_gate().config(escalation_rate=bad)


def test_the_rate_still_waits_for_warmup():
    gw = _rate_gate(rate=0.3, history=[1.0, 2.0, 3.0])
    assert gw._threshold() is None
