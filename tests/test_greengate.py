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
