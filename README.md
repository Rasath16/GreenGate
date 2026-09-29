# GreenGate

[![PyPI version](https://img.shields.io/pypi/v/greengate.svg)](https://pypi.org/project/greengate/)
[![Python](https://img.shields.io/pypi/pyversions/greengate.svg)](https://pypi.org/project/greengate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Cut the cost and carbon of LLM inference with one line of routing.** Easy queries are answered by a small local model; only genuinely hard ones escalate to a large model. Every query is carbon-accounted, honestly.

```bash
pip install greengate
```

```python
import greengate

gw = greengate.GreenGate(
    small="Qwen/Qwen2.5-0.5B-Instruct",   # runs locally, exposes logits
    large="gpt-4o-mini",                   # or any local model
    budget_g=0.5,                          # optional carbon ceiling
)

r = gw.route("Summarise this document ...")
print(r.response)     # answered by whichever tier was appropriate
print(r.decision)     # "LOCAL" or "ESCALATE"
print(r.carbon_g)     # gCO2 for this query, including any wasted small run

gw.profile()          # session totals: carbon, escalation rate, latency
```

## Why this exists

Today every query, easy or hard, is sent to the same large model. Most queries do not need it. GreenGate sits between your application and your models, measures how uncertain the small model is, and escalates only when that uncertainty is high.

Three things it does that other cascading systems do not:

1. **Measures real energy.** Local inference is metered with NVML at 100 ms across all GPUs, not estimated. API tiers use [EcoLogits](https://ecologits.ai/) and are labelled as estimates.
2. **Full carbon accounting.** When a query escalates, it is charged for *both* the discarded small-model run and the large-model run. Most published cascade savings omit the first, which overstates them.
3. **Routes vision queries too**, using the average token probability of the generated answer as the confidence signal.

## What to expect

From the evaluation in the accompanying study (ShareGPT, MMLU and VQAv2; measured on dual T4):

| Workload | Carbon reduction | Quality retention |
|---|---|---|
| MMLU (structured) | 29.5% | 95.0% |
| ShareGPT (open-ended) | 49.8% | 80.6% |
| ShareGPT (quality-first) | 14.5% | 90.1% |
| VQAv2 (vision) | escalates only 5% | exceeds both tiers |

Savings depend on four measurable things: the energy ratio between your tiers, your escalation rate, the retention you accept, and your hardware. GreenGate reports all of them rather than assuming them. On an inefficient local GPU, escalating to an efficient API can genuinely be greener — the profiler tells you which case you are in.

## Tiers

- **Small tier** must be a local open-weight model. The routing signal is computed from token logits, which hosted APIs do not expose. This is also where the privacy and cost win comes from.
- **Large tier** can be anything: another local model, or an OpenAI API model.

## Calibration

Language models are overconfident, so raw entropy thresholds are unreliable. GreenGate ships fitted temperature-scaling values for evaluated models and self-calibrates for anything else:

```python
gw.calibrate()   # ~15 min, fits T on held-out MMLU validation, saved to ~/.greengate/
```

## Modes and budgets

```python
gw.config(mode="green")      # escalate less; "balanced" and "quality" also available
gw.config(threshold=2.9)     # or set the entropy threshold explicitly
gw.config(budget_g=0.05)     # sliding-window carbon ceiling; escalation defers when exhausted
```

## Measuring inference you already run

The routing API above loads both tiers itself, which suits a controlled
comparison but not a service that already serves models through vLLM, TGI,
Ollama or a provider SDK. To profile an existing pipeline without
restructuring it, use the measurement API on its own:

```python
import greengate

with greengate.measure(n_queries=1, label="checkout-summariser") as m:
    answer = my_existing_pipeline(prompt)

print(m.report())
# checkout-summariser: 1 query in 2.914 s
#   431.72 J total, 431.72 J per query
#   0.068 g CO2 total, 0.068 g per query
#   148.2 W average, measured by gpu (pynvml)
```

Power is sampled per device, so anything else on the same GPU is counted in.
A measurement that shared the device says so, and `strict=True` raises instead
of warning:

```python
with greengate.measure(strict=True) as m:   # ContendedMeasurement if not alone
    ...
```

### Deciding whether a cascade is worth it

A cascade runs the small tier on every query and the large tier on the
escalated fraction, so relative to always using the large tier the saving is
`S = 1 - C_s/C_l - e`. Feed it measured costs:

```python
greengate.should_cascade(carbon_small=0.05, carbon_large=0.20,
                         escalation_rate=0.30)
# {'cost_ratio_small_over_large': 0.25,
#  'break_even_escalation_rate': 0.75,
#  'headroom': 0.45,
#  'predicted_saving': 0.45,
#  'verdict': 'saves 45.0% against always using the large tier; escalation
#              may rise to 75% before that is lost'}
```

If the small tier is not actually cheaper, it says so rather than reporting a
saving. That case is real: in this project's vision experiments a 2B model cost
more per query than a 4-bit 7B model, because image processing dominates and
parameter count does not.

### Services that batch

Under continuous batching many requests share the GPU at once and no
device-level meter can divide that energy between them. Rather than invent an
attribution, account over windows of work:

```python
ledger = greengate.ServiceLedger()

with greengate.measure(n_queries=128) as m:
    serve_batch_on_small_tier(...)
ledger.add(m, tier="small")

with greengate.measure(n_queries=37) as m:
    serve_batch_on_large_tier(...)
ledger.add(m, tier="large")

print(ledger.report())
# realised escalation rate, mean per-query costs, wasted carbon from
# discarded small-tier runs, and the break-even verdict
```

## Honest limitations

- On open-ended generation with small models, token entropy is a weak signal (near chance in our evaluation). It is informative on structured tasks and vision. Where it is weak, savings come from the cascade structure rather than from selective routing.
- Energy measurement requires an NVIDIA GPU (NVML). CPU runs fall back to a documented estimate.
- API-tier carbon is an estimate, not a measurement, and is not directly comparable to metered local figures.

- Device-level power cannot be attributed to individual requests when several run concurrently; measure exclusively, or account over windows with `ServiceLedger`.

## Install extras

```bash
pip install greengate[gpu]    # bitsandbytes + pynvml for quantised local models and metering
pip install greengate[api]    # openai + ecologits for API large tiers
pip install greengate[eval]   # pandas/matplotlib for the evaluation scripts
```

## Reproducing the evaluation

The `RUNBOOK.md` in this repository reproduces every published number: calibration, three deployment configurations, four baselines, threshold sweeps, grid conditions, trace replay against real Azure arrival traces, and three ablation studies. Raw per-query records for all runs are in `experiments/`.

## Citation

If you use GreenGate in academic work, please cite the accompanying study:

> T. R. Hemachandra, "GreenGate: A Confidence-Aware Cascading Framework for Optimizing Energy and Cost in Large Language and Multimodal Model Inference," BSc thesis, NSBM Green University, 2026.

## License

MIT
