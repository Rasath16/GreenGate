# GreenGate

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

## Honest limitations

- On open-ended generation with small models, token entropy is a weak signal (near chance in our evaluation). It is informative on structured tasks and vision. Where it is weak, savings come from the cascade structure rather than from selective routing.
- Energy measurement requires an NVIDIA GPU (NVML). CPU runs fall back to a documented estimate.
- API-tier carbon is an estimate, not a measurement, and is not directly comparable to metered local figures.

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
