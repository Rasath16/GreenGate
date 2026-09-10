# Hardware comparison run: NVIDIA A100-SXM4-80GB

Date: 2026-09-10
Repository commit evaluated: `bb62dc498caf9dd43dbf40d1c0bed9a7ef897ce9`

## Purpose

Every energy figure elsewhere in this project was measured on a Tesla T4, a
2018 Turing card with a 70 W board limit. A reasonable objection is that
carbon savings measured on old, slow hardware may not survive on the
datacentre GPUs that actually serve production inference. This run repeats
the MMLU cascade evaluation unchanged on an A100-SXM4-80GB (400 W, Ampere)
so that the two machines can be compared directly.

## Hardware and environment

| | Baseline | This run |
| --- | --- | --- |
| GPU | NVIDIA Tesla T4 | NVIDIA A100-SXM4-80GB |
| Board power limit | 70 W | 400 W |
| Memory | 16 GB GDDR6 | 80 GB HBM2e |
| Platform | Kaggle | RunPod (on-demand pod) |
| Driver | - | 570.133.20 |
| Python | - | 3.10.12 |
| torch | - | 2.8.0+cu128 |
| transformers | - | 5.17.0 |

Raw environment capture: `PROVENANCE.txt`. Full console output: `run_all.log`.

### Environment note

The RunPod PyTorch 2.8.0 template ships torch 2.8.0+cu128 against a CUDA 12.8
driver. Installing `transformers` pulls torch 2.14.0+cu130 as a dependency,
which reports `torch.cuda.is_available() == False` on that driver. The run
therefore pins torch back to 2.8.0+cu128 and removes the stale
torchvision 0.17.0 / torchaudio 2.2.0 that were compiled against torch 2.2.0
(their unresolved `torchvision::nms` operator surfaces confusingly as
`Could not import Qwen2ForCausalLM`). Anyone reproducing this run needs the
same two corrections.

## Protocol

Identical to the T4 runs in `results/zoo_*`: 300 MMLU questions, seed 42,
`mistralai/Mistral-7B-Instruct-v0.2` in 4-bit as the large tier, each
Qwen2.5 checkpoint at its native precision as the small tier, entropy
threshold 1.0 bits, full carbon accounting (escalated queries are charged for
both the discarded small-model run and the large-model run).

```
for M in Qwen2.5-0.5B Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct; do
  python eval_mmlu.py --n 300 \
    --small Qwen/$M \
    --large mistralai/Mistral-7B-Instruct-v0.2 --large-4bit \
    --outdir a100_$M
done
```

## Contents

```
PROVENANCE.txt              hardware, driver, library versions, commit
run_all.log                 full console output of all four runs
a100_<model>/records.csv    per-question energy, carbon, entropy, correctness
a100_<model>/summary.csv    policy comparison (B1-B4 and GreenGate)
a100_<model>/threshold_sweep.csv
```

This run is one of three. The comparison across all of them lives at
`experiments/compare_hardware.py`; reproduce it from the repository root:

```
python experiments/compare_hardware.py
```

## Results

See `experiments/compare_hardware.py` for the full three-accelerator table
(T4, A100, H100). This run's own figures, over 300 questions:

| Small model | Small tier (J) | Large tier (J) | C_s/C_l | Mean entropy | Carbon cut | Retention |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 542.4 | 3983.6 | 0.136 | 1.478 | 4.2% | 100.0% |
| Qwen2.5-1.5B | 930.0 | 3998.2 | 0.233 | 0.808 | 31.1% | 111.3% |
| Qwen2.5-3B | 1507.0 | 4017.5 | 0.375 | 0.488 | 39.9% | 122.6% |
| Qwen2.5-7B | 1920.9 | 4103.4 | 0.468 | 0.355 | 37.9% | 127.0% |

## Interpretation

Mean choice entropy matches the T4 and H100 runs to three decimal places on
every model, and small-tier accuracy differs by at most one question in 300.
The routing signal is a property of the model and the question, not of the
accelerator.

The A100 draws 5.7 times the board power of the T4 yet consumes 15 to 20 per
cent less energy for the same work, because it finishes sooner. Energy is
power integrated over time, so thermal design power is a poor proxy for
environmental cost.

On the cost ratio C_s/C_l that governs the break-even condition, this run
alone appears to show an improvement over the T4 (0.272 to 0.233 for the 1.5B
pair). **That reading does not survive the H100 run**, which returns 0.275
for the same pair. Across all three accelerators the ratio varies within a
modest band that is not ordered by hardware generation. The conclusion the
full series supports is bounded variation, not directional improvement; see
`experiments/hardware_h100_2026-09-10/README.md` and thesis Section 6.5.7.

## Excluded from the cross-hardware comparison

The T4 Qwen2.5-0.5B run in `results/records.csv` is not used. Its escalation
rate at threshold 1.0 is 64.7 per cent, whereas both the A100 and H100 0.5B
runs report 82.3 per cent and agree with each other. It predates the August
2026 profiler correction and is therefore treated as not comparable rather
than reported as a hardware effect.
