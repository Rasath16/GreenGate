# Hardware comparison run: NVIDIA H100 80GB HBM3

Date: 2026-09-10
Repository commit evaluated: `59c30d19bee470748b4c37caad2abf54f3cc7c35`

## Purpose

This is the third and largest accelerator in the cross-hardware series. The
baseline measurements throughout this project were taken on a Tesla T4 (70 W,
2018 Turing); `experiments/hardware_a100_2026-09-10/` repeats them on an
A100-SXM4-80GB (400 W, Ampere); this run completes the series on an H100
(700 W, Hopper), giving a tenfold range of board power across three
generations.

The third point matters more than a third point usually would. With only the
T4 and A100, the small-to-large cost ratio `C_s/C_l` appeared to improve on
newer hardware. The H100 does not reproduce that improvement, which converts
an apparent trend into what the evidence actually supports: bounded variation
with no ordering by generation.

## Hardware and environment

| | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 |
| Board power limit | 700 W |
| Memory | 81559 MiB HBM3 |
| Platform | RunPod (on-demand pod, no network volume) |
| Driver | 570.195.03 |
| Python | 3.10.12 |
| torch | 2.8.0+cu128 |
| transformers | 5.17.0 |

Raw environment capture: `PROVENANCE.txt`. Full console output: `run_all.log`.

### On the differing commit identifiers

This run records commit `59c30d1` while the A100 run records `bb62dc4`. The
difference between those two commits is the addition of the A100 result files
and a `.gitignore` change. `eval_mmlu.py` and the `greengate` package are
byte-identical across them, so the evaluation code is the same in both runs
and the differing hashes do not represent a change in method. Verify with:

```
git diff --stat bb62dc4 59c30d1 -- greengate/ eval_mmlu.py
```

### Environment note

The RunPod PyTorch 2.8.0 template ships torch 2.8.0+cu128 against a CUDA 12.8
driver. Installing `transformers` pulls torch 2.14.0+cu130 as a dependency,
which reports `torch.cuda.is_available() == False` on that driver. This run
therefore pins torch back to 2.8.0+cu128 and removes the stale
torchvision 0.17.0 / torchaudio 2.2.0 that were compiled against torch 2.2.0
(their unresolved `torchvision::nms` operator surfaces confusingly as
`Could not import Qwen2ForCausalLM`). Anyone reproducing this run needs both
corrections. Note also that a RunPod *network* volume imposes a storage quota
that the four-model sweep exhausts; use a local volume disk of 120 GB.

## Protocol

Identical to the T4 runs in `results/zoo_*` and the A100 run: 300 MMLU
questions, seed 42, `mistralai/Mistral-7B-Instruct-v0.2` in 4-bit as the
large tier, each Qwen2.5 checkpoint at its native precision as the small
tier, entropy threshold 1.0 bits, full carbon accounting (escalated queries
are charged for both the discarded small-model run and the large-model run).

```
for M in Qwen2.5-0.5B Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct; do
  python eval_mmlu.py --n 300 \
    --small Qwen/$M \
    --large mistralai/Mistral-7B-Instruct-v0.2 --large-4bit \
    --outdir h100_$M
done
```

## Contents

```
PROVENANCE.txt              hardware, driver, library versions, commit
run_all.log                 full console output of all four runs
h100_<model>/records.csv    per-question energy, carbon, entropy, correctness
h100_<model>/summary.csv    policy comparison (B1-B4 and GreenGate)
h100_<model>/threshold_sweep.csv
```

Reproduce the three-accelerator comparison from the repository root:

```
python experiments/compare_hardware.py
```

## Results

This run's figures, over 300 questions:

| Small model | Small tier (J) | Large tier (J) | C_s/C_l | Mean entropy | Carbon cut | Retention |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 863.8 | 4029.1 | 0.214 | 1.478 | -4.0% | 100.0% |
| Qwen2.5-1.5B | 1061.0 | 3865.1 | 0.275 | 0.808 | 27.0% | 110.6% |
| Qwen2.5-3B | 1470.8 | 3992.6 | 0.368 | 0.488 | 40.3% | 121.9% |
| Qwen2.5-7B | 1801.0 | 4010.6 | 0.449 | 0.355 | 40.2% | 126.2% |

Across all three accelerators:

| | T4 (70 W) | A100 (400 W) | H100 (700 W) |
| --- | --- | --- | --- |
| Mean entropy, 1.5B / 3B / 7B | 0.808 / 0.488 / 0.355 | 0.808 / 0.488 / 0.355 | 0.808 / 0.488 / 0.355 |
| C_s/C_l, 1.5B | 0.272 | 0.233 | 0.275 |
| C_s/C_l, 3B | 0.414 | 0.375 | 0.368 |
| C_s/C_l, 7B | 0.460 | 0.468 | 0.449 |
| Carbon cut, 1.5B | 26.2% | 31.1% | 27.0% |
| Carbon cut, 3B | 35.1% | 39.9% | 40.3% |

## Findings

**The routing signal is exactly hardware-invariant.** Mean choice entropy is
identical to three decimal places on all three accelerators for every model:
spread 0.0000 bits. Small-tier accuracy differs by at most one question in
300, consistent with floating-point non-determinism between Turing, Ampere
and Hopper kernels. A threshold calibrated on one machine therefore transfers
to another unchanged.

**Board power is a poor proxy for energy.** Both datacentre accelerators use
15 to 20 per cent *less* energy than the T4 for identical work while drawing
five to ten times its board power, because they finish sooner. Energy is
power integrated over time.

**The cost ratio varies but is not ordered by generation.** For the 1.5B
cascade `C_s/C_l` takes the values 0.272, 0.233 and 0.275 on T4, A100 and
H100. The A100 favours the small tier more than either of the others, so the
two-machine comparison suggested a trend that the third machine contradicts.
Measured carbon reduction for that cascade stays between 26.2 and 31.1 per
cent across the whole range, at retention varying by under two points.

**The break-even condition is real and marginal cascades are fragile.** The
0.5B configuration escalates 82 per cent of queries, close to break-even, and
the sign of the result changes between machines: +4.2 per cent carbon
reduction on the A100, -4.0 per cent on the H100, at identical routing
decisions and identical accuracy. A deployment operating that near the
boundary cannot infer from published figures whether cascading will help it,
which is the case the library is instrumented to decide empirically.

## Excluded from the cross-hardware comparison

The T4 Qwen2.5-0.5B run in `results/records.csv` is not used. Its escalation
rate at threshold 1.0 is 64.7 per cent, whereas both the A100 and H100 0.5B
runs report 82.3 per cent and agree with each other. It predates the August
2026 profiler correction and is therefore treated as not comparable rather
than reported as a hardware effect.
