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
compare_hardware.py         regenerates the comparison below
hardware_comparison.csv     output of compare_hardware.py
```

Reproduce the comparison from the repository root:

```
python experiments/hardware_a100_2026-09-10/compare_hardware.py
```

## Results

Energy over 300 questions, in joules:

| Small model | T4 small | A100 small | T4 large | A100 large | T4 ratio | A100 ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 1317.6 | 930.0 | 4848.0 | 3998.2 | 3.68 | 4.30 |
| Qwen2.5-3B | 1866.4 | 1507.0 | 4508.8 | 4017.5 | 2.42 | 2.67 |
| Qwen2.5-7B | 2270.3 | 1920.9 | 4933.0 | 4103.4 | 2.17 | 2.14 |

Quantities that should not depend on the GPU, and do not:

| Small model | T4 accuracy | A100 accuracy | T4 mean entropy | A100 mean entropy |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 0.570 | 0.567 | 0.808 | 0.808 |
| Qwen2.5-3B | 0.653 | 0.650 | 0.488 | 0.488 |
| Qwen2.5-7B | 0.687 | 0.687 | 0.355 | 0.355 |

Mean entropy is identical to three decimal places on all three model pairs.
Small-model accuracy differs by at most one question in 300, consistent with
floating-point non-determinism between Turing and Ampere kernels.

GreenGate at threshold 1.0:

| Small model | T4 carbon cut | A100 carbon cut | T4 retention | A100 retention |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 26.2% | 31.1% | 110.6% | 111.3% |
| Qwen2.5-3B | 35.1% | 39.9% | 121.2% | 122.6% |
| Qwen2.5-7B | 37.8% | 37.9% | 126.2% | 127.0% |

## Interpretation

The A100 draws 5.7 times the board power of the T4 yet consumes 15 to 30 per
cent *less* energy for the same work, because it finishes far sooner. Energy
is power integrated over time, and on this workload the speed advantage more
than offsets the higher draw.

The quantity that governs the break-even condition
`S = 1 - (C_s / C_l) - e` is the small-to-large cost ratio `C_s / C_l`. It
moves modestly and, for the two smaller cascades, favourably: 0.272 to 0.233
for the 1.5B pair and 0.414 to 0.375 for the 3B pair, while the 7B pair is
essentially unchanged at 0.460 to 0.468. Measured carbon reduction therefore
holds or improves on the newer hardware rather than eroding.

The practical claim this supports is narrow but real: the *routing decision*
transfers across hardware unchanged, because the confidence signal it depends
on is a property of the model and the question, not of the accelerator. The
*magnitude* of the saving does depend on the accelerator, which is why the
absolute carbon figures in this thesis are reported alongside the hardware
that produced them.

## Excluded from the comparison

The T4 Qwen2.5-0.5B run in `results/records.csv` is **not** used above. Its
escalation rate at threshold 1.0 is 64.7 per cent, whereas the A100 0.5B run
reports 82.3 per cent and a separate partial H100 run reports 82 per cent.
The two later runs agree with each other and disagree with the T4 one, which
predates the August 2026 profiler and entropy corrections. It is therefore
treated as not comparable rather than reported as a hardware effect.

## Partial H100 run

An earlier attempt on an H100 SXM (80 GB HBM3, 700 W) on the same date
completed two of the four model pairs before hitting a network-volume disk
quota. Its Qwen2.5-1.5B figures were 1094.6 J small, 3749.2 J large, ratio
3.43, small accuracy 0.567, large accuracy 0.533. These are consistent with
the A100 results and are noted here for completeness; the thesis comparison
uses the complete A100 series.
