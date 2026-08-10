# GreenGate — Full From-Scratch Evaluation Runbook

One clean Kaggle session (GPU T4 x2, Internet ON, ~7-8h GPU, ~$4 API total).
Secrets needed: `OPENAI_API_KEY`. All steps resume safely if interrupted.

## 0. Setup

```
!git clone https://github.com/Rasath16/GreenGate.git
%cd GreenGate
!pip install -q -U transformers bitsandbytes ecologits openai
```
```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
import pynvml; pynvml.nvmlInit()
print("GPUs:", pynvml.nvmlDeviceGetCount())   # expect 2
```

## 1. Calibration (both models, ~25 min)

```
!python calibrate_mmlu.py --n 150 --small mistralai/Mistral-7B-Instruct-v0.2 --small-4bit
!python calibrate_mmlu.py --n 150 --small Qwen/Qwen2.5-0.5B-Instruct --outfile results/calibration_qwen.json
```
Thesis Table: fitted T + ECE before/after per model.

## 2. Text passes (~3.5-4h)

```
!python eval_text.py --n 500 --small mistralai/Mistral-7B-Instruct-v0.2 --small-4bit --max-new-tokens 150
!python eval_small2.py --n 500 --small Qwen/Qwen2.5-0.5B-Instruct
!python make_main_records.py
```

## 3. MMLU (main pair, ~15 min)

```
!python eval_mmlu.py --n 300 --small Qwen/Qwen2.5-0.5B-Instruct --large mistralai/Mistral-7B-Instruct-v0.2 --large-4bit
```

## 4. Judging (cost-check FIRST, then ~$2.30 total)

```
!python judge_openai.py --records results/main_records.jsonl --out results/judgments_main.jsonl --limit 20
# CHECK platform.openai.com/usage (~$0.05 expected), then:
!python judge_openai.py --records results/main_records.jsonl --out results/judgments_main.jsonl
# secondary Mistral-vs-API analysis: reuse Mistral scores, judge only API answers
!python seed_judgments.py --src results/judgments_main.jsonl --dst results/judgments.jsonl --map large=small
!python judge_openai.py --records results/text_records.jsonl --out results/judgments.jsonl --tiers large
# local second judge (bias check, free)
!python judge_local.py --records results/main_records.jsonl --out results/judgments_local_main.jsonl --limit 200 --load-4bit
```

## 5. Vision (~1.5h GPU + ~$1.20 API)

```
!python eval_vision.py --n 500 --small HuggingFaceTB/SmolVLM-Instruct
```
Objective VQA accuracy — no judging cost.

## 6. Semantic entropy ablation (~1h)

```
!python eval_semantic.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --n 200 --k 3 --small Qwen/Qwen2.5-0.5B-Instruct
!python join_semantic.py
```

## 7. Measurement repetitions (variance table, ~30 min, $0)

```
!mkdir -p results/rep2 results/rep3
!cp results/calibration.json results/rep2/ ; !cp results/calibration.json results/rep3/
!python eval_text.py --n 50 --small mistralai/Mistral-7B-Instruct-v0.2 --small-4bit --max-new-tokens 150 --dry-run-api --outdir results/rep2
!python eval_text.py --n 50 --small mistralai/Mistral-7B-Instruct-v0.2 --small-4bit --max-new-tokens 150 --dry-run-api --outdir results/rep3
```
Compare per-query `small_energy_j` across runs -> mean +/- std.

## 8. Results tables

```
!python summarize_text.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --signal small_entropy_raw
!python summarize_text.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --signal small_entropy_max
!python summarize_text.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --signal semantic_entropy
!python summarize_text.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --signal semantic_entropy --floor 0.95
# secondary: Mistral vs API (dual-mode profiler demo)
!python summarize_text.py --records results/text_records.jsonl --judgments results/judgments.jsonl --signal small_entropy_raw
```

## 9. Trace replay (use t* from the best signal above)

```
!python eval_trace_replay.py --records results/main_records.jsonl --judgments results/judgments_main.jsonl --signal semantic_entropy --threshold <T_STAR> --budget-g 0.05 --window-s 600
```

## 10. Package + download, STOP SESSION

```
!zip -r results_final.zip results
```
Download `results_final.zip` from the file panel, then stop the session.

## 11. On the laptop afterwards

```
python export_annotation.py --records results/main_records.jsonl --n 100
# two annotators fill copies of annotation_sheet.csv, then:
python kappa.py --a results/annotator1.csv --b results/annotator2.csv
```
