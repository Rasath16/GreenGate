"""Populate results/ from the published experiment evidence.

The analysis scripts in this repository read their inputs from results/,
which is the working directory produced by a live run and is not tracked
in git. The evidence those runs produced is tracked, under experiments/,
one directory per run with its own README. This script copies the
published records into the layout the analysis scripts expect, so that a
fresh clone can recompute every reported figure without re-running any
inference.

The mapping below is also the provenance record: it states which run each
analysis input came from.

Run from the repository root:
    python bootstrap_results.py
    python eval_crossfamily.py        # and any other eval_*.py
"""
import os
import shutil
import sys

W2 = "experiments/week2_1p5B_and_zoo_2026-09-03"
SS = "experiments/scaling_series_2026-09-06"
API = "experiments/exp3_api_cascade_2026-08-19"
HA = "experiments/human_annotation_2026-09-03"

MAP = [
    # human annotation
    ("results/annotation_key.csv", f"{HA}/annotation_key.csv"),
    ("results/annotator1_results.csv", f"{HA}/annotator1_results.csv"),
    ("results/annotator2_results.csv", f"{HA}/annotator2_results.csv"),
    # API cascade (Experiment 3)
    ("results/api_cascade_records.jsonl", f"{API}/api_cascade_records.jsonl"),
    ("results/api_cascade_sweep.csv", f"{API}/api_cascade_sweep.csv"),
    ("results/judgments_api.jsonl", f"{API}/judgments_api.jsonl"),
    # calibration fitted on the held-out MMLU validation split
    ("results/calibration.json", f"{W2}/calibration.json"),
    ("results/calibration_qwen.json", f"{W2}/calibration_qwen.json"),
    ("results/calibration_qwen15.json", f"{SS}/cal_Qwen2.5-1.5B-Instruct.json"),
    ("results/cal_Phi-3.5-mini-instruct.json", f"{W2}/cal_Phi-3.5-mini-instruct.json"),
    ("results/cal_Qwen2.5-1.5B-Instruct.json", f"{SS}/cal_Qwen2.5-1.5B-Instruct.json"),
    ("results/cal_Qwen2.5-3B-Instruct.json", f"{W2}/cal_Qwen2.5-3B-Instruct.json"),
    ("results/cal_Qwen2.5-7B-Instruct.json", f"{SS}/cal_Qwen2.5-7B-Instruct.json"),
    ("results/cal_SmolLM2-1.7B-Instruct.json", f"{W2}/cal_SmolLM2-1.7B-Instruct.json"),
    ("results/cal_TinyLlama-1.1B-Chat-v1.0.json", f"{W2}/cal_TinyLlama-1.1B-Chat-v1.0.json"),
    # main text records and judgements
    ("results/main_records.jsonl", f"{W2}/main_records.jsonl"),
    ("results/main15_records.jsonl", f"{SS}/main15_records.jsonl"),
    ("results/qwen_records.jsonl", f"{W2}/qwen_records.jsonl"),
    ("results/q15/qwen_records.jsonl", f"{W2}/q15/qwen_records.jsonl"),
    ("results/text_records.jsonl", f"{W2}/text_records.jsonl"),
    ("results/text_records_small.jsonl", f"{W2}/text_records_small.jsonl"),
    ("results/judgments.jsonl", f"{W2}/judgments.jsonl"),
    ("results/judgments_main.jsonl", f"{W2}/judgments_main.jsonl"),
    ("results/judgments_main15.jsonl", f"{SS}/judgments_main15.jsonl"),
    ("results/judgments_local_main.jsonl", f"{W2}/judgments_local_main.jsonl"),
    # repeatability replicates
    ("results/rep2/calibration.json", f"{W2}/calibration.json"),
    ("results/rep2/text_records.jsonl", f"{W2}/rep2/text_records.jsonl"),
    ("results/rep2/text_records_small.jsonl", f"{W2}/rep2/text_records_small.jsonl"),
    ("results/rep3/calibration.json", f"{W2}/calibration.json"),
    ("results/rep3/text_records.jsonl", f"{W2}/rep3/text_records.jsonl"),
    ("results/rep3/text_records_small.jsonl", f"{W2}/rep3/text_records_small.jsonl"),
    # MMLU, semantic entropy, vision
    ("results/records.csv", f"{W2}/records.csv"),
    ("results/summary.csv", f"{W2}/summary.csv"),
    ("results/threshold_sweep.csv", f"{W2}/threshold_sweep.csv"),
    ("results/semantic_ablation.csv", f"{W2}/semantic_ablation.csv"),
    ("results/semantic_ablation_15B.csv", f"{SS}/semantic_ablation.csv"),
    ("results/vision_records.jsonl", f"{W2}/vision_records.jsonl"),
    ("results/vision_summary.csv", f"{W2}/vision_summary.csv"),
    ("results/vision_sweep.csv", f"{W2}/vision_sweep.csv"),
]

# model zoo: each small tier evaluated against the same large tier
ZOO = [
    ("zoo_Phi-3.5-mini-instruct", W2),
    ("zoo_Qwen2.5-1.5B-Instruct", SS),
    ("zoo_Qwen2.5-3B-Instruct", W2),
    ("zoo_Qwen2.5-7B-Instruct", SS),
    ("zoo_SmolLM2-1.7B-Instruct", W2),
    ("zoo_TinyLlama-1.1B-Chat-v1.0", W2),
]
for name, src in ZOO:
    for leaf in ("records.csv", "summary.csv", "threshold_sweep.csv"):
        MAP.append((f"results/{name}/{leaf}", f"{src}/{name}/{leaf}"))


def main():
    if not os.path.isdir("experiments"):
        sys.exit("Run this from the repository root (no experiments/ directory here).")
    copied = missing = kept = 0
    for dst, src in MAP:
        if not os.path.exists(src):
            print(f"  missing evidence: {src}")
            missing += 1
            continue
        if os.path.exists(dst):
            kept += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"\n  copied {copied}, already present {kept}, missing {missing}")
    if missing:
        print("  Some evidence is absent; the affected analyses cannot be rerun.")
    else:
        print("  results/ is populated. Every eval_*.py can now be run.")


if __name__ == "__main__":
    main()
