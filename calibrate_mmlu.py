"""
Temperature scaling calibration on the MMLU validation split.

Fits T for the small model, reports ECE before/after (thesis Table:
calibration proof), and saves T to results/calibration.json for use
by the evaluation scripts.

Usage:
    python calibrate_mmlu.py --n 150 --small Qwen/Qwen2.5-0.5B-Instruct
Smoke test (CPU):
    python calibrate_mmlu.py --n 25 --small gpt2
"""

import argparse
import json
from pathlib import Path

import torch

from greengate.mmlu import load_mmlu
from greengate.evaluator import ChoiceEvaluator
from greengate.calibration import fit_temperature, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="validation questions (held out from test)")
    ap.add_argument("--small", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--small-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Loading {args.n} MMLU VALIDATION questions (held out from evaluation)...")
    questions = load_mmlu(n_questions=args.n, seed=args.seed, split="validation")

    print(f"Collecting choice logits from {args.small}...")
    evaluator = ChoiceEvaluator(args.small, load_in_4bit=args.small_4bit)
    logits_rows, labels = [], []
    for i, q in enumerate(questions):
        r = evaluator.evaluate(q)
        logits_rows.append(r.choice_logits)
        labels.append(q.answer_idx)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(questions)}")

    logits = torch.tensor(logits_rows)
    labels_t = torch.tensor(labels)

    t_fit = fit_temperature(logits, labels_t)
    ece_before = ece(logits, labels_t, temperature=1.0)
    ece_after = ece(logits, labels_t, temperature=t_fit)

    print("\n" + "=" * 60)
    print("  TEMPERATURE SCALING CALIBRATION (Guo et al., 2017)")
    print("=" * 60)
    print(f"  Model:            {args.small}")
    print(f"  Validation size:  {len(questions)}")
    print(f"  Fitted T:         {t_fit:.3f}   ({'softens overconfidence' if t_fit > 1 else 'sharpens'})")
    print(f"  ECE before:       {ece_before:.4f}")
    print(f"  ECE after:        {ece_after:.4f}")
    print(f"  ECE reduction:    {(1 - ece_after / ece_before) * 100:.1f}%" if ece_before > 0 else "")
    print("=" * 60)

    out = {
        "model": args.small,
        "n_validation": len(questions),
        "temperature": round(t_fit, 4),
        "ece_before": round(ece_before, 5),
        "ece_after": round(ece_after, 5),
    }
    with open(outdir / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {outdir}/calibration.json")


if __name__ == "__main__":
    main()
