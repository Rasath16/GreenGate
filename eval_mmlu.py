"""
GreenGate MMLU Evaluation
=========================
Runs the small and large model ONCE per question, caches per-question
records, then simulates every routing policy offline from the cache:

  B1  Static large   (always large model)
  B2  Static small   (always small model)
  B3  Random router  (50% escalation, no small-model run first)
  B4  Blind cascade  (small runs, random gate at GreenGate's escalation rate)
  GG  GreenGate      (small runs, entropy gate at --threshold)

Because policies are simulated from cached records, the threshold sweep
costs nothing extra — each model does exactly one forward pass per question.

Full carbon accounting: when a cascade escalates, the query pays for BOTH
the (discarded) small-model run and the large-model run.

Usage (laptop CPU smoke test):
    python eval_mmlu.py --n 10 --small gpt2 --large gpt2-medium

Usage (Kaggle/Colab T4):
    python eval_mmlu.py --n 300 --small Qwen/Qwen2.5-0.5B --large mistralai/Mistral-7B-Instruct-v0.2 --large-4bit
"""

import argparse
import csv
import random
import sys
from pathlib import Path

from greengate.mmlu import load_mmlu
from greengate.evaluator import ChoiceEvaluator


def simulate_policy(records: list[dict], escalate_flags: list[bool],
                    runs_small_first: bool) -> dict:
    """Compute policy metrics from cached per-question records.

    escalate_flags[i] — whether query i is answered by the large model.
    runs_small_first  — cascade policies pay the small-model cost on
                        escalated queries too (full carbon accounting).
    """
    n = len(records)
    correct = 0
    carbon = 0.0
    energy = 0.0
    for rec, esc in zip(records, escalate_flags):
        if esc:
            correct += rec["large_correct"]
            carbon += rec["large_carbon"]
            energy += rec["large_energy"]
            if runs_small_first:  # wasted small run — full accounting
                carbon += rec["small_carbon"]
                energy += rec["small_energy"]
        else:
            correct += rec["small_correct"]
            carbon += rec["small_carbon"]
            energy += rec["small_energy"]
    accuracy = correct / n
    return {
        "accuracy": accuracy,
        "carbon_g": carbon,
        "energy_j": energy,
        "escalation_rate": sum(escalate_flags) / n,
        "gqos": accuracy / carbon if carbon > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of MMLU questions")
    ap.add_argument("--small", default="gpt2")
    ap.add_argument("--large", default="gpt2-medium")
    ap.add_argument("--large-4bit", action="store_true", help="load large model in 4-bit (GPU only)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="entropy threshold in bits (choice entropy range: 0-2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Loading {args.n} MMLU questions...")
    questions = load_mmlu(n_questions=args.n, seed=args.seed)

    # ---- Pass 1: small model on every question ----
    print(f"\n[1/2] Evaluating small model: {args.small}")
    small = ChoiceEvaluator(args.small)
    records = []
    for i, q in enumerate(questions):
        r = small.evaluate(q)
        records.append({
            "idx": i,
            "subject": q.subject,
            "small_correct": int(r.correct),
            "small_entropy": r.entropy,
            "small_energy": r.energy_joules,
            "small_carbon": r.carbon_grams,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(questions)}")
    del small  # free memory before loading the large model

    # ---- Pass 2: large model on every question ----
    print(f"\n[2/2] Evaluating large model: {args.large}")
    large = ChoiceEvaluator(args.large, load_in_4bit=args.large_4bit)
    for i, q in enumerate(questions):
        r = large.evaluate(q)
        records[i].update({
            "large_correct": int(r.correct),
            "large_entropy": r.entropy,
            "large_energy": r.energy_joules,
            "large_carbon": r.carbon_grams,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(questions)}")
    del large

    # ---- Simulate policies from cache ----
    n = len(records)
    gg_flags = [rec["small_entropy"] > args.threshold for rec in records]
    gg_rate = sum(gg_flags) / n

    policies = {
        "B1_static_large": simulate_policy(records, [True] * n, runs_small_first=False),
        "B2_static_small": simulate_policy(records, [False] * n, runs_small_first=False),
        "B3_random_50": simulate_policy(
            records, [random.random() < 0.5 for _ in range(n)], runs_small_first=False),
        "B4_blind_cascade": simulate_policy(
            records, [random.random() < gg_rate for _ in range(n)], runs_small_first=True),
        f"GreenGate_t{args.threshold}": simulate_policy(records, gg_flags, runs_small_first=True),
    }

    b1 = policies["B1_static_large"]
    print("\n" + "=" * 78)
    print(f"  RESULTS  ({n} MMLU questions, threshold = {args.threshold} bits)")
    print("=" * 78)
    header = f"  {'Policy':<22}{'Acc':>7}{'AccRet':>8}{'Carbon(g)':>11}{'CO2 cut':>9}{'Esc%':>7}{'gQoS':>9}"
    print(header)
    print("  " + "-" * 74)
    for name, p in policies.items():
        acc_ret = p["accuracy"] / b1["accuracy"] * 100 if b1["accuracy"] else 0
        co2_cut = (1 - p["carbon_g"] / b1["carbon_g"]) * 100 if b1["carbon_g"] else 0
        print(f"  {name:<22}{p['accuracy']:>7.3f}{acc_ret:>7.1f}%{p['carbon_g']:>11.4f}"
              f"{co2_cut:>8.1f}%{p['escalation_rate'] * 100:>6.0f}%{p['gqos']:>9.3f}")
    print("=" * 78)

    # ---- Threshold sweep (free — replayed from cache) ----
    sweep_rows = []
    for t in [round(0.1 * k, 1) for k in range(1, 20)]:
        flags = [rec["small_entropy"] > t for rec in records]
        p = simulate_policy(records, flags, runs_small_first=True)
        p["threshold"] = t
        sweep_rows.append(p)

    # ---- Write CSVs ----
    with open(outdir / "records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "accuracy", "accuracy_retention_pct", "carbon_g",
                    "carbon_reduction_pct", "escalation_rate", "gqos"])
        for name, p in policies.items():
            w.writerow([name, f"{p['accuracy']:.4f}",
                        f"{p['accuracy'] / b1['accuracy'] * 100:.2f}",
                        f"{p['carbon_g']:.6f}",
                        f"{(1 - p['carbon_g'] / b1['carbon_g']) * 100:.2f}",
                        f"{p['escalation_rate']:.3f}", f"{p['gqos']:.4f}"])

    with open(outdir / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "accuracy", "carbon_g", "escalation_rate", "gqos"])
        for row in sweep_rows:
            w.writerow([row["threshold"], f"{row['accuracy']:.4f}",
                        f"{row['carbon_g']:.6f}", f"{row['escalation_rate']:.3f}",
                        f"{row['gqos']:.4f}"])

    print(f"\nWrote {outdir}/records.csv, summary.csv, threshold_sweep.csv")

    # ---- Pareto plot (optional) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [r["carbon_g"] for r in sweep_rows]
        ys = [r["accuracy"] for r in sweep_rows]
        ax.plot(xs, ys, "o-", color="#2D6A4F", label="GreenGate (threshold sweep)")
        ax.scatter([b1["carbon_g"]], [b1["accuracy"]], marker="s", s=80,
                   color="#DC2626", zorder=5, label="B1 static large")
        b2 = policies["B2_static_small"]
        ax.scatter([b2["carbon_g"]], [b2["accuracy"]], marker="^", s=80,
                   color="#F59E0B", zorder=5, label="B2 static small")
        ax.set_xlabel("Total carbon (g CO2, full accounting)")
        ax.set_ylabel("Accuracy")
        ax.set_title("GreenGate accuracy-carbon Pareto curve (MMLU)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "pareto_curve.png", dpi=150)
        print(f"Wrote {outdir}/pareto_curve.png")
    except ImportError:
        print("matplotlib not installed — skipping Pareto plot")


if __name__ == "__main__":
    main()
