"""
GreenGate vision routing evaluation (VLM tier).

Small VLM answers every VQA question locally (measured energy, avg token
probability as confidence); GPT-4o-mini vision answers every question via
API (EcoLogits carbon). Accuracy is objective (VQA ground truth) — no LLM
judge needed. Routing policies + confidence sweep simulated offline, same
pattern as eval_mmlu.py.

Kaggle:
    python eval_vision.py --n 500 --small HuggingFaceTB/SmolVLM-Instruct
Smoke test (CPU, no API):
    python eval_vision.py --n 2 --small HuggingFaceTB/SmolVLM-256M-Instruct --dry-run-api
"""

import argparse
import csv
import json
import random
from pathlib import Path

from greengate.vqa import load_vqa, vqa_match
from greengate.visiongen import SmallVisionModel
from greengate.api_tier import APILargeTier


def simulate(records, flags, runs_small_first):
    n = len(records)
    correct = carbon = 0.0
    for rec, esc in zip(records, flags):
        if esc:
            correct += rec["large_correct"]
            carbon += rec["large_carbon_g"]
            if runs_small_first:
                carbon += rec["small_carbon_g"]  # full accounting
        else:
            correct += rec["small_correct"]
            carbon += rec["small_carbon_g"]
    return {"accuracy": correct / n, "carbon_g": carbon,
            "escalation_rate": sum(flags) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--small", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--small-4bit", action="store_true")
    ap.add_argument("--large", default="gpt-4o-mini")
    ap.add_argument("--dry-run-api", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out_path = outdir / "vision_records.jsonl"

    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["idx"]] = rec
        print(f"resuming: {len(done)} records complete")

    print(f"Small VLM: {args.small} | Large: {args.large}"
          + (" [DRY RUN]" if args.dry_run_api else ""))
    small = SmallVisionModel(args.small, load_in_4bit=args.small_4bit)
    large = APILargeTier(model=args.large, max_tokens=60, dry_run=args.dry_run_api)

    records = []
    with open(out_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(load_vqa(args.n, args.seed)):
            if i in done:
                records.append(done[i])
                continue
            s = small.answer(item["image"], item["question"])
            prompt = f"{item['question']} Answer briefly."
            l = large.query_vision(prompt, item["image"])
            rec = {
                "idx": i, "question": item["question"],
                "gt_answer": item["answer"],
                "small_response": s.response,
                "small_correct": int(vqa_match(s.response, item["answer"])),
                "small_conf": s.avg_token_prob,
                "small_entropy": s.entropy_mean,
                "small_energy_j": s.energy_joules,
                "small_carbon_g": s.carbon_grams,
                "small_latency_s": s.latency_s,
                "large_response": l.response,
                "large_correct": int(vqa_match(l.response, item["answer"])),
                "large_carbon_g": l.carbon_grams,
                "large_latency_s": l.latency_s,
                "large_carbon_source": l.carbon_source,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.n}")

    # ---- offline policy simulation (confidence signal: LOW prob -> escalate)
    n = len(records)
    rng = random.Random(args.seed)
    confs = sorted(r["small_conf"] for r in records)
    b1 = simulate(records, [True] * n, False)

    sweep = []
    for pct in range(5, 100, 5):
        t = confs[int(pct / 100 * (n - 1))]
        flags = [r["small_conf"] < t for r in records]  # low confidence escalates
        p = simulate(records, flags, True)
        p["conf_threshold"] = round(t, 4)
        p["retention_pct"] = p["accuracy"] / b1["accuracy"] * 100 if b1["accuracy"] else 0
        p["carbon_cut_pct"] = (1 - p["carbon_g"] / b1["carbon_g"]) * 100
        sweep.append(p)

    best = max((p for p in sweep if p["retention_pct"] >= 80),
               key=lambda p: p["carbon_cut_pct"], default=sweep[-1])
    t_op = best["conf_threshold"]
    gg_flags = [r["small_conf"] < t_op for r in records]
    gg_rate = sum(gg_flags) / n

    policies = {
        "B1_static_large": b1,
        "B2_static_small": simulate(records, [False] * n, False),
        "B3_random_50": simulate(records, [rng.random() < 0.5 for _ in range(n)], False),
        "B4_blind_cascade": simulate(records, [rng.random() < gg_rate for _ in range(n)], True),
        f"GreenGate_c{t_op}": simulate(records, gg_flags, True),
    }

    print("\n" + "=" * 74)
    print(f"  VISION ROUTING (VQAv2, {n} questions, conf threshold={t_op})")
    print("=" * 74)
    print(f"  {'Policy':<22}{'Acc':>7}{'Ret':>8}{'Carbon(g)':>11}{'CO2 cut':>9}{'Esc%':>6}")
    for name, p in policies.items():
        ret = p["accuracy"] / b1["accuracy"] * 100 if b1["accuracy"] else 0
        cut = (1 - p["carbon_g"] / b1["carbon_g"]) * 100 if b1["carbon_g"] else 0
        print(f"  {name:<22}{p['accuracy']:>7.3f}{ret:>7.1f}%{p['carbon_g']:>11.4f}"
              f"{cut:>8.1f}%{p['escalation_rate'] * 100:>5.0f}%")
    print("=" * 74)

    # signal validity: confidence of correct vs wrong small answers
    cor = [r["small_conf"] for r in records if r["small_correct"]]
    wrong = [r["small_conf"] for r in records if not r["small_correct"]]
    if cor and wrong:
        print(f"  signal: mean conf when correct={sum(cor)/len(cor):.3f} "
              f"vs wrong={sum(wrong)/len(wrong):.3f}")

    with open(outdir / "vision_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "accuracy", "carbon_g", "escalation_rate"])
        for name, p in policies.items():
            w.writerow([name, round(p["accuracy"], 4), round(p["carbon_g"], 5),
                        round(p["escalation_rate"], 3)])
    with open(outdir / "vision_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    print(f"\nWrote vision_records.jsonl, vision_summary.csv, vision_sweep.csv")


if __name__ == "__main__":
    main()
