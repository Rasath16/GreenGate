"""
Experiment 3 — API-vs-API cascade (the API-only developer persona).

Small = gpt-4o-mini, large = gpt-4o. Routing signal: average token
log-probability from the API's top-20 logprobs (exact logits are not
exposed by APIs — documented limitation). Carbon is EcoLogits-ESTIMATED
on both tiers (nothing can be physically measured here — reported as
estimates only). Monetary cost is exact, from token counts x published
prices: the unconditional savings claim for API-only developers.

No GPU needed — runs from a laptop.

    python eval_api_cascade.py --n 300
    python eval_api_cascade.py --n 3 --dry-run     # free pipeline test

Then judge quality:
    python judge_openai.py --records results/api_cascade_records.jsonl \
        --out results/judgments_api.jsonl
"""

import argparse
import csv
import json
import random
from pathlib import Path

from greengate.sharegpt import load_sharegpt
from greengate.api_tier import APILargeTier

# USD per 1M tokens (openai.com/api/pricing, checked 2026-08; cite in thesis)
PRICES = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICES[model]
    return tokens_in / 1e6 * p["in"] + tokens_out / 1e6 * p["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--small", default="gpt-4o-mini")
    ap.add_argument("--large", default="gpt-4o")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out_path = outdir / "api_cascade_records.jsonl"

    queries = load_sharegpt(n_queries=args.n, seed=args.seed)
    print(f"{len(queries)} ShareGPT queries (same set as the other experiments)")

    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["idx"]] = rec
        print(f"resuming: {len(done)} complete")

    small = APILargeTier(model=args.small, max_tokens=200,
                         dry_run=args.dry_run, logprobs=True)
    large = APILargeTier(model=args.large, max_tokens=200, dry_run=args.dry_run)

    records = []
    with open(out_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            if i in done:
                records.append(done[i])
                continue
            s = small.query(q)
            l = large.query(q)
            rec = {
                "idx": i, "query": q,
                "small_response": s.response,
                "small_avg_logprob": s.avg_logprob,
                "small_trunc_entropy": s.trunc_entropy,
                "small_carbon_g": s.carbon_grams,
                "small_cost_usd": cost_usd(args.small, s.input_tokens, s.output_tokens)
                                  if not args.dry_run else 0.0001,
                "small_latency_s": s.latency_s,
                "large_response": l.response,
                "large_carbon_g": l.carbon_grams,
                "large_cost_usd": cost_usd(args.large, l.input_tokens, l.output_tokens)
                                  if not args.dry_run else 0.001,
                "large_latency_s": l.latency_s,
                "carbon_source": s.carbon_source,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(queries)}")

    # ---- offline cost/carbon simulation (quality comes later via judging)
    n = len(records)
    rng = random.Random(args.seed)
    sigs = [r["small_avg_logprob"] if r["small_avg_logprob"] is not None else 0.0
            for r in records]

    def sim(flags, small_first):
        cost = carbon = 0.0
        for rec, esc in zip(records, flags):
            if esc:
                cost += rec["large_cost_usd"]
                carbon += rec["large_carbon_g"]
                if small_first:  # full accounting: wasted mini call
                    cost += rec["small_cost_usd"]
                    carbon += rec["small_carbon_g"]
            else:
                cost += rec["small_cost_usd"]
                carbon += rec["small_carbon_g"]
        return {"cost_usd": cost, "carbon_g": carbon,
                "escalation_rate": sum(flags) / n}

    b1 = sim([True] * n, False)
    sorted_sigs = sorted(sigs)
    sweep = []
    for pct in range(5, 100, 5):
        t = sorted_sigs[int(pct / 100 * (n - 1))]
        p = sim([s < t for s in sigs], True)  # LOW logprob -> escalate
        p["logprob_threshold"] = round(t, 4)
        p["cost_saving_pct"] = (1 - p["cost_usd"] / b1["cost_usd"]) * 100
        sweep.append(p)

    print("\n" + "=" * 70)
    print(f"  API-vs-API CASCADE ({args.small} -> {args.large}, {n} queries)")
    print("=" * 70)
    print(f"  B1 always-large:  ${b1['cost_usd']:.4f}  |  {b1['carbon_g']:.3f} g (estimated)")
    b2 = sim([False] * n, False)
    print(f"  B2 always-small:  ${b2['cost_usd']:.4f}  |  {b2['carbon_g']:.3f} g (estimated)")
    mid = sweep[len(sweep) // 2]
    print(f"  cascade @50pct:   ${mid['cost_usd']:.4f}  "
          f"({mid['cost_saving_pct']:.0f}% cost saved, esc {mid['escalation_rate']:.0%})")
    print("  NOTE: quality requires judging (judge_openai.py) before any")
    print("  savings-at-retention claim. Carbon here is estimate-only.")
    print("=" * 70)

    with open(outdir / "api_cascade_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    print(f"Wrote {out_path}, api_cascade_sweep.csv")


if __name__ == "__main__":
    main()
