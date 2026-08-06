"""
Trace replay simulation: sliding-window carbon budget under realistic
arrival patterns (Azure LLM Inference Traces 2024).

Replays the judged ShareGPT records with arrival timestamps drawn from
the Azure trace (or a Poisson fallback), comparing GreenGate WITH a
sliding-window budget vs WITHOUT. Shows budget enforcement working:
escalations get blocked when the window is exhausted, quality degrades
gracefully instead of carbon overshooting.

Runs entirely offline from cached records — no GPU, no API calls.

Usage:
    python eval_trace_replay.py --budget-g 2.0 --window-s 600
"""

import argparse
import csv
import json
import random
from pathlib import Path

from greengate.budget import SlidingWindowBudget

AZURE_TRACE_URL = ("https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
                   "master/data/AzureLLMInferenceTrace_conv.csv")


def load_arrival_times(n: int, seed: int) -> tuple[list[float], str]:
    """Inter-arrival times from the Azure 2024 conversation trace.

    Falls back to a Poisson process (exp inter-arrivals, mean 30s) if the
    trace cannot be downloaded — flagged in the output.
    """
    try:
        import io
        import urllib.request
        with urllib.request.urlopen(AZURE_TRACE_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
        # TIMESTAMP column: ISO timestamps; use successive deltas
        from datetime import datetime
        ts_key = next(k for k in rows[0] if "time" in k.lower())
        stamps = []
        for row in rows:
            try:
                stamps.append(datetime.fromisoformat(row[ts_key].replace("Z", "+00:00")))
            except ValueError:
                continue
        stamps.sort()
        deltas = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        deltas = [d for d in deltas if 0 <= d < 3600]
        rng = random.Random(seed)
        start = rng.randrange(max(1, len(deltas) - n))
        chosen = deltas[start:start + n]
        times, t = [], 0.0
        for d in chosen:
            t += d
            times.append(t)
        while len(times) < n:  # pad if trace segment too short
            t += rng.expovariate(1 / 30)
            times.append(t)
        return times, "azure_trace"
    except Exception as e:
        print(f"  Azure trace unavailable ({e}) — Poisson fallback (mean 30s)")
        rng = random.Random(seed)
        times, t = [], 0.0
        for _ in range(n):
            t += rng.expovariate(1 / 30)
            times.append(t)
        return times, "poisson_fallback"


def replay(records, scores, entropies, threshold, times,
           budget: SlidingWindowBudget | None, intensity=475.0, pue=1.2):
    qual = carbon = 0.0
    escalations = blocked = 0
    for rec, h, now in zip(records, entropies, times):
        c_small = rec["small_energy_j"] / 3_600_000.0 * pue * intensity  # J -> kWh -> g
        wants_escalation = h > threshold
        esc_cost = rec["large_carbon_g"] + c_small
        if wants_escalation and budget is not None and not budget.allows(now, esc_cost):
            wants_escalation = False
            blocked += 1
        if wants_escalation:
            qual += scores.get((rec["idx"], "large"), 0)
            spent = esc_cost
            escalations += 1
        else:
            qual += scores.get((rec["idx"], "small"), 0)
            spent = c_small
        carbon += spent
        if budget is not None:
            budget.record(now, spent)
    n = len(records)
    return {"quality": qual / n, "carbon_g": carbon,
            "escalation_rate": escalations / n, "blocked": blocked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/text_records.jsonl")
    ap.add_argument("--judgments", default="results/judgments.jsonl")
    ap.add_argument("--signal", default="small_entropy_cal")
    ap.add_argument("--threshold", type=float, required=True,
                    help="entropy threshold from summarize_text.py (t*)")
    ap.add_argument("--budget-g", type=float, default=2.0)
    ap.add_argument("--window-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    scores = {}
    for line in Path(args.judgments).read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        scores[(j["idx"], j["tier"])] = j["score"]
    records = [r for r in records
               if (r["idx"], "small") in scores and (r["idx"], "large") in scores]
    entropies = [r[args.signal] for r in records]

    times, source = load_arrival_times(len(records), args.seed)
    print(f"{len(records)} records | arrivals: {source} | "
          f"budget K={args.budget_g}g / {args.window_s:.0f}s window")

    no_budget = replay(records, scores, entropies, args.threshold, times, None)
    with_budget = replay(records, scores, entropies, args.threshold, times,
                         SlidingWindowBudget(args.budget_g, args.window_s))

    print("\n" + "=" * 70)
    print("  TRACE REPLAY — SLIDING WINDOW CARBON BUDGET")
    print("=" * 70)
    for name, p in [("GreenGate (no budget)", no_budget),
                    ("GreenGate (with budget)", with_budget)]:
        print(f"  {name:<26} quality={p['quality']:.3f}  carbon={p['carbon_g']:.4f}g"
              f"  esc={p['escalation_rate']:.0%}  blocked={p['blocked']}")
    print("=" * 70)

    with open(Path(args.outdir) / "trace_replay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "quality", "carbon_g", "escalation_rate",
                    "blocked", "arrival_source", "budget_g", "window_s"])
        for name, p in [("no_budget", no_budget), ("with_budget", with_budget)]:
            w.writerow([name, round(p["quality"], 4), round(p["carbon_g"], 5),
                        round(p["escalation_rate"], 3), p["blocked"],
                        source, args.budget_g, args.window_s])
    print(f"Wrote {args.outdir}/trace_replay.csv")


if __name__ == "__main__":
    main()
