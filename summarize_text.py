"""
Final results tables for the ShareGPT text evaluation.

Merges text_records.jsonl (energy/carbon/latency) with judgments.jsonl
(GPT-4o Likert scores) and simulates every routing policy offline:

  B1 static large, B2 static small, B3 random 50%, B4 blind cascade,
  GreenGate (calibrated entropy gate, threshold sweep)

Quality metric: mean judge score (1-5). Accuracy retention = mean score
as % of B1's mean score. gQoS = quality per gram CO2, reported with an
accuracy-retention floor (raw ratio metrics trivially favour the
cheapest model — discussed in thesis).

Grid conditions: local-tier carbon is recomputed from measured energy
under green (50 gCO2/kWh) and dirty (800 gCO2/kWh) grids. API-tier
carbon stays at the EcoLogits estimate (provider grid, not user grid).

Usage:
    python summarize_text.py
    python summarize_text.py --floor 0.85 --signal small_entropy_cal
"""

import argparse
import csv
import json
import random
from pathlib import Path

GRID_CONDITIONS = {"green_grid_50": 50.0, "world_avg_475": 475.0, "dirty_grid_800": 800.0}
PUE = 1.2


def local_carbon(rec: dict, intensity: float) -> float:
    """Recompute local-tier carbon from measured Joules under a grid intensity."""
    return rec["small_energy_j"] / 3_600_000.0 * PUE * intensity  # J -> kWh -> g


def simulate(records, scores, flags, runs_small_first, intensity=475.0):
    n = len(records)
    qual, carbon, latency = 0.0, 0.0, 0.0
    for rec, esc in zip(records, flags):
        s_small = scores.get((rec["idx"], "small"), 0)
        s_large = scores.get((rec["idx"], "large"), 0)
        c_small = local_carbon(rec, intensity)
        if esc:
            qual += s_large
            carbon += rec["large_carbon_g"]
            latency += rec["large_latency_s"]
            if runs_small_first:
                carbon += c_small                      # full accounting
                latency += rec["small_latency_s"]      # sequential cascade
        else:
            qual += s_small
            carbon += c_small
            latency += rec["small_latency_s"]
    return {
        "quality": qual / n,
        "carbon_g": carbon,
        "avg_latency_s": latency / n,
        "escalation_rate": sum(flags) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/text_records.jsonl")
    ap.add_argument("--judgments", default="results/judgments.jsonl")
    ap.add_argument("--signal", default="small_entropy_cal",
                    choices=["small_entropy_cal", "small_entropy_raw"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="report threshold (default: best gQoS above floor)")
    ap.add_argument("--floor", type=float, default=0.80,
                    help="accuracy-retention floor for gQoS ranking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    scores = {}
    for line in Path(args.judgments).read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        scores[(j["idx"], j["tier"])] = j["score"]

    # Keep only records with both judgments
    records = [r for r in records
               if (r["idx"], "small") in scores and (r["idx"], "large") in scores]
    n = len(records)
    if n == 0:
        raise SystemExit("No fully-judged records — run judge_openai.py first.")
    print(f"{n} fully-judged records | signal: {args.signal}")

    rng = random.Random(args.seed)
    entropies = [r[args.signal] for r in records]

    # Threshold sweep on the chosen signal (percentile grid → covers any scale)
    sorted_H = sorted(entropies)
    sweep = []
    for pct in range(5, 100, 5):
        t = sorted_H[int(pct / 100 * (n - 1))]
        flags = [h > t for h in entropies]
        p = simulate(records, scores, flags, True)
        p["threshold"] = round(t, 4)
        p["pct"] = pct
        sweep.append(p)

    b1 = simulate(records, scores, [True] * n, False)

    def enrich(p):
        p["retention_pct"] = p["quality"] / b1["quality"] * 100
        p["carbon_cut_pct"] = (1 - p["carbon_g"] / b1["carbon_g"]) * 100
        p["gqos"] = p["quality"] / p["carbon_g"] if p["carbon_g"] else 0
        return p

    sweep = [enrich(p) for p in sweep]

    # Operating threshold: best gQoS subject to retention floor (or user-set)
    if args.threshold is not None:
        t_op = args.threshold
    else:
        ok = [p for p in sweep if p["retention_pct"] >= args.floor * 100]
        t_op = (max(ok, key=lambda p: p["gqos"]) if ok
                else max(sweep, key=lambda p: p["retention_pct"]))["threshold"]

    gg_flags = [h > t_op for h in entropies]
    gg_rate = sum(gg_flags) / n

    policies = {
        "B1_static_large": enrich(simulate(records, scores, [True] * n, False)),
        "B2_static_small": enrich(simulate(records, scores, [False] * n, False)),
        "B3_random_50": enrich(simulate(records, scores,
                                        [rng.random() < 0.5 for _ in range(n)], False)),
        "B4_blind_cascade": enrich(simulate(records, scores,
                                            [rng.random() < gg_rate for _ in range(n)], True)),
        f"GreenGate_t{t_op}": enrich(simulate(records, scores, gg_flags, True)),
    }

    print("\n" + "=" * 86)
    print(f"  SHAREGPT TEXT EVALUATION  ({n} queries, judge-scored, "
          f"floor={args.floor:.0%}, t*={t_op})")
    print("=" * 86)
    print(f"  {'Policy':<22}{'Quality':>8}{'Ret':>8}{'Carbon(g)':>11}"
          f"{'CO2 cut':>9}{'Esc%':>6}{'Lat(s)':>8}{'gQoS':>8}")
    print("  " + "-" * 82)
    for name, p in policies.items():
        print(f"  {name:<22}{p['quality']:>8.3f}{p['retention_pct']:>7.1f}%"
              f"{p['carbon_g']:>11.4f}{p['carbon_cut_pct']:>8.1f}%"
              f"{p['escalation_rate'] * 100:>5.0f}%{p['avg_latency_s']:>8.2f}"
              f"{p['gqos']:>8.3f}")
    print("=" * 86)

    # Grid conditions for the GreenGate operating point + B1
    print(f"\n  GRID CONDITIONS (GreenGate t={t_op} vs B1)")
    grid_rows = []
    for cond, ci in GRID_CONDITIONS.items():
        gg = simulate(records, scores, gg_flags, True, intensity=ci)
        b1c = simulate(records, scores, [True] * n, False, intensity=ci)
        cut = (1 - gg["carbon_g"] / b1c["carbon_g"]) * 100
        grid_rows.append({"condition": cond, "intensity": ci,
                          "greengate_g": gg["carbon_g"], "b1_g": b1c["carbon_g"],
                          "carbon_cut_pct": cut})
        print(f"  {cond:<18} GG={gg['carbon_g']:>9.4f}g  B1={b1c['carbon_g']:>9.4f}g  cut={cut:.1f}%")

    outdir = Path(args.outdir)
    with open(outdir / "text_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "quality", "retention_pct", "carbon_g",
                    "carbon_cut_pct", "escalation_rate", "avg_latency_s", "gqos"])
        for name, p in policies.items():
            w.writerow([name] + [round(p[k], 5) for k in
                        ["quality", "retention_pct", "carbon_g", "carbon_cut_pct",
                         "escalation_rate", "avg_latency_s", "gqos"]])
    with open(outdir / "text_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    with open(outdir / "text_grid_conditions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)
    print(f"\nWrote text_summary.csv, text_sweep.csv, text_grid_conditions.csv")


if __name__ == "__main__":
    main()
