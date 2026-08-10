"""
GreenGate ShareGPT text evaluation — data collection phase.

Runs the SMALL local model (pynvml-measured) and the LARGE API tier
(EcoLogits-estimated) once per query, saving everything to JSONL.
Routing policies, threshold sweeps and grid conditions are simulated
offline afterwards; answers are scored once by judge_openai.py and
reused across all policies.

Smoke test (CPU, no API calls):
    python eval_text.py --n 3 --small gpt2 --dry-run-api

Kaggle run:
    python eval_text.py --n 500 --small mistralai/Mistral-7B-Instruct-v0.2 --small-4bit
    (or Qwen/Qwen2.5-0.5B-Instruct without --small-4bit)
"""

import argparse
import json
from pathlib import Path

from greengate.sharegpt import load_sharegpt
from greengate.textgen import SmallTextModel
from greengate.api_tier import APILargeTier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--small", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--small-4bit", action="store_true")
    ap.add_argument("--large", default="gpt-4o-mini")
    ap.add_argument("--dry-run-api", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out_path = outdir / "text_records.jsonl"

    # Fitted temperature from calibrate_mmlu.py (1.0 if not run yet)
    T = 1.0
    cal_path = Path(args.calibration)
    if cal_path.exists():
        T = json.loads(cal_path.read_text())["temperature"]
        print(f"Using fitted temperature T={T}")
    else:
        print("WARNING: no calibration.json found — using T=1.0 (uncalibrated)")

    print(f"Loading {args.n} ShareGPT queries (PII-scrubbed)...")
    queries = load_sharegpt(n_queries=args.n, seed=args.seed)
    print(f"  got {len(queries)} queries")

    # Resume support: skip already-completed queries.
    # Small-pass results are checkpointed incrementally to small_path so a
    # dead session never loses more than the current query.
    small_path = outdir / "text_records_small.jsonl"
    done = {}
    for path in (out_path, small_path):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                done.setdefault(rec["idx"], rec)
    if done:
        print(f"  resuming: {len(done)} small-pass records already complete")

    print(f"\n[1/2] Small model: {args.small}")
    need_small = [i for i in range(len(queries)) if i not in done]
    small = None
    if need_small:
        small = SmallTextModel(args.small, temperature_T=T,
                               max_new_tokens=args.max_new_tokens,
                               load_in_4bit=args.small_4bit)

    records = []
    with open(small_path, "a", encoding="utf-8") as ckpt:
        for i, q in enumerate(queries):
            if i in done:
                records.append(done[i])
                continue
            r = small.generate(q)
            rec = {
                "idx": i, "query": q,
                "small_response": r.response,
                "small_entropy_raw": r.entropy_raw,
                "small_entropy_cal": r.entropy_calibrated,
                "small_entropy_first": r.entropy_first,
                "small_entropy_max": r.entropy_max,
                "small_energy_j": r.energy_joules,
                "small_carbon_g": r.carbon_grams,
                "small_latency_s": r.latency_s,
                "small_tokens": r.output_tokens,
            }
            records.append(rec)
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(queries)}")
    if small is not None:
        del small

    print(f"\n[2/2] Large API tier: {args.large}"
          + (" [DRY RUN]" if args.dry_run_api else ""))
    large = APILargeTier(model=args.large, dry_run=args.dry_run_api)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if "large_response" not in rec:
                r = large.query(rec["query"])
                rec.update({
                    "large_response": r.response,
                    "large_model": r.model,
                    "large_in_tokens": r.input_tokens,
                    "large_out_tokens": r.output_tokens,
                    "large_latency_s": r.latency_s,
                    "large_energy_wh": r.energy_wh,
                    "large_carbon_g": r.carbon_grams,
                    "large_carbon_source": r.carbon_source,
                })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(records)}")

    total_small_c = sum(r["small_carbon_g"] for r in records)
    total_large_c = sum(r["large_carbon_g"] for r in records)
    print(f"\nDone. {len(records)} records -> {out_path}")
    print(f"Small tier total: {total_small_c:.3f} g CO2 (measured, pynvml)"
          if not args.dry_run_api else
          f"Small tier total: {total_small_c:.3f} g CO2")
    print(f"Large tier total: {total_large_c:.3f} g CO2 "
          f"({records[0].get('large_carbon_source', 'n/a')})")
    print("\nNext: python judge_openai.py  (scores each answer once, ~$2-3)")


if __name__ == "__main__":
    main()
