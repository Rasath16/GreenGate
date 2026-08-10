"""
Small-tier pass for the MAIN (local-vs-local) experiment.

Generates answers for the same 500 ShareGPT queries with the true small
model (Qwen2.5-0.5B-Instruct) and records multi-signal entropy
(mean/first/max token + calibrated) plus measured energy. The existing
Mistral-7B answers become the LARGE local tier via make_main_records.py.

Kaggle:
    python eval_small2.py --n 500 --small Qwen/Qwen2.5-0.5B-Instruct
Smoke test (CPU):
    python eval_small2.py --n 2 --small gpt2 --max-new-tokens 20 --outdir <tmp>
"""

import argparse
import json
from pathlib import Path

from greengate.sharegpt import load_sharegpt
from greengate.textgen import SmallTextModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--small", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--small-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42, help="MUST match eval_text.py")
    ap.add_argument("--calibration", default="results/calibration_qwen.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out_path = outdir / "qwen_records.jsonl"

    T = 1.0
    cal = Path(args.calibration)
    if cal.exists():
        T = json.loads(cal.read_text())["temperature"]
        print(f"Using fitted temperature T={T}")
    else:
        print(f"WARNING: {cal} not found — T=1.0 (run calibrate_mmlu.py "
              f"--small {args.small} --outfile {cal} first)")

    queries = load_sharegpt(n_queries=args.n, seed=args.seed)
    print(f"{len(queries)} queries (same seed as eval_text.py -> same set)")

    done = set()
    if out_path.exists():
        done = {json.loads(l)["idx"] for l in
                out_path.read_text(encoding="utf-8").splitlines()}
        print(f"resuming: {len(done)} already complete")

    model = SmallTextModel(args.small, temperature_T=T,
                           max_new_tokens=args.max_new_tokens,
                           load_in_4bit=args.small_4bit)

    with open(out_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            if i in done:
                continue
            r = model.generate(q)
            f.write(json.dumps({
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
                "small_model": args.small,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(queries)}")

    print(f"Done -> {out_path}")
    print("Next: python make_main_records.py")


if __name__ == "__main__":
    main()
