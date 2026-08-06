"""
Ablation A: Shannon entropy vs semantic entropy as the routing signal.

For a subset of judged ShareGPT queries, samples the small model k times
(temperature sampling), clusters answers by bidirectional NLI entailment,
and computes semantic entropy. Compares both signals on:

  1. Signal quality — AUROC at predicting "small answer is bad"
     (judge score <= 2), Shannon vs semantic
  2. Energy overhead — measured cost of k extra generations + NLI calls,
     as a multiple of the single-pass Shannon cost

Usage (Kaggle, after eval_text.py + judge_openai.py):
    python eval_semantic.py --n 100 --k 3 --small Qwen/Qwen2.5-0.5B-Instruct
Smoke test (CPU):
    python eval_semantic.py --n 2 --k 2 --small gpt2 --max-new-tokens 30
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from greengate.textgen import SmallTextModel
from greengate.semantic import NLIClusterer, semantic_entropy


def auroc(labels: list[int], scores: list[float]) -> float:
    """AUROC via rank statistic (no sklearn dependency)."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/text_records.jsonl")
    ap.add_argument("--judgments", default="results/judgments.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="samples per query")
    ap.add_argument("--small", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--small-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--nli", default="microsoft/deberta-base-mnli")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    scores = {}
    jpath = Path(args.judgments)
    if jpath.exists():
        for line in jpath.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            scores[(j["idx"], j["tier"])] = j["score"]
        records = [r for r in records if (r["idx"], "small") in scores]
        print(f"{len(records)} judged records available")
    else:
        print("WARNING: no judgments yet — running signal collection only")
    records = records[:args.n]

    model = SmallTextModel(args.small, max_new_tokens=args.max_new_tokens,
                           load_in_4bit=args.small_4bit)
    # enable sampling for diversity (semantic entropy needs variation)
    clusterer = NLIClusterer(args.nli)

    rows = []
    for i, rec in enumerate(records):
        query = rec["query"]
        answers, sample_energy = [], 0.0
        t0 = time.perf_counter()
        for _ in range(args.k):
            with torch.no_grad():
                prompt = model._chat_wrap(query)
                inputs = model.tokenizer(prompt, return_tensors="pt",
                                         truncation=True, max_length=1024)
                inputs = {kk: v.to(model.model.device) for kk, v in inputs.items()}
                model.profiler.start()
                out = model.model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    pad_token_id=model.tokenizer.pad_token_id)
                e, _ = model.profiler.stop()
                sample_energy += e
            gen = out[0][inputs["input_ids"].shape[1]:]
            answers.append(model.tokenizer.decode(gen, skip_special_tokens=True).strip())

        nli_t0 = time.perf_counter()
        clusters = clusterer.cluster(answers)
        se = semantic_entropy(clusters)
        wall = time.perf_counter() - t0

        rows.append({
            "idx": rec["idx"],
            "shannon_raw": rec["small_entropy_raw"],
            "shannon_cal": rec.get("small_entropy_cal", rec["small_entropy_raw"]),
            "semantic_entropy": se,
            "n_clusters": len(set(clusters)),
            "k": args.k,
            "extra_energy_j": sample_energy,
            "base_energy_j": rec["small_energy_j"],
            "overhead_x": (sample_energy + rec["small_energy_j"]) / rec["small_energy_j"]
                          if rec["small_energy_j"] > 0 else float("nan"),
            "wall_s": wall,
            "small_score": scores.get((rec["idx"], "small")),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)}")

    outp = Path(args.outdir) / "semantic_ablation.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {outp}")

    judged = [r for r in rows if r["small_score"] is not None]
    if judged:
        bad = [1 if r["small_score"] <= 2 else 0 for r in judged]
        a_sh = auroc(bad, [r["shannon_raw"] for r in judged])
        a_se = auroc(bad, [r["semantic_entropy"] for r in judged])
        mean_ovh = sum(r["overhead_x"] for r in judged) / len(judged)
        print("\n" + "=" * 62)
        print("  ABLATION A: SHANNON vs SEMANTIC ENTROPY")
        print("=" * 62)
        print(f"  AUROC (predicting bad small answer):")
        print(f"    Shannon (free):        {a_sh:.3f}")
        print(f"    Semantic (k={args.k}):       {a_se:.3f}")
        print(f"  Energy overhead of semantic: {mean_ovh:.1f}x the single pass")
        print("=" * 62)


if __name__ == "__main__":
    main()
