"""
Layer 2 accuracy methodology: export 100 queries for blind human annotation.

Produces:
  annotation_sheet.csv — what annotators fill in: query + answer with the
      tier HIDDEN and order shuffled (no bias toward either model).
      Give one copy to each of the two annotators; they fill 'score' (1-5)
      using the same rubric as the LLM judges.
  annotation_key.csv   — private mapping (row -> idx/tier). Do NOT show
      annotators. Used by kappa.py to align with LLM judgments.

Usage:
    python export_annotation.py --records results/main_records.jsonl --n 100
"""

import argparse
import csv
import json
import random
from pathlib import Path

RUBRIC_NOTE = ("Score 1-5: 1=unhelpful/wrong, 2=major errors, "
               "3=partially helpful, 4=helpful minor issues, 5=fully correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/main_records.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    rows = []
    for rec in sample:
        # one answer per query, tier chosen at random -> 50/50 blind mix
        tier = rng.choice(["small", "large"])
        rows.append({"idx": rec["idx"], "tier": tier, "query": rec["query"],
                     "answer": rec[f"{tier}_response"]})
    rng.shuffle(rows)

    outdir = Path(args.outdir)
    with open(outdir / "annotation_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "query", "answer", "score", RUBRIC_NOTE])
        for k, r in enumerate(rows):
            w.writerow([k, r["query"], r["answer"], "", ""])

    with open(outdir / "annotation_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "idx", "tier"])
        for k, r in enumerate(rows):
            w.writerow([k, r["idx"], r["tier"]])

    print(f"wrote {outdir}/annotation_sheet.csv ({len(rows)} rows) — give a COPY "
          f"to each annotator\nwrote {outdir}/annotation_key.csv — keep private")


if __name__ == "__main__":
    main()
