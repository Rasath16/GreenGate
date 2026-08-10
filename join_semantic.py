"""Join semantic_ablation.csv into main_records.jsonl as a routing signal.

Adds a 'semantic_entropy' field to matching records so summarize_text.py
can sweep it directly:  --signal semantic_entropy
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/main_records.jsonl")
    ap.add_argument("--semantic", default="results/semantic_ablation.csv")
    args = ap.parse_args()

    se = {}
    with open(args.semantic, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            se[int(row["idx"])] = float(row["semantic_entropy"])

    path = Path(args.records)
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    hit = 0
    for r in records:
        if r["idx"] in se:
            r["semantic_entropy"] = se[r["idx"]]
            hit += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"joined semantic entropy into {hit}/{len(records)} records -> {path}")


if __name__ == "__main__":
    main()
