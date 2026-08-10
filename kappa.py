"""
Inter-rater agreement for the human annotation layer.

Reads the two filled annotation sheets, reports Cohen's kappa (unweighted
and linear-weighted) between the human annotators, plus mean-score
comparison against the GPT-4o judge for the same (idx, tier) pairs.

Usage:
    python kappa.py --a results/annotator1.csv --b results/annotator2.csv
"""

import argparse
import csv
import json
from pathlib import Path


def read_sheet(path: str) -> dict[int, int]:
    scores = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s = (row.get("score") or "").strip()
            if s and s[0] in "12345":
                scores[int(row["row"])] = int(s[0])
    return scores


def cohens_kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    n = len(a)
    cats = sorted(set(a) | set(b))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1 / n
    pa = [sum(obs[i][j] for j in range(k)) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if not weighted:
            return 0.0 if i == j else 1.0
        return abs(cats[i] - cats[j]) / (max(cats) - min(cats) or 1)

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    pe = sum(w(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - po / pe if pe else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="annotator 1 filled sheet")
    ap.add_argument("--b", required=True, help="annotator 2 filled sheet")
    ap.add_argument("--key", default="results/annotation_key.csv")
    ap.add_argument("--judgments", default="results/judgments_main.jsonl")
    args = ap.parse_args()

    sa, sb = read_sheet(args.a), read_sheet(args.b)
    common = sorted(set(sa) & set(sb))
    if not common:
        raise SystemExit("no overlapping scored rows between the two sheets")
    a = [sa[r] for r in common]
    b = [sb[r] for r in common]

    print(f"rows scored by both annotators: {len(common)}")
    print(f"Cohen's kappa (unweighted):     {cohens_kappa(a, b):.3f}")
    print(f"Cohen's kappa (linear-weighted): {cohens_kappa(a, b, weighted=True):.3f}")
    exact = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
    print(f"exact agreement: {exact:.0%} | within-1: {within1:.0%}")
    print(f"annotator means: A={sum(a)/len(a):.2f}  B={sum(b)/len(b):.2f}")

    # Compare with GPT-4o judge on the same items
    key = {}
    with open(args.key, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key[int(row["row"])] = (int(row["idx"]), row["tier"])
    jpath = Path(args.judgments)
    if jpath.exists():
        judge = {}
        for line in jpath.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            judge[(j["idx"], j["tier"])] = j["score"]
        pairs = [(r, judge[key[r]]) for r in common if key.get(r) in judge]
        if pairs:
            human_mean = sum((sa[r] + sb[r]) / 2 for r, _ in pairs) / len(pairs)
            judge_mean = sum(s for _, s in pairs) / len(pairs)
            print(f"\nvs GPT-4o judge on {len(pairs)} overlapping items:")
            print(f"  human mean {human_mean:.2f} | judge mean {judge_mean:.2f} "
                  f"(diff {judge_mean - human_mean:+.2f})")


if __name__ == "__main__":
    main()
