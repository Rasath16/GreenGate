"""Copy cached judgments between files with a tier rename — never pay to
judge the same answer twice.

The Mistral answers appear as tier="large" in judgments_main.jsonl (main
experiment) and as tier="small" in judgments.jsonl (Mistral-vs-API
secondary analysis). After judging one file, seed the other:

    python seed_judgments.py --src results/judgments_main.jsonl \
        --dst results/judgments.jsonl --map large=small
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--map", required=True, help="srctier=dsttier, e.g. large=small")
    args = ap.parse_args()

    src_tier, dst_tier = args.map.split("=")

    existing = set()
    dst = Path(args.dst)
    if dst.exists():
        for line in dst.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            existing.add((j["idx"], j["tier"]))

    seeded = 0
    with open(dst, "a", encoding="utf-8") as f:
        for line in Path(args.src).read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            if j["tier"] == src_tier and (j["idx"], dst_tier) not in existing:
                f.write(json.dumps({"idx": j["idx"], "tier": dst_tier,
                                    "score": j["score"], "judge": j["judge"],
                                    "seeded": True}) + "\n")
                seeded += 1
    print(f"seeded {seeded} judgments ({src_tier} -> {dst_tier}) into {dst}")


if __name__ == "__main__":
    main()
