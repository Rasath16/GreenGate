"""
Assemble the MAIN experiment records: Qwen-0.5B (small) vs Mistral-7B (large).

Merges qwen_records.jsonl (small tier, fresh) with text_records.jsonl
(Mistral answers + measured energy, reused as the LOCAL large tier).
Both tiers carry measured Joules, so grid conditions rescale both.

Also seeds judgments_main.jsonl from existing judgments.jsonl: Mistral
answers were already judged there under tier="small" — they become
tier="large" here, so those GPT-4o calls are never paid twice.

Usage:
    python make_main_records.py
Then:
    python judge_openai.py --records results/main_records.jsonl --out results/judgments_main.jsonl
"""

import argparse
import json
from pathlib import Path

PUE = 1.2
WORLD_CI = 475.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="results/qwen_records.jsonl")
    ap.add_argument("--mistral", default="results/text_records.jsonl")
    ap.add_argument("--judgments-in", default="results/judgments.jsonl")
    ap.add_argument("--out", default="results/main_records.jsonl")
    ap.add_argument("--judgments-out", default="results/judgments_main.jsonl")
    args = ap.parse_args()

    qwen = {r["idx"]: r for r in
            (json.loads(l) for l in Path(args.qwen).read_text(encoding="utf-8").splitlines())}
    mistral = {r["idx"]: r for r in
               (json.loads(l) for l in Path(args.mistral).read_text(encoding="utf-8").splitlines())}

    common = sorted(set(qwen) & set(mistral))
    print(f"qwen={len(qwen)} mistral={len(mistral)} merged={len(common)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for i in common:
            q, m = qwen[i], mistral[i]
            assert q["query"] == m["query"], f"query mismatch at idx {i} — seed differs!"
            rec = dict(q)  # small_* fields from Qwen
            rec.update({
                "large_response": m["small_response"],       # Mistral answer
                "large_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "large_energy_j": m["small_energy_j"],       # measured -> grid-rescalable
                "large_carbon_g": m["small_energy_j"] / 3_600_000.0 * PUE * WORLD_CI,
                "large_latency_s": m["small_latency_s"],
                "large_out_tokens": m["small_tokens"],
                # keep the API tier for the secondary (dual-mode) analysis
                "api_response": m.get("large_response"),
                "api_carbon_g": m.get("large_carbon_g"),
                "api_latency_s": m.get("large_latency_s"),
                "api_carbon_source": m.get("large_carbon_source"),
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")

    # Seed judgments: (idx, small) in the old file = Mistral = (idx, large) here
    seeded, existing = 0, set()
    jout = Path(args.judgments_out)
    if jout.exists():
        for line in jout.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            existing.add((j["idx"], j["tier"]))
    with open(jout, "a", encoding="utf-8") as f:
        jin = Path(args.judgments_in)
        if jin.exists():
            for line in jin.read_text(encoding="utf-8").splitlines():
                j = json.loads(line)
                if j["tier"] == "small" and j["idx"] in set(common) \
                        and (j["idx"], "large") not in existing:
                    f.write(json.dumps({"idx": j["idx"], "tier": "large",
                                        "score": j["score"], "judge": j["judge"],
                                        "seeded_from": "mistral_as_small"}) + "\n")
                    seeded += 1
    print(f"seeded {seeded} Mistral judgments into {jout} (no re-judging cost)")
    print("\nNext: python judge_openai.py --records results/main_records.jsonl "
          "--out results/judgments_main.jsonl")


if __name__ == "__main__":
    main()
