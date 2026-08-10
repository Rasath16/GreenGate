"""
LLM-as-judge scoring (Layer 1 of the accuracy methodology).

Scores EACH answer once on a 1-5 Likert scale with GPT-4o; scores are
cached to JSONL and reused across every routing policy, so the judging
cost is paid exactly once (~$2-3 for 500 queries x 2 answers).

A local second judge (Llama-3-8B on Kaggle, judge_local.py) re-scores
a subset to check for GPT-4o self-favouritism bias, and 100 queries are
human-annotated (Layer 2, Cohen's kappa) per the methodology.

Usage:
    python judge_openai.py                    # judge everything not yet judged
    python judge_openai.py --limit 20         # first 20 (budget check)
    python judge_openai.py --judge-model gpt-4o-mini   # cheaper judge
"""

import argparse
import json
import time
from pathlib import Path

RUBRIC = """You are evaluating the quality of an AI assistant's answer to a user query.

User query:
{query}

Assistant answer:
{answer}

Rate the answer on a 1-5 scale:
1 = Completely unhelpful, wrong, or off-topic
2 = Mostly unhelpful or contains major errors
3 = Partially helpful but incomplete or with notable issues
4 = Helpful and correct with only minor issues
5 = Fully helpful, correct, and complete

Reply with ONLY the integer rating (1, 2, 3, 4, or 5)."""


def judge_one(client, judge_model: str, query: str, answer: str,
              retries: int = 3) -> int:
    prompt = RUBRIC.format(query=query[:1500], answer=answer[:2000])
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4, temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            for ch in text:
                if ch in "12345":
                    return int(ch)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return 0  # unparseable — flagged for manual review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/text_records.jsonl")
    ap.add_argument("--out", default="results/judgments.jsonl")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--tiers", default="small,large",
                    help="which tiers to judge, e.g. --tiers large")
    args = ap.parse_args()
    wanted_tiers = set(args.tiers.split(","))

    from openai import OpenAI
    client = OpenAI()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    out_path = Path(args.out)

    judged = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            judged[(j["idx"], j["tier"])] = j
    print(f"{len(judged)} judgments already cached")

    todo = []
    for rec in records:
        for tier, key in [("small", "small_response"), ("large", "large_response")]:
            if tier not in wanted_tiers:
                continue
            if (rec["idx"], tier) not in judged and rec.get(key):
                if rec[key].startswith("[DRY RUN"):
                    continue
                todo.append((rec["idx"], tier, rec["query"], rec[key]))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} answers to judge with {args.judge_model}")

    with open(out_path, "a", encoding="utf-8") as f:
        for k, (idx, tier, query, answer) in enumerate(todo):
            score = judge_one(client, args.judge_model, query, answer)
            f.write(json.dumps({"idx": idx, "tier": tier, "score": score,
                                "judge": args.judge_model}) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(todo)}")

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
