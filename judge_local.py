"""
Local second judge (bias check for GPT-4o self-favouritism).

Re-scores a subset of answers with an open-weight model on Kaggle
(default Qwen2.5-7B-Instruct — ungated; Llama-3-8B requires HF access
approval). Agreement between judges is reported by summarizing the two
judgment files; systematic divergence on GPT-4o-mini's own answers
would indicate self-favouritism bias.

Usage (Kaggle T4):
    python judge_local.py --limit 200
"""

import argparse
import json
from pathlib import Path

import torch

from judge_openai import RUBRIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/text_records.jsonl")
    ap.add_argument("--out", default="results/judgments_local.jsonl")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    kwargs = {"device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, **kwargs)
    model.eval()

    records = [json.loads(l) for l in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            done.add((j["idx"], j["tier"]))

    todo = []
    for rec in records:
        for tier, key in [("small", "small_response"), ("large", "large_response")]:
            if (rec["idx"], tier) not in done and rec.get(key) \
                    and not rec[key].startswith("[DRY RUN"):
                todo.append((rec["idx"], tier, rec["query"], rec[key]))
    todo = todo[:args.limit]
    print(f"{len(todo)} answers to judge locally with {args.judge_model}")

    @torch.no_grad()
    def judge(query: str, answer: str) -> int:
        prompt = RUBRIC.format(query=query[:1500], answer=answer[:2000])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for ch in reply:
            if ch in "12345":
                return int(ch)
        return 0

    with open(out_path, "a", encoding="utf-8") as f:
        for k, (idx, tier, query, answer) in enumerate(todo):
            score = judge(query, answer)
            f.write(json.dumps({"idx": idx, "tier": tier, "score": score,
                                "judge": args.judge_model}) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(todo)}")
    print(f"Done -> {out_path}")
    print("Compare with judgments.jsonl for the bias-check table "
          "(mean score per tier per judge).")


if __name__ == "__main__":
    main()
