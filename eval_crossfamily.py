"""Does the open-text signal failure hold for a different model family?

The finding that token entropy does not identify poor answers on
open-ended generation rests on one small model, Qwen2.5-1.5B. This
repeats the measurement with Phi-3.5-mini-instruct, a different vendor,
different training data and different tokeniser, against the same
Mistral-7B large tier, on the same 500 ShareGPT queries under the same
seed and the same GPT-4o judge.

Run from the repository root:
    python eval_crossfamily.py
"""
import json
import random

SEED = 1234
R = 2000
rng = random.Random(SEED)
CF = "experiments/crossfamily_2026-10-01"


def auroc(labels, scores):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n:
                w += 1
            elif p == n:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def load(records_path, judgments_path):
    recs = [json.loads(l) for l in open(records_path, encoding="utf-8")]
    sc = {}
    for line in open(judgments_path, encoding="utf-8"):
        j = json.loads(line)
        sc[(j["idx"], j["tier"])] = j["score"]
    recs = [r for r in recs if (r["idx"], "small") in sc and (r["idx"], "large") in sc]
    for r in recs:
        r["s_small"] = sc[(r["idx"], "small")]
        r["s_large"] = sc[(r["idx"], "large")]
        r["y"] = int(r["s_large"] > r["s_small"])
    return recs


def report(name, recs):
    n = len(recs)
    y = [r["y"] for r in recs]
    ent = [r["small_entropy_raw"] for r in recs]
    a = auroc(y, ent)
    boot = []
    for _ in range(R):
        pick = [rng.randrange(n) for _ in range(n)]
        boot.append(auroc([y[i] for i in pick], [ent[i] for i in pick]))
    boot = sorted(b for b in boot if b == b)
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    ms = sum(r["s_small"] for r in recs) / n
    ml = sum(r["s_large"] for r in recs) / n
    cs = sum(r["small_carbon_g"] for r in recs)
    cl = sum(r["large_carbon_g"] for r in recs)
    gap = (sum(ent[i] for i in range(n) if y[i]) / max(1, sum(y))
           - sum(ent[i] for i in range(n) if not y[i]) / max(1, n - sum(y)))
    print(f"\n  {name}")
    print(f"    escalation helps on            {sum(y) / n:.1%} of queries")
    print(f"    entropy AUROC                  {a:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"    entropy gap (helps minus not)  {gap:+.3f} bits")
    print(f"    judge score small / large      {ms:.2f} / {ml:.2f}")
    print(f"    retention with no escalation   {ms / ml * 100:.1f}%")
    print(f"    C_s/C_l (measured carbon)      {cs / cl:.3f}")
    return dict(name=name, auroc=a, lo=lo, hi=hi, ret=ms / ml * 100, ratio=cs / cl)


print("=" * 74)
print("  CROSS-FAMILY REPLICATION on 500 ShareGPT queries, Mistral-7B large tier")
print("=" * 74)
q = report("Qwen2.5-1.5B-Instruct  (original)",
           load("results/main15_records.jsonl", "results/judgments_main15.jsonl"))
p = report("Phi-3.5-mini-instruct  (replication)",
           load(f"{CF}/phi_records.jsonl", f"{CF}/judgments_phi.jsonl"))

print("\n" + "=" * 74)
print("  VERDICT")
print("=" * 74)
both_chance = q["lo"] < 0.5 < q["hi"] and p["lo"] < 0.5 < p["hi"]
print(f"  Qwen2.5-1.5B  AUROC {q['auroc']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}]")
print(f"  Phi-3.5-mini  AUROC {p['auroc']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]")
if both_chance:
    print("\n  Both intervals contain 0.5, so neither model's token entropy identifies")
    print("  poor answers on open-ended generation. The negative result is a property")
    print("  of the task at this scale rather than of one model family.")
else:
    print("\n  The intervals do not both contain chance; the finding is model-dependent")
    print("  and must be reported as such.")
print(f"\n  Phi answers at {p['ret']:.1f}% of the large tier's judged quality with no")
print(f"  escalation at all, against {q['ret']:.1f}% for Qwen2.5-1.5B.")
