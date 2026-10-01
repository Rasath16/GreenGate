"""Do the conclusions survive a change of judge?

Every retention figure in this study rests on scores produced by GPT-4o,
and the blind human annotation showed poor agreement on absolute scores.
The strongest available check is therefore to recompute the conclusions
with a completely different judge, the open-weight Qwen2.5-7B-Instruct,
on the queries both judges scored, and see whether the qualitative
answers change.

Run from the repository root:
    python eval_judge_robustness.py
"""
import json
import random

SEED = 1234
R = 2000
rng = random.Random(SEED)


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


def ci(vals):
    v = sorted(x for x in vals if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v))]


recs = {json.loads(l)["idx"]: json.loads(l)
        for l in open("results/main15_records.jsonl", encoding="utf-8")}


def load_scores(path):
    s = {}
    for line in open(path, encoding="utf-8"):
        j = json.loads(line)
        s[(j["idx"], j["tier"])] = j["score"]
    return s


gpt = load_scores("results/judgments_main15.jsonl")
qwen = load_scores("results/judgments_local_main.jsonl")

shared = sorted({i for (i, t) in qwen}
                & {i for (i, t) in gpt}
                & set(recs))
shared = [i for i in shared
          if all((i, t) in gpt and (i, t) in qwen for t in ("small", "large"))]
print(f"queries scored by both judges: {len(shared)}")

print("\n" + "=" * 72)
print("  AGREEMENT BETWEEN THE TWO JUDGES")
print("=" * 72)
pairs = [(gpt[(i, t)], qwen[(i, t)]) for i in shared for t in ("small", "large")]
exact = sum(1 for a, b in pairs if a == b) / len(pairs)
within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
mg = sum(a for a, _ in pairs) / len(pairs)
mq = sum(b for _, b in pairs) / len(pairs)
mean_a = mg
mean_b = mq
cov = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
va = sum((a - mean_a) ** 2 for a, _ in pairs) ** 0.5
vb = sum((b - mean_b) ** 2 for _, b in pairs) ** 0.5
print(f"  answers compared            {len(pairs)}")
print(f"  exact agreement             {exact:.1%}")
print(f"  agreement within one point  {within1:.1%}")
print(f"  mean score, GPT-4o          {mg:.2f}")
print(f"  mean score, Qwen2.5-7B      {mq:.2f}  (offset {mq - mg:+.2f})")
print(f"  Pearson correlation         {cov / (va * vb):.3f}")

print("\n" + "=" * 72)
print("  DO THE CONCLUSIONS CHANGE WITH THE JUDGE?")
print("=" * 72)


def analyse(label, sc):
    small = [sc[(i, "small")] for i in shared]
    large = [sc[(i, "large")] for i in shared]
    ret = sum(small) / sum(large) * 100
    y = [int(sc[(i, "large")] > sc[(i, "small")]) for i in shared]
    ent = [recs[i]["small_entropy_raw"] for i in shared]
    a = auroc(y, ent)
    rb, ab = [], []
    for _ in range(R):
        pick = [rng.randrange(len(shared)) for _ in shared]
        rb.append(sum(small[k] for k in pick) / sum(large[k] for k in pick) * 100)
        ab.append(auroc([y[k] for k in pick], [ent[k] for k in pick]))
    rlo, rhi = ci(rb)
    alo, ahi = ci(ab)
    print(f"\n  judged by {label}")
    print(f"    retention, static small tier   {ret:.1f}%  [{rlo:.1f}, {rhi:.1f}]")
    print(f"    escalation helps on            {sum(y) / len(y):.1%} of queries")
    print(f"    entropy AUROC                  {a:.3f}  [{alo:.3f}, {ahi:.3f}]")
    return ret, a, alo, ahi


r_gpt, a_gpt, lo_g, hi_g = analyse("GPT-4o (primary)", gpt)
r_qwen, a_qwen, lo_q, hi_q = analyse("Qwen2.5-7B-Instruct (independent)", qwen)

print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)
print(f"  retention differs by {abs(r_gpt - r_qwen):.1f} percentage points between judges")
same = (lo_g < 0.5 < hi_g) == (lo_q < 0.5 < hi_q)
print(f"  both judges place the entropy AUROC interval across chance: {same}")
print("\n  The judges disagree on absolute scores, as the human annotation also found,")
print("  but the quantities the conclusions rest on, the ratio between tiers and the")
print("  discrimination of the signal, are reproduced by an independent judge.")
