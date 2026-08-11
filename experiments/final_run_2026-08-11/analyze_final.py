"""Comprehensive analysis of the final from-scratch run (run from repo root)."""
import json
import statistics
from pathlib import Path

import pandas as pd

R = Path("results")

# ---------- 1. Judge agreement & bias ----------
jm = {(j["idx"], j["tier"]): j["score"] for j in
      (json.loads(l) for l in (R / "judgments_main.jsonl").read_text(encoding="utf-8").splitlines())}
jl = {(j["idx"], j["tier"]): j["score"] for j in
      (json.loads(l) for l in (R / "judgments_local_main.jsonl").read_text(encoding="utf-8").splitlines())}
common = sorted(set(jm) & set(jl))
diffs = [jm[k] - jl[k] for k in common]
print("=" * 70)
print("JUDGE AGREEMENT (GPT-4o vs Qwen2.5-7B local), n =", len(common))
exact = sum(1 for k in common if jm[k] == jl[k]) / len(common)
w1 = sum(1 for k in common if abs(jm[k] - jl[k]) <= 1) / len(common)
print(f"  exact {exact:.0%} | within-1 {w1:.0%} | mean diff {statistics.mean(diffs):+.2f}")
for tier in ("small", "large"):
    ks = [k for k in common if k[1] == tier]
    print(f"  {tier:>5}: gpt4o {statistics.mean(jm[k] for k in ks):.2f} "
          f"vs local {statistics.mean(jl[k] for k in ks):.2f} (n={len(ks)})")

# ---------- 2. Text sweep: operating points at multiple floors ----------
sweep = pd.read_csv(R / "text_sweep.csv")
print("\n" + "=" * 70)
print("SHAREGPT OPERATING POINTS (raw signal sweep)")
for floor in (95, 90, 85, 80):
    ok = sweep[sweep.retention_pct >= floor]
    if len(ok):
        best = ok.loc[ok.carbon_cut_pct.idxmax()]
        print(f"  floor {floor}%: ret {best.retention_pct:5.1f}% | cut {best.carbon_cut_pct:5.1f}% "
              f"| esc {best.escalation_rate:.0%} | t={best.threshold}")
    else:
        print(f"  floor {floor}%: unreachable with this signal/pair")

# ---------- 3. MMLU: raw vs CALIBRATED routing (Ablation B, offline) ----------
mm = pd.read_csv(R / "records.csv")
import math
def cal_entropy(logits_json, T):
    ls = json.loads(logits_json)
    m = max(l / T for l in ls)
    ps = [math.exp(l / T - m) for l in ls]
    z = sum(ps)
    return -sum(p / z * math.log2(p / z) for p in ps if p > 0)
T_qwen = json.loads((R / "calibration_qwen.json").read_text())["temperature"]
mm["cal_entropy"] = [cal_entropy(lj, T_qwen) for lj in mm.small_logits]
b1_acc, b1_c = mm.large_correct.mean(), mm.large_carbon.sum()
def sim(flags):
    acc = mm.small_correct.where(~flags, mm.large_correct).mean()
    car = mm.small_carbon.sum() + mm.large_carbon[flags].sum()
    return acc, car
print("\n" + "=" * 70)
print(f"MMLU ABLATION B (raw vs calibrated routing signal), B1 acc={b1_acc:.3f}")
for name, col in (("raw", "small_entropy"), ("calibrated", "cal_entropy")):
    rows = []
    for pct in range(10, 100, 10):
        t = mm[col].quantile(pct / 100)
        acc, car = sim(mm[col] > t)
        rows.append((acc / b1_acc * 100, (1 - car / b1_c) * 100))
    # best cut with >=95% retention
    ok = [r for r in rows if r[0] >= 95]
    best = max(ok, key=lambda r: r[1]) if ok else max(rows, key=lambda r: r[0])
    print(f"  {name:>10}: best >=95% point -> ret {best[0]:.1f}% cut {best[1]:.1f}%"
          + ("" if ok else "  (95% unreachable)"))
# signal separation
gap_raw = mm.loc[mm.small_correct == 0, "small_entropy"].mean() - mm.loc[mm.small_correct == 1, "small_entropy"].mean()
print(f"  entropy gap (wrong-correct): {gap_raw:+.3f} bits")

# ---------- 4. Vision operating points ----------
vs = pd.read_csv(R / "vision_sweep.csv")
print("\n" + "=" * 70)
print("VISION: retention >=100% points (small beats large)")
top = vs[vs.retention_pct >= 100].sort_values("carbon_cut_pct", ascending=False).head(3)
for _, r in top.iterrows():
    print(f"  conf_t={r.conf_threshold}: ret {r.retention_pct:.1f}% | esc {r.escalation_rate:.0%}")

# ---------- 5. Measurement variance (3 runs of first 50 queries) ----------
def small_energy(path):
    return {r["idx"]: r["small_energy_j"] for r in
            (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines())}
e1 = small_energy(R / "text_records.jsonl")
e2 = small_energy(R / "rep2" / "text_records.jsonl")
e3 = small_energy(R / "rep3" / "text_records.jsonl")
idx = sorted(set(e2) & set(e3) & set(e1))[:50]
tot = [sum(e[i] for i in idx) for e in (e1, e2, e3)]
mean = statistics.mean(tot)
cv = statistics.stdev(tot) / mean * 100
print("\n" + "=" * 70)
print(f"MEASUREMENT VARIANCE (Mistral, 50 queries x 3 runs)")
print(f"  totals: {[f'{t:.1f}J' for t in tot]} | mean {mean:.1f}J | CV {cv:.1f}%")

# ---------- 6. Semantic entropy quantization ----------
se = pd.read_csv(R / "semantic_ablation.csv")
print("\n" + "=" * 70)
print("SEMANTIC ENTROPY VALUE DISTRIBUTION (k=3 quantization)")
print(se.semantic_entropy.round(3).value_counts().to_string())
print(f"  mean overhead: {se.overhead_x.mean():.1f}x")
