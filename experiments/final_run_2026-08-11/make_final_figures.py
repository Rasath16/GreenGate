"""Thesis figures from the final run (run from repo root)."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("results")
OUT = Path("experiments/final_run_2026-08-11/figs")
OUT.mkdir(parents=True, exist_ok=True)

GREEN, DARK, RED, AMBER, GRAY, BLUE = "#2D6A4F", "#1B4332", "#DC2626", "#F59E0B", "#6B7280", "#1D4ED8"

# ---------- Fig 1: ShareGPT Pareto ----------
sweep = pd.read_csv(R / "text_sweep.csv")
summ = pd.read_csv(R / "text_summary.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(sweep.carbon_g, sweep.quality, "o-", color=GREEN, lw=2, ms=4,
        label="GreenGate threshold sweep (raw signal)")
for name, marker, color in [("B1_static_large", "s", RED), ("B2_static_small", "^", AMBER),
                            ("B3_random_50", "D", GRAY)]:
    r = summ.loc[name]
    ax.scatter([r.carbon_g], [r.quality], marker=marker, s=100, color=color,
               zorder=5, label=name.replace("_", " "))
for _, r in sweep.iloc[[1, 5, 9, 13]].iterrows():
    ax.annotate(f"esc {r.escalation_rate:.0%}", (r.carbon_g, r.quality),
                textcoords="offset points", xytext=(6, -12), fontsize=8, color=DARK)
ax.set_xlabel("Total carbon, full accounting (g CO$_2$, 500 queries)")
ax.set_ylabel("Mean judge score (1–5)")
ax.set_title("ShareGPT: quality–carbon trade-off (Qwen-0.5B → Mistral-7B, measured)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_sharegpt_pareto.png", dpi=200)

# ---------- Fig 2: MMLU raw vs calibrated Pareto + oracle ----------
mm = pd.read_csv(R / "records.csv")
T = json.loads((R / "calibration_qwen.json").read_text())["temperature"]
def cal_H(lj):
    ls = json.loads(lj); m = max(l / T for l in ls)
    ps = [math.exp(l / T - m) for l in ls]; z = sum(ps)
    return -sum(p / z * math.log2(p / z) for p in ps if p > 0)
mm["cal_entropy"] = [cal_H(x) for x in mm.small_logits]
b1a, b1c = mm.large_correct.mean(), mm.large_carbon.sum()
fig, ax = plt.subplots(figsize=(7.5, 5))
for col, color, label in [("small_entropy", GRAY, "raw entropy"),
                          ("cal_entropy", GREEN, "calibrated entropy (T=2.68)")]:
    xs, ys = [], []
    for pct in range(5, 100, 5):
        t = mm[col].quantile(pct / 100)
        f = mm[col] > t
        ys.append(mm.small_correct.where(~f, mm.large_correct).mean())
        xs.append(mm.small_carbon.sum() + mm.large_carbon[f].sum())
    ax.plot(xs, ys, "o-", color=color, lw=2, ms=4, label=label)
esc = (mm.small_correct == 0) & (mm.large_correct == 1)
oa = mm.small_correct.where(~esc, mm.large_correct).mean()
oc = mm.small_carbon.sum() + mm.large_carbon[esc].sum()
ax.scatter([b1c], [b1a], marker="s", s=110, color=RED, zorder=5, label="B1 static large")
ax.scatter([mm.small_carbon.sum()], [mm.small_correct.mean()], marker="^", s=110,
           color=AMBER, zorder=5, label="B2 static small")
ax.scatter([oc], [oa], marker="*", s=240, color=DARK, zorder=5, label="oracle router")
ax.set_xlabel("Total carbon, full accounting (g CO$_2$, 300 questions)")
ax.set_ylabel("MMLU accuracy")
ax.set_title("MMLU: Ablation B — raw vs calibrated routing signal")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_mmlu_ablation_b.png", dpi=200)

# ---------- Fig 3: Ablation A ROC (Shannon vs semantic) ----------
se = pd.read_csv(R / "semantic_ablation.csv").dropna(subset=["small_score"])
bad = (se.small_score <= 2).astype(int)
def roc(scores):
    pts = sorted(zip(scores, bad), key=lambda x: -x[0])
    P, N = bad.sum(), len(bad) - bad.sum()
    tpr, fpr, tp, fp = [0.0], [0.0], 0, 0
    for s, y in pts:
        tp, fp = tp + y, fp + (1 - y)
        tpr.append(tp / P); fpr.append(fp / N)
    return fpr, tpr
fig, ax = plt.subplots(figsize=(6, 5.6))
for col, color, label in [("shannon_raw", GRAY, "Shannon (AUROC 0.539, 1x energy)"),
                          ("semantic_entropy", GREEN, "Semantic k=3 (AUROC 0.439, 4.3x energy)")]:
    f, t = roc(se[col])
    ax.plot(f, t, color=color, lw=2.2, label=label)
ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1, label="chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Ablation A: neither signal detects bad answers on open generation")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_ablation_a_roc.png", dpi=200)

# ---------- Fig 4: vision confidence separation ----------
vr = [json.loads(l) for l in (R / "vision_records.jsonl").read_text(encoding="utf-8").splitlines()]
cor = [r["small_conf"] for r in vr if r["small_correct"]]
wrong = [r["small_conf"] for r in vr if not r["small_correct"]]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(cor, bins=28, alpha=0.65, color=GREEN,
        label=f"correct (mean {sum(cor)/len(cor):.3f}, n={len(cor)})")
ax.hist(wrong, bins=28, alpha=0.65, color=RED,
        label=f"wrong (mean {sum(wrong)/len(wrong):.3f}, n={len(wrong)})")
ax.set_xlabel("Average token probability of SmolVLM answer")
ax.set_ylabel("Questions")
ax.set_title("Vision routing signal: confidence separates correct from wrong (VQAv2)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig_vision_signal.png", dpi=200)

# ---------- Fig 5: ECE before/after both models ----------
cals = [json.loads((R / f).read_text()) for f in ("calibration.json", "calibration_qwen.json")]
fig, ax = plt.subplots(figsize=(6.5, 4))
names = [c["model"].split("/")[-1] for c in cals]
x = range(len(cals))
ax.bar([i - 0.18 for i in x], [c["ece_before"] for c in cals], 0.36,
       color=RED, alpha=0.8, label="ECE before")
ax.bar([i + 0.18 for i in x], [c["ece_after"] for c in cals], 0.36,
       color=GREEN, alpha=0.9, label="ECE after temperature scaling")
for i, c in enumerate(cals):
    ax.text(i + 0.18, c["ece_after"] + 0.005, f"T={c['temperature']:.2f}",
            ha="center", fontsize=9, color=DARK)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Expected Calibration Error")
ax.set_title("Temperature scaling calibration (MMLU validation, held out)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig_calibration_ece.png", dpi=200)

print("wrote 5 figures to", OUT)
