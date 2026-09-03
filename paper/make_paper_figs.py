"""Figures for the ICACT paper (run from repo root)."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8})
GREEN, DARK, RED, AMBER, GRAY = "#2D6A4F", "#1B4332", "#C1272D", "#E08A1E", "#6B7280"

# ---- Fig 1: signal informativeness vs small-model capability (zoo) ----
rows = []
for d in sorted(glob.glob("results/zoo_*")) + ["results"]:
    rec = os.path.join(d, "records.csv")
    if not os.path.exists(rec):
        continue
    slug = "Qwen2.5-0.5B" if d == "results" else os.path.basename(d)[4:]
    df = pd.read_csv(rec)
    gap = (df.loc[df.small_correct == 0, "small_entropy"].mean()
           - df.loc[df.small_correct == 1, "small_entropy"].mean())
    rows.append((slug.replace("-Instruct", "").replace("-Chat-v1.0", ""),
                 df.small_correct.mean(), gap))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
ax.scatter(xs, ys, s=45, color=GREEN, zorder=5)
for name, x, y in rows:
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, -3),
                fontsize=6, color=DARK)
ax.axvline(0.25, color=RED, ls=":", lw=1)
ax.text(0.253, 0.02, "4-choice chance", fontsize=6, color=RED, rotation=90, va="bottom")
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xlabel("Small-model MMLU accuracy", fontsize=8)
ax.set_ylabel("Entropy gap (bits)\nwrong minus correct", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_signal_vs_capability.png", dpi=300)
plt.close(fig)

# ---- Fig 2: ShareGPT Pareto, both small tiers, full vs naive accounting ----
import numpy as np
recs15 = [json.loads(l) for l in open("results/main15_records.jsonl", encoding="utf-8")]
sc = {}
for line in open("results/judgments_main15.jsonl", encoding="utf-8"):
    j = json.loads(line); sc[(j["idx"], j["tier"])] = j["score"]
recs15 = [r for r in recs15 if (r["idx"], "small") in sc and (r["idx"], "large") in sc]
PUE, CI = 1.2, 475.0

def curve(recs, scores, full=True):
    hs = sorted(r["small_entropy_raw"] for r in recs)
    out = []
    for pct in range(5, 100, 5):
        t = hs[int(pct / 100 * (len(hs) - 1))]
        q = c = 0.0
        for r in recs:
            cs = r["small_energy_j"] / 3.6e6 * PUE * CI
            cl = r["large_energy_j"] / 3.6e6 * PUE * CI
            if r["small_entropy_raw"] > t:
                q += scores[(r["idx"], "large")]; c += cl + (cs if full else 0)
            else:
                q += scores[(r["idx"], "small")]; c += cs
        out.append((c, q / len(recs)))
    return out

b1c = sum(r["large_energy_j"] for r in recs15) / 3.6e6 * PUE * CI
b1q = sum(sc[(r["idx"], "large")] for r in recs15) / len(recs15)
full_c = curve(recs15, sc, True); naive_c = curve(recs15, sc, False)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot([p[0] for p in full_c], [p[1] for p in full_c], "o-", color=GREEN,
        ms=3, lw=1.4, label="Full accounting")
ax.plot([p[0] for p in naive_c], [p[1] for p in naive_c], "s--", color=AMBER,
        ms=3, lw=1.2, label="Naive accounting")
ax.scatter([b1c], [b1q], marker="*", s=90, color=RED, zorder=5, label="Always-large")
ax.set_xlabel("Carbon (g CO$_2$, 500 queries)", fontsize=8)
ax.set_ylabel("Mean judge score (1-5)", fontsize=8)
ax.legend(fontsize=6.5, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_accounting_pareto.png", dpi=300)
plt.close(fig)

print("wrote 2 paper figures")
for name, acc, gap in rows:
    print(f"  {name}: acc={acc:.3f} gap={gap:+.3f}")
