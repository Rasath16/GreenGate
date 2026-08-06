"""Thesis-ready figures for the T4 pilot (run from this folder)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

rec = pd.read_csv("records.csv")
sweep = pd.read_csv("threshold_sweep.csv")

b1_acc, b1_c = rec.large_correct.mean(), rec.large_carbon.sum()
b2_acc, b2_c = rec.small_correct.mean(), rec.small_carbon.sum()

# Oracle: escalate exactly when small wrong & large right
esc = (rec.small_correct == 0) & (rec.large_correct == 1)
oracle_acc = rec.small_correct.where(~esc, rec.large_correct).mean()
oracle_c = rec.small_carbon.sum() + rec.large_carbon[esc].sum()

GREEN, DARK, RED, AMBER, GRAY = "#2D6A4F", "#1B4332", "#DC2626", "#F59E0B", "#6B7280"

# ---- Figure 1: Pareto curve ----
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(sweep.carbon_g, sweep.accuracy, "o-", color=GREEN, lw=2, ms=5,
        label="GreenGate threshold sweep (0.1–1.9 bits)")
for _, r in sweep[sweep.threshold.isin([1.2, 1.4, 1.7])].iterrows():
    ax.annotate(f"t={r.threshold}", (r.carbon_g, r.accuracy),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color=DARK)
ax.scatter([b1_c], [b1_acc], marker="s", s=110, color=RED, zorder=5, label="B1 static large")
ax.scatter([b2_c], [b2_acc], marker="^", s=110, color=AMBER, zorder=5, label="B2 static small")
ax.scatter([oracle_c], [oracle_acc], marker="*", s=220, color=DARK, zorder=5,
           label="Oracle router (upper bound)")
ax.axvline(b1_c, color=RED, ls=":", alpha=0.4)
ax.set_xlabel("Total carbon, full accounting (g CO$_2$, 300 queries)")
ax.set_ylabel("MMLU accuracy")
ax.set_title("Accuracy–carbon trade-off: GreenGate vs baselines (T4, pynvml)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_pareto.png", dpi=200)

# ---- Figure 2: entropy distributions (signal validity) ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(rec.loc[rec.small_correct == 1, "small_entropy"], bins=24, alpha=0.65,
        color=GREEN, label=f"Small model correct (mean {rec.loc[rec.small_correct==1,'small_entropy'].mean():.2f} bits)")
ax.hist(rec.loc[rec.small_correct == 0, "small_entropy"], bins=24, alpha=0.65,
        color=RED, label=f"Small model wrong (mean {rec.loc[rec.small_correct==0,'small_entropy'].mean():.2f} bits)")
ax.set_xlabel("Choice entropy of small model (bits)")
ax.set_ylabel("Questions")
ax.set_title("Routing signal validity: entropy separates correct from wrong answers")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig_entropy_dist.png", dpi=200)

print("Wrote fig_pareto.png, fig_entropy_dist.png")
