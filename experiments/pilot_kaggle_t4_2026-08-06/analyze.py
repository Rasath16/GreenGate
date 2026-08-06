"""Analysis of the Kaggle T4 300-question pilot run."""
import pandas as pd

rec = pd.read_csv("records.csv")
sweep = pd.read_csv("threshold_sweep.csv")

n = len(rec)
b1_acc = rec.large_correct.mean()
b1_carbon = rec.large_carbon.sum()
b2_acc = rec.small_correct.mean()
b2_carbon = rec.small_carbon.sum()

print(f"n={n}")
print(f"B1 large: acc={b1_acc:.3f} carbon={b1_carbon:.1f}g  avg/query={b1_carbon/n:.3f}g")
print(f"B2 small: acc={b2_acc:.3f} carbon={b2_carbon:.1f}g  avg/query={b2_carbon/n:.3f}g")
print(f"Energy ratio large/small: {b1_carbon/b2_carbon:.2f}x")

# Retention added to sweep
sweep["retention_pct"] = sweep.accuracy / b1_acc * 100
sweep["carbon_cut_pct"] = (1 - sweep.carbon_g / b1_carbon) * 100
print("\nSweep with retention:")
print(sweep[["threshold", "accuracy", "retention_pct", "carbon_cut_pct", "escalation_rate", "gqos"]].to_string(index=False))

# Question-mix analysis
both_right = ((rec.small_correct == 1) & (rec.large_correct == 1)).mean()
only_large = ((rec.small_correct == 0) & (rec.large_correct == 1)).mean()
only_small = ((rec.small_correct == 1) & (rec.large_correct == 0)).mean()
both_wrong = ((rec.small_correct == 0) & (rec.large_correct == 0)).mean()
print(f"\nQuestion mix: both_right={both_right:.1%} only_large={only_large:.1%} "
      f"only_small={only_small:.1%} both_wrong={both_wrong:.1%}")

# Oracle: escalate exactly when small is wrong and large is right
oracle_esc = (rec.small_correct == 0) & (rec.large_correct == 1)
oracle_acc = (rec.small_correct | rec.large_correct).mean() if False else \
    (rec.small_correct.where(~oracle_esc, rec.large_correct)).mean()
oracle_carbon = rec.small_carbon.sum() + rec.large_carbon[oracle_esc].sum()
print(f"\nOracle router: acc={oracle_acc:.3f} ({oracle_acc/b1_acc*100:.1f}% ret) "
      f"carbon={oracle_carbon:.1f}g ({(1-oracle_carbon/b1_carbon)*100:.1f}% cut) "
      f"esc={oracle_esc.mean():.1%}")

# Does entropy predict correctness? (signal quality)
correct_H = rec.loc[rec.small_correct == 1, "small_entropy"].mean()
wrong_H = rec.loc[rec.small_correct == 0, "small_entropy"].mean()
print(f"\nSignal check: mean entropy when small CORRECT={correct_H:.3f}, "
      f"when WRONG={wrong_H:.3f}  (gap={wrong_H-correct_H:+.3f} bits)")

# Matched-escalation comparison: entropy gate vs random gate
import numpy as np
rng = np.random.default_rng(0)
for target in [0.26, 0.50, 0.65]:
    t_row = sweep.iloc[(sweep.escalation_rate - target).abs().argmin()]
    rand_flags = rng.random(n) < t_row.escalation_rate
    rand_acc = rec.small_correct.where(~rand_flags, rec.large_correct).mean()
    print(f"esc~{t_row.escalation_rate:.0%}: entropy-gate acc={t_row.accuracy:.3f} "
          f"vs random-gate acc={rand_acc:.3f}  (+{(t_row.accuracy-rand_acc)*100:.1f} pts)")
