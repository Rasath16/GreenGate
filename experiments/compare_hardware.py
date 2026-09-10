"""Cross-hardware comparison: Tesla T4, A100-SXM4-80GB, H100 80GB HBM3.

The same protocol was run on all three accelerators - 300 MMLU questions,
seed 42, Mistral-7B-Instruct-v0.2 in 4-bit as the large tier, each Qwen2.5
checkpoint as the small tier, entropy threshold 1.0, full carbon accounting -
so every column below is like-for-like except the GPU.

The T4 Qwen2.5-0.5B run is deliberately absent: it predates the profiler
correction of thesis Section 5.4.3 and disagrees with both later runs, which
agree with each other. See the READMEs for details.

Run from the repository root:
    python experiments/compare_hardware.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

MODELS = ["Qwen2.5-0.5B", "Qwen2.5-1.5B-Instruct",
          "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]

# label -> (board power W, directory template); {} is the model name
GPUS = [
    ("T4", 70, os.path.join(ROOT, "results", "zoo_{}")),
    ("A100", 400, os.path.join(HERE, "hardware_a100_2026-09-10", "a100_{}")),
    ("H100", 700, os.path.join(HERE, "hardware_h100_2026-09-10", "h100_{}")),
]


def load(d):
    """Per-run energy, accuracy, mean entropy, and the t=1.0 policy row."""
    rec = pd.read_csv(os.path.join(d, "records.csv"))
    summ = pd.read_csv(os.path.join(d, "summary.csv"))
    gg = summ[summ.policy == "GreenGate_t1.0"].iloc[0]
    small_j, large_j = rec.small_energy.sum(), rec.large_energy.sum()
    return {
        "small_j": small_j,
        "large_j": large_j,
        "ratio": large_j / small_j,
        "cs_over_cl": small_j / large_j,
        "small_acc": rec.small_correct.mean(),
        "large_acc": rec.large_correct.mean(),
        "mean_entropy": rec.small_entropy.mean(),
        "gg_retention": gg.accuracy_retention_pct,
        "gg_carbon_cut": gg.carbon_reduction_pct,
        "gg_esc": gg.escalation_rate,
    }


rows = []
for model in MODELS:
    for gpu, watts, tmpl in GPUS:
        d = tmpl.format(model)
        if not os.path.isdir(d):
            continue
        r = load(d)
        rows.append({
            "small_model": model.replace("-Instruct", ""),
            "gpu": gpu,
            "tdp_w": watts,
            "small_j": round(r["small_j"], 1),
            "large_j": round(r["large_j"], 1),
            "ratio": round(r["ratio"], 2),
            "cs_cl": round(r["cs_over_cl"], 3),
            "small_acc": round(r["small_acc"], 3),
            "large_acc": round(r["large_acc"], 3),
            "entropy": round(r["mean_entropy"], 3),
            "gg_cut": round(r["gg_carbon_cut"], 1),
            "gg_ret": round(r["gg_retention"], 1),
            "gg_esc": round(r["gg_esc"], 3),
        })

df = pd.DataFrame(rows)
out = os.path.join(HERE, "hardware_comparison.csv")
df.to_csv(out, index=False)

pd.set_option("display.width", 220)
print("FULL SERIES")
print(df.to_string(index=False))

print("\nHARDWARE-INVARIANT QUANTITIES")
piv = df.pivot(index="small_model", columns="gpu", values="entropy")
print("mean choice entropy (bits)")
print(piv.to_string())
print("\nsmall-tier accuracy")
print(df.pivot(index="small_model", columns="gpu", values="small_acc").to_string())

print("\nENERGY: same work, different accelerator")
print("large tier (J), identical 4-bit Mistral-7B on every row")
print(df.pivot(index="small_model", columns="gpu", values="large_j").to_string())

print("\nBREAK-EVEN INPUT C_s/C_l (lower = wider escalation budget)")
print(df.pivot(index="small_model", columns="gpu", values="cs_cl").to_string())

print("\nGREENGATE AT t=1.0: carbon cut %")
print(df.pivot(index="small_model", columns="gpu", values="gg_cut").to_string())

# Consistency check on the quantities that must not move with hardware.
print("\nINVARIANCE CHECK")
for model, g in df.groupby("small_model"):
    if len(g) < 2:
        continue
    print(f"  {model:22s} entropy spread {g.entropy.max() - g.entropy.min():.4f} bits"
          f" | accuracy spread {g.small_acc.max() - g.small_acc.min():.4f}")

print("\nwrote", out)
