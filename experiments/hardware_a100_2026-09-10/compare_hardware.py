"""Cross-hardware comparison: Tesla T4 (70 W) vs A100-SXM4-80GB (400 W).

Both machines ran the identical protocol - 300 MMLU questions, seed 42,
Mistral-7B-Instruct-v0.2 in 4-bit as the large tier, each Qwen2.5 checkpoint
as the small tier - so every column below is like-for-like except the GPU.

Run from the repository root:
    python experiments/hardware_a100_2026-09-10/compare_hardware.py
"""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# T4 runs live in results/zoo_<model>/; A100 runs in this directory.
T4_DIRS = {
    "Qwen2.5-1.5B-Instruct": os.path.join(ROOT, "results", "zoo_Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-3B-Instruct": os.path.join(ROOT, "results", "zoo_Qwen2.5-3B-Instruct"),
    "Qwen2.5-7B-Instruct": os.path.join(ROOT, "results", "zoo_Qwen2.5-7B-Instruct"),
}
A100_DIRS = {
    os.path.basename(d).replace("a100_", ""): d
    for d in sorted(glob.glob(os.path.join(HERE, "a100_*")))
}


def load(d):
    """Per-run energy, accuracy, mean entropy, and the t=1.0 policy row."""
    rec = pd.read_csv(os.path.join(d, "records.csv"))
    summ = pd.read_csv(os.path.join(d, "summary.csv"))
    gg = summ[summ.policy == "GreenGate_t1.0"].iloc[0]
    return {
        "small_j": rec.small_energy.sum(),
        "large_j": rec.large_energy.sum(),
        "ratio": rec.large_energy.sum() / rec.small_energy.sum(),
        "cs_over_cl": rec.small_energy.sum() / rec.large_energy.sum(),
        "small_acc": rec.small_correct.mean(),
        "large_acc": rec.large_correct.mean(),
        "mean_entropy": rec.small_entropy.mean(),
        "gg_retention": gg.accuracy_retention_pct,
        "gg_carbon_cut": gg.carbon_reduction_pct,
        "gg_esc": gg.escalation_rate,
    }


rows = []
for model in T4_DIRS:
    if model not in A100_DIRS:
        continue
    t4, a100 = load(T4_DIRS[model]), load(A100_DIRS[model])
    rows.append({
        "small_model": model,
        "t4_small_j": round(t4["small_j"], 1),
        "t4_large_j": round(t4["large_j"], 1),
        "t4_ratio": round(t4["ratio"], 2),
        "a100_small_j": round(a100["small_j"], 1),
        "a100_large_j": round(a100["large_j"], 1),
        "a100_ratio": round(a100["ratio"], 2),
        "t4_cs_cl": round(t4["cs_over_cl"], 3),
        "a100_cs_cl": round(a100["cs_over_cl"], 3),
        "t4_small_acc": round(t4["small_acc"], 3),
        "a100_small_acc": round(a100["small_acc"], 3),
        "t4_entropy": round(t4["mean_entropy"], 3),
        "a100_entropy": round(a100["mean_entropy"], 3),
        "t4_gg_cut": round(t4["gg_carbon_cut"], 1),
        "a100_gg_cut": round(a100["gg_carbon_cut"], 1),
        "t4_gg_ret": round(t4["gg_retention"], 1),
        "a100_gg_ret": round(a100["gg_retention"], 1),
    })

df = pd.DataFrame(rows)
out = os.path.join(HERE, "hardware_comparison.csv")
df.to_csv(out, index=False)

pd.set_option("display.width", 200)
print("ENERGY (J, 300 questions)")
print(df[["small_model", "t4_small_j", "a100_small_j", "t4_large_j",
          "a100_large_j", "t4_ratio", "a100_ratio"]].to_string(index=False))
print("\nHARDWARE-INVARIANT QUANTITIES (should match)")
print(df[["small_model", "t4_small_acc", "a100_small_acc",
          "t4_entropy", "a100_entropy"]].to_string(index=False))
print("\nBREAK-EVEN INPUT  C_s/C_l  (lower = more headroom)")
print(df[["small_model", "t4_cs_cl", "a100_cs_cl"]].to_string(index=False))
print("\nGREENGATE AT t=1.0")
print(df[["small_model", "t4_gg_cut", "a100_gg_cut",
          "t4_gg_ret", "a100_gg_ret"]].to_string(index=False))
print("\nwrote", out)
