"""Train/validation/test protocol: thresholds are selected on a validation
split and reported only on a held-out test split. Also reports bootstrap
confidence intervals for every primary metric. Run from repo root."""
import json
import random
import pandas as pd

PUE, CI_WORLD = 1.2, 475.0
R = 2000
SEED = 1234


def ci(vals, lo=0.025, hi=0.975):
    v = sorted(vals)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def fmt(pt, lo, hi, unit="%"):
    return f"{pt:.1f}{unit} [{lo:.1f}, {hi:.1f}]"


# ============================ ShareGPT ============================
recs = [json.loads(l) for l in open("results/main15_records.jsonl", encoding="utf-8")]
sc = {}
for line in open("results/judgments_main15.jsonl", encoding="utf-8"):
    j = json.loads(line)
    sc[(j["idx"], j["tier"])] = j["score"]
recs = [r for r in recs if (r["idx"], "small") in sc and (r["idx"], "large") in sc]

rng = random.Random(SEED)
idx = list(range(len(recs)))
rng.shuffle(idx)
half = len(idx) // 2
val = [recs[i] for i in idx[:half]]
test = [recs[i] for i in idx[half:]]
print(f"ShareGPT: {len(recs)} judged records -> validation {len(val)}, test {len(test)}")


def sg_metrics(sample, t, signal="small_entropy_raw"):
    q = c = b1q = b1c = 0.0
    esc = 0
    for r in sample:
        cs = r["small_energy_j"] / 3.6e6 * PUE * CI_WORLD
        cl = r["large_energy_j"] / 3.6e6 * PUE * CI_WORLD
        b1q += sc[(r["idx"], "large")]
        b1c += cl
        if r[signal] > t:
            q += sc[(r["idx"], "large")]
            c += cl + cs
            esc += 1
        else:
            q += sc[(r["idx"], "small")]
            c += cs
    return (q / b1q * 100, (1 - c / b1c) * 100, esc / len(sample))


# --- select threshold on VALIDATION only, at a 95% retention floor ---
hs = sorted(r["small_entropy_raw"] for r in val)
best_t, best_cut = None, -1e9
for pct in range(2, 99):
    t = hs[int(pct / 100 * (len(hs) - 1))]
    ret, cut, _ = sg_metrics(val, t)
    if ret >= 95 and cut > best_cut:
        best_cut, best_t = cut, t
if best_t is None:                       # floor unreachable on val: take max retention
    best_t = max(hs, key=lambda t: sg_metrics(val, t)[0])
v_ret, v_cut, v_esc = sg_metrics(val, best_t)
print(f"  threshold selected on validation: {best_t:.4f} "
      f"(val: retention {v_ret:.1f}%, cut {v_cut:.1f}%, esc {v_esc:.0%})")

# --- report on TEST only ---
t_ret, t_cut, t_esc = sg_metrics(test, best_t)
rets, cuts = [], []
for _ in range(R):
    s = [test[rng.randrange(len(test))] for _ in range(len(test))]
    a, b, _ = sg_metrics(s, best_t)
    rets.append(a); cuts.append(b)
print("  HELD-OUT TEST RESULT:")
print(f"    retention   {fmt(t_ret, *ci(rets))}")
print(f"    energy cut  {fmt(t_cut, *ci(cuts))}")
print(f"    escalation  {t_esc:.0%}")

# --- baselines on the same test split, with CIs ---
def sg_baseline(sample, mode):
    q = c = b1q = b1c = 0.0
    for r in sample:
        cs = r["small_energy_j"] / 3.6e6 * PUE * CI_WORLD
        cl = r["large_energy_j"] / 3.6e6 * PUE * CI_WORLD
        b1q += sc[(r["idx"], "large")]; b1c += cl
        if mode == "large":
            q += sc[(r["idx"], "large")]; c += cl
        elif mode == "small":
            q += sc[(r["idx"], "small")]; c += cs
        else:                                   # random 50%
            if rng.random() < 0.5:
                q += sc[(r["idx"], "large")]; c += cl
            else:
                q += sc[(r["idx"], "small")]; c += cs
    return q / b1q * 100, (1 - c / b1c) * 100

for mode, label in (("small", "static small"), ("random", "random 50%")):
    pr, pc = sg_baseline(test, mode)
    rr, cc = [], []
    for _ in range(R):
        s = [test[rng.randrange(len(test))] for _ in range(len(test))]
        a, b = sg_baseline(s, mode)
        rr.append(a); cc.append(b)
    print(f"    {label:13s} retention {fmt(pr, *ci(rr))}  cut {fmt(pc, *ci(cc))}")

# ============================ MMLU ============================
mm = pd.read_csv("results/records.csv").to_dict("records")
rng2 = random.Random(SEED)
mi = list(range(len(mm)))
rng2.shuffle(mi)
h = len(mi) // 2
mval = [mm[i] for i in mi[:h]]
mtest = [mm[i] for i in mi[h:]]
print(f"\nMMLU: {len(mm)} questions -> validation {len(mval)}, test {len(mtest)}")


def mm_metrics(sample, t):
    acc = car = b1a = b1c = 0.0
    esc = 0
    for r in sample:
        b1a += r["large_correct"]; b1c += r["large_carbon"]
        if r["small_entropy"] > t:
            acc += r["large_correct"]; car += r["large_carbon"] + r["small_carbon"]; esc += 1
        else:
            acc += r["small_correct"]; car += r["small_carbon"]
    return acc / b1a * 100, (1 - car / b1c) * 100, esc / len(sample)


ents = sorted(r["small_entropy"] for r in mval)
bt, bc = None, -1e9
for pct in range(2, 99):
    t = ents[int(pct / 100 * (len(ents) - 1))]
    ret, cut, _ = mm_metrics(mval, t)
    if ret >= 95 and cut > bc:
        bc, bt = cut, t
if bt is None:
    bt = max(ents, key=lambda t: mm_metrics(mval, t)[0])
vr, vc, ve = mm_metrics(mval, bt)
print(f"  threshold selected on validation: {bt:.4f} (val: ret {vr:.1f}%, cut {vc:.1f}%)")
tr, tc, te = mm_metrics(mtest, bt)
rr, cc = [], []
for _ in range(R):
    s = [mtest[rng2.randrange(len(mtest))] for _ in range(len(mtest))]
    a, b, _ = mm_metrics(s, bt)
    rr.append(a); cc.append(b)
print("  HELD-OUT TEST RESULT:")
print(f"    retention   {fmt(tr, *ci(rr))}")
print(f"    energy cut  {fmt(tc, *ci(cc))}")
print(f"    escalation  {te:.0%}")

# ============================ Vision ============================
vr_ = [json.loads(l) for l in open("results/vision_records.jsonl", encoding="utf-8")]
rng3 = random.Random(SEED)
vi = list(range(len(vr_)))
rng3.shuffle(vi)
hv = len(vi) // 2
vval = [vr_[i] for i in vi[:hv]]
vtest = [vr_[i] for i in vi[hv:]]
print(f"\nVQAv2: {len(vr_)} questions -> validation {len(vval)}, test {len(vtest)}")


def v_metrics(sample, t):
    acc = b1 = 0.0
    esc = 0
    for r in sample:
        b1 += r["large_correct"]
        if r["small_conf"] < t:
            acc += r["large_correct"]; esc += 1
        else:
            acc += r["small_correct"]
    return acc / len(sample) * 100, acc / b1 * 100 if b1 else 0, esc / len(sample)


confs = sorted(r["small_conf"] for r in vval)
bt2, ba = None, -1e9
for pct in range(1, 60):
    t = confs[int(pct / 100 * (len(confs) - 1))]
    a, _, _ = v_metrics(vval, t)
    if a > ba:
        ba, bt2 = a, t
va, vrel, vesc = v_metrics(vval, bt2)
print(f"  threshold selected on validation: {bt2:.4f} (val acc {va:.1f}%, esc {vesc:.0%})")
ta, trel, tesc = v_metrics(vtest, bt2)
aa, rl = [], []
for _ in range(R):
    s = [vtest[rng3.randrange(len(vtest))] for _ in range(len(vtest))]
    a, b, _ = v_metrics(s, bt2)
    aa.append(a); rl.append(b)
print("  HELD-OUT TEST RESULT:")
print(f"    accuracy         {fmt(ta, *ci(aa))}")
print(f"    vs large tier    {fmt(trel, *ci(rl))}")
print(f"    escalation       {tesc:.0%}")
