"""Bootstrap confidence intervals for MMLU and vision results (run from repo root)."""
import json
import random
import pandas as pd

R = 2000
rng = random.Random(0)


def ci(vals, lo=0.025, hi=0.975):
    v = sorted(vals)
    return v[int(lo * len(v))], v[int(hi * len(v))]


# ---------------- MMLU ----------------
mm = pd.read_csv("results/records.csv").to_dict("records")
n = len(mm)


def mmlu_metrics(sample, thresh):
    acc = car = b1a = b1c = 0.0
    esc = 0
    for r in sample:
        b1a += r["large_correct"]; b1c += r["large_carbon"]
        if r["small_entropy"] > thresh:
            acc += r["large_correct"]; car += r["large_carbon"] + r["small_carbon"]; esc += 1
        else:
            acc += r["small_correct"]; car += r["small_carbon"]
    return acc / b1a * 100, (1 - car / b1c) * 100, esc / len(sample)


ent = sorted(r["small_entropy"] for r in mm)
t_star = ent[int(0.55 * (n - 1))]          # ~45% escalation, the reported operating point
pt_ret, pt_cut, pt_esc = mmlu_metrics(mm, t_star)
rets, cuts = [], []
for _ in range(R):
    s = [mm[rng.randrange(n)] for _ in range(n)]
    a, c, _ = mmlu_metrics(s, t_star)
    rets.append(a); cuts.append(c)
print("MMLU (n=%d, threshold=%.3f, escalation %.0f%%)" % (n, t_star, pt_esc * 100))
print("  retention  %.1f%%  95%% CI [%.1f, %.1f]" % (pt_ret, *ci(rets)))
print("  energy cut %.1f%%  95%% CI [%.1f, %.1f]" % (pt_cut, *ci(cuts)))

# entropy gap significance
gap_pt = (pd.DataFrame(mm).query("small_correct == 0").small_entropy.mean()
          - pd.DataFrame(mm).query("small_correct == 1").small_entropy.mean())
gaps = []
for _ in range(R):
    s = pd.DataFrame([mm[rng.randrange(n)] for _ in range(n)])
    w = s[s.small_correct == 0].small_entropy
    c = s[s.small_correct == 1].small_entropy
    if len(w) and len(c):
        gaps.append(w.mean() - c.mean())
lo, hi = ci(gaps)
print("  entropy gap %+.3f bits  95%% CI [%+.3f, %+.3f]  %s"
      % (gap_pt, lo, hi, "excludes zero" if lo > 0 else "includes zero"))

# ---------------- Vision ----------------
vr = [json.loads(l) for l in open("results/vision_records.jsonl", encoding="utf-8")]
nv = len(vr)
confs = sorted(r["small_conf"] for r in vr)
c_star = confs[int(0.05 * (nv - 1))]       # escalate least-confident 5%


def vis_metrics(sample, thresh):
    acc = b1 = 0.0
    esc = 0
    for r in sample:
        b1 += r["large_correct"]
        if r["small_conf"] < thresh:
            acc += r["large_correct"]; esc += 1
        else:
            acc += r["small_correct"]
    return acc / len(sample) * 100, acc / b1 * 100 if b1 else 0, esc / len(sample)


pa, pr, pe = vis_metrics(vr, c_star)
accs, rets_v = [], []
for _ in range(R):
    s = [vr[rng.randrange(nv)] for _ in range(nv)]
    a, r_, _ = vis_metrics(s, c_star)
    accs.append(a); rets_v.append(r_)
print("\nVQAv2 (n=%d, confidence threshold=%.3f, escalation %.0f%%)" % (nv, c_star, pe * 100))
print("  accuracy   %.1f%%  95%% CI [%.1f, %.1f]" % (pa, *ci(accs)))
print("  vs large   %.1f%%  95%% CI [%.1f, %.1f]" % (pr, *ci(rets_v)))

sm = sum(r["small_correct"] for r in vr) / nv * 100
lg = sum(r["large_correct"] for r in vr) / nv * 100
diffs = []
for _ in range(R):
    s = [vr[rng.randrange(nv)] for _ in range(nv)]
    diffs.append((sum(r["small_correct"] for r in s) - sum(r["large_correct"] for r in s)) / len(s) * 100)
lo, hi = ci(diffs)
print("  small minus large: %+.1f pts  95%% CI [%+.1f, %+.1f]  %s"
      % (sm - lg, lo, hi, "excludes zero" if lo > 0 else "includes zero"))

# confidence separation
cor = [r["small_conf"] for r in vr if r["small_correct"]]
wro = [r["small_conf"] for r in vr if not r["small_correct"]]
sep = sum(cor) / len(cor) - sum(wro) / len(wro)
seps = []
for _ in range(R):
    s = [vr[rng.randrange(nv)] for _ in range(nv)]
    c = [r["small_conf"] for r in s if r["small_correct"]]
    w = [r["small_conf"] for r in s if not r["small_correct"]]
    if c and w:
        seps.append(sum(c) / len(c) - sum(w) / len(w))
lo, hi = ci(seps)
print("  confidence separation %+.3f  95%% CI [%+.3f, %+.3f]  %s"
      % (sep, lo, hi, "excludes zero" if lo > 0 else "includes zero"))
