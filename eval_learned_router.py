"""Ablation D: a learned router against the intrinsic entropy signal.

The thesis routes on a calibrated intrinsic signal. The obvious alternative,
taken by RouteLLM and FrugalGPT, is to train a meta-classifier that predicts
whether the small tier will be good enough. This script tests that alternative
on the cached ShareGPT records, so the comparison costs no additional
inference and every policy is evaluated on identical data.

Three routers are compared against the entropy gate:

  prompt-only   features available BEFORE the small model runs (query text and
                surface features). Such a router can skip the small-model run
                on escalated queries, which changes the accounting.
  post-hoc      prompt features plus the small model's own signals, including
                the four entropy variants. This still pays for the small run.
  tf-idf        bag-of-words logistic regression on the query, the closest
                analogue to a lightweight learned router.

Labels come from the judge scores already collected: escalation is beneficial
when the large tier scores strictly higher than the small tier.

Run from the repository root:
    python eval_learned_router.py
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1234
FOLDS = 5
R_BOOT = 2000
PUE, CI_WORLD = 1.2, 475.0
OUT = Path("results")


def auroc(labels, scores):
    """Tie-corrected AUROC (Mann-Whitney U); ties count half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ----------------------------------------------------------------- data
recs = [json.loads(l) for l in open("results/main15_records.jsonl", encoding="utf-8")]
score = {}
for line in open("results/judgments_main15.jsonl", encoding="utf-8"):
    j = json.loads(line)
    score[(j["idx"], j["tier"])] = j["score"]
recs = [r for r in recs if (r["idx"], "small") in score and (r["idx"], "large") in score]
for r in recs:
    r["s_small"] = score[(r["idx"], "small")]
    r["s_large"] = score[(r["idx"], "large")]
    r["y"] = int(r["s_large"] > r["s_small"])          # escalation helps
print(f"judged records: {len(recs)} | escalation beneficial on "
      f"{sum(r['y'] for r in recs)} ({sum(r['y'] for r in recs) / len(recs):.1%})")

CODE = re.compile(r"```|def |class |import |SELECT |function|<[a-z]+>|\{|\}|;\s*$")


def prompt_features(r):
    q = r["query"]
    words = q.split()
    return [
        len(words),
        len(q),
        np.mean([len(w) for w in words]) if words else 0.0,
        q.count("?"),
        int(bool(CODE.search(q))),
        int(q.strip().lower().startswith(("what", "who", "when", "where", "which"))),
        int(q.strip().lower().startswith(("how", "why", "explain", "describe"))),
        int("http" in q.lower()),
        len(set(w.lower() for w in words)) / max(len(words), 1),
    ]


def posthoc_features(r):
    return prompt_features(r) + [
        r["small_entropy_raw"], r["small_entropy_cal"],
        r["small_entropy_first"], r["small_entropy_max"],
        r["small_tokens"],
        len(r["small_response"].split()),
        int("sorry" in r["small_response"].lower() or
            "cannot" in r["small_response"].lower() or
            "can't" in r["small_response"].lower()),
    ]


y = np.array([r["y"] for r in recs])
X_prompt = np.array([prompt_features(r) for r in recs], dtype=float)
X_post = np.array([posthoc_features(r) for r in recs], dtype=float)
queries = [r["query"] for r in recs]

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


def oof_scores(make_model, X, text=False):
    """Out-of-fold probabilities: every prediction comes from a model that
    never saw that query in training."""
    out = np.zeros(len(y))
    for tr, te in cv.split(np.zeros(len(y)), y):
        model = make_model()
        if text:
            model.fit([X[i] for i in tr], y[tr])
            out[te] = model.predict_proba([X[i] for i in te])[:, 1]
        else:
            model.fit(X[tr], y[tr])
            out[te] = model.predict_proba(X[te])[:, 1]
    return out


logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
gb = lambda: GradientBoostingClassifier(random_state=SEED, n_estimators=150, max_depth=2)
tfidf = lambda: make_pipeline(
    TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=1.0))

signals = {
    "entropy (raw)": np.array([r["small_entropy_raw"] for r in recs]),
    "entropy (calibrated)": np.array([r["small_entropy_cal"] for r in recs]),
    "learned: prompt-only, logistic": oof_scores(logit, X_prompt),
    "learned: prompt-only, gradient boosting": oof_scores(gb, X_prompt),
    "learned: prompt text, tf-idf": oof_scores(tfidf, queries, text=True),
    "learned: post-hoc (prompt + entropy)": oof_scores(logit, X_post),
    "learned: post-hoc, gradient boosting": oof_scores(gb, X_post),
}

rng = random.Random(SEED)
print("\nDISCRIMINATION: can the signal identify queries where escalation helps?")
print(f"  {'signal':42s} {'AUROC':>7s}  {'95% CI':>16s}")
auc_rows = {}
for name, s in signals.items():
    a = auroc(y, s)
    boot = []
    idx = list(range(len(y)))
    for _ in range(500):
        samp = [rng.randrange(len(y)) for _ in idx]
        boot.append(auroc(y[samp], s[samp]))
    boot = sorted(b for b in boot if b == b)
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    auc_rows[name] = (a, lo, hi)
    print(f"  {name:42s} {a:7.3f}  [{lo:.3f}, {hi:.3f}]")

# ----------------------------------------------------------------- routing
def metrics(sample, flags, runs_small_first):
    """Retention and carbon reduction against always using the large tier,
    under the study's full accounting convention."""
    q = c = bq = bc = 0.0
    for r, esc in zip(sample, flags):
        cs = r["small_carbon_g"]
        cl = r["large_carbon_g"]
        bq += r["s_large"]
        bc += cl
        if esc:
            q += r["s_large"]
            c += cl + (cs if runs_small_first else 0.0)
        else:
            q += r["s_small"]
            c += cs
    return q / bq * 100, (1 - c / bc) * 100


idx = list(range(len(recs)))
random.Random(SEED).shuffle(idx)
half = len(idx) // 2
val_i, test_i = idx[:half], idx[half:]
val = [recs[i] for i in val_i]
test = [recs[i] for i in test_i]
print(f"\nHELD-OUT PROTOCOL: threshold chosen on {len(val)} validation queries, "
      f"reported on {len(test)} test queries")


def at_rate(s, rate, runs_small_first):
    """Escalate the `rate` fraction of test queries the signal ranks most
    uncertain, with the cut-off taken from the validation half."""
    thr = float(np.quantile(s[val_i], 1 - rate))
    flags = s[test_i] > thr
    ret, cut = metrics(test, flags, runs_small_first)
    return ret, cut, flags.mean() * 100


RATES = [0.10, 0.20, 0.30, 0.40]
print("\nMATCHED-ESCALATION COMPARISON (held-out test half)")
print("  retention % / carbon reduction % at each escalation budget\n")
header = "  {:42s}".format("policy") + "".join(f"{int(r*100):>17d}%" for r in RATES)
print(header)
rows = []
r_small, c_small = metrics(test, np.zeros(len(test), bool), False)
print("  {:42s}".format("static small tier (no escalation)")
      + "".join(f"{r_small:8.1f} /{c_small:6.1f}" for _ in RATES))

for name, s in signals.items():
    runs_small = "prompt" not in name        # prompt-only routers skip the small run
    cells, rec = [], {"name": name, "runs_small_first": runs_small}
    for rate in RATES:
        ret, cut, esc = at_rate(s, rate, runs_small)
        cells.append(f"{ret:8.1f} /{cut:6.1f}")
        rec[f"e{int(rate*100)}"] = [round(ret, 1), round(cut, 1)]
    rows.append(rec)
    print(f"  {name:42s}" + "".join(cells))

# an oracle upper bound for reference
orac = np.array([r["y"] + rng.random() * 1e-6 for r in recs])
cells = []
for rate in RATES:
    ret, cut, esc = at_rate(orac, rate, True)
    cells.append(f"{ret:8.1f} /{cut:6.1f}")
print(f"  {'oracle (escalates only where it helps)':42s}" + "".join(cells))

OUT.mkdir(exist_ok=True)
with open(OUT / "learned_router.json", "w", encoding="utf-8") as f:
    json.dump({"auroc": {k: list(v) for k, v in auc_rows.items()}, "routing": rows}, f, indent=2)
print("\nwrote results/learned_router.json")

# ----------------------------------------------------------------- paired test
print("\nPAIRED DIFFERENCE vs the entropy gate at a 30% escalation budget")
print("  (bootstrap over test queries; a CI excluding zero means the")
print("   difference is not attributable to sampling)\n")
base = signals["entropy (raw)"]
tb = float(np.quantile(base[val_i], 0.70))
base_flags = base[test_i] > tb
for name in ("learned: prompt-only, logistic", "learned: post-hoc (prompt + entropy)",
             "entropy (calibrated)"):
    s = signals[name]
    thr = float(np.quantile(s[val_i], 0.70))
    flags = s[test_i] > thr
    runs_small = "prompt" not in name
    dr, dc = [], []
    for _ in range(R_BOOT):
        samp = [rng.randrange(len(test)) for _ in test]
        sub = [test[i] for i in samp]
        r1, c1 = metrics(sub, flags[samp], runs_small)
        r0, c0 = metrics(sub, base_flags[samp], True)
        dr.append(r1 - r0)
        dc.append(c1 - c0)
    dr.sort(); dc.sort()
    lo_r, hi_r = dr[int(.025 * R_BOOT)], dr[int(.975 * R_BOOT)]
    lo_c, hi_c = dc[int(.025 * R_BOOT)], dc[int(.975 * R_BOOT)]
    print(f"  {name:38s} retention {np.mean(dr):+5.1f} [{lo_r:+.1f}, {hi_r:+.1f}]"
          f"   carbon {np.mean(dc):+5.1f} [{lo_c:+.1f}, {hi_c:+.1f}]")
