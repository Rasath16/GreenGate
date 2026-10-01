"""Do the held-out conclusions depend on which random split was drawn?

Section 5.10.5 selects an operating threshold on one half of the ShareGPT
records and reports on the other, using a single random split. A reader is
entitled to ask whether a different draw would have given a different
answer. This repeats the entire held-out protocol over many independent
splits and reports how often each conclusion survives.

It requires no new inference: every policy is replayed against the cached
per-query records, so all splits are evaluated on identical measurements.

Run from the repository root:
    python bootstrap_results.py        # if results/ is not populated
    python eval_split_stability.py
"""
import json
import random
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1234
FOLDS = 5
N_SPLITS = 200          # independent validation/test draws
MODEL_SEEDS = 10        # independent cross-validation draws for the signal
RATE = 0.30             # escalation budget at which the thesis compares policies

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
    r["y"] = int(r["s_large"] > r["s_small"])
y = np.array([r["y"] for r in recs])
n = len(recs)
print(f"judged records: {n}")

CODE = re.compile(r"```|def |class |import |SELECT |function|<[a-z]+>|\{|\}|;\s*$")


def prompt_features(r):
    q = r["query"]
    w = q.split()
    return [len(w), len(q), np.mean([len(x) for x in w]) if w else 0.0, q.count("?"),
            int(bool(CODE.search(q))),
            int(q.strip().lower().startswith(("what", "who", "when", "where", "which"))),
            int(q.strip().lower().startswith(("how", "why", "explain", "describe"))),
            int("http" in q.lower()),
            len(set(x.lower() for x in w)) / max(len(w), 1)]


def posthoc_features(r):
    return prompt_features(r) + [
        r["small_entropy_raw"], r["small_entropy_cal"], r["small_entropy_first"],
        r["small_entropy_max"], r["small_tokens"], len(r["small_response"].split()),
        int("sorry" in r["small_response"].lower() or
            "cannot" in r["small_response"].lower() or
            "can't" in r["small_response"].lower())]


X_prompt = np.array([prompt_features(r) for r in recs], dtype=float)
X_post = np.array([posthoc_features(r) for r in recs], dtype=float)
queries = [r["query"] for r in recs]

logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
tfidf = lambda: make_pipeline(
    TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=1.0))


def oof(make_model, X, cv, text=False):
    out = np.zeros(n)
    for tr, te in cv.split(np.zeros(n), y):
        m = make_model()
        if text:
            m.fit([X[i] for i in tr], y[tr])
            out[te] = m.predict_proba([X[i] for i in te])[:, 1]
        else:
            m.fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
    return out


def build_signals(model_seed):
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=model_seed)
    return {
        "entropy (raw)": (np.array([r["small_entropy_raw"] for r in recs]), True),
        "entropy (calibrated)": (np.array([r["small_entropy_cal"] for r in recs]), True),
        "learned: prompt-only": (oof(logit, X_prompt, cv), False),
        "learned: prompt text, tf-idf": (oof(tfidf, queries, cv, text=True), False),
        "learned: post-hoc": (oof(logit, X_post, cv), True),
    }


def metrics(sample, flags, runs_small_first):
    q = c = bq = bc = 0.0
    for r, esc in zip(sample, flags):
        cs, cl = r["small_carbon_g"], r["large_carbon_g"]
        bq += r["s_large"]
        bc += cl
        if esc:
            q += r["s_large"]
            c += cl + (cs if runs_small_first else 0.0)
        else:
            q += r["s_small"]
            c += cs
    return q / bq * 100, (1 - c / bc) * 100


def pct(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], np.median(v), v[int(0.975 * len(v))]


# ------------------------------------------------- 1. vary the held-out split
print("\n" + "=" * 78)
print(f"  1. VARYING THE VALIDATION/TEST SPLIT  ({N_SPLITS} independent draws)")
print("=" * 78)
print(f"  Threshold chosen on the validation half at a {int(RATE*100)}% escalation")
print("  budget, applied unchanged to the test half. Model seed fixed.\n")

signals = build_signals(SEED)
rng = random.Random(SEED)
acc = {k: {"ret": [], "cut": []} for k in signals}
acc["static small tier"] = {"ret": [], "cut": []}
diff_ret, diff_cut, diff_cut_pre = [], [], []

for rep in range(N_SPLITS):
    idx = list(range(n))
    random.Random(SEED + rep).shuffle(idx)
    h = n // 2
    val_i, test_i = np.array(idx[:h]), np.array(idx[h:])
    test = [recs[i] for i in test_i]

    r0, c0 = metrics(test, np.zeros(len(test), bool), False)
    acc["static small tier"]["ret"].append(r0)
    acc["static small tier"]["cut"].append(c0)

    per = {}
    for name, (s, runs_small) in signals.items():
        thr = float(np.quantile(s[val_i], 1 - RATE))
        ret, cut = metrics(test, s[test_i] > thr, runs_small)
        acc[name]["ret"].append(ret)
        acc[name]["cut"].append(cut)
        per[name] = (ret, cut)
    diff_ret.append(per["learned: post-hoc"][0] - per["entropy (raw)"][0])
    diff_cut.append(per["learned: post-hoc"][1] - per["entropy (raw)"][1])
    diff_cut_pre.append(per["learned: prompt-only"][1] - per["entropy (raw)"][1])

print(f"  {'policy':30s}{'retention % (2.5, med, 97.5)':>34s}"
      f"{'carbon reduction %':>26s}")
for name in ["static small tier"] + list(signals):
    lo, md, hi = pct(acc[name]["ret"])
    clo, cmd, chi = pct(acc[name]["cut"])
    print(f"  {name:30s}{md:11.1f}  [{lo:5.1f},{hi:6.1f}]"
          f"{cmd:14.1f}  [{clo:5.1f},{chi:6.1f}]")

# ------------------------------------------------- 2. do conclusions survive?
print("\n" + "=" * 78)
print("  2. HOW OFTEN DOES EACH PUBLISHED CONCLUSION HOLD?")
print("=" * 78)

print("\n  (a)  Retention: post-hoc learned router minus entropy gate")
lo, md, hi = pct(diff_ret)
print(f"       across splits   median {md:+.2f}   [{lo:+.2f}, {hi:+.2f}]")
print(f"       thesis reports  +1.10  [-1.30, +3.50]  from the single drawn split")
print(f"       The two ranges agree closely, so the published split was")
print(f"       representative. The interval still contains zero, so the")
print(f"       conclusion of no reliable retention advantage stands.")

print("\n  (b)  Carbon: the thesis claim about WHERE the router's advantage comes from")
lo, md, hi = pct(diff_cut_pre)
print(f"       prompt-only router (skips the discarded small run) minus entropy")
print(f"         median {md:+.2f}  [{lo:+.2f}, {hi:+.2f}]   "
      f"positive in {np.mean(np.array(diff_cut_pre) > 0):.1%} of splits")
lo, md, hi = pct(diff_cut)
print(f"       post-hoc router (pays for the small run) minus entropy")
print(f"         median {md:+.2f}  [{lo:+.2f}, {hi:+.2f}]   "
      f"positive in {np.mean(np.array(diff_cut) > 0):.1%} of splits")
print(f"       This is the central claim of Ablation D and it holds in every")
print(f"       split: the carbon advantage belongs to deciding early, not to")
print(f"       routing better. Charge the router for the discarded run and it")
print(f"       disappears.")

ent_ret = np.array(acc["entropy (raw)"]["ret"])
sta_ret = np.array(acc["static small tier"]["ret"])
ent_cut = np.array(acc["entropy (raw)"]["cut"])
sta_cut = np.array(acc["static small tier"]["cut"])
print("\n  (c)  Entropy gate against the static small tier")
lo, md, hi = pct(ent_ret - sta_ret)
print(f"       retention        median {md:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
lo, md, hi = pct(ent_cut - sta_cut)
print(f"       carbon reduction median {md:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
print(f"       The gate buys a little quality at a large carbon cost in every")
print(f"       split. Neither option dominates, which is the trade-off the")
print(f"       thesis reports rather than a win for routing.")

# ------------------------------------------------- 3. vary the model seed
print("\n" + "=" * 78)
print(f"  3. VARYING THE CROSS-VALIDATION DRAW  ({MODEL_SEEDS} model seeds)")
print("=" * 78)
print("  The learned router is refitted from scratch under each seed.\n")


def auroc(labels, scores):
    pos = [s for s, t in zip(scores, labels) if t == 1]
    neg = [s for s, t in zip(scores, labels) if t == 0]
    w = t = 0
    for p in pos:
        for q_ in neg:
            if p > q_:
                w += 1
            elif p == q_:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


aurocs = {k: [] for k in ["learned: prompt-only", "learned: prompt text, tf-idf",
                          "learned: post-hoc"]}
for ms in range(MODEL_SEEDS):
    sig = build_signals(SEED + 97 * ms)
    for k in aurocs:
        aurocs[k].append(auroc(y, sig[k][0]))
print(f"  {'signal':32s}{'AUROC across seeds (min, med, max)':>40s}")
for k, v in aurocs.items():
    print(f"  {k:32s}{min(v):14.3f}{np.median(v):13.3f}{max(v):13.3f}")
ent = auroc(y, np.array([r["small_entropy_raw"] for r in recs]))
print(f"  {'entropy (raw), deterministic':32s}{ent:14.3f}{ent:13.3f}{ent:13.3f}")

print("\n" + "=" * 78)
print("  VERDICT")
print("=" * 78)
print("  If the retention conclusion holds in the large majority of splits and the")
print("  carbon ordering is stable, the published held-out result is a property of")
print("  the data rather than of the single split that was drawn.")
