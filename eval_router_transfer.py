"""Does a learned router need data from the deployment it runs in?

Ablation D showed a classifier trained on a workload can outrank the
intrinsic entropy signal on that same workload. Two questions follow, and
they decide whether such a router could be shipped with the library rather
than fitted per deployment:

  transfer      does a router trained on one workload work on another, or
                does it only learn the failure modes of one model pair?
  data scale    does accuracy improve with more training examples, or is
                the task itself the limit?

Both are answered offline from records already collected. MMLU labels are
free because ground truth exists; ShareGPT labels come from the judge
scores already gathered.

Run from the repository root:
    python eval_router_transfer.py
"""
import csv
import json
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1234
rng = random.Random(SEED)
CODE = re.compile(r"```|def |class |import |SELECT |function|<[a-z]+>|\{|\}|;\s*$")


def auroc(labels, scores):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n:
                w += 1
            elif p == n:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def feats(q):
    words = q.split()
    return [len(words), len(q),
            np.mean([len(w) for w in words]) if words else 0.0,
            q.count("?"), int(bool(CODE.search(q))),
            int(q.strip().lower().startswith(("what", "who", "when", "where", "which"))),
            int(q.strip().lower().startswith(("how", "why", "explain", "describe"))),
            int("http" in q.lower()),
            len(set(w.lower() for w in words)) / max(len(words), 1)]


# ---------------------------------------------------------------- ShareGPT
recs = [json.loads(l) for l in open("results/main15_records.jsonl", encoding="utf-8")]
sc = {}
for line in open("results/judgments_main15.jsonl", encoding="utf-8"):
    j = json.loads(line)
    sc[(j["idx"], j["tier"])] = j["score"]
recs = [r for r in recs if (r["idx"], "small") in sc and (r["idx"], "large") in sc]
sg_q = [r["query"] for r in recs]
sg_y = np.array([int(sc[(r["idx"], "large")] > sc[(r["idx"], "small")]) for r in recs])
sg_ent = np.array([r["small_entropy_raw"] for r in recs])

# ---------------------------------------------------------------- MMLU
from greengate.mmlu import load_mmlu                                   # noqa: E402
qs = load_mmlu(300, seed=42)
rows = list(csv.DictReader(open("results/records.csv", encoding="utf-8")))
assert all(q.subject == r["subject"] for q, r in zip(qs, rows)), "record alignment failed"
mm_q = [q.question for q in qs]
mm_y = np.array([int(r["small_correct"] == "0" and r["large_correct"] == "1") for r in rows])
mm_ent = np.array([float(r["small_entropy"]) for r in rows])

print(f"ShareGPT: {len(sg_y)} queries, escalation helps on {sg_y.mean():.1%}")
print(f"MMLU    : {len(mm_y)} questions, escalation helps on {mm_y.mean():.1%}")

Xsg = np.array([feats(q) for q in sg_q], dtype=float)
Xmm = np.array([feats(q) for q in mm_q], dtype=float)

logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
tfidf = lambda: make_pipeline(
    TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000))


def cv_auroc(X, y, make, text=False, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    out = np.zeros(len(y))
    for tr, te in cv.split(np.zeros(len(y)), y):
        m = make()
        if text:
            m.fit([X[i] for i in tr], y[tr])
            out[te] = m.predict_proba([X[i] for i in te])[:, 1]
        else:
            m.fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1]
    return auroc(y, out)


def transfer(Xtr, ytr, Xte, yte, make, text=False):
    m = make()
    m.fit(Xtr if not text else list(Xtr), ytr)
    p = m.predict_proba(Xte if not text else list(Xte))[:, 1]
    return auroc(yte, p)


print("\n" + "=" * 70)
print("  1. IN-DOMAIN: trained and tested on the same workload")
print("=" * 70)
print(f"  {'':34s}{'ShareGPT':>12s}{'MMLU':>12s}")
print(f"  {'entropy (no training)':34s}{auroc(sg_y, sg_ent):12.3f}{auroc(mm_y, mm_ent):12.3f}")
print(f"  {'classifier, prompt features':34s}"
      f"{cv_auroc(Xsg, sg_y, logit):12.3f}{cv_auroc(Xmm, mm_y, logit):12.3f}")
print(f"  {'classifier, query text (tf-idf)':34s}"
      f"{cv_auroc(sg_q, sg_y, tfidf, text=True):12.3f}"
      f"{cv_auroc(mm_q, mm_y, tfidf, text=True):12.3f}")

print("\n" + "=" * 70)
print("  2. TRANSFER: trained on one workload, tested on the other")
print("=" * 70)
print(f"  {'MMLU -> ShareGPT, prompt features':44s}"
      f"{transfer(Xmm, mm_y, Xsg, sg_y, logit):8.3f}")
print(f"  {'MMLU -> ShareGPT, query text':44s}"
      f"{transfer(mm_q, mm_y, sg_q, sg_y, tfidf, text=True):8.3f}")
print(f"  {'ShareGPT -> MMLU, prompt features':44s}"
      f"{transfer(Xsg, sg_y, Xmm, mm_y, logit):8.3f}")
print(f"  {'ShareGPT -> MMLU, query text':44s}"
      f"{transfer(sg_q, sg_y, mm_q, mm_y, tfidf, text=True):8.3f}")
print("\n  0.500 is chance. A shipped router is viable only if these hold up.")

print("\n" + "=" * 70)
print("  3. DOES MORE TRAINING DATA HELP?  (ShareGPT, held-out AUROC)")
print("=" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"  {'training examples':>18s}{'prompt features':>18s}{'query text':>14s}")
for size in (50, 100, 150, 200, 250, 300, 350, 400):
    a_p, a_t = [], []
    for rep in range(3):
        r2 = random.Random(SEED + rep)
        op = np.zeros(len(sg_y)); ot = np.zeros(len(sg_y))
        for tr, te in cv.split(np.zeros(len(sg_y)), sg_y):
            tr = list(tr)
            r2.shuffle(tr)
            sub = tr[:min(size, len(tr))]
            if len(set(sg_y[sub])) < 2:
                continue
            m = logit(); m.fit(Xsg[sub], sg_y[sub]); op[te] = m.predict_proba(Xsg[te])[:, 1]
            m = tfidf(); m.fit([sg_q[i] for i in sub], sg_y[sub])
            ot[te] = m.predict_proba([sg_q[i] for i in te])[:, 1]
        a_p.append(auroc(sg_y, op)); a_t.append(auroc(sg_y, ot))
    print(f"  {size:>18d}{np.mean(a_p):>18.3f}{np.mean(a_t):>14.3f}")
print("\n  A flat curve means the limit is the task, not the amount of data.")
