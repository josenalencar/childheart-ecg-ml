"""Item 2 — DeLong tests for every AUC comparison the study makes.

Not one significance test for a model comparison exists anywhere in the CHILDHEART code
base: the only P values in the manuscript come from the Mann-Whitney tests of chamber
status. Every claim of the form "X discriminates better than Y" is currently read off
overlapping confidence intervals, which is not a test, and is anti-conservative exactly
where it matters most here -- these are correlated ROC curves computed on the same
children, so the curves covary and independent CIs overstate the uncertainty of their
difference.

Implements the fast DeLong estimator of Sun & Xu (IEEE Signal Process Lett 2014), which is
the standard used by pROC::roc.test and by the DART study.

Comparisons produced:
  A. incumbent RandomForest vs each tuned ceiling estimator, per outcome (held-out test)
  B. every pairwise comparison among all four estimators, per outcome

Outputs (out/delong/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, itertools
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CEIL = os.path.join(HERE, 'out', 'ceiling')
OUT = os.path.join(HERE, 'out', 'delong'); os.makedirs(OUT, exist_ok=True)


# ----------------------------- fast DeLong -----------------------------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float); out[J] = T
    return out


def _structural_components(preds, m, n):
    """preds: (k, m+n) with the m positives first. Returns V10 (k,m), V01 (k,n), aucs (k,)."""
    k = preds.shape[0]
    tx, ty, tz = np.empty([k, m]), np.empty([k, n]), np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    return v01, v10, aucs


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC curves on the same subjects."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort')          # positives first
    y = y[order]
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    m = int(y.sum()); n = len(y) - m
    v01, v10, aucs = _structural_components(preds, m, n)
    s01 = np.cov(v01); s10 = np.cov(v10)
    S = s01 / m + s10 / n
    d = np.array([[1.0, -1.0]])
    var = float(d @ S @ d.T)
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return diff, 1.0, (np.nan, np.nan)
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(var)
    return diff, float(p), (round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4))


# ----------------------------- run comparisons -------------------------------
pred = pd.read_csv(os.path.join(CEIL, 'test_predictions.csv'))
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]
from sklearn.metrics import roc_auc_score

rows = []
for ycol, label in OUTCOMES:
    y = pred[ycol].values
    cols = [c for c in pred.columns if c.startswith(ycol + '__')]
    names = {c: c.split('__', 1)[1].replace('_', ' ') for c in cols}
    for a, b in itertools.combinations(cols, 2):
        diff, p, ci = delong_test(y, pred[a].values, pred[b].values)
        rows.append(dict(outcome=label,
                         model_A=names[a], auc_A=round(roc_auc_score(y, pred[a]), 4),
                         model_B=names[b], auc_B=round(roc_auc_score(y, pred[b]), 4),
                         auc_difference=round(diff, 4), diff_ci_lo=ci[0], diff_ci_hi=ci[1],
                         delong_p=round(p, 4),
                         significant_at_0_05=bool(p < 0.05)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'delong_pairwise_test.csv'), index=False)

inc = res[res.model_A.str.contains('as shipped') | res.model_B.str.contains('as shipped')]
inc.to_csv(os.path.join(OUT, 'delong_vs_incumbent.csv'), index=False)

print('== item 2: DeLong, incumbent RandomForest vs the tuned ceiling ==')
print(inc[['outcome', 'model_A', 'auc_A', 'model_B', 'auc_B',
           'auc_difference', 'delong_p']].to_string(index=False))
print('\nwrote', OUT)
