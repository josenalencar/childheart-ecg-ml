"""Items 4, 6, 10 — SHAP importance, bootstrap stability of the ranking, parsimony floor.

Item 4  The manuscript's mechanistic narrative ("the QRS-T angle family dominates
        internally and collapses externally") rests on Gini importance. Gini is biased
        toward continuous and high-cardinality predictors, which describes nearly all
        1,047 features here, so the ranking that carries the argument is the one ranking
        known to be unreliable for it. SHAP is already computed in train_one.py for the
        figures but never reported. Recomputed here as the primary ranking, with
        permutation importance on the held-out set as an unbiased cross-check.

Item 6  With 1,309 training children, 279 positives and 1,047 candidate features, feature
        importance rankings are unstable by construction. The ranking is bootstrapped so
        the paper can state how often its named features actually survive resampling
        instead of asserting a single ordering.

Item 10 A parsimony floor. The study reports one model family and no comparator, so a
        reader cannot tell whether 1,047 features buy anything over a handful. A logistic
        regression on the most stable features answers that. If it approaches the forest,
        that is the most clinically useful result in the paper.

Outputs (out/importance/).
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'importance'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS, N_BOOT = 42, 5, 60

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]
RF = lambda nj=N_JOBS: RandomForestClassifier(n_estimators=600, class_weight='balanced',
                                              random_state=SEED, n_jobs=nj, min_samples_leaf=2)

# ==================== items 4: SHAP vs Gini vs permutation ====================
rank_rows = []
for ycol, label in OUTCOMES:
    m = RF().fit(tr[FEAT].values, tr[ycol].values)
    gini = pd.Series(m.feature_importances_, index=FEAT)

    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(te[FEAT].values, check_additivity=False)
    sv1 = sv[..., 1] if (hasattr(sv, 'ndim') and sv.ndim == 3) else (sv[1] if isinstance(sv, list) else sv)
    shap_imp = pd.Series(np.abs(sv1).mean(axis=0), index=FEAT)

    # Permutation importance, computed only over the union of the Gini and SHAP top 60.
    # sklearn's permutation_importance has no feature-subset argument and permuting all
    # 1,047 features 20 times is ~21,000 re-scorings per outcome, for a ranking whose tail
    # is noise either way. The question that matters is whether the features the narrative
    # NAMES survive an unbiased estimator, so the subset is computed directly.
    cand = sorted(set(gini.sort_values(ascending=False).head(60).index) |
                  set(shap_imp.sort_values(ascending=False).head(60).index))
    Xte_p = te[FEAT].values.copy()
    yte_p = te[ycol].values
    base = roc_auc_score(yte_p, m.predict_proba(Xte_p)[:, 1])
    prng = np.random.default_rng(SEED)
    perm_imp = pd.Series(np.nan, index=FEAT, dtype=float)
    for c in cand:
        j = FEAT.index(c); drops = []
        col = Xte_p[:, j].copy()
        for _ in range(10):
            Xte_p[:, j] = prng.permutation(col)
            drops.append(base - roc_auc_score(yte_p, m.predict_proba(Xte_p)[:, 1]))
        Xte_p[:, j] = col
        perm_imp[c] = float(np.mean(drops))

    r = pd.DataFrame({'gini': gini, 'shap': shap_imp, 'permutation': perm_imp})
    # permutation is NaN outside the candidate subset, so ranks stay nullable
    # method='min' gives integer ranks (plain rank() returns x.5 on ties, which cannot
    # cast to a nullable integer), and the column stays float so NaN survives outside the
    # permutation subset.
    for c in ['gini', 'shap', 'permutation']:
        r[c + '_rank'] = r[c].rank(ascending=False, method='min')
    r['outcome'] = label
    r = r.sort_values('shap', ascending=False)
    r.head(30).reset_index().rename(columns={'index': 'feature'}).to_csv(
        os.path.join(OUT, f'importance_{ycol}.csv'), index=False)
    top_g = list(gini.sort_values(ascending=False).head(10).index)
    top_s = list(shap_imp.sort_values(ascending=False).head(10).index)
    rank_rows.append(dict(outcome=label,
                          top1_gini=top_g[0], top1_shap=top_s[0],
                          overlap_top10=len(set(top_g) & set(top_s)),
                          spearman_gini_shap=round(r['gini'].corr(r['shap'], method='spearman'), 3),
                          spearman_gini_perm=round(r[['gini','permutation']].dropna()
                                                   .corr(method='spearman').iloc[0,1], 3),
                          spearman_shap_perm=round(r[['shap','permutation']].dropna()
                                                   .corr(method='spearman').iloc[0,1], 3)))
    print(f'  {label:18s} top-1 Gini={top_g[0][:28]:28s} SHAP={top_s[0][:28]:28s} overlap {rank_rows[-1]["overlap_top10"]}/10')
pd.DataFrame(rank_rows).to_csv(os.path.join(OUT, 'gini_vs_shap_summary.csv'), index=False)

# ==================== item 6: bootstrap stability of the ranking ====================
print('\n== item 6: bootstrap stability of the top-20 (100 resamples) ==')
rng = np.random.default_rng(SEED)
stab_rows = []
for ycol, label in OUTCOMES:
    Xtr, ytr = tr[FEAT].values, tr[ycol].values
    counts = pd.Series(0, index=FEAT, dtype=int)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        if len(np.unique(ytr[idx])) < 2: continue
        mb = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=SEED, n_jobs=N_JOBS, min_samples_leaf=2)
        mb.fit(Xtr[idx], ytr[idx])
        top = pd.Series(mb.feature_importances_, index=FEAT).sort_values(ascending=False).head(20).index
        counts[top] += 1
    freq = (counts / N_BOOT).sort_values(ascending=False)
    freq.head(40).rename('top20_selection_frequency').reset_index().rename(
        columns={'index': 'feature'}).assign(outcome=label).to_csv(
        os.path.join(OUT, f'stability_{ycol}.csv'), index=False)
    base_top = pd.Series(RF().fit(Xtr, ytr).feature_importances_,
                         index=FEAT).sort_values(ascending=False).head(20).index
    stab_rows.append(dict(outcome=label,
                          n_features_ever_in_top20=int((counts > 0).sum()),
                          n_features_in_top20_over_80pct=int((freq >= 0.8).sum()),
                          n_features_in_top20_over_50pct=int((freq >= 0.5).sum()),
                          median_freq_of_pointestimate_top20=round(float(freq[base_top].median()), 2)))
    print(f'  {label:18s} {stab_rows[-1]["n_features_ever_in_top20"]:4d} features ever top-20; '
          f'{stab_rows[-1]["n_features_in_top20_over_80pct"]:3d} appear in >=80% of resamples')
pd.DataFrame(stab_rows).to_csv(os.path.join(OUT, 'stability_summary.csv'), index=False)

# ==================== item 10: parsimony floor ====================
print('\n== item 10: parsimony floor (logistic regression on the most stable features) ==')
par_rows = []
for ycol, label in OUTCOMES:
    freq = pd.read_csv(os.path.join(OUT, f'stability_{ycol}.csv')).set_index('feature')['top20_selection_frequency']
    full = RF().fit(tr[FEAT].values, tr[ycol].values)
    full_auc = roc_auc_score(te[ycol].values, full.predict_proba(te[FEAT].values)[:, 1])
    for k in [5, 10, 20]:
        sel = list(freq.head(k).index)
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000, class_weight='balanced', random_state=SEED))
        cv = cross_val_predict(lr, tr[sel].values, tr[ycol].values,
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               method='predict_proba', n_jobs=N_JOBS)[:, 1]
        lr.fit(tr[sel].values, tr[ycol].values)
        a = roc_auc_score(te[ycol].values, lr.predict_proba(te[sel].values)[:, 1])
        rf_k = RF().fit(tr[sel].values, tr[ycol].values)
        a_rf = roc_auc_score(te[ycol].values, rf_k.predict_proba(te[sel].values)[:, 1])
        par_rows.append(dict(outcome=label, k=k, features=';'.join(sel),
                             LR_cv_auc=round(roc_auc_score(tr[ycol].values, cv), 3),
                             LR_test_auc=round(a, 3), RF_kfeat_test_auc=round(a_rf, 3),
                             RF_1047_test_auc=round(full_auc, 3),
                             recovery_pct=round(100 * (a - 0.5) / (full_auc - 0.5), 1)))
        print(f'  {label:18s} k={k:2d}  LR {a:.3f} | RF({k}) {a_rf:.3f} | RF(1047) {full_auc:.3f} '
              f'| LR recovers {par_rows[-1]["recovery_pct"]:.0f}%')
pd.DataFrame(par_rows).to_csv(os.path.join(OUT, 'parsimony_floor.csv'), index=False)
print('\nwrote', OUT)
