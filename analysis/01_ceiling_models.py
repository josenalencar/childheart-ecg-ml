"""Item 1 — the machine-learning ceiling, tuned, with nested cross-validation.

The manuscript's thesis is that the internal-to-external gap is the surrogate LABEL and
not the model. That claim is only testable against a ceiling: if a stronger estimator on
the same features reaches materially higher held-out discrimination, the limitation is at
least partly the model. Until now this study has only ever fitted RandomForest (and
LogisticRegression in side scripts) with hyperparameters that were typed once and never
moved — no gradient boosting of any kind, and no search.

Three estimators on the identical split, seed and feature matrix:
  * RandomForest   -- the incumbent, both as-shipped and tuned
  * XGBoost        -- the estimator that beat RF by 0.029 in the sibling DART study
  * LightGBM       -- second GBDT family, guards against an xgboost-specific artefact

Tuning is by RandomizedSearchCV inside an OUTER 5-fold loop (nested CV), so the reported
cross-validated AUC carries no tuning optimism. The final model per estimator is then
refitted on all of UKB-DERIV... i.e. the training partition, with the search repeated once,
and scored on the untouched held-out test set.

Outputs (results/ceiling/):
  ceiling_nested_cv.csv   nested-CV AUC per estimator and outcome (tuning-optimism-free)
  ceiling_test.csv        held-out test AUC per estimator and outcome, with bootstrap CI
  ceiling_best_params.csv the hyperparameters chosen on the training partition
  test_predictions.csv    per-patient held-out probabilities, for the DeLong step (item 2)
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform
import xgboost as xgb, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'ceiling'); os.makedirs(OUT, exist_ok=True)
SEED = 42
N_JOBS = 5          # half the cores: heavy jobs must not lock the machine

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
print(f'features {len(FEAT)} | train {len(tr)} | test {len(te)}')

OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]

def spw(y):                      # scale_pos_weight, as DART used it
    return float((y == 0).sum() / max((y == 1).sum(), 1))

def estimators(y):
    """(name, estimator, search space). Grids are deliberately modest: 1,309 children and
    279 positives will not support a fine search, and a wide grid would itself overfit."""
    return [
        # max_features is capped well below 0.1: with 1,047 features, 0.3 means evaluating
        # 314 candidate splits per node and the search becomes the bottleneck without
        # buying accuracy. sqrt(1047) ~ 32 is the conventional choice and is what the
        # incumbent already uses.
        ('RandomForest', RandomForestClassifier(class_weight='balanced', random_state=SEED,
                                                n_jobs=1),
         {'n_estimators': randint(200, 700), 'min_samples_leaf': randint(1, 8),
          'max_features': ['sqrt', 'log2', 0.05], 'max_depth': [None, 12, 24]}),
        ('XGBoost', xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                      tree_method='hist', random_state=SEED, n_jobs=1,
                                      scale_pos_weight=spw(y)),
         {'n_estimators': randint(200, 800), 'max_depth': randint(2, 7),
          'learning_rate': uniform(0.01, 0.15), 'subsample': uniform(0.6, 0.4),
          'colsample_bytree': uniform(0.3, 0.7), 'min_child_weight': randint(1, 10),
          'reg_lambda': uniform(0.5, 4.0)}),
        ('LightGBM', lgb.LGBMClassifier(objective='binary', random_state=SEED, n_jobs=1,
                                        verbose=-1, class_weight='balanced'),
         {'n_estimators': randint(200, 800), 'num_leaves': randint(7, 63),
          'learning_rate': uniform(0.01, 0.15), 'subsample': uniform(0.6, 0.4),
          'subsample_freq': [1], 'colsample_bytree': uniform(0.3, 0.7),
          'min_child_samples': randint(5, 40), 'reg_lambda': uniform(0.5, 4.0)}),
    ]

rng = np.random.default_rng(SEED)
def boot_ci(y, p, n=1000):
    y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1: v.append(roc_auc_score(y[i], p[i]))
    return round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)

nested_rows, test_rows, param_rows = [], [], []
preds = te[['folio'] + [y for y, _ in OUTCOMES]].copy().reset_index(drop=True)

for ycol, label in OUTCOMES:
    ytr, yte = tr[ycol].values, te[ycol].values
    Xtr, Xte = tr[FEAT].values, te[FEAT].values
    print(f'\n=== {label}  (train pos {ytr.sum()} / test pos {yte.sum()}) ===')

    # ---- as-shipped incumbent, no tuning: the model currently in the manuscript -------
    inc = RandomForestClassifier(n_estimators=600, class_weight='balanced',
                                 random_state=SEED, n_jobs=N_JOBS, min_samples_leaf=2)
    oof = cross_val_predict(inc, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method='predict_proba', n_jobs=1)[:, 1]
    inc.fit(Xtr, ytr); p_inc = inc.predict_proba(Xte)[:, 1]
    a = roc_auc_score(yte, p_inc)
    nested_rows.append(dict(outcome=label, label=ycol, estimator='RandomForest (as shipped)',
                            nested_cv_auc=round(roc_auc_score(ytr, oof), 4), tuned=False))
    lo, hi = boot_ci(yte, p_inc)
    test_rows.append(dict(outcome=label, label=ycol, estimator='RandomForest (as shipped)',
                          test_auc=round(a, 4), ci_lo=lo, ci_hi=hi, tuned=False))
    preds[f'{ycol}__RandomForest_as_shipped'] = p_inc
    print(f'  RandomForest (as shipped)  nestedCV {roc_auc_score(ytr,oof):.4f}  TEST {a:.4f}')

    # ---- tuned estimators, nested CV --------------------------------------------------
    for name, est, grid in estimators(ytr):
        t0 = time.time()
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
        # Parallelism sits on the OUTER loop: joblib does not nest here, so a search with
        # n_jobs=5 inside a serial cross_val_predict ran on a single core. Five outer folds
        # and n_jobs=5 map one fold per worker, which is the requested half of the machine.
        search = RandomizedSearchCV(est, grid, n_iter=15, scoring='roc_auc', cv=inner,
                                    random_state=SEED, n_jobs=1, refit=True)
        nested = cross_val_predict(search, Xtr, ytr, cv=outer, method='predict_proba',
                                   n_jobs=N_JOBS)[:, 1]
        nested_auc = roc_auc_score(ytr, nested)
        # final model: one search on the whole training partition, scored on held-out test
        search.n_jobs = N_JOBS
        search.fit(Xtr, ytr)
        p = search.best_estimator_.predict_proba(Xte)[:, 1]
        a = roc_auc_score(yte, p); lo, hi = boot_ci(yte, p)
        nested_rows.append(dict(outcome=label, label=ycol, estimator=name,
                                nested_cv_auc=round(nested_auc, 4), tuned=True))
        test_rows.append(dict(outcome=label, label=ycol, estimator=name,
                              test_auc=round(a, 4), ci_lo=lo, ci_hi=hi, tuned=True))
        param_rows.append(dict(outcome=label, estimator=name,
                               **{k: str(v) for k, v in search.best_params_.items()}))
        preds[f'{ycol}__{name}'] = p
        print(f'  {name:12s} nestedCV {nested_auc:.4f}  TEST {a:.4f} ({lo}-{hi})  [{time.time()-t0:.0f}s]')

pd.DataFrame(nested_rows).to_csv(os.path.join(OUT, 'ceiling_nested_cv.csv'), index=False)
pd.DataFrame(test_rows).to_csv(os.path.join(OUT, 'ceiling_test.csv'), index=False)
pd.DataFrame(param_rows).to_csv(os.path.join(OUT, 'ceiling_best_params.csv'), index=False)
preds.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
print('\nwrote', OUT)
