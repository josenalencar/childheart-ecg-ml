"""Item 13 -- sensitivity analysis: median imputation vs native missing-value handling.

A co-author's question (2026-10-14): the pipeline median-imputes, but LightGBM and XGBoost
can both consume NaN directly, learning a default direction for the missing branch at every
split. Does the published discrimination depend on the imputation?

Design -- the two arms differ in ONE step and nothing else:
  'median'  the published matrix: missing values replaced by the TRAIN median before the
            age-specific z-transform (common_features.apply_agez)
  'native'  the identical matrix with the fillna removed, so NaN reaches the booster

Everything else is held fixed on purpose: the same feature columns, the same age-norms read
from the saved transform bundle (never refit), the same audited hyperparameters from
out/ceiling/ceiling_best_params.csv, the same fixed 80/20 split, the same held-out test
children and the same external ZZU records. The arms are therefore paired patient by
patient, so the internal contrast is testable with DeLong on correlated ROC curves.

Reported per outcome and per estimator (LightGBM, XGBoost):
  internal held-out AUC in both arms, the paired difference and its DeLong P
  external AUC in both arms under the manuscript's three case definitions, with the same
  paired DeLong test (the two arms score the same records, so the curves are correlated there too)
  the share of the matrix that imputation actually touches

The random forest is excluded by construction: scikit-learn's forest cannot take NaN, which
is itself part of the answer to the question.

Outputs (out/imputation_sensitivity/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
import numpy as np, pandas as pd, joblib
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_features import build_feature_matrix, clean_implausible, apply_agez
# The raw, pre-imputation internal matrix has to be rebuilt from the same extraction the
# published dataset was built from, so the vintage is read through batch_source rather than
# chosen again here. Point CHILDHEART_STUDY at the study directory (the one holding Data/ and
# code/); it defaults to the parent of this file's own directory.
STUDY = os.environ.get('CHILDHEART_STUDY', os.path.dirname(HERE))
if os.path.isdir(os.path.join(STUDY, 'code')):
    sys.path.append(os.path.join(STUDY, 'code'))
from batch_source import load_internal

OUT = os.path.join(HERE, 'out', 'imputation_sensitivity'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS = 42, 5
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]

# ---------------------------------------------------------------- fast DeLong (from 04)
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

def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort'); y = y[order]
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    m = int(y.sum()); n = len(y) - m; k = 2
    tx, ty, tz = np.empty([k, m]), np.empty([k, n]), np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = np.array([[1.0, -1.0]]); var = float(d @ S @ d.T)
    diff = float(aucs[0] - aucs[1])
    if var <= 0: return diff, 1.0, (np.nan, np.nan)
    se = np.sqrt(var)
    return diff, float(2 * stats.norm.sf(abs(diff / se))), (diff - 1.96 * se, diff + 1.96 * se)

# ------------------------------------------------- the transform, with and without fillna
bundle = joblib.load(os.path.join(HERE, 'data', 'agez_transform.joblib'))

def apply_agez_native(raw_df, age_months, bundle):
    """apply_agez with the median fill REMOVED: a missing value stays missing through the
    age-specific z-transform, so the booster receives NaN. Byte-for-byte the same norms."""
    cont, raw, FEAT = bundle['cont'], bundle['raw'], bundle['FEAT']
    centers, means, sds = bundle['centers'], bundle['means'], bundle['sds']
    gmean, gsd = bundle['gmean'], bundle['gsd']
    n = len(raw_df); ag = np.asarray(age_months, float)
    out = pd.DataFrame(index=raw_df.index)
    for j, c in enumerate(cont):
        col = pd.to_numeric(raw_df[c], errors='coerce') if c in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
        col = col.values                                          # <-- no .fillna(med[c])
        m = means[j].copy(); s = sds[j].copy(); ok = ~np.isnan(m)
        if ok.sum() >= 2:
            mm = np.interp(ag, centers[ok], m[ok]); ss = np.interp(ag, centers[ok], np.nan_to_num(s[ok], nan=gsd[j]))
        else:
            mm = np.full(n, gmean[j]); ss = np.full(n, gsd[j])
        ss = np.where(ss < 1e-6, gsd[j], ss)
        out[c] = np.clip((col - mm) / ss, -8, 8)                  # NaN propagates
    for c in raw:
        col = pd.to_numeric(raw_df[c], errors='coerce') if c in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
        out[c] = col.values                                       # <-- no .fillna(med[c])
    return out[FEAT]

# --------------------------------------------------------------- the internal cohort, raw
log('rebuilding the internal feature matrix before imputation')
published = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
published['folio'] = published['folio'].astype(str)
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV', 'age_months']
MODEL_FEAT = bundle['model_feat']
assert published['folio'].is_unique

au = load_internal(STUDY)
au['folio'] = au['filename'].astype(str).str.extract(r'(\d+)\.json')
man = pd.read_csv(os.path.join(STUDY, 'Data', '[OPEN] Crianças Qualquer doença - Pacientes.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip()
mg = pd.merge(man[['folio', 'Qualquer patologia', 'Doença AD', 'Doença AE', 'Doença VD', 'Doença VE']],
              au, on='folio')
feat = build_feature_matrix(mg); feat, _ = clean_implausible(feat)
feat['folio'] = mg['folio'].astype(str).values
# the batch carries its own age column; age_months comes from the manual sheet via the
# published dataset, exactly as build_agez_dataset.py assigned it
feat = feat.drop(columns=[c for c in ['age_months', 'age'] if c in feat.columns])
feat = feat.merge(published[['folio', 'split', 'age_months'] + [y for y, _ in OUTCOMES]],
                  on='folio', how='right')
assert 'age_months' in feat.columns
assert len(feat) == len(published), (len(feat), len(published))

Zi_native = apply_agez_native(feat, feat['age_months'].values, bundle)
Zi_native.insert(0, 'age_months', feat['age_months'].values)
Zi_native = Zi_native[MODEL_FEAT]
Zi_med = apply_agez(feat, feat['age_months'].values, bundle)
Zi_med.insert(0, 'age_months', feat['age_months'].values)
Zi_med = Zi_med[MODEL_FEAT]

# the rebuilt median arm must reproduce the published matrix, or the two arms are not paired
dmax = float(np.nanmax(np.abs(Zi_med.values - published[MODEL_FEAT].values)))
log(f'  rebuild check: max |rebuilt median arm - published matrix| = {dmax:.2e}')

split = feat['split'].values
Yi = {y: feat[y].values for y, _ in OUTCOMES}
tr_m, te_m = split == 'train', split == 'test'
nan_i = Zi_native.isna()
log(f'  internal: train {int(tr_m.sum())}, test {int(te_m.sum())}, {len(MODEL_FEAT)} features; '
    f'{100 * nan_i.values.mean():.2f}% of cells imputed in the published arm')

# --------------------------------------------------------------- the external cohort (ZZU)
log('building the external cohort')
ext = pd.read_csv(os.path.join(HERE, 'data', 'ecg_batch_analysis_14190_v4_E3.csv'), low_memory=False)
fe = build_feature_matrix(ext); fe, _ = clean_implausible(fe)
fe['_ECG_ID'] = ext['filename'].astype(str); fe['_error'] = ext['error']
lab = pd.read_excel(os.path.join(HERE, 'data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
keep = ['ECG_ID', 'is_primary_ecg', 'is_12lead', 'age_months', 'sex_M',
        'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
df = fe.merge(lab[keep].rename(columns={'sex_M': 'zzu_sex_M'}), left_on='_ECG_ID', right_on='ECG_ID', how='left')
err_ok = df['_error'].isna() | (df['_error'].astype(str).str.strip() == '') | (df['_error'].astype(str) == 'nan')
C = df[(df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1) & err_ok].copy()
C['sex_M'] = C['zzu_sex_M'].fillna(C['sex_M']).fillna(0).astype(int)

Xe_med = apply_agez(C, C['age_months'].values, bundle)
Xe_med.insert(0, 'age_months', C['age_months'].values); Xe_med = Xe_med[MODEL_FEAT]
Xe_native = apply_agez_native(C, C['age_months'].values, bundle)
Xe_native.insert(0, 'age_months', C['age_months'].values); Xe_native = Xe_native[MODEL_FEAT]

dx = [c for c in lab.columns if c.startswith('dx_')]
Cx = C.merge(lab[['ECG_ID'] + dx], on='ECG_ID', how='left')
for c in dx: Cx[c] = Cx[c].fillna(0).astype(int)
CH = [c for c in ['dx_ASD','dx_VSD','dx_AVSD','dx_ToF','dx_PDA','dx_pulm_stenosis',
                  'dx_aortic_stenosis','dx_coarctation','dx_complex_chd','dx_mitral_disease',
                  'dx_tricuspid_disease'] if c in Cx.columns]
SH = [c for c in ['dx_ASD','dx_VSD','dx_PDA'] if c in Cx.columns]
nonshunt_case = (Cx[CH].sum(1).gt(0) &
                 ~(Cx[SH].sum(1).gt(0) & Cx[[c for c in dx if c not in SH]].sum(1).eq(0))).values
free = (Cx.y_any == 0).values
MANIFESTING = {
    'y_LV': ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other', 'dx_aortic_stenosis', 'dx_coarctation'],
    'y_RV': ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension'],
    'y_RA': ['dx_tricuspid_disease', 'dx_pulm_hypertension', 'dx_complex_chd'],
    'y_LA': ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd'],
    'y_any': ['dx_complex_chd', 'dx_pulm_hypertension', 'dx_ToF'],
}
nan_e = Xe_native.isna()
log(f'  external cohort {len(C)}; disease-free {int(free.sum())}; non-shunt cases {int(nonshunt_case.sum())}; '
    f'{100 * nan_e.values.mean():.2f}% of cells imputed in the published arm')

# ----------------------------------------------------------- how much imputation touches
params = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'ceiling_best_params.csv'))
INT_PARAMS = {'n_estimators', 'min_samples_leaf', 'max_depth', 'max_leaf_nodes', 'num_leaves',
              'min_child_samples', 'subsample_freq', 'min_child_weight'}

def kwargs_for(label, estimator):
    r = params[(params.outcome == label) & (params.estimator == estimator)]
    kw = {}
    for k, v in r.iloc[0].items():
        if k in ('outcome', 'estimator') or pd.isna(v): continue
        s = str(v)
        try: kw[k] = int(s)
        except ValueError:
            try:
                f = float(s); kw[k] = int(f) if (k in INT_PARAMS and f.is_integer()) else f
            except ValueError: kw[k] = None if s == 'None' else s
    return kw

def model(estimator, label, y):
    kw = kwargs_for(label, estimator)
    if estimator == 'LightGBM':
        return lgb.LGBMClassifier(objective='binary', random_state=SEED, n_jobs=N_JOBS,
                                  verbose=-1, class_weight='balanced', **kw)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                             random_state=SEED, n_jobs=N_JOBS, scale_pos_weight=spw, **kw)

missrep = []
for cohort, M in [('internal', nan_i), ('external', nan_e)]:
    per_feat = M.mean()
    missrep.append(dict(cohort=cohort, n_records=len(M), n_features=M.shape[1],
                        pct_cells_imputed=round(100 * float(M.values.mean()), 3),
                        features_with_no_missing=int((per_feat == 0).sum()),
                        features_gt_5pct=int((per_feat > 0.05).sum()),
                        features_gt_20pct=int((per_feat > 0.20).sum()),
                        features_gt_50pct=int((per_feat > 0.50).sum()),
                        pct_records_with_any_missing=round(100 * float(M.any(axis=1).mean()), 1)))
pd.DataFrame(missrep).to_csv(os.path.join(OUT, 'missingness_profile.csv'), index=False)
top = (100 * nan_i.mean()).sort_values(ascending=False).head(30).round(2)
top.rename('pct_missing_internal').to_frame().join(
    (100 * nan_e.mean()).rename('pct_missing_external').round(2)).to_csv(
    os.path.join(OUT, 'missingness_top_features.csv'))

# ------------------------------------------------------------------------ the two arms
rows, ext_rows = [], []
for est in ['LightGBM', 'XGBoost']:
    for ycol, label in OUTCOMES:
        y = Yi[ycol]
        p_te, p_ext = {}, {}
        for arm, Xi, Xe in [('median', Zi_med, Xe_med), ('native', Zi_native, Xe_native)]:
            m = model(est, label, y[tr_m]).fit(Xi.values[tr_m], y[tr_m])
            p_te[arm] = m.predict_proba(Xi.values[te_m])[:, 1]
            p_ext[arm] = m.predict_proba(Xe.values)[:, 1]
        a_med = roc_auc_score(y[te_m], p_te['median'])
        a_nat = roc_auc_score(y[te_m], p_te['native'])
        diff, p, ci = delong_test(y[te_m], p_te['native'], p_te['median'])
        rows.append(dict(estimator=est, outcome=label, n_test=int(te_m.sum()),
                         events_test=int(y[te_m].sum()),
                         auc_median_imputed=round(a_med, 4), auc_native_nan=round(a_nat, 4),
                         diff_native_minus_median=round(diff, 4),
                         ci_low=round(ci[0], 4), ci_high=round(ci[1], 4),
                         delong_p=round(p, 4)))
        log(f'  {est:9s} {label:18s} median {a_med:.3f} | native {a_nat:.3f} | '
            f'diff {diff:+.3f} (P={p:.3f})')
        # external, the manuscript's three case definitions
        yx = Cx[ycol].values
        man_cols = [c for c in MANIFESTING[ycol] if c in Cx.columns]
        man_case = Cx[man_cols].sum(1).gt(0).values
        for defn, mask, ylab in [
                ('all ICD-10 codes', np.ones(len(Cx), bool), yx),
                ('non-shunt, CHILDHEART-like', free | nonshunt_case, None),
                ('ECG-manifesting lesions', free | man_case, None)]:
            yy = yx[mask] if ylab is not None else (man_case[mask] if defn.startswith('ECG') else nonshunt_case[mask]).astype(int)
            if len(np.unique(yy)) < 2: continue
            dx_, px_, cix_ = delong_test(yy, p_ext['native'][mask], p_ext['median'][mask])
            ext_rows.append(dict(estimator=est, outcome=label, case_definition=defn,
                                 n=int(mask.sum()), events=int(yy.sum()),
                                 auc_median_imputed=round(roc_auc_score(yy, p_ext['median'][mask]), 4),
                                 auc_native_nan=round(roc_auc_score(yy, p_ext['native'][mask]), 4),
                                 ci_low=round(cix_[0], 4), ci_high=round(cix_[1], 4),
                                 delong_p=round(px_, 4)))

R = pd.DataFrame(rows); R['delta_abs'] = R.diff_native_minus_median.abs()
R.drop(columns='delta_abs').to_csv(os.path.join(OUT, 'internal_heldout_imputation_arms.csv'), index=False)
E = pd.DataFrame(ext_rows)
E['diff_native_minus_median'] = (E.auc_native_nan - E.auc_median_imputed).round(4)
E.to_csv(os.path.join(OUT, 'external_imputation_arms.csv'), index=False)

log('summary (internal held-out)')
log(f'  largest |difference| {R.delta_abs.max():.4f}; median |difference| {R.delta_abs.median():.4f}; '
    f'significant at P<0.05: {int((R.delong_p < 0.05).sum())}/{len(R)}')
log('summary (external)')
log(f'  largest |difference| {E.diff_native_minus_median.abs().max():.4f}; '
    f'median |difference| {E.diff_native_minus_median.abs().median():.4f}; '
    f'significant at P<0.05: {int((E.delong_p < 0.05).sum())}/{len(E)}')
log(f'ALL COMPLETE -> {OUT}')
for f in sorted(os.listdir(OUT)): print('   ', f)
