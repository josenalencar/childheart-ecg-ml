"""Item 1, extended — does the ceiling's internal advantage transport?

The ceiling run answers the question that was never asked: gradient boosting beats the
incumbent random forest internally. That alone weakens the manuscript's thesis, because
"the gap is the label, not the model" is harder to sustain once a better model exists.

But it also creates the sharper test. If XGBoost's internal advantage over RF DISAPPEARS
in the external cohort, that is a second natural experiment of exactly the kind the E2->E3
extractor comparison already provides: a change that buys real discrimination against the
adjudicated reference standard and buys nothing against ICD codes. That would strengthen
the label thesis rather than weaken it, and on evidence that does not depend on any
subgroup restriction.

If instead the advantage transports, the honest conclusion is that part of the external gap
was the model after all, and the manuscript has to say so.

Each estimator is fitted on the internal training partition with the hyperparameters chosen
in 01_ceiling_models.py, then applied unchanged to the external cohort exactly as
run_external.py does: shared feature builder, shared implausible-value cleaning, saved
age-z norms applied without re-estimation, one primary 12-lead error-free ECG per patient.

Outputs (out/ceiling_external/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_features import build_feature_matrix, clean_implausible, apply_agez

OUT = os.path.join(HERE, 'out', 'ceiling_external'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS = 42, 5

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
bundle = joblib.load(os.path.join(HERE, 'data', 'agez_transform.joblib'))
params = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'ceiling_best_params.csv'))

# ---------------- external cohort, built exactly as run_external.py builds it -------------
print('building external cohort ...')
ext = pd.read_csv(os.path.join(HERE, 'data', 'ecg_batch_analysis_14190_v4_E3.csv'), low_memory=False)
feat = build_feature_matrix(ext)
feat, _ = clean_implausible(feat)
feat['_ECG_ID'] = ext['filename'].astype(str)
feat['_error'] = ext['error']
lab = pd.read_excel(os.path.join(HERE, 'data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
keep = ['ECG_ID', 'is_primary_ecg', 'is_12lead', 'age_months', 'sex_M',
        'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
df = feat.merge(lab[keep].rename(columns={'sex_M': 'zzu_sex_M'}),
                left_on='_ECG_ID', right_on='ECG_ID', how='left')
err_ok = df['_error'].isna() | (df['_error'].astype(str).str.strip() == '') | (df['_error'].astype(str) == 'nan')
C = df[(df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1) & err_ok].copy()
C['sex_M'] = C['zzu_sex_M'].fillna(C['sex_M']).fillna(0).astype(int)
X = apply_agez(C, C['age_months'].values, bundle)
X.insert(0, 'age_months', C['age_months'].values)
X = X[bundle['model_feat']]
print(f'external cohort: {len(C)}')

# non-shunt CHILDHEART-like definition, as used in the manuscript
dx = [c for c in lab.columns if c.startswith('dx_')]
Cx = C.merge(lab[['ECG_ID'] + dx], on='ECG_ID', how='left')
for c in dx: Cx[c] = Cx[c].fillna(0).astype(int)
CH = [c for c in ['dx_ASD','dx_VSD','dx_AVSD','dx_ToF','dx_PDA','dx_pulm_stenosis',
                  'dx_aortic_stenosis','dx_coarctation','dx_complex_chd','dx_mitral_disease',
                  'dx_tricuspid_disease'] if c in Cx.columns]
SH = [c for c in ['dx_ASD','dx_VSD','dx_PDA'] if c in Cx.columns]
has_ch = Cx[CH].sum(1).gt(0)
shunt_only = Cx[SH].sum(1).gt(0) & Cx[[c for c in dx if c not in SH]].sum(1).eq(0)
nonshunt_case = (has_ch & ~shunt_only).values
free = (Cx.y_any == 0).values

rng = np.random.default_rng(SEED)
def boot_ci(y, p, n=1000):
    y=np.asarray(y); p=np.asarray(p); v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))>1: v.append(roc_auc_score(y[i],p[i]))
    return round(float(np.percentile(v,2.5)),4), round(float(np.percentile(v,97.5)),4)

def build(name, outcome, ytr):
    row = params[(params.outcome == outcome) & (params.estimator == name)]
    kw = {}
    if len(row):
        for k, v in row.iloc[0].items():
            if k in ('outcome', 'estimator') or pd.isna(v): continue
            sv = str(v)
            # pandas widens a column to float as soon as any row is NaN, so integer
            # hyperparameters come back as '12.0' and sklearn rejects them by type.
            INT_PARAMS = {'n_estimators', 'min_samples_leaf', 'max_depth', 'max_leaf_nodes',
                          'num_leaves', 'min_child_samples', 'subsample_freq'}
            try:
                kw[k] = int(sv)
            except ValueError:
                try:
                    f = float(sv)
                    kw[k] = int(f) if (k in INT_PARAMS and f.is_integer()) else f
                except ValueError:
                    kw[k] = None if sv == 'None' else sv
    if name == 'RandomForest':
        return RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=N_JOBS, **kw)
    if name == 'XGBoost':
        return xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                                 random_state=SEED, n_jobs=N_JOBS,
                                 scale_pos_weight=float((ytr==0).sum()/max((ytr==1).sum(),1)), **kw)
    return lgb.LGBMClassifier(objective='binary', random_state=SEED, n_jobs=N_JOBS,
                              verbose=-1, class_weight='balanced', **kw)

OUTCOMES = [('y_any','ANY disease'),('y_RA','Right atrial'),('y_LA','Left atrial'),
            ('y_RV','Right ventricular'),('y_LV','Left ventricular')]
rows = []
for ycol, label in OUTCOMES:
    ytr = tr[ycol].values
    yext_all = Cx[ycol].astype(int).values
    for name in ['RandomForest', 'XGBoost', 'LightGBM']:
        est = build(name, label, ytr)
        est.fit(tr[FEAT].values, ytr)
        p_int = est.predict_proba(te[FEAT].values)[:, 1]
        a_int = roc_auc_score(te[ycol].values, p_int)
        p_ext = est.predict_proba(X.values)[:, 1]
        a_all = roc_auc_score(yext_all, p_ext)
        lo1, hi1 = boot_ci(yext_all, p_ext)
        # non-shunt CHILDHEART-like contrast
        if ycol == 'y_any':
            case = nonshunt_case
        else:
            case = nonshunt_case & (yext_all == 1)
        keepm = case | free
        y2 = case[keepm].astype(int); p2 = p_ext[keepm]
        a_ns = roc_auc_score(y2, p2); lo2, hi2 = boot_ci(y2, p2)
        rows.append(dict(outcome=label, estimator=name,
                         internal_test_auc=round(a_int,4),
                         external_all_icd_auc=round(a_all,4), ext_all_ci_lo=lo1, ext_all_ci_hi=hi1,
                         external_nonshunt_auc=round(a_ns,4), ext_ns_ci_lo=lo2, ext_ns_ci_hi=hi2,
                         internal_minus_external_all=round(a_int-a_all,4),
                         n_ext=len(yext_all), n_ext_events=int(yext_all.sum())))
        print(f'  {label:18s} {name:12s} int {a_int:.4f} | ext(all ICD) {a_all:.4f} | ext(non-shunt) {a_ns:.4f}')

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'ceiling_external.csv'), index=False)

print('\n== does the gradient-boosting advantage transport? ==')
for ycol, label in OUTCOMES:
    r = res[res.outcome == label].set_index('estimator')
    d_int = r.loc['XGBoost','internal_test_auc'] - r.loc['RandomForest','internal_test_auc']
    d_ext = r.loc['XGBoost','external_all_icd_auc'] - r.loc['RandomForest','external_all_icd_auc']
    d_ns  = r.loc['XGBoost','external_nonshunt_auc'] - r.loc['RandomForest','external_nonshunt_auc']
    print(f'  {label:18s} XGB-RF: internal {d_int:+.4f} | external(all) {d_ext:+.4f} | external(non-shunt) {d_ns:+.4f}')
print('\nwrote', OUT)
