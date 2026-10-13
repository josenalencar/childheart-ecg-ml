"""The ECG-manifesting subanalysis, for all three estimator families.

This contrast already existed for the random forest (results/external_validation/
MANIFESTING_ANALYSIS.csv: any disease 0.900). Today's ceiling run used the NON-SHUNT
definition instead, which is a different case definition -- non-shunt excludes isolated
ASD, VSD and PDA, whereas manifesting is a positive list of lesions that load or injure the
chamber directly. Quoting a random-forest-only 0.900 inside a three-estimator paragraph
would compare unlike things.

Lesion lists are copied verbatim from code/external_validation/manifesting_analysis.py so
the two analyses cannot drift.

Outputs (out/manifesting/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_features import build_feature_matrix, clean_implausible, apply_agez

OUT = os.path.join(HERE, 'out', 'manifesting'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS = 42, 5

MANIFESTING = {   # direct chamber load or myocardial injury; shunts deliberately excluded
    'y_LV': ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other', 'dx_aortic_stenosis', 'dx_coarctation'],
    'y_RV': ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension'],
    'y_RA': ['dx_tricuspid_disease', 'dx_pulm_hypertension', 'dx_complex_chd'],
    'y_LA': ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd'],
    'y_any': ['dx_complex_chd', 'dx_pulm_hypertension', 'dx_ToF'],
}
LABEL = {'y_any': 'ANY disease', 'y_RA': 'Right atrial', 'y_LA': 'Left atrial',
         'y_RV': 'Right ventricular', 'y_LV': 'Left ventricular'}

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr = ds[ds.split == 'train']
bundle = joblib.load(os.path.join(HERE, 'data', 'agez_transform.joblib'))
params = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'ceiling_best_params.csv'))

print('building external cohort ...')
ext = pd.read_csv(os.path.join(HERE, 'data', 'ecg_batch_analysis_14190_v4_E3.csv'), low_memory=False)
feat = build_feature_matrix(ext); feat, _ = clean_implausible(feat)
feat['_ECG_ID'] = ext['filename'].astype(str); feat['_error'] = ext['error']
lab = pd.read_excel(os.path.join(HERE, 'data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
keep = ['ECG_ID', 'is_primary_ecg', 'is_12lead', 'age_months', 'sex_M',
        'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
df = feat.merge(lab[keep].rename(columns={'sex_M': 'zzu_sex_M'}), left_on='_ECG_ID',
                right_on='ECG_ID', how='left')
ok = df['_error'].isna() | (df['_error'].astype(str).str.strip() == '') | (df['_error'].astype(str) == 'nan')
C = df[(df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1) & ok].copy()
C['sex_M'] = C['zzu_sex_M'].fillna(C['sex_M']).fillna(0).astype(int)
X = apply_agez(C, C['age_months'].values, bundle)
X.insert(0, 'age_months', C['age_months'].values); X = X[bundle['model_feat']]
dx = [c for c in lab.columns if c.startswith('dx_')]
Cx = C.merge(lab[['ECG_ID'] + dx], on='ECG_ID', how='left')
for c in dx: Cx[c] = Cx[c].fillna(0).astype(int)
free = (Cx.y_any == 0).values
print(f'external cohort {len(C)}; disease-free comparator {int(free.sum())}')

INT_PARAMS = {'n_estimators','min_samples_leaf','max_depth','max_leaf_nodes','num_leaves',
              'min_child_samples','subsample_freq'}
def build(name, outcome, ytr):
    r = params[(params.outcome == outcome) & (params.estimator == name)]
    kw = {}
    if len(r):
        for k, v in r.iloc[0].items():
            if k in ('outcome','estimator') or pd.isna(v): continue
            sv = str(v)
            try: kw[k] = int(sv)
            except ValueError:
                try:
                    f = float(sv); kw[k] = int(f) if (k in INT_PARAMS and f.is_integer()) else f
                except ValueError: kw[k] = None if sv == 'None' else sv
    if name == 'RandomForest':
        return RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=N_JOBS, **kw)
    if name == 'XGBoost':
        return xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                                 random_state=SEED, n_jobs=N_JOBS,
                                 scale_pos_weight=float((ytr==0).sum()/max((ytr==1).sum(),1)), **kw)
    return lgb.LGBMClassifier(objective='binary', random_state=SEED, n_jobs=N_JOBS,
                              verbose=-1, class_weight='balanced', **kw)

rng = np.random.default_rng(SEED)
def boot_ci(y, p, n=2000):
    y=np.asarray(y); p=np.asarray(p); v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))>1: v.append(roc_auc_score(y[i],p[i]))
    return round(float(np.percentile(v,2.5)),3), round(float(np.percentile(v,97.5)),3)

rows = []
for ycol, lesions in MANIFESTING.items():
    les = [c for c in lesions if c in Cx.columns]
    mani = (Cx[les].sum(axis=1) > 0).values
    ytr = tr[ycol].values
    for name in ['RandomForest', 'XGBoost', 'LightGBM']:
        est = build(name, LABEL[ycol], ytr)
        est.fit(tr[FEAT].values, ytr)
        p = est.predict_proba(X.values)[:, 1]
        keepm = mani | free
        y2 = mani[keepm].astype(int); p2 = p[keepm]
        a = roc_auc_score(y2, p2); lo, hi = boot_ci(y2, p2)
        rows.append(dict(outcome=LABEL[ycol], estimator=name,
                         lesions='+'.join(c.replace('dx_','') for c in les),
                         n_manifesting=int(mani.sum()), n_disease_free=int(free.sum()),
                         manifesting_auc=round(a,4), ci_lo=lo, ci_hi=hi))
        print(f'  {LABEL[ycol]:18s} {name:12s} manifesting AUC {a:.4f} ({lo}-{hi})  n={int(mani.sum())}')

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'manifesting_three_estimators.csv'), index=False)
print('\n== any disease, by estimator ==')
print(res[res.outcome == 'ANY disease'][['estimator','manifesting_auc','ci_lo','ci_hi','n_manifesting']].to_string(index=False))
print('\nwrote', OUT)
