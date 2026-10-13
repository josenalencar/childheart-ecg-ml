"""External calibration and decision-curve analysis.

Two gaps left by the first pass.

EXTERNAL CALIBRATION. Item 3 was completed internally only. That is the wrong way round
for this study: the internal probabilities are miscalibrated by construction, because the
case fraction was fixed by the case-control design, so the internal slope mostly measures
the design. The question that matters clinically is what the probabilities do in a cohort
with a real prevalence, which is what the external cohort provides.

DECISION CURVES. Discrimination and calibration say nothing about whether acting on the
model helps. For an instrument proposed as triage -- who gets an echocardiogram, and how
soon -- net benefit across the plausible range of threshold probabilities is the standard
evidence, and it is the one display item a clinical reader will actually use.

Outputs (out/external_calibration/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np, pandas as pd, joblib
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_features import build_feature_matrix, clean_implausible, apply_agez

OUT = os.path.join(HERE, 'out', 'external_calibration'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS = 42, 5

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
bundle = joblib.load(os.path.join(HERE, 'data', 'agez_transform.joblib'))
params = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'ceiling_best_params.csv'))
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]

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
CH = [c for c in ['dx_ASD','dx_VSD','dx_AVSD','dx_ToF','dx_PDA','dx_pulm_stenosis','dx_aortic_stenosis',
                  'dx_coarctation','dx_complex_chd','dx_mitral_disease','dx_tricuspid_disease'] if c in Cx.columns]
SH = [c for c in ['dx_ASD','dx_VSD','dx_PDA'] if c in Cx.columns]
nonshunt = (Cx[CH].sum(1).gt(0) & ~(Cx[SH].sum(1).gt(0) & Cx[[c for c in dx if c not in SH]].sum(1).eq(0))).values
free = (Cx.y_any == 0).values
print(f'external cohort {len(C)}')

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

def calib(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    lp = np.log(p/(1-p))
    slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    cil = float(sm.GLM(y, np.ones((len(y),1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return cil, slope

def net_benefit(y, p, thresholds):
    """Vickers & Elkin net benefit: (TP - FP * pt/(1-pt)) / N, with treat-all and treat-none."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); N = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        yh = (p >= pt).astype(int)
        tp = int(((yh==1)&(y==1)).sum()); fp = int(((yh==1)&(y==0)).sum())
        w = pt/(1-pt)
        rows.append(dict(threshold=round(pt,3),
                         net_benefit_model=round(tp/N - fp/N*w, 5),
                         net_benefit_treat_all=round(prev - (1-prev)*w, 5),
                         net_benefit_treat_none=0.0))
    return pd.DataFrame(rows)

cal_rows, dca_frames = [], []
THR = np.arange(0.01, 0.51, 0.01)
for ycol, label in OUTCOMES:
    ytr = tr[ycol].values
    yext = Cx[ycol].astype(int).values
    for name in ['RandomForest', 'XGBoost', 'LightGBM']:
        est = build(name, label, ytr); est.fit(tr[FEAT].values, ytr)
        p_int = est.predict_proba(te[FEAT].values)[:, 1]
        p_ext = est.predict_proba(X.values)[:, 1]
        for cohort, y, p in [
            ('internal held-out', te[ycol].values, p_int),
            ('external, all ICD+', yext, p_ext),
        ]:
            cil, sl = calib(y, p)
            cal_rows.append(dict(outcome=label, estimator=name, cohort=cohort,
                                 n=len(y), events=int(np.sum(y)),
                                 observed_prevalence=round(float(np.mean(y)), 4),
                                 mean_predicted=round(float(np.mean(p)), 4),
                                 calibration_in_the_large=round(cil, 4),
                                 calibration_slope=round(sl, 4),
                                 brier=round(brier_score_loss(y, p), 4),
                                 auc=round(roc_auc_score(y, p), 4)))
        # non-shunt contrast
        case = nonshunt if ycol == 'y_any' else (nonshunt & (yext == 1))
        keepm = case | free
        y2 = case[keepm].astype(int); p2 = p_ext[keepm]
        cil, sl = calib(y2, p2)
        cal_rows.append(dict(outcome=label, estimator=name, cohort='external, non-shunt',
                             n=len(y2), events=int(y2.sum()),
                             observed_prevalence=round(float(y2.mean()), 4),
                             mean_predicted=round(float(p2.mean()), 4),
                             calibration_in_the_large=round(cil, 4),
                             calibration_slope=round(sl, 4),
                             brier=round(brier_score_loss(y2, p2), 4),
                             auc=round(roc_auc_score(y2, p2), 4)))
        if name == 'RandomForest':
            for cohort, y, p in [('internal held-out', te[ycol].values, p_int),
                                 ('external, all ICD+', yext, p_ext)]:
                nb = net_benefit(y, p, THR); nb['outcome'] = label; nb['cohort'] = cohort
                dca_frames.append(nb)
    print(f'  {label} done')

cal = pd.DataFrame(cal_rows)
cal.to_csv(os.path.join(OUT, 'calibration_internal_external.csv'), index=False)
pd.concat(dca_frames).to_csv(os.path.join(OUT, 'decision_curves.csv'), index=False)

print('\n== calibration, RandomForest ==')
print(cal[cal.estimator == 'RandomForest'][
    ['outcome','cohort','n','events','observed_prevalence','mean_predicted',
     'calibration_in_the_large','calibration_slope','auc']].to_string(index=False))

# where does the model beat treat-all / treat-none?
dca = pd.concat(dca_frames)
print('\n== decision curve: threshold range where the model has the highest net benefit ==')
for (o, c), g in dca.groupby(['outcome', 'cohort']):
    win = g[(g.net_benefit_model > g.net_benefit_treat_all) & (g.net_benefit_model > 0)]
    rng = f'{win.threshold.min():.2f}-{win.threshold.max():.2f}' if len(win) else 'none'
    print(f'  {o:18s} {c:20s} {rng}')
print('\nwrote', OUT)
