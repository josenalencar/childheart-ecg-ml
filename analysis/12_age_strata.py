"""Every headline AUC of the primary model (LightGBM) split at 3 years, internal and external, 2026-10-13.

Why: the CHILDHEART papers report every result under and over 3 years, and the right-heart external validation
showed that age changes both what the ECG means (physiological right ventricular dominance in infants) and the
comparison between cohorts (the external cohort is much older). Script 11 reported the held-out test set in three
age bands; this reports the same split as the other CHILDHEART papers (< 3 y, >= 3 y), internally AND externally.

Inputs (no refit, nothing typed by hand):
  results/tripod_analyses/ceiling/test_predictions.csv            held-out test predictions (LightGBM column)
  results/tripod_analyses/lightgbm/external_predictions_lightgbm.csv  external predictions written by script 11
  Data/[OPEN] Crianças Qualquer doença - Pacientes.csv            internal age ("Idade (m)"), as script 11
  Data/ZZU_external_validation_clinical_textbook.xlsx             external age and labels
External contrasts as in the manuscript: all ICD-coded vs everyone else; ECG-manifesting lesions (lists copied from
script 10) vs disease-free. Bootstrap 1000 resamples for AUC CIs, as script 11's subgroups. Seed 42.
Outputs: results/tripod_analyses/age_strata/age_strata.csv (+ AGE_STRATA.md)
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); RF = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(RF, 'results', 'tripod_analyses', 'age_strata'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'), ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]
MANIFESTING = {'y_LV': ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other', 'dx_aortic_stenosis', 'dx_coarctation'],
               'y_RV': ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension'],
               'y_RA': ['dx_tricuspid_disease', 'dx_pulm_hypertension', 'dx_complex_chd'],
               'y_LA': ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd'],
               'y_any': ['dx_complex_chd', 'dx_pulm_hypertension', 'dx_ToF']}
def boot(y, p, n=1000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
AGES = [('all ages', lambda a: a == a), ('< 3 y', lambda a: a < 36), ('>= 3 y', lambda a: a >= 36)]
rows = []
def add(cohort, contrast, outcome, age, y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        rows.append(dict(cohort=cohort, contrast=contrast, outcome=outcome, age=age, cases=int(y.sum()), controls=int((1 - y).sum()), note='not estimable')); return
    lo, hi = boot(y, p)
    rows.append(dict(cohort=cohort, contrast=contrast, outcome=outcome, age=age, cases=int(y.sum()), controls=int((1 - y).sum()),
                     AUC=roc_auc_score(y, p), AUC_lo=lo, AUC_hi=hi, note=''))

# internal held-out test set
tp = pd.read_csv(os.path.join(RF, 'results', 'tripod_analyses', 'ceiling', 'test_predictions.csv')); tp['folio'] = tp['folio'].astype(str)
man = pd.read_csv(os.path.join(RF, 'Data', '[OPEN] Crianças Qualquer doença - Pacientes.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip(); man = man.drop_duplicates('folio').set_index('folio')
tp['age_months'] = pd.to_numeric(tp['folio'].map(man['Idade (m)']), errors='coerce')
assert tp['age_months'].notna().all(), 'test children without age'
for ycol, label in OUTCOMES:
    for al, af in AGES:
        m = af(tp.age_months).values
        add('Internal held-out test', 'current chamber disease vs all others', label, al, tp.loc[m, ycol], tp.loc[m, f'{ycol}__LightGBM'])

# external (ZZU)
ep = pd.read_csv(os.path.join(RF, 'results', 'tripod_analyses', 'lightgbm', 'external_predictions_lightgbm.csv'))
lab = pd.read_excel(os.path.join(RF, 'Data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
dx = [c for c in lab.columns if c.startswith('dx_')]
X = ep.merge(lab[['ECG_ID', 'age_months', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV'] + dx], on='ECG_ID', how='left')
for c in dx: X[c] = X[c].fillna(0).astype(int)
assert X['age_months'].notna().all() and len(X) == len(ep)
free = X.y_any == 0
for ycol, label in OUTCOMES:
    for al, af in AGES:
        m = af(X.age_months)
        add('External (ZZU)', 'all ICD-coded vs everyone else', label, al, X.loc[m, ycol], X.loc[m, f'{ycol}__LightGBM'])
        mani = X[MANIFESTING[ycol]].sum(axis=1) > 0; sel = m & (mani | free)
        add('External (ZZU)', 'ECG-manifesting lesions vs disease-free', label, al, mani[sel], X.loc[sel, f'{ycol}__LightGBM'])
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, 'age_strata.csv'), index=False)
L = ['# Primary model (LightGBM) by age, internal and external\n', T.round(3).to_markdown(index=False)]
open(os.path.join(OUT, 'AGE_STRATA.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
