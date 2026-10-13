"""Post-process the external validation: (1) internal-vs-external comparison table +
figure, for BOTH label organizations; (2) the 'surrogate-label dilution' diagnostic —
external AUC on ALL ICD-positives vs on the hemodynamically-significant / ECG-manifesting
lesion subset (controls unchanged). Writes the EXTERNAL_README.md.
"""
import os, json
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results', 'external_validation')
FIG = os.path.join(BASE, 'figures', 'external_validation')
MODELS = [('rf_any_disease', 'y_any', 'ANY disease'), ('rf_RA', 'y_RA', 'Right atrial'),
          ('rf_LA', 'y_LA', 'Left atrial'), ('rf_RV', 'y_RV', 'Right ventricular'),
          ('rf_LV', 'y_LV', 'Left ventricular')]

# ---- (1) internal vs external comparison ----
rows = []
for name, y, desc in MODELS:
    intm = json.load(open(os.path.join(BASE, 'results', name, 'metrics_agez.json')))
    r = dict(model=desc, label=y, internal_AUC=intm['test_roc_auc'],
             internal_AUC_CI=tuple(intm['test_roc_auc_95CI']))
    for MODE in ['textbook', 'empirical']:
        em = json.load(open(os.path.join(RES, MODE, name, 'metrics_external.json')))
        r[f'external_{MODE}_AUC'] = em['ext_roc_auc']
        r[f'external_{MODE}_AUC_CI'] = tuple(em['ext_roc_auc_95CI'])
        r[f'external_{MODE}_PRAUC'] = em['ext_pr_auc']
        r[f'external_{MODE}_sens'] = em['sensitivity']
        r[f'external_{MODE}_spec'] = em['specificity']
        r[f'external_{MODE}_npos'] = em['n_pos']
    rows.append(r)
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(RES, 'COMPARISON_internal_vs_external.csv'), index=False)
print('COMPARISON_internal_vs_external.csv\n', comp[['model','internal_AUC','external_textbook_AUC','external_empirical_AUC']].to_string(index=False))

# bar figure
fig, ax = plt.subplots(figsize=(9, 5))
labels = [d for _, _, d in MODELS]; x = np.arange(len(labels)); w = 0.27
ax.bar(x - w, comp['internal_AUC'], w, label='Internal (echo labels)', color='#2563eb')
ax.bar(x,     comp['external_textbook_AUC'], w, label='External ZZU — textbook', color='#16a34a')
ax.bar(x + w, comp['external_empirical_AUC'], w, label='External ZZU — empirical', color='#f59e0b')
ax.axhline(0.5, ls='--', color='gray', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right'); ax.set_ylabel('ROC-AUC')
ax.set_ylim(0.4, 1.0); ax.set_title('Internal (echo) vs External ZZU (ICD surrogate) — ROC-AUC')
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, 'internal_vs_external_auc.png'), dpi=130); plt.close()

# ---- (2) surrogate-label dilution diagnostic ----
# NOTE: this block requires BOTH the lesion flag and the chamber's ICD label, so its manifesting
# counts are smaller than the canonical analysis. The harmonised, publication-facing version —
# lesion-list-only case definition, two comparators, matched to the companion LV score paper —
# is manifesting_analysis.py -> results/external_validation/MANIFESTING_ANALYSIS.{csv,md}.
# For each chamber: AUC on ALL ICD-positives vs on the ECG-manifesting lesion subset.
# controls = all that-chamber-negatives (unchanged). Uses textbook predictions+labels.
pred = pd.read_csv(os.path.join(RES, 'textbook', 'predictions.csv'))
lab = pd.read_excel(os.path.join(BASE, 'Data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
dxcols = [c for c in lab.columns if c.startswith('dx_')]
m = pred.merge(lab[['ECG_ID'] + dxcols], on='ECG_ID', how='left')
HI = {  # hemodynamically-significant / ECG-manifesting lesions per chamber
    'y_RV': ('prob_rf_RV', ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension']),
    'y_LV': ('prob_rf_LV', ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other', 'dx_aortic_stenosis', 'dx_coarctation']),
    'y_RA': ('prob_rf_RA', ['dx_tricuspid_disease', 'dx_pulm_hypertension', 'dx_complex_chd']),
    'y_LA': ('prob_rf_LA', ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd']),
}
def auc_of(prob, pos_mask, ctrl_mask):
    yv = np.r_[np.ones(pos_mask.sum()), np.zeros(ctrl_mask.sum())]
    pp = np.r_[m.loc[pos_mask, prob].values, m.loc[ctrl_mask, prob].values]
    return round(roc_auc_score(yv, pp), 3)
drows = []
for y, (prob, hicols) in HI.items():
    ctrl = m[y] == 0
    allpos = m[y] == 1
    hi = allpos & (m[hicols].sum(axis=1) > 0)
    lo = allpos & (m[hicols].sum(axis=1) == 0)
    drows.append(dict(chamber=y, n_all_pos=int(allpos.sum()), AUC_all=auc_of(prob, allpos, ctrl),
        n_highyield=int(hi.sum()), AUC_highyield=auc_of(prob, hi, ctrl),
        n_lowyield=int(lo.sum()), AUC_lowyield=auc_of(prob, lo, ctrl) if lo.sum() > 5 else None,
        highyield_lesions='+'.join(c.replace('dx_', '') for c in hicols)))
diag = pd.DataFrame(drows)
diag.to_csv(os.path.join(RES, 'highyield_dilution_diagnostic.csv'), index=False)
print('\nhighyield_dilution_diagnostic.csv\n', diag.to_string(index=False))

# dilution figure
fig, ax = plt.subplots(figsize=(8, 5))
ch = diag['chamber'].tolist(); x = np.arange(len(ch)); w = 0.38
ax.bar(x - w/2, diag['AUC_all'], w, label='ALL ICD-positives', color='#94a3b8')
ax.bar(x + w/2, diag['AUC_highyield'], w, label='ECG-manifesting lesions only', color='#dc2626')
ax.axhline(0.5, ls='--', color='gray', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(ch); ax.set_ylabel('External ROC-AUC'); ax.set_ylim(0.4, 0.95)
ax.set_title('Surrogate-label dilution: signal recovers on ECG-manifesting lesions')
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, 'surrogate_dilution.png'), dpi=130); plt.close()
print('\nsaved figures/external_validation/{internal_vs_external_auc,surrogate_dilution}.png')
