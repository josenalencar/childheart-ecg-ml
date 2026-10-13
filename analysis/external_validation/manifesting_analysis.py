"""manifesting_analysis.py — the ECG-manifesting (surrogate-label dilution) analysis,
harmonised with the companion CHILDHEART LV score paper.

Two things are fixed relative to the earlier `compare_and_diagnose.py` diagnostic:

1. CASE DEFINITION. The manifesting group is defined by the LESION LIST ALONE, not by the
   lesion list AND the chamber's ICD crosswalk flag. Requiring both dropped 13 children with
   `dx_cardiomyopathy_other` whose ICD code the crosswalk did not map to the LV, and it is
   what made this study report n=124 where the score paper reports n=137. The manifesting
   group is therefore NOT a strict subset of the headline positives, which is stated.

2. COMPARATOR. Two are reported for every chamber, so the two papers are directly comparable:
     - vs disease-free (y_any == 0)                          = the score paper's analysis B
     - vs ICD-silent positives + disease-free                 = the score paper's analysis B2
   The second contrast FOLDS the chamber's ICD-positive-but-not-manifesting children INTO the
   control group. That is the point of it: if those children carry no electrical signal for the
   chamber, moving them from cases to controls should barely change the AUC, which is the direct
   demonstration that the headline figure is diluted by the surrogate label rather than limited
   by the model. A comparator that merely excluded them would not test this.

3. LEFT-ATRIAL LESION LIST. The previous list (mitral disease + VSD + PDA) captured 730 of 763
   positives, so the split tested nothing. The governing principle — taken from the LV list,
   which contains the cardiomyopathies, aortic stenosis and coarctation but NOT the shunts — is
   that a manifesting lesion loads or injures the chamber DIRECTLY, rather than volume-loading
   it through a shunt. Applied to the left atrium that gives mitral disease, atrioventricular
   septal defect and complex CHD; pulmonary hypertension is excluded as a right-sided load.

Output: results/external_validation/MANIFESTING_ANALYSIS.csv + .md
Usage:  python3 manifesting_analysis.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE, 'code'))
from engine_provenance import engine_string
RES  = os.path.join(BASE, 'results', 'external_validation')
rng  = np.random.default_rng(42)

MANIFESTING = {   # direct chamber load or myocardial injury; shunts deliberately excluded
    'y_LV': ('prob_rf_LV', ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other',
                            'dx_aortic_stenosis', 'dx_coarctation'], 'Left ventricular'),
    'y_RV': ('prob_rf_RV', ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension'], 'Right ventricular'),
    'y_RA': ('prob_rf_RA', ['dx_tricuspid_disease', 'dx_pulm_hypertension',
                            'dx_complex_chd'], 'Right atrial'),
    'y_LA': ('prob_rf_LA', ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd'], 'Left atrial'),
    # ANY disease has no single chamber to load, so its manifesting set is the lesions that
    # load ANY chamber enough to be expected on the surface ECG. This row existed only as a
    # typed constant in generate_manuscript_figures.py (the 0.86 bar of Figure 3, whose
    # footnote names exactly these three lesions) with no script that produced it; it is
    # computed here so the figure reads the same file as the other four bars.
    'y_any': ('prob_rf_any_disease', ['dx_complex_chd', 'dx_pulm_hypertension', 'dx_ToF'],
              'Any disease'),
}

def auc_ci(y, p, n=2000):
    y = np.asarray(y); p = np.asarray(p, float)
    a = roc_auc_score(y, p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return a, np.percentile(v, 2.5), np.percentile(v, 97.5)

def contrast(m, prob, pos, ctrl):
    y = np.r_[np.ones(int(pos.sum())), np.zeros(int(ctrl.sum()))]
    p = np.r_[m.loc[pos, prob].values, m.loc[ctrl, prob].values]
    return auc_ci(y, p)

pred = pd.read_csv(os.path.join(RES, 'textbook', 'predictions.csv'))
lab  = pd.read_excel(os.path.join(BASE, 'Data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
dx   = [c for c in lab.columns if c.startswith('dx_')]
m    = pred.merge(lab[['ECG_ID'] + dx], on='ECG_ID', how='left')
free = m['y_any'] == 0

rows = []
for ycol, (prob, lesions, label) in MANIFESTING.items():
    lesions = [c for c in lesions if c in m.columns]
    mani   = m[lesions].sum(axis=1) > 0
    allpos = m[ycol] == 1
    silent = allpos & (~mani)
    a_all  = contrast(m, prob, allpos, m[ycol] == 0)
    a_free = contrast(m, prob, mani,   free)
    a_neg  = contrast(m, prob, mani,   silent | free)
    a_sil  = contrast(m, prob, silent, free) if silent.sum() > 10 else (np.nan,)*3
    rows.append(dict(chamber=label, lesions='+'.join(c.replace('dx_', '') for c in lesions),
        n_all_ICD_pos=int(allpos.sum()), AUC_all=round(a_all[0], 3),
        n_manifesting=int(mani.sum()),
        n_manifesting_not_ICD_pos=int((mani & (m[ycol] == 0)).sum()),
        AUC_manifesting_vs_diseasefree=round(a_free[0], 3),
        CI_manifesting_vs_diseasefree=f"({a_free[1]:.3f}-{a_free[2]:.3f})",
        AUC_manifesting_vs_silent_plus_free=round(a_neg[0], 3),
        CI_manifesting_vs_silent_plus_free=f"({a_neg[1]:.3f}-{a_neg[2]:.3f})",
        n_silent=int(silent.sum()),
        AUC_silent_vs_diseasefree=(round(a_sil[0], 3) if np.isfinite(a_sil[0]) else None)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'MANIFESTING_ANALYSIS.csv'), index=False)

L = ["# ECG-manifesting analysis — external ZZU cohort\n",
     f"Engine: {engine_string()}\n",
     f"Disease-free comparator n = {int(free.sum())}. Manifesting groups are defined by the lesion "
     "list alone and are therefore not strict subsets of the ICD-positive groups; the number of "
     "manifesting children lacking that chamber's ICD code is given.\n",
     "| Chamber | Manifesting lesions | All ICD+ (n / AUC) | Manifesting n | not ICD+ | vs disease-free | vs ICD-silent + disease-free | ICD-silent n / AUC |",
     "|---|---|---|---|---|---|---|---|"]
for r in rows:
    sil = f"{r['n_silent']} / {r['AUC_silent_vs_diseasefree']}" if r['AUC_silent_vs_diseasefree'] else f"{r['n_silent']} / —"
    L.append(f"| {r['chamber']} | {r['lesions']} | {r['n_all_ICD_pos']} / {r['AUC_all']} | {r['n_manifesting']} | "
             f"{r['n_manifesting_not_ICD_pos']} | **{r['AUC_manifesting_vs_diseasefree']}** "
             f"{r['CI_manifesting_vs_diseasefree']} | {r['AUC_manifesting_vs_silent_plus_free']} "
             f"{r['CI_manifesting_vs_silent_plus_free']} | {sil} |")
L.append("\n*Manifesting lesions load or injure the chamber directly; left-to-right shunts are excluded "
         "from every list, since a shunt volume-loads a chamber without necessarily producing an "
         "electrical substrate. Pulmonary hypertension is assigned to the right-sided chambers only.*")
open(os.path.join(RES, 'MANIFESTING_ANALYSIS.md'), 'w').write("\n".join(L) + "\n")
print("\n".join(L))
