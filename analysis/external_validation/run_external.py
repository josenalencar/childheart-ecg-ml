"""EXTERNAL VALIDATION on the ZZU pediatric ECG database.
Scores the 5 CHILDHEART random forests on the ZZU cohort using EXACTLY the internal
methodology:
  - features built + IMPLAUSIBLE VALUES CLEANED with the shared common_features module
    (same physiologic bounds as CHILDHEART);
  - age-z transform applied with the SAVED internal TRAIN-HEALTHY norms (never refit);
  - one-vs-rest labels (controls = everyone without that chamber's disease, incl. other
    chambers) from the ICD-10 -> chamber crosswalk;
  - models scored with the threshold chosen INTERNALLY (Youden on train OOF) = honest
    transported decision rule (no peeking at external labels).

Run for BOTH label organizations and report both:
  - textbook  = primary-chamber load physiology only
  - empirical = textbook + secondary chambers CHILDHEART echo shows >=50%
               (cardiomyopathy/myocarditis ->+LA, Tetralogy of Fallot ->+RA)

Cohort: one ECG per patient (is_primary_ecg), 12-lead only, batch 'error' empty.

Usage:  python3 run_external.py
Outputs: results/external_validation/<mode>/...  figures/external_validation/<mode>/...
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json
import pandas as pd, numpy as np, joblib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, brier_score_loss)
from statsmodels.stats.proportion import proportion_confint

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, 'code'))
from common_features import build_feature_matrix, clean_implausible, apply_agez
from batch_source import load_external
from engine_provenance import engine_string

DATA = os.path.join(BASE, 'Data')
RES = os.path.join(BASE, 'results', 'external_validation')
FIG = os.path.join(BASE, 'figures', 'external_validation')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
MODELS = [('rf_any_disease', 'y_any', 'ANY disease'), ('rf_RA', 'y_RA', 'Right atrial'),
          ('rf_LA', 'y_LA', 'Left atrial'), ('rf_RV', 'y_RV', 'Right ventricular'),
          ('rf_LV', 'y_LV', 'Left ventricular')]
rng = np.random.default_rng(42)

def boot_ci(yv, p, fn, n=2000):
    yv = np.asarray(yv); p = np.asarray(p); v = []
    for _ in range(n):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2: continue
        v.append(fn(yv[idx], p[idx]))
    return (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3))
def simel(tp, fp, fn, tn):
    a, b, c, d = tp, fp, fn, tn
    if min(a, b, c, d) == 0: a, b, c, d = a + .5, b + .5, c + .5, d + .5
    se = a / (a + c); sp = d / (b + d); lrp = se / (1 - sp); lrn = (1 - se) / sp
    selp = np.sqrt(1/a - 1/(a+c) + 1/b - 1/(b+d)); seln = np.sqrt(1/c - 1/(a+c) + 1/d - 1/(b+d))
    return (round(lrp, 2), (round(np.exp(np.log(lrp)-1.96*selp), 2), round(np.exp(np.log(lrp)+1.96*selp), 2)),
            round(lrn, 2), (round(np.exp(np.log(lrn)-1.96*seln), 2), round(np.exp(np.log(lrn)+1.96*seln), 2)))
def wci(k, nn): lo, hi = proportion_confint(k, nn, method='wilson'); return (round(lo*100, 1), round(hi*100, 1))

# ----- load external batch features, build + clean (SHARED methodology) -----
# The vintage lives in ONE place (code/batch_source.py) and the engine that produced it is
# READ from the engine (code/engine_provenance.py), because a typed engine string has twice
# outlived the extraction it described. CH_VINTAGE=v3_D2 reproduces the September run.
print('engine:', engine_string())
ext = load_external(BASE)
feat = build_feature_matrix(ext)
feat, rep = clean_implausible(feat)
rep.to_csv(os.path.join(RES, 'implausible_clean_report_external.csv'), index=False)
print('external implausible values nulled: %d across %d features (same bounds as internal)'
      % (int(rep.n_nulled.sum()) if len(rep) else 0, len(rep)))
feat['_ECG_ID'] = ext['filename'].astype(str)
feat['_error'] = ext['error']
bundle = joblib.load(os.path.join(BASE, 'results', 'agez_transform.joblib'))

AGE_BINS = [('1mo-3y', 0, 36), ('3-8y', 36, 96), ('>8y', 96, 1e9)]

for MODE in ['textbook', 'empirical']:
    print(f'\n################ EXTERNAL MODE = {MODE} ################')
    od_r = os.path.join(RES, MODE); od_f = os.path.join(FIG, MODE)
    os.makedirs(od_r, exist_ok=True); os.makedirs(od_f, exist_ok=True)
    lab = pd.read_excel(os.path.join(DATA, f'ZZU_external_validation_clinical_{MODE}.xlsx'), 'all_ecgs')
    keep = ['ECG_ID', 'is_primary_ecg', 'is_12lead', 'age_months', 'sex_M',
            'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
    labk = lab[keep].rename(columns={'sex_M': 'zzu_sex_M'})  # authoritative sex from ZZU Gender
    df = feat.merge(labk, left_on='_ECG_ID', right_on='ECG_ID', how='left')
    # cohort: one ECG/patient, 12-lead, no batch error
    err_ok = df['_error'].isna() | (df['_error'].astype(str).str.strip() == '') | (df['_error'].astype(str) == 'nan')
    cohort = (df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1) & err_ok
    n_total = len(df); n_prim = int((df['is_primary_ecg'] == 1).sum())
    n_prim12 = int(((df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1)).sum())
    C = df[cohort].copy()
    C['sex_M'] = C['zzu_sex_M'].fillna(C['sex_M']).fillna(0).astype(int)  # ZZU sex, fallback batch
    print(f'cohort: {len(C)} (primary {n_prim}; primary+12lead {n_prim12}; '
          f'excluded {n_prim12-len(C)} errored 12-lead)')
    # build model matrix via SAVED norms (override sex_M with ZZU truth)
    raw_in = C.copy()
    X = apply_agez(raw_in, C['age_months'].values, bundle)   # cont (z) + raw
    X.insert(0, 'age_months', C['age_months'].values)         # age_months is a model feature
    X = X[bundle['model_feat']]                               # exact training column order
    summary = []
    preds = C[['_ECG_ID', 'age_months', 'sex_M', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']].rename(columns={'_ECG_ID': 'ECG_ID'}).reset_index(drop=True)
    for name, y, desc in MODELS:
        rf = joblib.load(os.path.join(BASE, 'results', name, 'model_agez.joblib'))
        thr = json.load(open(os.path.join(BASE, 'results', name, 'metrics_agez.json')))['threshold']
        yv = C[y].astype(int).values
        p = rf.predict_proba(X.values)[:, 1]
        preds[f'prob_{name}'] = p
        auc = roc_auc_score(yv, p); ap = average_precision_score(yv, p)
        yp = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yv, yp, labels=[0, 1]).ravel()
        sens = tp/(tp+fn)*100 if tp+fn else 0; spec = tn/(tn+fp)*100 if tn+fp else 0
        ppv = tp/(tp+fp)*100 if tp+fp else 0; npv = tn/(tn+fn)*100 if tn+fn else 0
        lrp, lrp_ci, lrn, lrn_ci = simel(tp, fp, fn, tn)
        strata = []
        for slab, lo, hi in AGE_BINS:
            m = (C.age_months > lo) & (C.age_months <= hi) if lo > 0 else (C.age_months <= hi)
            yy = yv[m.values]; pp = p[m.values]
            a = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
            ypp = (pp >= thr).astype(int); tn2, fp2, fn2, tp2 = confusion_matrix(yy, ypp, labels=[0, 1]).ravel()
            strata.append(dict(stratum=slab, n=int(m.sum()), n_pos=int(yy.sum()),
                auc=round(a, 3) if a == a else None,
                sens=round(tp2/(tp2+fn2)*100, 0) if tp2+fn2 else None,
                spec=round(tn2/(tn2+fp2)*100, 0) if tn2+fp2 else None))
        M = dict(model=desc, label=y, mode=MODE, transported_threshold=round(thr, 3),
            n=int(len(yv)), n_pos=int(yv.sum()), n_controls=int((yv == 0).sum()), prevalence=round(yv.mean(), 3),
            ext_roc_auc=round(auc, 3), ext_roc_auc_95CI=boot_ci(yv, p, roc_auc_score),
            ext_pr_auc=round(ap, 3), ext_pr_auc_95CI=boot_ci(yv, p, average_precision_score),
            sensitivity=round(sens, 1), sensitivity_95CI=wci(tp, tp+fn),
            specificity=round(spec, 1), specificity_95CI=wci(tn, tn+fp),
            PPV=round(ppv, 1), PPV_95CI=wci(tp, tp+fp), NPV=round(npv, 1), NPV_95CI=wci(tn, tn+fn),
            LR_pos=lrp, LR_pos_95CI=lrp_ci, LR_neg=lrn, LR_neg_95CI=lrn_ci,
            brier=round(brier_score_loss(yv, p), 3),
            confusion=dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn)), strata=strata)
        os.makedirs(os.path.join(od_r, name), exist_ok=True)
        json.dump(M, open(os.path.join(od_r, name, 'metrics_external.json'), 'w'), indent=2)
        pd.DataFrame(strata).to_csv(os.path.join(od_r, name, 'stratum_external.csv'), index=False)
        # figures
        os.makedirs(os.path.join(od_f, name), exist_ok=True)
        fpr, tpr, _ = roc_curve(yv, p); plt.figure(figsize=(5, 5)); plt.plot(fpr, tpr, color='#2563eb', lw=2, label=f'AUC={auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=.7); plt.xlabel('1-Specificity'); plt.ylabel('Sensitivity')
        plt.title(f'{desc} — EXTERNAL ROC ({MODE})'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(od_f, name, 'roc_external.png'), dpi=120); plt.close()
        pr, rc, _ = precision_recall_curve(yv, p); plt.figure(figsize=(5, 5)); plt.plot(rc, pr, color='#16a34a', lw=2, label=f'AP={ap:.2f}')
        plt.axhline(yv.mean(), ls='--', color='gray', label=f'baseline={yv.mean():.2f}'); plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'{desc} — EXTERNAL PR ({MODE})'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(od_f, name, 'pr_external.png'), dpi=120); plt.close()
        plt.figure(figsize=(4, 4)); cm = np.array([[tn, fp], [fn, tp]]); plt.imshow(cm, cmap='Blues')
        for i in range(2):
            for j2 in range(2): plt.text(j2, i, cm[i, j2], ha='center', va='center', fontsize=12)
        plt.xticks([0, 1], ['pred-', 'pred+']); plt.yticks([0, 1], ['true-', 'true+'])
        plt.title(f'{desc} — EXTERNAL confusion ({MODE})'); plt.tight_layout()
        plt.savefig(os.path.join(od_f, name, 'confusion_external.png'), dpi=120); plt.close()
        plt.figure(figsize=(5, 5)); dfp = pd.DataFrame({'prob': p, 'class': np.where(yv == 1, desc, 'control')})
        sns.violinplot(data=dfp, x='class', y='prob', cut=0, inner='quartile', palette=['#16a34a', '#dc2626'])
        plt.axhline(thr, ls='--', color='k', lw=.8); plt.title(f'{desc} — EXTERNAL prob by class ({MODE})')
        plt.tight_layout(); plt.savefig(os.path.join(od_f, name, 'prob_by_class_external.png'), dpi=120); plt.close()
        summary.append(dict(model=desc, label=y, n=int(len(yv)), n_pos=int(yv.sum()),
            ext_auc=round(auc, 3), ext_auc_CI=M['ext_roc_auc_95CI'], pr_auc=round(ap, 3),
            sens=round(sens, 0), spec=round(spec, 0), LRp=lrp, LRn=lrn, threshold=round(thr, 3)))
        print(f'  {name}: ext AUC={auc:.3f} {M["ext_roc_auc_95CI"]} PR-AUC={ap:.2f} '
              f'sens={sens:.0f} spec={spec:.0f} LR+={lrp} (n_pos {int(yv.sum())}/{len(yv)})')
    pd.DataFrame(summary).to_csv(os.path.join(od_r, 'SUMMARY_external.csv'), index=False)
    preds.to_csv(os.path.join(od_r, 'predictions.csv'), index=False)
    print(f'  saved {od_r}/SUMMARY_external.csv + predictions.csv')
print('\nDONE. results/external_validation/{textbook,empirical}/  figures/external_validation/{textbook,empirical}/')
