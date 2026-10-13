"""Item 11 -- the LightGBM-primary analyses that existed only for the random forest.

LightGBM is the primary estimator of this paper: the selection rule, applied to TRAINING
data only, is the highest nested-CV ROC-AUC for the primary outcome (any disease), where
LightGBM scored 0.9831 against 0.9793 for XGBoost and 0.9757 for the tuned forest
(out/ceiling/ceiling_nested_cv.csv). Scripts 02, 03 and the decision-curve half of 08 were
written when the forest was primary and report the forest only, so the paper would otherwise
describe a LightGBM model with a random forest's calibration, subgroups, feature importance
and net benefit. Everything here is recomputed for LightGBM with the hyperparameters already
tuned and audited in out/ceiling/ceiling_best_params.csv; 01_ceiling_models.py is NOT re-run,
so the audited nested-CV selection and the held-out test predictions stay byte-identical.

Items, following the handoff of 2026-10-13:
 1 Test-set operating characteristics with no refit, read from out/ceiling/test_predictions.csv:
   PR-AUC, Brier, and sensitivity/specificity/PPV/NPV at two thresholds, both chosen on
   out-of-fold TRAIN predictions and applied unchanged to the test set -- the Youden point and
   the point giving at least 95% specificity. A threshold picked on the test set would report
   the best of 328 children rather than a decision rule.
 2 Subgroups by sex and age band on the held-out test set (TRIPOD+AI 12d/14).
 3 Split lottery: the headline AUC is one draw from a distribution: 100 stratified splits.
 4 Internal calibration for LightGBM, cross-checked against the LightGBM "internal held-out"
   rows that 08 already wrote, so the two scripts cannot disagree silently.
 5 SHAP importance (TreeExplainer), bootstrap stability of the top-20 and the parsimony floor.
   This is what decides whether the manuscript's QRS-T-angle narrative, built on the forest's
   Gini ranking, survives under the primary model. Whatever comes out is what gets reported.
 6 Decision curves for LightGBM. 08 guards its DCA block with `if name == 'RandomForest'`,
   so decision_curves.csv is the forest's net benefit and nothing else.
 7 Transportable-feature retrain. The univariate half of that analysis is model-agnostic and
   is kept from the archived cross-cohort run; the "retrained on the transportable subset"
   half was a forest and is redone here, reusing the same leak-safe design: the external
   cohort is split once at patient level stratified by disease (seed 42), features are chosen
   on the internal training partition and the external SELECTION half, and every external
   number is computed in the REPORTING half only.

No number is typed by hand anywhere: every value is computed or read from a CSV.
Outputs (out/lightgbm/).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json, time
import numpy as np, pandas as pd, joblib
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_features import build_feature_matrix, clean_implausible, apply_agez

OUT = os.path.join(HERE, 'out', 'lightgbm'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS, N_BOOT, N_LOTTERY, N_SPLIT_SEED = 42, 5, 60, 100, 1000
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]
log(f'internal: train {len(tr)}, test {len(te)}, {len(FEAT)} features')

# ------------------------------------------------------------------ the primary estimator
params = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'ceiling_best_params.csv'))
INT_PARAMS = {'n_estimators', 'min_samples_leaf', 'max_depth', 'max_leaf_nodes', 'num_leaves',
              'min_child_samples', 'subsample_freq'}

def lgbm_kwargs(label):
    """The tuned hyperparameters for this outcome, read from the audited ceiling run.

    pandas widens a column to float when any row is missing, so an integer hyperparameter
    comes back as e.g. 12.0, which LightGBM rejects; INT_PARAMS puts those back.
    """
    r = params[(params.outcome == label) & (params.estimator == 'LightGBM')]
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

def LGB(label, n_jobs=N_JOBS, **over):
    kw = lgbm_kwargs(label); kw.update(over)
    return lgb.LGBMClassifier(objective='binary', random_state=SEED, n_jobs=n_jobs,
                              verbose=-1, class_weight='balanced', **kw)

pd.DataFrame([dict(outcome=l, **lgbm_kwargs(l)) for _, l in OUTCOMES]).to_csv(
    os.path.join(OUT, 'lightgbm_hyperparameters.csv'), index=False)

rng = np.random.default_rng(SEED)
def boot_ci(y, p, fn, n=2000):
    y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            try: v.append(fn(y[i], p[i]))
            except Exception: pass
    if not v: return (np.nan, np.nan)
    return round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)

# ===================== item 1: operating characteristics, no refit =====================
log('item 1: out-of-fold train thresholds -> held-out test')
tp = pd.read_csv(os.path.join(HERE, 'out', 'ceiling', 'test_predictions.csv'))
oof_store, op_rows = {}, []

def point_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn_, tp_ = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp_ / (tp_ + fn_) if (tp_ + fn_) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(threshold=round(float(thr), 4),
                sensitivity=round(float(sens), 4), specificity=round(float(spec), 4),
                ppv=round(float(tp_ / (tp_ + fp)), 4) if (tp_ + fp) else np.nan,
                npv=round(float(tn / (tn + fn_)), 4) if (tn + fn_) else np.nan,
                lr_pos=round(float(sens / (1 - spec)), 3) if spec < 1 else np.nan,
                tp=int(tp_), fp=int(fp), tn=int(tn), fn=int(fn_))

for ycol, label in OUTCOMES:
    ytr = tr[ycol].values
    oof = cross_val_predict(LGB(label, n_jobs=1), tr[FEAT].values, ytr,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method='predict_proba', n_jobs=N_JOBS)[:, 1]
    oof_store[ycol] = oof
    fpr, tpr, thr = roc_curve(ytr, oof)
    thr_youden = float(thr[np.argmax(tpr - fpr)])
    ok = np.where((1 - fpr) >= 0.95)[0]
    thr_spec95 = float(thr[ok[np.argmax(tpr[ok])]])

    y, p = te[ycol].values, tp[f'{ycol}__LightGBM'].values
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p, roc_auc_score)
    pr = average_precision_score(y, p)
    prlo, prhi = boot_ci(y, p, average_precision_score)
    base = dict(outcome=label, n_test=len(y), events_test=int(y.sum()),
                prevalence=round(float(y.mean()), 4),
                auc=round(float(auc), 4), auc_lo=lo, auc_hi=hi,
                pr_auc=round(float(pr), 4), pr_auc_lo=prlo, pr_auc_hi=prhi,
                pr_auc_baseline=round(float(y.mean()), 4),
                brier=round(float(brier_score_loss(y, p)), 4),
                brier_null=round(float(brier_score_loss(y, np.full(len(y), y.mean()))), 4),
                oof_train_auc=round(float(roc_auc_score(ytr, oof)), 4))
    for tname, t in [('Youden (OOF train)', thr_youden), ('>=95% specificity (OOF train)', thr_spec95)]:
        m = point_metrics(y, p, t)
        sl, sh = boot_ci(y, p, lambda yy, pp, tt=t: (yy[(pp >= tt)].sum() / max(yy.sum(), 1)))
        op_rows.append({**base, 'threshold_rule': tname, **m,
                        'sensitivity_lo': sl, 'sensitivity_hi': sh})
    log(f'  {label:18s} AUC {auc:.3f} PR-AUC {pr:.3f} | Youden thr {thr_youden:.3f} | spec95 thr {thr_spec95:.3f}')

op = pd.DataFrame(op_rows)
op.to_csv(os.path.join(OUT, 'test_operating_characteristics.csv'), index=False)
pd.DataFrame({'folio': tr['folio'].values,
              **{f'{y}__oof_LightGBM': oof_store[y] for y, _ in OUTCOMES},
              **{y: tr[y].values for y, _ in OUTCOMES}}).to_csv(
    os.path.join(OUT, 'oof_train_predictions.csv'), index=False)

# ============================== item 2: subgroups ==============================
log('item 2: subgroups on the held-out test set')
man = pd.read_csv(os.path.join(HERE, 'data', 'manual_sheet.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip()
man = man.drop_duplicates('folio').set_index('folio')
te2 = te.copy(); te2['folio'] = te2['folio'].astype(str)
te2['sex_M'] = te2['folio'].map(man['Sexo'].astype(str).str.upper().str[0].eq('M'))
te2['age_months'] = pd.to_numeric(te2['folio'].map(man['Idade (m)']), errors='coerce')
tp2 = tp.copy(); tp2['folio'] = tp2['folio'].astype(str)
te2 = te2.merge(tp2[['folio'] + [f'{y}__LightGBM' for y, _ in OUTCOMES]], on='folio', how='left')

sub = []
for ycol, label in OUTCOMES:
    y = te2[ycol].values; p = te2[f'{ycol}__LightGBM'].values
    groups = [('all', np.ones(len(te2), bool)),
              ('female', (~te2.sex_M.fillna(False)).values),
              ('male', te2.sex_M.fillna(False).values),
              ('age <3y', (te2.age_months < 36).values),
              ('age 3-8y', ((te2.age_months >= 36) & (te2.age_months < 96)).values),
              ('age >=8y', (te2.age_months >= 96).values)]
    for gname, mask in groups:
        mask = np.asarray(mask, bool)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            sub.append(dict(outcome=label, subgroup=gname, n=int(mask.sum()),
                            events=int(y[mask].sum()), auc=None, auc_lo=None, auc_hi=None,
                            note='not estimable')); continue
        lo, hi = boot_ci(y[mask], p[mask], roc_auc_score, n=1000)
        sub.append(dict(outcome=label, subgroup=gname, n=int(mask.sum()),
                        events=int(y[mask].sum()),
                        auc=round(float(roc_auc_score(y[mask], p[mask])), 3),
                        auc_lo=lo, auc_hi=hi, note=''))
pd.DataFrame(sub).to_csv(os.path.join(OUT, 'subgroup_fairness_test.csv'), index=False)
log(f'  wrote subgroups for {len(OUTCOMES)} outcomes')

# ============================== item 4: internal calibration ==============================
log('item 4: internal calibration (LightGBM)')
def calib(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    cil = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                       offset=lp).fit().params[0])
    return dict(calibration_in_the_large=round(cil, 4), calibration_slope=round(slope, 4),
                mean_predicted=round(float(p.mean()), 4), observed=round(float(y.mean()), 4),
                brier=round(float(brier_score_loss(y, p)), 4),
                auc=round(float(roc_auc_score(y, p)), 4))

cal_rows = []
for ycol, label in OUTCOMES:
    cal_rows.append(dict(outcome=label, cohort='internal held-out', n=len(te),
                         events=int(te[ycol].sum()), **calib(te[ycol].values, tp[f'{ycol}__LightGBM'].values)))
    cal_rows.append(dict(outcome=label, cohort='out-of-fold train', n=len(tr),
                         events=int(tr[ycol].sum()), **calib(tr[ycol].values, oof_store[ycol])))
cal = pd.DataFrame(cal_rows)
cal.to_csv(os.path.join(OUT, 'calibration_internal.csv'), index=False)

# cross-check against what 08 already wrote, so the two scripts cannot disagree silently
prev = pd.read_csv(os.path.join(HERE, 'out', 'external_calibration', 'calibration_internal_external.csv'))
chk = prev[(prev.estimator == 'LightGBM') & (prev.cohort == 'internal held-out')]
mrg = cal[cal.cohort == 'internal held-out'].merge(chk, on='outcome', suffixes=('_new', '_08'))
mrg['slope_abs_diff'] = (mrg.calibration_slope_new - mrg.calibration_slope_08).abs()
mrg[['outcome', 'calibration_slope_new', 'calibration_slope_08', 'slope_abs_diff',
     'auc_new', 'auc_08']].to_csv(os.path.join(OUT, 'calibration_crosscheck_vs_script08.csv'), index=False)
log(f'  calibration cross-check: max |slope diff| vs script 08 = {mrg.slope_abs_diff.max():.4f}')

# ============================== item 3: split lottery ==============================
log(f'item 3: split lottery ({N_LOTTERY} stratified splits)')
lot = []
for ycol, label in OUTCOMES:
    X, y = ds[FEAT].values, ds[ycol].values
    aucs = []
    for i in range(N_LOTTERY):
        itr, ite = train_test_split(np.arange(len(ds)), test_size=0.2,
                                    random_state=N_SPLIT_SEED + i, stratify=ds['y_any'].values)
        if len(np.unique(y[ite])) < 2: continue
        m = LGB(label).fit(X[itr], y[itr])
        aucs.append(roc_auc_score(y[ite], m.predict_proba(X[ite])[:, 1]))
    a = np.array(aucs)
    fixed = float(roc_auc_score(te[ycol].values, tp[f'{ycol}__LightGBM'].values))
    lot.append(dict(outcome=label, n_splits=len(a), mean=round(float(a.mean()), 3),
                    sd=round(float(a.std()), 3), p2_5=round(float(np.percentile(a, 2.5)), 3),
                    p97_5=round(float(np.percentile(a, 97.5)), 3),
                    min=round(float(a.min()), 3), max=round(float(a.max()), 3),
                    reported_fixed_split=round(fixed, 3),
                    percentile_of_reported=round(float((a < fixed).mean() * 100), 1)))
    log(f'  lottery {label:18s} {a.mean():.3f} +/- {a.std():.3f} [{a.min():.3f}-{a.max():.3f}]'
        f'  reported split {fixed:.3f} = p{lot[-1]["percentile_of_reported"]:.0f}')
pd.DataFrame(lot).to_csv(os.path.join(OUT, 'split_lottery.csv'), index=False)

# ================= item 5: SHAP, bootstrap stability, parsimony floor =================
log('item 5a: SHAP importance (TreeExplainer)')
rank_rows = []
for ycol, label in OUTCOMES:
    m = LGB(label).fit(tr[FEAT].values, tr[ycol].values)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(te[FEAT].values)
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEAT)
    gain = pd.Series(m.booster_.feature_importance(importance_type='gain'), index=FEAT)
    split = pd.Series(m.booster_.feature_importance(importance_type='split'), index=FEAT)
    r = pd.DataFrame({'shap': shap_imp, 'gain': gain, 'split': split})
    for c in ['shap', 'gain', 'split']:
        r[c + '_rank'] = r[c].rank(ascending=False, method='min')
    r['outcome'] = label
    r.sort_values('shap', ascending=False).head(30).reset_index().rename(
        columns={'index': 'feature'}).to_csv(os.path.join(OUT, f'importance_{ycol}.csv'), index=False)
    top_s = list(shap_imp.sort_values(ascending=False).head(10).index)
    top_g = list(gain.sort_values(ascending=False).head(10).index)
    rank_rows.append(dict(outcome=label, top1_shap=top_s[0], top1_gain=top_g[0],
                          overlap_top10=len(set(top_s) & set(top_g)),
                          spearman_shap_gain=round(float(r['shap'].corr(r['gain'], method='spearman')), 3),
                          top10_shap=';'.join(top_s)))
    log(f'  {label:18s} top-1 SHAP={top_s[0][:34]:34s} gain={top_g[0][:28]:28s} overlap {rank_rows[-1]["overlap_top10"]}/10')
pd.DataFrame(rank_rows).to_csv(os.path.join(OUT, 'shap_vs_gain_summary.csv'), index=False)

log(f'item 5b: bootstrap stability of the top-20 ({N_BOOT} resamples)')
stab_rows = []
for ycol, label in OUTCOMES:
    Xtr, ytr = tr[FEAT].values, tr[ycol].values
    counts = pd.Series(0, index=FEAT, dtype=int); done = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        if len(np.unique(ytr[idx])) < 2: continue
        mb = LGB(label).fit(Xtr[idx], ytr[idx])
        imp = pd.Series(mb.booster_.feature_importance(importance_type='gain'), index=FEAT)
        counts[imp.sort_values(ascending=False).head(20).index] += 1; done += 1
    freq = (counts / max(done, 1)).sort_values(ascending=False)
    freq.head(40).rename('top20_selection_frequency').reset_index().rename(
        columns={'index': 'feature'}).assign(outcome=label).to_csv(
        os.path.join(OUT, f'stability_{ycol}.csv'), index=False)
    base = LGB(label).fit(Xtr, ytr)
    base_top = pd.Series(base.booster_.feature_importance(importance_type='gain'),
                         index=FEAT).sort_values(ascending=False).head(20).index
    stab_rows.append(dict(outcome=label, n_resamples=done,
                          n_features_ever_in_top20=int((counts > 0).sum()),
                          n_features_in_top20_over_80pct=int((freq >= 0.8).sum()),
                          n_features_in_top20_over_50pct=int((freq >= 0.5).sum()),
                          median_freq_of_pointestimate_top20=round(float(freq[base_top].median()), 2)))
    log(f'  {label:18s} {stab_rows[-1]["n_features_ever_in_top20"]:4d} ever top-20; '
        f'{stab_rows[-1]["n_features_in_top20_over_80pct"]:3d} in >=80% of resamples')
pd.DataFrame(stab_rows).to_csv(os.path.join(OUT, 'stability_summary.csv'), index=False)

log('item 5c: parsimony floor')
par_rows = []
for ycol, label in OUTCOMES:
    freq = pd.read_csv(os.path.join(OUT, f'stability_{ycol}.csv')).set_index('feature')['top20_selection_frequency']
    full_auc = float(roc_auc_score(te[ycol].values, tp[f'{ycol}__LightGBM'].values))
    for k in [5, 10, 20]:
        sel = list(freq.head(k).index)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000,
                           class_weight='balanced', random_state=SEED))
        cv = cross_val_predict(lr, tr[sel].values, tr[ycol].values,
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               method='predict_proba', n_jobs=N_JOBS)[:, 1]
        lr.fit(tr[sel].values, tr[ycol].values)
        a = float(roc_auc_score(te[ycol].values, lr.predict_proba(te[sel].values)[:, 1]))
        mk = LGB(label).fit(tr[sel].values, tr[ycol].values)
        a_lgb = float(roc_auc_score(te[ycol].values, mk.predict_proba(te[sel].values)[:, 1]))
        par_rows.append(dict(outcome=label, k=k, features=';'.join(sel),
                             LR_cv_auc=round(float(roc_auc_score(tr[ycol].values, cv)), 3),
                             LR_test_auc=round(a, 3), LGBM_kfeat_test_auc=round(a_lgb, 3),
                             LGBM_1047_test_auc=round(full_auc, 3),
                             recovery_pct=round(100 * (a - 0.5) / (full_auc - 0.5), 1)))
        log(f'  {label:18s} k={k:2d}  LR {a:.3f} | LGBM({k}) {a_lgb:.3f} | LGBM(1047) {full_auc:.3f}'
            f' | LR recovers {par_rows[-1]["recovery_pct"]:.0f}%')
pd.DataFrame(par_rows).to_csv(os.path.join(OUT, 'parsimony_floor.csv'), index=False)

log('STAGE A complete (items 1-5); external cohort next')

# ============================ the external cohort (ZZU) ============================
log('building the external cohort')
bundle = joblib.load(os.path.join(HERE, 'data', 'agez_transform.joblib'))
ext = pd.read_csv(os.path.join(HERE, 'data', 'ecg_batch_analysis_14190_v4_E3.csv'), low_memory=False)
feat = build_feature_matrix(ext); feat, _ = clean_implausible(feat)
feat['_ECG_ID'] = ext['filename'].astype(str); feat['_error'] = ext['error']
lab = pd.read_excel(os.path.join(HERE, 'data', 'ZZU_external_validation_clinical_textbook.xlsx'), 'all_ecgs')
keep = ['ECG_ID', 'is_primary_ecg', 'is_12lead', 'age_months', 'sex_M',
        'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
df = feat.merge(lab[keep].rename(columns={'sex_M': 'zzu_sex_M'}),
                left_on='_ECG_ID', right_on='ECG_ID', how='left')
err_ok = df['_error'].isna() | (df['_error'].astype(str).str.strip() == '') | (df['_error'].astype(str) == 'nan')
C = df[(df['is_primary_ecg'] == 1) & (df['is_12lead'] == 1) & err_ok].copy()
C['sex_M'] = C['zzu_sex_M'].fillna(C['sex_M']).fillna(0).astype(int)
Xe = apply_agez(C, C['age_months'].values, bundle)
Xe.insert(0, 'age_months', C['age_months'].values)
Xe = Xe[bundle['model_feat']]

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
# the manuscript's ECG-manifesting lists, copied verbatim from 10_manifesting_three_estimators.py
MANIFESTING = {
    'y_LV': ['dx_DCM', 'dx_HCM', 'dx_cardiomyopathy_other', 'dx_aortic_stenosis', 'dx_coarctation'],
    'y_RV': ['dx_ToF', 'dx_pulm_stenosis', 'dx_pulm_hypertension'],
    'y_RA': ['dx_tricuspid_disease', 'dx_pulm_hypertension', 'dx_complex_chd'],
    'y_LA': ['dx_mitral_disease', 'dx_AVSD', 'dx_complex_chd'],
    'y_any': ['dx_complex_chd', 'dx_pulm_hypertension', 'dx_ToF'],
}
log(f'  external cohort {len(C)}; disease-free {int(free.sum())}; non-shunt cases {int(nonshunt_case.sum())}')

# ============================== item 6: decision curves ==============================
log('item 6: decision curves for LightGBM')
THR = np.arange(0.01, 0.51, 0.01)
def net_benefit(y, p, thresholds):
    """Vickers & Elkin net benefit: (TP - FP * pt/(1-pt)) / N, with treat-all and treat-none."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); N = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        yh = (p >= pt).astype(int)
        tpn = int(((yh == 1) & (y == 1)).sum()); fpn = int(((yh == 1) & (y == 0)).sum())
        w = pt / (1 - pt)
        rows.append(dict(threshold=round(float(pt), 3),
                         net_benefit_model=round(tpn / N - fpn / N * w, 5),
                         net_benefit_treat_all=round(float(prev - (1 - prev) * w), 5),
                         net_benefit_treat_none=0.0))
    return pd.DataFrame(rows)

dca_frames, ext_pred = [], {}
for ycol, label in OUTCOMES:
    m = LGB(label).fit(tr[FEAT].values, tr[ycol].values)
    p_ext = m.predict_proba(Xe.values)[:, 1]
    ext_pred[ycol] = p_ext
    yext = Cx[ycol].values
    for cohort, y, p in [('internal held-out', te[ycol].values, tp[f'{ycol}__LightGBM'].values),
                         ('external, all ICD+', yext, p_ext)]:
        nb = net_benefit(y, p, THR); nb['outcome'] = label; nb['cohort'] = cohort
        nb['estimator'] = 'LightGBM'
        dca_frames.append(nb)
    # the useful range: thresholds where the model beats both default strategies
    for cohort, y, p in [('internal held-out', te[ycol].values, tp[f'{ycol}__LightGBM'].values),
                         ('external, all ICD+', yext, p_ext)]:
        nb = net_benefit(y, p, THR)
        better = nb[(nb.net_benefit_model > nb.net_benefit_treat_all) &
                    (nb.net_benefit_model > nb.net_benefit_treat_none)]
        log(f'  {label:18s} {cohort:20s} useful range '
            f'{("%.2f-%.2f" % (better.threshold.min(), better.threshold.max())) if len(better) else "none"}')
pd.concat(dca_frames).to_csv(os.path.join(OUT, 'decision_curves_lightgbm.csv'), index=False)
pd.DataFrame({'ECG_ID': Cx['ECG_ID'].values,
              **{f'{y}__LightGBM': ext_pred[y] for y, _ in OUTCOMES}}).to_csv(
    os.path.join(OUT, 'external_predictions_lightgbm.csv'), index=False)

# ===================== item 7: transportable-feature retrain =====================
log('item 7: transportable features (leak-safe halves, seed 42)')
CUT = 0.55
def uauc(x, y):
    """Univariate ROC-AUC by rank, NaN-safe, computed for all features at once."""
    ok = ~np.isnan(x)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2: return np.nan
    return roc_auc_score(y[ok], x[ok])

# one patient-level split of the external cohort, stratified by disease status
idx_sel, idx_rep = train_test_split(np.arange(len(Cx)), test_size=0.5, random_state=SEED,
                                    stratify=Cx['y_any'].values)
sel_mask = np.zeros(len(Cx), bool); sel_mask[idx_sel] = True
log(f'  selection half {int(sel_mask.sum())}, reporting half {int((~sel_mask).sum())}')

tr_arr = tr[FEAT].values
Xe_arr = Xe.values
trans_rows, fam_rows = [], []
for ycol, label in OUTCOMES:
    ytr = tr[ycol].values
    ns_case = nonshunt_case if ycol == 'y_any' else (nonshunt_case & (Cx[ycol].values == 1))
    keep_ns = ns_case | free
    y_ns = ns_case[keep_ns].astype(int)
    # univariate AUC internally (train) and in the external SELECTION half
    sel_ns = keep_ns & sel_mask
    rep_ns = keep_ns & ~sel_mask
    ui, us = {}, {}
    for j, f in enumerate(FEAT):
        ui[f] = uauc(tr_arr[:, j], ytr)
        ys = ns_case[sel_ns].astype(int)
        us[f] = uauc(Xe_arr[sel_ns, j], ys)
    ui = pd.Series(ui); us = pd.Series(us)
    # a feature discriminates in a direction; take distance from chance in both cohorts
    di = (ui - 0.5).abs() + 0.5
    dsel = (us - 0.5).abs() + 0.5
    transportable = [f for f in FEAT if pd.notna(di[f]) and pd.notna(dsel[f])
                     and min(di[f], dsel[f]) >= CUT]
    pd.DataFrame({'feature': FEAT, 'internal_train_uauc': [ui[f] for f in FEAT],
                  'external_selection_uauc': [us[f] for f in FEAT],
                  'transportable': [f in set(transportable) for f in FEAT]}).to_csv(
        os.path.join(OUT, f'univariate_transportability_{ycol}.csv'), index=False)

    # full model vs model retrained on the transportable subset, both reported in the
    # REPORTING half only
    y_rep = ns_case[rep_ns].astype(int)
    full = LGB(label).fit(tr_arr, ytr)
    p_rep_full = full.predict_proba(Xe_arr[rep_ns])[:, 1]
    a_full_rep = roc_auc_score(y_rep, p_rep_full)
    a_full_int = float(roc_auc_score(te[ycol].values, tp[f'{ycol}__LightGBM'].values))

    if len(transportable) >= 10:
        cols = [FEAT.index(f) for f in transportable]
        sub_m = LGB(label).fit(tr_arr[:, cols], ytr)
        a_sub_rep = roc_auc_score(y_rep, sub_m.predict_proba(Xe_arr[rep_ns][:, cols])[:, 1])
        a_sub_int = roc_auc_score(te[ycol].values, sub_m.predict_proba(te[FEAT].values[:, cols])[:, 1])
        # the same contrast against the ECG-manifesting definition
        les = [c for c in MANIFESTING[ycol] if c in Cx.columns]
        mani = (Cx[les].sum(axis=1) > 0).values
        keep_m = (mani | free) & ~sel_mask
        y_m = mani[keep_m].astype(int)
        a_full_man = roc_auc_score(y_m, full.predict_proba(Xe_arr[keep_m])[:, 1])
        a_sub_man = roc_auc_score(y_m, sub_m.predict_proba(Xe_arr[keep_m][:, cols])[:, 1])
    else:
        a_sub_rep = a_sub_int = a_full_man = a_sub_man = np.nan
    trans_rows.append(dict(outcome=label, n_transportable=len(transportable),
                           n_candidate=len(FEAT),
                           internal_full=round(a_full_int, 4),
                           internal_transportable=round(float(a_sub_int), 4),
                           external_nonshunt_reporting_full=round(float(a_full_rep), 4),
                           external_nonshunt_reporting_transportable=round(float(a_sub_rep), 4),
                           external_manifesting_reporting_full=round(float(a_full_man), 4),
                           external_manifesting_reporting_transportable=round(float(a_sub_man), 4),
                           n_reporting=int(rep_ns.sum()), events_reporting=int(y_rep.sum())))
    log(f'  {label:18s} {len(transportable):4d}/{len(FEAT)} transportable | '
        f'int {a_full_int:.3f}->{a_sub_int:.3f} | ext non-shunt {a_full_rep:.3f}->{a_sub_rep:.3f}')
pd.DataFrame(trans_rows).to_csv(os.path.join(OUT, 'transportable_retrain_lightgbm.csv'), index=False)

log('ALL ITEMS COMPLETE -> ' + OUT)
for f in sorted(os.listdir(OUT)): print('   ', f)
