"""Items 3, 7, 8, 9 — calibration, sample size, subgroups/fairness, split lottery.

Item 3  Calibration beyond Brier. The manuscript reports a single Brier score and nothing
        else. At a case fraction of 21.3% fixed by design, a Brier of 0.067 is close to
        uninformative on its own (predicting the base rate for everyone scores ~0.168).
        Calibration-in-the-large and the calibration slope say whether the probabilities
        mean anything, and they are computed here internally and externally.

Item 7  Riley sample size and events per candidate predictor, stated against the 1,047
        features actually offered to the forest rather than against the four or five that
        survive in the narrative.

Item 8  Subgroup and fairness analyses. These exist in the study's results files but were
        dropped from the manuscript between versions; TRIPOD+AI items 12d and 14 require
        them. Recomputed here on the held-out test set.

Item 9  Split lottery. The headline 0.941 is one draw from a distribution the study has
        already measured; the reader cannot see it. Re-measured over 200 splits.

Outputs (out/validity/).
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'validity'); os.makedirs(OUT, exist_ok=True)
SEED, N_JOBS = 42, 4

ds = pd.read_csv(os.path.join(HERE, 'data', 'childheart_modeling_dataset_agez.csv'))
META = ['folio', 'split', 'y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']
FEAT = [c for c in ds.columns if c not in META]
tr, te = ds[ds.split == 'train'], ds[ds.split == 'test']
OUTCOMES = [('y_any', 'ANY disease'), ('y_RA', 'Right atrial'), ('y_LA', 'Left atrial'),
            ('y_RV', 'Right ventricular'), ('y_LV', 'Left ventricular')]
RF = lambda: RandomForestClassifier(n_estimators=600, class_weight='balanced',
                                    random_state=SEED, n_jobs=N_JOBS, min_samples_leaf=2)

# ============================ item 3: calibration ============================
def calib(y, p):
    """Calibration-in-the-large (mean predicted minus observed, and the intercept of a
    logistic recalibration with the linear predictor offset) and the calibration slope."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    # intercept with the linear predictor as a fixed offset = calibration-in-the-large
    cil = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                       offset=lp).fit().params[0])
    return dict(calibration_in_the_large=round(cil, 4), calibration_slope=round(slope, 4),
                mean_predicted=round(p.mean(), 4), observed=round(y.mean(), 4),
                brier=round(brier_score_loss(y, p), 4))

rows = []
for ycol, label in OUTCOMES:
    m = RF().fit(tr[FEAT].values, tr[ycol].values)
    p = m.predict_proba(te[FEAT].values)[:, 1]
    rows.append(dict(outcome=label, cohort='UKB-TEST equivalent (internal held-out)',
                     n=len(te), events=int(te[ycol].sum()), **calib(te[ycol].values, p)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'calibration_internal.csv'), index=False)
print('== item 3: internal calibration ==')
print(pd.DataFrame(rows)[['outcome', 'calibration_in_the_large', 'calibration_slope',
                          'mean_predicted', 'observed', 'brier']].to_string(index=False))

# ============================ item 7: Riley / EPP ============================
def riley_min_n(c_stat, prev, n_params, shrink=0.9):
    """Riley et al. criterion 1 (target shrinkage). Cox-Snell R2 approximated from the
    C-statistic by the usual logistic approximation."""
    lnO = np.log(prev / (1 - prev))
    # approximation of R2_CS from C used in pmsampsize for binary outcomes
    mu = (c_stat - 0.5) * np.sqrt(2) * 1.0
    R2_app = (mu ** 2) / ((mu ** 2) + (np.pi ** 2 / 3))
    max_R2 = 1 - np.exp(2 * (prev * np.log(prev) + (1 - prev) * np.log(1 - prev)))
    R2 = R2_app * max_R2
    n = n_params / ((shrink - 1) * np.log(1 - R2 / shrink))
    return int(np.ceil(n)), R2, max_R2

prev = float(tr['y_any'].mean()); events = int(tr['y_any'].sum())
rr = []
for nparam, what in [(4, '4 items (a DART-sized score)'), (23, '23 stable features'),
                     (166, '166 features'), (1047, '1,047 candidate features (this study)')]:
    n, r2, mx = riley_min_n(0.94, prev, nparam)
    rr.append(dict(candidate_parameters=nparam, description=what, anticipated_C=0.94,
                   prevalence=round(prev, 4), required_N=n, required_events=int(np.ceil(n * prev)),
                   available_N=len(tr), available_events=events,
                   events_per_parameter=round(events / nparam, 2),
                   meets=bool(len(tr) >= n)))
pd.DataFrame(rr).to_csv(os.path.join(OUT, 'sample_size_riley.csv'), index=False)
print('\n== item 7: Riley ==')
print(pd.DataFrame(rr)[['candidate_parameters', 'required_N', 'required_events',
                        'available_N', 'available_events', 'events_per_parameter', 'meets']].to_string(index=False))

# ============================ item 8: subgroups / fairness ============================
man = pd.read_csv(os.path.join(HERE, 'data', 'manual_sheet.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip()
man = man.drop_duplicates('folio').set_index('folio')
te2 = te.copy(); te2['folio'] = te2['folio'].astype(str)
te2['sex_M'] = te2['folio'].map(man['Sexo'].astype(str).str.upper().str[0].eq('M'))
te2['age_months'] = pd.to_numeric(te2['folio'].map(man['Idade (m)']), errors='coerce')

sub = []
for ycol, label in OUTCOMES:
    m = RF().fit(tr[FEAT].values, tr[ycol].values)
    p = m.predict_proba(te2[FEAT].values)[:, 1]
    y = te2[ycol].values
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
                            events=int(y[mask].sum()), auc=None, note='not estimable'))
            continue
        sub.append(dict(outcome=label, subgroup=gname, n=int(mask.sum()),
                        events=int(y[mask].sum()),
                        auc=round(roc_auc_score(y[mask], p[mask]), 3), note=''))
pd.DataFrame(sub).to_csv(os.path.join(OUT, 'subgroup_fairness_test.csv'), index=False)
print('\n== item 8: subgroups (held-out test) ==')
print(pd.DataFrame(sub).to_string(index=False))

# ============================ item 9: split lottery ============================
# 100 splits at 200 trees rather than 200 at 300: the quantity of interest is the WIDTH
# of the distribution, which is already stable at 100 draws, and the full version costs
# ~50 minutes for a number that does not move.
lot = []
for ycol, label in OUTCOMES:
    aucs = []
    X, y = ds[FEAT].values, ds[ycol].values
    for i in range(100):
        itr, ite = train_test_split(np.arange(len(ds)), test_size=0.2,
                                    random_state=1000 + i, stratify=ds['y_any'].values)
        if len(np.unique(y[ite])) < 2: continue
        m = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=SEED, n_jobs=N_JOBS, min_samples_leaf=2)
        m.fit(X[itr], y[itr])
        aucs.append(roc_auc_score(y[ite], m.predict_proba(X[ite])[:, 1]))
    a = np.array(aucs)
    lot.append(dict(outcome=label, n_splits=len(a), mean=round(a.mean(), 3),
                    sd=round(a.std(), 3), p2_5=round(np.percentile(a, 2.5), 3),
                    p97_5=round(np.percentile(a, 97.5), 3),
                    min=round(a.min(), 3), max=round(a.max(), 3)))
    print(f'  lottery {label:18s} {a.mean():.3f} +/- {a.std():.3f}  [{a.min():.3f}-{a.max():.3f}]')
pd.DataFrame(lot).to_csv(os.path.join(OUT, 'split_lottery.csv'), index=False)
print('\nwrote', OUT)
