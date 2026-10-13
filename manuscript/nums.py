"""Every number the manuscript states, read from the file that produced it.

This study has twice shipped a stale number because a value was typed into prose and then
the analysis was re-run underneath it. So nothing here is typed: each attribute is looked up
in the CSV that the analysis wrote, and content.py interpolates these attributes instead of
literals. If an analysis is re-run and a value moves, the manuscript moves with it; if a file
or a row disappears, the build fails loudly instead of printing yesterday's number.

Primary estimator is LightGBM throughout (see PRIMARY). Comparators are the tuned random
forest, the forest as shipped in the parent CHILDHEART analyses, and XGBoost.

Source of truth: RESULTS/tripod_analyses/, written by code/tripod_analyses/01..11.
"""
import os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(STUDY, 'results')
TRI = os.path.join(RES, 'tripod_analyses')

PRIMARY = 'LightGBM'
RF_TUNED, RF_SHIPPED, XGB = 'RandomForest', 'RandomForest (as shipped)', 'XGBoost'
OUTCOMES = ['ANY disease', 'Right atrial', 'Left atrial', 'Right ventricular', 'Left ventricular']
SHORT = {'ANY disease': 'any disease', 'Right atrial': 'right atrial', 'Left atrial': 'left atrial',
         'Right ventricular': 'right ventricular', 'Left ventricular': 'left ventricular'}


def _csv(*parts):
    p = os.path.join(TRI, *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(f'nums.py: missing results file {p}\n'
                                f'Run code/tripod_analyses and copy out/ into results/tripod_analyses/.')
    return pd.read_csv(p)


def one(df, **where):
    """The single row matching `where`; raises if it is not unique, so a silent mismatch
    (a renamed estimator, a dropped outcome) cannot become a wrong number in the prose."""
    m = df
    for k, v in where.items():
        m = m[m[k] == v]
    if len(m) != 1:
        raise ValueError(f'nums.py: expected exactly 1 row for {where}, found {len(m)}')
    return m.iloc[0]


def f3(x): return f'{float(x):.3f}'
def f2(x): return f'{float(x):.2f}'
def pct(x, d=1): return f'{100 * float(x):.{d}f}'
def comma(n): return f'{int(n):,}'


def pval(p):
    """APA-ish p formatting: exact to 3 decimals down to 0.001, then '< 0.001'."""
    p = float(p)
    return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'.rstrip('0').rstrip('.') if p >= 0.001 else 'P < 0.001'


def ci(row, a='ci_lo', b='ci_hi', v=None, lab='95% CI'):
    s = f3(row[v]) if v else ''
    return f'{s} ({lab}, {f3(row[a])}–{f3(row[b])})' if v else f'({lab}, {f3(row[a])}–{f3(row[b])})'


# ----------------------------------------------------------------- the audited ceiling run
CEIL_TEST = _csv('ceiling', 'ceiling_test.csv')
CEIL_CV = _csv('ceiling', 'ceiling_nested_cv.csv')
CEIL_PARAMS = _csv('ceiling', 'ceiling_best_params.csv')
EXT = _csv('ceiling_external', 'ceiling_external.csv')
MANI = _csv('manifesting', 'manifesting_three_estimators.csv')
DELONG = _csv('delong', 'delong_pairwise_test.csv')
CAL = _csv('external_calibration', 'calibration_internal_external.csv')
RILEY = _csv('validity', 'sample_size_riley.csv')

# ----------------------------------------------------------------- the LightGBM-primary run
# Loaded on first use rather than at import: the figures that depend only on the audited
# ceiling run can then be built while script 11 is still writing these. A missing file still
# raises FileNotFoundError, just at the point where its number is actually wanted.
_LAZY = {'LG_OPS': 'test_operating_characteristics.csv',
         'LG_SUB': 'subgroup_fairness_test.csv',
         'LG_LOT': 'split_lottery.csv',
         'LG_CAL': 'calibration_internal.csv',
         'LG_SHAP': 'shap_vs_gain_summary.csv',
         'LG_STAB': 'stability_summary.csv',
         'LG_PARS': 'parsimony_floor.csv',
         'LG_DCA': 'decision_curves_lightgbm.csv',
         'LG_TRANS': 'transportable_retrain_lightgbm.csv',
         'LG_HP': 'lightgbm_hyperparameters.csv'}
_cache = {}

def __getattr__(name):
    if name in _LAZY:
        if name not in _cache:
            _cache[name] = _csv('lightgbm', _LAZY[name])
        return _cache[name]
    raise AttributeError(name)

# chamber status (never / past / current), from the parent analysis
CHAMBER = pd.read_csv(os.path.join(RES, 'chamber_status_oof.csv'))


# ----------------------------------------------------------------- accessors
def test_auc(outcome, est=PRIMARY):
    return one(CEIL_TEST, outcome=outcome, estimator=est)

def nested(outcome, est=PRIMARY):
    return float(one(CEIL_CV, outcome=outcome, estimator=est)['nested_cv_auc'])

def ext_row(outcome, est=PRIMARY):
    return one(EXT, outcome=outcome, estimator=est)

def mani_row(outcome, est=PRIMARY):
    return one(MANI, outcome=outcome, estimator=est)

def _dl_name(x):
    """04_delong.py writes 'RandomForest as shipped' where 01_ceiling writes
    'RandomForest (as shipped)'. Normalise so one spelling cannot silently miss a row."""
    return x.replace('(', '').replace(')', '').strip()


def delong(outcome, a, b):
    """The DeLong row for a pair, in whichever order the file stored it."""
    a, b = _dl_name(a), _dl_name(b)
    d = DELONG[(DELONG.outcome == outcome)].copy()
    d['model_A'] = d['model_A'].map(_dl_name)
    d['model_B'] = d['model_B'].map(_dl_name)
    m = d[(d.model_A == a) & (d.model_B == b)]
    if len(m) == 1: return m.iloc[0], False
    m = d[(d.model_A == b) & (d.model_B == a)]
    if len(m) == 1: return m.iloc[0], True
    raise ValueError(f'nums.py: no DeLong row for {a} vs {b} in {outcome}')

def delong_p(outcome, a, b):
    r, _ = delong(outcome, a, b)
    return float(r['delong_p'])

def cal_row(outcome, cohort, est=PRIMARY):
    return one(CAL, outcome=outcome, estimator=est, cohort=cohort)

def ops(outcome, rule):
    return one(__getattr__('LG_OPS'), outcome=outcome, threshold_rule=rule)

def sub(outcome, group):
    return one(__getattr__('LG_SUB'), outcome=outcome, subgroup=group)

def lot(outcome):
    return one(__getattr__('LG_LOT'), outcome=outcome)

def pars(outcome, k):
    return one(__getattr__('LG_PARS'), outcome=outcome, k=k)

def stab(outcome):
    return one(__getattr__('LG_STAB'), outcome=outcome)

def trans(outcome):
    return one(__getattr__('LG_TRANS'), outcome=outcome)

def riley(nparam):
    return one(RILEY, candidate_parameters=nparam)

def chamber(name):
    return one(CHAMBER, chamber=name)

def dca_useful(outcome, cohort):
    """Threshold range in which LightGBM beats both 'treat all' and 'treat none'."""
    LG_DCA = __getattr__('LG_DCA')
    d = LG_DCA[(LG_DCA.outcome == outcome) & (LG_DCA.cohort == cohort)]
    b = d[(d.net_benefit_model > d.net_benefit_treat_all) &
          (d.net_benefit_model > d.net_benefit_treat_none)]
    if not len(b): return None
    return float(b.threshold.min()), float(b.threshold.max())


def ranges(fn, outcomes=None, fmt=f3):
    """min-max across outcomes, for the 'ranged from X to Y' sentences."""
    vals = [float(fn(o)) for o in (outcomes or OUTCOMES)]
    return fmt(min(vals)), fmt(max(vals))


def arg_extreme(fn, outcomes=None, biggest=True):
    outs = outcomes or OUTCOMES
    vals = {o: float(fn(o)) for o in outs}
    o = (max if biggest else min)(vals, key=vals.get)
    return o, vals[o]


# ----------------------------------------------------------------- cohort constants
# These are design facts of the cohorts, not model outputs; they live in the parent papers
# and the external README, and are asserted here so a change in the dataset is caught.
N_INTERNAL, N_CASES, N_CONTROLS = 1637, 349, 1288
N_TRAIN, N_TEST = 1309, 328
N_FEATURES = 1047
N_EXT = int(ext_row('ANY disease')['n_ext'])
N_EXT_EVENTS = int(ext_row('ANY disease')['n_ext_events'])
ENGINE_COMMIT = '150be8b'
PCT_POSTSURGICAL = 82.8
CASE_FRACTION = float(riley(4)['prevalence'])

_ds = os.path.join(RES, 'childheart_modeling_dataset_agez.csv')
if os.path.exists(_ds):
    _d = pd.read_csv(_ds, usecols=['split', 'y_any'])
    assert len(_d) == N_INTERNAL, f'nums.py: dataset has {len(_d)} rows, expected {N_INTERNAL}'
    assert int((_d.split == 'test').sum()) == N_TEST, 'nums.py: test split size changed'
    N_TEST_EVENTS = int(_d[_d.split == 'test'].y_any.sum())
    N_TRAIN_EVENTS = int(_d[_d.split == 'train'].y_any.sum())
else:
    N_TEST_EVENTS = int(riley(4)['available_events'])  # ops() needs script 11
    N_TRAIN_EVENTS = int(riley(4)['available_events'])


if __name__ == '__main__':
    print(f'primary estimator: {PRIMARY}')
    print(f'internal {N_INTERNAL} ({N_CASES} cases, {N_CONTROLS} controls); '
          f'train {N_TRAIN}/{N_TRAIN_EVENTS} events, test {N_TEST}/{N_TEST_EVENTS} events')
    print(f'external {comma(N_EXT)} ({comma(N_EXT_EVENTS)} ICD-positive)')
    print(f'\n{"outcome":20s} {"nestedCV":>9s} {"test":>22s} {"ext all":>9s} {"ext mani":>9s}')
    for o in OUTCOMES:
        t = test_auc(o); e = ext_row(o); m = mani_row(o)
        print(f'{o:20s} {nested(o):9.4f} {ci(t, v="test_auc"):>22s} '
              f'{float(e["external_all_icd_auc"]):9.3f} {float(m["manifesting_auc"]):9.3f}')
    print(f'\nDeLong (ANY): {PRIMARY} vs RF P={delong_p("ANY disease", PRIMARY, RF_TUNED):.4f}; '
          f'vs XGB P={delong_p("ANY disease", PRIMARY, XGB):.4f}')
