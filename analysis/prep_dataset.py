"""Build the shared modeling dataset: batch ECG features + 5 chamber-disease labels
+ a fixed stratified 80/20 train/test split. All 5 random forests load THIS file so
features and the test patients are identical/comparable.

Feature construction + implausible-value cleaning live in common_features.py (shared
verbatim with the external ZZU validation). Labels are one-vs-rest: for each chamber,
y=1 iff that chamber's Doença col=='Sim'; everyone else (incl. other-chamber-diseased)
is a control. y_any = any disease."""
import pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
from common_features import build_feature_matrix, clean_implausible
from batch_source import load_internal
from engine_provenance import engine_string

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Data')
# Which extraction of the features this is built from lives in ONE place, batch_source.py,
# and the engine that produced it is READ from the engine rather than typed (see
# engine_provenance.py). CH_VINTAGE=v3_D2 reproduces the September run, v2/v1 the older ones.
print('engine:', engine_string())
au = load_internal(BASE)
au['folio'] = au['filename'].astype(str).str.extract(r'(\d+)\.json')
man = pd.read_csv(os.path.join(DATA, '[OPEN] Crianças Qualquer doença - Pacientes.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip()
mg = pd.merge(man[['folio', 'Qualquer patologia', 'Doença AD', 'Doença AE', 'Doença VD', 'Doença VE']], au, on='folio')

# ----- labels (one-vs-rest; controls = everyone without that chamber's disease) -----
lab = pd.DataFrame({'folio': mg['folio']})
lab['y_any'] = (mg['Qualquer patologia'] == 'Sim').astype(int)
lab['y_RA']  = (mg['Doença AD'] == 'Sim').astype(int)
lab['y_LA']  = (mg['Doença AE'] == 'Sim').astype(int)
lab['y_RV']  = (mg['Doença VD'] == 'Sim').astype(int)
lab['y_LV']  = (mg['Doença VE'] == 'Sim').astype(int)

# ----- features (shared builder) + implausible-value cleaning (shared) -----
# CRITICAL: build features from the MERGED frame `mg`, NOT from `au`. `au` is in the
# xlsx row order while the labels (`lab`) come from `mg` (manual-sheet/merge order).
# Building from `au` and concatenating positionally paired 77.5% of patients' ECG
# features with the WRONG patient's diagnosis (only 22.5% of rows happened to align),
# which silently degraded every model (true AUC ~0.92-0.96 collapsed to ~0.71-0.77).
# Fixed 2026-06-10. mg carries all batch columns, aligned 1:1 with the labels.
feat = build_feature_matrix(mg)
feat, rep = clean_implausible(feat)
if len(rep):
    rep.to_csv(os.path.join(BASE, 'results', 'implausible_clean_report.csv'), index=False)
    print('implausible values nulled: %d across %d features (-> median-imputed). report: results/implausible_clean_report.csv'
          % (int(rep.n_nulled.sum()), len(rep)))
    print(rep.head(8).to_string(index=False))

# drop near-empty (>80% missing) and constant columns, then median-impute
miss = feat.isna().mean(); feat = feat[miss[miss <= 0.8].index]
nun = feat.nunique(dropna=True); feat = feat[nun[nun > 1].index]
feat = feat.fillna(feat.median(numeric_only=True))
feat = feat.loc[:, feat.notna().all()]

ds = pd.concat([lab.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
# ----- fixed stratified 80/20 split (by any-disease, shared) -----
tr, te = train_test_split(ds.index, test_size=0.20, random_state=42, stratify=ds['y_any'])
ds['split'] = 'train'; ds.loc[te, 'split'] = 'test'

out = os.path.join(BASE, 'results', 'childheart_modeling_dataset.csv')
ds.to_csv(out, index=False)
fcols = list(feat.columns)
print('saved:', out)
print('rows:', len(ds), ' features:', len(fcols))
print('split: train', int((ds.split == "train").sum()), ' test', int((ds.split == "test").sum()))
print('positives per label (train/test):')
for y in ['y_any', 'y_RA', 'y_LA', 'y_RV', 'y_LV']:
    print(f'  {y}: train {int(ds[ds.split=="train"][y].sum())}  test {int(ds[ds.split=="test"][y].sum())}')
open(os.path.join(BASE, 'results', 'feature_list.txt'), 'w').write('\n'.join(fcols))
