"""The internal-vs-external demographics table, from the cohorts as they are actually scored.

`demographics_internal_vs_external.csv` had been written inline in June 2026 and never
regenerated, so it still reported a tested external cohort of 10,337 after the cohort had
become 10,352 — small enough to pass unread, which is exactly why it survived. It is a
script now, and it reads the same `predictions.csv` the external metrics come from, so the
row counts cannot drift apart from the AUCs beside them again.

Output: results/external_validation/demographics_internal_vs_external.csv
Usage:  python3 demographics_table.py
"""
import os, sys
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE, 'code'))
from engine_provenance import engine_string
RES = os.path.join(BASE, 'results', 'external_validation')

print('engine:', engine_string())


def row(name, age, male, yany):
    age = pd.to_numeric(pd.Series(age), errors='coerce').dropna()
    q1, q3 = age.quantile(0.25), age.quantile(0.75)
    return dict(cohort=name, n=len(age),
                age_mo_median=int(round(age.median())),
                age_mo_IQR=f'[{q1:.0f}-{q3:.0f}]',
                pct_under_3y=f'{100 * (age < 36).mean():.0f}%',
                pct_male=f'{100 * np.mean(male):.0f}%',
                prev_any=f'{100 * np.mean(yany):.0f}%')


ds = pd.read_csv(os.path.join(BASE, 'results', 'childheart_modeling_dataset_agez.csv'))
man = pd.read_csv(os.path.join(BASE, 'Data', '[OPEN] Crianças Qualquer doença - Pacientes.csv'), dtype=str)
man['folio'] = man['Folio'].astype(str).str.strip()
sexm = man.drop_duplicates('folio').set_index('folio')['Sexo'].astype(str).str.upper().str[0].eq('M')
internal = row('CHILDHEART (internal)', ds['age_months'],
               ds['folio'].astype(str).map(sexm).fillna(False).astype(int), ds['y_any'])

pred = pd.read_csv(os.path.join(RES, 'textbook', 'predictions.csv'))
external = row('ZZU (external, tested)', pred['age_months'], pred['sex_M'], pred['y_any'])

out = pd.DataFrame([internal, external])
out.to_csv(os.path.join(RES, 'demographics_internal_vs_external.csv'), index=False)
print(out.to_string(index=False))
print('\nsaved', os.path.join(RES, 'demographics_internal_vs_external.csv'))
