"""Single source of truth for WHICH extraction of the ECG features a script reads.

Both cohorts have now been re-extracted four times, and the failure mode is always
the same one: a script keeps opening the file it was written against while the rest
of the study moves on, so two halves of one analysis end up on two different
measurement engines with nothing in the numbers to say so. That has already
happened twice here — `SUMMARY_all_models_agez.csv` carried a June LV AUC after the
August rescore, and `validate_batch_vs_manual.py` / `generate_manuscript_figures.py`
were still reading the JUNE tables after the September migration to D2 claimed the
whole study sat on one engine.

    from batch_source import load_internal, load_external
    au  = load_internal(BASE)     # CHILDHEART 1637, DataFrame with a 'filename' column
    ext = load_external(BASE)     # ZZU 14,190

`CH_VINTAGE` selects the vintage in one place for every script:

    v5_D22            delineator D2.2, extractor E3+, contract M2 2026-10-13 (right-heart letter)
    v4_E3  (default)  delineator D2, extractor E3, contract M2   2026-09-02
    v3_D2             delineator D2, extractor E2                2026-09-01
    v2                delineator D1, engine 2026-08
    v1                engine 2026-06 (the original xlsx/csv pair)

A measurement is comparable with another only when the delineator version, the
measurement contract AND the extractor version all match — see
`ecg-vcg-viewer/js/extractorVersion.js`, which is where those three strings live and
where `engine_provenance.py` reads them from rather than repeating them here.
"""
import os
import pandas as pd

# vintage -> (internal file, external file, human label, engine commit that produced it)
#
# The commit is the one the TABLE was written by, which is not the commit checked out now:
# ecg-vcg-viewer keeps moving, and a report that prints today's HEAD claims a provenance the
# numbers do not have. v4_E3 was extracted at `e9494a5` and re-extracted from the raw captures
# at `150be8b` on 2026-09-09; the two agreed to within 1e-9 on every numeric cell of both
# cohorts (1,637 and 14,190 records, 1,107 columns), which is what makes `150be8b` the honest
# stamp for it -- the commits in between touched no measurement code.
VINTAGES = {
    # v5_D22: re-extracted on the Mac mini 2026-10-13 for the right-heart score letter only (the LV
    # paper stays on v4_E3). 600b72f = origin/main 41fef0f + ecg-vcg-viewer PR #272, which re-derives
    # S-after-R / QS depth / QRS morphology label on the final QRS window and exports them, plus the
    # M1.1 QRS lobes (js/qrsMorphology.js) and the QM2 class (js/qrsMorphClass.js).
    'v5_D22': ('ecg_batch_analysis_1637_v5_D22.csv', 'ecg_batch_analysis_14190_v5_D22.csv',
               'delineator D2.2, extractor E3 (+29 Sep QRS-shape columns), contract M2 (2026-10-13)', '600b72f'),
    'v4_E3': ('ecg_batch_analysis_1637_v4_E3.csv', 'ecg_batch_analysis_14190_v4_E3.csv',
              'delineator D2, extractor E3, contract M2 (2026-09-02)', '150be8b'),
    'v3_D2': ('ecg_batch_analysis_1637_v3_D2.csv', 'ecg_batch_analysis_14190_v3_D2.csv',
              'delineator D2, extractor E2 (2026-09-01)', 'ec3e2f6'),
    'v2':    ('ecg_batch_analysis_1637_v2.csv', 'ecg_batch_analysis_14190_v2.csv',
              'delineator D1, engine 2026-08', '?'),
    'v1':    ('ecg_batch_analysis_1637.xlsx', 'ecg_batch_analysis_14190.csv',
              'engine 2026-06', '?'),
}
DEFAULT = 'v4_E3'


def vintage():
    v = os.environ.get('CH_VINTAGE', DEFAULT)
    if v not in VINTAGES:
        raise SystemExit(f'CH_VINTAGE={v!r} unknown; choose from {list(VINTAGES)}')
    return v


def _load(base, which, verbose):
    v = vintage()
    fn = VINTAGES[v][which]
    path = os.path.join(base, 'Data', fn)
    if not os.path.exists(path):
        raise SystemExit(f'feature table missing for CH_VINTAGE={v}: {path}')
    df = pd.read_excel(path) if fn.endswith('.xlsx') else pd.read_csv(path, low_memory=False)
    if verbose:
        side = 'internal CHILDHEART' if which == 0 else 'external ZZU'
        print(f'{side} batch source: {fn} ({VINTAGES[v][2]}, {df.shape[1]} cols)')
    return df


def extraction_commit(v=None):
    """The ecg-vcg-viewer commit the selected feature tables were written by."""
    return VINTAGES[v or vintage()][3]


def load_internal(base, verbose=True):
    """CHILDHEART 1637 feature table for the selected vintage."""
    return _load(base, 0, verbose)


def load_external(base, verbose=True):
    """ZZU 14,190 feature table for the selected vintage."""
    return _load(base, 1, verbose)
