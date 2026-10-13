"""Shared feature construction + implausible-value cleaning.
Imported by BOTH the internal CHILDHEART pipeline (prep_dataset.py) and the external
ZZU validation (external_validation/run_external.py) so the methodology is byte-for-byte
identical on both cohorts.
"""
import pandas as pd, numpy as np

# Minnesota T / ST code -> ordinal severity
TMAP = {'5-0': 0, '5-4': 1, '5-3': 1, '5-2': 2, '5-1': 3}
SMAP = {'4-0': 0, '4-3': 1, '4-2': 2, '4-1': 3}

# Physiologic plausibility bounds for global intervals/durations (ms) and HR (bpm)
GLOBAL_BOUNDS = {'RR_ms': (150, 2500), 'HR_bpm': (25, 320), 'PR_ms': (50, 400),
    'P_duration_ms': (10, 220), 'QRS_duration_ms': (30, 250), 'QT_ms': (150, 800),
    'QTc_ms': (200, 700), 'TpTe_max_ms': (0, 320), 'TpTe_mean_ms': (0, 320)}


def build_feature_matrix(au):
    """Replicate the batch->feature construction: numeric cols + sex_M + Minnesota ordinals.
    (Column selection / imputation are done by the caller.)"""
    feat = pd.DataFrame(index=au.index)
    num = au.select_dtypes(include=[np.number]).columns
    feat[num] = au[num]
    feat['sex_M'] = (au['sex'].astype(str).str.upper().str[0] == 'M').astype(int)
    for c in au.columns:
        if c.endswith('_T_minnesota'):  feat[c + '_ord'] = au[c].map(TMAP)
        if c.endswith('_ST_minnesota'): feat[c + '_ord'] = au[c].map(SMAP)
    return feat


def clean_implausible(feat):
    """Null physiologically impossible MEASURED values (set NaN; caller median-imputes).
    The VALUE is removed, the patient is kept. Returns (feat_in_place, report_df).
    Rules: global interval bounds above; component durations 0-300 (no negatives);
    intrinsicoid 0-150; Tpe 0-320; |amplitude| <= 8 mV (negative T kept = inversion)."""
    report = []
    def _nz(col, mask):
        m = np.asarray(mask.fillna(False)) if hasattr(mask, 'fillna') else np.asarray(mask)
        n = int(m.sum())
        if n:
            feat.loc[m, col] = np.nan
            report.append({'feature': col, 'n_nulled': n, 'pct': round(100 * n / len(feat), 1)})
    for c, (lo, hi) in GLOBAL_BOUNDS.items():
        if c in feat:
            v = pd.to_numeric(feat[c], errors='coerce'); _nz(c, (v < lo) | (v > hi))
    for c in list(feat.columns):
        if c in GLOBAL_BOUNDS: continue
        v = pd.to_numeric(feat[c], errors='coerce')
        if c.endswith(('_dur_ms', '_duration_ms')):   _nz(c, (v < 0) | (v > 300))
        # T "intrinsicoid" = J -> T-peak time. The engine anchors the T wave at the J
        # point (ecg-vcg-viewer 9a5ed19, 2026-08-04), which shifted this family ~+80 ms;
        # the old 0-150 bound was written for the previous T-onset anchor and under the
        # new convention nulls ~73% of physiologic values. Bound it like Tpe (0-320).
        elif c.endswith('_T_intrinsicoid_ms'):        _nz(c, (v < 0) | (v > 320))
        elif c.endswith('_intrinsicoid_ms'):          _nz(c, (v < 0) | (v > 150))
        elif c.endswith('_Tpe_ms'):                   _nz(c, (v < 0) | (v > 320))
        elif c.endswith(('_amp_uV', '_amplitude_uV')): _nz(c, v.abs() > 8000)
    rep = pd.DataFrame(report).sort_values('n_nulled', ascending=False) if report else pd.DataFrame(columns=['feature','n_nulled','pct'])
    return feat, rep


def apply_agez(raw_df, age_months, bundle):
    """Apply a FITTED age-z transform (bundle from build_agez_dataset.py) to any cohort.
    Continuous features -> age-specific z-score via the saved TRAIN-HEALTHY norms
    (linear-interpolated across bin centers, clipped +-8); raw binary/ordinal kept.
    Missing values imputed with the saved TRAIN medians. Returns a DataFrame with the
    exact model feature columns/order (bundle['FEAT']). USE THE SAME NORMS — never refit."""
    cont, raw, FEAT = bundle['cont'], bundle['raw'], bundle['FEAT']
    centers, means, sds = bundle['centers'], bundle['means'], bundle['sds']
    gmean, gsd, med = bundle['gmean'], bundle['gsd'], bundle['train_medians']
    n = len(raw_df); ag = np.asarray(age_months, float)
    out = pd.DataFrame(index=raw_df.index)
    for j, c in enumerate(cont):
        col = pd.to_numeric(raw_df[c], errors='coerce') if c in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
        col = col.fillna(med[c]).values
        m = means[j].copy(); s = sds[j].copy(); ok = ~np.isnan(m)
        if ok.sum() >= 2:
            mm = np.interp(ag, centers[ok], m[ok]); ss = np.interp(ag, centers[ok], np.nan_to_num(s[ok], nan=gsd[j]))
        else:
            mm = np.full(n, gmean[j]); ss = np.full(n, gsd[j])
        ss = np.where(ss < 1e-6, gsd[j], ss)
        out[c] = np.clip((col - mm) / ss, -8, 8)
    for c in raw:
        col = pd.to_numeric(raw_df[c], errors='coerce') if c in raw_df.columns else pd.Series(np.nan, index=raw_df.index)
        out[c] = col.fillna(med[c]).values
    return out[FEAT]
