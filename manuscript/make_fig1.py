"""Figure 1: flow of participants through both cohorts (STARD item 19).

Drawn with scifig so it is the same visual system as Figure 2 — the same blue header bands,
square flow boxes, italic 'n =' lines and 5 pt notes — rather than a generic CONSORT diagram.

Two columns: a, the derivation cohort, where cases and the comparison group were recruited
separately and only the cases were imaged; b, the external cohort, where one frozen set of
models meets three different case definitions. Every count is read from the label workbook and
the result files by check_fig1_counts.py; the exclusions sum to the difference between the
boxes they sit between.

    python3 make_fig1.py   ->  Fig1.pdf / .svg / .png / .tiff
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SCIFIG_PATH", ""))  # in-house figure toolkit (not public)
from scifig import SciFigure
import nums as N

W, H = 174.0, 178.0
f = SciFigure(W, H)
L, Rt = 3.0, W - 3.0
FX, FW = 46.0, 78.0          # flow-box centre and width
EX, EW = 102.0, 69.0         # exclusion / side-box left edge and width

_m = N.mani_row('ANY disease')
_e = N.ext_row('ANY disease')

# ===================================================== a: derivation cohort
f.panel_letter("a", 0, 0)
f.band(L, Rt, 3.0, "Derivation cohort: CHILDHEART, Instituto Dante Pazzanese, Brazil", icon="heart")

f.nbox("scr", FX, 15.0, FW,
       "Consecutive admissions aged 0-14 years screened\nfor structural heart disease", "376")
f.nbox("cas", FX, 34.0, FW,
       "Structural disease confirmed; chamber involvement\nadjudicated by three pediatric cardiologists blinded\nto the ECG, from echocardiography and clinical,\nimaging and surgical records",
       str(N.N_CASES), fill="deriv")
f.link("scr", "cas", ["Excluded (n = 27)",
                      "Isolated arrhythmia, no structural disease",
                      "Ventricular pacing",
                      "Unavailable or illegible tracing"], ex_left=EX, ex_w=EW)

f.nbox("ctl", EX + EW / 2, 38.0, EW,
       "Comparison group, recruited separately\ntele-electrocardiography in general care, with no\nclinical indication of heart disease, NOT imaged",
       N.comma(N.N_CONTROLS), fill="box")

f.nbox("coh", 87.0, 57.0, FW + 14,
       "Derivation cohort analysed, one 12-lead ECG per child",
       N.comma(N.N_INTERNAL),
       f"cases {N.pct(N.CASE_FRACTION)}% by design, not by observed prevalence", fill="deriv")
f.link("cas", "coh")
f.link("ctl", "coh")

f.nbox("trn", 55.0, 77.0, 56.0, "Training",
       N.comma(N.N_TRAIN), f"{N.N_TRAIN_EVENTS} with structural disease", fill="box")
f.nbox("tst", 119.0, 77.0, 56.0, "Held-out test",
       str(N.N_TEST), f"{N.N_TEST_EVENTS} with structural disease", fill="eval")
f.split_to("coh", ["trn", "tst"], 68.0, "Fixed stratified 80/20 split")
f.caption(87.0, 86.0,
          ["Cases and the comparison group were recruited separately, from different care settings, and only the cases were imaged",
           "(PROBAST participant domain: high risk). The case fraction is therefore fixed by design and is not an estimate of prevalence."])

# ===================================================== b: external cohort
Y = 96.0
f.panel_letter("b", 0, Y - 4.0)
f.band(L, Rt, Y, "External validation cohort: Zhengzhou University pediatric ECG database, China", icon="ecg_trace")

f.nbox("src", FX, Y + 12.0, FW,
       "Pediatric 12-lead ECG database\ntracings from 11,643 children", "14,190")
f.nbox("ext", FX, Y + 31.0, FW,
       "Primary 12-lead records analysed\nmodels applied frozen, no refitting",
       N.comma(N.N_EXT),
       f"{N.comma(N.N_EXT_EVENTS)} ICD-10 positive; {N.comma(int(_m['n_disease_free']))} disease-free",
       fill="eval")
f.link("src", "ext", ["Excluded (n = 3,838)",
                      "Nine-lead, lacking V2, V4 and V6: n = 1,856",
                      "Repeat tracing, not the child's first: n = 1,979",
                      "Signal-extraction failure: n = 3"], ex_left=EX, ex_w=EW)

cw = 52.0
f.nbox("d1", 32.0, Y + 51.0, cw, "All ICD-10 codes",
       N.comma(N.N_EXT_EVENTS), f"ROC-AUC {N.f3(_e['external_all_icd_auc'])}", fill="box")
f.nbox("d2", 87.0, Y + 51.0, cw, "Non-shunt, CHILDHEART-like",
       "724", f"ROC-AUC {N.f3(_e['external_nonshunt_auc'])}", fill="box")
f.nbox("d3", 142.0, Y + 51.0, cw, "ECG-manifesting lesions",
       str(int(_m['n_manifesting'])), f"ROC-AUC {N.f3(_m['manifesting_auc'])}", fill="deriv")
f.split_to("ext", ["d1", "d2", "d3"], Y + 42.0, "Three pre-specified case definitions, models unchanged")
f.caption(87.0, Y + 60.0,
          ["Only what counts as a case changes across the three definitions; discrimination is that of the primary model (LightGBM)",
           "against the disease-free children, and is shown for all estimators in Fig. 2."])

yk = Y + 70.0
f.legend_key(10.0, yk, "deriv", "Adjudicated, or physiology-anchored, reference standard")
f.legend_key(70.0, yk, "eval", "Set on which a reported estimate is computed")
f.legend_key(128.0, yk, "box", "Diagnosis-code label, or not imaged")

f.save(os.path.join(HERE, "Fig1"))
print("wrote Fig1")
