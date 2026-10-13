"""Figure 2: the reference standard, and what changes when it changes.

Panel a names what each external case definition counts as a case; panel b is the original
matplotlib output embedded as vector (make_fig2_panelb.py), built at its slot width so it is
placed at scale 1.0 and never resized. Participant counts live in Figure 1 and are deliberately
not repeated here.

    python3 make_fig2.py        ->  Fig2.pdf / .svg / .png / .tiff
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SCIFIG_PATH", ""))  # in-house figure toolkit (not public)
from scifig import SciFigure
import nums as N

W = 174.0
H_A, H_B = 33.0, 78.0
f = SciFigure(W, H_A + H_B + 2)

# ------------------------------------------------------------------ panel a
f.panel_letter("a", 0, 0)
f.band(6, W - 2, 3.2, "What the external reference standard counts as a case")

_m = N.mani_row('ANY disease')
_e = N.ext_row('ANY disease')
yb = 13.5
MARG, GAP = 6.0, 5.0
colw = (W - 2 * MARG - 2 * GAP) / 3
cx = [MARG + colw / 2 + i * (colw + GAP) for i in range(3)]

f.nbox("all", cx[0], yb, colw,
       "All ICD-10 codes\nany cardiac code, including isolated\nshunts that need not load a chamber",
       f"{int(_e['n_ext_events']):,} cases", "", fill="box")
f.nbox("ns", cx[1], yb, colw,
       "Non-shunt, CHILDHEART-like\nisolated atrial or ventricular septal\ndefect and patent ductus excluded",
       "724 cases", "", fill="eval")
f.nbox("man", cx[2], yb, colw,
       "ECG-manifesting\nlesions that load or injure a chamber: complex\ndisease, pulmonary hypertension, tetralogy of Fallot",
       f"{int(_m['n_manifesting'])} cases", "", fill="deriv")

f.caption(W / 2, yb + 13.0,
          ["The same frozen models are applied under all three definitions; the comparison group is the "
           f"{int(_m['n_disease_free']):,} children with no cardiac code.",
           "Only what counts as a case changes from left to right."])

# ------------------------------------------------------------------ panel b
ytop = H_A
info = f.embed_plot(os.path.join(HERE, "Fig2_panelb.pdf"), 0, ytop + 1.5, w=W)
f.panel_letter("b", 0, ytop + 0.5)   # after embed_plot: the plot paints an opaque background
print("embedded plot scale:", round(info.get("scale", float("nan")), 3))

f.save(os.path.join(HERE, "Fig2"))
print("wrote Fig2")
