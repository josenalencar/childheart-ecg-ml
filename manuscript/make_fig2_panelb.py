"""Figure 2, panel b: discrimination for any structural disease, by estimator and by stage.

Built at the slot size it will occupy in the composite (174 x 78 mm) in Arial at 6 pt, so
scifig's embed_plot places it at scale 1.0 and nothing is resized after the fact.

The panel makes one comparison visible: reading down the left plot, the four estimators sit
apart internally and move together externally; reading the right plot, the internal-to-external
difference is the same for all of them. Numbers come from nums.py, never typed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'sans-serif',
                            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                            'axes.linewidth': 0.25, 'xtick.major.width': 0.25,
                            'ytick.major.width': 0.25, 'font.size': 7.5})
import matplotlib.pyplot as plt
import numpy as np
import nums as N

MM = 1 / 25.4
W, H = 174 * MM, 78 * MM
# The forest as shipped in the parent analyses was never applied externally, so it has no
# row in the external files; it appears in Table 3 and in the text, not here.
EST = [(N.RF_TUNED, 'Random forest',       '#4B5563'),
       (N.XGB,      'XGBoost',             '#317EC2'),
       (N.PRIMARY,  'LightGBM (primary)',  '#009E73')]
O = 'ANY disease'

# stage -> (auc, lo, hi) per estimator; CIs exist for the test and external stages only
def stages(est):
    t = N.test_auc(O, est); e = N.ext_row(O, est); m = N.mani_row(O, est)
    return [('Nested cross-validation\n(training data only)', N.nested(O, est), None, None),
            ('Internal held-out test', float(t['test_auc']), float(t['ci_lo']), float(t['ci_hi'])),
            ('External, all ICD-10 codes', float(e['external_all_icd_auc']),
             float(e['ext_all_ci_lo']), float(e['ext_all_ci_hi'])),
            ('External, non-shunt', float(e['external_nonshunt_auc']),
             float(e['ext_ns_ci_lo']), float(e['ext_ns_ci_hi'])),
            ('External, ECG-manifesting', float(m['manifesting_auc']),
             float(m['ci_lo']), float(m['ci_hi']))]

LAB = [s[0] for s in stages(N.PRIMARY)]
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1], left=0.205, right=0.975,
                      top=0.855, bottom=0.185, wspace=0.26)
ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

off = np.linspace(-0.23, 0.23, len(EST))
for k, (est, lab, col) in enumerate(EST):
    S = stages(est)
    ys = np.arange(len(S))[::-1] + off[k]
    xs = [s[1] for s in S]
    for y, (_, a, lo, hi) in zip(ys, S):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=0.6, solid_capstyle='butt', zorder=2)
    ax.plot(xs, ys, 'o', ms=3.2, color=col, mec='white', mew=0.25, zorder=3, label=lab)

ax.set_yticks(np.arange(len(LAB))[::-1]); ax.set_yticklabels(LAB, fontsize=7.5)
ax.set_xlabel('ROC-AUC for any structural heart disease', fontsize=7.5)
ax.set_xlim(0.60, 1.03); ax.set_ylim(-0.6, len(LAB) - 0.4)
ax.axhline(2.5, color='#CBD2D9', lw=0.4, ls=(0, (2.5, 2)), zorder=1)
ax.annotate('below this line the models are unchanged;\nonly the reference standard changes',
            xy=(1.021, 2.46), fontsize=7, color='#6B7280', va='top', ha='right')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.tick_params(length=1.6, pad=1.4)
ax.grid(axis='x', color='#EEF1F4', lw=0.35, zorder=0); ax.set_axisbelow(True)

gaps = [float(N.ext_row(O, est)['internal_minus_external_all']) for est, _, _ in EST]
ys2 = np.arange(len(EST))[::-1]
for y, (est, lab, col), g in zip(ys2, EST, gaps):
    ax2.plot([0, g], [y, y], color=col, lw=1.6, solid_capstyle='butt')
    ax2.plot([g], [y], 'o', ms=3.2, color=col, mec='white', mew=0.25)
    ax2.annotate(f'{g:.3f}', (g, y), textcoords='offset points', xytext=(3.2, 0),
                 ha='left', va='center', fontsize=7.5, color=col)
ax2.set_yticks(ys2); ax2.set_yticklabels([lab for _, lab, _ in EST], fontsize=7.5)
ax2.set_xlim(0, max(gaps) * 1.38); ax2.set_ylim(-0.6, len(EST) - 0.4)
ax2.set_xlabel('Internal held-out minus\nexternal (all ICD-10)', fontsize=7.5)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)
ax2.tick_params(length=1.6, pad=1.4)
ax2.grid(axis='x', color='#EEF1F4', lw=0.35, zorder=0); ax2.set_axisbelow(True)
ax2.annotate(f'range {min(gaps):.3f}–{max(gaps):.3f}', xy=(0.03, -0.46),
             fontsize=7, color='#6B7280', ha='left', va='center')

h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.165, 0.995), ncol=4, frameon=False,
           fontsize=7.5, handletextpad=0.35, columnspacing=1.8, markerscale=1.1)
OUTDIR = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(OUTDIR, 'Fig2_panelb.pdf'))
print('wrote Fig2_panelb.pdf  (%.0f x %.0f mm)' % (W / MM, H / MM))
print('gaps:', ', '.join(f'{lab.split(" (")[0]} {g:.3f}' for (_, lab, _), g in zip(EST, gaps)))
