#!/usr/bin/env python3
"""One figure from strategy_test.py: the combined band (generator + theory + learning) of the test histogram by the
three strategies, relative to the central prediction (after the calculation is imposed):
  joint refit draws  S5, every source drawn at once and each draw re-solved (the exact band, MC_N draws)
  refit + quadrature the r8 bands: re-solved generator (S0) + re-solved theory (S4) + learning, in quadrature
  linear joint       the linear joint covariance at the box centre (LJ), one solve
  log-linear         log w linear in the inputs at the box centre, the multipliers re-solved per draw (LL, when present)

    python plot_strategy_test.py output/strategy_test_MIXGEOV3S9.json [out.pdf]
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figs_r8 import style, trim, window, TW, METHOD, GEN, BASE, REF, LEARN_TEXT

src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src.replace('.json', '.pdf')
d = json.load(open(src)); S = d['S5']['after']
B = [k for k in d['labels'] if k.startswith('bin_')]
edges = np.array([float(k.split('_')[1]) for k in B] + [float(B[-1].split('_')[2])])
y0 = np.array([d['central']['after'][k] for k in B])
rel = lambda r: 100*np.array([r[k] for k in B])/y0
style()
fig, ax = plt.subplots(figsize=(TW, 3.0))
window(ax, 0.05, 1/3)
ax.text(0.19, 57, 'moments imposed', ha='center', va='top', fontsize=7.5, color=REF)
r = rel(S['sd']['joint'])
ax.stairs(r, edges, baseline=-r, fill=True, color=BASE, alpha=0.45, lw=0, label='exact (joint refit)')
lines = [(rel(S['quadrature_r8']), METHOD, '-', 'refit + quadrature'),
         (rel(d['linear_joint']['box_centre']['band']['after']), GEN, (0, (4, 2)), 'linear')]
if 'LL' in d:
    lines.append((rel(d['LL']['band_resolved']), LEARN_TEXT, (0, (5, 1.5, 1, 1.5)), 'log-linear'))
for r, c, ls, lab in lines:
    ax.stairs(r, edges, baseline=None, color=c, ls=ls, lw=1.3, label=lab)
    ax.stairs(-r, edges, baseline=None, color=c, ls=ls, lw=1.3)
ax.axhline(0, color=REF, lw=0.5)
ax.set_xlim(0, 0.5); ax.set_ylim(-65, 65)
ax.set_xlabel(r'$\tau = 1-T$'); ax.set_ylabel(r'combined uncertainty (\%)')
ax.legend(loc='lower center', bbox_to_anchor=(0.38, 0.0), fontsize=7.5)
trim(ax)
fig.savefig(out); print('wrote', out)
