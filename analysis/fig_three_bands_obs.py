#!/usr/bin/env python3
"""Figure: the three uncertainties and their combination on the event-shape means, the window
mean, the window fraction, the mean charged multiplicity and the mean baryon count, in percent of
the prediction with the calculation imposed. The generator band before the calculation is drawn
as a grey tick, so the collapse of the constrained moment is visible next to the unconstrained
ones.

    python fig_three_bands_obs.py output/three_bands_MIXGEO_w.json figs/fig_three_bands_obs.pdf
"""
import sys, json, numpy as np, matplotlib.pyplot as plt
from money_style import setup, save, trim, BLUE, RED, GREY, INK, MUTED
setup(usetex=True)
js, out = sys.argv[1:3]
J = json.load(open(js)); c = J['central']; g = J['generator_band']
OBS = [('tau_win', r'$\langle\tau\rangle_{\rm win}$'), ('pwin', r'$P_{\rm win}$'), ('thrust', r'$\langle 1-T\rangle$'),
       ('B_total', r'$\langle B_{\rm tot}\rangle$'), ('rho_heavy', r'$\langle\rho_H\rangle$'), ('nch', r'$\langle n_{\rm ch}\rangle$'),
       ('nbaryon', r'$\langle n_{\rm b}\rangle$')]
keys = [k for k, _ in OBS]; labels = [l for _, l in OBS]
pred = np.array([c['imposed'][k] for k in keys])
pct = lambda v: 100*np.array([v[k] for k in keys])/np.abs(pred)
gen_before = 100*np.array([g['before'][k]['sd'] for k in keys])/np.abs(np.array([c['generator_only'][k] for k in keys]))
gen = 100*np.array([g['after'][k]['sd'] for k in keys])/np.abs(pred)
theory, learn, comb = pct(c['theory_band']), pct(c['learning_band']), pct(c['combined'])
FLOOR = 1e-3
x = np.arange(len(keys)); wdt = 0.2
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.bar(x - 1.5*wdt, np.maximum(gen, FLOOR), wdt, color=RED, label='generator', zorder=3)
ax.bar(x - 0.5*wdt, np.maximum(theory, FLOOR), wdt, color=BLUE, label='theory', zorder=3)
ax.bar(x + 0.5*wdt, np.maximum(np.nan_to_num(learn), FLOOR), wdt, color=INK, label='learning', zorder=3)
ax.bar(x + 1.5*wdt, np.maximum(comb, FLOOR), wdt, facecolor='none', edgecolor=INK, hatch='////', lw=0.8, label='combined', zorder=3)
for xi, v in zip(x, gen_before):
    ax.hlines(v, xi - 1.5*wdt - 0.5*wdt, xi - 1.5*wdt + 0.5*wdt, color=GREY, lw=2.4, zorder=4)
ax.hlines([], [], [], color=GREY, lw=2.4, label='generator, before the calculation')
ax.set_yscale('log'); ax.set_ylim(FLOOR, 2000); ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('uncertainty (percent)')
ax.legend(fontsize=11.5, ncol=3, loc='upper left', frameon=False, columnspacing=1.2, handlelength=1.4); trim(ax)
save(fig, out)
