#!/usr/bin/env python3
"""fig_anchored_fit.pdf: the anchored sample at the FITTED point against LEP, with the fit itself.

(a) the thrust distribution as a ratio to DELPHI, normalized over the fitted bins: the unanchored
    generator at the profiled nuisance parameters with their band, and the anchored sample with the
    theory band (scale variations, half-sum of squares) and the nuisance band; the window shaded.
(b) the width of each band in percent of the prediction against tau.
(c) the two-parameter 68 and 95 percent regions of the fit in the coupling and alpha_0, with the
    one-parameter profile errors.
(d) the mean thrust over the full range against OPAL, and (e) the charged multiplicity against L3,
    before and after anchoring, inner bar the nuisance band, outer the total with the theory band in
    quadrature.

Reads the fit JSON, *_exchange.json and *_exchange_dist.npz (bands_exact.py), the one-parameter
profiles *_rows.json (profile_rows.py) and the two-parameter surface *_surface.json (surface_rows.py).
"""
import os, sys, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from money_style import setup, save, trim, panel_title, BLUE, RED, OCHRE, GREY, INK
setup(usetex=True)
fitp = sys.argv[1] if len(sys.argv) > 1 else 'output/fit_C_1M_v2.json'
fit = json.load(open(fitp)); ex = json.load(open(fitp.replace('.json', '_exchange.json'))); D = np.load(fitp.replace('.json', '_exchange_dist.npz'), allow_pickle=True)
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('FIG_OUT', 'output/fig_anchored_fit.pdf')
DAT = dict(opal=(0.06671, 0.00068), l3=(18.63, 0.11)); WLO, WHI = 0.05, 1/3
lo, hi, y, e = D[f'{TEST}_lo'], D[f'{TEST}_hi'], D[f'{TEST}_data'], D[f'{TEST}_err']; ctr = 0.5*(lo + hi)
un, an, tsd, nsd, kb, ka = (D[f'{TEST}_{k}'] for k in ('unanchored', 'anchored', 'theory', 'np_model', 'knobs_before', 'knobs_after'))
# The stored densities are normalized over all of the experiment's bins. The fit compares shapes
# over the bins it uses, tau >= FIRST_BIN, so both sides are renormalized over those here. Drawing
# the other normalization would put the excluded peak's mismatch into every bin of the window.
FIRST = float(fit['first_bin']); use = lo >= FIRST - 1e-12; w = hi - lo
Sd = float((y[use]*w[use]).sum()); Sp = float((an[use]*w[use]).sum()); Su = float((un[use]*w[use]).sum())
y, e = y/Sd, e/Sd
an, tsd, nsd, ka = an/Sp, tsd/Sp, nsd/Sp, ka/Sp
un, kb = un/Su, kb/Su
fig = plt.figure(figsize=(11.0, 6.4)); gs = GridSpec(2, 4, figure=fig, width_ratios=[2.3, 1.3, 1.0, 1.0], height_ratios=[1, 1], wspace=0.6, hspace=0.45)
# (a) ratio to data
ax = fig.add_subplot(gs[0, :2])
ax.errorbar(ctr, y/y, yerr=e/y, fmt='o', ms=3.2, color=INK, zorder=5, label=TEST.upper())
ax.fill_between(ctr, (un-kb)/y, (un+kb)/y, step='mid', color=GREY, alpha=0.40, lw=0, label='unanchored, nuisance band')
ax.step(ctr, un/y, where='mid', color=GREY, lw=1.3)
ax.fill_between(ctr, (an-tsd)/y, (an+tsd)/y, step='mid', color=BLUE, alpha=0.22, lw=0, label='anchored, theory band')
ax.fill_between(ctr, (an-ka)/y, (an+ka)/y, step='mid', color=RED, alpha=0.85, lw=0, label='anchored, nuisance band')
ax.step(ctr, an/y, where='mid', color=BLUE, lw=1.7, label='anchored, best fit')
ax.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0); ax.set_xlim(0, min(0.5, hi.max())); ax.set_ylim(0.55, 1.85)
ax.set_xlabel(r'$\tau = 1-T$'); ax.set_ylabel(f'prediction / {TEST.upper()}'); ax.legend(fontsize=12, loc='upper left', ncol=2, frameon=False); panel_title(ax, '(a)')
# (b) band widths
ax = fig.add_subplot(gs[0, 2:])
rel = lambda b, p: 100*b/np.maximum(p, 1e-12)
ax.step(ctr, rel(kb, un), where='mid', color=GREY, lw=2.0, label='nuisance, before')
ax.step(ctr, rel(ka, an), where='mid', color=RED, lw=2.0, label='nuisance, after')
ax.step(ctr, rel(tsd, an), where='mid', color=BLUE, lw=2.0, label='theory')
ax.step(ctr, rel(nsd, an), where='mid', color=BLUE, lw=1.0, ls='--', label='np model')
ax.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0); ax.set_xlim(0, min(0.5, hi.max())); ax.set_yscale('log'); ax.set_ylim(0.05, 100)
ax.set_xlabel(r'$\tau$'); ax.set_ylabel('band width (percent)'); ax.legend(fontsize=12, ncol=2, loc='upper right', frameon=False, columnspacing=1.0); panel_title(ax, '(b)')
# (d) mean thrust, (e) charged multiplicity
for k, (col, key, dat, lab, ttl) in enumerate([(gs[1, 2], 'thrust', DAT['opal'], r'$\langle 1-T\rangle$', '(d)'), (gs[1, 3], 'nch', DAT['l3'], r'$\langle n_{\rm ch}\rangle$', '(e)')]):
    ax = fig.add_subplot(col); r = ex['rows'][key]
    ax.axhspan(dat[0]-dat[1], dat[0]+dat[1], color=GREY, alpha=0.35, lw=0); ax.axhline(dat[0], color=INK, lw=0.8)
    for x, v, kb_, ka_, th_, c in [(0, r['unanchored'], r['knobs_before'], None, 0.0, GREY), (1, r['anchored'], None, r['knobs_after'], r['theory'], BLUE)]:
        inner = kb_ if kb_ is not None else ka_
        ax.errorbar([x], [v], yerr=[np.hypot(inner, th_)], fmt='none', ecolor=c, elinewidth=1.2, capsize=4)
        ax.errorbar([x], [v], yerr=[inner], fmt='o', color=c, ms=5, elinewidth=3.0, capsize=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['before', 'after'], fontsize=13); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(lab); panel_title(ax, ttl)
# (c) the fit surface
ax = fig.add_subplot(gs[1, :2])
if os.path.exists(fitp.replace('.json', '_surface.json')):
    # the surface with the nuisance parameters minimized at every point, the targets splined
    sf = json.load(open(fitp.replace('.json', '_surface.json')))
    prof, A, A0 = np.array(sf['chi2']), np.array(sf['alphas']), np.array(sf['alpha0'])
else:
    prof = np.array(fit['profile']); A, A0 = np.array(fit['alphas']), np.array(fit['alpha0'])
dchi = prof - prof.min()
# two-dimensional confidence regions, whose levels are 2.30 and 6.18, not 1 and 4: the bars are
# the one-dimensional profile errors and are a different statement, so both are labelled
cs = ax.contour(A, A0, dchi.T, levels=[2.30, 6.18], colors=[BLUE, GREY], linewidths=[1.8, 1.0])
# a legend instead of inline labels, which would cut gaps into the contours
from matplotlib.lines import Line2D
reg = [Line2D([], [], color=BLUE, lw=1.8, label=r'$68\%$, two parameters'), Line2D([], [], color=GREY, lw=1.0, label=r'$95\%$, two parameters')]
pr = fit['fit_profile']
if os.path.exists(fitp.replace('.json', '_rows.json')):
    # the one-parameter values and errors of the paper, from the continuous profiles of profile_rows.py
    rr = json.load(open(fitp.replace('.json', '_rows.json')))
    pr = dict(alpha_s=rr['variations']['central'], alpha_0=rr.get('alpha_0_profile', pr['alpha_0']))
f = dict(alpha_s=pr['alpha_s']['value'], alpha_0=pr['alpha_0']['value'])
ax.plot([f['alpha_s']], [f['alpha_0']], 'o', color=RED, ms=6)
ax.errorbar([f['alpha_s']], [f['alpha_0']], xerr=[[pr['alpha_s']['err_lo']], [pr['alpha_s']['err_hi']]],
            yerr=[[pr['alpha_0']['err_lo']], [pr['alpha_0']['err_hi']]], fmt='none', ecolor=RED, elinewidth=1.4,
            capsize=3, label='one-parameter errors')
ax.legend(handles=reg + [ax.get_legend_handles_labels()[0][0]], fontsize=11, loc='upper left', frameon=False)
ax.set_xlabel(r'$\alpha_s(M_Z)$'); ax.set_ylabel(r'$\alpha_0(2~\mathrm{GeV})$'); panel_title(ax, '(c)')
# the region the 95 percent contour occupies, with a margin, and ticks every 0.004 in the coupling
inside = dchi <= 9.0
xa = A[np.any(inside, 1)]; ya = A0[np.any(inside, 0)]
ax.set_xlim(max(A.min(), xa.min() - 0.002), min(A.max(), xa.max() + 0.002)); ax.set_ylim(max(A0.min(), ya.min() - 0.02), min(A0.max(), ya.max() + 0.02))
lo_t = np.ceil(ax.get_xlim()[0]/0.004 - 1e-9)*0.004; ax.set_xticks(np.round(np.arange(lo_t, ax.get_xlim()[1] + 1e-9, 0.004), 3))
save(fig, OUT); print('wrote', OUT)
