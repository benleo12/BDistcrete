#!/usr/bin/env python3
"""Figure: the thrust distribution of the reweighted sample against LEP data, with each of the
three uncertainties on its own panel and their combination on the main one.

(a) The ratio to the measurement on the experiment's bins, normalized over the fitted bins: the
    generator alone (grey line) with all its uncertainties, and with the NNLL+NLO calculation
    imposed (blue line) with the three uncertainties combined in quadrature. The window is shaded.
(b) The generator band in percent of the prediction, before and after the calculation.
(c) The theory band, with the dispersive-model variations separately.
(d) The learning band.
The combined width is repeated as a thin dotted line on (b) to (d), so each band can be read
against the total on the same axis.

    python fig_three_bands.py output/three_bands_MIXGEO_w.npz output/three_bands_MIXGEO_w.json figs/fig_three_bands.pdf
"""
import sys, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from money_style import setup, save, trim, panel_title, BLUE, RED, GREY, OCHRE, INK, MUTED
setup(usetex=True)
npz, js, out = sys.argv[1:4]
D = np.load(npz); J = json.load(open(js))
lo, hi, y, e = D['lo'], D['hi'], D['data'], D['err']; edges = np.r_[lo, hi[-1]]
WLO, WHI = J['calc']['window']
h0, h1 = D['h_central_before'], D['h_central']
g0, g1, th, le, le0 = D['gen_hist_before'], D['gen_hist'], D['theory_hist'], D['learn_hist'], D['learn_hist_before']
c0, c1 = D['comb_hist_before'], D['comb_hist']
nphist = np.sqrt(np.mean(D['np_hist']**2, 0))
has_learn = np.isfinite(le).all()
TEST = J['test'].upper()

fig = plt.figure(figsize=(11.0, 7.4))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1.0], hspace=0.42, wspace=0.30)
ax = fig.add_subplot(gs[0, :]); bx, cx, dx = (fig.add_subplot(gs[1, i]) for i in range(3))


def band(a, lo_v, hi_v, **kw):
    a.stairs(hi_v, edges, baseline=lo_v, fill=True, **kw)


# (a) the ratio to the data with the combined bands
ax.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0)
band(ax, (h0 - c0)/y, (h0 + c0)/y, color=GREY, alpha=0.35, lw=0, label='generator only, all uncertainties')
ax.stairs(h0/y, edges, baseline=None, color=GREY, lw=1.4)
band(ax, (h1 - c1)/y, (h1 + c1)/y, color=BLUE, alpha=0.28, lw=0, label='calculation imposed, all uncertainties')
ax.stairs(h1/y, edges, baseline=None, color=BLUE, lw=1.8)
ax.errorbar(0.5*(lo + hi), y/y, yerr=e/y, fmt='o', ms=3.4, color=INK, zorder=5, label=TEST)
ax.set_xlim(0, edges[-1]); ax.set_ylim(0.55, 1.85); ax.set_xlabel(r'$\tau = 1-T$'); ax.set_ylabel(f'prediction / {TEST}')
ax.legend(fontsize=12.5, loc='upper left', ncol=1, frameon=False); panel_title(ax, '(a) generator, theory and learning uncertainties combined'); trim(ax)

# (b) to (d): each band in percent of the prediction, the combined width repeated
rel = lambda b, p: 100*b/np.maximum(p, 1e-12)
for a, title in ((bx, '(b) generator'), (cx, '(c) theory'), (dx, '(d) learning')):
    a.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0)
    a.stairs(rel(c1, h1), edges, baseline=None, color=INK, lw=1.0, ls=(0, (1.5, 1.5)), label='all three combined')
    a.set_xlim(0, edges[-1]); a.set_yscale('log'); a.set_ylim(0.001, 500); a.set_yticks([0.01, 0.1, 1, 10, 100])
    a.set_xlabel(r'$\tau$'); panel_title(a, title, fs=14.5); trim(a)
bx.stairs(rel(g0, h0), edges, baseline=None, color=GREY, lw=2.0, label='before the calculation')
bx.stairs(rel(g1, h1), edges, baseline=None, color=RED, lw=2.0, label='calculation imposed')
cx.stairs(rel(th, h1), edges, baseline=None, color=BLUE, lw=2.0, label='scales and scheme')
cx.stairs(rel(nphist, h1), edges, baseline=None, color=BLUE, lw=1.2, ls='--', label='dispersive model')
if has_learn:
    dx.stairs(rel(le0, h0), edges, baseline=None, color=GREY, lw=2.0, label='before the calculation')
    dx.stairs(rel(le, h1), edges, baseline=None, color=INK, lw=2.0, label='calculation imposed')
bx.set_ylabel('band width (percent)')
for a in (bx, cx, dx):
    a.legend(fontsize=10.5, loc='upper left', frameon=False, handlelength=1.3, borderaxespad=0.2, labelspacing=0.25)
save(fig, out)
