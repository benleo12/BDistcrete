#!/usr/bin/env python3
"""fig_anchored_fit.pdf: the anchored sample at the FITTED point against LEP, with the fit itself.

(a) the thrust distribution as a ratio to DELPHI, normalized over the fitted bins: the unanchored
    generator at the profiled nuisance parameters with their band, and the anchored sample with the
    theory band (scale variations, half-sum of squares) and the nuisance band; the window shaded.
(b) the width of each band in percent of the prediction against tau.
(c) the coupling and alpha_0: the contour where the chi^2 of the fit to data rises by one, whose
    projections are the Delta chi^2 = 1 intervals, against the one-standard-deviation ellipse of the
    quoted errors, the scatter of the fits to forty pseudo-data sets generated at the fitted point
    (drawn as dots). Both ellipses are at the level whose projections are the one-parameter errors.
(d) the mean of 1-T over the full range against OPAL, and (e) the charged multiplicity against L3,
    before and after anchoring; the bar is the nuisance band, after anchoring combined in quadrature
    with the theory band.

Every histogram is drawn on the experiment's own bin edges.

    python fig_anchored_fit.py output/profile_MIX17ext_central.json

Reads the fit JSON, *_exchange.json and *_exchange_dist.npz (bands_exact.py), the one-parameter
profiles *_rows.json (profile_rows.py), the two-parameter surface *_surface.json (surface_rows.py)
and the pseudo-data fits at the fitted point (PSEUDO_DIR, pseudo_rows.py with the truth at the fit).
"""
import os, sys, glob, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from money_style import setup, save, trim, panel_title, BLUE, RED, OCHRE, GREY, INK, MUTED
setup(usetex=True)
fitp = sys.argv[1] if len(sys.argv) > 1 else 'output/profile_MIX17ext_central.json'
fit = json.load(open(fitp)); ex = json.load(open(fitp.replace('.json', '_exchange.json'))); D = np.load(fitp.replace('.json', '_exchange_dist.npz'), allow_pickle=True)
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('FIG_OUT', 'output/fig_anchored_fit.pdf')
PSEUDO = os.environ.get('PSEUDO_DIR', 'output/pseudo_atfit_sets')
DAT = dict(opal=(0.06671, 0.00068), l3=(18.63, 0.11)); WLO, WHI = 0.05, 1/3
lo, hi, y, e = D[f'{TEST}_lo'], D[f'{TEST}_hi'], D[f'{TEST}_data'], D[f'{TEST}_err']; ctr = 0.5*(lo + hi)
assert np.allclose(lo[1:], hi[:-1]), 'the bins must be contiguous to draw them as stairs'
edges = np.r_[lo, hi[-1]]
un, an, tsd, nsd, kb, ka = (D[f'{TEST}_{k}'] for k in ('unanchored', 'anchored', 'theory', 'np_model', 'knobs_before', 'knobs_after'))
# The stored densities are normalized over all of the experiment's bins. The fit compares shapes
# over the bins it uses, tau >= FIRST_BIN, so both sides are renormalized over those here. Drawing
# the other normalization would put the excluded peak's mismatch into every bin of the window.
FIRST = float(fit['first_bin']); use = lo >= FIRST - 1e-12; w = hi - lo
Sd = float((y[use]*w[use]).sum()); Sp = float((an[use]*w[use]).sum()); Su = float((un[use]*w[use]).sum())
y, e = y/Sd, e/Sd
an, tsd, nsd, ka = an/Sp, tsd/Sp, nsd/Sp, ka/Sp
un, kb = un/Su, kb/Su


def band(ax, lo_v, hi_v, **kw):
    """a band that follows the bin edges exactly"""
    ax.stairs(hi_v, edges, baseline=lo_v, fill=True, **kw)


fig = plt.figure(figsize=(11.0, 6.4)); gs = GridSpec(2, 4, figure=fig, width_ratios=[2.3, 1.3, 1.0, 1.0], height_ratios=[1, 1], wspace=0.6, hspace=0.5)
# (a) ratio to data
ax = fig.add_subplot(gs[0, :2])
ax.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0)
band(ax, (un-kb)/y, (un+kb)/y, color=GREY, alpha=0.40, lw=0, label='unanchored, nuisance band')
ax.stairs(un/y, edges, baseline=None, color=GREY, lw=1.3)
band(ax, (an-tsd)/y, (an+tsd)/y, color=BLUE, alpha=0.22, lw=0, label='anchored, perturbative band')
band(ax, (an-ka)/y, (an+ka)/y, color=RED, alpha=0.85, lw=0, label='anchored, nuisance band')
ax.stairs(an/y, edges, baseline=None, color=BLUE, lw=1.7, label='anchored, best fit')
ax.errorbar(ctr, y/y, yerr=e/y, fmt='o', ms=3.2, color=INK, zorder=5, label=TEST.upper())
ax.set_xlim(0, edges[-1]); ax.set_ylim(0.55, 1.85)
ax.set_xlabel(r'$\tau = 1-T$'); ax.set_ylabel(f'prediction / {TEST.upper()}')
h, l = ax.get_legend_handles_labels()
order = [l.index(s) for s in ('unanchored, nuisance band', 'anchored, perturbative band', 'anchored, nuisance band', 'anchored, best fit', TEST.upper())]
ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=12, loc='upper left', ncol=2, frameon=False)
panel_title(ax, '(a)'); trim(ax)
# (b) band widths
ax = fig.add_subplot(gs[0, 2:])
rel = lambda b, p: 100*b/np.maximum(p, 1e-12)
ax.axvspan(WLO, WHI, color=OCHRE, alpha=0.09, lw=0)
ax.stairs(rel(kb, un), edges, baseline=None, color=GREY, lw=2.0, label='nuisance, before')
ax.stairs(rel(ka, an), edges, baseline=None, color=RED, lw=2.0, label='nuisance, after')
ax.stairs(rel(tsd, an), edges, baseline=None, color=BLUE, lw=2.0, label='perturbative')
ax.stairs(rel(nsd, an), edges, baseline=None, color=BLUE, lw=1.0, ls='--', label='nonperturbative')
ax.set_xlim(0, edges[-1]); ax.set_yscale('log'); ax.set_ylim(0.05, 2000)
ax.set_yticks([0.1, 1, 10, 100])
ax.set_xlabel(r'$\tau$'); ax.set_ylabel('band width (percent)')
ax.legend(fontsize=11.5, ncol=2, loc='upper right', frameon=False, columnspacing=0.7, handlelength=1.0, borderaxespad=0.2); panel_title(ax, '(b)'); trim(ax)
# (d) mean of 1-T, (e) charged multiplicity
for col, key, dat, lab, ttl in [(gs[1, 2], 'thrust', DAT['opal'], r'$\langle 1-T\rangle$', '(d)'), (gs[1, 3], 'nch', DAT['l3'], r'$\langle n_{\rm ch}\rangle$', '(e)')]:
    ax = fig.add_subplot(col); r = ex['rows'][key]
    ax.axhspan(dat[0]-dat[1], dat[0]+dat[1], color=GREY, alpha=0.35, lw=0); ax.axhline(dat[0], color=INK, lw=0.8)
    for x, v, err, c in [(0, r['unanchored'], r['knobs_before'], GREY), (1, r['anchored'], np.hypot(r['knobs_after'], r['theory']), BLUE)]:
        ax.errorbar([x], [v], yerr=[err], fmt='o', color=c, ms=5, elinewidth=2.2, capsize=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['before', 'after'], fontsize=13); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(lab); panel_title(ax, ttl); trim(ax)
# (c) the fit to data against the quoted errors
ax = fig.add_subplot(gs[1, :2])
sf = json.load(open(fitp.replace('.json', '_surface.json')))
prof, A, A0 = np.array(sf['chi2']), np.array(sf['alphas']), np.array(sf['alpha0'])
# the surface is computed on a grid of 0.001 by 0.01, and its contour is drawn from a cubic spline
# through it, so that the grid does not draw corners; the spline's projections are checked below
from scipy.interpolate import RectBivariateSpline
spl = RectBivariateSpline(A, A0, prof, kx=3, ky=3, s=0)
Af, A0f = np.linspace(A.min(), A.max(), 481), np.linspace(A0.min(), A0.max(), 601)
dchi = spl(Af, A0f); dchi -= dchi.min()
rr = json.load(open(fitp.replace('.json', '_rows.json')))['variations']['central']
f = dict(alpha_s=rr['value'], alpha_0=rr['alpha_0_at_min'])
S = [json.load(open(p)) for p in sorted(glob.glob(f'{PSEUDO}/pseudo_atfit_*_set[0-9][0-9][0-9].json'))]
assert len(S) == 40, f'expected the forty sets at the fitted point, found {len(S)}'
assert all(abs(s['truth']['alpha_s'] - f['alpha_s']) < 1e-6 for s in S), 'the sets must be generated at the fitted point'
pa = np.array([s['alpha_s'] for s in S]); p0 = np.array([s['alpha_0_at_min'] for s in S])
cov = np.cov(pa, p0)
print(f'pseudo-data at the fit: sd {np.sqrt(cov[0, 0]):.5f}, {np.sqrt(cov[1, 1]):.4f}, correlation {cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]):+.3f}')
ax.plot(pa, p0, 'o', ms=3.4, color=GREY, alpha=0.75, mew=0, zorder=2)
# the ellipse whose projections are one standard deviation, the same level as Delta chi^2 = 1
val, vec = np.linalg.eigh(cov)
ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
ax.add_patch(Ellipse((f['alpha_s'], f['alpha_0']), 2*np.sqrt(val[1]), 2*np.sqrt(val[0]), angle=ang,
                     fill=False, color=BLUE, lw=1.8, zorder=3))
ax.contour(Af, A0f, dchi.T, levels=[1.0], colors=[RED], linewidths=[1.8], zorder=4)
ax.plot([f['alpha_s']], [f['alpha_0']], 'o', color=RED, ms=6, zorder=5)
m = dchi <= 1.0
print(f"Delta chi2 = 1 projections of the drawn contour: alpha_s {Af[m.any(1)].min():.4f} to {Af[m.any(1)].max():.4f} "
      f"(profile {rr['lo']:.4f} to {rr['hi']:.4f}), alpha_0 {A0f[m.any(0)].min():.3f} to {A0f[m.any(0)].max():.3f}")
leg = [Line2D([], [], color=RED, lw=1.8, label=r'fit to data, $\Delta\chi^2=1$'),
       Line2D([], [], color=BLUE, lw=1.8, label='quoted errors, from pseudo-data'),
       Line2D([], [], color=GREY, marker='o', ms=4, lw=0, alpha=0.75, label='fits to pseudo-data')]
ax.legend(handles=leg, fontsize=12, loc='upper left', frameon=False)
ax.set_xlabel(r'$\alpha_s(M_Z)$'); ax.set_ylabel(r'$\alpha_0(2~\mathrm{GeV})$'); panel_title(ax, '(c)'); trim(ax)
ax.set_xlim(0.1065, 0.1335); ax.set_ylim(0.235, 0.62)
ax.set_xticks([0.110, 0.115, 0.120, 0.125, 0.130]); ax.set_yticks([0.3, 0.4, 0.5, 0.6])
inside = (pa > ax.get_xlim()[0]) & (pa < ax.get_xlim()[1]) & (p0 > ax.get_ylim()[0]) & (p0 < ax.get_ylim()[1])
assert inside.all(), f'{(~inside).sum()} pseudo-data fits fall outside panel (c)'
save(fig, OUT); print('wrote', OUT)
