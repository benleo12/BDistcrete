#!/usr/bin/env python3
"""The data figures of the paper at their printed size, in one style. plot_envelope.py draws Figs. 8 to 10
in the same style.

Figures are drawn at the width they have on the page (TW = 5.95 in, the JHEP text width, or a stated
fraction of it), so the fonts are the printed fonts: 9 pt axis labels and panel titles, 8 pt tick
labels, 7.5 pt legends and annotations. Each color keeps one meaning across the paper, as set out
in the block of constants below.

    python figs_r8.py               # the nine data figures of the paper, output/r8_<name>.pdf
    python figs_r8.py bands obs     # only the figures named

The nine, in the order of the paper: closure_mix (Fig. 3), map (Fig. 4), coverage (Fig. 5), kscan
(Fig. 6), cost (Fig. 7), bands (Fig. 11), obs (Fig. 12), toy (Fig. 13) and aleph (Fig. 14). widebox
(the wide coupling scan, whose numbers the paper quotes without a figure) and closure (Stage C
reweighted to a held-out point) are not in the paper and are drawn only when named. R8_OUT sets the output folder (default output). XP_NULL sets the dotted tick of
Fig. 4 (default the mean of output/xp_null.json).

The linear mixture: a figure drawn from result files of the linear mixture (names ending in _lin: BANDS_NPZ,
BANDS_JSON, VARS_JSON; COST_SUFFIX=_lin for the cost benchmark, read from BENCH_DIR, default output) checks that each of them holds mix = linear and is
written as r8_<name>_lin.pdf; each benchmark file must hold the device and threads of its name. Fig. 11 compares
with DELPHI at its hadron level (D8, the *_conv predictions of
three_bands.py and theory_variations.py) when the files hold it (FIG_CONV=0 draws the uncorrected predictions), and
takes its vertical range from what it draws.

Ranges (rounds B10, B11 and B21): every axis keeps the published range when what it draws fits in it, and otherwise
widens only as far as needed (widen()), so that no value of the rerun is hidden: Figs. 3 (the reweighted ratio and its
band, and since B21 the dashed unweighted reference, each side alone), 5, 6 (points and error bars; since B21 none
ends on the axis), 7, 11, 12 (every bar but the two bands of <tau>_win that the calculation removes, its learning band,
which the caption puts below the axis, and its generator band with the calculation imposed, which the text says
disappears; neither lowers the axis, and the job log says where both lie) and 14 (the data points with their error bars
and the calculation inside the window, with ticks for any span of the ratio panel; since B21 also every data point below
tau = 0.5 in the upper panel and the theory band in the ratio panel; a data point outside the window and beyond the
range of a panel is drawn as an arrow at its edge). Fig. 13 shows the central 99.4 percent of the events, as published,
with the axes half a hexagon wider (B21), so that the hexagons on its edge are whole.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter, LogLocator, NullFormatter, MultipleLocator, FormatStrFormatter
# One visual language for every figure of the paper. Each color has one job everywhere (Okabe-Ito
# hues, checked with the dataviz validator for color-vision deficiency):
#   METHOD  the classifier-weighted prediction: reweighted sample, factorized network, learned functions,
#           the prediction with the calculation imposed (ordered series: one ramp of this hue)
#   GEN     the generator band, and the generator models in the schematics
#   THEORY  the calculation: its curve, its theory band, and the window in which it is imposed
#   LEARN   the learning band
#   DATA    measurements and fresh generator runs, the truth a prediction is compared with
#   BASE    what the method starts from or is compared with: the unweighted reference, the generators
#           alone, a single reference run, alternative networks (BASE_DARK for a second one)
#   REF     reference lines (one, an expectation, a bound) and annotations
# Marks: data are black points with error bars, filled for the main data set and open for a second
# one. Predictions are lines or stairs with a translucent band of their own color. Variants of one
# series differ by line style or marker fill, never by a new color.
METHOD, GEN, THEORY, LEARN = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
DATA, BASE, BASE_DARK, REF = '#1A1A1A', '#8C8C8C', '#4D4D4D', '#5A5D62'
METHOD_DARK, LEARN_TEXT = '#00446B', '#A8538A'
BAND_ALPHA, WINDOW_ALPHA = 0.25, 0.08
INK, MUTED = DATA, REF


def ramp(n):
    """An ordinal ramp of METHOD from light to dark, for series in a natural order (the stages)."""
    base = np.array([0, 114, 178])/255.0; out = []
    for t in np.linspace(-0.45, 0.4, n):
        c = base + (1 - base)*(-t) if t < 0 else base*(1 - t)
        out.append(tuple(c))
    return out

TW = 5.95                                   # JHEP text width in inches
OUT = os.environ.get('R8_OUT', 'output')


def style():
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'axes.labelpad': 2.5, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3, 'legend.columnspacing': 1.0,
        'lines.linewidth': 1.1, 'lines.markersize': 3.2, 'errorbar.capsize': 0,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015, 'figure.dpi': 150,
    })


def trim(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(which='both', top=False, right=False)


def title(ax, s):
    ax.set_title(s, loc='left', fontsize=9, pad=4)


def unit_line(ax, y=1.0, **kw):
    ax.axhline(y, color=REF, lw=0.7, ls=(0, (4, 3)), zorder=1, **kw)


def save(fig, name):
    path = os.path.join(OUT, f'r8_{name}.pdf'); fig.savefig(path); plt.close(fig); print('wrote', path)


def linear_inputs(*pairs):
    """pairs of (path, loaded json or npz). True when any path names a result of the linear mixture (ends in _lin),
    in which case every one of them must hold mix = linear."""
    lin = any(os.path.splitext(p)[0].endswith('_lin') for p, _ in pairs)
    if lin:
        for p, d in pairs:
            m = str(d['mix']) if (hasattr(d, 'files') and 'mix' in d.files) or (isinstance(d, dict) and 'mix' in d) else None
            assert m == 'linear', f'{p} does not hold the linear mixture (mix = {m!r})'
    return lin


def nice_range(vals, pad=0.06):
    """A vertical range that holds every finite value of vals, rounded out to steps of 0.05, and ticks for it."""
    v = np.concatenate([np.ravel(np.asarray(a, float)) for a in vals]); v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max()); sp = max(hi - lo, 0.1)
    lo, hi = np.floor((lo - pad*sp)*20)/20, np.ceil((hi + pad*sp)*20)/20
    st = next(t for t in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0) if (hi - lo)/t <= 7)
    return (lo, hi), [t for t in np.arange(np.ceil(lo/st)*st, hi + 1e-9, st)]


def half_ticks(ax):
    """Round B21: a ratio panel that was widened keeps the ticks of its neighbours, steps of 0.5 with one decimal, where
    matplotlib would switch to whole numbers; only the ticks change, and a span above 4 keeps matplotlib's choice."""
    lo, hi = ax.get_ylim()
    if hi - lo <= 4:
        ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def widen(lo, hi, vals, pad=0.05, log=False, clear=0.0):
    """The published range (lo, hi), widened only as far as needed to hold every finite value of vals (positive ones
    on a log axis), with a margin of pad of the new height (in decades on a log axis). Unchanged when all fit. Round
    B21: with clear, a value closer to an edge than clear of the height counts as beyond it (an error bar that would end
    on the axis)."""
    v = np.concatenate([np.ravel(np.asarray(a, float)) for a in vals]) if len(vals) else np.array([])
    v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
    f, g = (np.log10, lambda x: 10**x) if log else (lambda x: x, lambda x: x)
    lo_c, hi_c = (g(f(lo) + clear*(f(hi) - f(lo))), g(f(hi) - clear*(f(hi) - f(lo)))) if clear else (lo, hi)
    if not len(v) or (v.min() >= lo_c and v.max() <= hi_c):
        return lo, hi
    a, b = f(min(lo, v.min())), f(max(hi, v.max())); m = pad*(b - a)
    return g(a - m) if v.min() < lo_c else lo, g(b + m) if v.max() > hi_c else hi


def edge_arrows(ax, x, y, lo, hi, log=False, **kw):
    """Points of (x, y) beyond the vertical range (lo, hi), drawn as arrows just inside its edge, so that none is
    hidden. Returns how many."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ins = (lambda e, s: e*(hi/lo)**(0.035*s)) if log else (lambda e, s: e + 0.035*s*(hi - lo))
    n = 0
    for sel, yy, mk in ((np.isfinite(y) & (y > hi), ins(hi, -1), '^'), (np.isfinite(y) & (y < lo), ins(lo, 1), 'v')):
        if sel.any():
            ax.plot(x[sel], np.full(sel.sum(), yy), mk, ms=3.4, zorder=7, **kw); n += int(sel.sum())
    return n


def window(ax, lo, hi):
    ax.axvspan(lo, hi, color=THEORY, alpha=WINDOW_ALPHA, lw=0, zorder=0)


# ---- Fig. closure: Stage C reweighted to a held-out point ------------------------------------------
def closure():
    src = os.environ.get('HERO_FROM', 'output/fig_closure_C_P2augfullfac0_6901_report.npz')
    z = np.load(src)
    P = {o: {k: z[f'{o}__{k}'] for k in ('bins', 'dw', 'ew', 'du', 'dt', 'et', 'width')} for o in [str(x) for x in z['panels']]}
    PANELS = [('1_minus_thrust', r'$1-T$', r'$\frac{1}{\sigma}\frac{d\sigma}{d(1-T)}$', True, '(a) thrust'),
              ('mult_total', r'multiplicity $n$', r'$\frac{1}{\sigma}\frac{d\sigma}{dn}$', False, '(b) multiplicity'),
              ('nbaryon', r'baryon count $n_{\rm b}$', r'$\frac{1}{\sigma}\frac{d\sigma}{dn_{\rm b}}$', True, '(c) baryon count'),
              ('x_p', r'energy fraction $E_i/E_{\rm vis}$', r'$\frac{1}{n}\frac{dn}{d(E_i/E_{\rm vis})}$', True, '(d) energy fraction')]
    fig = plt.figure(figsize=(TW, 4.35))
    outer = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, left=0.1, right=0.99, top=0.95, bottom=0.08)
    for i, (obs, xlab, ylab, logy, ttl) in enumerate(PANELS):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i//2, i % 2], height_ratios=[2.1, 1], hspace=0.07)
        axd = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=axd)
        p = P[obs]; bins, dw, ew, du, dt, et = (p[k] for k in ('bins', 'dw', 'ew', 'du', 'dt', 'et'))
        ctr = 0.5*(bins[1:] + bins[:-1])
        axd.stairs(du, bins, baseline=None, color=BASE, lw=1.0, ls=(0, (3, 2)), label='unweighted reference')
        axd.stairs(dw + ew, bins, baseline=dw - ew, fill=True, color=METHOD, alpha=BAND_ALPHA, lw=0)
        axd.stairs(dw, bins, baseline=None, color=METHOD, lw=1.2, label='reweighted reference')
        axd.errorbar(ctr, dt, yerr=et, fmt='o', color=DATA, ms=2.0, lw=0.7, zorder=5, label='fresh run')
        if logy:
            axd.set_yscale('log'); axd.set_ylim(max(dt[dt > 0].min()*0.3, 1e-6), None)
        if obs == 'mult_total':
            axd.set_xlim(0, 92); axd.set_ylim(0, 1.3*np.nanmax(dt))
        axd.set_ylabel(ylab, fontsize=9.5)
        plt.setp(axd.get_xticklabels(), visible=False)
        axd.text(0.97, 0.93, rf'width ${float(p["width"]):.2f}$', transform=axd.transAxes, ha='right', va='top',
                 fontsize=8, color=REF, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
        title(axd, ttl); trim(axd)
        if i == 0:
            axd.legend(loc='lower left', fontsize=7.5)
        ok = dt > 0
        nan = lambda v: np.where(ok, v, np.nan)
        rw = nan(dw/np.where(ok, dt, 1)); rwe = nan(ew/np.where(ok, dt, 1)); ru = nan(du/np.where(ok, dt, 1))
        rte = nan(et/np.where(ok, dt, 1))
        axr.axhline(1, color=REF, lw=0.6)
        axr.stairs(1 + rte, bins, baseline=1 - rte, fill=True, color=BASE, alpha=0.28, lw=0)
        axr.stairs(ru, bins, baseline=None, color=BASE, lw=1.0, ls=(0, (3, 2)))
        axr.stairs(rw + rwe, bins, baseline=rw - rwe, fill=True, color=METHOD, alpha=BAND_ALPHA, lw=0)
        axr.stairs(rw, bins, baseline=None, color=METHOD, lw=1.1)
        span = np.nanmax(np.abs(np.r_[ru[ok], rw[ok]] - 1))
        lim = 0.14 if span < 0.12 else min(1.15*span, 0.8)
        axr.set_ylim(1 - lim, 1 + lim)
        if obs == 'mult_total':
            axr.set_xlim(0, 92)
        axr.set_xlabel(xlab)
        axr.set_ylabel('ratio to\nfresh run', fontsize=8, linespacing=0.95)
        trim(axr)
    save(fig, 'closure')


# ---- Fig. closure of the seventeen-parameter mixture at a Sherpa and a Herwig held-out point -------
def closure_mix():
    z = np.load(os.environ.get('MIXCL_IN', 'output/fig_closure_mix.npz'))
    rids = [int(r) for r in z['rids']]
    COLS = [('1_minus_thrust', r'$1-T$', r'$\frac{1}{\sigma}\frac{d\sigma}{d(1-T)}$', True, 'thrust'),
            ('mult_total', r'multiplicity $n$', r'$\frac{1}{\sigma}\frac{d\sigma}{dn}$', False, 'multiplicity'),
            ('nbaryon', r'baryon count $n_{\rm b}$', r'$\frac{1}{\sigma}\frac{d\sigma}{dn_{\rm b}}$', True, 'baryon count')]
    GEN = {7: 'Sherpa', 8: 'Herwig'}
    fig = plt.figure(figsize=(TW, 4.95))
    outer = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.42, left=0.085, right=0.99, top=0.86, bottom=0.08)
    tags = iter('abcdef')
    for r, rid in enumerate(rids):
        for c, (obs, xlab, ylab, logy, name) in enumerate(COLS):
            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c], height_ratios=[2.0, 1], hspace=0.07)
            axd = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=axd)
            g = lambda k: z[f'{rid}__{obs}__{k}']
            bins, dw, ew, du, dt, et = g('bins'), g('dw'), g('ew'), g('du'), g('dt'), g('et')
            ctr = 0.5*(bins[1:] + bins[:-1])
            axd.stairs(du, bins, baseline=None, color=BASE, lw=1.0, ls=(0, (3, 2)), label='unweighted reference')
            axd.stairs(dw + ew, bins, baseline=dw - ew, fill=True, color=METHOD, alpha=BAND_ALPHA, lw=0)
            axd.stairs(dw, bins, baseline=None, color=METHOD, lw=1.2, label='reweighted reference')
            axd.errorbar(ctr, dt, yerr=et, fmt='o', color=DATA, ms=1.8, lw=0.7, zorder=5, label='fresh run')
            # the ranges follow the data (round B10): the published ones whenever everything fits in them
            if logy:
                low = dt[dt > 0].min()*0.3; other = np.r_[du[du > 0], dw[dw > 0]]
                if len(other) and other.min() < low:                    # the reference or the reweighted sample lower
                    low = 0.9*other.min()
                axd.set_yscale('log'); axd.set_ylim(max(low, 1e-6), None)
            xhi = 80.0
            if obs == 'mult_total':
                # 0 to 80, unless a bin from 80 up holds 5 percent of the peak (the published bins there hold 2 or less)
                hh = np.maximum.reduce([dt, du, dw]); big = (bins[:-1] >= 80) & (hh >= 0.05*hh.max())
                xhi = float(max(80.0, bins[1:][big].max())) if big.any() else 80.0
                top = max(1.3*np.nanmax(dt), 1.05*np.nanmax(np.r_[du, dw + ew, dt + et]))
                axd.set_xlim(0, xhi); axd.set_ylim(0, top)
            axd.set_ylabel(ylab, fontsize=9.5)
            plt.setp(axd.get_xticklabels(), visible=False)
            axd.text(0.97, 0.93, rf'width ${float(g("width")):.2f}$', transform=axd.transAxes, ha='right', va='top',
                     fontsize=7.5, color=REF, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
            title(axd, f'({next(tags)}) {name}'); trim(axd)
            if c == 0:
                fig.text(0.53, outer[r, 0].get_position(fig).y1 + 0.052,
                         f'held-out {GEN[rid // 1000]} point of the seventeen-parameter mixture',
                         ha='center', va='bottom', fontsize=9)
            if r == 0 and c == 0:
                fig.legend(*axd.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.55, 1.0), ncol=3, fontsize=7.5)
            ok = dt > 0
            nan = lambda v: np.where(ok, v, np.nan)
            rw = nan(dw/np.where(ok, dt, 1)); rwe = nan(ew/np.where(ok, dt, 1)); ru = nan(du/np.where(ok, dt, 1))
            rte = nan(et/np.where(ok, dt, 1))
            axr.axhline(1, color=REF, lw=0.6)
            axr.stairs(1 + rte, bins, baseline=1 - rte, fill=True, color=BASE, alpha=0.28, lw=0)
            axr.stairs(ru, bins, baseline=None, color=BASE, lw=1.0, ls=(0, (3, 2)))
            axr.stairs(rw + rwe, bins, baseline=rw - rwe, fill=True, color=METHOD, alpha=BAND_ALPHA, lw=0)
            axr.stairs(rw, bins, baseline=None, color=METHOD, lw=1.1)
            span = np.nanmax(np.abs(np.r_[ru[ok], rw[ok]] - 1))
            lim = 0.14 if span < 0.12 else min(1.15*span, 0.8)
            # the reweighted ratio and its band always fit
            need = np.nanmax(np.abs(np.r_[(rw + rwe)[ok], (rw - rwe)[ok]] - 1))
            if need > lim:
                lim = 1.15*need
            # and so does the dashed unweighted reference (round B21; before, it could leave the panel in sparse tails
            # beyond a ratio of 1.8, as in (b) below n = 14): each side widened alone, with the same margin
            ylo, yhi = 1 - lim, 1 + lim
            if np.nanmax(ru[ok]) > yhi:
                yhi = 1 + 1.15*(np.nanmax(ru[ok]) - 1)
            if np.nanmin(ru[ok]) < ylo:
                ylo = 1 - 1.15*(1 - np.nanmin(ru[ok]))
            axr.set_ylim(ylo, yhi)
            if (ylo, yhi) != (1 - lim, 1 + lim):
                half_ticks(axr)
            if obs == 'mult_total':
                axr.set_xlim(0, xhi)
            axr.set_xlabel(xlab)
            if c == 0:
                axr.set_ylabel('ratio to\nfresh run', fontsize=8, linespacing=0.95)
            trim(axr)
    save(fig, 'closure_mix')


# ---- Fig. map: Stage C across its box ------------------------------------------------------------
def map_():
    d = json.load(open(os.environ.get('MAP_IN', 'output/valid_map_C_P2augfullfac0_report.json')))
    xp_null = os.environ.get('XP_NULL')   # the dotted tick, default the mean of output/xp_null.json
    if xp_null is None and os.path.exists('output/xp_null.json'):
        xp_null = json.load(open('output/xp_null.json'))['mean']
    xp_null = float(xp_null if xp_null is not None else 'nan')
    pts = {int(k): v for k, v in d.items() if k != 'summary' and v.get('kind') == 'valid'}
    anc = {int(k): v for k, v in d.items() if k != 'summary' and v.get('kind') == 'anchor'}
    OBS = [('1_minus_thrust', r'$1-T$'), ('B_total', r'$B_{\rm tot}$'), ('rho_heavy', r'$\rho_H$'),
           ('mult_total', 'mult.'), ('x_p', r'$E_i/E_{\rm vis}$'), ('strange', 'kaons'),
           ('joint_thrust_mult', 'joint'), ('nbaryon', 'baryons')]
    fig = plt.figure(figsize=(TW, 2.45))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1], wspace=0.32, left=0.075, right=0.99, top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0]); rng = np.random.default_rng(3)
    for i, (o, lab) in enumerate(OBS):
        va = np.array([p[o] for p in pts.values()]); vb = np.array([p[o] for p in anc.values()])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(va)), va, s=8, color=METHOD, alpha=0.85, lw=0, zorder=3,
                   label='twelve further runs' if i == 0 else None)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vb)), vb, s=12, marker='D', facecolor='white', edgecolor=METHOD,
                   lw=0.8, zorder=4, label='five held-out points' if i == 0 else None)
        ax.plot([i - 0.26, i + 0.26], [np.nanmean(va)]*2, color=METHOD_DARK, lw=1.5, zorder=5,
                label='mean of the further runs' if i == 0 else None)
        if o == 'x_p' and np.isfinite(xp_null):
            ax.plot([i - 0.3, i + 0.3], [xp_null]*2, color=REF, lw=0.9, ls=(0, (1.2, 1.2)), zorder=2)
    unit_line(ax)
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([l for _, l in OBS], fontsize=7.5)
    ax.set_xlim(-0.6, len(OBS) - 0.4)
    ax.set_ylabel(r'closure width'); ax.legend(loc='upper left', fontsize=7.5)
    title(ax, '(a) closure width per observable'); trim(ax)
    bx = fig.add_subplot(gs[0, 1])
    norm = [(0.120, 0.008), (0.475, 0.175), (1.30, 0.50)]
    for coll, mk, fc, lab, s in ((pts, 'o', METHOD, 'further runs', 10), (anc, 'D', 'white', 'held-out points', 13)):
        dd = [max(abs((p['theta'][i] - norm[i][0])/norm[i][1]) for i in range(3)) for p in coll.values()]
        zz = [abs(p.get('nbaryon_z', np.nan)) for p in coll.values()]
        bx.scatter(dd, zz, s=s, marker=mk, facecolor=fc, edgecolor=METHOD, lw=0.8, zorder=3, label=lab)
    bx.axhline(2, color=REF, lw=0.7, ls=(0, (4, 3)))
    bx.set_xlabel('distance from the middle', labelpad=1.5); bx.set_ylabel(r'$|$difference$|$ / s.d.')
    bx.set_ylim(0, None); bx.legend(loc='upper left', fontsize=7.5)
    title(bx, '(b) mean baryon count'); trim(bx)
    save(fig, 'map')


# ---- Fig. widebox: three references over a wide coupling range -------------------------------------
def widebox():
    w = json.load(open(os.environ.get('WIDEBOX_IN', 'output/widebox_wb18.json')))
    fig = plt.figure(figsize=(TW, 2.35))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.3, left=0.08, right=0.99, top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    for key, col, mk, ls, mfc, lab in (('single_centre', BASE, 'o', '-', BASE, r'single run at $\alpha_s^{\rm sh}=0.14$'),
                                       ('pool_all', METHOD, 's', '-', METHOD, 'union of all 21 runs'),
                                       ('local_union', METHOD, 'D', (0, (3, 1.5)), 'white', 'union of the 7 nearest runs')):
        d = w[key]; a = sorted(float(x) for x in d); y = [d[f'{x:g}'] if f'{x:g}' in d else d[str(x)] for x in a]
        ax.plot(a, y, marker=mk, color=col, ms=3.4, lw=1.1, ls=ls, mfc=mfc, mec=col, mew=0.9, label=lab)
    unit_line(ax)
    ax.axvline(0.14, color=REF, lw=0.6, ls=(0, (1, 2)), ymax=0.5)
    ax.set_xlabel(r'held-out shower coupling $\alpha_s^{\rm sh}$'); ax.set_ylabel('closure width')
    ax.legend(loc='upper right', fontsize=7.5); title(ax, '(a) closure width'); trim(ax)
    bx = fig.add_subplot(gs[0, 1])
    ws = np.load(os.environ.get('WIDEBOX_WS', 'output/wb/W_P2wb18singlefac0_w_9100.npy'))
    wu = np.load(os.environ.get('WIDEBOX_WU', 'output/wb/W_P2wb18local9100fac0_w_9100.npy'))
    bins = np.linspace(-6, 3, 61)
    curves = [(ws, BASE, '-', 'single run'), (wu, METHOD, (0, (3, 1.5)), '7 nearest runs')]
    fpool = os.environ.get('WIDEBOX_WP', 'output/wb/W_P2wb18poolfac0_w_9100.npy')
    if os.path.exists(fpool):
        curves.insert(1, (np.load(fpool), METHOD, '-', 'all 21 runs'))
    for wv, col, ls, lab in curves:
        wv = wv/wv.sum(); h, _ = np.histogram(np.log10(np.clip(wv*len(wv), 1e-8, None)), bins)
        bx.stairs(h/h.sum(), bins, color=col, lw=1.1, ls=ls, label=lab)
    bx.axvline(np.log10(7), color=REF, lw=0.8, ls=(0, (1, 1.5)), label=r'bounds $w\le M$')
    if os.path.exists(fpool):
        bx.axvline(np.log10(21), color=REF, lw=0.8, ls=(0, (1, 1.5)))
    bx.set_yscale('log'); bx.set_xlim(-2.9, 3.4); bx.set_ylim(1e-5, 2e3)
    bx.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])); bx.yaxis.set_minor_locator(NullLocator())
    bx.yaxis.set_major_formatter(FixedFormatter([r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$']))
    bx.set_xlabel(r'$\log_{10}$ of the weight (unit mean)'); bx.set_ylabel('fraction of events')
    bx.legend(loc='upper left', fontsize=7.5, handlelength=1.6); title(bx, r'(b) weights at $\alpha_s^{\rm sh}=0.083$'); trim(bx)
    save(fig, 'widebox')


# ---- Fig. rank: closure width against the number of terms -------------------------------------------
def kscan():
    rs = json.load(open(os.environ.get('RANK_IN', 'output/rank_scan_final.json')))
    R4 = ramp(4)
    SERIES = [('A', R4[0], 'o', 1, 'Stage A, $d=1$'), ('B', R4[1], 'o', 2, 'Stage B, $d=2$'),
              ('C', R4[2], 'o', 3, 'Stage C, $d=3$'), ('E', R4[3], 'o', 8, 'Stage D (Sherpa), $d=8$')]
    fig = plt.figure(figsize=(TW, 2.45))
    gs = GridSpec(1, 2, figure=fig, wspace=0.2, left=0.08, right=0.99, top=0.9, bottom=0.17)
    for i, (key, ttl, ticks, top) in enumerate((('scan', '(a) standard test', [1, 1.5, 2, 3, 4], 5.0),
                                                ('scan_strict', '(b) strict test', [1, 1.5, 2, 3, 5, 8], 9.0))):
        ax = fig.add_subplot(gs[0, i]); ends = []
        for st, col, mk, dd, lab in SERIES:
            sc = rs[st][key]; KK = sorted(int(k) for k in sc)
            yy = [sc[str(k)]['width'] for k in KK]; ee = [sc[str(k)].get('spread', 0) for k in KK]
            ax.errorbar(KK, yy, yerr=ee, marker=mk, ms=3.0, color=col, lw=1.0, elinewidth=0.6, label=lab)
            ends += [np.add(yy, ee), np.subtract(yy, ee), yy]
            K2 = 1 + dd + dd*(dd + 1)//2
            ax.axvline(K2, color=col, ls=(0, (4, 3)), lw=0.7, alpha=0.55, ymax=0.5, zorder=1)
        unit_line(ax)
        # the published range (0.85, 5 or 9), widened only as far as every point and error bar needs (round B10), and
        # so that no error bar ends on the axis (round B21: one within 3 percent of the height of an edge widens it)
        y0, y1 = widen(0.85, top, ends, log=True, clear=0.03)
        ticks = sorted(set(ticks + [t for t in (0.5, 0.6, 0.7) if y0 <= t <= 0.85/1.15]
                           + [t for t in (12, 20, 30, 50, 80, 120, 200) if 1.5*max(ticks) <= t <= y1]))
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(y0, y1)
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
        tk = [1, 2, 3, 4, 8, 16, 32, 48]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
        ax.set_xlabel(r'number of terms $K$')
        if i == 0:
            ax.set_ylabel('closure width'); ax.legend(loc='upper right', fontsize=7.5)
        title(ax, ttl); trim(ax)
    save(fig, 'kscan')


# ---- Fig. cost: the network part of a fit step ------------------------------------------------------
def cost():
    sfx = os.environ.get('COST_SUFFIX', '')                          # _lin: the benchmark of the linear mixture
    bd = os.environ.get('BENCH_DIR', 'output')
    fc, fg = f'{bd}/bench_final_cpu_4{sfx}.json', f'{bd}/bench_final_cuda_12{sfx}.json'
    c, g = json.load(open(fc)), json.load(open(fg))
    lin = linear_inputs((fc, c), (fg, g))
    for f, d, dev, thr in ((fc, c, 'cpu', 4), (fg, g, 'cuda', 12)):      # never one device's run under the other's name
        assert (d.get('device'), d.get('threads')) == (dev, thr), f'{f} records device {d.get("device")!r} with {d.get("threads")!r} threads'
    nc, ng, rest = c['n_network_step_seconds'], g['n_network_step_seconds'], c['b_tilt_and_chi2_seconds']
    gpu = g.get('gpu', 'GPU').replace('NVIDIA ', '')
    ROWS = [('factorized', 'factorized network', METHOD), ('late', r'$\theta$ after the sum', BASE),
            ('concatenation', 'DCTR network', BASE_DARK)]

    def human(s):
        if s >= 60: return f'{s/60:.0f} min'
        if s >= 1: return f'{s:.2g} s'
        if s >= 0.1: return f'{s:.2f} s'
        return f'{1000*s:.2g} ms'
    fig, ax = plt.subplots(figsize=(TW, 1.8))
    y = np.arange(len(ROWS))[::-1].astype(float)
    ax.axvline(rest, color=REF, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.text(rest*1.15, y.max() + 0.42, ('maximum-entropy weights of both generators\non the CPU, the same for every network' if lin
                                        else 'maximum-entropy weight on the CPU,\nthe same for every network'),
            fontsize=7.5, color=REF, va='top', ha='left', linespacing=1.0)
    for yi, (k, lab, col) in zip(y, ROWS):
        ax.plot([ng[k], nc[k]], [yi, yi], color=col, lw=1.0, alpha=0.55, zorder=1)
        ax.plot(ng[k], yi, 'o', ms=5, mfc='white', mec=col, mew=1.2, zorder=3)
        ax.plot(nc[k], yi, 'o', ms=5, color=col, zorder=3)
        for xv in (ng[k], nc[k]):
            ax.text(xv, yi - 0.2, human(xv), ha='center', va='top', fontsize=7.5)
    tt = [v for k, _, _ in ROWS for v in (ng[k], nc[k])] + [rest]
    # the published range 2 ms to 8000 s, widened by a factor two beyond any time outside it (round B10)
    ax.set_xscale('log'); ax.set_xlim(min(2e-3, min(tt)/2), max(8e3, 2*max(tt)))
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _ in ROWS]); ax.set_ylim(y.min() - 0.6, y.max() + 0.5)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(which='both', top=False)
    ax.set_xlabel('time of the network at one parameter point [s]')
    from matplotlib.lines import Line2D
    # above the axes in one row, as in Fig. 5 (round B21: at the lower left the dashed line crossed "threads")
    ax.legend(handles=[Line2D([0], [0], marker='o', lw=0, ms=5, color=REF, label='one CPU process, 4 threads'),
                       Line2D([0], [0], marker='o', lw=0, ms=5, mfc='white', mec=REF, mew=1.2, label=f'one {gpu}')],
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    save(fig, 'cost' + ('_lin' if lin else ''))


# ---- Fig. three bands: the thrust distribution ------------------------------------------------------
def bands():
    """(a) the generators alone and (b) with the calculation imposed, as ratios to the data, each with the
    mixture at its midpoint (solid, with its band) and each generator alone at its default tune (Sherpa
    dashed, Herwig dotted, theory_variations.py). The band of (b) is signed: the combined band above the
    prediction collects the upward theory variations and the one below the downward ones. (c)-(e) each band
    per bin in percent of the prediction, the theory band split into its two sides."""
    fn, fj, fv = (os.environ.get(k, d) for k, d in (('BANDS_NPZ', 'output/three_bands_MIXGEOV3S9.npz'),
                                                    ('BANDS_JSON', 'output/three_bands_MIXGEOV3S9.json'),
                                                    ('VARS_JSON', 'output/theory_variations_MIX.json')))
    D, J, V = np.load(fn), json.load(open(fj)), json.load(open(fv))
    lin = linear_inputs((fn, D), (fj, J), (fv, V))
    # D8: against DELPHI at its hadron level, when both files hold the corrected predictions
    CV = '_conv' if (os.environ.get('FIG_CONV', '1') == '1' and 'h_central_conv' in D.files and 'h_imposed_conv' in V['xi0']) else ''
    # the bins below the window come first when the files have them (keys ending in _low), with the
    # normalization over the bins above tau = 0.05 unchanged
    low = 'lo_low' in D.files and len(D['lo_low']) > 0 and 'lo_low' in V
    cat = (lambda k: np.r_[D[k + '_low'], D[k]]) if low else (lambda k: D[k])
    vcat = (lambda d, k: np.r_[d[k + '_low'], d[k]]) if low else (lambda d, k: np.asarray(d[k]))
    lo, hi, y, e = cat('lo'), cat('hi'), cat('data'), cat('err'); edges = np.r_[lo, hi[-1]]
    assert np.allclose(vcat(V, 'lo'), lo) and np.allclose(vcat(V, 'data'), y)
    WLO, WHI = J['calc']['window']; TEST = J['test'].upper()
    h0, h1 = cat('h_central_before' + CV), cat('h_central' + CV)
    g0, g1, le, le0 = cat('gen_hist_before' + CV), cat('gen_hist' + CV), cat('learn_hist' + CV), cat('learn_hist_before' + CV)
    thu, thd = cat('theory_hist_up' + CV), cat('theory_hist_down' + CV)
    c0, cu, cd = cat('comb_hist_before' + CV), cat('comb_hist_up' + CV), cat('comb_hist_down' + CV)
    st = cat('stat_hist' + CV)
    single = {xi: (vcat(V[f'xi{xi}'], 'h_gen_only' + CV), vcat(V[f'xi{xi}'], 'h_imposed' + CV)) for xi in (0, 1)}
    fig = plt.figure(figsize=(TW, 5.7))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.12, left=0.085, right=0.99,
                  top=0.955, bottom=0.065)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # the vertical range holds every curve, band and data point drawn (below the window the ratio to DELPHI
    # reaches far from one: about 2 in the first bin with the all-particle table)
    YL, YT = nice_range([(h0 + c0)/y, (h0 - c0)/y, (h1 + cu)/y, (h1 - cd)/y, 1 + e/y, 1 - e/y]
                        + [single[xi][k]/y for xi in (0, 1) for k in (0, 1)])
    for row, (ttl, h, up, dn, col, k) in enumerate((
            ('(a) generators alone', h0, c0, c0, BASE, 0),
            ('(b) calculation imposed', h1, cu, cd, METHOD, 1))):
        ax = fig.add_subplot(gs[row, :])
        window(ax, WLO, WHI)
        lc = col if k else BASE_DARK; al = BAND_ALPHA if k else 0.3
        ax.stairs((h + up)/y, edges, baseline=(h - dn)/y, fill=True, color=col, alpha=al, lw=0)
        ax.stairs(h/y, edges, baseline=None, color=lc, lw=1.3 if k else 1.1)
        ax.stairs(single[0][k]/y, edges, baseline=None, color=lc, lw=1.0, ls=(0, (4, 2)))
        ax.stairs(single[1][k]/y, edges, baseline=None, color=lc, lw=1.2, ls=(0, (1, 1)))
        eb = ax.errorbar(0.5*(lo + hi), y/y, yerr=e/y, fmt='o', ms=2.6, lw=0.8, color=DATA, zorder=5)
        ax.set_xlim(0, edges[-1]); ax.set_ylim(*YL)
        ax.set_yticks(YT)
        ax.set_ylabel(f'ratio to {TEST}')
        if row == 0:
            # 4.5 points above the axis, clear of the inward ticks of 3 points (round B21: the tick at 0.2 cut into it)
            ax.annotate('window of the calculation', xy=(0.5*(WLO + WHI), 0), xycoords=('data', 'axes fraction'), xytext=(0, 4.5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=7, color=REF)
        hs = [(Patch(color=col, alpha=al, lw=0), Line2D([], [], color=lc, lw=1.2)),
              Line2D([], [], color=lc, lw=1.0, ls=(0, (4, 2))), Line2D([], [], color=lc, lw=1.2, ls=(0, (1, 1))), eb]
        ls_ = [r'mixture at $\xi=\frac12$ with its band',
               'Sherpa alone', 'Herwig alone', TEST]
        # in two columns at the top of the window, the one region that no curve or band reaches
        ax.legend(hs, ls_, loc='upper left', fontsize=7, ncol=2, handlelength=1.9,
                  bbox_to_anchor=((0.095 if low else 0.0)/edges[-1], 1.0), columnspacing=1.2, frameon=False)
        title(ax, ttl); trim(ax)
        ax.set_xlabel(r'$\tau=1-T$', labelpad=1.5)
    rel = lambda b, p: 100*b/np.maximum(p, 1e-12)
    axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    curves = [
        ('(c) generator band', [(rel(g0, h0), BASE, '-', 'generators alone'), (rel(g1, h1), GEN, '-', 'calculation imposed')]),
        ('(d) theory band', [(rel(thu, h1), THEORY, '-', 'above the prediction'), (rel(thd, h1), THEORY, (0, (3, 1.5)), 'below the prediction')]),
        ('(e) learning band', [(rel(le0, h0), BASE, '-', 'generators alone'), (rel(le, h1), LEARN, '-', 'calculation imposed'),
                               (rel(st, h1), METHOD, (0, (1, 1.2)), 'statistical error')])]
    # the published range 0.003 to 1000 percent for all three panels, widened only if a band leaves it (round B10)
    PL = widen(3e-3, 1e3, [v for _, cs in curves for v, _, _, _ in cs], log=True)
    DEC = [10.0**k for k in range(-12, 13) if 2*PL[0] <= 10.0**k <= PL[1]/5]      # 0.01 to 100 in the published range
    for j, (a, (ttl, cs)) in enumerate(zip(axes, curves)):
        window(a, WLO, WHI)
        for v, col, ls, lab in cs:
            a.stairs(v, edges, baseline=None, color=col, lw=1.2, ls=ls, label=lab)
        a.set_xlim(0, edges[-1]); a.set_yscale('log'); a.set_ylim(*PL)
        a.yaxis.set_major_locator(FixedLocator(DEC))
        a.yaxis.set_minor_locator(NullLocator())
        a.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in DEC]))
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
        a.set_xlabel(r'$\tau$', labelpad=1.5); title(a, ttl); trim(a)
        a.legend(loc='upper right' if j == 1 else 'upper left', fontsize=7, ncol=1, handlelength=1.4)
    axes[0].set_ylabel('percent of prediction', fontsize=8.5)
    save(fig, 'bands' + ('_lin' if lin else ''))


# ---- Fig. three bands of the means -------------------------------------------------------------------
def obs():
    """The bands of seven means in percent of the prediction with the calculation imposed. The theory band
    is signed, so it has two bars, its side above the prediction (light) and below it (dark)."""
    fj = os.environ.get('BANDS_JSON', 'output/three_bands_MIXGEOV3S9.json'); J = json.load(open(fj))
    lin = linear_inputs((fj, J))
    c = J['central']; g = J['generator_band']
    OBS = [('tau_win', r'$\langle\tau\rangle_{\rm win}$'), ('pwin', r'$P_{\rm win}$'), ('thrust', r'$\langle 1-T\rangle$'),
           ('B_total', r'$\langle B_{\rm tot}\rangle$'), ('rho_heavy', r'$\langle\rho_H\rangle$'),
           ('nch', r'$\langle n_{\rm ch}\rangle$'), ('nbaryon', r'$\langle n_{\rm b}\rangle$')]
    keys = [k for k, _ in OBS]; pred = np.array([c['imposed'][k] for k in keys])
    pct = lambda v: 100*np.array([v[k] for k in keys])/np.abs(pred)
    gen_before = 100*np.array([g['before'][k]['sd'] for k in keys])/np.abs(np.array([c['generator_only'][k] for k in keys]))
    gen = 100*np.array([g['after'][k]['sd'] for k in keys])/np.abs(pred)
    thu, thd, learn = pct(c['theory_up']), pct(c['theory_down']), pct(c['learning_band'])
    # the published axis from 0.001 to 300 percent. Below it may lie only the two bands of tau_win that the calculation
    # removes: its learning band (the caption of Fig. 12, main.tex 1758) and its generator band with the calculation
    # imposed (main.tex 1706: "its generator band disappears"). Rounds B10 and B11: any other bar or tick below 0.001, or
    # above 300, widens the axis; these two never lower its floor, and the log says where both lie
    FLOOR, TOP = 1e-3, 300.0
    bars = dict(generator=gen, generators_alone=gen_before, theory_above=thu, theory_below=thd, learning=np.nan_to_num(learn))
    TAU = (('learning', 'tau_win'), ('generator', 'tau_win'))
    others = [v for n, a in bars.items() for k, v in zip(keys, a) if v > 0 and ((n, k) not in TAU or v >= FLOOR)]
    lo_, TOP = widen(FLOOR, TOP, others, log=True)
    if lo_ < FLOOR:
        FLOOR = 10.0**np.floor(np.log10(lo_))
    tw = float(np.nan_to_num(learn[keys.index('tau_win')]))
    print(f'obs: the learning band of tau_win is {tw:.2g} percent of the prediction: ' +
          ('below the axis, as the caption of Fig. 12 says' if tw < FLOOR else f'inside the axis (from {FLOOR:g}): the caption sentence must go'))
    tg = float(np.nan_to_num(gen[keys.index('tau_win')]))
    print(f'obs: the generator band of tau_win with the calculation imposed is {tg:.2g} percent of the prediction: ' +
          (f'below the axis (from {FLOOR:g})' if tg < FLOOR else f'a bar inside the axis (from {FLOOR:g}); main.tex 1706 '
           'says it disappears (published: 0.0027 percent, a short bar)'))
    x = np.arange(len(keys)); wdt = 0.19
    fig, ax = plt.subplots(figsize=(TW, 2.3))
    ax.bar(x - 1.5*wdt, np.maximum(gen, FLOOR), wdt, color=GEN, lw=0, label='generator', zorder=3)
    ax.bar(x - 0.5*wdt, np.maximum(thu, FLOOR), wdt, color=THEORY, alpha=0.45, lw=0, label='theory, above', zorder=3)
    ax.bar(x + 0.5*wdt, np.maximum(thd, FLOOR), wdt, color=THEORY, lw=0, label='theory, below', zorder=3)
    ax.bar(x + 1.5*wdt, np.maximum(np.nan_to_num(learn), FLOOR), wdt, color=LEARN, lw=0, label='learning', zorder=3)
    for xi, v in zip(x, gen_before):
        ax.hlines(v, xi - 2*wdt, xi - wdt, color=BASE_DARK, lw=1.6, zorder=4)
    ax.hlines([], [], [], color=BASE_DARK, lw=1.6, label='generator, generators alone')
    ax.set_yscale('log'); ax.set_ylim(FLOOR, TOP)
    DEC = [10.0**k for k in range(-12, 13) if 5*FLOOR <= 10.0**k <= TOP/2]          # 0.01 to 100 in the published axis
    ax.yaxis.set_major_locator(FixedLocator(DEC)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in DEC]))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in OBS], fontsize=8.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('percent of prediction', fontsize=8.5)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.16), fontsize=7.5, handlelength=1.3, columnspacing=1.0)
    trim(ax)
    save(fig, 'obs' + ('_lin' if lin else ''))


# ---- Fig. coverage: does the learning band cover the error of the prediction? ----------------------
def coverage():
    """Pulls of the reweighted means against the fresh runs at the 48 held-out points of the strict test,
    for the three sets of members together (144 points per observable), without and with the learning
    band; the numbers are the root mean square over all of them. The vertical range comes from the data, as in
    bands(): the published range (-10.9, 8.6) when every point fits in it, else wider, so that no point is hidden;
    the row of rms values stays at the bottom, below the lowest point."""
    files = os.environ.get('COV_IN', 'output/wifi_coverage_MIXGEOV3S9.json output/wifi_coverage_MIXGEOV3S9s1.json '
                                     'output/wifi_coverage_MIXGEOV3S9s2.json').split()
    d = dict(points=[q for f in files for q in json.load(open(f))['points']])
    OBS = [('thrust', r'$\langle 1-T\rangle$'), ('B_total', r'$\langle B_{\rm tot}\rangle$'), ('rho_heavy', r'$\langle\rho_H\rangle$'),
           ('nch', r'$\langle n_{\rm ch}\rangle$'), ('nbaryon', r'$\langle n_{\rm b}\rangle$')]
    za = np.array([q[o][k] for q in d['points'] for o, _ in OBS for k in ('z_without', 'z_with')], float)
    za = za[np.isfinite(za)]
    # the published range, the rms row 0.3 above its bottom, and its lowest point at -9.3 at most, kept as fractions
    # of the height (the row's height in points is fixed): the top clears the highest point by 0.4, and the bottom
    # goes down until the lowest point lies that fraction up, above the row
    BOT, TOP = -10.9, 8.6; ROW, CLEAR = 0.3/(TOP - BOT), (-9.3 - BOT)/(TOP - BOT)
    hi = max(TOP, float(za.max()) + 0.4) if len(za) else TOP
    lo = min(BOT, (float(za.min()) - CLEAR*hi)/(1 - CLEAR)) if len(za) else BOT
    yrow = lo + ROW*(hi - lo)
    st = next(s for s in (2, 5, 10, 20, 50, 100, 200) if (hi - lo)/s <= 12)
    t_lo = -st*np.floor(6/st) if lo >= BOT else st*np.ceil((lo + 1.2*CLEAR*(hi - lo))/st)
    t_hi = st*np.floor(6/st) if hi <= TOP else st*np.floor((hi - 0.3)/st)
    fig, ax = plt.subplots(figsize=(TW, 2.3))
    rng = np.random.default_rng(5)
    ax.axhspan(-1, 1, color=BASE, alpha=0.12, lw=0, zorder=0)
    for y in (-2, 2):
        ax.axhline(y, color=REF, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=REF, lw=0.5, zorder=1)
    for i, (o, lab) in enumerate(OBS):
        zw = np.array([q[o]['z_without'] for q in d['points']]); zb = np.array([q[o]['z_with'] for q in d['points']])
        ax.scatter(i - 0.17 + rng.uniform(-0.1, 0.1, len(zw)), zw, s=5, color=BASE, lw=0, alpha=0.8, zorder=3,
                   label='statistical uncertainty only' if i == 0 else None)
        ax.scatter(i + 0.17 + rng.uniform(-0.1, 0.1, len(zb)), zb, s=5, color=LEARN, lw=0, alpha=0.9, zorder=3,
                   label='with the learning band' if i == 0 else None)
        for x0, z, col in ((i - 0.17, zw, BASE_DARK), (i + 0.17, zb, LEARN_TEXT)):
            ax.text(x0, yrow, rf'${np.sqrt(np.mean(z**2)):.2f}$', ha='center', va='bottom', fontsize=7.5, color=col)
    # inside the axes, 3 points from its left edge (round B21: it overlapped the axis line)
    ax.annotate('rms', xy=(0, yrow), xycoords=('axes fraction', 'data'), xytext=(3, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=7.5, color=REF)
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([l for _, l in OBS], fontsize=8.5)
    ax.tick_params(axis='x', length=0); ax.set_xlim(-0.8, len(OBS) - 0.45); ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(np.arange(t_lo, t_hi + 1e-9, st)))
    ax.set_ylabel('pull')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=2, markerscale=1.6)
    trim(ax)
    save(fig, 'coverage')


# ---- Fig. the calculation against ALEPH and DELPHI (App. C) --------------------------------------
def aleph():
    """The NNLL+NLO calculation with the dispersive shift at the input values of the demonstration, with its
    theory band (Eq. thband: each scale pair once per side with its larger displacement, the change of scheme once, the three
    dispersive-model variations as their root mean square, each side kept apart), against ALEPH and
    DELPHI, both normalized to the total rate as published (calc_bins.py)."""
    J = json.load(open(os.environ.get('CALC_BINS', 'output/calc_bins_nlo_point.json')))
    WLO, WHI = J['window']
    PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
    SCH = "(1.0, 1.0, 'modR', 1.0)"; NPV = ['milan_up', 'milan_dn', 'sub2', 'sub3']   # those present are used

    def band(c):
        cen = np.array(c['central']); dv = {k: np.array(v) - cen for k, v in c.items() if k != 'central'}
        def side(f):
            t = sum(np.maximum(f(dv[a]), f(dv[b]))**2 for a, b in PAIRS) + f(dv[SCH])**2
            npv = [k for k in NPV if k in dv]
            return np.sqrt(t + sum(f(dv[k])**2 for k in npv)/len(npv))
        return cen, side(lambda v: np.maximum(v, 0)), side(lambda v: np.maximum(-v, 0))

    A, Dl = J['aleph'], J['delphi']
    lo, hi = np.array(A['lo']), np.array(A['hi']); edges = np.r_[lo, hi[-1]]
    assert np.allclose(lo[1:], hi[:-1])
    cen, up, dn = band(A['calc'])
    ctr = 0.5*(lo + hi); inw = (ctr >= WLO) & (ctr < WHI)
    ya, ea = np.array(A['y']), np.hypot(A['stat'], A['sys'])
    ld, hd = np.array(Dl['lo']), np.array(Dl['hi']); cd = np.array(Dl['calc']['central'])
    yd, ed = np.array(Dl['y']), np.hypot(Dl['stat'], Dl['sys'])
    fig = plt.figure(figsize=(TW, 3.9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1], hspace=0.08, left=0.085, right=0.99, top=0.975, bottom=0.1)
    ax, ar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    for a in (ax, ar):
        window(a, WLO, WHI)
    ax.stairs(np.where(inw, cen + up, np.nan), edges, baseline=np.where(inw, cen - dn, np.nan), fill=True, color=THEORY, alpha=BAND_ALPHA, lw=0)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.stairs(np.where(inw, cen, np.nan), edges, baseline=None, color=THEORY, lw=1.3)
    out = (~inw) & (lo >= 0.02)                   # outside the window, without the region the shift empties
    ax.stairs(np.where(out, cen, np.nan), edges, baseline=None, color=THEORY, lw=1.0, ls=(0, (3, 2)))
    ha = ax.errorbar(ctr, ya, yerr=ea, fmt='o', ms=2.8, lw=0.8, mfc='white', mec=DATA, ecolor=DATA, mew=0.8, zorder=5)
    hd_ = ax.errorbar(0.5*(ld + hd), yd, yerr=ed, fmt='o', ms=2.6, lw=0.8, color=DATA, zorder=6)
    # rounds B10 and B11: the published ranges, widened for a data point with its error bar (the positive ends on the
    # log axis) or the calculation inside the window; round B21: the upper panel for every data point below tau = 0.5
    # with its error bar too (the ALEPH point at 0.405 and the bar at 0.395 were cut at the edge), and the ratio panel
    # for the theory band. A data point outside the window and beyond the range of a panel (the tail above tau = 1/3 in
    # the ratio panel) is drawn as an arrow at its edge, so none is hidden
    cdc = 0.5*(ld + hd); inwd = (cdc >= WLO) & (cdc < WHI); ina, ind = ctr < 0.5, cdc < 0.5
    UL = widen(2e-3, 40, [ya[ina], (ya + ea)[ina], (ya - ea)[ina], yd[ind], (yd + ed)[ind], (yd - ed)[ind], cen[inw]],
               log=True)
    ax.set_yscale('log'); ax.set_xlim(0, 0.5); ax.set_ylim(*UL)
    edge_arrows(ax, ctr[~inw & (ctr < 0.5)], ya[~inw & (ctr < 0.5)], *UL, log=True, mfc='white', mec=DATA, mew=0.8)
    edge_arrows(ax, cdc[~inwd & (cdc < 0.5)], yd[~inwd & (cdc < 0.5)], *UL, log=True, color=DATA)
    ax.set_ylabel(r'$\frac{1}{\sigma}\frac{d\sigma}{d\tau}$')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.text(0.5*(WLO + WHI), 22, 'window of the calculation', ha='center', fontsize=7.5, color=REF)
    ax.legend([(Patch(color=THEORY, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=THEORY, lw=1.3)),
               Line2D([], [], color=THEORY, lw=1.0, ls=(0, (3, 2))), ha, hd_],
              ['calculation and its theory band', 'calculation outside the window',
               'ALEPH', 'DELPHI'], loc='lower left', fontsize=7.5, handlelength=2.0)
    trim(ax)
    # ratio panel: data over the calculation, inside the window and up to tau = 0.4
    ar.stairs(np.where(inw, 1 + up/cen, np.nan), edges, baseline=np.where(inw, 1 - dn/cen, np.nan), fill=True, color=THEORY, alpha=BAND_ALPHA, lw=0)
    unit_line(ar)
    rea = ar.errorbar(ctr - 0.002, ya/cen, yerr=ea/cen, fmt='o', ms=2.8, lw=0.8, mfc='white', mec=DATA, ecolor=DATA, mew=0.8, zorder=5)
    red = ar.errorbar(0.5*(ld + hd) + 0.002, yd/cd, yerr=ed/cd, fmt='o', ms=2.6, lw=0.8, color=DATA, zorder=6)
    with np.errstate(divide='ignore', invalid='ignore'):          # the shift empties the calculation's first bin
        ra, rd = ya/cen, yd/cd
        RL = widen(0.7, 1.3, [ra[inw], ((ya + ea)/cen)[inw], ((ya - ea)/cen)[inw], rd[inwd], ((yd + ed)/cd)[inwd],
                              ((yd - ed)/cd)[inwd], (1 + up/cen)[inw], (1 - dn/cen)[inw]])
    ar.set_xlim(0, 0.5); ar.set_ylim(*RL)
    # the marker of a point beyond the range, which edge_arrows draws as an arrow, is left out and its error bar kept
    # (round B21: the ALEPH point at 0.375 lies just below the widened range and showed half at the edge)
    for eb, r in ((rea, ra), (red, rd)):
        eb[0].set_ydata(np.where((r >= RL[0]) & (r <= RL[1]), r, np.nan))
    # at most seven steps over any span (round B11: steps up to 20, and beyond them 1, 2 or 5 times a power of ten);
    # the ticks are the multiples of the step inside the range
    span = RL[1] - RL[0]
    st = next((s for s in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0) if span/s <= 7), None)
    if st is None:
        e = 10.0**np.floor(np.log10(span/7)); st = next(m*e for m in (1.0, 2.0, 5.0, 10.0) if span/(m*e) <= 7)
    ar.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2] if RL == (0.7, 1.3) else
                  [t for t in np.round(np.arange(np.ceil(RL[0]/st)*st, RL[1], st), 1) if RL[0] + 0.3*st <= t <= RL[1] - 0.3*st])
    edge_arrows(ar, (ctr - 0.002)[~inw & (ctr < 0.5)], ra[~inw & (ctr < 0.5)], *RL, mfc='white', mec=DATA, mew=0.8)
    edge_arrows(ar, (cdc + 0.002)[~inwd & (cdc < 0.5)], rd[~inwd & (cdc < 0.5)], *RL, color=DATA)
    ar.set_xlabel(r'$\tau=1-T$'); ar.set_ylabel('data / calculation'); trim(ar)
    save(fig, 'aleph')


# ---- Fig. toy: the Gaussian toy with a known answer ----------------------------------------------
def toy():
    z = np.load(os.environ.get('TOY_NPZ', 'output/toy_final_K48_s0.npz'))
    J = json.load(open(os.environ.get('TOY_JSON', 'output/toy_final_K48_s0.json')))
    fig = plt.figure(figsize=(TW, 3.75))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.3, left=0.09, right=0.99, top=0.94, bottom=0.11)

    def density(ax, xe, xl, xlab, ylab, slope, corr, ttl):
        lo, hi = np.percentile(np.r_[xe, xl], [0.3, 99.7])
        hb = ax.hexbin(xe, xl, gridsize=45, extent=(lo, hi, lo, hi), bins='log', cmap='Blues', mincnt=1, lw=0, zorder=2)
        # the hexagons on the edge of this range are centred on it, so the axes reach the larger half-size of a hexagon
        # beyond it, in x and y alike, and the diagonal with them (round B21: the corner hexagons were cut)
        p = float(np.abs(hb.get_paths()[0].vertices).max()); lo, hi = lo - p, hi + p
        ax.plot([lo, hi], [lo, hi], color=REF, lw=0.7, ls=(0, (4, 3)), zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.text(0.97, 0.06, rf'slope ${slope:.3f}$, correlation ${corr:.4f}$', transform=ax.transAxes, ha='right',
                fontsize=7.5, color=REF)
        title(ax, ttl); trim(ax)
    density(fig.add_subplot(gs[0, 0]), z['f_exact'], z['f_learned'], r'exact $f_{\rm exact}$', r'learned $f$',
            J['slope'], J['corr'], r'(a) log ratio at $\log s=0.21$')
    density(fig.add_subplot(gs[0, 1]), z['df_exact'], z['df_learned'], r'exact $\Delta f_{\rm exact}$', r'learned $\Delta f$',
            J['dslope'], J['dcorr'], r'(b) difference between $\log s=\pm0.21$')
    s, coef, ex = z['s_grid'], z['coef'], z['coef_exact']
    for j, (ttl, ylab) in enumerate(((r'(c) constant term', r'coefficient of $1$'),
                                     (r'(d) term in $\sum_i x_i^2$', r'coefficient of $\sum_i x_i^2$'))):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(s, ex[:, j], color=DATA, lw=1.0, label='exact')
        ax.plot(s, coef[:, j], 'o', color=METHOD, ms=3.0, label='learned')
        ax.axhline(0, color=REF, lw=0.5); ax.axvline(0, color=REF, lw=0.5, ls=(0, (1, 2)))
        ax.set_xlabel(r'parameter $\log s$'); ax.set_ylabel(ylab)
        if j == 0:
            ax.legend(loc='upper right', fontsize=7.5)
        title(ax, ttl); trim(ax)
    save(fig, 'toy')


# the data figures of the paper, Figs. 3 to 7 and 11 to 14 in order: what runs without arguments
PAPER = ['closure_mix', 'map', 'coverage', 'kscan', 'cost', 'bands', 'obs', 'toy', 'aleph']


if __name__ == '__main__':
    funcs = {'closure': closure, 'map': map_, 'widebox': widebox, 'kscan': kscan, 'cost': cost, 'bands': bands, 'obs': obs, 'coverage': coverage, 'closure_mix': closure_mix,
             'toy': toy, 'aleph': aleph}
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__); sys.exit(0)
    names = sys.argv[1:] or PAPER
    bad = [n for n in names if n not in funcs]
    if bad:
        sys.exit(f'figs_r8.py: no figure {", ".join(bad)}. The figures are {", ".join(funcs)}.')
    style()
    for n in names:
        funcs[n]()
