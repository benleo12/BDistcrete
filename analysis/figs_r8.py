#!/usr/bin/env python3
"""Every data figure of the r8 paper at its printed size, in one style.

Figures are drawn at the width they have on the page (TW = 5.95 in, the JHEP text width, or a stated
fraction of it), so the fonts are the printed fonts: 9 pt axis labels and panel titles, 8 pt tick
labels, 7.5 pt legends and annotations. Each color keeps one meaning across the paper, as set out
in the block of constants below.

    python figs_r8.py               # the ten data figures of the paper, output/r8_<name>.pdf
    python figs_r8.py bands obs     # only the figures named

The ten, in the order of the paper: closure_mix (Fig. 3), toy (Fig. 4), map (Fig. 5), widebox
(Fig. 6), coverage (Fig. 7), kscan (Fig. 8), cost (Fig. 9), bands (Fig. 10), obs (Fig. 11) and
aleph (Fig. 12). closure (Stage C reweighted to a held-out point) is not in the paper and is drawn
only when named. R8_OUT sets the output folder (default output). XP_NULL sets the dotted tick of
Fig. 5 (default the mean of output/xp_null.json).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter, LogLocator, NullFormatter
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
            if logy:
                axd.set_yscale('log'); axd.set_ylim(max(dt[dt > 0].min()*0.3, 1e-6), None)
            if obs == 'mult_total':
                axd.set_xlim(0, 80); axd.set_ylim(0, 1.3*np.nanmax(dt))
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
            axr.set_ylim(1 - lim, 1 + lim)
            if obs == 'mult_total':
                axr.set_xlim(0, 80)
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
                   label='twelve blind runs' if i == 0 else None)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vb)), vb, s=12, marker='D', facecolor='white', edgecolor=METHOD,
                   lw=0.8, zorder=4, label='five held-out points' if i == 0 else None)
        ax.plot([i - 0.26, i + 0.26], [np.nanmean(va)]*2, color=METHOD_DARK, lw=1.5, zorder=5,
                label='mean of the blind runs' if i == 0 else None)
        if o == 'x_p' and np.isfinite(xp_null):
            ax.plot([i - 0.3, i + 0.3], [xp_null]*2, color=REF, lw=0.9, ls=(0, (1.2, 1.2)), zorder=2)
    unit_line(ax)
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([l for _, l in OBS], fontsize=7.5)
    ax.set_xlim(-0.6, len(OBS) - 0.4)
    ax.set_ylabel(r'closure width'); ax.legend(loc='upper left', fontsize=7.5)
    title(ax, '(a) closure width per observable'); trim(ax)
    bx = fig.add_subplot(gs[0, 1])
    norm = [(0.120, 0.008), (0.475, 0.175), (1.30, 0.50)]
    for coll, mk, fc, lab, s in ((pts, 'o', METHOD, 'blind runs', 10), (anc, 'D', 'white', 'held-out points', 13)):
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
        ax = fig.add_subplot(gs[0, i])
        for st, col, mk, dd, lab in SERIES:
            sc = rs[st][key]; KK = sorted(int(k) for k in sc)
            yy = [sc[str(k)]['width'] for k in KK]; ee = [sc[str(k)].get('spread', 0) for k in KK]
            ax.errorbar(KK, yy, yerr=ee, marker=mk, ms=3.0, color=col, lw=1.0, elinewidth=0.6, label=lab)
            K2 = 1 + dd + dd*(dd + 1)//2
            ax.axvline(K2, color=col, ls=(0, (4, 3)), lw=0.7, alpha=0.55, ymax=0.5, zorder=1)
        unit_line(ax)
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(0.85, top)
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
    def load(dev):
        f = {'cpu': 'output/bench_final_cpu_4.json', 'gpu': 'output/bench_final_cuda_12.json'}[dev]
        return json.load(open(f))
    c, g = load('cpu'), load('gpu')
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
    ax.text(rest*1.15, y.max() + 0.42, 'maximum-entropy weight on the CPU,\nthe same for every network',
            fontsize=7.5, color=REF, va='top', ha='left', linespacing=1.0)
    for yi, (k, lab, col) in zip(y, ROWS):
        ax.plot([ng[k], nc[k]], [yi, yi], color=col, lw=1.0, alpha=0.55, zorder=1)
        ax.plot(ng[k], yi, 'o', ms=5, mfc='white', mec=col, mew=1.2, zorder=3)
        ax.plot(nc[k], yi, 'o', ms=5, color=col, zorder=3)
        for xv in (ng[k], nc[k]):
            ax.text(xv, yi - 0.2, human(xv), ha='center', va='top', fontsize=7.5)
    ax.set_xscale('log'); ax.set_xlim(2e-3, 8e3)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _ in ROWS]); ax.set_ylim(y.min() - 0.6, y.max() + 0.5)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(which='both', top=False)
    ax.set_xlabel('time of the network at one parameter point [s]')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker='o', lw=0, ms=5, color=REF, label='one CPU process, 4 threads'),
                       Line2D([0], [0], marker='o', lw=0, ms=5, mfc='white', mec=REF, mew=1.2, label=f'one {gpu}')],
              loc='lower left', fontsize=7.5)
    save(fig, 'cost')


# ---- Fig. three bands: the thrust distribution ------------------------------------------------------
def bands():
    """(a) the generators alone and (b) with the calculation imposed, as ratios to the data, each with the
    mixture at its midpoint (solid, with its band) and each generator alone at its default tune (Sherpa
    dashed, Herwig dotted, theory_variations.py). The band of (b) is signed: the combined band above the
    prediction collects the upward theory variations and the one below the downward ones. (c)-(e) each band
    per bin in percent of the prediction, the theory band split into its two sides."""
    D = np.load(os.environ.get('BANDS_NPZ', 'output/three_bands_MIXGEOV3S9.npz'))
    J = json.load(open(os.environ.get('BANDS_JSON', 'output/three_bands_MIXGEOV3S9.json')))
    V = json.load(open(os.environ.get('VARS_JSON', 'output/theory_variations_MIX.json')))
    # the bins below the window come first when the files have them (keys ending in _low), with the
    # normalization over the bins above tau = 0.05 unchanged
    low = 'lo_low' in D.files and len(D['lo_low']) > 0 and 'lo_low' in V
    cat = (lambda k: np.r_[D[k + '_low'], D[k]]) if low else (lambda k: D[k])
    vcat = (lambda d, k: np.r_[d[k + '_low'], d[k]]) if low else (lambda d, k: np.asarray(d[k]))
    lo, hi, y, e = cat('lo'), cat('hi'), cat('data'), cat('err'); edges = np.r_[lo, hi[-1]]
    assert np.allclose(vcat(V, 'lo'), lo) and np.allclose(vcat(V, 'data'), y)
    WLO, WHI = J['calc']['window']; TEST = J['test'].upper()
    h0, h1 = cat('h_central_before'), cat('h_central')
    g0, g1, le, le0 = cat('gen_hist_before'), cat('gen_hist'), cat('learn_hist'), cat('learn_hist_before')
    thu, thd = cat('theory_hist_up'), cat('theory_hist_down')
    c0, cu, cd = cat('comb_hist_before'), cat('comb_hist_up'), cat('comb_hist_down')
    st = cat('stat_hist')
    single = {xi: (vcat(V[f'xi{xi}'], 'h_gen_only'), vcat(V[f'xi{xi}'], 'h_imposed')) for xi in (0, 1)}
    fig = plt.figure(figsize=(TW, 5.7))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.12, left=0.085, right=0.99,
                  top=0.955, bottom=0.065)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # the bins below the window reach 0.46 to 1.29 of DELPHI, so the range takes them in
    YL = (0.42, 1.42) if low else (0.68, 1.42)
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
        ax.set_yticks([0.6, 0.8, 1.0, 1.2] if low else [0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_ylabel(f'ratio to {TEST}')
        if row == 0:
            ax.text(0.5*(WLO + WHI), YL[0] + 0.012, 'window of the calculation', ha='center', va='bottom', fontsize=7, color=REF)
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
    for j, (a, (ttl, cs)) in enumerate(zip(axes, curves)):
        window(a, WLO, WHI)
        for v, col, ls, lab in cs:
            a.stairs(v, edges, baseline=None, color=col, lw=1.2, ls=ls, label=lab)
        a.set_xlim(0, edges[-1]); a.set_yscale('log'); a.set_ylim(3e-3, 1e3)
        a.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
        a.yaxis.set_minor_locator(NullLocator())
        a.yaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '10', '100']))
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
        a.set_xlabel(r'$\tau$', labelpad=1.5); title(a, ttl); trim(a)
        a.legend(loc='upper right' if j == 1 else 'upper left', fontsize=7, ncol=1, handlelength=1.4)
    axes[0].set_ylabel('percent of prediction', fontsize=8.5)
    save(fig, 'bands')


# ---- Fig. three bands of the means -------------------------------------------------------------------
def obs():
    """The bands of seven means in percent of the prediction with the calculation imposed. The theory band
    is signed, so it has two bars, its side above the prediction (light) and below it (dark)."""
    J = json.load(open(os.environ.get('BANDS_JSON', 'output/three_bands_MIXGEOV3S9.json')))
    c = J['central']; g = J['generator_band']
    OBS = [('tau_win', r'$\langle\tau\rangle_{\rm win}$'), ('pwin', r'$P_{\rm win}$'), ('thrust', r'$\langle 1-T\rangle$'),
           ('B_total', r'$\langle B_{\rm tot}\rangle$'), ('rho_heavy', r'$\langle\rho_H\rangle$'),
           ('nch', r'$\langle n_{\rm ch}\rangle$'), ('nbaryon', r'$\langle n_{\rm b}\rangle$')]
    keys = [k for k, _ in OBS]; pred = np.array([c['imposed'][k] for k in keys])
    pct = lambda v: 100*np.array([v[k] for k in keys])/np.abs(pred)
    gen_before = 100*np.array([g['before'][k]['sd'] for k in keys])/np.abs(np.array([c['generator_only'][k] for k in keys]))
    gen = 100*np.array([g['after'][k]['sd'] for k in keys])/np.abs(pred)
    thu, thd, learn = pct(c['theory_up']), pct(c['theory_down']), pct(c['learning_band'])
    FLOOR = 1e-3; x = np.arange(len(keys)); wdt = 0.19
    fig, ax = plt.subplots(figsize=(TW, 2.3))
    ax.bar(x - 1.5*wdt, np.maximum(gen, FLOOR), wdt, color=GEN, lw=0, label='generator', zorder=3)
    ax.bar(x - 0.5*wdt, np.maximum(thu, FLOOR), wdt, color=THEORY, alpha=0.45, lw=0, label='theory, above', zorder=3)
    ax.bar(x + 0.5*wdt, np.maximum(thd, FLOOR), wdt, color=THEORY, lw=0, label='theory, below', zorder=3)
    ax.bar(x + 1.5*wdt, np.maximum(np.nan_to_num(learn), FLOOR), wdt, color=LEARN, lw=0, label='learning', zorder=3)
    for xi, v in zip(x, gen_before):
        ax.hlines(v, xi - 2*wdt, xi - wdt, color=BASE_DARK, lw=1.6, zorder=4)
    ax.hlines([], [], [], color=BASE_DARK, lw=1.6, label='generator, generators alone')
    ax.set_yscale('log'); ax.set_ylim(FLOOR, 300)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '10', '100']))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in OBS], fontsize=8.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('percent of prediction', fontsize=8.5)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.16), fontsize=7.5, handlelength=1.3, columnspacing=1.0)
    trim(ax)
    save(fig, 'obs')


# ---- Fig. coverage: does the learning band cover the error of the prediction? ----------------------
def coverage():
    """Pulls of the reweighted means against the fresh runs at the 48 held-out points of the strict test,
    for the three sets of members together (144 points per observable), without and with the learning
    band; the numbers are the root mean square over all of them."""
    files = os.environ.get('COV_IN', 'output/wifi_coverage_MIXGEOV3S9.json output/wifi_coverage_MIXGEOV3S9s1.json '
                                     'output/wifi_coverage_MIXGEOV3S9s2.json').split()
    d = dict(points=[q for f in files for q in json.load(open(f))['points']])
    OBS = [('thrust', r'$\langle 1-T\rangle$'), ('B_total', r'$\langle B_{\rm tot}\rangle$'), ('rho_heavy', r'$\langle\rho_H\rangle$'),
           ('nch', r'$\langle n_{\rm ch}\rangle$'), ('nbaryon', r'$\langle n_{\rm b}\rangle$')]
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
            ax.text(x0, -10.6, rf'${np.sqrt(np.mean(z**2)):.2f}$', ha='center', va='bottom', fontsize=7.5, color=col)
    ax.text(-0.6, -10.6, 'rms', ha='right', va='bottom', fontsize=7.5, color=REF)
    lo_pt = min(min(q[o]['z_without'] for q in d['points']) for o, _ in OBS); assert lo_pt > -9.3, lo_pt
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([l for _, l in OBS], fontsize=8.5)
    ax.tick_params(axis='x', length=0); ax.set_xlim(-0.8, len(OBS) - 0.45); ax.set_ylim(-10.9, 8.6)
    ax.yaxis.set_major_locator(FixedLocator([-6, -4, -2, 0, 2, 4, 6]))
    ax.set_ylabel('pull')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=2, markerscale=1.6)
    trim(ax)
    save(fig, 'coverage')


# ---- Fig. the calculation against ALEPH and DELPHI (App. C) --------------------------------------
def aleph():
    """The NNLL+NLO calculation with the dispersive shift at the input values of the demonstration, with its
    theory band (Eq. thband: the two scale pairs at half weight, the change of scheme once, the three
    dispersive-model variations as their root mean square, each side kept apart), against ALEPH and
    DELPHI, both normalized to the total rate as published (calc_bins.py)."""
    J = json.load(open(os.environ.get('CALC_BINS', 'output/calc_bins_nlo_point.json')))
    WLO, WHI = J['window']
    PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
    SCH = "(1.0, 1.0, 'modR', 1.0)"; NPV = ['milan_up', 'milan_dn', 'sub3']

    def band(c):
        cen = np.array(c['central']); dv = {k: np.array(v) - cen for k, v in c.items() if k != 'central'}
        def side(f):
            t = sum(0.5*f(dv[a])**2 + 0.5*f(dv[b])**2 for a, b in PAIRS) + f(dv[SCH])**2
            return np.sqrt(t + sum(f(dv[k])**2 for k in NPV)/3)
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
    ax.set_yscale('log'); ax.set_xlim(0, 0.5); ax.set_ylim(2e-3, 40)
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
    ar.errorbar(ctr - 0.002, ya/cen, yerr=ea/cen, fmt='o', ms=2.8, lw=0.8, mfc='white', mec=DATA, ecolor=DATA, mew=0.8, zorder=5)
    ar.errorbar(0.5*(ld + hd) + 0.002, yd/cd, yerr=ed/cd, fmt='o', ms=2.6, lw=0.8, color=DATA, zorder=6)
    ar.set_xlim(0, 0.5); ar.set_ylim(0.7, 1.3); ar.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
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
        ax.hexbin(xe, xl, gridsize=45, extent=(lo, hi, lo, hi), bins='log', cmap='Blues', mincnt=1, lw=0, zorder=2)
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


# the data figures of the paper, Figs. 3 to 12 in order: what runs without arguments
PAPER = ['closure_mix', 'toy', 'map', 'widebox', 'coverage', 'kscan', 'cost', 'bands', 'obs', 'aleph']


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
