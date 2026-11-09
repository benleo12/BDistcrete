#!/usr/bin/env python3
"""Figures of envelope_test.py: the envelope of the parameter variations against the linear covariance.

  env_<box>        thrust and charged multiplicity, prior only (generator parameters and learning): the draws
                   relative to the centre, the one-sigma band of the draws and of the linear covariance, and the
                   ratio of each linear SD to the SD of the draws, with its bootstrap uncertainty
  corr_<box>       correlation matrices of the thrust and multiplicity bins, prior only: the draws below the
                   diagonal, the linear covariance above
  env_after_<box>  the thrust distribution with the calculation imposed in the window (the application)
  ratios           the SD ratio of the linear covariance to the draws for every box, prior only

    python plot_envelope.py output/envelope_test_MIXGEOV3S9 full tune tune_xifull tune_sherpa tune_herwig
Writes <R8_OUT>/r8_env_<box>.pdf and so on, in the style of figs_r8.py. SUFFIX=_lin reads the files of the
linear mixture (<prefix>_<box>_lin.npz: boxes tune, full, tune_sherpa, tune_herwig, full_sherpa, full_herwig,
the tune-sized ranges with xi over [0, 1]), checks that each file (and the json next to it) holds mix = linear,
and appends the suffix to every figure name.

Ranges (rounds B10 and B21): each panel keeps its published range when what it draws fits, and otherwise widens only
as far as needed (figs_r8.widen): the panels of the ratio of the linear band to the spread of the draws and of the
fraction inside the band, and the linear band and (since B21, each side alone) the band of the draws of the upper
panels, are always drawn whole. Otherwise the draws of an upper panel keep the published range (it holds 95 percent of
the draws in every bin, up to a ratio of 1.8), so in tail bins some draws may leave it, as in the published figures.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figs_r8 as F

PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'output/envelope_test_MIXGEOV3S9'
SUFFIX = os.environ.get('SUFFIX', ''); LINEAR = SUFFIX.endswith('_lin')
BOXES = sys.argv[2:] or (['tune', 'full', 'tune_sherpa', 'tune_herwig', 'full_sherpa', 'full_herwig'] if LINEAR else
                         ['full', 'tune', 'tune_xifull', 'tune_sherpa', 'tune_herwig'])
NBOOT = 300
BOX_TITLE = dict(full='full box', tune=r'tune-sized box, $\xi\in[0,1]$' if LINEAR else 'tune-sized box',
                 tune_xifull=r'tune-sized box, $\xi\in[0,1]$',
                 tune_sherpa=r'tune-sized box, Sherpa ($\xi=0$)', tune_herwig=r'tune-sized box, Herwig ($\xi=1$)',
                 full_sherpa=r'full box, Sherpa ($\xi=0$)', full_herwig=r'full box, Herwig ($\xi=1$)')
# bins under 100 effective reference events at the centre (66, 7 and 223 raw events): masked in the case plots
THIN = {'thr_0.400_0.500', 'nch_0_2', 'nch_2_4'}
BOX_SHORT = dict(full='full box', tune='tune-sized ranges', tune_xifull=r'tune-sized, $\xi\in[0,1]$', tune_sherpa='Sherpa alone', tune_herwig='Herwig alone',
                 full_sherpa='Sherpa alone, full box', full_herwig='Herwig alone, full box')
TUNE_LABEL = r'tune-sized, $\xi\in[0,1]$' if LINEAR else r'tune-sized, $\xi\in[1/3,2/3]$'
MODELS = [('lin_flat', 'linear', '-'), ('lin_z', r'slopes in $z$', (0, (1, 1.2)))]


def load(box):
    f = f'{PREFIX}_{box}{SUFFIX}.npz'
    d = np.load(f, allow_pickle=True)
    if LINEAR:                                                      # a _lin file must hold the linear mixture
        assert 'mix' in d.files and str(d['mix']) == 'linear', f'{f} does not hold the linear mixture'
        fj = f[:-4] + '.json'
        if os.path.exists(fj):
            import json
            m = json.load(open(fj)).get('mix')
            assert m == 'linear', f'{fj} does not hold the linear mixture (mix = {m!r})'
    lab = [str(l) for l in d['labels']]; no = len(lab)
    idx = dict(thrust=[i for i, l in enumerate(lab) if l.startswith('thr_')], nch=[i for i, l in enumerate(lab) if l.startswith('nch_')],
               rho=[i for i, l in enumerate(lab) if l.startswith('rho_')])
    return d, lab, no, idx


def model_cov(d, name, side, no, sets=('theta', 'omega', 'c')):
    """The linear covariance of one model on one side (before: generator parameters and learning; after: also the
    calculation), from the stored blocks."""
    s = slice(0, no) if side == 'before' else slice(no, 2*no)
    if f'C_{name}' not in d.files:
        return None
    xq = name == 'xi_quad'
    C = d[f'C_{name}'][s, s].copy() if 'theta' in sets else 0
    if 'omega' in sets:
        C = C + d['C_learn_xq' if xq and 'C_learn_xq' in d.files else 'C_learn'][s, s]
    if side == 'after' and 'c' in sets:
        C = C + d['C_theory_xq' if xq and 'C_theory_xq' in d.files else 'C_theory'][s, s]
    return C


def boot_sd(Y, rng):
    b = rng.integers(0, len(Y), (NBOOT, len(Y)))
    return np.array([Y[i].std(0, ddof=1) for i in b]).std(0, ddof=1)


def edges_of(d, key):
    if key == 'thrust':
        return np.asarray(d['thrust_edges'], float), r'$\tau=1-T$'
    if key == 'nch':
        e = np.asarray(d['nch_edges'], float).copy(); e[-1] = e[-2] + 2
        return e, r'$n_{\rm ch}$'
    return np.asarray(d['rho_edges'], float), r'$\rho_H$'


def upper_lim(dev, sl, band=()):
    """The range of an upper panel (ratio to the centre): 1 -+ 1.15 times the largest 97.5th percentile of the draws'
    deviation, between 0.05 and 0.8, as published (the draws of a tail bin may leave the panel there); round B10: widened
    when the linear band (sl) would leave it, so the linear band is always drawn whole; round B21: each side widened alone
    when the band of the draws (band: its lower and upper edges) would leave it, with the same margin, so it is drawn
    whole too (over the full box it left the panel in the last thrust bin and at both ends of the multiplicity)."""
    lim = float(np.clip(1.15*np.nanmax(dev), 0.05, 0.8))
    need = float(np.nanmax(sl)) if np.size(sl) else 0.0
    lim = 1.15*need if np.isfinite(need) and need > lim else lim
    b = np.concatenate([np.ravel(np.asarray(v, float)) for v in band]) if len(band) else np.array([])
    b = b[np.isfinite(b)]
    lo = 1 - 1.15*(1 - b.min()) if len(b) and b.min() < 1 - lim else 1 - lim
    hi = 1 + 1.15*(b.max() - 1) if len(b) and b.max() > 1 + lim else 1 + lim
    return lo, hi


def set_upper(ax, dev, sl, band):
    """Set the range of an upper panel (upper_lim); when the band of the draws widened it (round B21), the panel keeps
    the ticks of its neighbours (figs_r8.half_ticks)."""
    lo, hi = upper_lim(dev, sl, band)
    ax.set_ylim(lo, hi)
    if (lo, hi) != upper_lim(dev, sl):
        F.half_ticks(ax)


def env(box, side='before', keys=('thrust', 'nch')):
    d, lab, no, idx = load(box); rng = np.random.default_rng(1)
    s = slice(0, no) if side == 'before' else slice(no, 2*no)
    Y = d['Y_joint'][:, s]; yc = d['y_central'][s]; se = Y.std(0, ddof=1); sb = boot_sd(Y, rng)
    fig = plt.figure(figsize=(F.TW, 3.6 if len(keys) > 1 else 3.4))
    gs = GridSpec(2, len(keys), figure=fig, height_ratios=[1.35, 1], hspace=0.08, wspace=0.22, left=0.09, right=0.99, top=0.9, bottom=0.12)
    for c, key in enumerate(keys):
        ii = idx[key]; e, xl = edges_of(d, key)
        ax = fig.add_subplot(gs[0, c]); bx = fig.add_subplot(gs[1, c], sharex=ax)
        for k in range(len(Y)):
            ax.stairs(Y[k, ii]/yc[ii], e, baseline=None, color=F.BASE, lw=0.35, alpha=0.12, zorder=1)
        mu = Y[:, ii].mean(0)/yc[ii]; sr = se[ii]/yc[ii]
        ax.stairs(mu + sr, e, baseline=mu - sr, fill=True, color=F.GEN, alpha=F.BAND_ALPHA, lw=0, zorder=2)
        ax.stairs(mu, e, baseline=None, color=F.GEN, lw=1.0, zorder=3)
        Cm = model_cov(d, 'lin_flat', side, no)
        if Cm is not None:
            sl = np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/yc[ii]
            for sg in (1, -1):
                ax.stairs(1 + sg*sl, e, baseline=None, color=F.METHOD, lw=1.1, zorder=4)
        F.unit_line(ax)
        dev = np.nanpercentile(np.abs(Y[:, ii]/yc[ii] - 1), 97.5, axis=0)       # the panel holds 95% of the draws in every bin
        set_upper(ax, dev, sl if Cm is not None else [0], (mu - sr, mu + sr))
        ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity', 'rho': 'heavy jet mass'}[key])
        if side == 'after':
            ax.axvspan(1/20, 1/3, color=F.THEORY, alpha=F.WINDOW_ALPHA, lw=0, zorder=0)
        # ratio panel
        bx.axhline(1, color=F.REF, lw=0.6)
        rse = sb[ii]/se[ii]; drawn = [1 + rse, 1 - rse]
        bx.stairs(1 + rse, e, baseline=1 - rse, fill=True, color=F.BASE, alpha=0.28, lw=0)
        for name, nm, ls in MODELS:
            Cm = model_cov(d, name, side, no)
            if Cm is None:
                continue
            r = np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii]; drawn.append(r)
            bx.stairs(r, e, baseline=None, color=F.METHOD, lw=1.0, ls=ls, label=nm)
        bx.set_ylim(*F.widen(0.3, 1.7, drawn)); bx.set_xlim(e[0], e[-1]); bx.set_xlabel(xl, labelpad=1.5)
        bx.set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$' if c == 0 else '')
        if side == 'after':
            bx.axvspan(1/20, 1/3, color=F.THEORY, alpha=F.WINDOW_ALPHA, lw=0, zorder=0)
        for a in (ax, bx):
            F.trim(a)
    h = [plt.Line2D([], [], color=F.BASE, lw=0.6), plt.Rectangle((0, 0), 1, 1, color=F.GEN, alpha=F.BAND_ALPHA),
         plt.Line2D([], [], color=F.METHOD, lw=1.1)] + [plt.Line2D([], [], color=F.METHOD, lw=1.0, ls=ls) for n, _, ls in MODELS[1:]
                                                         if f'C_{n}' in d.files]
    lb = [f'{len(Y)} variations', r'variations, $\pm1\sigma$', r'linear, $\pm1\sigma$'] + [nm for n, nm, _ in MODELS[1:] if f'C_{n}' in d.files]
    fig.legend(h, lb, loc='upper center', ncol=len(h), fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=1.8)
    fig.text(0.99, 0.905, BOX_TITLE.get(box, box) + ('' if side == 'before' else ', with the calculation'), ha='right', va='bottom',
             fontsize=7.5, color=F.REF)
    F.save(fig, f'env_{box}' if side == 'before' else f'env_after_{box}')


def corrfig(box, keys=('thrust', 'nch')):
    d, lab, no, idx = load(box)
    Y = d['Y_joint'][:, :no]; Ce = np.cov(Y, rowvar=False); Cm = model_cov(d, 'lin_flat', 'before', no)
    corr = lambda A: A/np.sqrt(np.outer(np.diag(A), np.diag(A)))
    fig, axs = plt.subplots(1, len(keys), figsize=(F.TW, 2.9), gridspec_kw=dict(wspace=0.25, left=0.07, right=0.9, bottom=0.14, top=0.88))
    for ax, key in zip(np.atleast_1d(axs), keys):
        ii = np.array([i for i in idx[key] if np.diag(Ce)[i] > 0]); A, B = corr(Ce[np.ix_(ii, ii)]), corr(Cm[np.ix_(ii, ii)])
        # origin='lower' draws row i at height i, so the cells with column > row (triu) lie below the diagonal
        M = np.where(np.triu(np.ones_like(A, bool), 1), A, np.where(np.tril(np.ones_like(A, bool), -1), B, 1.0))
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', interpolation='nearest')
        ax.plot([-0.5, len(ii) - 0.5], [-0.5, len(ii) - 0.5], color=F.REF, lw=0.5)
        e, xl = edges_of(d, key); centres = [e[list(idx[key]).index(i)] for i in ii]
        step = max(1, len(ii)//6); ticks = np.arange(0, len(ii), step)
        fmt = (lambda v: f'{v:.2f}') if key != 'nch' else (lambda v: f'{int(v)}')
        ax.set_xticks(ticks); ax.set_xticklabels([fmt(centres[t]) for t in ticks], fontsize=6.5)
        ax.set_yticks(ticks); ax.set_yticklabels([fmt(centres[t]) for t in ticks], fontsize=6.5)
        ax.set_xlabel(xl + ' (lower edge)', labelpad=1.5)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity', 'rho': 'heavy jet mass'}[key])
        bb = dict(fc='white', ec='none', alpha=0.85, pad=1.0)
        ax.text(0.97, 0.05, 'variations', transform=ax.transAxes, ha='right', fontsize=7, color=F.INK, bbox=bb)
        ax.text(0.03, 0.95, 'linear', transform=ax.transAxes, ha='left', va='top', fontsize=7, color=F.INK, bbox=bb)
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.62]); fig.colorbar(im, cax=cax).set_label('correlation', fontsize=7.5)
    fig.text(0.9, 0.93, BOX_TITLE.get(box, box), ha='right', fontsize=7.5, color=F.REF)
    F.save(fig, f'corr_{box}')


def ratios(boxes, key='thrust'):
    fig, axs = plt.subplots(1, 2, figsize=(F.TW, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.08, right=0.99, bottom=0.17, top=0.86))
    cols = F.ramp(len(boxes)); drawn = []
    for ax, key in zip(axs, ('thrust', 'nch')):
        ax.axhline(1, color=F.REF, lw=0.6)
        for col, box in zip(cols, boxes):
            if not os.path.exists(f'{PREFIX}_{box}{SUFFIX}.npz'):
                continue
            d, lab, no, idx = load(box); ii = idx[key]; e, xl = edges_of(d, key)
            se = d['Y_joint'][:, :no].std(0, ddof=1); Cm = model_cov(d, 'lin_flat', 'before', no)
            r = np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii]; drawn.append(r)
            ax.stairs(r, e, baseline=None, color=col, lw=1.1, label=BOX_TITLE.get(box, box))
        ax.set_xlim(e[0], e[-1]); ax.set_xlabel(xl, labelpad=1.5); F.trim(ax)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key])
    axs[0].set_ylim(*F.widen(0.3, 1.7, drawn))                      # shared by both panels
    axs[0].set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$'); axs[1].legend(fontsize=6.5, loc='upper right')
    F.save(fig, 'env_ratios')


def paper_panels(ax, bx, Y, yc, sl, e, sb, mask=None):
    """One distribution of the paper figures (Figs. 8 and 10, redrawn Nov 9 for legibility): in the upper panel the
    spread of the variations, their mean -+ one SD as an orange band, against the linear band, the prediction at the center
    -+ its linear SD as blue lines, both relative to that prediction; in the lower panel the ratio of the two SDs, with
    the bootstrap uncertainty of the variations' SD in light gray. The individual draws and the slopes in z are left out
    (the text quotes the latter); env() and env_cases still draw both."""
    se = Y.std(0, ddof=1); mu = Y.mean(0)/yc; sr = se/yc; band = sl/yc
    ax.stairs(mu + sr, e, baseline=mu - sr, fill=True, color=F.GEN, alpha=0.3, lw=0, zorder=2)
    for sg in (1, -1):
        ax.stairs(1 + sg*band, e, baseline=None, color=F.METHOD, lw=1.3, zorder=4)
    F.unit_line(ax)
    (lo, hi), ticks = F.nice_range([mu - sr, mu + sr, 1 - band, 1 + band], pad=0.08)
    ax.set_ylim(lo, hi); ax.set_yticks(ticks)
    rel = sb/se; r = sl/se
    if mask is not None:
        r = np.where(mask, np.nan, r)
    bx.stairs(1 + rel, e, baseline=1 - rel, fill=True, color=F.BASE, alpha=0.25, lw=0, zorder=1)
    bx.axhline(1, color=F.REF, lw=0.6, zorder=2)
    bx.stairs(r, e, baseline=None, color=F.METHOD, lw=1.3, zorder=3)
    bx.set_ylim(*F.widen(0.4, 1.6, [r, 1 + rel, 1 - rel])); bx.set_xlim(e[0], e[-1])


def paper_legend(fig, n, window=False, ncol=3):
    h = [plt.Rectangle((0, 0), 1, 1, color=F.GEN, alpha=0.3, lw=0), plt.Line2D([], [], color=F.METHOD, lw=1.3),
         plt.Rectangle((0, 0), 1, 1, color=F.BASE, alpha=0.25, lw=0)]
    lb = [f'{n} variations, mean $\\pm1\\sigma$', r'linear covariance, $\pm1\sigma$', r'bootstrap uncertainty of $\sigma_{\rm variations}$']
    if window:
        h.append(plt.Line2D([], [], color=F.THEORY, lw=0.9, ls=(0, (3, 2)))); lb.append('edges of the window')
    fig.legend(h, lb, loc='upper center', ncol=ncol, fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=1.8)


def env_pair(boxes=('tune', 'full'), keys=('thrust', 'nch'), save_as='envelope', short=BOX_SHORT, mask_thin=False):
    """The paper figure (Fig. 8): paper_panels() for several boxes stacked, prior only."""
    H = 2.75*len(boxes); fig = plt.figure(figsize=(F.TW, H))
    outer = GridSpec(len(boxes), 1, figure=fig, hspace=0.32, left=0.09, right=0.99, top=1 - 0.36/H, bottom=0.36/H)
    rng = np.random.default_rng(1)
    for r, box in enumerate(boxes):
        d, lab, no, idx = load(box)
        Y = d['Y_joint'][:, :no]; yc = d['y_central'][:no]; sb = boot_sd(Y, rng)
        sl = np.sqrt(np.maximum(np.diag(model_cov(d, 'lin_flat', 'before', no)), 0))
        gs = outer[r].subgridspec(2, len(keys), height_ratios=[1.25, 0.75], hspace=0.07, wspace=0.2)
        for c, key in enumerate(keys):
            ii = idx[key]; e, xl = edges_of(d, key)
            ax = fig.add_subplot(gs[0, c]); bx = fig.add_subplot(gs[1, c], sharex=ax)
            mask = np.array([lab[i] in THIN for i in ii]) if mask_thin else None
            paper_panels(ax, bx, Y[:, ii], yc[ii], sl[ii], e, sb[ii], mask)
            ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
            F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key] + ', ' + short.get(box, box))
            bx.set_xlabel(xl, labelpad=1.5); bx.set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$' if c == 0 else '')
            for a in (ax, bx):
                F.trim(a)
    paper_legend(fig, len(Y))
    F.save(fig, save_as)


def corr_pair(boxes=('tune', 'full'), keys=('thrust', 'nch'), save_as='envelope_corr', short=BOX_SHORT):
    """The paper figure: correlation matrices for two boxes, variations below the diagonal, linear above."""
    corr = lambda A: A/np.sqrt(np.outer(np.diag(A), np.diag(A)))
    fig, axs = plt.subplots(len(boxes), len(keys), figsize=(F.TW, 5.6),
                            gridspec_kw=dict(wspace=0.28, hspace=0.42, left=0.08, right=0.89, bottom=0.08, top=0.95))
    for r, box in enumerate(boxes):
        d, lab, no, idx = load(box)
        Ce = np.cov(d['Y_joint'][:, :no], rowvar=False); Cm = model_cov(d, 'lin_flat', 'before', no)
        for c, key in enumerate(keys):
            ax = axs[r, c]
            ii = np.array([i for i in idx[key] if np.diag(Ce)[i] > 0]); A, B = corr(Ce[np.ix_(ii, ii)]), corr(Cm[np.ix_(ii, ii)])
            # origin='lower' draws row i at height i, so the cells with column > row (triu) lie below the diagonal
            M = np.where(np.triu(np.ones_like(A, bool), 1), A, np.where(np.tril(np.ones_like(A, bool), -1), B, 1.0))
            im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', interpolation='nearest')
            ax.plot([-0.5, len(ii) - 0.5], [-0.5, len(ii) - 0.5], color=F.REF, lw=0.5)
            e, xl = edges_of(d, key); lows = [e[list(idx[key]).index(i)] for i in ii]
            step = max(1, len(ii)//6); ticks = np.arange(0, len(ii), step)
            fmt = (lambda v: f'{v:.2f}') if key != 'nch' else (lambda v: f'{int(v)}')
            ax.set_xticks(ticks); ax.set_xticklabels([fmt(lows[t]) for t in ticks], fontsize=6.5)
            ax.set_yticks(ticks); ax.set_yticklabels([fmt(lows[t]) for t in ticks], fontsize=6.5)
            ax.set_xlabel(xl + ' (lower bin edge)', labelpad=1.5)
            F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key] + ', ' + short.get(box, box))
            bb = dict(fc='white', ec='none', alpha=0.85, pad=1.0)
            ax.text(0.97, 0.04, 'variations', transform=ax.transAxes, ha='right', fontsize=7, color=F.INK, bbox=bb)
            ax.text(0.03, 0.96, 'linear', transform=ax.transAxes, ha='left', va='top', fontsize=7, color=F.INK, bbox=bb)
    cax = fig.add_axes([0.91, 0.25, 0.015, 0.5]); fig.colorbar(im, cax=cax).set_label('correlation', fontsize=7.5)
    F.save(fig, save_as)


def ratio_summary(boxes=('tune', 'tune_herwig', 'tune_sherpa', 'tune_xifull', 'full')):
    """The linear band over the spread of the variations, prior only, for every box on one panel per distribution,
    with the bootstrap uncertainty of the tune-sized box in gray."""
    style = dict(tune=(F.METHOD, '-', 1.5), tune_herwig=('#56B4E9', '-', 1.1), tune_sherpa=(F.THEORY, (0, (4, 1.5)), 1.1),
                 tune_xifull=(F.GEN, '-', 1.5), full=(F.BASE_DARK, (0, (1, 1.2)), 1.2),
                 full_herwig=('#56B4E9', (0, (1, 1.2)), 1.1), full_sherpa=(F.THEORY, (0, (1, 1.2)), 1.1))
    label = dict(BOX_SHORT, tune=TUNE_LABEL)
    fig, axs = plt.subplots(1, 2, figsize=(F.TW, 2.75), sharey=True, gridspec_kw=dict(wspace=0.07, left=0.08, right=0.99, bottom=0.15, top=0.8))
    rng = np.random.default_rng(1)
    d0, _, no0, idx0 = load('tune'); Y0 = d0['Y_joint'][:, :no0]; rel = boot_sd(Y0, rng)/Y0.std(0, ddof=1); drawn = []
    for ax, key in zip(axs, ('thrust', 'nch')):
        e0, _ = edges_of(d0, key); r0 = rel[idx0[key]]; drawn += [1 + r0, 1 - r0]
        ax.stairs(1 + r0, e0, baseline=1 - r0, fill=True, color=F.BASE, alpha=0.28, lw=0, label='bootstrap uncertainty')
        ax.axhline(1, color=F.REF, lw=0.6)
        for box in boxes:
            d, lab, no, idx = load(box); ii = idx[key]; e, xl = edges_of(d, key); col, ls, lw = style[box]
            se = d['Y_joint'][:, :no].std(0, ddof=1); Cm = model_cov(d, 'lin_flat', 'before', no)
            v = np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii]; v[[lab[i] in THIN for i in ii]] = np.nan; drawn.append(v)
            ax.stairs(v, e, baseline=None, color=col, ls=ls, lw=lw, label=label.get(box, box))
        ax.set_xlim(e[0], e[-1]); ax.set_xlabel(xl, labelpad=1.5); F.trim(ax)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key])
    axs[0].set_ylim(*F.widen(0.25, 1.25, drawn))                    # shared by both panels
    axs[0].set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=2.2)
    F.save(fig, 'env_cases_ratio')


def coverage_summary(boxes=('tune', 'tune_herwig', 'tune_sherpa', 'tune_xifull', 'full')):
    """How many of the variations lie inside the linear band around the prediction at the center, prior only, against
    the 68 percent of a Gaussian one sigma, for slopes in the parameters (top) and in z (bottom). In gray the binomial
    error of a fraction from the number of draws."""
    style = dict(tune=(F.METHOD, '-', 1.5), tune_herwig=('#56B4E9', '-', 1.1), tune_sherpa=(F.THEORY, (0, (4, 1.5)), 1.1),
                 tune_xifull=(F.GEN, '-', 1.5), full=(F.BASE_DARK, (0, (1, 1.2)), 1.2),
                 full_herwig=('#56B4E9', (0, (1, 1.2)), 1.1), full_sherpa=(F.THEORY, (0, (1, 1.2)), 1.1))
    label = dict(BOX_SHORT, tune=TUNE_LABEL)
    fig, axs = plt.subplots(2, 2, figsize=(F.TW, 4.4), sharey=True, sharex='col', gridspec_kw=dict(wspace=0.07, hspace=0.14, left=0.08, right=0.99, bottom=0.095, top=0.87))
    drawn = []
    for r, (model, tag) in enumerate((('lin_flat', 'slopes in the parameters'), ('lin_z', r'slopes in $z$'))):
        for c, key in enumerate(('thrust', 'nch')):
            ax = axs[r, c]
            for box in boxes:
                d, lab, no, idx = load(box); ii = idx[key]; e, xl = edges_of(d, key); col, ls, lw = style[box]
                Y = d['Y_joint'][:, :no]; yc = d['y_central'][:no]; sl = np.sqrt(np.maximum(np.diag(model_cov(d, model, 'before', no)), 0))
                v = 100*(np.abs(Y[:, ii] - yc[ii]) <= sl[ii]).mean(0); v[[lab[i] in THIN for i in ii]] = np.nan; drawn.append(v)
                ax.stairs(v, e, baseline=None, color=col, ls=ls, lw=lw, label=label.get(box, box))
            se = 100*np.sqrt(0.68*0.32/len(Y)); ax.axhspan(68 - se, 68 + se, color=F.BASE, alpha=0.28, lw=0, label=f'68 percent, sampling error of {len(Y)} draws')
            ax.axhline(68, color=F.REF, lw=0.6); ax.set_xlim(e[0], e[-1]); F.trim(ax)
            F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key] + ', ' + tag)
            if r == 1:
                ax.set_xlabel(xl, labelpad=1.5)
        axs[r, 0].set_ylabel(r'inside the band [\%]')
    axs[0, 0].set_ylim(*F.widen(35, 95, drawn))                     # shared by all four panels
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=2.2)
    F.save(fig, 'env_cases_coverage')


def env_after_pair(boxes=('tune', 'full')):
    """The application figure (Fig. 10): thrust with the calculation imposed in the window, drawn as Fig. 8."""
    fig = plt.figure(figsize=(F.TW, 2.75))
    gs = GridSpec(2, len(boxes), figure=fig, height_ratios=[1.25, 0.75], hspace=0.07, wspace=0.2, left=0.09, right=0.99, top=0.83, bottom=0.14)
    rng = np.random.default_rng(1)
    for c, box in enumerate(boxes):
        d, lab, no, idx = load(box); s = slice(no, 2*no)
        Y = d['Y_joint'][:, s]; yc = d['y_central'][s]; sb = boot_sd(Y, rng)
        sl = np.sqrt(np.maximum(np.diag(model_cov(d, 'lin_flat', 'after', no)), 0))
        ii = idx['thrust']; e, xl = edges_of(d, 'thrust')
        ax = fig.add_subplot(gs[0, c]); bx = fig.add_subplot(gs[1, c], sharex=ax)
        for a in (ax, bx):                                            # the window as two edges, not a fill under the band
            for x in (1/20, 1/3):
                a.axvline(x, color=F.THEORY, lw=0.9, ls=(0, (3, 2)), zorder=5)
        paper_panels(ax, bx, Y[:, ii], yc[ii], sl[ii], e, sb[ii])
        ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
        F.title(ax, 'thrust with the calculation, ' + {'tune': 'tune-sized ranges', 'full': 'full box'}.get(box, box))
        bx.set_xlabel(xl, labelpad=1.5); bx.set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$' if c == 0 else '')
        for a in (ax, bx):
            F.trim(a)
    paper_legend(fig, len(Y), window=True, ncol=2)
    F.save(fig, 'envelope_after')


if __name__ == '__main__':
    F.style()
    if os.environ.get('PNG'):                                       # also write a png of every figure, for a quick look
        _save = F.save
        def _both(fig, name):
            fig.savefig(os.path.join(F.OUT, f'r8_{name}.png'), dpi=200); _save(fig, name)
        F.save = _both
    if SUFFIX:                                                      # new names, so the earlier figures stay
        _plain = F.save
        F.save = lambda fig, name: _plain(fig, name + SUFFIX)
    have = [b for b in BOXES if os.path.exists(f'{PREFIX}_{b}{SUFFIX}.npz')]
    for b in have:
        env(b); corrfig(b); env(b, 'after', keys=('thrust',))
    if len(have) > 1:
        ratios(have)
    if 'tune' in have and 'full' in have:
        env_pair(); corr_pair(); env_after_pair()
    if LINEAR and {'tune', 'tune_herwig', 'tune_sherpa', 'full'} <= set(have):
        cases = [b for b in ('tune', 'tune_herwig', 'tune_sherpa', 'full_herwig', 'full_sherpa', 'full') if b in have]
        env_pair(('tune', 'tune_herwig', 'tune_sherpa', 'full'), save_as='env_cases', short=dict(BOX_SHORT, tune=TUNE_LABEL), mask_thin=True)
        corr_pair(('tune', 'full'), save_as='corr_cases', short=dict(tune=TUNE_LABEL, full='full box')); ratio_summary(cases); coverage_summary(cases)
    elif set(BOXES) >= {'tune', 'tune_herwig', 'tune_sherpa', 'tune_xifull', 'full'} and set(have) == set(BOXES):
        # the cases one at a time: where the linear covariance holds and what breaks it
        sh = dict(BOX_SHORT, tune=r'tune-sized, $\xi\in[1/3,2/3]$')
        env_pair(('tune', 'tune_herwig', 'tune_sherpa', 'tune_xifull'), save_as='env_cases', short=sh, mask_thin=True)
        corr_pair(('tune', 'tune_xifull'), save_as='corr_cases', short=dict(tune=r'$\xi\in[1/3,2/3]$', tune_xifull=r'$\xi\in[0,1]$')); ratio_summary(); coverage_summary()
