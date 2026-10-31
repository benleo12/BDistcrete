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
Writes <R8_OUT>/r8_env_<box>.pdf and so on, in the style of figs_r8.py.
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
BOXES = sys.argv[2:] or ['full', 'tune', 'tune_xifull', 'tune_sherpa', 'tune_herwig']
NBOOT = 300
BOX_TITLE = dict(full='full box', tune='tune-sized box', tune_xifull=r'tune-sized box, $\xi\in[0,1]$',
                 tune_sherpa=r'tune-sized box, Sherpa ($\xi=0$)', tune_herwig=r'tune-sized box, Herwig ($\xi=1$)')
MODELS = [('lin_flat', 'linear', '-'), ('lin_z', r'slopes in $z$', (0, (1, 1.2)))]


def load(box):
    d = np.load(f'{PREFIX}_{box}.npz', allow_pickle=True)
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
        lim = float(np.clip(1.15*np.nanmax(dev), 0.05, 0.8)); ax.set_ylim(1 - lim, 1 + lim)
        ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity', 'rho': 'heavy jet mass'}[key])
        if side == 'after':
            ax.axvspan(1/20, 1/3, color=F.THEORY, alpha=F.WINDOW_ALPHA, lw=0, zorder=0)
        # ratio panel
        bx.axhline(1, color=F.REF, lw=0.6)
        rse = sb[ii]/se[ii]
        bx.stairs(1 + rse, e, baseline=1 - rse, fill=True, color=F.BASE, alpha=0.28, lw=0)
        for name, nm, ls in MODELS:
            Cm = model_cov(d, name, side, no)
            if Cm is None:
                continue
            r = np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii]
            bx.stairs(r, e, baseline=None, color=F.METHOD, lw=1.0, ls=ls, label=nm)
        bx.set_ylim(0.3, 1.7); bx.set_xlim(e[0], e[-1]); bx.set_xlabel(xl, labelpad=1.5)
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
    cols = F.ramp(len(boxes))
    for ax, key in zip(axs, ('thrust', 'nch')):
        ax.axhline(1, color=F.REF, lw=0.6)
        for col, box in zip(cols, boxes):
            if not os.path.exists(f'{PREFIX}_{box}.npz'):
                continue
            d, lab, no, idx = load(box); ii = idx[key]; e, xl = edges_of(d, key)
            se = d['Y_joint'][:, :no].std(0, ddof=1); Cm = model_cov(d, 'lin_flat', 'before', no)
            ax.stairs(np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii], e, baseline=None, color=col, lw=1.1, label=BOX_TITLE.get(box, box))
        ax.set_xlim(e[0], e[-1]); ax.set_ylim(0.3, 1.7); ax.set_xlabel(xl, labelpad=1.5); F.trim(ax)
        F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key])
    axs[0].set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$'); axs[1].legend(fontsize=6.5, loc='upper right')
    F.save(fig, 'env_ratios')


def env_pair(boxes=('tune', 'full'), keys=('thrust', 'nch')):
    """The paper figure: env() for two boxes stacked, prior only."""
    fig = plt.figure(figsize=(F.TW, 6.3))
    outer = GridSpec(2, 1, figure=fig, hspace=0.3, left=0.09, right=0.99, top=0.93, bottom=0.07)
    rng = np.random.default_rng(1)
    for r, box in enumerate(boxes):
        d, lab, no, idx = load(box)
        Y = d['Y_joint'][:, :no]; yc = d['y_central'][:no]; se = Y.std(0, ddof=1); sb = boot_sd(Y, rng)
        gs = outer[r].subgridspec(2, len(keys), height_ratios=[1.3, 1], hspace=0.08, wspace=0.22)
        for c, key in enumerate(keys):
            ii = idx[key]; e, xl = edges_of(d, key)
            ax = fig.add_subplot(gs[0, c]); bx = fig.add_subplot(gs[1, c], sharex=ax)
            for k in range(len(Y)):
                ax.stairs(Y[k, ii]/yc[ii], e, baseline=None, color=F.BASE, lw=0.35, alpha=0.12, zorder=1)
            mu = Y[:, ii].mean(0)/yc[ii]; sr = se[ii]/yc[ii]
            ax.stairs(mu + sr, e, baseline=mu - sr, fill=True, color=F.GEN, alpha=F.BAND_ALPHA, lw=0, zorder=2)
            ax.stairs(mu, e, baseline=None, color=F.GEN, lw=1.0, zorder=3)
            sl = np.sqrt(np.maximum(np.diag(model_cov(d, 'lin_flat', 'before', no)), 0))[ii]/yc[ii]
            for sg in (1, -1):
                ax.stairs(1 + sg*sl, e, baseline=None, color=F.METHOD, lw=1.1, zorder=4)
            F.unit_line(ax)
            dev = np.nanpercentile(np.abs(Y[:, ii]/yc[ii] - 1), 97.5, axis=0)
            lim = float(np.clip(1.15*np.nanmax(dev), 0.05, 0.8)); ax.set_ylim(1 - lim, 1 + lim)
            ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
            F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key] + ', ' + {'tune': 'tune-sized ranges', 'full': 'full box'}.get(box, box))
            bx.axhline(1, color=F.REF, lw=0.6)
            rse = sb[ii]/se[ii]
            bx.stairs(1 + rse, e, baseline=1 - rse, fill=True, color=F.BASE, alpha=0.28, lw=0)
            for name, nm, ls in MODELS:
                Cm = model_cov(d, name, 'before', no)
                if Cm is not None:
                    bx.stairs(np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii], e, baseline=None, color=F.METHOD, lw=1.0, ls=ls)
            bx.set_ylim(0.3, 1.7); bx.set_xlim(e[0], e[-1]); bx.set_xlabel(xl, labelpad=1.5)
            bx.set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$' if c == 0 else '')
            for a in (ax, bx):
                F.trim(a)
    h = [plt.Line2D([], [], color=F.BASE, lw=0.6), plt.Rectangle((0, 0), 1, 1, color=F.GEN, alpha=F.BAND_ALPHA),
         plt.Line2D([], [], color=F.METHOD, lw=1.1)] + [plt.Line2D([], [], color=F.METHOD, lw=1.0, ls=ls) for _, _, ls in MODELS[1:]]
    lb = ['400 variations', r'variations, $\pm1\sigma$', r'linear, $\pm1\sigma$'] + [nm for _, nm, _ in MODELS[1:]]
    fig.legend(h, lb, loc='upper center', ncol=len(h), fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=1.8)
    F.save(fig, 'envelope')


def corr_pair(boxes=('tune', 'full'), keys=('thrust', 'nch')):
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
            F.title(ax, {'thrust': 'thrust', 'nch': 'charged multiplicity'}[key] + ', ' + {'tune': 'tune-sized ranges', 'full': 'full box'}.get(box, box))
            bb = dict(fc='white', ec='none', alpha=0.85, pad=1.0)
            ax.text(0.97, 0.04, 'variations', transform=ax.transAxes, ha='right', fontsize=7, color=F.INK, bbox=bb)
            ax.text(0.03, 0.96, 'linear', transform=ax.transAxes, ha='left', va='top', fontsize=7, color=F.INK, bbox=bb)
    cax = fig.add_axes([0.91, 0.25, 0.015, 0.5]); fig.colorbar(im, cax=cax).set_label('correlation', fontsize=7.5)
    F.save(fig, 'envelope_corr')


def env_after_pair(boxes=('tune', 'full')):
    """The application figure: thrust with the calculation imposed in the window, variations against linear."""
    fig = plt.figure(figsize=(F.TW, 3.3))
    gs = GridSpec(2, len(boxes), figure=fig, height_ratios=[1.3, 1], hspace=0.08, wspace=0.22, left=0.09, right=0.99, top=0.86, bottom=0.13)
    rng = np.random.default_rng(1)
    for c, box in enumerate(boxes):
        d, lab, no, idx = load(box); s = slice(no, 2*no)
        Y = d['Y_joint'][:, s]; yc = d['y_central'][s]; se = Y.std(0, ddof=1); sb = boot_sd(Y, rng)
        ii = idx['thrust']; e, xl = edges_of(d, 'thrust')
        ax = fig.add_subplot(gs[0, c]); bx = fig.add_subplot(gs[1, c], sharex=ax)
        for a in (ax, bx):
            a.axvspan(1/20, 1/3, color=F.THEORY, alpha=F.WINDOW_ALPHA, lw=0, zorder=0)
        for k in range(len(Y)):
            ax.stairs(Y[k, ii]/yc[ii], e, baseline=None, color=F.BASE, lw=0.35, alpha=0.12, zorder=1)
        mu = Y[:, ii].mean(0)/yc[ii]; sr = se[ii]/yc[ii]
        ax.stairs(mu + sr, e, baseline=mu - sr, fill=True, color=F.GEN, alpha=F.BAND_ALPHA, lw=0, zorder=2)
        ax.stairs(mu, e, baseline=None, color=F.GEN, lw=1.0, zorder=3)
        sl = np.sqrt(np.maximum(np.diag(model_cov(d, 'lin_flat', 'after', no)), 0))[ii]/yc[ii]
        for sg in (1, -1):
            ax.stairs(1 + sg*sl, e, baseline=None, color=F.METHOD, lw=1.1, zorder=4)
        F.unit_line(ax)
        dev = np.nanpercentile(np.abs(Y[:, ii]/yc[ii] - 1), 97.5, axis=0)
        lim = float(np.clip(1.15*np.nanmax(dev), 0.05, 0.8)); ax.set_ylim(1 - lim, 1 + lim)
        ax.set_ylabel('ratio to center' if c == 0 else ''); plt.setp(ax.get_xticklabels(), visible=False)
        F.title(ax, 'thrust with the calculation, ' + {'tune': 'tune-sized ranges', 'full': 'full box'}.get(box, box))
        bx.axhline(1, color=F.REF, lw=0.6); rse = sb[ii]/se[ii]
        bx.stairs(1 + rse, e, baseline=1 - rse, fill=True, color=F.BASE, alpha=0.28, lw=0)
        for name, nm, ls in MODELS:
            Cm = model_cov(d, name, 'after', no)
            if Cm is not None:
                bx.stairs(np.sqrt(np.maximum(np.diag(Cm), 0))[ii]/se[ii], e, baseline=None, color=F.METHOD, lw=1.0, ls=ls)
        bx.set_ylim(0.3, 1.7); bx.set_xlim(e[0], e[-1]); bx.set_xlabel(xl, labelpad=1.5)
        bx.set_ylabel(r'$\sigma_{\rm linear}/\sigma_{\rm variations}$' if c == 0 else '')
        for a in (ax, bx):
            F.trim(a)
    h = [plt.Line2D([], [], color=F.BASE, lw=0.6), plt.Rectangle((0, 0), 1, 1, color=F.GEN, alpha=F.BAND_ALPHA),
         plt.Line2D([], [], color=F.METHOD, lw=1.1)] + [plt.Line2D([], [], color=F.METHOD, lw=1.0, ls=ls) for _, _, ls in MODELS[1:]] \
        + [plt.Rectangle((0, 0), 1, 1, color=F.THEORY, alpha=0.2)]
    lb = ['400 variations', r'variations, $\pm1\sigma$', r'linear, $\pm1\sigma$'] + [nm for _, nm, _ in MODELS[1:]] + ['window']
    fig.legend(h, lb, loc='upper center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.54, 1.0), handlelength=1.8)
    F.save(fig, 'envelope_after')


if __name__ == '__main__':
    F.style()
    if os.environ.get('PNG'):                                       # also write a png of every figure, for a quick look
        _save = F.save
        def _both(fig, name):
            fig.savefig(os.path.join(F.OUT, f'r8_{name}.png'), dpi=200); _save(fig, name)
        F.save = _both
    have = [b for b in BOXES if os.path.exists(f'{PREFIX}_{b}.npz')]
    for b in have:
        env(b); corrfig(b); env(b, 'after', keys=('thrust',))
    if len(have) > 1:
        ratios(have)
    if 'tune' in have and 'full' in have:
        env_pair(); corr_pair(); env_after_pair()
