#!/usr/bin/env python3
"""The upgraded figure set. Distributions first, statistics second: the hero figure shows the
actual reweighted spectra landing on a fresh generator run, with pulls, at an off-centre
held-out point. All styling through money_style."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from money_style import setup, save, trim, ideal, panel_tag, panel_title, BLUE, RED, OCHRE, GREY, TEAL, INK, MUTED
from ab_analysis import Cond
from r2_ladder import make_bins, strange_frac
from xp_closure import per_event_bincontents, XP_BINS

MODELS = 'output/models'
HELD = 6901; THETA = (0.124, 0.55, 1.5)      # off-centre in two of three parameters


def _hist_w(vals, w, bins):
    h, _ = np.histogram(vals, bins, weights=w)
    h2, _ = np.histogram(vals, bins, weights=w**2)
    W = h.sum(); bw = np.diff(bins)
    dens = h/(W*bw)
    err = np.sqrt(h2)/(W*bw)
    return dens, err


def _hist_u(vals, bins):
    n, _ = np.histogram(vals, bins)
    N = n.sum(); bw = np.diff(bins)
    return n/(N*bw), np.sqrt(n)/(N*bw)


def _steps(ax, bins, y, **kw):
    ax.stairs(y, bins, **kw)


def hero():
    cond = Cond('C'); ref = np.load(f'{MODELS}/C_ref.npz')
    import pandas as pd
    d = np.load(f'data_stageC/particles_full_{HELD}.npz')
    sh = pd.read_csv(f'data_stageC/shapes_run_{HELD:04d}.csv')
    f = cond.f_at(np.array(THETA)); w = np.exp((f-f.max())/cond.T); w /= w.sum()
    uni = np.full(cond.Nref, 1.0/cond.Nref)

    PANELS = [
        ('1_minus_thrust', r'$1-T$', r'$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}(1-T)$', True),
        ('mult_total', r'multiplicity $N$', r'$P(N)$', False),
        ('nbaryon', r'baryon count $N_{\mathrm{baryon}}$', r'$P(N_{\mathrm{baryon}})$', True),
        ('x_p', r'$E_i/E_{\mathrm{vis}}$', r'$(1/n)\,\mathrm{d}n/\mathrm{d}(E_i/E_{\mathrm{vis}})$', True),
    ]
    fig = plt.figure(figsize=(11.2, 4.6))
    gs = GridSpec(2, 4, height_ratios=[2.6, 1.0], hspace=0.06, wspace=0.30,
                  left=0.06, right=0.99, top=0.91, bottom=0.12)
    TITLES = ['(a) thrust', '(b) multiplicity',
              '(c) baryon count', '(d) energy fraction']
    for i, (obs, xlab, ylab, logy) in enumerate(PANELS):
        axd = fig.add_subplot(gs[0, i]); axp = fig.add_subplot(gs[1, i], sharex=axd)
        panel_title(axd, TITLES[i], fs=14.5)
        if obs == 'x_p':
            # per-particle spectrum with event-level effective counts for the pulls
            rC = per_event_bincontents(ref['particles'][:, :, 0], ref['mask'], XP_BINS)
            tC = per_event_bincontents(d['particles'][:, :, 0], d['mask'], XP_BINS)
            bins = XP_BINS; bw = np.diff(bins)
            def dens_eff(C, wts):
                v = C*wts[:, None]; cont = v.sum(0)
                v2 = (C**2)*(wts[:, None]**2); s2 = v2.sum(0)
                W = cont.sum()
                neff = np.where(s2 > 0, cont**2/np.where(s2 > 0, s2, 1), 0)
                dens = cont/(W*bw)
                err = np.where(neff > 0, dens/np.sqrt(np.maximum(neff, 1)), 0)
                return dens, err
            dw, ew = dens_eff(rC, w)
            du, _ = dens_eff(rC, uni)
            wt_t = np.full(len(tC), 1.0/len(tC))
            dt, et = dens_eff(tC, wt_t)
        else:
            if obs == 'nbaryon':
                rv = ref['nbaryon'].astype(float); tv = d['nbaryon'].astype(float)
            else:
                rv = ref[obs].astype(float); tv = sh[obs].values
            bins = make_bins(np.r_[rv, tv], obs)
            dw, ew = _hist_w(rv, w, bins)
            du, _ = _hist_w(rv, uni, bins)
            dt, et = _hist_u(tv, bins)
        # upper: distributions
        _steps(axd, bins, du, color=GREY, lw=1.3, ls='--', label='unweighted')
        _steps(axd, bins, dw, color=BLUE, lw=1.7, label='reweighted')
        axd.fill_between(np.repeat(bins, 2)[1:-1], np.repeat(dw-ew, 2), np.repeat(dw+ew, 2),
                         color=BLUE, alpha=0.22, lw=0, step=None)
        ctr = 0.5*(bins[1:]+bins[:-1])
        axd.errorbar(ctr, dt, yerr=et, fmt='o', color=INK, ms=2.8, lw=0.9, capsize=0,
                     label='fresh run', zorder=5)
        if logy:
            axd.set_yscale('log')
            lo = max(dt[dt > 0].min()*0.3, 1e-6)
            axd.set_ylim(lo, None)
        if obs == 'mult_total':
            axd.set_xlim(0, 92); axd.set_ylim(0, 1.3*np.nanmax(dt))
        axd.set_ylabel(ylab, fontsize=16)
        plt.setp(axd.get_xticklabels(), visible=False)
        trim(axd)
        if i == 0:
            leg = axd.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=12,
                             frameon=True, facecolor='white', framealpha=0.95,
                             edgecolor='none')
            leg.set_zorder(7)
        # per-panel closure width from the validation map, ties the picture to the ruler
        # per-point widths from the validation map, never hardcoded
        _vm = json.load(open('output/valid_map.json'))[str(HELD)]
        WIDTHS = {o: _vm[o] for o in ('1_minus_thrust', 'mult_total', 'nbaryon', 'x_p')}
        axd.text(0.96, 0.86, rf'width $={WIDTHS[obs]:.2f}$', transform=axd.transAxes,
                 ha='right', fontsize=13, color=MUTED, zorder=6,
                 bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
        # lower: ratio to the fresh run (the physicist's view; grey departs, blue hugs one)
        ok = dt > 0
        rw = np.where(ok, dw/np.where(ok, dt, 1), np.nan)
        rwe = np.where(ok, ew/np.where(ok, dt, 1), np.nan)
        ru = np.where(ok, du/np.where(ok, dt, 1), np.nan)
        rte = np.where(ok, et/np.where(ok, dt, 1), np.nan)
        axp.axhline(1, color=MUTED, lw=0.7)
        axp.fill_between(ctr[ok], 1-rte[ok], 1+rte[ok], color=GREY, alpha=0.3, lw=0,
                         step='mid')
        axp.plot(ctr[ok], ru[ok], color=GREY, lw=1.2, ls='--', drawstyle='steps-mid')
        axp.plot(ctr[ok], rw[ok], color=BLUE, lw=1.6, drawstyle='steps-mid')
        axp.fill_between(ctr[ok], rw[ok]-rwe[ok], rw[ok]+rwe[ok], color=BLUE, alpha=0.25,
                         lw=0, step='mid')
        span = np.nanmax(np.abs(np.r_[ru[ok], rw[ok]]-1))
        lim = 0.14 if span < 0.12 else min(1.15*span, 0.6)
        axp.set_ylim(1-lim, 1+lim)
        if obs == 'mult_total':
            axp.set_xlim(0, 92)
        axp.set_xlabel(xlab, fontsize=17.5)
        if i == 0: axp.set_ylabel('ratio to\nfresh run', fontsize=13.5)
        trim(axp)
    save(fig, 'output/fig_p_closure.pdf')


def head():
    fig = plt.figure(figsize=(5.2, 3.7))
    gs = GridSpec(1, 1, left=0.15, right=0.98, top=0.95, bottom=0.16)
    # (a) cliffs: generator stages only (the toy scan lives in Sec 5.1 / fig_p_toy)
    ax = fig.add_subplot(gs[0, 0])
    rs = json.load(open('output/rank_scan.json'))
    # Stage E (d=8) is included only when its scan is present AND was run at the same seed
    # count as A, B and C. A curve from a different ensemble size does not belong on these
    # axes: the seed count moves both the width and its spread.
    SERIES = [('A', BLUE, 'o', 1), ('B', OCHRE, 's', 2), ('C', RED, '^', 3)]
    ens_abc = {rs[st].get('rank_ens', rs[st].get('recipe', {}).get('ens'))
               for st in ('A', 'B', 'C') if st in rs}
    if 'E' in rs:
        ens_e = rs['E'].get('rank_ens', rs['E'].get('recipe', {}).get('ens'))
        if len(ens_abc) == 1 and ens_e == next(iter(ens_abc)):
            SERIES.append(('E', TEAL, 'D', 8))
        else:
            print(f'fig_p_head: LEAVING Stage E OFF panel (a), it ran with {ens_e} seeds '
                  f'against {ens_abc} for A, B and C')
    for st, col, mk, dd in SERIES:
        sc = rs[st]['scan']
        KK = sorted(int(k) for k in sc); yy = [sc[str(k)]['width'] for k in KK]
        ee = [sc[str(k)].get('spread', 0) for k in KK]
        ax.errorbar(KK, yy, yerr=ee, marker=mk, ms=5.0, color=col, lw=1.6, capsize=2,
                    label=(rf'Stage {st}, $d={dd}$' if st != 'E' else rf'Sherpa, $d={dd}$'))
        K2 = 1 + dd + dd*(dd+1)//2
        ax.axvline(K2, color=col, ls=(0, (5, 3)), lw=0.9, alpha=0.5, zorder=1,
                   ymax=0.55)
    ideal(ax, 1.0)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    _yt = [1, 1.5, 2, 3, 4]
    ax.yaxis.set_major_locator(FixedLocator(_yt)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in _yt]))
    _tk = [1, 2, 3, 4, 8, 16] + ([32, 48] if any(t[0] == 'E' for t in SERIES) else [])
    ax.set_xticks(_tk)
    ax.set_xticklabels([str(t) for t in _tk])
    ax.set_xlabel(r'number of terms $K$')
    ax.set_ylabel(r'closure width $\sqrt{\chi^2/\mathrm{ndf}}$')
    ax.legend(fontsize=12.5, loc='upper right', frameon=True, facecolor='white',
              framealpha=0.9, edgecolor='none')
    trim(ax)
    save(fig, 'output/fig_p_head.pdf')


def toy():
    fvf = json.load(open('output/toy_fvf_data.json'))
    fig = plt.figure(figsize=(11.6, 3.6))
    gs = GridSpec(1, 3, wspace=0.32, left=0.055, right=0.985, top=0.88, bottom=0.165)
    # (a) recovery of the known answer
    key = '0.12' if '0.12' in fvf else list(fvf)[0]; t = fvf[key]
    xe, xl = np.array(t['f_exact']), np.array(t['f_learned'])
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = np.percentile(np.r_[xe, xl], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=MUTED, lw=1.0, zorder=3)
    ax.hexbin(xe, xl, gridsize=48, cmap='Blues', mincnt=1, extent=(lo, hi, lo, hi),
              linewidths=0)
    ax.set_xlabel(r'exact $f_{\mathrm{exact}}(\Phi)$')
    ax.set_ylabel(r'learned $f(\Phi)$')
    ax.text(0.95, 0.07, rf"slope $={t['slope']:.2f}$, $r={t['corr']:.3f}$",
            transform=ax.transAxes, ha='right', fontsize=13.5)
    panel_title(ax, '(a) log density ratio'); trim(ax)
    # (b) event side: differencing cancels b(theta)
    e1, l1 = np.array(fvf['0.12']['f_exact']), np.array(fvf['0.12']['f_learned'])
    e2, l2 = np.array(fvf['0.2']['f_exact']), np.array(fvf['0.2']['f_learned'])
    dx, dy = e2 - e1, l2 - l1
    ax = fig.add_subplot(gs[0, 1])
    lo, hi = np.percentile(dx, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=MUTED, lw=1.0, zorder=3)
    ax.hexbin(dx, dy, gridsize=46, cmap='Blues', mincnt=1, extent=(lo, hi, lo, hi),
              linewidths=0)
    cc = np.polyfit(dx, dy, 1); r = np.corrcoef(dx, dy)[0, 1]
    ax.set_xlabel(r'exact $\Delta f_{\mathrm{exact}} \propto \sum_i x_i^2$')
    ax.set_ylabel(r'learned $\Delta f$')
    ax.text(0.95, 0.07, rf'slope $={cc[0]:.2f}$, $r={r:.3f}$', transform=ax.transAxes,
            ha='right', fontsize=13.5)
    panel_title(ax, '(b) function of the event'); trim(ax)
    # (c) parameter side: recovered coefficients against the analytic curves
    ab = json.load(open('output/toy_ab_data.json'))
    ss = np.array(ab['ss'])
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ss, ab['an_c0'], color=MUTED, lw=1.5, zorder=2)
    ax.plot(ss, ab['an_c1'], color=MUTED, lw=1.5, zorder=2, label='exact')
    ax.plot(ss[::3], np.array(ab['c0'])[::3], 'o', color=BLUE, ms=4.4, zorder=3,
            label=r'learned, constant')
    ax.plot(ss[::3], np.array(ab['c1'])[::3], 's', color=OCHRE, ms=4.4, zorder=3,
            label=r'learned, $\sum_i x_i^2$')
    ax.set_xlabel(r'toy parameter $\log\sigma$')
    ax.set_ylabel(r'coefficient in $f$')
    ax.set_ylim(-2.9, 3.6)
    ax.legend(fontsize=12.5, loc='lower left')
    panel_title(ax, '(c) function of the parameter'); trim(ax)
    save(fig, 'output/fig_p_toy.pdf')

def anymap():
    d = json.load(open('output/valid_map.json'))
    pts = {int(k): v for k, v in d.items() if k != 'summary' and v.get('kind') == 'valid'}
    # held-out ladder points; run 6902 duplicates the training run 6813 event for event (no random
    # seed was set for the ladder runs), so it is not a held-out test and is left out
    anc = {int(k): v for k, v in d.items() if k != 'summary' and v.get('kind') == 'anchor' and int(k) != 6902}
    OBS = [('1_minus_thrust', r'thrust'), ('B_total', r'$B_{\rm tot}$'),
           ('rho_heavy', r'$\rho_H$'), ('mult_total', r'mult.'), ('x_p', r'$E_i/E_{\rm vis}$'),
           ('strange', r'strange'), ('joint_thrust_mult', r'joint'), ('nbaryon', r'baryons')]
    fig = plt.figure(figsize=(10.4, 4.0))
    gs = GridSpec(1, 2, width_ratios=[1.55, 1], wspace=0.24, left=0.06, right=0.985,
                  top=0.88, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(3)
    for i, (o, lab) in enumerate(OBS):
        va = np.array([p[o] for p in pts.values()])
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(va)), va, s=15, color=BLUE, alpha=0.75,
                   edgecolors='none', zorder=3, label='12 fresh runs' if i == 0 else None)
        vb = np.array([p[o] for p in anc.values()])
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(vb)), vb, s=22, marker='D', color=OCHRE,
                   alpha=0.9, edgecolors='none', zorder=4,
                   label='held-out ladder points' if i == 0 else None)
        ax.plot([i-0.25, i+0.25], [np.nanmean(va)]*2, color=INK, lw=1.8, zorder=5)
    ideal(ax, 1.0)
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([l for _, l in OBS], fontsize=14, rotation=25, ha='right', rotation_mode='anchor')
    ax.set_ylabel(r'closure width $\sqrt{\chi^2/\mathrm{ndf}}$')
    ax.legend(loc='upper left', fontsize=13.5)
    panel_title(ax, '(a) closure width by observable'); trim(ax)
    ax = fig.add_subplot(gs[0, 1])
    norm = [(0.120, 0.008), (0.475, 0.175), (1.30, 0.50)]
    for coll, col, mk, lab in [(pts, BLUE, 'o', '12 fresh runs'), (anc, OCHRE, 'D', 'held-out ladder points')]:
        dd = [max(abs((p['theta'][i]-norm[i][0])/norm[i][1]) for i in range(3)) for p in coll.values()]
        zz = [abs(p.get('nbaryon_z', np.nan)) for p in coll.values()]
        ax.scatter(dd, zz, s=24, marker=mk, color=col, edgecolors='none', zorder=3, label=lab)
    ax.axhline(2, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel(r'distance to box centre (half-widths)')
    ax.set_ylabel(r'$|z|$ of the mean baryon count')
    ax.set_ylim(0, None); ax.legend(loc='upper left', fontsize=13.5)
    panel_title(ax, '(b) mean baryon count'); trim(ax)
    save(fig, 'output/fig_p_map.pdf')


def widebox():
    w = json.load(open('output/widebox_ctrl.json'))
    fig = plt.figure(figsize=(9.8, 3.9))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.26, left=0.07, right=0.985,
                  top=0.88, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    for key, col, mk, lab in [('single_centre', RED, 'o', 'single central run'),
                              ('pool_all', OCHRE, 's', 'union of all 21 runs'),
                              ('local_union', BLUE, 'D', 'union of the 7 nearest runs')]:
        d = w[key]; a = sorted(float(x) for x in d)
        y = [d[f'{x:g}'] if f'{x:g}' in d else d[str(x)] for x in a]
        ax.plot(a, y, marker=mk, color=col, ms=5.5, lw=1.7, label=lab)
    ideal(ax, 1.0, label='statistical expectation')
    ax.axvline(0.14, color=MUTED, lw=0.8, ls=(0, (1, 2)), ymax=0.56)
    ax.annotate('reference\ncentre', (0.145, 3.1), ha='left', fontsize=13, color=MUTED)
    ax.set_xlabel(r'held-out $\alpha_s(M_Z)$')
    ax.set_ylabel(r'closure width $\sqrt{\chi^2/\mathrm{ndf}}$')
    ax.legend(loc='upper right', fontsize=13.5, frameon=True, facecolor='white',
              framealpha=0.95, edgecolor='none')
    panel_title(ax, '(a) closure width'); trim(ax)
    # (b) the mechanism: weight distributions at the far-edge target alpha_s = 0.083. The
    # union mixture bounds the weight by the number of pooled runs; the single reference
    # must span the full density ratio and its weights stretch over decades.
    ax = fig.add_subplot(gs[0, 1])
    ws = np.load('output/wb_weights_single_0.083.npy')
    wu = np.load('output/wb_weights_union_0.083.npy')
    for wv, col, lab in [(ws, RED, 'single central run'), (wu, BLUE, '7 nearest runs')]:
        lw10 = np.log10(np.clip(wv*len(wv), 1e-8, None))
        neff = 1.0/np.sum(wv**2)
        bins = np.linspace(-6, 3, 61)
        h, _ = np.histogram(lw10, bins)
        ax.stairs(h/h.sum(), bins, color=col, lw=1.7,
                  label=(rf'{lab}, $N_{{\rm eff}}={neff:.0f}$' if neff < 1e3 else
                         rf'{lab}, $N_{{\rm eff}}={neff/1e3:.1f}$k'))
    ax.axvline(np.log10(7), color=BLUE, lw=0.9, ls=(0, (2, 2)),
               label=r'mixture bound $w\le M$')
    ax.set_yscale('log')
    ax.set_xlim(-2.9, 3.2)
    ax.set_ylim(top=300)
    ax.set_xlabel(r'$\log_{10}$ of the weight (unit mean)')
    ax.set_ylabel(r'fraction of reference events')
    ax.legend(loc='upper left', fontsize=13, frameon=True, facecolor='white',
              framealpha=0.9, edgecolor='none')
    panel_title(ax, r'(b) weights at $\alpha_s=0.083$'); trim(ax)
    save(fig, 'output/fig_p_widebox.pdf')


if __name__ == '__main__':
    import sys
    setup(usetex=True)
    which = sys.argv[1:] or ['hero', 'head', 'toy', 'anymap', 'widebox']
    for name in which:
        try:
            {'hero': hero, 'head': head, 'toy': toy, 'anymap': anymap, 'widebox': widebox}[name]()
            print(f'{name} OK')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'{name} FAILED: {e}')
