#!/usr/bin/env python3
"""Cost of the three networks in the seventeen-parameter fit of Sec. 6 (Sec. 5.8), from the
bench_cost.py results on the 1.15M reference events of the mixture model.

(a) Time of the network part of one fit step, that is the weights of every reference event and
    their derivatives in the seventeen generator parameters, on one CPU process with the four
    threads the fits used and on one A100. The dashed line is the rest of the step, the
    maximum-entropy reweighting and the chi^2, which is the same for every network.
(b) Memory kept between steps: the stored a(Phi), the pooled features, or the particles.

    python fig_cost.py            # reads output/bench/bench_cost_*.json, writes output/fig_cost.pdf
"""
import json, glob, os, re
import numpy as np
from money_style import setup, save, panel_title, BLUE, GREY, INK, MUTED
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROWS = [('factorized', 'factorized'), ('late', r'$\theta$ after the sum'), ('early', 'concatenation (DCTR)')]
COLOR = {'factorized': BLUE, 'late': GREY, 'early': GREY}
FIT_THREADS = 4             # threads per process in the real-data profile fits


def load():
    runs = {}
    for f in sorted(glob.glob('output/bench/bench_cost_*.json')):
        if re.search(r'smoke|gctest|failedrun', f):
            continue
        d = json.load(open(f))
        key = 'cuda' if d['device'] == 'cuda' else int(d['threads'])
        r = runs.setdefault(key, {})
        for sec in ('n_network_step_seconds', 'b_fit_step_seconds', 'cache_GB'):
            for k, v in (d.get(sec) or {}).items():
                r.setdefault(sec, {}).setdefault(k, v)
        if d.get('b_tilt_and_chi2_seconds') is not None:
            r.setdefault('tilt', d['b_tilt_and_chi2_seconds'])
    return runs


def human(sec):
    if sec >= 60:
        return f'{sec/60:.0f}\\,min'
    if sec >= 1:
        return f'{sec:.2g}\\,s'
    if sec >= 0.1:
        return f'{sec:.2f}\\,s'
    return f'{1000*sec:.2g}\\,ms'


def clean(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0, left=False, right=False)
    ax.tick_params(axis='x', which='both', top=False)


def main():
    setup()
    runs = load()
    cpu, gpu = runs[FIT_THREADS], runs['cuda']
    nc, ng, shared = cpu['n_network_step_seconds'], gpu['n_network_step_seconds'], cpu['tilt']
    mem = cpu['cache_GB']
    memv = {'factorized': mem['factorized'], 'late': mem['late'], 'early': mem['early_particles']}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.8, 3.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.06))
    y = np.arange(len(ROWS))[::-1].astype(float)

    ax.axvline(shared, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.text(shared*1.12, y.max() + 0.42, 'reweighting and $\\chi^2$ on the CPU,\nthe same for every network',
            fontsize=11.5, color=MUTED, va='top', ha='left', linespacing=1.1)
    for yi, (k, lab) in zip(y, ROWS):
        c = COLOR[k]
        ax.plot([ng[k], nc[k]], [yi, yi], color=c, lw=1.3, alpha=0.5, zorder=1, solid_capstyle='round')
        ax.plot(ng[k], yi, 'o', ms=8, mfc='white', mec=c, mew=1.7, zorder=3)
        ax.plot(nc[k], yi, 'o', ms=8, color=c, zorder=3)
        ax.text(ng[k], yi - 0.19, human(ng[k]), ha='center', va='top', fontsize=12, color=INK)
        ax.text(nc[k], yi - 0.19, human(nc[k]), ha='center', va='top', fontsize=12, color=INK)
    ax.set_xscale('log'); ax.set_xlim(2e-3, 8e3)
    ax.set_yticks(y); ax.set_yticklabels([lab for k, lab in ROWS], fontsize=14)
    ax.set_ylim(y.min() - 0.62, y.max() + 0.5)
    ax.set_xlabel('time [s]')
    ax.legend(handles=[Line2D([0], [0], marker='o', lw=0, ms=8, color=MUTED, label=f'CPU, {FIT_THREADS} threads'),
                       Line2D([0], [0], marker='o', lw=0, ms=8, mfc='white', mec=MUTED, mew=1.7, label='A100')],
              loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=12, handletextpad=0.2, borderaxespad=0.3)
    panel_title(ax, '(a) network time per step of the fit')
    clean(ax)

    for yi, (k, lab) in zip(y, ROWS):
        bx.barh(yi, memv[k], height=0.36, color=COLOR[k], zorder=2)
        bx.text(memv[k] + 0.1, yi, f'{memv[k]:.1f}\\,GB', va='center', fontsize=12, color=INK)
    bx.set_xlim(0, 5.8); bx.set_xticks([0, 2, 4])
    bx.set_xlabel('memory [GB]')
    panel_title(bx, '(b) memory kept between steps')
    clean(bx)

    os.makedirs('output', exist_ok=True)
    save(fig, 'output/fig_cost.pdf')
    json.dump({str(k): v for k, v in runs.items()}, open('output/fig_cost_values.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
