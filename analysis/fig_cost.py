#!/usr/bin/env python3
"""The cost of the three networks at one parameter point (Sec. 3.6, Fig. 7), from bench_cost_final.py on the
1.15M reference events of the seventeen-parameter mixture, with the memory each design keeps between points.

(a) Time of the network part at one point: the weights of every reference event and the derivatives of one
    prediction (one backward pass), on one CPU process with four threads and on one GPU. The dashed line is the rest
    of a step, the maximum-entropy weight (linear mixture: of both generators) and the chi^2, the same for every
    network.
(b) Memory kept between points: the stored a(Phi), the pooled features, or the particles.

SUFFIX=_lin (the default) reads the benchmark of the linear mixture, bench_final_cpu_4_lin.json and
bench_final_cuda_12_lin.json, in which a point needs the weights of both generators; SUFFIX= (empty) reads the earlier
draft's geometric mixture, bench_final_cpu_4.json and bench_final_cuda_12.json, for comparison. Each file must hold
the mixture, device and threads its name says. figs_r8.py cost draws the paper's figure from the same files (COST_SUFFIX).

    python fig_cost.py            # BENCH_DIR (default output) -> FIG_OUT (default output)/fig_cost<SUFFIX>.pdf
"""
import json, os
import numpy as np
from money_style import setup, save, panel_title, BLUE, GREY, INK, MUTED
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROWS = [('factorized', 'factorized'), ('late', r'$\theta$ after the sum'), ('concatenation', 'concatenation (DCTR)')]
COLOR = {'factorized': BLUE, 'late': GREY, 'concatenation': GREY}
MEMKEY = {'factorized': 'factorized', 'late': 'late', 'concatenation': 'concatenation_particles'}
FIT_THREADS = 4             # threads of the CPU process
SFX = os.environ.get('SUFFIX', '_lin')
BENCH_DIR, FIG_OUT = os.environ.get('BENCH_DIR', 'output'), os.environ.get('FIG_OUT', 'output')


def load():
    runs = {}
    for key, f in ((FIT_THREADS, f'bench_final_cpu_{FIT_THREADS}{SFX}.json'), ('cuda', f'bench_final_cuda_12{SFX}.json')):
        d = json.load(open(os.path.join(BENCH_DIR, f)))
        mix = d.get('mix', 'geometric')                            # files before the linear mixture hold no mix
        assert mix == ('linear' if SFX.endswith('_lin') else 'geometric'), f'{f} holds mix = {mix}'
        dev, thr = ('cpu', FIT_THREADS) if key == FIT_THREADS else ('cuda', 12)     # the device and threads of its name
        assert (d.get('device'), d.get('threads')) == (dev, thr), f'{f} records device {d.get("device")!r} with {d.get("threads")!r} threads'
        runs[key] = d
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
    nc, ng, shared = cpu['n_network_step_seconds'], gpu['n_network_step_seconds'], cpu['b_tilt_and_chi2_seconds']
    memv = {k: cpu['store_GB'][MEMKEY[k]] for k, _ in ROWS}
    rows = [(k, lab) for k, lab in ROWS if k in nc and k in ng]
    lin = SFX.endswith('_lin')

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.8, 3.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.06))
    y = np.arange(len(rows))[::-1].astype(float)

    ax.axvline(shared, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.text(shared*1.12, y.max() + 0.42, ('maximum-entropy weights of both generators\nand $\\chi^2$ on the CPU, the same for every network'
                                          if lin else 'reweighting and $\\chi^2$ on the CPU,\nthe same for every network'),
            fontsize=11.5, color=MUTED, va='top', ha='left', linespacing=1.1)
    for yi, (k, lab) in zip(y, rows):
        c = COLOR[k]
        ax.plot([ng[k], nc[k]], [yi, yi], color=c, lw=1.3, alpha=0.5, zorder=1, solid_capstyle='round')
        ax.plot(ng[k], yi, 'o', ms=8, mfc='white', mec=c, mew=1.7, zorder=3)
        ax.plot(nc[k], yi, 'o', ms=8, color=c, zorder=3)
        ax.text(ng[k], yi - 0.19, human(ng[k]), ha='center', va='top', fontsize=12, color=INK)
        ax.text(nc[k], yi - 0.19, human(nc[k]), ha='center', va='top', fontsize=12, color=INK)
    ax.set_xscale('log'); ax.set_xlim(2e-3, 8e3)
    ax.set_yticks(y); ax.set_yticklabels([lab for k, lab in rows], fontsize=14)
    ax.set_ylim(y.min() - 0.62, y.max() + 0.5)
    ax.set_xlabel('time [s]')
    ax.legend(handles=[Line2D([0], [0], marker='o', lw=0, ms=8, color=MUTED, label=f'CPU, {FIT_THREADS} threads'),
                       Line2D([0], [0], marker='o', lw=0, ms=8, mfc='white', mec=MUTED, mew=1.7,
                              label=gpu.get('gpu', 'GPU').replace('NVIDIA ', ''))],
              loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=12, handletextpad=0.2, borderaxespad=0.3)
    panel_title(ax, '(a) network time at one point' + (' (both generators)' if lin else ''))
    clean(ax)

    for yi, (k, lab) in zip(y, rows):
        bx.barh(yi, memv[k], height=0.36, color=COLOR[k], zorder=2)
        bx.text(memv[k] + 0.1, yi, f'{memv[k]:.1f}\\,GB', va='center', fontsize=12, color=INK)
    bx.set_xlim(0, 5.8); bx.set_xticks([0, 2, 4])
    bx.set_xlabel('memory [GB]')
    panel_title(bx, '(b) memory kept between points')
    clean(bx)

    os.makedirs(FIG_OUT, exist_ok=True)
    save(fig, os.path.join(FIG_OUT, f'fig_cost{SFX}.pdf'))
    json.dump({str(k): {s: v.get(s) for s in ('mix', 'n_network_step_seconds', 'a_weights_seconds', 'b_tilt_and_chi2_seconds', 'store_GB')}
               for k, v in runs.items()}, open(os.path.join(FIG_OUT, f'fig_cost_values{SFX}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
