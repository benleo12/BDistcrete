#!/usr/bin/env python3
"""The factorization, opened up and compared to truth on both sides.

(a) toy, event side: differencing the learned logit between two target widths cancels
    b(theta) exactly, so Delta f isolates the learned a(Phi). Against the exact
    Delta f_exact, which is affine in the true sufficient statistic sum x^2.
(b) toy, parameter side: the recovered b(s) coefficients against the analytic curves.
(c) generator, parameter side (Stage A): the theta-modes of the trained f(Phi,theta),
    from an SVD over events x theta grid, against the constant / linear / quadratic
    shapes the score expansion predicts.
(d) generator, event side: the score direction extracted independently from two ensemble
    members, event by event. Agreement means the score is a property of the generator
    family, not of one network's noise.
Writes output/fig_p_factor.pdf."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from money_style import setup, save, trim, panel_title, BLUE, RED, OCHRE, GREY, INK, MUTED
from ab_analysis import Cond

NSUB = 6000


def main():
    setup(usetex=True)
    fig = plt.figure(figsize=(9.8, 3.9))
    gs = GridSpec(1, 2, wspace=0.27, left=0.075, right=0.985, top=0.87, bottom=0.15)

    # toy slope/r kept for the provenance dump (drawn in fig_p_toy)
    fvf = json.load(open('output/toy_fvf_data.json'))
    e1, l1 = np.array(fvf['0.12']['f_exact']), np.array(fvf['0.12']['f_learned'])
    e2, l2 = np.array(fvf['0.2']['f_exact']), np.array(fvf['0.2']['f_learned'])
    dx, dy = e2 - e1, l2 - l1

    # ---- (a) generator parameter side: theta-modes of Stage A ----
    cond = Cond('A')
    rng = np.random.default_rng(11)
    sub = rng.choice(cond.Nref, NSUB, replace=False)
    tgrid = np.linspace(0.108, 0.132, 41)
    thn = (tgrid - cond.norm[0, 0])/cond.norm[0, 1]
    F = np.zeros((NSUB, len(tgrid)))
    for mi in range(cond.ENS):
        B = np.stack([cond.bnet(mi, [t]) for t in thn])            # (grid, K)
        F += cond.AE[mi][sub] @ B.T
    F /= cond.ENS
    U, sv, Vt = np.linalg.svd(F, full_matrices=False)
    ax = fig.add_subplot(gs[0, 0])
    cols = [GREY, BLUE, OCHRE]
    x = (tgrid - 0.120)/0.010
    _r2s = []
    for j in range(3):
        mode = sv[j]*Vt[j]
        mode = mode*np.sign(mode[-1] - mode[0]) if j > 0 else mode*np.sign(mode.mean())
        mode = mode/np.max(np.abs(mode))
        cc = np.polyfit(x, mode, 2)
        fit = np.polyval(cc, x)
        r2 = 1 - np.sum((mode-fit)**2)/np.sum((mode-mode.mean())**2)
        _r2s.append(float(r2))
        ax.plot(tgrid, mode, color=cols[j], lw=1.8, label=rf'mode {j+1}')
        ax.plot(tgrid, fit, color=cols[j], lw=1.0, ls=(0, (4, 3)))
    ax.plot([], [], color=INK, lw=1.0, ls=(0, (4, 3)), label='degree-2 fit')
    ax.set_ylim(-1.15, 1.62)
    ax.set_xlabel(r'$\alpha_s(M_Z)$'); ax.set_ylabel(r'$\theta$-mode (normalized)')
    ax.legend(fontsize=13, loc='upper left', ncol=2, columnspacing=1.0)
    panel_title(ax, r'(a) parameter side: $\theta$-modes of $b$')
    trim(ax)

    # ---- (b) generator event side: score reproducibility across the ensemble ----
    ax = fig.add_subplot(gs[0, 1])
    scores = []
    for mi in range(cond.ENS):
        B = np.stack([cond.bnet(mi, [t]) for t in thn])
        Fm = cond.AE[mi][sub] @ B.T
        scores.append((Fm @ thn)/np.sum(thn**2))                     # regression on theta
    r1 = np.corrcoef(scores[0], scores[1])[0, 1]
    h0 = 0.5*(scores[0]+scores[1]); h1 = 0.5*(scores[2]+scores[3])
    rh = np.corrcoef(h0, h1)[0, 1]
    lo, hi = np.percentile(np.r_[h0, h1], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=MUTED, lw=1.0, zorder=3)
    ax.hexbin(h0, h1, gridsize=46, cmap='Blues', mincnt=1, extent=(lo, hi, lo, hi),
              linewidths=0)
    ax.set_xlabel(r'score $\hat S(\Phi)$, ensemble half $(1{,}2)$')
    ax.set_ylabel(r'score $\hat S(\Phi)$, half $(3{,}4)$')
    ax.text(0.05, 0.93, rf'halves $r={rh:.2f}$' '\n' rf'single members $r={r1:.2f}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=14)
    panel_title(ax, '(b) event side: score reproducibility')
    trim(ax)

    save(fig, 'output/fig_p_factor.pdf')
    import json as _json
    _sl = float(np.polyfit(dx, dy, 1)[0])
    _json.dump(dict(toy_slope=_sl, toy_r=float(np.corrcoef(dx, dy)[0, 1]),
                    mode_r2=_r2s, sv4_over_sv1=float(sv[3]/sv[0]),
                    halves_r=float(rh), singles_r=float(r1)),
               open('output/factor_fig_values.json', 'w'), indent=1)
    print(f'FACTOR FIG DONE  toy slope {_sl:.3f}  member-agreement r={rh:.3f}')


if __name__ == '__main__':
    main()
