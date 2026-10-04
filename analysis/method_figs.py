#!/usr/bin/env python3
"""Publication illustrations for the method sections.

fig_m22 (Sec 2.2): why the classification objective yields the density ratio.
  (a) the two samples as a labeled classification problem
  (b) the population minimizer is the local label fraction, with the gradient-balance
      mechanism stated on the panel
  (c) the payoff: exponentiating the logit reweights the reference onto the target

fig_m23 (Sec 2.3): why the factorized head has rank d+1.
  (a) per-event logit against the parameter: every event's curve in the exact toy family
  (b) the d+1 shared shapes that span all of them
  (c) the event-by-parameter logit matrix as a sum of d+1 outer products
All in the paper's figure style."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from money_style import setup, save, trim, panel_title, BLUE, RED, OCHRE, GREY, INK, MUTED

S0 = 1.0


def qref(x): return np.exp(-x**2/(2*S0**2))/np.sqrt(2*np.pi*S0**2)


def qtgt(x, s1=1.15): return np.exp(-x**2/(2*s1**2))/np.sqrt(2*np.pi*s1**2)


def fig22():
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.7))
    x = np.linspace(-3.7, 3.7, 500)

    ax = axs[0]
    ax.plot(x, qref(x), color=GREY, lw=1.9, label=r'$q_{\mathrm{ref}}$, label $0$')
    ax.plot(x, qtgt(x), color=BLUE, lw=1.9, label=r'$q(\theta)$, label $1$')
    xr = rng.normal(0, 1.0, 45); xt = rng.normal(0, 1.15, 45)
    ax.plot(xr, np.full_like(xr, -0.014), '|', color=GREY, ms=6, alpha=0.8)
    ax.plot(xt, np.full_like(xt, -0.034), '|', color=BLUE, ms=6, alpha=0.8)
    ax.set_ylim(-0.05, 0.56)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'density')
    ax.legend(fontsize=13.5, loc='upper right')
    panel_title(ax, r'(a) two labeled samples'); trim(ax)

    ax = axs[1]
    sstar = qtgt(x)/(qtgt(x)+qref(x))
    ax.plot(x, sstar, color=BLUE, lw=1.9)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_ylim(0.42, 0.93)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$s^{\star}(x)$')
    ax.text(0.5, 0.86, r'$q\,(1-s)=q_{\mathrm{ref}}\,s$', transform=ax.transAxes,
            ha='center', fontsize=15.5, color=INK)
    ax.text(0.5, 0.72,
            r'$\Rightarrow\; s^{\star}=\dfrac{q}{q+q_{\mathrm{ref}}}$',
            transform=ax.transAxes, ha='center', fontsize=16, color=INK)
    panel_title(ax, r'(b) minimizer of the cross entropy'); trim(ax)

    ax = axs[2]
    N = 400000
    xs = rng.normal(0, 1.0, N)
    bins = np.linspace(-3.7, 3.7, 38); bw = bins[1]-bins[0]
    hu, _ = np.histogram(xs, bins, density=True)
    wts = qtgt(xs)/qref(xs)
    hw, _ = np.histogram(xs, bins, weights=wts); hw /= wts.sum()*bw
    ctr = 0.5*(bins[1:]+bins[:-1])
    ax.stairs(hu, bins, color=GREY, lw=1.5, ls='--', label=r'$q_{\mathrm{ref}}$')
    ax.stairs(hw, bins, color=BLUE, lw=1.9, label=r'$q_{\mathrm{ref}}\times w$')
    ax.plot(ctr[::2], qtgt(ctr[::2]), 'o', color=INK, ms=2.6, label=r'$q(\theta)$')
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'density')
    ax.set_ylim(0, 0.63)
    ax.text(0.05, 0.82, r'$w=e^{f^{\star}}$',
            transform=ax.transAxes, fontsize=14, va='center')
    ax.text(0.078, 0.705, r'$=s^{\star}\!/(1-s^{\star})$',
            transform=ax.transAxes, fontsize=14, va='center')
    ax.legend(fontsize=13.5, loc='upper right')
    panel_title(ax, r'(c) reweighted reference'); trim(ax)

    fig.tight_layout(w_pad=1.6)
    save(fig, 'output/fig_m22.pdf')


def fig23():
    # exact toy family: f(x_i; theta) = 10 log(1/sig) + (1/2 - 1/(2 sig^2)) * Q_i,
    # sig = e^theta, so every event's curve lives in the span of TWO theta-shapes
    th = np.linspace(-0.25, 0.25, 120)
    sig = np.exp(th)
    g0 = 10*np.log(1.0/sig)                    # event-independent shape (normalization)
    g1 = 0.5*(1.0 - 1.0/sig**2)                # shape multiplying the sufficient statistic
    rng = np.random.default_rng(5)
    Q = np.sort(rng.chisquare(10, 7))          # sum of 10 squares per event

    fig = plt.figure(figsize=(11.0, 3.7))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.32, left=0.06, right=0.99,
                  top=0.87, bottom=0.15)

    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.Blues(np.linspace(0.45, 0.95, len(Q)))
    for qi, c in zip(Q, cmap):
        ax.plot(th, g0 + qi*g1, color=c, lw=1.5)
    ax.set_xlabel(r'parameter $\theta$')
    ax.set_ylabel(r'$f_{\mathrm{exact}}(\Phi_i,\theta)$')
    panel_title(ax, r'(a) one curve per event $\Phi_i$')
    trim(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(th, g0/np.max(np.abs(g0)), color=OCHRE, lw=1.9, label=r'shape 1: $c(\theta)$')
    ax.plot(th, g1/np.max(np.abs(g1)), color=BLUE, lw=1.9,
            label=r'shape 2: coeff.\ of $S(\Phi)$')
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.set_ylim(-1.1, 1.75)
    ax.set_xlabel(r'parameter $\theta$'); ax.set_ylabel(r'shared shapes (normalized)')
    ax.legend(fontsize=13, loc='upper left')
    panel_title(ax, r'(b) the $n_s{+}1=2$ spanning shapes')
    trim(ax)

    # (c) the logit matrix and its rank-(d+1) factorization
    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    panel_title(ax, r'(c) rank-$(n_s{+}1)$ factorization')
    the = np.linspace(-0.25, 0.25, 30)
    sige = np.exp(the)
    G0 = 10*np.log(1.0/sige); G1 = 0.5*(1.0-1.0/sige**2)
    Qe = np.sort(rng.chisquare(10, 24))
    F = G0[None, :] + Qe[:, None]*G1[None, :]
    vmax = np.max(np.abs(F))
    TOP, H = 0.10, 0.74
    RY, RH = 0.42, 0.12
    def tall(x0, w, M, label):
        axi = ax.inset_axes([x0, TOP, w, H])
        vm = np.max(np.abs(M))
        axi.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values(): sp.set_linewidth(0.7)
        axi.set_title(label, fontsize=14, pad=4)
    def row(x0, w, M, label):
        axi = ax.inset_axes([x0, RY, w, RH])
        vm = np.max(np.abs(M))
        axi.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values(): sp.set_linewidth(0.7)
        axi.set_title(label, fontsize=14, pad=3)
    tall(0.00, 0.26, F, r'$f(\Phi_i,\theta_j)$')
    ax.text(0.315, 0.47, r'$=$', fontsize=16, ha='center')
    tall(0.36, 0.055, np.ones((24, 1)), r'$\mathbf{1}_i$')
    ax.text(0.455, 0.47, r'$\otimes$', fontsize=14, ha='center')
    row(0.49, 0.17, G0[None, :], r'$c(\theta_j)$')
    ax.text(0.705, 0.47, r'$+$', fontsize=16, ha='center')
    tall(0.745, 0.055, Qe[:, None], r'$S(\Phi_i)$')
    ax.text(0.84, 0.47, r'$\otimes$', fontsize=14, ha='center')
    row(0.875, 0.125, G1[None, :], r'$b_2(\theta_j)$')
    save(fig, 'output/fig_m23.pdf')


if __name__ == '__main__':
    setup(usetex=True)
    fig22()
    fig23()
    print('METHOD FIGS DONE')
