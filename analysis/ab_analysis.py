#!/usr/bin/env python3
"""Method-sell analysis for the bilinear conditional head f(Phi,theta)=<a(Phi),b(theta)>.

Three things a machine-learning reader will demand, all read straight off the EXPORTED
network (output/models/<stage>_cond.npz), no retraining:

 (a) intrinsic rank. Sample the learned function on a cloud of theta across the training box
     and singular-value-decompose the event x theta matrix f(Phi_i,theta_j). The number of
     dominant singular values is the intrinsic dimension of the theta-dependence. It sits at
     d+1 for d parameters (d score directions plus the normalization mode), which is exactly
     why the closure cliff falls where it does. This is measured on the function itself, not
     through closure, so it is independent evidence for the same statement.

 (b) interpretable score. S_p(Phi)=d f/d theta_p is the per-event sensitivity the network
     assigns to parameter p. Profiled against the physical observable that parameter drives,
     it is monotonic, so the internal score is a physical response and not an opaque feature.

 (c) it costs nothing. The same closure as the standard concatenation head (DCTR), which has
     no rank structure and hands back no analytic theta-derivative. The bilinear head matches
     its accuracy and additionally yields b(theta), whose derivative is the response matrix J
     that the maximum-entropy uncertainty propagation needs.

Writes output/ab_analysis.json and output/fig_m_ab.pdf.
"""
import json, numpy as np
from money_style import setup, save, trim, ideal, panel_tag, BLUE, RED, OCHRE, GREY, INK, MUTED
import matplotlib.pyplot as plt

MODELS = 'output/models'
STAGES = ['A', 'B', 'C']
ACTS = {'relu': lambda x: np.maximum(x, 0.0),
        'silu': lambda x: x/(1.0+np.exp(-x)),
        'gelu': lambda x: 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))}


class Cond:
    def __init__(self, stage):
        c = np.load(f'{MODELS}/{stage}_cond.npz')
        self.c = c
        self.K = int(c['K']); self.nt = int(c['ntheta']); self.ENS = int(c['ens'])
        self.norm = c['norm']; self.act = ACTS[str(c['act'])]; self.T = float(c['temperature'])
        self.AE = c['AE']                      # (ENS, Nref, K)  a(Phi) cached
        self.Nref = self.AE.shape[1]

    def tn(self, th):
        return ((np.asarray(th, float) - self.norm[:, 0])/self.norm[:, 1])

    def bnet(self, mi, thn):
        x = np.atleast_2d(thn).astype(np.float64); li = 0
        while f'B{mi}_W{li}' in self.c:
            x = x @ self.c[f'B{mi}_W{li}'].T + self.c[f'B{mi}_b{li}']
            if f'B{mi}_W{li+1}' in self.c: x = self.act(x)
            li += 1
        return x[0]                            # (K,)

    def f_grid(self, thetas, sub):
        """f(Phi_i, theta_j) for events `sub` and a list of physical thetas -> (ngrid, nsub)."""
        F = np.empty((len(thetas), len(sub)))
        for j, th in enumerate(thetas):
            acc = np.zeros(len(sub))
            for mi in range(self.ENS):
                acc += self.AE[mi, sub] @ self.bnet(mi, self.tn(th))
            F[j] = acc/self.ENS
        return F

    def f_at(self, theta):
        acc = np.zeros(self.Nref)
        for mi in range(self.ENS):
            acc += self.AE[mi] @ self.bnet(mi, self.tn(theta))
        return acc/self.ENS

    def score(self, theta, p, rel=0.25):
        h = rel*self.norm[p, 1]
        tp = np.array(theta, float); tp[p] += h
        tm = np.array(theta, float); tm[p] -= h
        return (self.f_at(tp) - self.f_at(tm))/(2*h)


def box_cloud(cond, npts, seed=0, half=1.0):
    """A cloud of theta covering +-`half` standardized units in every parameter. The default
    stays inside every stage's training box (A +-1.33, B +-0.83, C +-1.0 sigma), so the
    spectrum measures the score expansion where it was actually trained, not an extrapolation
    whose curvature would inflate the tail."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-half, half, size=(npts, cond.nt))
    return [cond.norm[:, 0] + u[i]*cond.norm[:, 1] for i in range(npts)]


def svd_spectrum(cond, npts=48, nsub=5000, seed=0):
    rng = np.random.default_rng(seed+1)
    sub = rng.choice(cond.Nref, min(nsub, cond.Nref), replace=False)
    F = cond.f_grid(box_cloud(cond, npts, seed), sub)     # (npts, nsub)
    s = np.linalg.svd(F - F.mean(0, keepdims=True)*0, compute_uv=False)   # keep normalization mode
    return (s/s[0]).tolist()


def main():
    setup(usetex=True)
    out = {}
    # ---------- (a) intrinsic-rank spectra ----------
    spec = {}
    for st in STAGES:
        cond = Cond(st)
        spec[st] = svd_spectrum(cond)
        print(f'[{st}] d={cond.nt}  top singular values (normalized): '
              f'{np.round(spec[st][:6], 4).tolist()}')
    out['svd_spectrum'] = spec

    # ---------- (b) interpretable score: Stage A alpha_s score vs 1-thrust ----------
    cA = Cond('A'); refA = np.load(f'{MODELS}/A_ref.npz')
    thr = refA['1_minus_thrust'].astype(float)
    SA = cA.score(cA.norm[:, 0], 0)            # d f / d alpha_s at box centre
    # profile the mean score in bins of 1-thrust
    qs = np.quantile(thr, np.linspace(0.02, 0.98, 13))
    xb = 0.5*(qs[1:]+qs[:-1]); yb = []; ye = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (thr >= lo) & (thr < hi)
        yb.append(SA[m].mean()); ye.append(SA[m].std()/max(np.sqrt(m.sum()), 1))
    r = np.corrcoef(thr, SA)[0, 1]
    out['score_profile'] = dict(x=xb.tolist(), y=yb, ye=ye, corr=float(r))
    print(f'[A] corr(1-thrust, d f/d alpha_s) = {r:.3f}')

    # ---------- (c) bilinear vs concat closure ----------
    lad = json.load(open('output/ladder_final.json'))
    con = json.load(open('output/concat_baseline.json'))
    comp = {s: dict(bilinear=lad[s]['master'], concat=con[s]['master']) for s in ['A', 'B'] if s in con}
    out['head_compare'] = comp
    print('[compare]', comp)

    # ================= figure =================
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.15))
    # (a) cumulative variance of the theta->f map: saturates at d+1 for every stage
    cols = {'A': BLUE, 'B': OCHRE, 'C': RED}
    dd = {'A': 1, 'B': 2, 'C': 3}
    for st in STAGES:
        s = np.array(spec[st]); e = np.cumsum(s**2)/np.sum(s**2)
        k = np.arange(1, len(e)+1)
        ax[0].plot(k[:7], e[:7], marker='o', ms=4.5, color=cols[st], lw=1.4,
                   label=rf'$d={dd[st]}$ parameters')
        ax[0].axvline(dd[st]+1, color=cols[st], ls=(0, (2, 2)), lw=0.9, alpha=0.7)
    ax[0].axhline(0.99, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax[0].text(6.4, 0.986, r'$99\%$', color=MUTED, fontsize=8, ha='right', va='top')
    ax[0].set_xlabel(r'bilinear rank $K$')
    ax[0].set_ylabel(r'variance of $f(\Phi,\theta)$ captured')
    ax[0].set_ylim(0.55, 1.015); ax[0].set_xlim(0.7, 6.5)
    ax[0].legend(loc='lower right', fontsize=8.4)
    panel_tag(ax[0], '(a)', loc=(0.03, 0.30))
    # (b)
    ax[1].axhline(0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax[1].errorbar(out['score_profile']['x'], out['score_profile']['y'],
                   yerr=out['score_profile']['ye'], marker='o', ms=4.5, color=BLUE,
                   lw=1.4, capsize=2)
    ax[1].set_xlabel(r'$1-T$'); ax[1].set_ylabel(r'$\langle\, \partial f/\partial\alpha_s \,\rangle$')
    ax[1].text(0.95, 0.08, rf'$r={r:.2f}$', transform=ax[1].transAxes, ha='right',
               va='bottom', fontsize=10, color=INK)
    panel_tag(ax[1], '(b)')
    # (c)
    labs = list(comp); x = np.arange(len(labs)); wd = 0.36
    bl = [comp[s]['bilinear'] for s in labs]; cc = [comp[s]['concat'] for s in labs]
    ax[2].bar(x-wd/2, bl, wd, color=BLUE, label=r'bilinear $\langle a,b\rangle$', zorder=2)
    ax[2].bar(x+wd/2, cc, wd, color=GREY, label=r'concatenation (DCTR)', zorder=2)
    ideal(ax[2], 1.0)
    ax[2].set_xticks(x); ax[2].set_xticklabels([rf'Stage {s}' for s in labs])
    ax[2].set_ylabel(r'match width $\sqrt{\chi^2/\mathrm{ndf}}$')
    ax[2].set_ylim(0.9, max(max(bl), max(cc))*1.12)
    ax[2].legend(loc='upper left', fontsize=8.6)
    panel_tag(ax[2], '(c)')
    for a in ax: trim(a)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'output/fig_m_ab.pdf')
    json.dump(out, open('output/ab_analysis.json', 'w'), indent=1)
    print('AB ANALYSIS DONE')


if __name__ == '__main__':
    main()
