#!/usr/bin/env python3
"""figs/fig_thrust_vs_aleph.pdf: the matched NNLL+NNLO+shift thrust distribution against ALEPH.
Same ingredients as notes/thrust_tutorial/figs_tutorial.py (Fig 1). The perturbative envelope,
each variation with the (alpha_s, alpha_0) pair refitted, is drawn ONLY inside the fit window,
where the calculation is fitted, validated and used by the anchor. Outside the window the
central curve is dashed and no band is drawn: the refitted variations do not measure the
failure modes there (rigid shift below tau=0.05, effectively NLO four-parton kinematics
beyond tau=1/3), so any band drawn there would understate the uncertainty."""
import sys, os, json, numpy as np
sys.path.insert(0, '.'); os.chdir('.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import np_shift as N
from fit_alpha0_2d import load_grid, ASGRID, CSV
from match_v3 import fo_cumulants, fo_shift, logR, modR
from fit_alpha0 import aleph
from money_style import setup, save, trim, BLUE, GREY, INK, OCHRE
setup(usetex=True)
MZ = 91.1876; WLO, WHI = 0.05, 1.0/3.0
from thrust_chain import fo_total
A = load_grid(CSV); fo = fo_cumulants(); t_fo, cumA = fo['LO']; _, cumB = fo['NLO']; _, cumC = fo['NNLO']

def build(mu, xv, scheme, asmz, cfac=1.0):
    M = A[(mu, xv)]; tau = M[:, 0]; c = M[:, 2:6]; sig = M[:, 8 + ASGRID.index(asmz)]
    i = np.searchsorted(t_fo, tau[0]); sl = slice(i, i + len(tau))
    Ash, Bsh, Csh = fo_shift(*fo_total(cumA[sl], cumB[sl], cumC[sl]*cfac), np.log(mu**2))   # sigma-normalized
    asmu = N.alpha_s(mu*MZ, asmz); ab = asmu/(2*np.pi)
    S = [ab*Ash, ab**2*Bsh, ab**3*Csh]; R = [c[:, 1]*asmu, c[:, 2]*asmu**2, c[:, 3]*asmu**3]
    return tau, (logR(sig, R, S, 3) if scheme == 'logR' else modR(tau, sig, R, S, 3))

def binned(mu, xv, scheme, asmz, a0, lo, hi, cfac=1.0):
    ia = int(np.clip(np.searchsorted(ASGRID, asmz) - 1, 0, len(ASGRID) - 2))
    a1, a2 = ASGRID[ia], ASGRID[ia + 1]; w = (asmz - a1)/(a2 - a1)
    out = 0
    for aa, ww in ((a1, 1 - w), (a2, w)):
        tau, m = build(mu, xv, scheme, aa, cfac); S, _ = N.sanitize(m/m[-1])
        t = tau + N.shift(mu, a0, asmz=aa)
        out = out + ww*(N.cum_eval(hi, t, S) - N.cum_eval(lo, t, S))/(hi - lo)
    return out

lo, hi, y, e = aleph(); ctr = 0.5*(lo + hi)
fits = json.load(open('output/moments_joint.json'))['fits']
CS = json.load(open('output/chain_summary.json')); ASC, A0C = CS['alpha_s'], CS['alpha_0']
cen = binned(1.0, 1.0, 'logR', ASC, A0C, lo, hi)
curves = []
for k, (asm, a0, c2) in fits.items():
    mu, xv, sch, cf = eval(k); curves.append(binned(mu, xv, sch, asm, a0, lo, hi, cf))
env_lo, env_hi = np.min(curves, 0), np.max(curves, 0)
win = (ctr >= WLO) & (ctr < WHI); nan = np.where(win, 1.0, np.nan)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.06})
assert np.allclose(lo[1:], hi[:-1]), 'the bins must be contiguous to draw them as stairs'
edges = np.r_[lo, hi[-1]]
# a bin belongs to the window when its centre does; the curve inside and outside are drawn on the
# true bin edges, and the dashed curve joins the solid one at the window's first and last bin edges
for a, ylo, yhi, c_, band_lab in ((ax[0], env_lo, env_hi, cen, 'perturbative variations, refitted'),
                                  (ax[1], env_lo/y, env_hi/y, cen/y, None)):
    a.axvspan(WLO, WHI, color=OCHRE, alpha=0.10, lw=0)
    a.stairs(np.where(win, yhi, np.nan), edges, baseline=np.where(win, ylo, np.nan), fill=True,
             color=BLUE, alpha=0.25, lw=0, label=band_lab)
    a.stairs(np.where(win, c_, np.nan), edges, baseline=None, color=BLUE, lw=1.7,
             label=(r'NNLL+NNLO with dispersive shift' if a is ax[0] else None))
    a.stairs(np.where(win, np.nan, c_), edges, baseline=None, color=BLUE, lw=1.3, ls='--',
             label=('outside the fit window' if a is ax[0] else None))
    trim(a)
ax[0].errorbar(ctr, y, yerr=e, fmt='o', ms=3.4, color=INK, label='ALEPH 91.2 GeV', zorder=5)
ax[0].set_yscale('log'); ax[0].set_ylabel(r'$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}\tau$'); ax[0].set_ylim(3e-4, 60)
ax[0].legend(fontsize=11, frameon=False, loc='lower left', bbox_to_anchor=(0.07, 0.0)); ax[0].text(0.19, 30, 'fit window', ha='center', color=GREY, fontsize=11)
ax[1].axhline(1, color=GREY, lw=0.8)
ax[1].errorbar(ctr, np.ones_like(y), yerr=e/y, fmt='none', ecolor=INK, elinewidth=0.8)
ax[1].set_ylim(0.3, 1.7); ax[1].set_yticks([0.5, 1.0, 1.5]); ax[1].set_ylabel('theory / data'); ax[1].set_xlabel(r'$\tau = 1-T$'); ax[1].set_xlim(0, edges[-1])
r_ = cen/y; hid = (r_ < 0.3) | (r_ > 1.7); print(f'ratio range {np.nanmin(r_):.2f} to {np.nanmax(r_):.2f}, bins outside the lower panel: {ctr[hid]}')
save(fig, 'output/fig_thrust_vs_aleph.pdf'); print('wrote output/fig_thrust_vs_aleph.pdf')
pulls = ((cen - y)/e); print(f'window pull mean {pulls[win].mean():+.2f} rms {np.sqrt((pulls[win]**2).mean()):.2f}; '
                             f'tail (tau>1/3) theory/data min {np.min((cen/y)[ctr >= WHI]):.2f}')
