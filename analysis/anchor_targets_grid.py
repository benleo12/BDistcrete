#!/usr/bin/env python3
"""Theory targets for the fit: the fourteen windowed conditional thrust moments on a grid of
the two theory parameters, the MSbar coupling and the dispersive alpha_0, with the theory
covariance at FIXED parameters and no prefit to any experiment.

This replaces anchor_targets_window_cond.py for the coupling fit. That script evaluated the
targets at the ALEPH-fitted pair and built Sigma_c from variations each REFITTED to ALEPH,
which is the right band for a fixed-coupling prediction and the wrong one for an extraction,
where the refit spread is the perturbative error ON THE COUPLING and must not be pre-absorbed
into the targets. Here every variation is evaluated at the same (alpha_s, alpha_0) as the
central value, and the experimental displacement vector is gone: the data enter the fit, not
the targets.

Stored per grid node: central moments, window fraction, the twelve scale/scheme/fixed-order
variations, and two nonperturbative model variations (Milan factor +-20 percent, and the
O(alpha_s^3) term of the renormalon subtraction switched on), kept separate so the fit can
refit under each. Sigma_pert (theory_cov.py) is the half-sum of squared displacements over the
log-R scale pairs and the third-order pair plus the scheme difference once, the covariance the tilt
is regularized with and the propagated bands use.

    python anchor_targets_grid.py            # writes output/thrust_targets_grid.npz
"""
import os, sys, json, importlib
import numpy as np
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
from theory_cov import sigma_pert
from anchor_targets_window import WLO, WHI, keys, FUN   # window edges and the moment functions

TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:] + TF[:-1])
FW = np.array([FUN[k](TMID) for k in keys])
ASG = list(TC.ASGRID)                                   # the calculation's own alpha_s nodes
A0G = np.round(np.arange(float(os.environ.get('A0_MIN', '0.25')), float(os.environ.get('A0_MAX', '0.65')) + 1e-4, 0.01), 4)
OUT_GRID = os.environ.get('TARGETS_OUT', 'output/thrust_targets_grid.npz')
SCALE_VARS = TC.VARS[1:]                                 # the twelve, the first entry is central
NP_VARS = ['milan_up', 'milan_dn', 'sub3']

_cache = {}
def cumulant(mu, xv, sch, asmz, cfac):
    key = (mu, xv, sch, asmz, cfac)
    if key not in _cache:
        tau, m = TC.build(mu, xv, sch, asmz, cfac)
        S, _ = N.sanitize(m/m[-1])
        _cache[key] = (tau, S)
    return _cache[key]

def moments_at(mu, xv, sch, asmz, a0, cfac, milan=None, sub=None):
    """Windowed conditional moments and window fraction at one node of the grid."""
    tau, S = cumulant(mu, xv, sch, asmz, cfac)
    old_m, old_s = N.MILAN, N.SUB_ORDER
    if milan is not None: N.MILAN = milan
    if sub is not None: N.SUB_ORDER = sub
    try:
        t = tau + N.shift(mu, a0, asmz=asmz)
    finally:
        N.MILAN, N.SUB_ORDER = old_m, old_s
    dS = np.diff(N.cum_eval(TF, t, S))
    pw = float(dS.sum())
    return (FW @ dS)/pw, pw

nA, n0, nk = len(ASG), len(A0G), len(keys)
central = np.zeros((nA, n0, nk)); pwin = np.zeros((nA, n0))
scale = np.zeros((nA, n0, len(SCALE_VARS), nk)); pwin_scale = np.zeros((nA, n0, len(SCALE_VARS)))
npv = np.zeros((nA, n0, len(NP_VARS), nk))
sigma = np.zeros((nA, n0, nk, nk))
for i, a in enumerate(ASG):
    for j, a0 in enumerate(A0G):
        c, p = moments_at(1.0, 1.0, 'logR', a, a0, 1.0); central[i, j] = c; pwin[i, j] = p
        for v, (mu, xv, sch, cf) in enumerate(SCALE_VARS):
            scale[i, j, v], pwin_scale[i, j, v] = moments_at(mu, xv, sch, a, a0, cf)
        npv[i, j, 0], _ = moments_at(1.0, 1.0, 'logR', a, a0, 1.0, milan=1.2*N.MILAN)
        npv[i, j, 1], _ = moments_at(1.0, 1.0, 'logR', a, a0, 1.0, milan=0.8*N.MILAN)
        npv[i, j, 2], _ = moments_at(1.0, 1.0, 'logR', a, a0, 1.0, sub=3)
        sigma[i, j] = sigma_pert(c, scale[i, j], [str(v) for v in SCALE_VARS])   # Eq. (sigmac), theory_cov.py
    print(f'alpha_s {a:.3f} done', flush=True)
np.savez(OUT_GRID, keys=np.array(keys), alphas=np.array(ASG), alpha0=A0G,
         central=central, pwin=pwin, scale=scale, pwin_scale=pwin_scale,
         scale_labels=np.array([str(v) for v in SCALE_VARS]), np_vars=npv, np_labels=np.array(NP_VARS),
         sigma_pert=sigma, window=np.array([WLO, WHI]), milan=N.MILAN, sub_order_central=N.SUB_ORDER)
# consistency with the fixed-coupling targets at the ALEPH-fitted pair, which those used
CS = json.load(open('output/chain_summary.json'))
T = np.load('output/thrust_anchor_targets_cond.npz', allow_pickle=True)
c_fit, _ = moments_at(1.0, 1.0, 'logR', ASG[min(range(nA), key=lambda i: abs(ASG[i]-CS['alpha_s']))], CS['alpha_0'], 1.0)
ia = int(np.clip(np.searchsorted(ASG, CS['alpha_s'])-1, 0, nA-2)); w = (CS['alpha_s']-ASG[ia])/(ASG[ia+1]-ASG[ia])
c_int = (1-w)*moments_at(1.0, 1.0, 'logR', ASG[ia], CS['alpha_0'], 1.0)[0] + w*moments_at(1.0, 1.0, 'logR', ASG[ia+1], CS['alpha_0'], 1.0)[0]
print('at the ALEPH-fitted pair, interpolated between nodes, relative difference to the old central targets:')
print('   ' + '  '.join(f'{k} {100*(c_int[n]/T["central"][n]-1):+.3f}%' for n, k in enumerate(keys)))
sd_old = np.sqrt(np.diag(T['Sigma_c'])); sd_new = np.sqrt(np.diag((1-w)*sigma[ia, np.argmin(abs(A0G-CS['alpha_0']))] + w*sigma[ia+1, np.argmin(abs(A0G-CS['alpha_0']))]))
print('relative theory width per moment, refit band (old) -> fixed-parameter band (new):')
print('   ' + '  '.join(f'{k} {100*sd_old[n]/abs(T["central"][n]):.2f}->{100*sd_new[n]/abs(c_int[n]):.2f}%' for n, k in enumerate(keys)))
print(f'window fraction at the fitted pair: {pwin[ia, np.argmin(abs(A0G-CS["alpha_0"]))]:.4f}; grid {nA} x {n0} nodes; wrote {OUT_GRID}')
