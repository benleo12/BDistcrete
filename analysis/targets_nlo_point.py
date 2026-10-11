#!/usr/bin/env python3
"""The calculated moments of the NNLL+NLO chain at one fixed point, in the file layout of the theory
grid, so every reweighting script reads it as a grid with a single node.

The point is the pair (alpha_s, alpha_0) of the chain's own fit to the ALEPH window
(THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python thrust_chain.py). At that point: the fourteen windowed
moments of Eq. (thrustmoments), the window fraction, every scale and scheme variation of the
calculation at the SAME point (not refitted), the three variations of the dispersive model (Milan
factor +-20 percent, third-order subtraction term), and the covariance Sigma_c of Eq. (sigmac): the
two log-R scale pairs at half weight plus the modified-R scheme at central scales once. The paper's
diagonal floor is added by the reader (directlib.Model.sigma).

    THRUST_ORDER=2 python targets_nlo_point.py         # writes output/thrust_targets_nlo_point.npz
"""
import os, sys, json
os.environ.setdefault('THRUST_ORDER', '2')
assert os.environ['THRUST_ORDER'] == '2', 'this builder is for the NNLL+NLO chain'
import numpy as np
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
WLO, WHI = 0.05, 1.0/3.0                          # the window, as in anchor_targets_window.py
AB = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0), 'tlog2t': (1, 2), 't2logt': (2, 1),
      'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3), 't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
keys = list(N.G); assert set(keys) == set(AB)
FUN = {k: (lambda t, a=a, b=b: t**a*np.log(t)**b*((t >= WLO) & (t < WHI))) for k, (a, b) in AB.items()}

SFX = os.environ.get('CHAIN_SUFFIX', '_nlo'); OUT = os.environ.get('TARGETS_OUT', 'output/thrust_targets_nlo_point.npz')
TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:] + TF[:-1]); FW = np.array([FUN[k](TMID) for k in keys])
CS = json.load(open(f'output/chain_summary{SFX}.json')); asc, a0c = float(CS['alpha_s']), float(CS['alpha_0'])
assert asc in TC.ASGRID or min(abs(asc - a) for a in TC.ASGRID) < 1e-9 or True
SCALE_VARS = TC.VARS[1:]; NP_VARS = ['milan_up', 'milan_dn', 'sub3']
assert all(v[3] == 1.0 for v in SCALE_VARS), 'the NLO chain has no third-order variation'
PAIRS = [((2.0, 1.0, 'logR', 1.0), (0.5, 1.0, 'logR', 1.0)), ((1.0, 2.0, 'logR', 1.0), (1.0, 0.5, 'logR', 1.0))]
SCHEME = (1.0, 1.0, 'modR', 1.0)
_cache = {}


def cumulant(mu, xv, sch, asmz):
    key = (mu, xv, sch, asmz)
    if key not in _cache:
        tau, m = TC.build(mu, xv, sch, asmz); S, _ = N.sanitize(m/m[-1]); _cache[key] = (tau, S)
    return _cache[key]


def moments_at(mu, xv, sch, asmz, a0, milan=None, sub=None):
    """Windowed moments and window fraction at the pair, interpolated between the coupling nodes."""
    ia = int(np.clip(np.searchsorted(TC.ASGRID, asmz) - 1, 0, len(TC.ASGRID) - 2))
    a1, a2 = TC.ASGRID[ia], TC.ASGRID[ia + 1]; wgt = (asmz - a1)/(a2 - a1)
    out = np.zeros(len(keys)); pw = 0.0
    old_m, old_s = N.MILAN, N.SUB_ORDER
    if milan is not None: N.MILAN = milan
    if sub is not None: N.SUB_ORDER = sub
    try:
        for aa, ww in ((a1, 1 - wgt), (a2, wgt)):
            tau, S = cumulant(mu, xv, sch, aa); t = tau + N.shift(mu, a0, asmz=aa)
            dS = np.diff(N.cum_eval(TF, t, S)); out += ww*(FW @ dS)/dS.sum(); pw += ww*float(dS.sum())
    finally:
        N.MILAN, N.SUB_ORDER = old_m, old_s
    return out, pw


central, pwin = moments_at(1.0, 1.0, 'logR', asc, a0c)
scale = np.array([moments_at(*v[:3], asc, a0c)[0] for v in SCALE_VARS])
pwin_scale = np.array([moments_at(*v[:3], asc, a0c)[1] for v in SCALE_VARS])
npv = np.array([moments_at(1.0, 1.0, 'logR', asc, a0c, milan=1.2*N.MILAN)[0], moments_at(1.0, 1.0, 'logR', asc, a0c, milan=0.8*N.MILAN)[0],
                moments_at(1.0, 1.0, 'logR', asc, a0c, sub=3)[0]])
idx = {v: i for i, v in enumerate(SCALE_VARS)}
Sig = np.zeros((len(keys), len(keys)))
for a, b in PAIRS:
    for v in (a, b):
        d = scale[idx[v]] - central; Sig += 0.5*np.outer(d, d)
d = scale[idx[SCHEME]] - central; Sig += np.outer(d, d)
sd = np.sqrt(np.diag(Sig))
np.savez(OUT, keys=np.array(keys), alphas=np.array([asc]), alpha0=np.array([a0c]),
         central=central[None, None], pwin=np.array([[pwin]]), scale=scale[None, None], pwin_scale=pwin_scale[None, None],
         scale_labels=np.array([str(v) for v in SCALE_VARS]), np_vars=npv[None, None], np_labels=np.array(NP_VARS),
         sigma_pert=Sig[None, None], window=np.array([WLO, WHI]), milan=N.MILAN, sub_order_central=N.SUB_ORDER,
         order=2, chain_summary=json.dumps(CS))
print(f'NNLL+NLO point: alpha_s = {asc:.4f}, alpha_0 = {a0c:.4f}, window fraction {pwin:.4f}, {len(SCALE_VARS)} scale and scheme variations')
for k, cc, s in zip(keys, central, sd):
    print(f'  {k:8s} {cc:11.5g}  +- {s/abs(cc):6.2%}')
print(f'wrote {OUT}')
