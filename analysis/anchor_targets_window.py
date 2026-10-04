#!/usr/bin/env python3
"""Anchor targets for the WINDOWED basis: the 14 functions tau^a ln^b tau (1<=a+b<=4) restricted to
0.05 <= tau < 1/3, the range in which the calculation is fitted and validated. Central values and
Sigma_c (12 refitted variations + experimental pair, no floor: anchor_thrust.py adds it) from the
matched calculation exactly as thrust_chain.py does for the full-range moments. Writes
output/thrust_anchor_targets.npz in the same format (the full-range file is kept as
thrust_anchor_targets_full.npz)."""
import numpy as np, json, sys
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
WLO, WHI = 0.05, 1.0/3.0
AB = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0), 'tlog2t': (1, 2), 't2logt': (2, 1),
      'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3), 't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
keys = list(N.G); assert set(keys) == set(AB)
def win_funcs():
    return {k: (lambda t, a=a, b=b: t**a*np.log(t)**b*((t >= WLO) & (t < WHI))) for k, (a, b) in AB.items()}
FUN = win_funcs()
TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:]+TF[:-1]); FW = np.array([FUN[k](TMID) for k in keys])
def functional(mu, xv, sch, asmz, a0, cfac):
    ia = int(np.clip(np.searchsorted(TC.ASGRID, asmz)-1, 0, len(TC.ASGRID)-2)); a1, a2 = TC.ASGRID[ia], TC.ASGRID[ia+1]; w = (asmz-a1)/(a2-a1)
    out = np.zeros(len(keys))
    for aa, ww in ((a1, 1-w), (a2, w)):   # window edges exact: cumulant interpolated onto a fine grid inside the window
        tau, m = TC.build(mu, xv, sch, aa, cfac); S, _ = N.sanitize(m/m[-1]); t = tau + N.shift(mu, a0, asmz=aa)
        out += ww*(FW @ np.diff(N.cum_eval(TF, t, S)))
    return out
CS = json.load(open('output/chain_summary.json')); fits = json.load(open('output/moments_joint.json'))['fits']
asc, a0c, sa, s0, rho = CS['alpha_s'], CS['alpha_0'], CS['alpha_s_err'], CS['alpha_0_err'], CS['rho']
central = functional(1.0, 1.0, 'logR', asc, a0c, 1.0); vecs = []; labels = []
for k, (asm, a0, c2) in fits.items():
    mu, xv, sch, cf = eval(k); vecs.append(functional(mu, xv, sch, asm, a0, cf)); labels.append(k)
for s in (+1, -1):
    vecs.append(functional(1.0, 1.0, 'logR', asc+s*sa, a0c+s*rho*s0, 1.0)); labels.append(f'exp{s:+d}')
V = np.array(vecs); D = V-central; Sig = (D.T@D)/2.0
sd = np.sqrt(np.diag(Sig))
np.savez('output/thrust_anchor_targets.npz', keys=np.array(keys), central=central, Sigma_c=Sig, variations=V, labels=np.array(labels),
         window=np.array([WLO, WHI]))
print('windowed targets:', {k: f'{c:.5g} ({s/abs(c):.2%})' for k, c, s in zip(keys, central, sd)})
print('rank of Sigma_c:', np.linalg.matrix_rank(Sig), 'wrote output/thrust_anchor_targets.npz (window %.2f-%.4f)' % (WLO, WHI))
