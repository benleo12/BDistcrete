#!/usr/bin/env python3
"""CONDITIONAL windowed targets (window edges exact, see functional): the fourteen moments normalized INSIDE the window,
c_i = <tau^a ln^b tau | 0.05 <= tau < 1/3>, i.e. the windowed integral divided by the window
fraction P(window) of the same calculation. The total-cross-section normalization, which
carries the peak region the calculation does not describe, cancels. Sigma_c from the same
12 refitted variations + experimental pair applied to the RATIOS. Also stores P(window)
central and per variation. Writes output/thrust_anchor_targets_cond.npz."""
import numpy as np, json, sys
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
from anchor_targets_window import WLO, WHI, AB, keys, FUN
# The window edges are exact: the cumulant is interpolated onto a fine grid inside [WLO, WHI) and integrated there.
# (Deciding membership by the midpoint of the calculation's own 0.0025-wide cells let a slice below 0.05 and one
# above 1/3 into the window, by a different amount for every variation because each has its own shift; that
# edge slice, not the scale variation, dominated the earlier Sigma_c.)
TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:]+TF[:-1]); FW = np.array([FUN[k](TMID) for k in keys])
def functional(mu, xv, sch, asmz, a0, cfac):
    ia = int(np.clip(np.searchsorted(TC.ASGRID, asmz)-1, 0, len(TC.ASGRID)-2)); a1, a2 = TC.ASGRID[ia], TC.ASGRID[ia+1]; w = (asmz-a1)/(a2-a1)
    mom = np.zeros(len(keys)); pw = 0.0
    for aa, ww in ((a1, 1-w), (a2, w)):
        tau, m = TC.build(mu, xv, sch, aa, cfac); S, _ = N.sanitize(m/m[-1]); t = tau + N.shift(mu, a0, asmz=aa)
        dS = np.diff(N.cum_eval(TF, t, S)); pw += ww*float(dS.sum()); mom += ww*(FW @ dS)
    return mom/pw, pw
CS = json.load(open('output/chain_summary.json')); fits = json.load(open('output/moments_joint.json'))['fits']
asc, a0c, sa, s0, rho = CS['alpha_s'], CS['alpha_0'], CS['alpha_s_err'], CS['alpha_0_err'], CS['rho']
central, pw0 = functional(1.0, 1.0, 'logR', asc, a0c, 1.0); vecs, pws, labels = [], [], []
for k, (asm, a0, c2) in fits.items():
    mu, xv, sch, cf = eval(k); v, p = functional(mu, xv, sch, asm, a0, cf); vecs.append(v); pws.append(p); labels.append(k)
for s in (+1, -1):
    v, p = functional(1.0, 1.0, 'logR', asc+s*sa, a0c+s*rho*s0, 1.0); vecs.append(v); pws.append(p); labels.append(f'exp{s:+d}')
V = np.array(vecs); D = V-central; Sig = (D.T@D)/2.0; sd = np.sqrt(np.diag(Sig))
np.savez('output/thrust_anchor_targets_cond.npz', keys=np.array(keys), central=central, Sigma_c=Sig, variations=V, labels=np.array(labels),
         window=np.array([WLO, WHI]), pwin_central=pw0, pwin_variations=np.array(pws))
T = np.load('output/thrust_anchor_targets.npz'); sdt = np.sqrt(np.diag(T['Sigma_c']))
print(f'P(window) central {pw0:.4f}, over variations {min(pws):.4f}..{max(pws):.4f}')
print('relative width per moment, total-normalized -> conditional:')
for i, k in enumerate(keys): print(f'  {k:8s} {sdt[i]/abs(T["central"][i]):6.2%} -> {sd[i]/abs(central[i]):6.2%}')
