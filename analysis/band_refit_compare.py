#!/usr/bin/env python3
"""The sentence of Sec. 6.2 on refitting the variations: at the ALEPH-fitted pair of App. D, the band
on the conditional mean thrust inside the window with every variation evaluated at the same
parameters (the construction of the targets), and with every variation refitted to ALEPH first
(moments_joint.json), both combined as in Eq. (sigmac) (theory_cov.band).

    python band_refit_compare.py        (writes output/band_refit_compare.json)
"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
from anchor_targets_window import WLO, WHI, keys, FUN
from theory_cov import band

# the functional of anchor_targets_window_cond.py (not imported: that module writes its targets on import)
TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:] + TF[:-1]); FW = np.array([FUN[k](TMID) for k in keys])
def functional(mu, xv, sch, asmz, a0, cfac):
    """windowed moments normalized inside the window, and the window fraction, linear between the coupling nodes"""
    ia = int(np.clip(np.searchsorted(TC.ASGRID, asmz) - 1, 0, len(TC.ASGRID) - 2)); a1, a2 = TC.ASGRID[ia], TC.ASGRID[ia + 1]
    w = (asmz - a1)/(a2 - a1); mom = np.zeros(len(keys)); pw = 0.0
    for aa, ww in ((a1, 1 - w), (a2, w)):
        tau, m = TC.build(mu, xv, sch, aa, cfac); S, _ = N.sanitize(m/m[-1]); t = tau + N.shift(mu, a0, asmz=aa)
        dS = np.diff(N.cum_eval(TF, t, S)); pw += ww*float(dS.sum()); mom += ww*(FW @ dS)
    return mom/pw, pw

CS = json.load(open('output/chain_summary.json')); fits = json.load(open('output/moments_joint.json'))['fits']
asc, a0c = CS['alpha_s'], CS['alpha_0']; k = keys.index('tau')
c0 = functional(1.0, 1.0, 'logR', asc, a0c, 1.0)[0][k]
labs = [l for l in fits if l != "(1.0, 1.0, 'logR', 1.0)"]
fixed, refit = [], []
for l in labs:
    mu, xv, sch, cf = eval(l); a, a0, _ = fits[l]
    fixed.append(functional(mu, xv, sch, asc, a0c, cf)[0][k] - c0)
    refit.append(functional(mu, xv, sch, a, a0, cf)[0][k] - c0)
bf, br = float(band(fixed, labs)), float(band(refit, labs))
print(f'conditional mean thrust in the window at the ALEPH pair ({asc:.4f}, {a0c:.3f}): {c0:.5f}')
print(f'band, variations at fixed parameters: {bf:.5f} ({100*bf/c0:.2f} percent); refitted: {br:.5f} ({100*br/c0:.2f} percent); ratio {bf/br:.2f}')
for l, f, r in zip(labs, fixed, refit):
    print(f'  {l:28s} fixed {f:+.5f}  refitted {r:+.5f}')
json.dump(dict(alpha_s=asc, alpha_0=a0c, mean=c0, band_fixed=bf, band_refit=br, ratio=bf/br,
               fixed=dict(zip(labs, fixed)), refit=dict(zip(labs, refit))), open('output/band_refit_compare.json', 'w'), indent=1)
