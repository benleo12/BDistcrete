#!/usr/bin/env python3
"""Two numbers App. D quotes, on the current chain (thrust_chain.py, sigma-normalized, PCHIP):
  1. the clipping of the cumulant (np_shift.sanitize): its dip below zero at the ALEPH-fitted pair,
     and how much it moves <ln^4 tau> there and, under every variation, at alpha_s = 0.118 with
     alpha_0 at its best value for that coupling;
  2. with NP_SUB_ORDER=3 set in the environment, the ALEPH fit with the third-order term of the
     renormalon subtraction (run the script twice, with and without it).

    python appD_checks.py;  NP_SUB_ORDER=3 python appD_checks.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, '.')
sys.argv = sys.argv[:1]
import np_shift as N, thrust_chain as TC

def ln4(mu, xv, sch, a, a0, cf, clean):
    tau, m = TC.build(mu, xv, sch, a, cf); sh = N.shift(mu, a0, asmz=a)
    return N.moments(tau + sh, m/m[-1], clean=clean, lo=sh)['log4t'], float((m/m[-1]).min())

out = {}
if N.SUB_ORDER >= 3:
    asm, a0, c2, _ = TC.profile(1.0, 1.0, 'logR', 1.0)
    print(f'third-order subtraction: alpha_s {asm:.4f}, alpha_0 {a0:.3f}, chi2 {c2:.2f}')
    out['sub3'] = dict(alpha_s=asm, alpha_0=a0, chi2=c2)
    json.dump(out, open('output/appD_checks_sub3.json', 'w'), indent=1); sys.exit()
CS = json.load(open('output/chain_summary.json')); a, a0 = CS['alpha_s'], CS['alpha_0']
# the fitted pair, interpolated linearly between the two coupling nodes around it
ia = int(np.clip(np.searchsorted(TC.ASGRID, a) - 1, 0, len(TC.ASGRID) - 2)); w = (a - TC.ASGRID[ia])/(TC.ASGRID[ia + 1] - TC.ASGRID[ia])
cl = un = dip = 0.0
for aa, ww in ((TC.ASGRID[ia], 1 - w), (TC.ASGRID[ia + 1], w)):
    c, d = ln4(1.0, 1.0, 'logR', aa, a0, 1.0, True); u, _ = ln4(1.0, 1.0, 'logR', aa, a0, 1.0, False)
    cl += ww*c; un += ww*u; dip = min(dip, d)
print(f'fitted pair: cumulant minimum {dip:.1e}, <ln^4 tau> moved by {100*abs(cl/un - 1):.3f} percent')
out['fitted'] = dict(dip=dip, rel=abs(cl/un - 1))
rel = []
for mu, xv, sch, cf in TC.VARS:
    # alpha_0 at its best value for alpha_s = 0.118 under this variation
    tau, m = TC.build(mu, xv, sch, 0.118, cf); S, _ = N.sanitize(m/m[-1])
    c2 = [TC.chi2_of(tau, S, mu, 0.118, x) for x in TC.A0GRID]; a0b = float(TC.A0GRID[int(np.argmin(c2))])
    c, d = ln4(mu, xv, sch, 0.118, a0b, cf, True); u, _ = ln4(mu, xv, sch, 0.118, a0b, cf, False)
    rel.append(abs(c/u - 1)); print(f'  alpha_s 0.118 {(mu, xv, sch, cf)}: alpha_0 {a0b:.3f}, minimum {d:.1e}, moved {100*rel[-1]:.3f} percent')
print(f'at alpha_s = 0.118: largest change over the variations {100*max(rel):.2f} percent')
out['at_0118_max_rel'] = max(rel)
json.dump(out, open('output/appD_checks.json', 'w'), indent=1)
