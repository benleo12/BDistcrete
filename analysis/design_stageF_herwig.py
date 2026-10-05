#!/usr/bin/env python3
"""Stage F: the Herwig 7.3.0p1 variation box, public-release build.

Centred on the ACTUAL 7.3.0p1 shipped defaults, probed from the CVMFS release we generate with
(/cvmfs/.../herwig++/7.3.0p1-18d2e/.../share/Herwig/defaults/{Shower,Hadronization}.in):
  AlphaQCDFSR:AlphaIn 0.102337   PTCutOff:pTmin 0.654714 GeV
  ClMaxLight 3.528693   ClPowLight 1.849375   PSplitLight 0.914156
  ClSmrLight 0.78   PwtSquark 0.374094   PwtDIquark 0.33107

Two axes are CORRELATED LOCI, not rectangles. arXiv:1904.11866 (JHEP 04 (2020) 019) Table 1 gives
six full retunes of the same model on the same LEP data, differing only in the shower recoil
scheme. In that family the shower coupling and the shower cutoff rise together, and so do the
cluster-fission mass and power. Sampling them as independent rectangles admits corners no
published tune realises, so we sample position along each locus and allow only a narrow
perpendicular band. The locus DIRECTION and relative WIDTH come from the 7.2-era six-tune family;
they are applied about the 7.3 centres because the 7.3 retune moved the central values.

  six-tune family, arXiv:1904.11866 Table 1:
    AlphaIn  0.1074 0.1087 0.1136 0.1186 0.1244 0.1262   (span +-8.1% about the mid)
    pTmin    0.900  0.924  0.933  0.958  1.136  1.223    (span +-15.2%)
    ClMax    3.003  3.141  3.639  3.649  3.653  4.204    (span +-16.7%)
    ClPow    1.353  1.424  2.000  2.575  2.780  3.000    (span +-37.8%)

The remaining four axes are scalars. PwtSquark and PwtDIquark use the 7.3 normalisation: the
pre-7.2 values (0.597-0.737) are on a DIFFERENT normalisation, from before non-perturbative
g -> s sbar splitting was added as a second strangeness source, and must never be mixed in.
ClSmrLight is the honest wide one: no published Herwig tune has ever fitted it, and the one
Bayesian study to float it (arXiv:2302.01139 Table 3) prefers 0.675 against the 0.78 default.
NOT VARIED, and documented as traps in the release: ClusterFissioner:FissionPwtSquark is inert
unless Fission is set to 'new'; PartonSplitter:SplitPwtSquark 0.824135 was tuned to 7 TeV pp
minimum bias, never to e+e-.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import qmc

D_ALPHA, D_PTMIN = 0.102337, 0.654714
D_CLMAX, D_CLPOW = 3.528693, 1.849375
# Herwig HARD-LIMITS AlphaQCDFSR:AlphaIn at 0.10 (probed directly on the 7.3.0p1 build: 0.09 is
# rejected, 0.10 accepted). The 7.3 default is 0.102337, so there is almost no room BELOW it and
# the coupling axis has to be one-sided upward. Endpoints are given explicitly rather than as a
# relative width, so no inflation can push a parameter through a limit.
# t = -1 .. +1 runs along each locus, both members rising together, per arXiv:1904.11866 Table 1.
LOC = {
    'alpha_fsr': (0.1005, 0.1300),   # floor is Herwig's own limit, top covers the six-tune family
    'ptmin':     (0.50,   1.30  ),   # rises with the coupling
    'clmax':     (2.60,   4.80  ),
    'clpow':     (0.75,   3.20  ),   # floor above the Bayesian lower mode of arXiv:2302.01139
}
PERP = 0.30                        # perpendicular band, as a fraction of the half-range

def locus(t, perp, k1, k2):
    """t in [-1,1] along the locus (both rise together), perp in [-1,1] across it."""
    out = []
    for k, sgn in ((k1, +1.0), (k2, -1.0)):
        lo_, hi_ = LOC[k]; mid = 0.5*(lo_+hi_); half = 0.5*(hi_-lo_)
        v = mid + half*t + sgn*half*PERP*perp
        out.append(float(np.clip(v, lo_, hi_)))
    return tuple(out)

AXES = ['t_shower', 't_fission', 'psplit', 'pwtsquark', 'pwtdiquark', 'clsmr']
BOX  = [(-1, 1), (-1, 1), (0.65, 1.15), (0.22, 0.58), (0.20, 0.50), (0.40, 1.20)]
N_TRAIN, N_HELD, CONTRACT, SEED = 64, 8, 0.15, 20260909
d = len(AXES); lo = np.array([b[0] for b in BOX]); hi = np.array([b[1] for b in BOX])

def maximin(n, seed, trials=400):
    best, sc = None, -np.inf
    for s in range(trials):
        x = qmc.LatinHypercube(d=d, seed=seed+s, optimization='random-cd').random(n=n)
        dd = np.linalg.norm(x[:,None]-x[None,:], axis=-1); m = dd[np.triu_indices(n,1)].min()
        if m > sc: sc, best = m, x
    return best, sc

X, s1 = maximin(N_TRAIN, SEED); print(f'train {N_TRAIN} pts, min pair distance {s1:.4f}')
Y, s2 = maximin(N_HELD, SEED+7777); print(f'held  {N_HELD} pts (contracted {CONTRACT}), min dist {s2:.4f}')
tr = lo + X*(hi-lo)
hd = (lo+CONTRACT*(hi-lo)) + Y*((hi-lo)*(1-2*CONTRACT))
rng = np.random.default_rng(SEED)

def rows_from(P, role, rid0):
    out = []; rid = rid0
    for p in P:
        t, tf, ps, sq, dq, sm = p
        perp_s, perp_f = rng.uniform(-1,1,2)
        a, pt = locus(t,  perp_s, 'alpha_fsr', 'ptmin')
        cm, cp = locus(tf, perp_f, 'clmax', 'clpow')
        out.append(dict(run_id=rid, role=role, alpha_fsr=a, ptmin=pt, clmax=cm, clpow=cp,
                        psplit=ps, pwtsquark=sq, pwtdiquark=dq, clsmr=sm)); rid += 1
    return out

rows = rows_from(tr, 'train', 8000) + rows_from(hd, 'held', 8200)
mid = dict(alpha_fsr=D_ALPHA, ptmin=D_PTMIN, clmax=D_CLMAX, clpow=D_CLPOW,
           psplit=0.914156, pwtsquark=0.374094, pwtdiquark=0.33107, clsmr=0.78)
rid = 8300
for v in (0.40, 0.60, 0.78, 1.00, 1.20):       # ClSmrLight has never been tuned: scan it explicitly
    r = dict(mid); r.update(run_id=rid, role='scan_clsmr', clsmr=v); rows.append(r); rid += 1
df = pd.DataFrame(rows)[['run_id','role','alpha_fsr','ptmin','clmax','clpow','psplit','pwtsquark','pwtdiquark','clsmr']]
os.makedirs('configs', exist_ok=True)
df.to_csv('configs/stageF_design.csv', index=False)
print(f'\nwrote configs/stageF_design.csv: {len(df)} runs'); print(df.groupby('role').size().to_string())
print('\nrealised ranges (training set):')
for c, dflt in (('alpha_fsr',D_ALPHA),('ptmin',D_PTMIN),('clmax',D_CLMAX),('clpow',D_CLPOW),
                ('psplit',0.914156),('pwtsquark',0.374094),('pwtdiquark',0.33107),('clsmr',0.78)):
    v = df[df.role=='train'][c]; print(f'  {c:<12} 7.3 default {dflt:<10.5g} sampled [{v.min():.4f}, {v.max():.4f}]')
t = df[df.role=='train']
print(f'\nlocus correlations (should be strongly positive, not 0):')
print(f'  corr(alpha_fsr, ptmin) = {np.corrcoef(t.alpha_fsr, t.ptmin)[0,1]:+.3f}')
print(f'  corr(clmax, clpow)     = {np.corrcoef(t.clmax, t.clpow)[0,1]:+.3f}')
