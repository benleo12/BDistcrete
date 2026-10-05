#!/usr/bin/env python3
"""Stage E design: the redesigned Sherpa hadronization box.

Six axes, chosen for mechanical independence and measured potency (see notes in the paper):
  ALPHAS(MZ)        shower coupling
  AHADIC PT_MAX     the transverse lever. KT_0 is HELD at 1.21 because at fixed PT_MAX it
                    moves the mean cluster kt by 5 percent, and because the code only ever
                    uses GAMMA_L/KT_0^2 (Cluster_Splitter.C:155), so KT_0 and GAMMA_L are one
                    axis. Upper edge 1.00 GeV = the shower cutoff, which the model paper
                    (Chahal & Krauss, SciPost Phys. 13 (2022) 019, Eq. 2.6) defines the cap to be.
  AHADIC ALPHA_L    small-z exponent of the cluster-decay weight z^a (1-z)^b
  AHADIC GAMMA_L    the kt/mass damping, carrying the GAMMA_L/KT_0^2 direction
  AHADIC STRANGE_FRACTION
  AHADIC BARYON_FRACTION

Published anchor for every central value and its replica interval: Knobbe, Krauss, Reichelt,
Schumann, EPJC 84 (2024) 83 [arXiv:2306.03682] Appendix B Table 2, tuned with Sherpa 3.0beta,
the CSS dipole shower and MEPS@NLO at LEP1. PT_MAX is absent from that table; its scan range
[0.5,3.0] comes from the history matching of JHEP 08 (2026) [arXiv:2602.22324] Appendix A Table 1,
whose text states that "the PT_MAX and ALPHA_x parameters seem to drive much of the output changes".
Ranges below inflate the replica intervals, which are fit spread rather than model uncertainty.

ALPHA_G and BETA_L are HELD, but measured rather than assumed: a one-at-a-time scan of each is
generated so the paper can show they are weak instead of asserting it.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import qmc

AXES = [
    ('alphas',           'ALPHAS(MZ)',          0.118, 0.110, 0.130),
    ('pt_max',           'PT_MAX',              0.68,  0.40,  1.00),
    ('alpha_l',          'ALPHA_L',             3.9,   2.5,   5.0 ),
    ('gamma_l',          'GAMMA_L',             0.48,  0.28,  0.95),
    ('strange_fraction', 'STRANGE_FRACTION',    0.46,  0.36,  0.58),
    ('baryon_fraction',  'BARYON_FRACTION',     0.17,  0.11,  0.28),
    # Added 2026-09-10 after the one-at-a-time scans MEASURED both to be live on multiplicity
    # (chi2 against a flat line, ndf 4: ALPHA_G 155, BETA_L 128). They were held on code-reading
    # arguments -- ALPHA_G sits near the exactly-flat point a=1, BETA_L enters as beta*threshold/Q
    # -- and both arguments concern event shapes, where they are indeed weak. Ranges are the
    # history-matching scan ranges of arXiv:2602.22324 Table 1, which are sourced.
    ('alpha_g',          'ALPHA_G',             0.97,  0.60,  1.90),
    ('beta_l',           'BETA_L',              0.18,  0.04,  0.40),
]
N_TRAIN, N_HELD, CONTRACT, SEED = 96, 8, 0.15, 20260910   # 96 for d=8
lo = np.array([a[3] for a in AXES]); hi = np.array([a[4] for a in AXES]); d = len(AXES)

def maximin(n, seed, trials=400):
    best, score = None, -np.inf
    for s in range(trials):
        x = qmc.LatinHypercube(d=d, seed=seed+s, optimization='random-cd').random(n=n)
        dist = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
        m = dist[np.triu_indices(n, 1)].min()
        if m > score: score, best = m, x
    return best, score

X, sc = maximin(N_TRAIN, SEED); print(f'train: {N_TRAIN} pts, min pair distance {sc:.4f}')
Y, sc2 = maximin(N_HELD, SEED+9999); print(f'held : {N_HELD} pts (contracted {CONTRACT}), min dist {sc2:.4f}')
train = lo + X*(hi-lo)
held  = (lo+CONTRACT*(hi-lo)) + Y*((hi-lo)*(1-2*CONTRACT))

rows = []; rid = 7000
for p in train:
    rows.append(dict(run_id=rid, role='train', **{a[0]: v for a, v in zip(AXES, p)})); rid += 1
rid = 7200
for p in held:
    rows.append(dict(run_id=rid, role='held', **{a[0]: v for a, v in zip(AXES, p)})); rid += 1
# one-at-a-time scans of the two HELD knobs, at the box centre, to measure their potency
mid = {a[0]: a[2] for a in AXES}
rid = 7300
for v in (0.60, 0.80, 0.97, 1.40, 1.90):
    r = dict(mid); r.update(run_id=rid, role='scan_alpha_g', alpha_g=v); rows.append(r); rid += 1
rid = 7400
for v in (0.02, 0.10, 0.18, 0.28, 0.40):
    r = dict(mid); r.update(run_id=rid, role='scan_beta_l', beta_l=v); rows.append(r); rid += 1
df = pd.DataFrame(rows)[['run_id','role']+[a[0] for a in AXES]]
os.makedirs('configs', exist_ok=True)
df.to_csv('configs/stageE_design.csv', index=False)
print(f'\nwrote configs/stageE_design.csv: {len(df)} runs')
print(df.groupby('role').size().to_string())
print('\nper-axis span of the training set:')
for a in AXES:
    c = df[df.role=='train'][a[0]]; print(f'  {a[1]:<18} default {a[2]:<7} box [{a[3]}, {a[4]}]   sampled [{c.min():.4f}, {c.max():.4f}]')
