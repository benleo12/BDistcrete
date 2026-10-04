#!/usr/bin/env python3
"""Combine EERAD3 NNLO production logs into the CUMULANT coefficient, rank by rank.

Why not bin by bin: EERAD3's local (antenna) subtraction puts a real-emission event and its
counter-event at slightly different thrust values, so in a 0.0025-wide histogram they land in
adjacent bins as a +-1e7 pair (rank 199 of prod2 carries +7.7e7 at tau=0.254). Such pairs
cancel in any integral over both bins (every rank's printed <1-T> coefficient is normal, and
plain, trimmed and median agree at 885 vs the published 867.6 +- 21) but they dominate every
per-bin mean, and a symmetric trim applied BIN BY BIN and then integrated acquires a
sample-dependent bias: 10-25% low in the fit window and up to 70% low in the tail for prod2,
2-9% HIGH for the noisier y0=1e-7 sample. Integrating first (per-rank cumulant, where the pairs
cancel unless they straddle tau) and trimming across ranks afterwards is stable between the
two independent productions to 2% at every tau, which is also its bootstrap error.

usage: eerad_cumulant_combine.py [--dirs logs/eerad_NNLO logs/eerad_NNLO_prod4 ...] [--trim 0.10]
       [--out output/eerad_NNLO_cum.npy] [--check logs/eerad_NNLO_y7]
writes rows (grid, C, SE_boot); C(tau) = -int_tau A3 in the fo_cumulants convention.
"""
import numpy as np, glob, sys, re
from scipy import stats
import eerad_nnlo_combine as E

def per_rank_cumulants(dirs):
    U = []; grid = None; ids = []
    for d in dirs:
        for p in sorted(glob.glob(f'{d}/prod_*.log')):
            B = E.get_block(p, '1/sig')
            if B is None or len(B) != 200 or not np.isfinite(B[:, 1]).all(): continue
            U.append(B[:, 1]); grid = B[:, 0]; ids.append(p)
    U = np.array(U); bw = float(np.diff(grid).mean()); n = len(U)
    above = np.concatenate([np.cumsum((U*bw)[:, ::-1], axis=1)[:, ::-1][:, 1:], np.zeros((n, 1))], axis=1)
    return grid, -(above + 0.5*U*bw), ids

def rank_trim(Cr, trim, nboot=400, seed=0):
    ct = np.array([stats.trim_mean(Cr[:, j], trim) for j in range(Cr.shape[1])])
    rng = np.random.default_rng(seed); n = len(Cr)
    bs = np.array([[stats.trim_mean(Cr[idx, j], trim) for j in range(Cr.shape[1])]
                   for idx in (rng.integers(0, n, n) for _ in range(nboot))])
    return ct, bs.std(0)

if __name__ == '__main__':
    a = sys.argv
    dirs = ['logs/eerad_NNLO']
    if '--dirs' in a:
        dirs = []
        for d in a[a.index('--dirs')+1:]:
            if d.startswith('--'): break
            dirs.append(d)
    trim = float(a[a.index('--trim')+1]) if '--trim' in a else 0.10
    out = a[a.index('--out')+1] if '--out' in a else 'output/eerad_NNLO_cum.npy'
    grid, Cr, ids = per_rank_cumulants(dirs)
    C, SE = rank_trim(Cr, trim)
    np.save(out, np.vstack([grid, C, SE]))
    print(f'{len(Cr)} ranks from {dirs}; trim {trim:.0%}; saved {out}')
    chk = a[a.index('--check')+1] if '--check' in a else None
    if chk:
        g2, Cr2, _ = per_rank_cumulants([chk]); C2, SE2 = rank_trim(Cr2, trim)
        print(f'check sample {chk}: {len(Cr2)} ranks')
    print(' tau      C        SE    rel' + ('     C_check   pull' if chk else ''))
    for t0 in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.36, 0.40]:
        i = np.argmin(abs(grid-t0)); line = f' {grid[i]:.3f} {C[i]:9.1f} {SE[i]:7.1f} {SE[i]/abs(C[i]):6.1%}'
        if chk: line += f'  {C2[i]:9.1f} {(C2[i]-C[i])/np.hypot(SE[i],SE2[i]):+6.2f}'
        print(line)
