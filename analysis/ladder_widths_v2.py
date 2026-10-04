#!/usr/bin/env python3
"""The paper's ladder closure (tab:ladder) recomputed on the corrected samples.

Same estimator as the training run and as recalibrate_T.py: bins fixed from reference plus the
report half of each held run, per-bin pulls with both sides' multinomial errors, width =
sqrt(chi2/ndf) about zero with null 1, the master width the mean over held runs and observables,
the uniform-weight control, and N_eff. Evaluated at the head's shipped temperature on the
REPORT half, which is the paper's number. Both the v1 and v2 data are run so the change from
the exact thrust axis is seen directly.

    STAGEC_DATA=data_stageC_v2 python ladder_widths_v2.py C output/models/C_ref_v2.npz
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head
import r2_ladder as R
from make_stage_table import strange_frac
tag, refpath = sys.argv[1], sys.argv[2]
cfg = R.STAGES[tag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
h = Head(f'output/models/{tag}_cond.npz')
A = h.pack(np.asarray(np.load(f'output/models/{tag}_cond.npz', mmap_mode='r')['AE']))
ref = np.load(refpath, mmap_mode='r'); robs = {o: np.asarray(ref[o], np.float64) for o in OBS}
def target(rid, o):
    if o in ('nbaryon', 'strange'):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        return d['nbaryon'].astype(float) if o == 'nbaryon' else strange_frac(d['particles'], d['mask']).astype(float)
    return pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')[o].values.astype(float)
uni = np.full(len(robs[OBS[0]]), 1.0/len(robs[OBS[0]]))
per = {}; uper = {}; neffs = {}
for rid, th in cfg['held'].items():
    T_ = {o: target(rid, o) for o in OBS}; n = len(T_[OBS[0]])
    perm = np.random.default_rng(1000 + rid).permutation(n); rep = perm[n//2:]
    w = h.weights(A, np.array(th)); neffs[rid] = h.n_eff(w)/len(w)
    for o in OBS:
        b = R.make_bins(np.r_[robs[o], T_[o][rep]], o)
        p, nb = R.pulls(robs[o], w, T_[o][rep], b); per[rid, o] = R.width_of(p)
        pu, _ = R.pulls(robs[o], uni, T_[o][rep], b); uper[rid, o] = R.width_of(pu)
master = float(np.mean([per[k] for k in per])); umaster = float(np.mean([uper[k] for k in uper]))
print(f'{tag} on {DATA} with {os.path.basename(refpath)}, T={h.T}: report-split width {master:.3f} '
      f'(uniform control {umaster:.2f}), N_eff/N {min(neffs.values()):.2f} to {max(neffs.values()):.2f}')
for o in OBS: print(f'   {o:16s} width {np.mean([per[r, o] for r in cfg["held"]]):.3f}   uniform {np.mean([uper[r, o] for r in cfg["held"]]):.2f}')
