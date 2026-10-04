#!/usr/bin/env python3
"""Binned closure widths (the statistic of Sec. 4, the paper's tab:ladder convention) for the
eight-parameter Sherpa and Herwig stages and the seventeen-parameter mixture, on the corrected
shapes. Same estimator as ladder_widths_v2.py: bins from reference plus the report half of each
held run, per-bin pulls with both sides' multinomial errors, width = sqrt(chi2/ndf) about zero,
master width = mean over held runs and observables, at the shipped temperature.

    python ladder_widths_big.py E output/models/E_cond.npz output/models/E_ref.npz
    STAGEF_CSV=stageF_design_aug.csv python ladder_widths_big.py F output/models/Fauglong_cond.npz output/models/Fauglong_ref.npz
    DM_DATA=data_stageDM17aug_v2 python ladder_widths_big.py MIX output/models/MIX17aug_cond.npz output/models/MIX17aug_ref.npz
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head, MixtureHead
import r2_ladder as R
from make_stage_table import strange_frac
tag, export, refpath = sys.argv[1], sys.argv[2], sys.argv[3]
if tag == 'MIX':
    from mixture_cfg import mixture_cfg
    cfg, _ = mixture_cfg(os.environ['DM_DATA'])
else:
    cfg = R.STAGES[tag]
DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
kind = str(np.load(export).get('head_kind', 'cond'))
h = (MixtureHead if kind == 'mixture' else Head)(export)
A = h.pack(np.asarray(np.load(export, mmap_mode='r')['AE']))
ref = np.load(refpath); robs = {o: np.asarray(ref[o], np.float64) for o in OBS}
assert len(robs[OBS[0]]) == A.shape[0]
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
print(f'{tag} ({os.path.basename(export)}) on {DATA}, T={h.T}: report-split width {master:.3f} '
      f'(uniform control {umaster:.2f}), N_eff/N {min(neffs.values()):.2f} to {max(neffs.values()):.2f}, {len(cfg["held"])} held runs', flush=True)
res = dict(tag=tag, export=export, T=h.T, master=master, uniform=umaster, neff=[min(neffs.values()), max(neffs.values())],
           per_obs={o: float(np.mean([per[r, o] for r in cfg['held']])) for o in OBS},
           per_obs_uniform={o: float(np.mean([uper[r, o] for r in cfg['held']])) for o in OBS},
           per_run={str(r): {o: float(per[r, o]) for o in OBS} for r in cfg['held']})
for o in OBS: print(f'   {o:16s} width {res["per_obs"][o]:.3f}   uniform {res["per_obs_uniform"][o]:.2f}', flush=True)
json.dump(res, open(f'output/widths_v2_{os.path.basename(export).replace("_cond.npz", "")}.json', 'w'), indent=1)
