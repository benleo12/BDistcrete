#!/usr/bin/env python3
"""Single networks against the mean of members, on a trained Stage A to C model (App. B).

Each member of the ensemble is used alone, calibrated on its own on the selection half (the
temperature grid of r2_ladder.py), and scored on the comparison half, with the splits, the bins,
the pulls and the closure width of r2_ladder.py. The mean of members is scored the same way, which
reproduces the training's own result and checks the script.

    python single_members.py <model tag> <stage A|B|C> <data dir>
e.g. python single_members.py C_P2augfullfac0 C ../gen/data_abc
Writes output/single_members_<model tag>.json."""
import json, sys
import numpy as np
import pandas as pd
from ab_analysis import Cond
from r2_ladder import STAGES, make_bins, pulls, width_of, strange_frac

model, stage, data = sys.argv[1:4]
cfg = STAGES[stage]; OBS = cfg['obs'] + cfg['flav_obs']
cond = Cond(model); ref = np.load(f'output/models/{model}_ref.npz')
robs = {o: np.asarray(ref[o], float) for o in OBS}
GRID = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)] + [1.75, 2.0, 2.5, 3.0]))

held = {}
for rid, theta in cfg['held'].items():
    sh = pd.read_csv(f'{data}/shapes_run_{rid:04d}.csv'); d = np.load(f'{data}/particles_full_{rid:04d}.npz')
    t = {o: sh[o].values.astype(float) for o in cfg['obs']}
    if 'nbaryon' in OBS: t['nbaryon'] = d['nbaryon'].astype(float)
    if 'strange' in OBS: t['strange'] = strange_frac(d['particles'], d['mask']).astype(float)
    perm = np.random.default_rng(1000 + rid).permutation(len(sh)); n = len(sh)//2
    held[rid] = dict(theta=np.array(theta, float), obs=t, stop=perm[:n], rep=perm[n:],
                     bins={o: make_bins(np.r_[robs[o], t[o][perm[n:]]], o) for o in OBS})

# the log ratio of every member at every held-out point, on the whole reference
F = {rid: np.stack([cond.AE[mi] @ cond.bnet(mi, cond.tn(h['theta'])) for mi in range(cond.ENS)]) for rid, h in held.items()}


def closure(f_of, T, split):
    ws = []
    for rid, h in held.items():
        f = f_of(rid); w = np.exp((f - f.max())/T); w /= w.sum()
        ws.append(np.mean([width_of(pulls(robs[o], w, h['obs'][o][h[split]], h['bins'][o])[0]) for o in OBS]))
    return float(np.mean(ws))


def calibrated(f_of):
    T = min(GRID, key=lambda T: abs(closure(f_of, T, 'stop') - 1.0))
    return T, closure(f_of, T, 'rep')


out = {}
T, w = calibrated(lambda rid: F[rid].mean(0))
out['mean_of_members'] = dict(T=T, width=w)
print(f'[{model}] mean of members: T={T}, comparison-half width {w:.3f}', flush=True)
for mi in range(cond.ENS):
    T, w = calibrated(lambda rid, mi=mi: F[rid][mi])
    out[f'member_{mi}'] = dict(T=T, width=w)
    print(f'[{model}] member {mi} alone: T={T}, comparison-half width {w:.3f}', flush=True)
json.dump(out, open(f'output/single_members_{model}.json', 'w'), indent=1)
