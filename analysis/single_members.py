#!/usr/bin/env python3
"""Single networks against the ensemble average, on a trained Stage A to C model (App. A.2).

Each member of the ensemble is used alone, calibrated on its own on the selection half (the
temperature grid of r2_ladder.py), and scored on the comparison half, with the splits, the bins,
the pulls and the closure width of r2_ladder.py. The mean of members is scored the same way, which
reproduces the training's own result and checks the script.

    python single_members.py <model tag> <stage A|B|C> <data dir>
e.g. python single_members.py C_P2augfullfac0 C ../gen/data_abc
Writes output/single_members_<model tag>.json (SM_OUT overrides it).

The rerun (score_obs.py): BEAM_PHOTONS=remove takes the beam photons out of the multiplicity and the
kaon fraction on both sides (the reference's from the particles of SM_REF_PARTICLES, default the
bundle) and checks the reference shapes against the old bundle SM_REF_ORIG. SM_REF overrides the
bundle output/models/<model tag>_ref.npz. The calibration constants are fitted here, as before."""
import json, os, sys
import numpy as np
import pandas as pd
from ab_analysis import Cond
from r2_ladder import STAGES, make_bins, pulls, width_of, register_sorted
import score_obs as SO

model, stage, data = sys.argv[1:4]
cfg = STAGES[stage]; OBS = cfg['obs'] + cfg['flav_obs']
REF = SO.need(os.environ.get('SM_REF', f'output/models/{model}_ref.npz'))
REF_PARTICLES = os.environ.get('SM_REF_PARTICLES', REF); REF_ORIG = os.environ.get('SM_REF_ORIG', '')
cond = Cond(model)
robs = SO.reference_obs(REF, OBS, particles=REF_PARTICLES, orig=REF_ORIG or None)
for _o in OBS:
    register_sorted(robs[_o])          # LADDER_FAST_HIST=1: the same histograms, without sorting them again per T
GRID = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)] + [1.75, 2.0, 2.5, 3.0]))

held = {}
for rid, theta in cfg['held'].items():
    sh = SO.read_shapes(f'{data}/shapes_run_{rid:04d}.csv'); d = np.load(SO.need(f'{data}/particles_full_{rid:04d}.npz'))
    t = {o: sh[o].values.astype(float) for o in cfg['obs']}
    P, M = d['particles'], d['mask']
    nb = SO.n_beam(P, M) if SO.REMOVE else None
    if 'mult_total' in t: t['mult_total'] = SO.mult_total(t['mult_total'], P, M, where=f'{data} run {rid}', nb=nb)
    if 'nbaryon' in OBS: t['nbaryon'] = d['nbaryon'].astype(float)
    if 'strange' in OBS: t['strange'] = SO.strange_frac(P, M, nb=nb).astype(float)
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
if SO.REMOVE:
    out['provenance'] = dict(SO.describe(), ref=REF, ref_particles=REF_PARTICLES, ref_orig=REF_ORIG or None,
                             data=data, data_realpath=os.path.realpath(data))
json.dump(out, open(os.environ.get('SM_OUT', f'output/single_members_{model}.json'), 'w'), indent=1)
