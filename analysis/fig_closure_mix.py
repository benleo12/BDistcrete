#!/usr/bin/env python3
"""The data of the closure figure of the seventeen-parameter mixture: the reference reweighted with the
mean of members (MixtureHead of the export, at its calibration constant) to one Sherpa and one Herwig
held-out point, the median of each generator's closure widths on the standard test, against the
comparison half of the fresh run there, for thrust, the multiplicity and the baryon count. Bins and
widths as in ladder_widths_big.py. Writes output/fig_closure_mix.npz.

    DM_DATA=data_stagePURE17_v3 python fig_closure_mix.py 7214 8211
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import MixtureHead
import r2_ladder as R
from mixture_cfg import mixture_cfg

TAG = os.environ.get('MIX_TAG', 'MIXGEOV3S9')
RIDS = [int(x) for x in sys.argv[1:]] or [7214, 8211]
cfg, _ = mixture_cfg(os.environ.get('DM_DATA', 'data_stagePURE17_v3')); DATA = cfg['data']
h = MixtureHead(f'output/models/{TAG}_cond.npz')
A = h.pack(np.asarray(np.load(f'output/models/{TAG}_cond.npz', mmap_mode='r')['AE']))
ref = np.load(f'output/models/{TAG}_ref_v2_slim.npz')
OBS = ['1_minus_thrust', 'mult_total', 'nbaryon']
robs = {o: np.asarray(ref[o], np.float64) for o in OBS}
N = len(robs[OBS[0]]); uni = np.full(N, 1.0/N)


def target(rid, o):
    if o == 'nbaryon':
        return np.load(f'{DATA}/particles_full_{rid:04d}.npz')['nbaryon'].astype(float)
    return pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')[o].values.astype(float)


def dens_w(v, w, bins):
    hh, _ = np.histogram(v, bins, weights=w); h2, _ = np.histogram(v, bins, weights=w**2)
    W = hh.sum(); bw = np.diff(bins)
    return hh/(W*bw), np.sqrt(h2)/(W*bw)


def dens_u(v, bins):
    n, _ = np.histogram(v, bins); bw = np.diff(bins)
    return n/(n.sum()*bw), np.sqrt(n)/(n.sum()*bw)


out = {}
for rid in RIDS:
    th = np.array(cfg['held'][rid]); w = h.weights(A, th)
    T_ = {o: target(rid, o) for o in OBS}; n = len(T_[OBS[0]])
    rep = np.random.default_rng(1000 + rid).permutation(n)[n//2:]          # the comparison half, as ladder_widths_big
    for o in OBS:
        tv = T_[o][rep]; bins = R.make_bins(np.r_[robs[o], tv], o)
        dw, ew = dens_w(robs[o], w, bins); du, _ = dens_w(robs[o], uni, bins); dt, et = dens_u(tv, bins)
        p, _ = R.pulls(robs[o], w, tv, bins)
        for k, v in (('bins', bins), ('dw', dw), ('ew', ew), ('du', du), ('dt', dt), ('et', et)):
            out[f'{rid}__{o}__{k}'] = np.asarray(v, float)
        out[f'{rid}__{o}__width'] = np.array(R.width_of(p))
    out[f'{rid}__theta'] = th
    print(rid, 'widths', {o: round(float(out[f"{rid}__{o}__width"]), 3) for o in OBS}, 'xi', th[-1], flush=True)
np.savez('output/fig_closure_mix.npz', rids=np.array(RIDS), obs=np.array(OBS), **out)
print('wrote output/fig_closure_mix.npz')
