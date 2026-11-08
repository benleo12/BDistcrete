#!/usr/bin/env python3
"""The data of the closure figure of the seventeen-parameter mixture: the reference reweighted with the
mean of members (MixtureHead of the export, at its calibration constant) to one Sherpa and one Herwig
held-out point, the median of each generator's closure widths on the standard test, against the
comparison half of the fresh run there, for thrust, the multiplicity and the baryon count. Bins and
widths as in ladder_widths_big.py. Writes output/fig_closure_mix.npz.

    DM_DATA=data_stagePURE17_v3 python fig_closure_mix.py 7214 8211

The rerun (score_obs.py): DM_DATA is required. BEAM_PHOTONS=remove takes the beam photons out of the
multiplicity on both sides (the reference's from REF_PARTICLES, the particles of its events) and
checks the reference shapes against the old bundle REF_ORIG; MIX_REF overrides the bundle and
CLOSURE_OUT the output file. With T_DIR set the export must carry the refitted temperature.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import MixtureHead
import r2_ladder as R
from mixture_cfg import mixture_cfg
import score_obs as SO

TAG = os.environ.get('MIX_TAG', 'MIXGEOV3S9')
SO.need(f"{os.environ['DM_DATA']}/meta.json")
cfg, _ = mixture_cfg(os.environ['DM_DATA']); DATA = cfg['data']
RIDS = [int(x) for x in sys.argv[1:]] or [7214, 8211]
# CLOSURE_MEDIAN_FROM=<widths_v2 json of the standard test>: the point of median closure width of each
# generator, as in the caption (the mean over the observables of each held-out run, the upper of the
# two middle runs of 24), in place of the points on the command line. On the published widths it
# gives 7214 and 8211.
if os.environ.get('CLOSURE_MEDIAN_FROM'):
    _pr = json.load(open(SO.need(os.environ['CLOSURE_MEDIAN_FROM'])))['per_run']
    RIDS = []
    for _xi in (0.0, 1.0):
        _m = {int(r): float(np.mean(list(v.values()))) for r, v in _pr.items() if cfg['held'][int(r)][-1] == _xi}
        _k = sorted(_m, key=lambda r: _m[r]); RIDS.append(_k[len(_k)//2])
    print(f'points of median closure width: {RIDS} (from {os.environ["CLOSURE_MEDIAN_FROM"]})', flush=True)
EXPORT = SO.need(f'output/models/{TAG}_cond.npz')
REF = SO.need(os.environ.get('MIX_REF', f'output/models/{TAG}_ref_v2_slim.npz'))
h = MixtureHead(EXPORT)
SO.check_export_T(EXPORT, h.T)
A = h.pack(np.asarray(np.load(EXPORT, mmap_mode='r')['AE']))
OBS = ['1_minus_thrust', 'mult_total', 'nbaryon']
REF_PARTICLES, REF_ORIG = os.environ.get('REF_PARTICLES', ''), os.environ.get('REF_ORIG', '')
if SO.REMOVE:
    assert REF_PARTICLES and REF_ORIG, 'BEAM_PHOTONS=remove: set REF_PARTICLES and REF_ORIG'
POBS = os.environ.get('POBS_DIR', '')
robs = SO.reference_obs(REF, OBS, particles=REF_PARTICLES or None, orig=REF_ORIG or None,
                        cache=(os.path.join(POBS, os.path.basename(os.path.realpath(REF_PARTICLES)).replace('.npz', '_pobs.npz'))
                               if POBS and REF_PARTICLES else None))
N = len(robs[OBS[0]]); uni = np.full(N, 1.0/N)


def targets(rid):
    sh = SO.read_shapes(f'{DATA}/shapes_run_{rid:04d}.csv')
    d = np.load(SO.need(f'{DATA}/particles_full_{rid:04d}.npz'))
    out = {o: sh[o].values.astype(float) for o in OBS if o != 'nbaryon'}
    out['nbaryon'] = d['nbaryon'].astype(float)
    if SO.REMOVE:
        out['mult_total'] = SO.mult_total(out['mult_total'], d['particles'], d['mask'], where=f'{DATA} run {rid}').astype(float)
    return out


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
    T_ = targets(rid); n = len(T_[OBS[0]])
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
OUT = os.environ.get('CLOSURE_OUT', 'output/fig_closure_mix.npz')
extra = dict(beam_photons=np.array(SO.MODE), T=np.array(h.T), ref=np.array(REF)) if (SO.REMOVE or SO.T_DIR) else {}
np.savez(OUT, rids=np.array(RIDS), obs=np.array(OBS), **out, **extra)
print(f'wrote {OUT}')
