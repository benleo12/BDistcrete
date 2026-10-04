#!/usr/bin/env python3
"""Mean baryon count and strange-hadron fraction of the seven-parameter mixture stage (Sec. 5.6),
reweighted against fresh, at each held-out run: the relative difference in percent and the
statistical resolution of the comparison, from the report half of each run as in the closure test.

    EXPORT=output/models/DMDMXSc0_cond.npz python rate_check_mix.py output/models/DMDMXSc0_ref.npz
"""
import sys, os, json
import numpy as np
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import MixtureHead
from make_stage_table import strange_frac

refpath = sys.argv[1]; DATA = os.environ.get('DM_DATA', 'data_stageDM2'); export = os.environ['EXPORT']
meta = json.load(open(f'{DATA}/meta.json'))
held = {int(m['rid']): tuple(m['theta']) for m in meta['held'] if int(m['rid']) not in (9932,)}
h = MixtureHead(export); A = h.pack(np.asarray(np.load(export, mmap_mode='r')['AE']))
ref = np.load(refpath, mmap_mode='r')
robs = {o: np.asarray(ref[o], np.float64) for o in ('nbaryon', 'strange')}
out = {}
print(f'{os.path.basename(export)} at T = {h.T}')
for rid, th in held.items():
    d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
    tv = {'nbaryon': d['nbaryon'].astype(float), 'strange': strange_frac(d['particles'], d['mask']).astype(float)}
    n = len(d['mask']); rep = np.random.default_rng(1000 + rid).permutation(n)[n//2:]
    w = h.weights(A, np.array(th)); w = w/w.sum()
    row = {}
    for o in ('nbaryon', 'strange'):
        x = tv[o][rep]; mf = x.mean(); ef = x.std(ddof=1)/np.sqrt(len(x))
        mr = float(w @ robs[o]); er = float(np.sqrt(np.sum(w**2*(robs[o] - mr)**2)))
        row[o] = dict(fresh=mf, reweighted=mr, rel_diff_pct=100*(mr - mf)/mf, resolution_pct=100*np.hypot(ef, er)/mf,
                      z=(mr - mf)/np.hypot(ef, er))
    out[rid] = dict(theta=list(th), **row)
    print(f'  run {rid} fraction {th[-1]:.3f}: baryons {row["nbaryon"]["rel_diff_pct"]:+.2f}% (resolution {row["nbaryon"]["resolution_pct"]:.2f}%, z {row["nbaryon"]["z"]:+.1f}), '
          f'strange {row["strange"]["rel_diff_pct"]:+.2f}% (resolution {row["strange"]["resolution_pct"]:.2f}%)')
json.dump(out, open(f'output/rate_check_mix_{os.path.basename(export).replace("_cond.npz", "")}.json', 'w'), indent=1)
