#!/usr/bin/env python3
"""Mean-rate check of the ladder (Sec. 5.2): reweighted against fresh mean of the baryon count and
the strange-hadron fraction at each held-out point, |z| in units of the combined statistical error,
averaged over the held-out points, with and without the held-out runs that duplicate training runs."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head
import r2_ladder as R
from make_stage_table import strange_frac
tag = sys.argv[1]; HALF = os.environ.get('HALF', 'report')
cfg = R.STAGES[tag]; DATA = cfg['data']
h = Head(f'output/models/{tag}_cond.npz')
A = h.pack(np.asarray(np.load(f'output/models/{tag}_cond.npz', mmap_mode='r')['AE']))
ref = np.load(f'output/models/{tag}_ref.npz', mmap_mode='r')
OBS = [o for o in ('nbaryon', 'strange') if o in cfg['flav_obs']]
robs = {o: np.asarray(ref[o], np.float64) for o in OBS}
uni = np.full(len(robs[OBS[0]]), 1.0/len(robs[OBS[0]]))
def stats(x, w):
    m = float(w @ x); v = float(np.sum(w**2*(x - m)**2)); return m, v
rows = {}
for rid, th in cfg['held'].items():
    d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
    tv = {'nbaryon': d['nbaryon'].astype(float), 'strange': strange_frac(d['particles'], d['mask']).astype(float)}
    n = len(d['mask']); perm = np.random.default_rng(1000 + rid).permutation(n)
    sel = perm[n//2:] if HALF == 'report' else np.arange(n)
    w = h.weights(A, np.array(th))
    for o in OBS:
        t = tv[o][sel]; mt = t.mean(); vt = t.var()/len(t)
        mw, vw = stats(robs[o], w); mu, vu = stats(robs[o], uni)
        rows[rid, o] = dict(z=(mw - mt)/np.sqrt(vw + vt), z_uni=(mu - mt)/np.sqrt(vu + vt), rel=100*(mw/mt - 1))
for drop in ([], [{'B': 6704, 'C': 6902}[tag]]):
    keep = [r for r in cfg['held'] if r not in drop]
    for o in OBS:
        z = [abs(rows[r, o]['z']) for r in keep]; zu = [abs(rows[r, o]['z_uni']) for r in keep]; rel = [abs(rows[r, o]['rel']) for r in keep]
        print(f'{tag} {o:8s} drop {drop}: mean |z| uniform {np.mean(zu):.1f} -> reweighted {np.mean(z):.2f}; mean |rel| {np.mean(rel):.2f}%, max |rel| {np.max(rel):.2f}%')
