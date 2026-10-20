#!/usr/bin/env python3
"""The signed displacement of every theory variation of Sec. 4, bin by bin and for the means, at the
central point of three_bands.py (the default tunes, xi = 1/2), with the calculation imposed: the four
scale variations, the change of scheme and the three variations of the dispersive model. three_bands.py
combines them into a symmetric band and keeps only the combination. Also the generator-only and
imposed predictions at xi = 0 and xi = 1 (each generator alone at its default tune). Same Model, grid
and test data as three_bands.py. Writes output/theory_variations_<tag>.json.

    python theory_variations.py MIX output/models/MIXGEOV3S9_cond.npz output/models/MIXGEOV3S9_ref_v2_slim.npz \
        output/thrust_targets_nlo_point.npz
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model

tag, export, ref, grid = sys.argv[1:5]
TEST = os.environ.get('TEST_DATA', 'delphi')
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
G = M.G; NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy()
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy(); wid = x['wid'].numpy()


def predict(theta, which='central', anchored=True):
    with torch.no_grad():
        lw, lw0, lam, Pw, its = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, which=which, anchored=anchored)
        w = torch.exp(lw); o = M.observables(w); o = {k: float(o[k]) for k in OBK if k in o}
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
        h = cnt[use]/(wid[use]*cnt[use].sum())
        neff = float(1.0/(w**2).sum())
    return o, h, neff


central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64)
o_c, h_c, ne_c = predict(central); o_c0, h_c0, ne_c0 = predict(central, anchored=False)
out = dict(tag=tag, test=TEST, central=central.tolist(), lo=x['lo'][use].tolist(), hi=x['hi'][use].tolist(),
           data=(x['y'].numpy()[use]).tolist(), err=(x['e'].numpy()[use]).tolist(),
           h_imposed=h_c.tolist(), h_gen_only=h_c0.tolist(), o_imposed=o_c, o_gen_only=o_c0,
           neff_imposed=ne_c, neff_gen_only=ne_c0, n_ref=int(M.N), variations={})
for kind, labels in (('scale', [str(l) for l in G['scale_labels']]), ('np', [str(l) for l in G['np_labels']])):
    for v, lab in enumerate(labels):
        ov, hv, _ = predict(central, which=f'{kind}:{v}')
        out['variations'][f'{kind}:{lab}'] = dict(dh=(hv - h_c).tolist(), do={k: ov[k] - o_c[k] for k in o_c})
for xi in (0.0, 1.0):
    th = central.copy(); th[-1] = xi
    o1, h1, n1 = predict(th); o0, h0, n0 = predict(th, anchored=False)
    out[f'xi{int(xi)}'] = dict(h_imposed=h1.tolist(), h_gen_only=h0.tolist(), o_imposed=o1, o_gen_only=o0,
                               neff_imposed=n1, neff_gen_only=n0)
json.dump(out, open(f'output/theory_variations_{tag}.json', 'w'), indent=1)
print('wrote', f'output/theory_variations_{tag}.json', 'neff (imposed, gen only) at xi=1/2:', ne_c, ne_c0)
