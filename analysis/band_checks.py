#!/usr/bin/env python3
"""Checks behind the bands of Sec. 4 (paper r8), on the Model, grid, central point and Latin hypercube of
three_bands.py (same SEED and NODES):
  stat      the statistical error of every prediction from the finite reference sample, at the central
            point, for the generators alone and with the calculation imposed. With weights summing to one,
            a mean has var = sum_i w_i^2 (O_i - <O>)^2, the window mean the same over the window divided by
            P_win^2, and a bin of the distribution normalized over the bins above 0.05 the same with O_i the
            indicator of the bin among the events in those bins, divided by the bin width.
  xi part   the generator band with only the mixing fraction varied (the sixteen generator parameters at
            the central point) and with the mixing fraction fixed at the central point (the sixteen varied)
  largest weight relative to the mean weight, at the central point and at xi = 0 and 1
Writes output/band_checks_<tag>.json.

    CENTRAL=... python band_checks.py MIX <export> <ref> output/thrust_targets_nlo_point.npz
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model

tag, export, ref, grid = sys.argv[1:5]
tag = os.environ.get('TAG', tag)
NODES = int(os.environ.get('NODES', '200')); SEED = int(os.environ.get('SEED', '11'))
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('OUT', f'output/band_checks_{tag}')
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy()
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy(); wid = x['wid'].numpy()
central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64); assert len(central) == NT
per_event = {'thrust': M.thr, 'nch': M.nch, **{k: M.extra[k] for k in ('B_total', 'rho_heavy', 'nbaryon')}}
inwin = np.zeros(M.N, bool); inwin[M.IW.numpy()] = True


def weights(theta, anchored=True):
    with torch.no_grad():
        lw, _, _, _, _ = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, anchored=anchored)
        w = torch.exp(lw).numpy()
    return w/w.sum()


def predict(w):
    o = {k: float(w @ v) for k, v in per_event.items()}
    o['pwin'] = float(w[inwin].sum()); o['tau_win'] = float((w[inwin] @ M.thr[inwin])/o['pwin'])
    cnt = np.zeros(x['nb']); np.add.at(cnt, x['idx'].numpy(), w[x['ins'].numpy()])
    return o, cnt[use]/(wid[use]*cnt[use].sum())


def stat_errors(w):
    o, h = predict(w); w2 = w**2
    err = {k: float(np.sqrt(w2 @ (v - o[k])**2)) for k, v in per_event.items()}
    err['pwin'] = float(np.sqrt(w2 @ (inwin - o['pwin'])**2))
    err['tau_win'] = float(np.sqrt(w2[inwin] @ (M.thr[inwin] - o['tau_win'])**2)/o['pwin'])
    ins = x['ins'].numpy(); idx = x['idx'].numpy(); ub = np.asarray(use)          # use is an index array of bins
    sel = np.isin(idx, ub); ev = ins[sel]; b = idx[sel]
    W = w[ev].sum(); hb = []
    for j, bb in enumerate(ub):
        ind = (b == bb).astype(float); p = (w[ev] @ ind)/W
        hb.append(float(np.sqrt(w2[ev] @ (ind - p)**2)/W/wid[bb]))
    return o, h, err, np.array(hb)


out = dict(tag=tag, central=central.tolist(), nodes=NODES, seed=SEED, n_ref=int(M.N))
for name, anchored in (('generators_alone', False), ('imposed', True)):
    w = weights(central, anchored); o, h, e, he = stat_errors(w)
    out[f'stat_{name}'] = dict(means=o, mean_err=e, hist=h.tolist(), hist_err=he.tolist(), hist_rel_err=(he/h).tolist(),
                               max_weight_over_mean=float(w.max()*len(w)))
    print(f'[{tag}] stat {name}: <1-T> {o["thrust"]:.5f} +- {e["thrust"]:.1e}, bins rel err {100*(he/h).min():.2f}-{100*(he/h).max():.2f}% '
          f'[{time.time()-t0:.0f} s]', flush=True)
for xi in (0.0, 1.0):
    th = central.copy(); th[-1] = xi
    out[f'max_weight_over_mean_xi{int(xi)}'] = {nm: float(weights(th, a).max()*M.N) for nm, a in (('generators_alone', False), ('imposed', True))}
rng = np.random.default_rng(SEED)
U = (np.argsort(rng.random((NODES, NT)), axis=0) + rng.random((NODES, NT)))/NODES*2 - 1
nodes = cen + hw*U
for part, mk in (('xi_only', lambda th: np.r_[central[:-1], th[-1]]), ('xi_fixed', lambda th: np.r_[th[:-1], central[-1]])):
    res = {}
    for name, anchored in (('before', False), ('after', True)):
        rows, H = [], []
        for th in nodes:
            o, h = predict(weights(mk(th), anchored)); rows.append([o[k] for k in OBK]); H.append(h)
        rows, H = np.array(rows), np.array(H)
        res[name] = dict(sd={k: float(rows[:, j].std(ddof=1)) for j, k in enumerate(OBK)}, hist_sd=H.std(0, ddof=1).tolist(),
                         mean={k: float(rows[:, j].mean()) for j, k in enumerate(OBK)}, hist_mean=H.mean(0).tolist())
    out[part] = res
    print(f'[{tag}] {part}: <1-T> sd before {res["before"]["sd"]["thrust"]:.5f} after {res["after"]["sd"]["thrust"]:.5f} [{time.time()-t0:.0f} s]', flush=True)
json.dump(out, open(OUT + '.json', 'w'), indent=1)
print('wrote', OUT + '.json')
