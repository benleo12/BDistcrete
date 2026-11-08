#!/usr/bin/env python3
"""Checks behind the bands of Sec. 5, on the Model, grid, central point and Latin hypercube of
three_bands.py (same SEED and NODES):
  stat      the statistical error of every prediction from the finite reference sample, at the central
            point, for the generators alone and with the calculation imposed. With weights summing to one,
            a mean has var = sum_i w_i^2 (O_i - <O>)^2, the window mean the same over the window divided by
            P_win^2, and a bin of the distribution normalized over the bins above 0.05 the same with O_i the
            indicator of the bin among the events in those bins, divided by the bin width.
  xi part   the generator band with only the mixing fraction varied (the sixteen generator parameters at
            the central point) and with the mixing fraction fixed at the central point (the sixteen varied)
  largest weight relative to the mean weight, at the central point and at xi = 0 and 1
MIX=linear (the default) does the same for the linear mixture of three_bands.py (same MIX, BOX, NODES and SEED;
BOX is full or tune): each generator's weights at xi = 0 and 1, normalized and tilted on their own, the predictions
mixed with (1 - xi, xi), the distribution normalized to the total rate (a bin is then the fraction of all events in
it over its width), the statistical error from the per-event terms of both generators together, with the
calculation imposed taken as three_bands.py takes it (directlib.stat_parts: the multipliers solved again on every
resampled sample, so the projection on the constraint functions is removed), the largest weight that of the mixture
(1 - xi) w_S + xi w_H; with xi alone varied the prediction is exactly (1 - xi) y_S + xi y_H at the central
parameters. Writes output/band_checks_<tag>.json (with _lin when linear). NODE_LIMIT (tests) evaluates only the
first NODE_LIMIT points of the hypercube.

    CENTRAL=... python band_checks.py MIX <export> <ref> output/thrust_targets_nlo_point.npz
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, at_fraction, lin_box, stat_parts

tag, export, ref, grid = sys.argv[1:5]
tag = os.environ.get('TAG', tag)
MIX = os.environ.get('MIX', 'linear'); assert MIX in ('linear', 'geometric'), MIX
LINEAR = MIX == 'linear'
NODES = int(os.environ.get('NODES', '2000' if LINEAR else '200')); SEED = int(os.environ.get('SEED', '11'))
NODE_LIMIT = int(os.environ.get('NODE_LIMIT', '0'))
BOX = os.environ.get('BOX', 'full'); TUNE = float(os.environ.get('TUNE_SCALE', str(1/3)))
assert not LINEAR or BOX in ('full', 'tune'), f'BOX={BOX}: band_checks.py takes full or tune, as three_bands.py'
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('OUT', f'output/band_checks_{tag}' + ('_lin' if LINEAR else ''))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy()
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy(); wid = x['wid'].numpy()
central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64); assert len(central) == NT
per_event = {'thrust': M.thr, 'nch': M.nch, **{k: M.extra[k] for k in ('B_total', 'rho_heavy', 'nbaryon')}}
inwin = np.zeros(M.N, bool); inwin[M.IW.numpy()] = True
GENS = (('S', 0.0), ('H', 1.0))                                    # the generators and their xi in the head


def weights(theta, anchored=True):
    with torch.no_grad():
        lw, _, _, _, _ = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, anchored=anchored)
        w = torch.exp(lw).numpy()
    return w/w.sum()


def predict(w):
    o = {k: float(w @ v) for k, v in per_event.items()}
    o['pwin'] = float(w[inwin].sum()); o['tau_win'] = float((w[inwin] @ M.thr[inwin])/o['pwin'])
    cnt = np.zeros(x['nb']); np.add.at(cnt, x['idx'].numpy(), w[x['ins'].numpy()])
    return o, cnt[use]/wid[use] if LINEAR else cnt[use]/(wid[use]*cnt[use].sum())


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


# ---- the linear mixture ------------------------------------------------------------------------------
def gen_weights(theta, anchored=True):
    """{g: weights of generator g, theta at xi = 0 or 1} for the generators with a share at theta."""
    xi = float(theta[-1])
    return {g: weights(at_fraction(theta, xg), anchored) for g, xg in GENS if (1 - xi if g == 'S' else xi) != 0}


def predict_lin(theta, anchored=True, ws=None):
    """(1 - xi) and xi times each generator's (means, histogram); ws reuses the weights of gen_weights."""
    xi = float(theta[-1]); ws = gen_weights(theta, anchored) if ws is None else ws; a = dict(S=1 - xi, H=xi)
    ps = {g: predict(w) for g, w in ws.items()}
    return {k: sum(a[g]*p[0][k] for g, p in ps.items()) for k in OBK}, sum(a[g]*p[1] for g, p in ps.items())


def stat_errors_lin(theta, anchored=True):
    """stat_errors of the mixture, as three_bands.py: the same reference events carry both generators, so the error
    of (1 - xi) <O>_S + xi <O>_H is sqrt(sum_i ((1 - xi) u_S,i + xi u_H,i)^2), with u_g = w_g (O - <O>_g) for the
    generators alone and, with the calculation imposed, the same minus its projection on the constraint functions
    (directlib.stat_parts); a bin the same with O_i its indicator over its width. Also returns the weights of the
    mixture."""
    xi = float(theta[-1]); a = dict(S=1 - xi, H=xi)
    parts = {g: stat_parts(M, at_fraction(theta, xg), anchored) for g, xg in GENS if a[g] != 0}
    ws = {g: p[0] for g, p in parts.items()}
    o, h = predict_lin(theta, anchored, ws)
    comb = lambda make: float(np.sqrt(np.sum(sum(a[g]*u(make(w)) for g, (w, u) in parts.items())**2)))
    err = {k: comb(lambda w, v=v: v - w @ v) for k, v in per_event.items()}
    err['pwin'] = comb(lambda w: inwin - w[inwin].sum())
    err['tau_win'] = comb(lambda w: np.where(inwin, M.thr - (w[inwin] @ M.thr[inwin])/w[inwin].sum(), 0.0)/w[inwin].sum())
    binof = np.full(M.N, -1); binof[x['ins'].numpy()] = x['idx'].numpy()
    hb = np.array([comb(lambda w, ind=(binof == bb).astype(float): ind - w @ ind)/wid[bb] for bb in use])
    return o, h, err, hb, sum(a[g]*w for g, w in ws.items())


out = dict(tag=tag, central=central.tolist(), nodes=NODES, seed=SEED, n_ref=int(M.N))
if LINEAR:
    out.update(mix=MIX, box=BOX, normalization='total rate',
               stat_imposed_rule='multipliers solved again on every resampled sample (directlib.stat_parts, as three_bands.py)')
for name, anchored in (('generators_alone', False), ('imposed', True)):
    if LINEAR:
        o, h, e, he, w = stat_errors_lin(central, anchored)
    else:
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
if LINEAR and BOX != 'full':                                        # the nodes of three_bands.py
    blo, bhi = lin_box(cen, hw, central, BOX, TUNE, M.NS); nodes = (blo + bhi)/2 + (bhi - blo)/2*U
else:
    nodes = cen + hw*U
if NODE_LIMIT:
    nodes = nodes[:NODE_LIMIT]                                     # tests: the first points of the same hypercube
    out['nodes_evaluated'] = len(nodes)
for part, mk in (('xi_only', lambda th: np.r_[central[:-1], th[-1]]), ('xi_fixed', lambda th: np.r_[th[:-1], central[-1]])):
    res = {}
    for name, anchored in (('before', False), ('after', True)):
        rows, H = [], []
        if LINEAR and part == 'xi_only':                            # (1 - xi) y_S + xi y_H with y_g at the central point
            ys = {g: predict(w) for g, w in gen_weights(np.r_[central[:-1], 0.5], anchored).items()}
        for th in nodes:
            if LINEAR and part == 'xi_only':
                xi = th[-1]
                o = {k: (1 - xi)*ys['S'][0][k] + xi*ys['H'][0][k] for k in OBK}; h = (1 - xi)*ys['S'][1] + xi*ys['H'][1]
            else:
                o, h = predict_lin(mk(th), anchored) if LINEAR else predict(weights(mk(th), anchored))
            rows.append([o[k] for k in OBK]); H.append(h)
        rows, H = np.array(rows), np.array(H)
        res[name] = dict(sd={k: float(rows[:, j].std(ddof=1)) for j, k in enumerate(OBK)}, hist_sd=H.std(0, ddof=1).tolist(),
                         mean={k: float(rows[:, j].mean()) for j, k in enumerate(OBK)}, hist_mean=H.mean(0).tolist())
    out[part] = res
    print(f'[{tag}] {part}: <1-T> sd before {res["before"]["sd"]["thrust"]:.5f} after {res["after"]["sd"]["thrust"]:.5f} [{time.time()-t0:.0f} s]', flush=True)
json.dump(out, open(OUT + '.json', 'w'), indent=1)
print('wrote', OUT + '.json')
