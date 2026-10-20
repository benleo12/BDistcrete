#!/usr/bin/env python3
"""Does the learning band cover what the emulator actually gets wrong?

At every held-out point of a data set, the prediction of the paper (mean of members, no
calculation imposed) of the observable means is compared with the held-out run itself:

    z = (prediction - truth) / sqrt(stat_pred^2 + stat_truth^2 [+ learning^2])

stat_pred is the statistical error of the reweighted reference mean, stat_truth that of the
held-out run's mean, and learning the learning band at that point (the covariance of the wifi
weights propagated as in three_bands.py). If the band is an honest uncertainty on the prediction,
the rms of z over the held-out points is near one with it and above one without it.

    python wifi_coverage.py MIX output/models/<TAG>_cond.npz output/models/<TAG>_ref_v2_slim.npz \\
        output/wifi_<TAG>_fit.npz <held data dir> <theory point file>
Writes output/wifi_coverage_<TAG>.json (COVERAGE_OUT overrides it)."""
import os, sys, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model

tag, export, ref, wfit, data, grid = sys.argv[1:7]
TAG = os.path.basename(export).replace('_cond.npz', '')
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(), ae_dtype='float32', grid=grid)
W = np.load(wfit); C = W['C']; w_pub = W['w_published']
OBS = {'thrust': '1_minus_thrust', 'B_total': 'B_total', 'rho_heavy': 'rho_heavy', 'nch': None, 'nbaryon': None}
meta = json.load(open(f'{data}/meta.json'))
ref_x = {k: (M.THRT.numpy() if k == 'thrust' else M.NCHT.numpy() if k == 'nch' else M.extra[k]) for k in OBS}
rows = []
for h in meta['held']:
    rid, theta = h['rid'], np.array(h['theta'], np.float64)
    sh = pd.read_csv(f'{data}/shapes_run_{rid:04d}.csv')
    z = np.load(f'{data}/particles_full_{rid:04d}.npz')
    truth_x = {'thrust': sh['1_minus_thrust'].values, 'B_total': sh['B_total'].values, 'rho_heavy': sh['rho_heavy'].values,
               'nch': ((z['particles'][..., 5] != 0) & z['mask'].astype(bool)).sum(1).astype(np.float64),
               'nbaryon': z['nbaryon'].astype(np.float64)}
    # the prediction and its learning band, at the weights of the published network (mean of members)
    wt = torch.tensor(np.asarray(w_pub, np.float64), requires_grad=True); M.set_wifi(wt)
    lw, _, _, _, _ = M.weights(torch.as_tensor(theta), 0, 0, anchored=False)
    wv = torch.exp(lw); o = M.observables(wv)
    rec = dict(rid=rid, held_set=h.get('held_set', ''), n_held=len(sh))
    w_np = wv.detach().numpy()
    for k in OBS:
        g = torch.autograd.grad(o[k], wt, retain_graph=True)[0].numpy()
        pred = float(o[k]); learn = float(np.sqrt(max(g @ C @ g, 0)))
        stat_pred = float(np.sqrt(np.sum(w_np**2 * (ref_x[k] - pred)**2)))
        tx = truth_x[k]; truth = float(tx.mean()); stat_truth = float(tx.std(ddof=1)/np.sqrt(len(tx)))
        s0 = np.hypot(stat_pred, stat_truth)
        rec[k] = dict(pred=pred, truth=truth, learn=learn, stat_pred=stat_pred, stat_truth=stat_truth,
                      z_without=(pred - truth)/s0, z_with=(pred - truth)/np.sqrt(s0**2 + learn**2))
    M.set_wifi(None)
    rows.append(rec)
    print(f'[{TAG}] {rid}: <1-T> {rec["thrust"]["pred"]:.5f} vs {rec["thrust"]["truth"]:.5f}, '
          f'z {rec["thrust"]["z_without"]:+.2f} -> {rec["thrust"]["z_with"]:+.2f} [{time.time()-t0:.0f} s]', flush=True)
summary = {k: dict(rms_z_without=float(np.sqrt(np.mean([r[k]['z_without']**2 for r in rows]))),
                   rms_z_with=float(np.sqrt(np.mean([r[k]['z_with']**2 for r in rows]))),
                   median_learn_over_stat=float(np.median([r[k]['learn']/np.hypot(r[k]['stat_pred'], r[k]['stat_truth']) for r in rows])),
                   rms_rel_error=float(np.sqrt(np.mean([((r[k]['pred'] - r[k]['truth'])/r[k]['truth'])**2 for r in rows]))))
           for k in OBS}
for k, s in summary.items():
    print(f'[{TAG}] {k:9s} rms z {s["rms_z_without"]:.2f} without the learning band, {s["rms_z_with"]:.2f} with it; '
          f'band/stat {s["median_learn_over_stat"]:.2f}; rms relative error {100*s["rms_rel_error"]:.3f}%')
json.dump(dict(tag=TAG, data=data, wifi_fit=wfit, n_points=len(rows), summary=summary, points=rows),
          open(os.environ.get('COVERAGE_OUT', f'output/wifi_coverage_{TAG}.json'), 'w'), indent=1)
