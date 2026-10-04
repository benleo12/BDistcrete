#!/usr/bin/env python3
"""App. E: why the diagonal term of Sigma_c is not propagated as an uncertainty. At the fitted point
of a direct-profile fit, the targets are drawn NDRAW times with independent Gaussian noise of the
diagonal term's size (0.2 percent of each moment) and the tilt is re-solved for each draw. Reported:
the relative entropy of the tilted sample from the unanchored one, mean over the draws, against that
of the anchor itself, and the spread of the window fraction, which no constraint fixes.

    python floor_noise_direct.py output/profile_MIX17ext_central.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, tilt_on_t, fitted_point

src = sys.argv[1]; fit = json.load(open(src))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float64'),
          grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
a, a0, u, _ = fitted_point(src)
c, _, _, S, _, _ = TargetInterp(M.G, 'central')(a, a0)
cT = torch.tensor(c); Sig = torch.tensor(S) + torch.diag((M.floor_rel*torch.abs(cT))**2)
win = torch.zeros(M.N, dtype=torch.float64).index_fill(0, M.IW, 1.0)
with torch.no_grad():
    lg = M.logit(M.theta(torch.tensor(u))); lw0 = lg - torch.logsumexp(lg, 0)
    def solve(ct):
        lw, _, _, _ = tilt_on_t(M, lw0, ct, Sig, torch.zeros(M.MW.shape[1]))
        w = torch.exp(lw); return float((w*(lw - lw0)).sum()), float(w.double() @ win)
    kl0, pw0 = solve(cT)
    rng = np.random.default_rng(11); kls, pws = [], []
    for d in range(int(os.environ.get('NDRAW', '100'))):
        k, p = solve(cT*(1 + M.floor_rel*torch.from_numpy(rng.standard_normal(len(c)))))
        kls.append(k); pws.append(p)
kls, pws = np.array(kls), np.array(pws)
print(f'anchor at the fitted point: KL {kl0:.5f} nats, window fraction {pw0:.4f}')
print(f'draws with independent noise of {100*M.floor_rel:.1f} percent: KL mean {kls.mean():.3f} nats ({kls.mean()/kl0:.0f} times the anchor), '
      f'window fraction spread {100*pws.std(ddof=1)/pw0:.1f} percent, largest change {100*np.abs(pws/pw0 - 1).max():.1f} percent')
json.dump(dict(kl_anchor=kl0, pwin_anchor=pw0, kl_mean=float(kls.mean()), kl_ratio=float(kls.mean()/kl0),
               pwin_spread_rel=float(pws.std(ddof=1)/pw0), pwin_maxchange_rel=float(np.abs(pws/pw0 - 1).max())),
          open(src.replace('.json', '_floornoise.json'), 'w'), indent=1)
