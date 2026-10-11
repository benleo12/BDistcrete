#!/usr/bin/env python3
"""Diagnostics of the wifi weights against the published network, out of sample.

On the held-out runs (comparison halves, as in the closure), the symmetrized MLC loss of the log
ratio is evaluated for the published weights (w_ij = delta_ij/(M T)), the wifi fit, and a
diagonal-only wifi fit (M + 1 weights), with the numerator the held-out events at their own theta
and the denominator a subsample of the reference at the same theta. A lower loss out of sample
means a better density ratio in the full phase space, whatever the binned closure width on six
observables says. The closure widths of all three are given alongside.

    python wifi_diag.py output/wifi_MIXGEO_basis.npz output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '.')
from r2_ladder import make_bins, pulls, width_of

bas, export, refpath = sys.argv[1:4]
B = np.load(bas); c = np.load(export, mmap_mode='r'); F = np.load(bas.replace('_basis.npz', '_fit.npz'))
ENS, K = int(B['ens']), int(B['K']); M2 = ENS*ENS; OBS = [str(o) for o in B['obs']]; kind = str(B['kind']); T = float(c['temperature'])
NDEN = int(os.environ.get('NDEN', '20000')); t0 = time.time()
design = lambda f: np.column_stack([np.ones(len(f)), np.asarray(f, np.float64)])
Xn, Xd = design(B['num_f']), design(B['den_f'])


def loss(w, Xn, Xd):
    zn, zd = Xn @ w, Xd @ w
    return float(np.mean(-zn + np.exp(-zn) - 1) + np.mean(zd + np.exp(zd) - 1))


def fit(Xn, Xd, mask):
    """Newton on the loss over the weights selected by mask (others zero)."""
    w = np.zeros(M2 + 1); w[1:] = np.eye(ENS).ravel()/(ENS*T); w = w*mask
    idx = np.where(mask)[0]
    for it in range(60):
        zn, zd = Xn @ w, Xd @ w; en, ed = np.exp(-zn), np.exp(zd)
        g = ((Xn*(-(1 + en))[:, None]).sum(0)/len(Xn) + (Xd*(1 + ed)[:, None]).sum(0)/len(Xd))[idx]
        H = ((Xn[:, idx]*en[:, None]).T @ Xn[:, idx]/len(Xn) + (Xd[:, idx]*ed[:, None]).T @ Xd[:, idx]/len(Xd))
        step = np.linalg.solve(H + 1e-12*np.eye(len(idx)), g); L0 = loss(w, Xn, Xd); s = 1.0
        while s > 1e-6:
            wt = w.copy(); wt[idx] -= s*step
            if loss(wt, Xn, Xd) <= L0 - 1e-4*s*(g @ step):
                break
            s *= 0.5
        w = wt
        if np.linalg.norm(g) < 1e-9:
            break
    return w


w_pub = np.zeros(M2 + 1); w_pub[1:] = np.eye(ENS).ravel()/(ENS*T)
w_wifi = F['w']
diag_mask = np.zeros(M2 + 1, bool); diag_mask[0] = True; diag_mask[1:] = np.eye(ENS, dtype=bool).ravel()
w_diag = fit(Xn, Xd, diag_mask)
print(f'fit-set loss: published {loss(w_pub, Xn, Xd):.5f}  wifi {loss(w_wifi, Xn, Xd):.5f}  diagonal-only {loss(w_diag, Xn, Xd):.5f}', flush=True)
print('diagonal-only weights:', np.round(w_diag[diag_mask], 4))

# ---- out of sample: the held-out runs ---------------------------------------------------------
AE = np.asarray(c['AE']); norm = np.asarray(c['norm'], np.float64)
act = {'relu': lambda z: np.maximum(z, 0.0), 'silu': lambda z: z/(1 + np.exp(-z))}[str(c['act'])]; Lnl = int(c['nlayers'])
def bnet(prefix, mi, x):
    for li in range(Lnl):
        x = x @ c[f'{prefix}{mi}_W{li}'].T + c[f'{prefix}{mi}_b{li}']
        if li < Lnl - 1: x = act(x)
    return x
def btilde(theta):
    x = ((np.asarray(theta, np.float64) - norm[:, 0])/norm[:, 1]).reshape(1, -1)
    if kind == 'geometric':
        ns = c['B0_W0'].shape[1]; nh = c['BH0_W0'].shape[1]; fr = float(np.clip(theta[-1], 0, 1))
        return np.stack([(1 - fr)*bnet('B', mi, x[:, :ns]).ravel() + fr*bnet('BH', mi, x[:, ns:ns + nh]).ravel() for mi in range(ENS)])
    return np.stack([bnet('B', mi, x).ravel() for mi in range(ENS)])
ref = np.load(refpath); robs = {o: np.asarray(ref[o], np.float64) for o in OBS}; Nref = AE.shape[1]
rng = np.random.default_rng(3)
# a third variant: the weights fitted on the SELECTION halves of the held-out runs, where the
# published calibration constant is fitted, and judged like everything else on the comparison halves
FREF = {}; SEL = {}
for rid, th in zip([int(r) for r in B['held_rids']], B['held_theta']):
    Bt = btilde(th); FREF[rid] = np.einsum('nik,jk->nij', AE.transpose(1, 0, 2).astype(np.float64), Bt).reshape(Nref, M2)
    n = len(B[f'held_f_{rid}']); perm = np.random.default_rng(1000 + rid).permutation(n); SEL[rid] = (perm[:n//2], perm[n//2:])
Xn_h = design(np.concatenate([B[f'held_f_{rid}'][SEL[rid][0]] for rid in FREF]))
Xd_h = design(np.concatenate([FREF[rid][np.sort(rng.choice(Nref, 4000, replace=False))] for rid in FREF]))
w_held = fit(Xn_h, Xd_h, np.ones(M2 + 1, bool))
print('held-fit weights:', np.round(w_held, 4))
W = {'published': w_pub, 'wifi': w_wifi, 'diagonal': w_diag, 'held-fit': w_held}
Lheld = {k: [] for k in W}; per = {k: {} for k in W}
for rid, th in zip([int(r) for r in B['held_rids']], B['held_theta']):
    Fref = FREF[rid]; rep = SEL[rid][1]
    Xh = design(B[f'held_f_{rid}'][rep]); Xr = design(Fref[np.sort(rng.choice(Nref, NDEN, replace=False))])
    for name, wv in W.items():
        Lheld[name].append(loss(wv, Xh, Xr))
        z = wv[0] + Fref @ wv[1:]; wr = np.exp(z - z.max()); wr /= wr.sum()
        for o in OBS:
            tv = np.asarray(B[f'held_{o}_{rid}'], np.float64)[rep]; bins = make_bins(np.r_[robs[o], tv], o)
            p, nb = pulls(robs[o], wr, tv, bins); per[name][(rid, o)] = width_of(p)
out = {}
for name in W:
    widths = {o: float(np.mean([per[name][(r, o)] for r in [int(x) for x in B['held_rids']]])) for o in OBS}
    out[name] = dict(held_loss=float(np.mean(Lheld[name])), held_loss_per_run=[float(x) for x in Lheld[name]],
                     closure=float(np.mean(list(widths.values()))), per_obs=widths, w=W[name].tolist())
    print(f'{name:10s}: held-out loss {out[name]["held_loss"]:.5f}   closure {out[name]["closure"]:.3f}   ' + ' '.join(f'{o} {v:.2f}' for o, v in widths.items()), flush=True)
json.dump(out, open(bas.replace('_basis.npz', '_diag.json'), 'w'), indent=1)
print(f'wrote {bas.replace("_basis.npz", "_diag.json")} [{time.time()-t0:.0f} s]')
