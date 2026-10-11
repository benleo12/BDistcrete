#!/usr/bin/env python3
"""The wifi weights and their covariance (Benevedes and Thaler, arXiv:2506.00113).

The log ratio is modelled as log r(Phi, theta) = w_0 + sum_ij w_ij f_ij(Phi, theta) with the M^2
basis functions f_ij = <a_i, b_j> of wifi_embed.py. The weights minimize the symmetrized maximum
likelihood classifier loss on the fit set,

    L = < -w.f + exp(-w.f) - 1 >_num + < w.f + exp(w.f) - 1 >_den ,

whose minimum over a well-specified basis is the true log ratio. The loss is convex, so Newton's
method converges. The covariance of the weights is the sandwich C = V^-1 U V^-1 with V the Hessian
of L and U the covariance of its gradient, estimated from the fit set. The denominator rows reuse
pooled events across the parameter points, so U sums each event's contributions before taking the
covariance over events.

Closure: the weights of the reference sample at every held-out point are rebuilt from the export,
e^{w.f}, and compared with the held-out run as in the paper (comparison half, closure width), for
the published network (w_ij = delta_ij/(M T)) and for the wifi fit, so the fit cannot silently
make the weights worse.

    python wifi_fit.py output/wifi_MIXGEO_basis.npz output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '.')
from r2_ladder import make_bins, pulls, width_of

bas, export, refpath = sys.argv[1], sys.argv[2], sys.argv[3]
B = np.load(bas); c = np.load(export, mmap_mode='r')
ENS, K = int(B['ens']), int(B['K']); M2 = ENS*ENS; OBS = [str(o) for o in B['obs']]
kind = str(B['kind']); T = float(c['temperature'])
OUT = os.environ.get('WIFI_FIT_OUT', bas.replace('_basis.npz', '_fit'))
t0 = time.time()


def design(f):
    return np.column_stack([np.ones(len(f)), f.astype(np.float64)])   # w_0 first


Xn, Xd = design(B['num_f']), design(B['den_f'])
Nn, Nd = len(Xn), len(Xd)
print(f'{M2 + 1} weights, numerator {Nn} rows, denominator {Nd} rows [{time.time()-t0:.0f} s]', flush=True)


def loss_grad_hess(w):
    zn, zd = Xn @ w, Xd @ w
    en, ed = np.exp(-zn), np.exp(zd)
    L = np.mean(-zn + en - 1) + np.mean(zd + ed - 1)
    gn = (Xn*(-(1 + en))[:, None]).sum(0)/Nn; gd = (Xd*(1 + ed)[:, None]).sum(0)/Nd
    H = (Xn*en[:, None]).T @ Xn/Nn + (Xd*ed[:, None]).T @ Xd/Nd
    return L, gn + gd, H


# the published network is the starting point: w_ij = delta_ij/(M T), w_0 = 0
w = np.zeros(M2 + 1); w[1:] = np.eye(ENS).ravel()/(ENS*T)
L0 = loss_grad_hess(w)[0]
for it in range(60):
    L, g, H = loss_grad_hess(w)
    step = np.linalg.solve(H + 1e-12*np.eye(len(w)), g)
    s = 1.0
    while s > 1e-6:
        wt = w - s*step
        if loss_grad_hess(wt)[0] <= L - 1e-4*s*(g @ step):
            break
        s *= 0.5
    w = wt
    if np.linalg.norm(g) < 1e-9:
        break
L, g, V = loss_grad_hess(w)
print(f'Newton: {it+1} iterations, loss {L0:.6f} (published weights) -> {L:.6f}, |grad| {np.linalg.norm(g):.1e}', flush=True)

# the sandwich: U from the per-event gradient contributions
zn, zd = Xn @ w, Xd @ w
an = Xn*(-(1 + np.exp(-zn)))[:, None]/Nn                    # numerator rows, one event each
bd = Xd*(1 + np.exp(zd))[:, None]/Nd                         # denominator rows, events reused across thetas
eid = B['den_eid']; ue, inv = np.unique(eid, return_inverse=True)
bde = np.zeros((len(ue), len(w))); np.add.at(bde, inv, bd)   # per-event sums
def cov_of_sums(S):
    return S.T @ S - np.outer(S.sum(0), S.sum(0))/len(S)
U = cov_of_sums(an) + cov_of_sums(bde)
Vi = np.linalg.inv(V); Cw = Vi @ U @ Vi
sd = np.sqrt(np.diag(Cw)); corr = Cw/np.outer(sd, sd)
print('weights (w_0 then w_ij row-major):'); print(np.round(w, 4))
print('their errors:'); print(np.round(sd, 4), flush=True)

# ---- closure on the held-out runs, published against wifi ------------------------------------
AE = np.asarray(c['AE'])                                      # (ENS, N, K)
norm = np.asarray(c['norm'], np.float64); act = {'relu': lambda z: np.maximum(z, 0.0), 'silu': lambda z: z/(1 + np.exp(-z))}[str(c['act'])]
Lnl = int(c['nlayers'])
def bnet(prefix, mi, x):
    for li in range(Lnl):
        x = x @ c[f'{prefix}{mi}_W{li}'].T + c[f'{prefix}{mi}_b{li}']
        if li < Lnl - 1:
            x = act(x)
    return x
def btilde(theta):
    x = ((np.asarray(theta, np.float64) - norm[:, 0])/norm[:, 1]).reshape(1, -1)
    if kind == 'geometric':
        ns = c['B0_W0'].shape[1]; nh = c['BH0_W0'].shape[1]; fr = float(np.clip(theta[-1], 0, 1))
        return np.stack([(1 - fr)*bnet('B', mi, x[:, :ns]).ravel() + fr*bnet('BH', mi, x[:, ns:ns + nh]).ravel() for mi in range(ENS)])
    return np.stack([bnet('B', mi, x).ravel() for mi in range(ENS)])
ref = np.load(refpath); robs = {o: np.asarray(ref[o], np.float64) for o in OBS}; Nref = len(robs[OBS[0]])
assert AE.shape[1] == Nref, (AE.shape, Nref)
w_pub = np.zeros(M2 + 1); w_pub[1:] = np.eye(ENS).ravel()/(ENS*T)
res = {'published': {}, 'wifi': {}}; per = {'published': {}, 'wifi': {}}
for rid, th in zip([int(r) for r in B['held_rids']], B['held_theta']):
    Bt = btilde(th)
    Fref = np.einsum('nik,jk->nij', AE.transpose(1, 0, 2).astype(np.float64), Bt).reshape(Nref, M2)
    n = len(B[f'held_f_{rid}']); perm = np.random.default_rng(1000 + rid).permutation(n); rep = perm[n//2:]
    for name, wv in (('published', w_pub), ('wifi', w)):
        z = wv[0] + Fref @ wv[1:]; wr = np.exp(z - z.max()); wr /= wr.sum()
        for o in OBS:
            tv = np.asarray(B[f'held_{o}_{rid}'], np.float64)[rep]
            bins = make_bins(np.r_[robs[o], tv], o); p, nb = pulls(robs[o], wr, tv, bins); per[name][(rid, o)] = width_of(p)
for name in per:
    res[name] = dict(master=float(np.mean(list(per[name].values()))),
                     per_obs={o: float(np.mean([per[name][(r, o)] for r in [int(x) for x in B['held_rids']]])) for o in OBS})
    print(f'closure {name:9s}: width {res[name]["master"]:.3f}  ' + ' '.join(f'{o} {v:.2f}' for o, v in res[name]['per_obs'].items()), flush=True)
# the learning band on the log ratio itself, averaged over the fit set: sqrt(f^T C f)
sig2 = np.einsum('ni,ij,nj->n', Xn, Cw, Xn)
print(f'learning uncertainty on log w over the fit set: median {np.sqrt(np.median(sig2)):.4f}, 95th percentile {np.sqrt(np.percentile(sig2, 95)):.4f}')
np.savez(OUT + '.npz', w=w, C=Cw, V=V, U=U, w_published=w_pub, ens=ENS, K=K, kind=kind)
json.dump(dict(export=export, basis=bas, n_weights=len(w), loss_published=float(L0), loss_wifi=float(L), newton_iterations=int(it + 1),
               w=w.tolist(), w_err=sd.tolist(), corr_max_offdiag=float(np.abs(corr - np.eye(len(w))).max()),
               closure=res, logw_sigma_median=float(np.sqrt(np.median(sig2))), logw_sigma_p95=float(np.sqrt(np.percentile(sig2, 95))),
               fit_split=int(B['fit_split'])), open(OUT + '.json', 'w'), indent=1)
print(f'wrote {OUT}.npz and {OUT}.json [{time.time()-t0:.0f} s]')
