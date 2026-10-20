#!/usr/bin/env python3
"""The Gaussian toy of the paper with the paper's network and training (r2_ladder.CondPFN and the
procedure of r2_ladder.run_stage): four members on bootstrap resamples, Adam with a cosine step size
from 1e-3 to 1e-5, the checkpoint whose closure width on the selection halves is closest to one, the
calibration constant kappa fitted on the same halves by the grid of r2_ladder, and the closure test of
r2_ladder (make_bins, pulls, width_of) on the comparison halves.

An event is ten particles with one coordinate each, x_i ~ N(0, sigma), and the parameter is
s = log sigma. Against a reference at s = 0 the exact log ratio is
    f_exact = -10 s + (1 - e^{-2s})/2 * sum_i x_i^2,
a sum of two products, K = 2 with a(Phi) = (1, sum_i x_i^2). The reference has 1e5 events at s = 0,
the training runs 2e4 events at 25 widths spread over [-0.25, 0.25], and six held-out widths have
1e5 events each, halved into a selection and a comparison half of 5e4. The networks train for
2.8e4 steps, so that each reference event enters the training about 290 times, as in the
seventeen-parameter mixture. The observables are the coordinates of all particles and the mean
square of the coordinates of an event.

Besides the closure test it records the straight-line fit of the learned log ratio against the exact
one at s = 0.21, the same for the difference between s = -0.21 and s = 0.21, which is linear in
sum_i x_i^2, and the coefficients of 1 and sum_i x_i^2 in the learned log ratio over a grid of widths.

    TOY_K=48 TOY_SEED=0 python toy_final.py      # output/toy_final_K48_s0.json and .npz
"""
import os, sys, json, time
os.environ.setdefault('LADDER_ACT', 'silu'); os.environ.setdefault('LADDER_EMB_SCALE', 'auto')
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
from r2_ladder import CondPFN, make_bins, pulls, width_of, emb_all

K = int(os.environ.get('TOY_K', '48')); SEED = int(os.environ.get('TOY_SEED', '0'))
ENS, NP = 4, 10
S_TRAIN = np.linspace(-0.25, 0.25, 25); N_TRAIN = 20000
N_REF = 100000
S_HELD = [-0.21, -0.13, -0.05, 0.05, 0.13, 0.21]; N_HELD = 100000
STEPS = int(os.environ.get('TOY_STEPS', '28000')); CK_EVERY = 500; LR = 1e-3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.environ.get('TOY_OUT', f'output/toy_final_K{K}_s{SEED}')
t0 = time.time()


def sample(s, n, rng):
    return rng.normal(0.0, np.exp(s), (n, NP)).astype(np.float32)


def exact(x, s):
    return -NP*s + 0.5*(1.0 - np.exp(-2.0*s))*(x.astype(np.float64)**2).sum(1)


# the samples are drawn once with a fixed seed, shared by every training, so the trainings of the
# K scan and the three seeds differ only in their networks, bootstrap resamples and batches
rng = np.random.default_rng(12345)
xref = sample(0.0, N_REF, rng)
xtr = [sample(s, N_TRAIN, rng) for s in S_TRAIN]
xheld = [sample(s, N_HELD, rng) for s in S_HELD]
allx = np.concatenate(xtr); MU, SD = float(allx.mean()), float(allx.std())   # inputs shifted and scaled over all training runs


def feat(x):
    return torch.tensor(((x - MU)/SD)[..., None], dtype=torch.float32)       # (events, 10, 1)


tn = lambda s: s/0.25                          # the parameter shifted and scaled to [-1, 1]
ref_f = feat(xref).to(DEV); ref_m = torch.ones(N_REF, NP, device=DEV)
tr_f = [feat(x).to(DEV) for x in xtr]
batch_m = torch.ones(2048, NP, device=DEV)

OBS = ['coordinate', 'mean_square']


def obs(x):
    return {'coordinate': x.ravel().astype(np.float64), 'mean_square': (x.astype(np.float64)**2).mean(1)}


robs = obs(xref)
split = [np.random.default_rng(1000 + j).permutation(N_HELD) for j in range(len(S_HELD))]
H = {'sel': [obs(xheld[j][split[j][:N_HELD//2]]) for j in range(len(S_HELD))],
     'rep': [obs(xheld[j][split[j][N_HELD//2:]]) for j in range(len(S_HELD))]}
BINS = {(j, o): make_bins(np.r_[robs[o], H['rep'][j][o]], o) for j in range(len(S_HELD)) for o in OBS}


def normw(f, kappa=1.0):
    w = np.exp((f - f.max())/kappa); return w/w.sum()


def closure(F, kappa, which, uniform=False):
    """mean closure width over the held-out widths and the observables, and its parts"""
    per = {}
    for j, s in enumerate(S_HELD):
        w = np.full(N_REF, 1.0/N_REF) if uniform else normw(F[j], kappa); po = {}
        for o in OBS:
            wo = np.repeat(w, NP) if o == 'coordinate' else w      # every particle carries its event's weight
            p, _ = pulls(robs[o], wo, H[which][j][o], BINS[(j, o)]); po[o] = width_of(p)
        per[f'{s:+.2f}'] = po
    return float(np.mean([np.mean(list(v.values())) for v in per.values()])), per


# ---- the members ---------------------------------------------------------------------------------
models, opts, schs, gens, pools = [], [], [], [], []
for mi in range(ENS):
    sd = 100*SEED + mi
    torch.manual_seed(sd); m = CondPFN(1, 1, K=K).to(DEV)
    m.autoscale(ref_f[:4096], ref_m[:4096])
    models.append(m)
    opts.append(torch.optim.Adam(m.parameters(), LR))
    schs.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[-1], STEPS, eta_min=LR/100))
    gens.append(torch.Generator().manual_seed(sd))
    g = np.random.default_rng(80000 + 100000*mi + 7*SEED)
    pools.append([np.sort(g.choice(N_TRAIN, N_TRAIN, replace=True)) for _ in S_TRAIN])   # bootstrap per run
bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)


def mean_logits(svals):
    """the mean of members of the learned log ratio on the reference at each width in svals"""
    with torch.no_grad():
        Es = [emb_all(m.eval(), ref_f, ref_m) for m in models]
        out = [np.mean([m.f_from(E, torch.full((N_REF, 1), tn(s), device=DEV)).cpu().numpy()
                        for m, E in zip(models, Es)], 0) for s in svals]
    for m in models:
        m.train()
    return out


best, traj, step = None, [], 0
for ck in range(CK_EVERY, STEPS + 1, CK_EVERY):
    for mi, m in enumerate(models):
        opt, sch, g = opts[mi], schs[mi], gens[mi]
        for _ in range(ck - step):
            i = int(torch.randint(len(S_TRAIN), (1,), generator=g))
            ir = torch.randint(N_REF, (1024,), generator=g).to(DEV)
            pool = pools[mi][i]; it = torch.from_numpy(pool[torch.randint(len(pool), (1024,), generator=g).numpy()]).to(DEV)
            fb = torch.cat([ref_f[ir], tr_f[i][it]])
            th = torch.full((2048, 1), tn(S_TRAIN[i]), device=DEV)
            loss = bce(m.f_from(m.emb(fb, batch_m), th), lab)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    step = ck
    ms, _ = closure(mean_logits(S_HELD), 1.0, 'sel'); traj.append((ck, ms))
    if np.isfinite(ms) and (best is None or abs(ms - 1.0) < best[0]):
        best = (abs(ms - 1.0), ck, ms, [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models])
    if ck % 5000 == 0:
        print(f'[toy K={K} s{SEED}] step {ck}: selection width {ms:.3f} [{time.time()-t0:.0f} s]', flush=True)
for m, st in zip(models, best[3]):
    m.load_state_dict(st)
print(f'[toy K={K} s{SEED}] selected step {best[1]}, selection width {best[2]:.3f}', flush=True)

# ---- calibration on the selection halves, closure on the comparison halves ------------------------
F = mean_logits(S_HELD)
grid = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)] + [1.75, 2.0, 2.5, 3.0]))
kappa = min(grid, key=lambda k: abs(closure(F, k, 'sel')[0] - 1.0))
width, per = closure(F, kappa, 'rep'); width_raw, _ = closure(F, 1.0, 'rep')
uni, per_uni = closure(F, 1.0, 'rep', uniform=True)

# ---- the learned log ratio against the exact one ---------------------------------------------------
Q = (xref.astype(np.float64)**2).sum(1)
j5 = S_HELD.index(0.21); j0 = S_HELD.index(-0.21)
fl, fe = F[j5], exact(xref, 0.21)
slope, icpt = np.polyfit(fe, fl, 1); corr = float(np.corrcoef(fe, fl)[0, 1])
dl, de = F[j5] - F[j0], exact(xref, 0.21) - exact(xref, -0.21)
dslope, _ = np.polyfit(de, dl, 1); dcorr = float(np.corrcoef(de, dl)[0, 1])
sg = np.linspace(-0.25, 0.25, 21); Fg = mean_logits(sg); A = np.vstack([np.ones_like(Q), Q]).T
coef = np.array([np.linalg.lstsq(A, f, rcond=None)[0] for f in Fg])          # (21, 2): constant, coefficient of Q
coef_exact = np.vstack([-NP*sg, 0.5*(1.0 - np.exp(-2.0*sg))]).T
res = dict(K=K, seed=SEED, steps=STEPS, best_step=best[1], selection_width=best[2], trajectory=traj, kappa=kappa,
           width=width, width_raw=width_raw, per_point=per, uniform=uni, uniform_per_point=per_uni,
           slope=float(slope), corr=corr, dslope=float(dslope), dcorr=dcorr,
           coef_rms_dev_const=float(np.sqrt(np.mean((coef[:, 0] - coef_exact[:, 0])**2))),
           coef_rms_dev_q=float(np.sqrt(np.mean((coef[:, 1] - coef_exact[:, 1])**2))),
           coef_q_ratio_at_021=float(coef[-1, 1]/coef_exact[-1, 1]) if coef_exact[-1, 1] else None,
           n_ref=N_REF, n_train=N_TRAIN, n_runs=len(S_TRAIN), n_held=N_HELD//2, held=S_HELD, seconds=time.time() - t0)
pick = np.random.default_rng(1).choice(N_REF, 20000, replace=False)
np.savez(OUT + '.npz', f_exact=fe[pick], f_learned=fl[pick], df_exact=de[pick], df_learned=dl[pick],
         s_grid=sg, coef=coef, coef_exact=coef_exact, kappa=kappa)
json.dump(res, open(OUT + '.json', 'w'), indent=1)
print(f'[toy K={K} s{SEED}] kappa {kappa}, closure width {width:.3f} (raw {width_raw:.3f}, uniform {uni:.1f}), '
      f'slope {slope:.3f} corr {corr:.4f}, difference slope {dslope:.3f} corr {dcorr:.4f}, '
      f'{time.time()-t0:.0f} s -> {OUT}.json', flush=True)
