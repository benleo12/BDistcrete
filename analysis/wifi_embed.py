#!/usr/bin/env python3
"""The basis functions of the wifi ensemble on the fit set and on the held-out runs.

With M ensemble members, each with its stored event functions a_i(Phi) and parameter functions
b_j(theta) (for a mixture head b_j(theta) = (1 - xi) b_{S,j}(theta_S) + xi b_{H,j}(theta_H)),
the wifi basis is every product f_ij(Phi, theta) = <a_i(Phi), b_j(theta)>, M^2 functions, plus
the constant. The published network is the special case w_ij = delta_ij / (M T).

This script runs where the data are. It embeds the fit-set events of every training run with the
trunk, evaluates the M^2 basis functions on them at their own run's theta (the numerator of the
wifi loss) and, for the denominator, on a subsample of the pooled fit set paired with the same
theta. It also evaluates the basis on every event of every held-out run at that run's theta and
stores the held-out observables, so the closure test with wifi weights can be run elsewhere with
only the export (for the reference side) and the files written here.

    DM_DATA=data_stagePURE17 python wifi_embed.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_trunk.pt
    python wifi_embed.py E output/models/E_cond.npz output/models/E_trunk.pt      (no fit split: non-reference events)

Writes output/wifi_<export tag>_basis.npz.
"""
import os, sys, json, time
import numpy as np, torch, pandas as pd
sys.path.insert(0, '.')

tag, export, trunk = sys.argv[1], sys.argv[2], sys.argv[3]
c = np.load(export)
kind = str(c.get('head_kind', 'cond'))
os.environ['LADDER_HEAD_KIND'] = kind
os.environ.setdefault('LADDER_ACT', str(c['act']))
os.environ.setdefault('LADDER_EMB_SCALE', str(c['emb_scale']))
os.environ.setdefault('LADDER_K', str(int(c['K_bilinear'])))
import r2_ladder as R
from r2_ladder import build_feats, apply_feature_norm, strange_frac, make_model, emb_all, fit_split

if tag == 'MIX':
    from mixture_cfg import mixture_cfg
    cfg, _ = mixture_cfg(os.environ['DM_DATA'])
else:
    cfg = R.STAGES[tag]
DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
norm = np.asarray(c['norm'], np.float64); ENS, K, NT = int(c['ens']), int(c['K']), int(c['ntheta'])
NFIT, NDEN = int(os.environ.get('NFIT_PER', '4000')), int(os.environ.get('NDEN_PER', '4000'))
OUT = os.environ.get('WIFI_OUT', f'output/wifi_{os.path.basename(export).replace("_cond.npz", "")}_basis.npz')
stem = export.replace('_cond.npz', '')
split_file = f'{stem}_fitsplit.npz'
SPLIT = np.load(split_file) if os.path.exists(split_file) else None
nref_per = int(SPLIT['nref_per']) if SPLIT is not None else R.NREF_PER
if tag == 'MIX' and SPLIT is None:
    nref_per = min(cfg and 80000, max(1000, 1150000//len(cfg['train'])))
print(f'[{tag}] {kind} head, {ENS} members, K={K}, d={NT}; fit split '
      f'{"from " + split_file if SPLIT is not None else "ABSENT, using non-reference events (same data as training)"}', flush=True)

# ---- the trunk, as trained -----------------------------------------------------------------
t = torch.load(trunk, map_location='cpu', weights_only=False)
if 'LADDER_MIX_SPLIT' not in os.environ and kind in R.MIXTURE_KINDS:
    os.environ['LADDER_MIX_SPLIT'] = f"{c['B0_W0'].shape[1]},{c['BH0_W0'].shape[1]}"
FMU, FSD = t['feat_mu'], t['feat_sd']
Cdim = len(FMU)
models = []
for mi, st in enumerate(t['models']):
    m = make_model(Cdim, NT, int(c['K_bilinear'])).to(R.DEV)
    m.load_state_dict(st); m.emb_scale = float(c['emb_scale_used'][mi]); m.emb_auto = False; m.eval(); models.append(m)
act = {'relu': lambda z: np.maximum(z, 0.0), 'silu': lambda z: z/(1 + np.exp(-z))}[str(c['act'])]
L = int(c['nlayers'])


def bnet(prefix, mi, x):
    for li in range(L):
        x = x @ c[f'{prefix}{mi}_W{li}'].T + c[f'{prefix}{mi}_b{li}']
        if li < L - 1:
            x = act(x)
    return x


def btilde(theta):
    """(ENS, K): the parameter vector of every member at one physical theta."""
    x = ((np.asarray(theta, np.float64) - norm[:, 0])/norm[:, 1]).reshape(1, -1)
    if kind in R.MIXTURE_KINDS:
        assert kind == 'geometric', 'the additive mixture head is not an inner product in theta'
        ns = c['B0_W0'].shape[1]; nh = c['BH0_W0'].shape[1]; fr = float(np.clip(theta[-1], 0, 1))
        return np.stack([(1 - fr)*bnet('B', mi, x[:, :ns]).ravel() + fr*bnet('BH', mi, x[:, ns:ns + nh]).ravel() for mi in range(ENS)])
    return np.stack([bnet('B', mi, x).ravel() for mi in range(ENS)])


def avecs(part, msk):
    """(n, ENS, K): a_i(Phi) of every member for the events given."""
    f, mk = build_feats(part, msk, cfg['flavor']); f = apply_feature_norm(f, mk, FMU, FSD)
    f, mk = f.to(R.DEV), mk.to(R.DEV); out = []
    with torch.no_grad():
        for m in models:
            E = emb_all(m, f, mk); out.append(np.concatenate([m.A(E[i:i + 10000]).cpu().numpy() for i in range(0, len(E), 10000)]))
    return np.stack(out, 1).astype(np.float32)


def basis(a, B):
    """(n, ENS*ENS): f_ij = <a_i, b_j> for a (n, ENS, K) and B (ENS, K); index i*ENS + j."""
    return np.einsum('nik,jk->nij', a.astype(np.float64), B).reshape(len(a), ENS*ENS)


def obsvals(d, sh):
    return {o: (sh[o].values.astype(np.float64) if o in cfg['obs'] else
                (d['nbaryon'].astype(np.float64) if o == 'nbaryon' else strange_frac(d['particles'], d['mask']).astype(np.float64)))
            for o in OBS}


t0 = time.time()
rids = [int(r) for r in SPLIT['rids']] if SPLIT is not None else list(cfg['train'])
num_f, num_rid, pool_a, pool_eid = [], [], [], []
theta_of = {}
for k, rid in enumerate(rids):
    d = np.load(f'{DATA}/particles_full_{rid:04d}.npz'); n = len(d['mask'])
    if SPLIT is not None:
        fit = SPLIT[f'fit_{rid}']
    else:
        ref_idx = np.random.default_rng(50000 + rid).choice(n, min(nref_per, n), replace=False)
        fit, _ = fit_split(rid, n, ref_idx, 0.2)
    sub = np.sort(np.random.default_rng(90000 + rid).choice(fit, min(NFIT, len(fit)), replace=False))
    a = avecs(d['particles'][sub], d['mask'][sub])
    theta_of[rid] = np.array(cfg['train'][rid], np.float64)
    num_f.append(basis(a, btilde(theta_of[rid])).astype(np.float32)); num_rid.append(np.full(len(sub), rid))
    pool_a.append(a); pool_eid.append(np.arange(len(sub)) + 10**6*k)
    if k % 20 == 0:
        print(f'  run {rid} ({k+1}/{len(rids)}): {len(sub)} fit events embedded [{time.time()-t0:.0f} s]', flush=True)
pool_a = np.concatenate(pool_a); pool_eid = np.concatenate(pool_eid); NP = len(pool_a)
# the denominator: for every run's theta, a subsample of the pooled fit set (equal counts per run,
# so the pool is a sample of the pooled reference density)
den_f, den_rid, den_eid = [], [], []
for k, rid in enumerate(rids):
    sel = np.sort(np.random.default_rng(95000 + rid).choice(NP, min(NDEN, NP), replace=False))
    den_f.append(basis(pool_a[sel], btilde(theta_of[rid])).astype(np.float32)); den_rid.append(np.full(len(sel), rid)); den_eid.append(pool_eid[sel])
print(f'[{tag}] numerator {sum(len(x) for x in num_f)} rows, denominator {sum(len(x) for x in den_f)} rows over {NP} pooled events [{time.time()-t0:.0f} s]', flush=True)
# the held-out runs: the basis on every event, plus the observables
held = {}
for rid, th in cfg['held'].items():
    d = np.load(f'{DATA}/particles_full_{rid:04d}.npz'); sh = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
    a = avecs(d['particles'], d['mask']); ov = obsvals(d, sh)
    held[f'held_f_{rid}'] = basis(a, btilde(np.array(th, np.float64))).astype(np.float32)
    for o in OBS:
        held[f'held_{o}_{rid}'] = ov[o].astype(np.float32)
    print(f'  held {rid}: {len(a)} events [{time.time()-t0:.0f} s]', flush=True)
np.savez_compressed(OUT, kind=kind, ens=ENS, K=K, ntheta=NT, obs=np.array(OBS), rids=np.array(rids),
                    theta=np.array([theta_of[r] for r in rids]), held_rids=np.array(list(cfg['held'])),
                    held_theta=np.array([cfg['held'][r] for r in cfg['held']], np.float64),
                    num_f=np.concatenate(num_f), num_rid=np.concatenate(num_rid),
                    den_f=np.concatenate(den_f), den_rid=np.concatenate(den_rid), den_eid=np.concatenate(den_eid),
                    fit_split=int(SPLIT is not None), **held)
print(f'wrote {OUT} [{time.time()-t0:.0f} s]')
