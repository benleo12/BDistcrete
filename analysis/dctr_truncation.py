#!/usr/bin/env python3
"""How many terms the concatenation (DCTR) network's learned log ratio needs, by a test that can fail.

The trained network's log ratio on the reference events at a set of parameter points (the held-out
points and 48 points uniform in the box, as in concat_baseline.py) forms a matrix of events by points.
Truncating it to its leading r singular components and recomputing the closure width at the held-out
points, with the published evaluation (same reference subsample, bins, report half and temperature),
gives the closure as a function of r. TEST_PLAN_2026-10-04.md fixes the reading: the learned ratio is
low rank for a stage if the truncated network reaches the full network's closure, within the seed
spread, at r no larger than K2 = 1 + d + d(d+1)/2 (testable for d = 1, 2, 3, 8, since at d = 17 the
matrix has fewer columns than K2). Also reported: the singular-value counts at 1, 3 and 10 percent of
the largest, and the noise floor, the spectrum of the difference of two of the four networks.

The data preparation repeats concat_baseline.run_stage line for line, so the evaluation is the
published one. Run with the settings of the run that trained the weights, for example

    CONCAT_NREF_PER=12000 CONCAT_DROP=6902 python dctr_truncation.py C output/models/concat_C_s10.pt
    CONCAT_NREF_PER=5989 DM_DATA=data_stageDM17aug_v2 python dctr_truncation.py MIX output/models/concat_MIX.pt
"""
import sys, os, json
import numpy as np, torch, pandas as pd
from r2_ladder import (STAGES, build_feats, fit_feature_norm, apply_feature_norm, pulls, width_of,
                       make_bins, strange_frac, NREF_PER)
from concat_baseline import ConcatPFN, LatePFN, MixtureOf, MODE, HEAD

DEV = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
tag, wpath = sys.argv[1], sys.argv[2]
OUT = os.environ.get('TRUNC_OUT', f'output/dctr_truncation_{tag}_{os.path.basename(wpath).replace(".pt", "")}.json')

# ---- data, exactly as concat_baseline.run_stage ------------------------------------------------
if tag == 'MIX':
    from mixture_cfg import mixture_cfg
    cfg, _ = mixture_cfg(os.environ['DM_DATA'])
else:
    cfg = STAGES[tag]
drop = {int(x) for x in os.environ.get('CONCAT_DROP', '').split(',') if x}
if drop:
    cfg = dict(cfg); cfg['held'] = {k: v for k, v in cfg['held'].items() if k not in drop}
DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
nref_per = int(os.environ.get('CONCAT_NREF_PER', str(NREF_PER)))
norm = np.array(cfg['norm']); d = cfg['ntheta']
def tn(theta): return ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32)
SH, NB, STR, tf, tm = {}, {}, {}, {}, {}
for rid in list(cfg['train']) + list(cfg['held']):
    dd = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
    SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
    if 'nbaryon' in cfg['flav_obs']: NB[rid] = dd['nbaryon'].astype(np.float32)
    if 'strange' in cfg['flav_obs']: STR[rid] = strange_frac(dd['particles'], dd['mask']).astype(np.float32)
    if rid in cfg['train']:
        tf[rid], tm[rid] = build_feats(dd['particles'], dd['mask'], cfg['flavor'])
def obsvals(rid, o, idx=None):
    v = (SH[rid][o].values if o in cfg['obs'] else (NB[rid] if o == 'nbaryon' else STR[rid]))
    return v if idx is None else v[idx]
FMU, FSD = fit_feature_norm([tf[r] for r in cfg['train']], [tm[r] for r in cfg['train']])
for rid in cfg['train']:
    tf[rid] = apply_feature_norm(tf[rid], tm[rid], FMU, FSD)
rng = np.random.default_rng(0); rf, rm, robs = [], [], {o: [] for o in OBS}
for rid in cfg['train']:
    n = len(tf[rid]); idx = rng.choice(n, min(nref_per, n), replace=False)
    rf.append(tf[rid][idx]); rm.append(tm[rid][idx])
    for o in OBS: robs[o].append(obsvals(rid, o, idx))
ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
robs = {o: np.concatenate(v) for o, v in robs.items()}; Nref = len(ref_f)
split = {rid: np.random.default_rng(1000+rid).permutation(len(SH[rid])) for rid in cfg['held']}
stop_idx = {rid: split[rid][:len(split[rid])//2] for rid in cfg['held']}
rep_idx = {rid: split[rid][len(split[rid])//2:] for rid in cfg['held']}
BINS = {(rid, o): make_bins(np.r_[robs[o], obsvals(rid, o, rep_idx[rid])], o) for rid in cfg['held'] for o in OBS}
C = ref_f.shape[-1]

# ---- the trained networks ----------------------------------------------------------------------
# The pooled-feature scale (LADDER_EMB_SCALE=auto) is set at the first training step from the initial
# weights and is not in the state dict. Newer runs store it beside the weights. For older runs it is
# recovered by replaying that step: the same seed builds the same initial network, and the same
# generator draws the same first batch as concat_baseline.run_stage.
Base = LatePFN if MODE == 'late' else ConcatPFN
make = (lambda: MixtureOf(Base, C, d)) if HEAD == 'mixture' else (lambda: Base(C, d))
off = int(os.environ.get('CONCAT_SEED_OFFSET', '0'))
side = wpath.replace('.pt', '_embscale.json')
scales = json.load(open(side)) if os.path.exists(side) else None
rids = list(cfg['train']); lab_rows = 2048
models = []
for mi, st in enumerate(torch.load(wpath, map_location='cpu')):
    torch.manual_seed(mi + off); m = make().to(DEV)
    core = getattr(m, 'g', m)
    if scales is not None:
        core.emb_scale = float(scales[mi])
    else:
        g = torch.Generator().manual_seed(mi + off)
        j = rids[int(torch.randint(len(rids), (1,), generator=g))]
        ir = torch.randint(Nref, (1024,), generator=g); it = torch.randint(len(tf[j]), (1024,), generator=g)
        fb = torch.cat([ref_f[ir], tf[j][it].to(DEV)]); mb = torch.cat([ref_m[ir], tm[j][it].to(DEV)])
        th = torch.tensor(tn(cfg['train'][j]), device=DEV).float().expand(lab_rows, d)
        with torch.no_grad():
            m(fb, mb, th)
    m.load_state_dict(st); core.emb_auto = False; m.eval(); models.append(m)
print(f'[{tag}] {len(models)} networks from {wpath}, scales ' +
      ('stored' if scales is not None else 'replayed') + f' {[round(getattr(x, "g", x).emb_scale, 4) for x in models]}, '
      f'reference {Nref}, d={d}, device {DEV}', flush=True)
del tf, tm

held = list(cfg['held'])
U = np.random.default_rng(0).uniform(-1.0, 1.0, size=(48, d)).astype(np.float32)
cols = [tn(cfg['held'][r]) for r in held] + list(U)
with torch.no_grad():
    per = np.stack([np.stack([m.logit_on(ref_f, ref_m, torch.tensor(u, device=DEV).reshape(1, d)) for u in cols], 1)
                    for m in models])                                       # (networks, events, points)
F = per.mean(0)                                                             # the ensemble's log ratio
Ff = F.astype(np.float64)

def closure_from(Fm, T, which='report'):
    idxs = stop_idx if which == 'stop' else rep_idx; out = []
    for j, rid in enumerate(held):
        f = Fm[:, j]; w = np.exp((f - f.max())/T); w /= w.sum(); po = []
        for o in OBS:
            p, nb = pulls(robs[o], w, obsvals(rid, o, idxs[rid]), BINS[(rid, o)]); po.append(width_of(p, nb))
        out.append(np.mean(po))
    return float(np.mean(out))

grid = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)] + [1.75, 2.0, 2.5, 3.0]))
T = min(grid, key=lambda t: abs(closure_from(Ff, t, 'stop') - 1))
full = closure_from(Ff, T)
# acceptance check: the reloaded networks must reproduce the published evaluation of these weights
pub = os.environ.get('TRUNC_PUBLISHED')
if pub:
    p = json.load(open(pub))[tag]
    print(f'[{tag}] reproduction: closure {full:.4f} at T={T} against published {p["master"]:.4f} at T={p["temperature"]}', flush=True)
Uv, S, Vt = np.linalg.svd(Ff, full_matrices=False)
sv = S/S[0]
noise = np.linalg.svd((per[0] - per[1]).astype(np.float64), compute_uv=False)/S[0]
K2 = 1 + d + d*(d + 1)//2
rs = [r for r in range(1, len(S) + 1) if r <= 12 or r % 4 == 0 or r == K2 or r == d + 1]
trunc = {r: closure_from((Uv[:, :r]*S[:r]) @ Vt[:r], T) for r in rs}
res = dict(tag=tag, weights=wpath, scales='stored' if scales is not None else 'replayed', published=pub, d=d, K2=K2, n_points=len(cols), n_reference=int(Nref), T=T, closure_full=full,
           closure_truncated={str(r): w for r, w in trunc.items()}, singular_values=sv.tolist(),
           counts={'1pct': int((sv > 0.01).sum()), '3pct': int((sv > 0.03).sum()), '10pct': int((sv > 0.10).sum())},
           noise_floor=noise.tolist(), count_above_noise=int((sv > noise.max()).sum()))
json.dump(res, open(OUT, 'w'), indent=1)
print(f"[{tag}] T={T}  full closure {full:.3f}  counts 1/3/10 pct {res['counts']}  above noise {res['count_above_noise']}")
print('  truncated: ' + '  '.join(f'r={r}:{w:.3f}' for r, w in trunc.items()))
print('wrote', OUT)
