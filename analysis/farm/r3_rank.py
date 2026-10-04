#!/usr/bin/env python3
"""R3(a): does the required bilinear rank MOVE with the number of parameters?

The log-ratio of a smooth family is, to first order in theta,
    log q(Phi;theta)/q(Phi;theta_0) = sum_p (theta-theta_0)_p S_p(Phi) + c(theta),
one score function per parameter plus a theta-only normalization term. Represented as an
inner product <a(Phi), b(theta)>, that needs rank d+1 for d parameters: d event-side score
directions and one constant direction. The prediction is therefore a CLIFF in closure at
K = d+1 that moves as d changes, and it is the sharpest interpretability claim in the paper.

Stage A has d=1 (cliff predicted at K=2) and Stage B has d=2 (cliff predicted at K=3), so
running both on the same ruler and the same budget tests the movement rather than just the
existence of a cliff.

Every K is trained to the SAME fixed step count with NO early stopping, so no K can win by
being selected more favourably; the only difference between points is the rank. Writes
output/rank_scan.json."""
import json
import numpy as np, torch, torch.nn as nn, pandas as pd
from r2_ladder import (STAGES, CondPFN, build_feats, pulls, width_of, make_bins,
                       strange_frac, NREF_PER, emb_all, fit_feature_norm, apply_feature_norm)
DEV = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ENS_SCAN, STEPS_SCAN, LR = 2, 36000, 1e-3     # same committed recipe as the ladder
import os as _os
KS = [int(k) for k in _os.environ.get('RANK_KS', '1,2,3,4,6,8,16').split(',')]
SCAN_STAGES = list(_os.environ.get('RANK_STAGES', 'AB'))
ENS_SCAN = int(_os.environ.get('RANK_ENS', '2'))


def prep(tag):
    cfg = STAGES[tag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
    norm = np.array(cfg['norm'])
    def tn(theta): return ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32)
    SH, NB, STR, tf, tm = {}, {}, {}, {}, {}
    for rid in list(cfg['train']) + list(cfg['held']):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        assert len(SH[rid]) == len(d['mask'])
        if 'nbaryon' in cfg['flav_obs']: NB[rid] = d['nbaryon'].astype(np.float32)
        if 'strange' in cfg['flav_obs']: STR[rid] = strange_frac(d['particles'], d['mask']).astype(np.float32)
        if rid in cfg['train']:
            tf[rid], tm[rid] = build_feats(d['particles'], d['mask'], cfg['flavor'])
    def obsvals(rid, o, idx=None):
        v = (SH[rid][o].values if o in cfg['obs'] else (NB[rid] if o == 'nbaryon' else STR[rid]))
        return v if idx is None else v[idx]
    # standardize features with statistics pooled over all training runs (committed recipe)
    FMU, FSD = fit_feature_norm([tf[r] for r in cfg['train']], [tm[r] for r in cfg['train']])
    for rid in cfg['train']:
        tf[rid] = apply_feature_norm(tf[rid], tm[rid], FMU, FSD)
    rng = np.random.default_rng(0)
    rf, rm, robs = [], [], {o: [] for o in OBS}
    for rid in cfg['train']:
        n = len(tf[rid]); idx = rng.choice(n, min(NREF_PER, n), replace=False)
        rf.append(tf[rid][idx]); rm.append(tm[rid][idx])
        for o in OBS: robs[o].append(obsvals(rid, o, idx))
    ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
    robs = {o: np.concatenate(v) for o, v in robs.items()}
    # report on the same disjoint half the ladder reports on
    rep = {}
    for rid in cfg['held']:
        p = np.random.default_rng(1000+rid).permutation(len(SH[rid])); rep[rid] = p[len(p)//2:]
    BINS = {(rid, o): make_bins(np.r_[robs[o], obsvals(rid, o, rep[rid])], o)
            for rid in cfg['held'] for o in OBS}
    return cfg, OBS, tn, tf, tm, ref_f, ref_m, robs, rep, obsvals, BINS


def scan_stage(tag):
    cfg, OBS, tn, tf, tm, ref_f, ref_m, robs, rep, obsvals, BINS = prep(tag)
    Nref = len(ref_f); Cdim = ref_f.shape[-1]; rids = list(cfg['train'])
    d = cfg['ntheta']
    print(f'[{tag}] d={d} (cliff predicted at K={d+1}); ref {Nref}, C={Cdim}')
    out = {}
    for Kv in KS:
        models = []
        for sd in range(ENS_SCAN):
            torch.manual_seed(sd)
            m = CondPFN(Cdim, d, K=Kv).to(DEV)
            opt = torch.optim.Adam(m.parameters(), LR)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS_SCAN, eta_min=LR/100)
            g = torch.Generator().manual_seed(sd)
            bce = nn.BCEWithLogitsLoss()
            lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
            for _ in range(STEPS_SCAN):
                j = rids[int(torch.randint(len(rids), (1,), generator=g))]
                ir = torch.randint(Nref, (1024,), generator=g)
                it = torch.randint(len(tf[j]), (1024,), generator=g)
                fb = torch.cat([ref_f[ir], tf[j][it].to(DEV)])
                mb = torch.cat([ref_m[ir], tm[j][it].to(DEV)])
                th = torch.tensor(tn(cfg['train'][j]), device=DEV).float().expand(2048, d)
                loss = bce(m.f_from(m.emb(fb, mb), th), lab)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            m.eval(); models.append(m)
        Es = [emb_all(m, ref_f, ref_m) for m in models]
        ws = []
        for rid, theta in cfg['held'].items():
            th = torch.tensor(tn(theta), device=DEV).float().expand(Nref, d)
            with torch.no_grad():
                f = np.mean([m.f_from(E, th).cpu().numpy() for m, E in zip(models, Es)], 0)
            w = np.exp(f - f.max()); w /= w.sum()
            ps, nbt = [], 0
            for o in OBS:
                ot = obsvals(rid, o, rep[rid]); b = BINS[(rid, o)]
                p, nb = pulls(robs[o], w, ot, b); ps.append(p); nbt += nb
            ws.append(width_of(np.concatenate(ps), nbt))
        out[Kv] = dict(width=float(np.mean(ws)), spread=float(np.std(ws)))
        print(f'[{tag}] K={Kv:>2}: width={out[Kv]["width"]:.3f} +- {out[Kv]["spread"]:.3f}')
        del models, Es
    return dict(d=d, predicted_cliff=d+1, scan={str(k): v for k, v in out.items()})


def main():
    res = {}
    for tag in SCAN_STAGES:
        res[tag] = scan_stage(tag)
        json.dump(res, open('output/rank_scan.json', 'w'), indent=1)
    for tag, r in res.items():
        w = {int(k): v['width'] for k, v in r['scan'].items()}
        ks = sorted(w)
        drops = [(k, w[ks[i-1]] - w[k]) for i, k in enumerate(ks) if i > 0]
        cliff = max(drops, key=lambda t: t[1])[0] if drops else None
        print(f'{tag}: d={r["d"]} predicted cliff K={r["predicted_cliff"]}, largest drop at K={cliff}')
    print('R3 RANK SCAN DONE')


if __name__ == '__main__':
    main()
