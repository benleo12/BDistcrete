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
import json, os, time as _time
import numpy as np, torch, torch.nn as nn, pandas as pd
from r2_ladder import (ACT_NAME as _ACT_NAME, EMB_SCALE as _EMB,
                       STAGES, CondPFN, build_feats, pulls, width_of, make_bins,
                       strange_frac, NREF_PER, emb_all, fit_feature_norm, apply_feature_norm,
                       MMAP, MMAP_DIR)
DEV = ('cuda' if torch.cuda.is_available() else
       'mps' if torch.backends.mps.is_available() else 'cpu')
ENS_SCAN, STEPS_SCAN, LR = 2, 36000, 1e-3     # same committed recipe as the ladder
import os as _os
KS = [int(k) for k in _os.environ.get('RANK_KS', '1,2,3,4,6,8,16').split(',')]
SCAN_STAGES = list(_os.environ.get('RANK_STAGES', 'AB'))
# One K per process is what makes this a job array: each task trains a single rank and
# writes its own file, so a task that dies costs one rank rather than the whole scan.
# RANK_OUT keeps the tasks from clobbering each other; merge_rank_scan.py joins them.
OUT = _os.environ.get('RANK_OUT', 'output/rank_scan.json')
ENS_SCAN = int(_os.environ.get('RANK_ENS', str(ENS_SCAN)))
STEPS_SCAN = int(_os.environ.get('RANK_STEPS', str(STEPS_SCAN)))


def prep(tag):
    cfg = STAGES[tag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
    norm = np.array(cfg['norm'])
    def tn(theta): return ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32)
    # The scan needs particles only for the training runs; the held runs enter through their
    # observable columns alone. What still does not fit is the training side: 96 runs of 80k
    # events at 100 x 7 float32 is about 21 GB, the same dense path that took the laptop down.
    # LADDER_LOWMEM=1 releases each run's array as soon as its features and reference
    # subsample are taken, and LADDER_MMAP=1 additionally holds the features on disk as
    # float16 and gathers them per batch. Both default off, so the published A/B scan at d=1,2
    # runs on exactly the code path it ran on before.
    LOWMEM = os.environ.get('LADDER_LOWMEM', '') == '1'
    SH, NB, STR, tf, tm, SUBIDX, HEAD = {}, {}, {}, {}, {}, {}, {}
    # Loading is compression-bound: every run is a zlib npz, and on a shared node with a
    # quarter of the cores it dominates a short job. A 30 minute smoke test died in here with
    # no output at all, so the loop now says where it is and how fast it is going.
    _rids = list(cfg['train']) + list(cfg['held'])
    _t0 = _time.time()
    for _k, rid in enumerate(_rids):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        assert len(SH[rid]) == len(d['mask'])
        if 'nbaryon' in cfg['flav_obs']: NB[rid] = d['nbaryon'].astype(np.float32)
        if 'strange' in cfg['flav_obs']: STR[rid] = strange_frac(d['particles'], d['mask']).astype(np.float32)
        if rid in cfg['train']:
            tf[rid], tm[rid] = build_feats(d['particles'], d['mask'], cfg['flavor'])
            if MMAP:
                os.makedirs(MMAP_DIR, exist_ok=True)
                HEAD[rid] = (tf[rid][:4000].clone(), tm[rid][:4000].clone())
                np.save(f'{MMAP_DIR}/f_{rid}.npy', tf[rid].numpy().astype(np.float16))
                np.save(f'{MMAP_DIR}/m_{rid}.npy', tm[rid].numpy().astype(np.uint8))
                tf[rid] = np.load(f'{MMAP_DIR}/f_{rid}.npy', mmap_mode='r')
                tm[rid] = np.load(f'{MMAP_DIR}/m_{rid}.npy', mmap_mode='r')
            if LOWMEM:
                _n = len(tf[rid])
                SUBIDX[rid] = np.random.default_rng(50000 + rid).choice(_n, min(NREF_PER, _n), replace=False)
        del d
        if _k % 8 == 7 or _k == len(_rids) - 1:
            _el = _time.time() - _t0
            print(f'[{tag}] loaded {_k+1}/{len(_rids)} runs in {_el:.0f}s '
                  f'({_el/(_k+1):.2f}s per run, {_el/(_k+1)*len(_rids):.0f}s projected)',
                  flush=True)
    def obsvals(rid, o, idx=None):
        v = (SH[rid][o].values if o in cfg['obs'] else (NB[rid] if o == 'nbaryon' else STR[rid]))
        return v if idx is None else v[idx]
    # standardize features with statistics pooled over all training runs (committed recipe).
    # Under MMAP the pooled statistics come from the first 4000 events of each run, held back
    # before the features went to disk, because the memmaps are float16 and are normalized on
    # the way out instead of in place.
    if MMAP:
        FMU, FSD = fit_feature_norm([HEAD[r][0] for r in cfg['train']], [HEAD[r][1] for r in cfg['train']])
        HEAD.clear()
    else:
        FMU, FSD = fit_feature_norm([tf[r] for r in cfg['train']], [tm[r] for r in cfg['train']])
        for rid in cfg['train']:
            tf[rid] = apply_feature_norm(tf[rid], tm[rid], FMU, FSD)

    def fetch(rid, idx):
        """Normalized (features, mask) rows of a training run; from the memmap under MMAP."""
        if MMAP:
            f = torch.from_numpy(np.asarray(tf[rid][idx]).astype(np.float32))
            m = torch.from_numpy(np.asarray(tm[rid][idx]).astype(np.float32))
            return apply_feature_norm(f, m, FMU, FSD), m
        return tf[rid][idx], tm[rid][idx]

    print(f'[{tag}] feature norm done at {_time.time()-_t0:.0f}s', flush=True)
    rng = np.random.default_rng(0)
    rf, rm, robs = [], [], {o: [] for o in OBS}
    for rid in cfg['train']:
        n = len(tf[rid])
        idx = SUBIDX[rid] if LOWMEM else rng.choice(n, min(NREF_PER, n), replace=False)
        _f, _m = fetch(rid, idx); rf.append(_f); rm.append(_m)
        for o in OBS: robs[o].append(obsvals(rid, o, idx))
    ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
    robs = {o: np.concatenate(v) for o, v in robs.items()}
    # report on the same disjoint half the ladder reports on
    rep = {}
    for rid in cfg['held']:
        p = np.random.default_rng(1000+rid).permutation(len(SH[rid])); rep[rid] = p[len(p)//2:]
    print(f'[{tag}] reference gathered at {_time.time()-_t0:.0f}s', flush=True)
    BINS = {(rid, o): make_bins(np.r_[robs[o], obsvals(rid, o, rep[rid])], o)
            for rid in cfg['held'] for o in OBS}
    return cfg, OBS, tn, tf, tm, ref_f, ref_m, robs, rep, obsvals, BINS, fetch


def scan_stage(tag):
    cfg, OBS, tn, tf, tm, ref_f, ref_m, robs, rep, obsvals, BINS, fetch = prep(tag)
    Nref = len(ref_f); Cdim = ref_f.shape[-1]; rids = list(cfg['train'])
    d = cfg['ntheta']
    print(f'[{tag}] d={d} (cliff predicted at K={d+1}, exhausted at K2={1+d+d*(d+1)//2}); ref {Nref}, C={Cdim}')
    print(f'[{tag}] dev={DEV} ens={ENS_SCAN} steps={STEPS_SCAN} Ks={KS} mmap={MMAP}')
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
            _ts = _time.time()
            for _st in range(STEPS_SCAN):
                if _st in (200, 2000) or (_st and _st % 6000 == 0):
                    _r = (_time.time() - _ts)/_st
                    print(f'[{tag}] K={Kv} seed={sd} step {_st}: {_r*1000:.1f} ms per step, '
                          f'{_r*STEPS_SCAN*ENS_SCAN/3600:.2f} h projected for this rank',
                          flush=True)
                j = rids[int(torch.randint(len(rids), (1,), generator=g))]
                ir = torch.randint(Nref, (1024,), generator=g)
                it = torch.randint(len(tf[j]), (1024,), generator=g)
                _fj, _mj = fetch(j, np.sort(it.numpy()) if MMAP else it)
                fb = torch.cat([ref_f[ir], _fj.to(DEV)])
                mb = torch.cat([ref_m[ir], _mj.to(DEV)])
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
        if DEV == 'cuda': torch.cuda.empty_cache()
    # Record the recipe with the numbers. Ranks are only comparable if every point was
    # trained identically, and a scan assembled from files written by different jobs cannot
    # check that after the fact unless the files say so.
    return dict(d=d, predicted_cliff=d+1, steps=STEPS_SCAN, ens=ENS_SCAN, lr=LR,
                act=_ACT_NAME, emb_scale=str(_EMB), nref=int(Nref),
                scan={str(k): v for k, v in out.items()})


def main():
    res = {}
    for tag in SCAN_STAGES:
        res[tag] = scan_stage(tag)
        json.dump(res, open(OUT, 'w'), indent=1)
    for tag, r in res.items():
        w = {int(k): v['width'] for k, v in r['scan'].items()}
        ks = sorted(w)
        drops = [(k, w[ks[i-1]] - w[k]) for i, k in enumerate(ks) if i > 0]
        cliff = max(drops, key=lambda t: t[1])[0] if drops else None
        print(f'{tag}: d={r["d"]} predicted cliff K={r["predicted_cliff"]}, largest drop at K={cliff}')
    print('R3 RANK SCAN DONE')


if __name__ == '__main__':
    main()
