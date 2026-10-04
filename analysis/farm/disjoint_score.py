#!/usr/bin/env python3
"""Data-disjoint score reproducibility (the honest version of the fig_p_factor panel d test).

Two Stage A models are trained under the committed recipe, but each sees a DISJOINT half of
every training run's events, so the pair shares no training data at all (and differs in seed).
Their per-event score estimates are then compared on the common exported reference events,
against the shared-data seed-only baseline r=0.61 of the production ensemble. If the disjoint
r is close to the shared r, seed noise dominates and sample noise is subleading, which is
what the ensemble-average argument needs. Writes output/disjoint_score.json."""
import json
import numpy as np
import torch
import torch.nn as nn
from r2_ladder import (STAGES, CondPFN, build_feats, fit_feature_norm, apply_feature_norm,
                       emb_all)

DEV = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
STEPS, K, LR = 36000, 24, 1e-3
cfg = STAGES['A']


def main():
    torch.manual_seed(0)
    tf, tm, half = {}, {}, {}
    for rid in cfg['train']:
        d = np.load(f"{cfg['data']}/particles_full_{rid:04d}.npz")
        f, m = build_feats(d['particles'], d['mask'], cfg['flavor'])
        tf[rid], tm[rid] = f, m
        p = np.random.default_rng(9000+rid).permutation(len(f))
        half[rid] = (p[:len(p)//2], p[len(p)//2:])
    FMU, FSD = fit_feature_norm([tf[r] for r in cfg['train']], [tm[r] for r in cfg['train']])
    for rid in cfg['train']:
        tf[rid] = apply_feature_norm(tf[rid], tm[rid], FMU, FSD)
    norm = np.array(cfg['norm'])
    def tn(a): return float((a - norm[0, 0])/norm[0, 1])
    rids = list(cfg['train'])
    models = []
    for h_outer in (0, 1):
        h = h_outer
        # pooled reference for this half, mirroring the production construction: an equal
        # subsample per training run, drawn only from this half's events
        rng = np.random.default_rng(50+h)
        rf, rm = [], []
        for rid in rids:
            idx = rng.choice(half[rid][h], min(6000, len(half[rid][h])), replace=False)
            rf.append(tf[rid][idx]); rm.append(tm[rid][idx])
        ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
        Nref = len(ref_f)
        for attempt in range(3):
            torch.manual_seed(100 + h + 10*attempt)
            m = CondPFN(tf[rids[0]].shape[-1], 1, K=K).to(DEV)
            opt = torch.optim.Adam(m.parameters(), LR)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR/100)
            g = torch.Generator().manual_seed(100 + h + 10*attempt)
            bce = nn.BCEWithLogitsLoss()
            lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
            diverged = False
            # production objective exactly (r2_ladder loop): pooled-union reference batch,
            # target batch from run j's half, ONE theta conditioning the entire batch
            for st in range(STEPS):
                j = rids[int(torch.randint(len(rids), (1,), generator=g))]
                hi = half[j][h]
                it = hi[torch.randint(len(hi), (1024,), generator=g).numpy()]
                ir = torch.randint(Nref, (1024,), generator=g)
                fb = torch.cat([ref_f[ir], tf[j][it].to(DEV)])
                mb = torch.cat([ref_m[ir], tm[j][it].to(DEV)])
                th = torch.full((2048, 1), tn(cfg['train'][j][0]), device=DEV)
                loss = bce(m.f_from(m.emb(fb, mb), th), lab)
                if not torch.isfinite(loss):
                    print(f'half {h} attempt {attempt}: loss non-finite at step {st}, retrying')
                    diverged = True
                    break
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                if (st+1) % 6000 == 0:
                    print(f'half {h} attempt {attempt}: step {st+1} loss {float(loss):.4f}')
            if not diverged:
                break
        m.eval(); models.append(m)
    # common evaluation events: the exported production reference
    ref = np.load('output/models/A_ref.npz')
    fe, me = build_feats(ref['particles'], ref['mask'], cfg['flavor'])
    c = np.load('output/models/A_cond.npz')
    fe = apply_feature_norm(fe, me, torch.tensor(c['feat_mu']), torch.tensor(c['feat_sd']))
    sub = np.random.default_rng(11).choice(len(fe), 6000, replace=False)
    fe, me = fe[sub].to(DEV), me[sub].to(DEV)
    tgrid = np.linspace(0.108, 0.132, 41)
    thn = ((tgrid - norm[0, 0])/norm[0, 1]).astype(np.float32)
    S = []
    for hi, m in enumerate(models):
        E = emb_all(m, fe, me)
        with torch.no_grad():
            F = np.stack([m.f_from(E, torch.full((len(sub), 1), float(t), device=DEV)).cpu().numpy()
                          for t in thn], 1)
        s = (F @ thn)/np.sum(thn**2)
        print(f'half {hi}: score nan-count {int(np.isnan(s).sum())}, std {np.nanstd(s):.4f}, '
              f'F nan-count {int(np.isnan(F).sum())}')
        S.append(s)
    np.savez('output/disjoint_scores.npz', s0=S[0], s1=S[1], sub=sub)
    ok = np.isfinite(S[0]) & np.isfinite(S[1])
    r = float(np.corrcoef(S[0][ok], S[1][ok])[0, 1])
    print(f'DISJOINT-DATA single-model score correlation r = {r:.3f} '
          f'({int(ok.sum())}/{len(sub)} finite; shared-data baseline 0.61)')
    json.dump(dict(r_disjoint_singles=r, r_shared_singles=0.607, n_eval=int(ok.sum()),
                   steps=STEPS, note='each model trained on a disjoint half of every training run'),
              open('output/disjoint_score.json', 'w'), indent=1)
    print('DISJOINT SCORE DONE')


if __name__ == '__main__':
    main()
