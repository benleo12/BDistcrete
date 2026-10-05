#!/usr/bin/env python3
"""Stage D: reweighting across a DISCRETE algorithm change, three showers, one binary.

CSS (8200), Dire (8201) and Alaric (8202) were generated at identical settings by the same
alaric-merge build, so the only thing that differs between two samples is the shower
algorithm. For each ordered pair we train an ensemble classifier on one half of each sample,
reweight the SOURCE's held-out half, and measure the closure against the TARGET's held-out
half with the same ruler as the ladder (sqrt(chi^2/ndf) about zero, null 1).

Two numbers per pair make the point: the width BEFORE reweighting (how far apart the two
showers are on this observable) and AFTER. Both representations are run, because the
heavy-hemisphere representation is what previously failed to close B_total and rho_heavy,
and the claim is that this was a representation artifact rather than a limit of the method.
Writes output/stageD3.json."""
import json, itertools
import numpy as np, torch, torch.nn as nn, pandas as pd
from r2_ladder import (CondPFN, build_feats, pulls, width_of, make_bins,
                       fit_feature_norm, apply_feature_norm)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DATA = 'data_stageD3'
SHOWERS = {8200: 'CSS', 8201: 'Dire', 8202: 'Alaric'}
ENS, K, STEPS = 4, 24, 12000
OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy']
NMAX = 125000                      # per-sample half


def load(rep):
    F, M, SH = {}, {}, {}
    for rid in SHOWERS:
        d = np.load(f'{DATA}/particles_{rep}_{rid}.npz')
        F[rid], M[rid] = build_feats(d['particles'], d['mask'], flavor=False)
        SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid}.csv')
        assert len(SH[rid]) == len(d['mask']), f'{rid}: csv/npz length mismatch'
    # pooled standardization across all three showers, applied identically, ladder recipe
    FMU, FSD = fit_feature_norm([F[r] for r in SHOWERS], [M[r] for r in SHOWERS])
    for rid in F:
        F[rid] = apply_feature_norm(F[rid], M[rid], FMU, FSD)
    return F, M, SH


def split(n, rid):
    p = np.random.default_rng(4000+rid).permutation(n)
    return p[:n//2], p[n//2:]


def train_pair(src, tgt, F, M, tr):
    """Binary ensemble classifier, source=0 vs target=1, on the TRAIN halves only."""
    models = []
    zero = torch.zeros(2048, 1, device=DEV)
    for sd in range(ENS):
        torch.manual_seed(sd)
        m = CondPFN(F[src].shape[-1], 1, K=K).to(DEV)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
        g = torch.Generator().manual_seed(sd)
        bce = nn.BCEWithLogitsLoss()
        lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
        si, ti = tr[src], tr[tgt]
        for _ in range(STEPS):
            a = si[torch.randint(len(si), (1024,), generator=g).numpy()]
            b = ti[torch.randint(len(ti), (1024,), generator=g).numpy()]
            fb = torch.cat([F[src][a].to(DEV), F[tgt][b].to(DEV)])
            mb = torch.cat([M[src][a].to(DEV), M[tgt][b].to(DEV)])
            loss = bce(m.f_from(m.emb(fb, mb), zero), lab)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval(); models.append(m)
    return models


def select_T(models, F, M, SH, src, tgt, stop_s, stop_t):
    """Fit the temperature on a stop split by driving the closure width to its null of one.
    Two shower algorithms are genuinely separable, so an uncalibrated logit concentrates the
    weights (observed: N_eff collapsing to tens of events out of 125k). The ladder recipe
    controls this with calibration at the null, ported here."""
    w0, _ = weights_for(models, F, M, src, stop_s)
    f = np.log(np.clip(w0, 1e-300, None))          # recover logits up to a constant
    best = (None, 1.0)
    for T in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        w = np.exp((f - f.max())/T); w /= w.sum()
        ps, nbt = [], 0
        for o in OBS:
            os_ = SH[src][o].values[stop_s]; ot = SH[tgt][o].values[stop_t]
            b = make_bins(np.r_[os_, ot], o)
            p, nb = pulls(os_, w, ot, b); ps.append(p); nbt += nb
        wd = width_of(np.concatenate(ps), nbt)
        if best[0] is None or abs(wd-1) < best[0]:
            best = (abs(wd-1), T)
    return best[1]


def weights_for(models, F, M, src, idx):
    fs = []
    for m in models:
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 20000):
                j = idx[i:i+20000]
                E = m.emb(F[src][j].to(DEV), M[src][j].to(DEV))
                out.append(m.f_from(E, torch.zeros(len(j), 1, device=DEV)).cpu().numpy())
        fs.append(np.concatenate(out))
    f = np.mean(fs, 0)                      # mean of logits, as in the ladder
    w = np.exp(f - f.max()); w /= w.sum()
    return w, float(1.0/np.sum(w**2))


def main():
    res = {}
    for rep in ['full', 'hemi']:
        print(f'=== representation: {rep} ===')
        F, M, SH = load(rep)
        tr, rp = {}, {}
        for rid in SHOWERS:
            a, b = split(len(F[rid]), rid); tr[rid], rp[rid] = a[:NMAX], b[:NMAX]
        res[rep] = {}
        for src, tgt in itertools.permutations(SHOWERS, 2):
            if src > tgt: continue                     # one direction per unordered pair
            name = f'{SHOWERS[src]}_to_{SHOWERS[tgt]}'
            models = train_pair(src, tgt, F, M, tr)
            # stop split: first quarter of each report half, disjoint from the reported events
            ns, nt = len(rp[src])//4, len(rp[tgt])//4
            T = select_T(models, F, M, SH, src, tgt, rp[src][:ns], rp[tgt][:nt])
            rep_s, rep_t = rp[src][ns:], rp[tgt][nt:]
            w0, _ = weights_for(models, F, M, src, rep_s)
            f = np.log(np.clip(w0, 1e-300, None))
            w = np.exp((f - f.max())/T); w /= w.sum(); neff = float(1.0/np.sum(w**2))
            uni = np.full(len(rep_s), 1.0/len(rep_s))
            row = {}
            for o in OBS:
                if o not in SH[src].columns: continue
                os_ = SH[src][o].values[rep_s]; ot = SH[tgt][o].values[rep_t]
                b = make_bins(np.r_[os_, ot], o)
                pa, na = pulls(os_, w, ot, b)
                pb, nb = pulls(os_, uni, ot, b)
                row[o] = dict(after=width_of(pa, na), before=width_of(pb, nb))
            row['N_eff'] = neff; row['N_ref'] = len(rep_s); row['temperature'] = T
            res[rep][name] = row
            summ = {o: f"{v['before']:.1f}->{v['after']:.2f}" for o, v in row.items() if isinstance(v, dict)}
            print(f'  {name}: {summ}  N_eff={neff:.0f}/{len(rp[src])}')
        json.dump(res, open('output/stageD3.json', 'w'), indent=1)
    print('STAGE D3 ANALYSIS DONE')


if __name__ == '__main__':
    main()
