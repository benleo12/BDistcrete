#!/usr/bin/env python3
"""B2 ablation: the DCTR-standard concatenation baseline against our bilinear head.

The reviewer standard (Baldi et al. 1601.07913; DCTR Andreassen-Nachman 1907.08209) is to make
the classifier theta-aware by CONCATENATING theta with the per-particle inputs and letting a
generic network learn f(Phi,theta). Our bilinear head f=<a(Phi),b(theta)> restricts the log
ratio to K products, which keeps the event side free of theta: one pass per event, then one
inner product per event at each new theta. A referee will ask whether that restriction
underfits relative to the unrestricted concatenation network.

This trains the concatenation baseline under the IDENTICAL recipe (same data, standardization,
SiLU, lr, steps, ensemble, temperature calibration, ruler) and reports the closure width, so it
can be placed next to the bilinear ladder result. If the two agree, the low-rank head does not
underfit. It also stores the singular values of the trained network's logits over events x
theta (dctr_rank.py counts them). Writes output/concat_baseline.json, or CONCAT_OUT.

Concatenation is literal DCTR: theta is appended to every particle's feature vector, so the
event embedding depends on theta (unlike the bilinear head, where the embedding is theta-free).
CONCAT_MODE=late appends theta after the sum over particles instead (LatePFN).
Usage: python concat_baseline.py A B   (default: A B)."""
import sys, os, json
import numpy as np, torch, torch.nn as nn, pandas as pd
from r2_ladder import (STAGES, build_feats, fit_feature_norm, apply_feature_norm,
                       pulls, width_of, make_bins, strange_frac, NREF_PER,
                       PHI_SIZES, F_SIZES, _mlp, EMB_CHUNK)
DEV = ('cuda' if torch.cuda.is_available() else
       'mps' if torch.backends.mps.is_available() else 'cpu')
ENS, STEPS, LR = 4, 36000, 1e-3
CKPTS = [6000, 12000, 18000, 24000, 30000, 36000]


class ConcatPFN(nn.Module):
    """Standard parameterized PFN: theta appended to each particle, generic scalar head.
    Phi: (C+d) -> 256, masked sum -> 256, F: 256 -> ... -> 1."""
    def __init__(s, C, ntheta):
        super().__init__()
        s.phi, L = _mlp(C + ntheta, PHI_SIZES)
        s.head, _ = _mlp(L, F_SIZES, 1)
        s.ntheta = ntheta
        # same pooled-sum conditioning fix as CondPFN (LADDER_EMB_SCALE=auto), so the
        # baseline comparison stays architecture-for-architecture under the fixed recipe
        from r2_ladder import EMB_SCALE
        s.emb_auto = (EMB_SCALE == 'auto')
        s.emb_scale = 1.0 if s.emb_auto else float(EMB_SCALE)
        for mod in s.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu'); nn.init.zeros_(mod.bias)

    def forward(s, f, m, th):
        # th: [B, ntheta] -> broadcast to every particle and concatenate
        B, P, _ = f.shape
        thp = th.unsqueeze(1).expand(B, P, s.ntheta)
        x = torch.cat([f, thp], dim=-1)
        h = s.phi(x.reshape(B*P, -1)).reshape(B, P, -1) * m.unsqueeze(-1)
        E = h.sum(1)
        if s.emb_auto and s.emb_scale == 1.0:
            with torch.no_grad():
                lin0 = [l for l in s.head if isinstance(l, nn.Linear)][0]
                s.emb_scale = float(lin0(E).std())
        if s.emb_scale != 1.0:
            E = E/s.emb_scale
        return s.head(E).squeeze(-1)

    @torch.no_grad()
    def logit_on(s, f, m, th_row, chunk=EMB_CHUNK):
        """f for a whole reference at one theta, chunked (embedding is theta-dependent here)."""
        out = []
        for i in range(0, len(f), chunk):
            fb = f[i:i+chunk]; mb = m[i:i+chunk]
            thb = th_row.expand(len(fb), s.ntheta)
            out.append(s.forward(fb, mb, thb).cpu().numpy())
        return np.concatenate(out)


class LatePFN(nn.Module):
    """Late fusion: the per-particle map and the sum over particles see only the event, and theta
    joins the pooled features before the head, so the pooled features can be computed once and
    cached across theta. Phi: C -> 256, masked sum -> 256, F: (256 + d) -> ... -> 1. Same sizes,
    initialization and embedding-scale fix as ConcatPFN."""
    def __init__(s, C, ntheta):
        super().__init__()
        s.phi, L = _mlp(C, PHI_SIZES)
        s.head, _ = _mlp(L + ntheta, F_SIZES, 1)
        s.ntheta = ntheta
        from r2_ladder import EMB_SCALE
        s.emb_auto = (EMB_SCALE == 'auto')
        s.emb_scale = 1.0 if s.emb_auto else float(EMB_SCALE)
        for mod in s.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu'); nn.init.zeros_(mod.bias)

    def pooled(s, f, m):
        B, P, _ = f.shape
        h = s.phi(f.reshape(B*P, -1)).reshape(B, P, -1) * m.unsqueeze(-1)
        return h.sum(1)

    def head_on(s, E, th):
        x = torch.cat([E, th], dim=-1)
        if s.emb_auto and s.emb_scale == 1.0:
            with torch.no_grad():
                lin0 = [l for l in s.head if isinstance(l, nn.Linear)][0]
                s.emb_scale = float(lin0(x).std())
        if s.emb_scale != 1.0:
            x = torch.cat([E/s.emb_scale, th], dim=-1)
        return s.head(x).squeeze(-1)

    def forward(s, f, m, th):
        return s.head_on(s.pooled(f, m), th)

    @torch.no_grad()
    def logit_on(s, f, m, th_row, chunk=EMB_CHUNK):
        out = []
        for i in range(0, len(f), chunk):
            fb = f[i:i+chunk]; mb = m[i:i+chunk]
            out.append(s.forward(fb, mb, th_row.expand(len(fb), s.ntheta)).cpu().numpy())
        return np.concatenate(out)


MODE = os.environ.get('CONCAT_MODE', 'early')     # early: theta on every particle (DCTR); late: after the sum
Net = LatePFN if MODE == 'late' else ConcatPFN
SUFFIX = '_late' if MODE == 'late' else ''


def run_stage(tag):
    if tag == 'MIX':
        # the seventeen-parameter mixture: its design comes from the data set's own meta.json
        from mixture_cfg import mixture_cfg
        cfg, _ = mixture_cfg(os.environ['DM_DATA'])
    else:
        cfg = STAGES[tag]
    # CONCAT_DROP removes held-out runs that duplicate training runs (Stage B 6704, Stage C 6902)
    drop = {int(x) for x in os.environ.get('CONCAT_DROP', '').split(',') if x}
    if drop:
        cfg = dict(cfg); cfg['held'] = {k: v for k, v in cfg['held'].items() if k not in drop}
        print(f'[{tag}] dropping held-out runs {sorted(drop)}, {len(cfg["held"])} remain', flush=True)
    DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
    # CONCAT_NREF_PER matches the reference size of the factorized model being compared
    nref_per = int(os.environ.get('CONCAT_NREF_PER', str(NREF_PER)))
    norm = np.array(cfg['norm']); d = cfg['ntheta']
    def tn(theta): return ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32)
    SH, NB, STR, tf, tm = {}, {}, {}, {}, {}
    for rid in list(cfg['train']) + list(cfg['held']):
        dd = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        assert len(SH[rid]) == len(dd['mask'])
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
    robs = {o: np.concatenate(v) for o, v in robs.items()}; Nref = len(ref_f); Cdim = ref_f.shape[-1] - 0
    split = {rid: np.random.default_rng(1000+rid).permutation(len(SH[rid])) for rid in cfg['held']}
    stop_idx = {rid: split[rid][:len(split[rid])//2] for rid in cfg['held']}
    rep_idx = {rid: split[rid][len(split[rid])//2:] for rid in cfg['held']}
    BINS = {(rid, o): make_bins(np.r_[robs[o], obsvals(rid, o, rep_idx[rid])], o)
            for rid in cfg['held'] for o in OBS}
    C = ref_f.shape[-1]
    print(f'[{tag}] {MODE} fusion baseline: ref {Nref}, C={C}, d={d}')

    models, opts, schs, gens = [], [], [], []
    for sd in range(ENS):
        torch.manual_seed(sd); m = Net(C, d).to(DEV); models.append(m)
        opts.append(torch.optim.Adam(m.parameters(), LR))
        schs.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[-1], STEPS, eta_min=LR/100))
        gens.append(torch.Generator().manual_seed(sd))
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    rids = list(cfg['train'])

    def logits_at(theta):
        thr = torch.tensor(tn(theta), device=DEV).float().reshape(1, d)
        for m in models: m.eval()
        f = np.mean([m.logit_on(ref_f, ref_m, thr) for m in models], 0)
        for m in models: m.train()
        return f

    _lcache = {}
    def closure(which, T=1.0, cached=False):
        idxs = stop_idx if which == 'stop' else rep_idx; out = {}
        for rid, theta in cfg['held'].items():
            if not cached or rid not in _lcache:
                _lcache[rid] = logits_at(theta)
            f = _lcache[rid]; w = np.exp((f - f.max())/T); w /= w.sum(); po = {}
            for o in OBS:
                ot = obsvals(rid, o, idxs[rid]); b = BINS[(rid, o)]
                p, nb = pulls(robs[o], w, ot, b); po[o] = width_of(p, nb)
            out[rid] = po
        return out
    def master(cl): return float(np.mean([np.mean(list(po.values())) for po in cl.values()]))

    step = 0; best = None
    for ck in CKPTS:
        for mi, m in enumerate(models):
            opt, sch, g = opts[mi], schs[mi], gens[mi]
            for _ in range(ck - step):
                j = rids[int(torch.randint(len(rids), (1,), generator=g))]
                ir = torch.randint(Nref, (1024,), generator=g); it = torch.randint(len(tf[j]), (1024,), generator=g)
                fb = torch.cat([ref_f[ir], tf[j][it].to(DEV)]); mb = torch.cat([ref_m[ir], tm[j][it].to(DEV)])
                th = torch.tensor(tn(cfg['train'][j]), device=DEV).float().expand(2048, d)
                loss = bce(m(fb, mb, th), lab); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step = ck
        ms = master(closure('stop'))
        if best is None or abs(ms-1) < best[0]:
            best = (abs(ms-1), ck, ms, [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models])
        print(f'[{tag}] step {ck}: stop-width={ms:.3f}')
    for m, st in zip(models, best[3]): m.load_state_dict(st)
    os.makedirs('output/models', exist_ok=True)
    torch.save([m.state_dict() for m in models], f'output/models/concat_{tag}{SUFFIX}.pt')
    # Intrinsic rank of the learned f(Phi,theta): SVD of the event x theta matrix on 5000 reference
    # events and 48 thetas uniform in the normalized training box (same protocol as
    # ab_analysis.svd_spectrum for the bilinear head, so the two spectra are comparable).
    rng_s = np.random.default_rng(0); sub = torch.tensor(rng_s.choice(Nref, min(5000, Nref), replace=False), device=DEV)
    U = rng_s.uniform(-1.0, 1.0, size=(48, d)).astype(np.float32)
    for m in models: m.eval()
    Fm = np.array([np.mean([m.logit_on(ref_f[sub], ref_m[sub], torch.tensor(u, device=DEV).reshape(1, d)) for m in models], 0) for u in U])
    for m in models: m.train()
    sv = np.linalg.svd(Fm, compute_uv=False)
    print(f'[{tag}] concat SVD spectrum (normalized): ' + ' '.join(f'{x:.4f}' for x in (sv/sv[0])[:12]))
    grid = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)]
                      + [1.75, 2.0, 2.5, 3.0]))
    _lcache.clear()
    T = min(grid, key=lambda t: abs(master(closure('stop', T=t, cached=True)) - 1))
    rep = closure('report', T=T, cached=True)
    rep1 = closure('report', T=1.0, cached=True)
    per_obs = {o: float(np.mean([rep[rid][o] for rid in cfg['held']])) for o in OBS}
    res = dict(master=master(rep), temperature=T, per_obs=per_obs, best_step=best[1], master_T1=master(rep1),
               per_obs_T1={o: float(np.mean([rep1[rid][o] for rid in cfg['held']])) for o in OBS},
               n_reference=int(Nref), n_held=len(cfg['held']),
               head='late_fusion' if MODE == 'late' else 'concat_mlp_dctr', phi_sizes=list(PHI_SIZES), f_sizes=list(F_SIZES), svd=(sv/sv[0]).tolist(), svd_theta_cloud=U.tolist())
    print(f"[{tag}] CONCAT RESULT width={res['master']:.3f} (T={T})  per_obs={ {o: round(v,2) for o,v in per_obs.items()} }")
    return res


def main():
    OUT = os.environ.get('CONCAT_OUT', 'output/concat_baseline.json')
    stages = [a for a in sys.argv[1:] if a in STAGES or a == 'MIX'] or ['A', 'B']
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for s in stages:
        out[s] = run_stage(s); json.dump(out, open(OUT, 'w'), indent=1)
    print('CONCAT BASELINE DONE:', {s: round(out[s]['master'], 2) for s in stages})


if __name__ == '__main__':
    main()
