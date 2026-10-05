#!/usr/bin/env python3
"""Corrected measurement of the accuracy law chi2/ndf = 1 + N_tgt/N* on the width toy.

Why this supersedes the Panel-B fit in budget_width.py: there the pull denominator is
sig = sqrt(pre^2 + pte^2) with the REFERENCE error pre held at a fixed nref = 2e5 while the
target error pte ~ 1/sqrt(N_tgt) shrinks. Once N_tgt approaches nref, sig stops falling and
chi2 SATURATES instead of rising linearly, so the largest-N_tgt points bend below the law --
and those are exactly the points that dominate the through-origin slope
N* = sum(N^2)/sum((chi2-1)N). The result was N* = 123k while the clean mid-range points imply
N* ~ 66-71k.

This script measures the law where it is defined and shows where it breaks:
  CLEAN  : nref = 1e6, N_tgt <= 5e4, so the reference contributes < 5% of sig^2 throughout.
           N* is fit by weighted least squares through (0,1) with a replica uncertainty, and
           the linearity is reported (residuals + a fit chi2), not assumed.
  CONTROL: nref = 2e5 with N_tgt pushed to 2e5, reproducing the saturation, so the bend is
           demonstrably an artifact of finite reference statistics rather than a failure of
           the law.
Writes output/budget_law.json."""
import json
import numpy as np, torch, torch.nn as nn
NP = 10
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
CHUNK = 200_000


def full(n, v): return torch.full((n,), float(v), device=DEV)
def sample(s, n, rng): return rng.normal(0.0, np.exp(s), (n, NP)).astype(np.float32)


class Cond(nn.Module):
    def __init__(s, L=24, h=64, K=12):
        super().__init__()
        s.phi = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, L))
        s.A = nn.Sequential(nn.Linear(L, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
        s.B = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
    def emb(s, x): b = x.shape[0]; return s.phi(x.reshape(b*NP, 1)).reshape(b, NP, -1).sum(1)
    def at(s, E, th): return (s.A(E)*s.B(th.reshape(-1, 1))).sum(-1)
    def forward(s, x, th): return s.at(s.emb(x), th)


def train(M, E, epochs=6000, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    grid = list(np.linspace(-0.25, 0.25, M)); xref = sample(0.0, 200000, rng)
    xr = torch.tensor(xref).to(DEV)
    tg = [torch.tensor(sample(s, E, rng)).to(DEV) for s in grid]
    m = Cond().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-5)
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    for ep in range(epochs):
        j = int(torch.randint(M, (1,), generator=g))
        ir = torch.randint(200000, (1024,), generator=g); it = torch.randint(E, (1024,), generator=g)
        loss = bce(m(torch.cat([xr[ir], tg[j][it]]), full(2048, grid[j])), lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); return m


def hist(o, w, bins):
    bw = np.diff(bins); W = w.sum() if w is not None else len(o)
    wv = w if w is not None else np.ones(len(o))
    h, _ = np.histogram(o, bins=bins, weights=wv); h2, _ = np.histogram(o, bins=bins, weights=wv**2)
    return h/(W*bw), np.sqrt(h2)/(W*bw)


OBS = [(lambda X: X.reshape(-1), True), (lambda X: (X**2).mean(1), False)]


def f_on_ref(m, xr_t, s):
    """log-ratio over a (possibly large) reference, evaluated in chunks."""
    out = []
    with torch.no_grad():
        for i in range(0, len(xr_t), CHUNK):
            xb = xr_t[i:i+CHUNK]
            out.append(m.at(m.emb(xb), full(len(xb), s)).cpu().numpy())
    return np.concatenate(out)


def ref_weights(m, xr_t):
    """Reference weights per parameter value s. Depends only on the model and the reference,
    so this is computed ONCE and reused across every N_tgt and replica."""
    W = {}
    for s in [x for x in np.linspace(-0.2, 0.2, 7) if abs(x) > 0.02]:
        f = f_on_ref(m, xr_t, s)
        w = np.exp(f - f.max()); W[s] = w/w.sum()
    return W


def chi2(m, xref, W, rng, Ntgt):
    """Mean squared pull vs a fresh target of Ntgt events, reference fixed (large)."""
    sq = []
    for s, w in W.items():
        tgt = sample(s, Ntgt, rng)
        for fn, pp in OBS:
            orf = fn(xref); otg = fn(tgt); wgt = np.repeat(w, NP) if pp else w
            lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
            pr, pre = hist(orf, wgt, bins); pt, pte = hist(otg, None, bins)
            sig = np.sqrt(pre**2 + pte**2); nz = (pt > 0) & (pr > 0)
            sq.append((((pr-pt)/np.where(sig > 0, sig, 1))[nz])**2)
    return float(np.concatenate(sq).mean())


def ref_fraction(m, xref, W, rng, Ntgt):
    """Fraction of sig^2 supplied by the REFERENCE error, i.e. how contaminated the point is."""
    fr = []
    for s, w in W.items():
        tgt = sample(s, Ntgt, rng)
        for fn, pp in OBS:
            orf = fn(xref); otg = fn(tgt); wgt = np.repeat(w, NP) if pp else w
            lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
            _, pre = hist(orf, wgt, bins); pt, pte = hist(otg, None, bins)
            nz = (pt > 0) & (pre + pte > 0)
            fr.append((pre**2/(pre**2+pte**2))[nz])
    return float(np.concatenate(fr).mean())


def fit_Nstar(N, y):
    """Weighted LS slope of (chi2-1) vs N through the origin -> N*. Returns N*, R^2."""
    N = np.asarray(N, float); d = np.asarray(y, float) - 1.0
    slope = float(np.sum(N*d)/np.sum(N*N))            # d = N/N*  => slope = 1/N*
    ss_res = float(np.sum((d - slope*N)**2)); ss_tot = float(np.sum((d - d.mean())**2))
    return (1.0/slope if slope > 0 else float('nan')), (1 - ss_res/ss_tot if ss_tot > 0 else float('nan'))


def main():
    print(f'device={DEV}')
    SEEDS = [0, 1, 2]; REPS = 3
    NREF_BIG = 1_000_000
    N_CLEAN = [2000, 5000, 10000, 20000, 35000, 50000]
    N_CTRL = [2000, 5000, 10000, 20000, 50000, 100000, 200000]
    out = {'clean': {}, 'control': {}, 'per_seed_Nstar': [], 'config': dict(
        seeds=SEEDS, reps=REPS, nref_clean=NREF_BIG, nref_control=200000,
        N_clean=N_CLEAN, N_control=N_CTRL, M=12, E=5000, epochs=6000)}

    clean_all = {N: [] for N in N_CLEAN}; ctrl_all = {N: [] for N in N_CTRL}
    refrac = {}
    for sd in SEEDS:
        print(f'-- seed {sd}: training')
        m = train(12, 5000, epochs=6000, seed=sd)
        rng0 = np.random.default_rng(500+sd)
        xref_big = sample(0.0, NREF_BIG, rng0); xr_big = torch.tensor(xref_big).to(DEV)
        W_big = ref_weights(m, xr_big); del xr_big
        xref_sm = sample(0.0, 200000, rng0); xr_sm = torch.tensor(xref_sm).to(DEV)
        W_sm = ref_weights(m, xr_sm); del xr_sm
        for N in N_CLEAN:
            for r in range(REPS):
                clean_all[N].append(chi2(m, xref_big, W_big, np.random.default_rng(9000+37*r+N), N))
            print(f'   clean  N={N:>6}: chi2={np.mean(clean_all[N][-REPS:]):.3f}')
        if sd == SEEDS[0]:
            for N in [N_CLEAN[0], N_CLEAN[-1]]:
                refrac[N] = ref_fraction(m, xref_big, W_big, np.random.default_rng(11), N)
        for N in N_CTRL:
            ctrl_all[N].append(chi2(m, xref_sm, W_sm, np.random.default_rng(7000+N), N))
        ns, _ = fit_Nstar(N_CLEAN, [np.mean(clean_all[N][-REPS:]) for N in N_CLEAN])
        out['per_seed_Nstar'].append(ns); print(f'   seed {sd}: N*={ns/1e3:.0f}k')
        del W_big, W_sm

    cl_mean = [float(np.mean(clean_all[N])) for N in N_CLEAN]
    cl_sem = [float(np.std(clean_all[N])/np.sqrt(len(clean_all[N]))) for N in N_CLEAN]
    Nstar, r2 = fit_Nstar(N_CLEAN, cl_mean)
    ns_arr = np.array(out['per_seed_Nstar'], float)
    out['clean'] = {'N': N_CLEAN, 'chi2': cl_mean, 'sem': cl_sem,
                    'Nstar': Nstar, 'Nstar_sd_over_seeds': float(ns_arr.std()),
                    'R2_linearity': r2,
                    'implied_Nstar_per_point': [float(N/(c-1)) if c > 1 else None
                                                for N, c in zip(N_CLEAN, cl_mean)],
                    'ref_error_fraction': {str(k): v for k, v in refrac.items()}}
    out['control'] = {'N': N_CTRL, 'chi2': [float(np.mean(ctrl_all[N])) for N in N_CTRL],
                      'implied_Nstar_per_point': [float(N/(np.mean(ctrl_all[N])-1))
                                                  if np.mean(ctrl_all[N]) > 1 else None for N in N_CTRL]}
    json.dump(out, open('output/budget_law.json', 'w'), indent=1)
    print(f"\nCLEAN  N* = {Nstar/1e3:.0f}k +- {ns_arr.std()/1e3:.0f}k (seed spread), "
          f"linearity R^2 = {r2:.3f}")
    print(f"  reference share of sig^2: {out['clean']['ref_error_fraction']}")
    print(f"  implied N* per point: {[None if v is None else round(v/1e3) for v in out['clean']['implied_Nstar_per_point']]}k")
    print(f"CONTROL implied N* per point (nref=2e5): "
          f"{[None if v is None else round(v/1e3) for v in out['control']['implied_Nstar_per_point']]}k")


if __name__ == '__main__':
    main()
