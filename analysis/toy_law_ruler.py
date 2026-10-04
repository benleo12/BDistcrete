#!/usr/bin/env python3
"""Re-score the toy subsection's REMAINING numbers on the paper's ruler (r2_ladder.pulls /
width_of) at the event level, so the whole of Sec. 5.1 sits on one estimator.

Two studies are covered, both of which currently use the private kscan-style estimator
(independent-side error quadrature, no in-range renormalization, no five-effective-event
rule, ndf = nbins, per-particle entries counted as independent):

  --mode split   split_scan.py  -> the dense-coverage comparison at fixed M*E = 60000
  --mode law     budget_law.py  -> the accuracy law chi2/ndf = 1 + N_tgt/N*

Training in both modes is a byte-for-byte copy of the original script's training (same seeds,
same schedule, same architecture); only the estimator changes.

The accuracy law is LINEAR IN chi2/ndf, not in the width, so every law quantity here is
reported in chi2/ndf. On the paper's ruler chi2/ndf is the pooled sum of squared pulls over
the sum of (nbins - 1), the square of the width. Both the old and the new number are emitted
side by side for every point.

Usage:
  python toy_law_ruler.py --mode split --M 12 --E 5000 --seed 0 --out FILE
  python toy_law_ruler.py --mode law --seed 0 --out FILE
KSCAN_DEV=cpu|mps chooses the device (default cpu).
"""
import os, sys, json, time
os.environ.setdefault('LADDER_ACT', 'silu')
import numpy as np, torch, torch.nn as nn
from r2_ladder import make_bins, pulls, width_of, MIN_COUNT
from kscan_toy_ruler import counts_per_event, hist_dens_ev, pulls_ev, SVALS

DEV = os.environ.get('KSCAN_DEV', 'cpu')
NP = 10
CHUNK = 200_000


def full(n, v): return torch.full((n,), float(v), device=DEV)
def sample(s, n, rng): return rng.normal(0.0, np.exp(s), (n, NP)).astype(np.float32)


class Cond(nn.Module):
    """Identical to split_scan.Cond and budget_law.Cond (L=24, h=64, K=12)."""
    def __init__(s, L=24, h=64, K=12):
        super().__init__()
        s.phi = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, L))
        s.A = nn.Sequential(nn.Linear(L, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
        s.B = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
    def emb(s, x):
        b = x.shape[0]
        return s.phi(x.reshape(b * NP, 1)).reshape(b, NP, -1).sum(1)
    def at(s, E, th): return (s.A(E) * s.B(th.reshape(-1, 1))).sum(-1)
    def forward(s, x, th): return s.at(s.emb(x), th)


# ---------------------------------------------------------------- estimators
def old_hist(o, w, bins):
    bw = np.diff(bins); W = w.sum() if w is not None else len(o)
    wv = w if w is not None else np.ones(len(o))
    h, _ = np.histogram(o, bins=bins, weights=wv); h2, _ = np.histogram(o, bins=bins, weights=wv ** 2)
    return h / (W * bw), np.sqrt(h2) / (W * bw)


def old_sq(orf, otg, w, per_particle):
    wgt = np.repeat(w, NP) if per_particle else w
    lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
    pr, pre = old_hist(orf, wgt, bins); pt, pte = old_hist(otg, None, bins)
    sig = np.sqrt(pre ** 2 + pte ** 2); nz = (pt > 0) & (pr > 0)
    return (((pr - pt) / np.where(sig > 0, sig, 1))[nz]) ** 2


def ev_stats(vals2d, w, bins, chunk=CHUNK):
    """Chunked event-level (density, per-bin effective count, in-range effective total).

    Identical algebra to kscan_toy_ruler.hist_dens_ev; chunked over events only, because the
    million-event reference of the clean arm would otherwise need a (10^6 x nbin) matrix and
    its square simultaneously. The pooling is per event, so chunking changes nothing."""
    bw = np.diff(bins); nb = len(bw)
    h = np.zeros(nb); h2 = np.zeros(nb); tot2 = 0.0
    for i in range(0, len(vals2d), chunk):
        c = counts_per_event(vals2d[i:i + chunk], bins)
        wc = c * w[i:i + chunk, None]
        h += wc.sum(0); h2 += (wc ** 2).sum(0)
        tot2 += float(((c.sum(1) * w[i:i + chunk]) ** 2).sum())
    W = float(h.sum())
    if W <= 0 or tot2 <= 0: return np.zeros(nb), np.zeros(nb), 0.0
    neff = np.where(h2 > 0, h ** 2 / np.where(h2 > 0, h2, 1.0), 0.0)
    return h / (W * bw), neff, W * W / tot2


def pulls_from(pa, na, Na, pb, nb_, Nb, bins):
    """r2_ladder.pulls, given both sides' densities and effective counts."""
    bw = np.diff(bins)
    if Na <= 0 or Nb <= 0: return np.array([]), 0
    qa, qb = pa * bw, pb * bw
    qhat = (Na * qa + Nb * qb) / (Na + Nb)
    sa = np.where(na > 0, na, 1.0); sb = np.where(nb_ > 0, nb_, 1.0)
    var = qhat ** 2 * (1.0 - qhat) * (1.0 / sa + 1.0 / sb) / bw ** 2
    sig = np.sqrt(np.maximum(var, 0.0))
    keep = (sig > 0) & (na >= MIN_COUNT) & (nb_ >= MIN_COUNT)
    if keep.sum() == 0: return np.array([]), 0
    return (pa[keep] - pb[keep]) / sig[keep], int(keep.sum())


def selftest():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (4000, NP)); b = rng.normal(0, 1.05, (2500, NP))
    bins = make_bins(np.r_[a.reshape(-1), b.reshape(-1)], 'x')
    wa = rng.random(len(a)); wa /= wa.sum(); wb = np.full(len(b), 1.0 / len(b))
    p0, n0 = pulls_ev(counts_per_event(a, bins), wa, counts_per_event(b, bins), wb, bins)
    pa, na, Na = ev_stats(a, wa, bins, chunk=997)
    pb, nb_, Nb = ev_stats(b, wb, bins, chunk=997)
    p1, n1 = pulls_from(pa, na, Na, pb, nb_, Nb, bins)
    assert n0 == n1 and np.allclose(p0, p1, rtol=1e-9, atol=1e-9), np.abs(p0 - p1).max()
    print(f'[selftest] chunked ev_stats == kscan_toy_ruler.pulls_ev ({n0} bins, '
          f'max|dp|={np.abs(p0-p1).max():.2e})', flush=True)


def score(xe, w, tgt):
    """One (reference, weights, target) comparison.

    Returns the pooled chi2/ndf on the PAPER's ruler (sum of squared pulls over sum of
    nbins-1, over the two observables), the per-observable widths, and the old kscan
    pooled chi2/ndf on the same weights."""
    res = dict(p2=0.0, ndf=0, w_pp=None, w_ms=None, nb_pp=0, nb_ms=0)
    # per-particle value, event-level effective counts (the x_p recipe)
    flat_r = xe.reshape(-1); flat_t = tgt.reshape(-1)
    b2 = make_bins(np.r_[flat_r, flat_t], 'x')
    pa, na, Na = ev_stats(xe, w, b2)
    wt = np.full(len(tgt), 1.0 / len(tgt))
    pb, nb_, Nb = ev_stats(tgt, wt, b2)
    p, nbk = pulls_from(pa, na, Na, pb, nb_, Nb, b2)
    res['p2'] += float(np.sum(p ** 2)); res['ndf'] += max(nbk - 1, 1)
    res['w_pp'] = width_of(p); res['nb_pp'] = nbk
    # per-event mean square (already one number per event)
    orf = (xe ** 2).mean(1); otg = (tgt ** 2).mean(1)
    b1 = make_bins(np.r_[orf, otg], 'mean_sq')
    p, nbk = pulls(orf, w, otg, b1)
    res['p2'] += float(np.sum(p ** 2)); res['ndf'] += max(nbk - 1, 1)
    res['w_ms'] = width_of(p); res['nb_ms'] = nbk
    res['old_sq'] = [old_sq(flat_r, flat_t, w, True), old_sq(orf, otg, w, False)]
    return res


def combine(rs):
    """Pooled chi2/ndf over a list of per-(s) score() results, both estimators."""
    p2 = sum(r['p2'] for r in rs); ndf = sum(r['ndf'] for r in rs)
    old = np.concatenate([x for r in rs for x in r['old_sq']])
    return dict(chi2_paper=p2 / max(ndf, 1), ndf_paper=ndf,
                chi2_old=float(old.mean()), nbins_old=int(len(old)),
                width_paper=float(np.mean([r['w_pp'] for r in rs] + [r['w_ms'] for r in rs])),
                width_pp=float(np.mean([r['w_pp'] for r in rs])),
                width_ms=float(np.mean([r['w_ms'] for r in rs])),
                nb_pp=float(np.mean([r['nb_pp'] for r in rs])),
                nb_ms=float(np.mean([r['nb_ms'] for r in rs])))


# ---------------------------------------------------------------- mode: split (split_scan.py)
def run_split(M, E, seed):
    """Training copied verbatim from split_scan.run_split."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ss = np.sort(rng.uniform(-0.25, 0.25, M)).astype(np.float32) if M > 12 else \
        np.linspace(-0.25, 0.25, M).astype(np.float32)
    X = np.stack([sample(s, E, rng) for s in ss]) if E > 1 else \
        np.stack([sample(s, 1, rng) for s in ss])
    X = torch.tensor(X.reshape(M * E, NP)).to(DEV)
    TH = torch.tensor(np.repeat(ss, E)).to(DEV)
    xref = sample(0.0, 200000, rng); xr = torch.tensor(xref).to(DEV)
    m = Cond().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000, eta_min=2e-5)
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    t0 = time.time()
    for ep in range(8000):
        it = torch.randint(M * E, (1024,), generator=g)
        ir = torch.randint(200000, (1024,), generator=g)
        th = torch.cat([TH[it], TH[it]])
        loss = bce(m.at(m.emb(torch.cat([xr[ir], X[it]])), th), lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 2000 == 0: print(f'   M={M} E={E} sd={seed} step {ep} ({time.time()-t0:.0f}s)', flush=True)
    m.eval()
    # evaluation: same samples and same rng stream as split_scan
    erng = np.random.default_rng(500 + seed)
    xe = sample(0.0, 100000, erng); xet = torch.tensor(xe).to(DEV)
    rs = []
    for s in SVALS:
        with torch.no_grad():
            f = m.at(m.emb(xet), full(100000, s)).cpu().numpy()
        w = np.exp(f - f.max()); w /= w.sum()
        tgt = sample(s, 50000, erng)
        rs.append(score(xe, w, tgt))
    return combine(rs)


# ---------------------------------------------------------------- mode: law (budget_law.py)
def train_law(M, E, epochs, seed):
    """Training copied verbatim from budget_law.train."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    grid = list(np.linspace(-0.25, 0.25, M)); xref = sample(0.0, 200000, rng)
    xr = torch.tensor(xref).to(DEV)
    tg = [torch.tensor(sample(s, E, rng)).to(DEV) for s in grid]
    m = Cond().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-5)
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    t0 = time.time()
    for ep in range(epochs):
        j = int(torch.randint(M, (1,), generator=g))
        ir = torch.randint(200000, (1024,), generator=g); it = torch.randint(E, (1024,), generator=g)
        loss = bce(m(torch.cat([xr[ir], tg[j][it]]), full(2048, grid[j])), lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 2000 == 0: print(f'   law sd={seed} step {ep} ({time.time()-t0:.0f}s)', flush=True)
    m.eval(); return m


def ref_weights(m, xref):
    W = {}
    for s in SVALS:
        out = []
        with torch.no_grad():
            for i in range(0, len(xref), CHUNK):
                xb = torch.tensor(xref[i:i + CHUNK]).to(DEV)
                out.append(m.at(m.emb(xb), full(len(xb), s)).cpu().numpy())
        f = np.concatenate(out)
        w = np.exp(f - f.max()); W[s] = w / w.sum()
    return W


def run_law(seed):
    N_CLEAN = [2000, 5000, 10000, 20000, 35000, 50000]
    N_CTRL = [2000, 5000, 10000, 20000, 50000, 100000, 200000]
    REPS = 3
    m = train_law(12, 5000, 6000, seed)
    rng0 = np.random.default_rng(500 + seed)
    xref_big = sample(0.0, 1_000_000, rng0)
    W_big = ref_weights(m, xref_big)
    xref_sm = sample(0.0, 200000, rng0)
    W_sm = ref_weights(m, xref_sm)
    out = dict(seed=seed, clean={}, control={})
    for N in N_CLEAN:
        acc = []
        for r in range(REPS):
            rng = np.random.default_rng(9000 + 37 * r + N)
            acc.append(combine([score(xref_big, W_big[s], sample(s, N, rng)) for s in SVALS]))
        out['clean'][str(N)] = dict(
            chi2_paper=float(np.mean([a['chi2_paper'] for a in acc])),
            chi2_old=float(np.mean([a['chi2_old'] for a in acc])),
            chi2_paper_reps=[a['chi2_paper'] for a in acc],
            chi2_old_reps=[a['chi2_old'] for a in acc],
            width_paper=float(np.mean([a['width_paper'] for a in acc])),
            ndf_paper=acc[0]['ndf_paper'], nbins_old=acc[0]['nbins_old'])
        print(f'   clean N={N:>6}: paper chi2/ndf={out["clean"][str(N)]["chi2_paper"]:.3f} '
              f'old={out["clean"][str(N)]["chi2_old"]:.3f}', flush=True)
    for N in N_CTRL:
        rng = np.random.default_rng(7000 + N)
        a = combine([score(xref_sm, W_sm[s], sample(s, N, rng)) for s in SVALS])
        out['control'][str(N)] = dict(chi2_paper=a['chi2_paper'], chi2_old=a['chi2_old'],
                                      ndf_paper=a['ndf_paper'], nbins_old=a['nbins_old'])
        print(f'   ctrl  N={N:>6}: paper chi2/ndf={a["chi2_paper"]:.3f} old={a["chi2_old"]:.3f}',
              flush=True)
    return out


def run_null(seed):
    """The estimator's OWN null on this observable pair, as a function of N_tgt.

    The accuracy law chi2/ndf = 1 + N_tgt/N* forces the intercept to 1, which is only correct
    if the estimator reads exactly 1 for a PERFECT model. Here the weights are the analytic
    log-ratio of the toy (epsilon = 0 by construction), so whatever this reads IS the
    intercept, and N* can be fit to (chi2 - c) rather than to (chi2 - 1). No training."""
    N_CLEAN = [2000, 5000, 10000, 20000, 35000, 50000]
    REPS = 3
    rng0 = np.random.default_rng(500 + seed)
    xref = sample(0.0, 1_000_000, rng0)
    s2 = (xref ** 2).sum(1)
    W = {}
    for s in SVALS:
        f = -0.5 * (np.exp(-2 * s) - 1.0) * s2
        w = np.exp(f - f.max()); W[s] = w / w.sum()
    out = dict(seed=seed, exact=True, clean={})
    for N in N_CLEAN:
        acc = []
        for r in range(REPS):
            rng = np.random.default_rng(9000 + 37 * r + N)
            acc.append(combine([score(xref, W[s], sample(s, N, rng)) for s in SVALS]))
        out['clean'][str(N)] = dict(
            chi2_paper=float(np.mean([a['chi2_paper'] for a in acc])),
            chi2_old=float(np.mean([a['chi2_old'] for a in acc])),
            chi2_paper_reps=[a['chi2_paper'] for a in acc],
            chi2_old_reps=[a['chi2_old'] for a in acc])
        print(f'   null N={N:>6}: paper chi2/ndf={out["clean"][str(N)]["chi2_paper"]:.3f} '
              f'old={out["clean"][str(N)]["chi2_old"]:.3f}', flush=True)
    return out


def main():
    mode = 'split'; M, E, seed, fn = 12, 5000, 0, None
    for i, a in enumerate(sys.argv):
        if a == '--mode': mode = sys.argv[i + 1]
        if a == '--M': M = int(sys.argv[i + 1])
        if a == '--E': E = int(sys.argv[i + 1])
        if a == '--seed': seed = int(sys.argv[i + 1])
        if a == '--out': fn = sys.argv[i + 1]
    selftest()
    if mode == 'split':
        r = run_split(M, E, seed); r.update(M=M, E=E, seed=seed)
        print(f'SPLIT M={M} E={E} sd={seed}: paper chi2/ndf={r["chi2_paper"]:.3f} '
              f'(width {r["width_paper"]:.3f})  old chi2/ndf={r["chi2_old"]:.3f}', flush=True)
        fn = fn or f'output/_law/split_M{M}_E{E}_s{seed}.json'
    elif mode == 'null':
        r = run_null(seed)
        fn = fn or f'output/_law/null_s{seed}.json'
    else:
        r = run_law(seed)
        fn = fn or f'output/_law/law_s{seed}.json'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(r, open(fn, 'w'), indent=1)
    print('->', fn)


if __name__ == '__main__':
    main()
