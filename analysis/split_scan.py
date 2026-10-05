#!/usr/bin/env python3
"""Jesse's split test: at FIXED total training statistics M x E = 60000 events, scan the
split between number of parameter points M and events per point E, from (12, 5000) all the
way to (60000, 1): one event per theta. The conditional network f(Phi,s)=a(Phi).b(s) pools
statistics across theta, so closure should be roughly independent of the split. Width toy
(true f known); theta values drawn uniformly in the box; every batch mixes many thetas
(each label-1 event carries its own theta; the paired reference events reuse the same
thetas, the standard parameterized-classifier construction). Scored as chi2/ndf at 5x10^4
validation events per held-out point, the same ruler as the K-scan.
Writes output/split_scan.json."""
import numpy as np, torch, torch.nn as nn, json
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NP_ = 10
TOTAL = 60000
SPLITS = [(12, 5000), (60, 1000), (600, 100), (6000, 10), (60000, 1)]


def sample(s, n, rng):
    return rng.normal(0.0, np.exp(s), (n, NP_)).astype(np.float32)


class Cond(nn.Module):
    def __init__(s, L=24, h=64, K=12):
        super().__init__()
        s.phi = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, L))
        s.A = nn.Sequential(nn.Linear(L, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
        s.B = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
    def emb(s, x):
        b = x.shape[0]
        return s.phi(x.reshape(b * NP_, 1)).reshape(b, NP_, -1).sum(1)
    def at(s, E, th): return (s.A(E) * s.B(th.reshape(-1, 1))).sum(-1)


def hist(o, w, bins):
    bw = np.diff(bins); W = w.sum() if w is not None else len(o)
    wv = w if w is not None else np.ones(len(o))
    h, _ = np.histogram(o, bins=bins, weights=wv); h2, _ = np.histogram(o, bins=bins, weights=wv ** 2)
    return h / (W * bw), np.sqrt(h2) / (W * bw)


def run_split(M, E, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ss = np.sort(rng.uniform(-0.25, 0.25, M)).astype(np.float32) if M > 12 else \
         np.linspace(-0.25, 0.25, M).astype(np.float32)
    X = np.stack([sample(s, E, rng) for s in ss]) if E > 1 else \
        np.stack([sample(s, 1, rng) for s in ss])                      # (M, E, NP)
    X = torch.tensor(X.reshape(M * E, NP_)).to(DEV)
    TH = torch.tensor(np.repeat(ss, E)).to(DEV)                        # theta of each target event
    xref = sample(0.0, 200000, rng); xr = torch.tensor(xref).to(DEV)
    m = Cond().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000, eta_min=2e-5)
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    for ep in range(8000):
        it = torch.randint(M * E, (1024,), generator=g)
        ir = torch.randint(200000, (1024,), generator=g)
        th = torch.cat([TH[it], TH[it]])                               # refs reuse the target thetas
        loss = bce(m.at(m.emb(torch.cat([xr[ir], X[it]])), th), lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    # closure at held-out widths, 5e4 validation events per point
    sq = []
    erng = np.random.default_rng(500 + seed)
    xe = sample(0.0, 100000, erng); xet = torch.tensor(xe).to(DEV)
    for s in [x for x in np.linspace(-0.2, 0.2, 7) if abs(x) > 0.02]:
        with torch.no_grad():
            f = m.at(m.emb(xet), torch.full((100000,), float(s), device=DEV)).cpu().numpy()
        w = np.exp(f - f.max()); w /= w.sum()
        tgt = sample(s, 50000, erng)
        for fn, pp in [(lambda X: X.reshape(-1), True), (lambda X: (X ** 2).mean(1), False)]:
            orf = fn(xe); otg = fn(tgt); wgt = np.repeat(w, NP_) if pp else w
            lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
            pr, pre = hist(orf, wgt, bins); pt, pte = hist(otg, None, bins)
            sig = np.sqrt(pre ** 2 + pte ** 2); nz = (pt > 0) & (pr > 0)
            sq.append((((pr - pt) / np.where(sig > 0, sig, 1))[nz]) ** 2)
    return float(np.concatenate(sq).mean())


def main():
    print(f'device={DEV}, total training events fixed at {TOTAL}')
    out = {}
    for M, E in SPLITS:
        assert M * E == TOTAL
        vals = [run_split(M, E, sd) for sd in (0, 1)]
        out[f'{M}x{E}'] = dict(M=M, E=E, chi2=float(np.mean(vals)), spread=float(np.std(vals)))
        print(f'M={M:6d} thetas x E={E:5d} events each: chi2/ndf = {np.mean(vals):.2f} +- {np.std(vals):.2f}')
    json.dump(out, open('output/split_scan.json', 'w'), indent=1)
    print('SPLIT SCAN DONE')


if __name__ == '__main__':
    main()
