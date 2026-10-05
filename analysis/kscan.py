#!/usr/bin/env python3
"""K-scan: closure and effective rank vs the inner-product dimension K, at fixed
everything else, on the width TOY (true rank 2). Expect K=1 to fail, K>=2 flat.
For each K: train a small ensemble, evaluate held-out closure (match width), and the
normalised singular values of f over events x theta grid (how many modes the fit uses).
Writes output/kscan_toy.json.

The Sherpa arm of this scan (formerly sherpa_scan) has been REMOVED. It was superseded by
r3_rank.py -> output/rank_scan.json, which runs the same scan on Stages A and B through
r2_ladder's pulls/width_of. The removed version scored with stageA_union_bilinear.pull_width
(no in-range renormalization, independent-side error quadrature, and bins kept when EITHER
side was populated) and then took np.std() of the pulls about their sample mean, which
Sec. 4 of the paper explicitly forbids because it deletes the coherent offset a
normalization error produces. Nothing read its output/kscan_sherpa.json.

NOTE on the estimator below: the closure computed here is NOT the paper's ruler. See
kscan_toy_ruler.py, which retrains these same models and re-scores them with
r2_ladder.pulls/width_of at the event level (output/kscan_toy_ruler.json). That is the file
the published rank figure uses."""
import numpy as np, torch, torch.nn as nn, json
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
KS = [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------- toy (width)
def toy_scan():
    NP = 10
    def full(n, v): return torch.full((n,), float(v), device=DEV)
    def sample(s, n, rng): return rng.normal(0.0, np.exp(s), (n, NP)).astype(np.float32)

    class Cond(nn.Module):
        def __init__(s, K):
            super().__init__()
            L, h = 24, 64
            s.phi = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, L))
            s.A = nn.Sequential(nn.Linear(L, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
            s.B = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, K))
        def emb(s, x):
            b = x.shape[0]
            return s.phi(x.reshape(b * NP, 1)).reshape(b, NP, -1).sum(1)
        def at(s, E, th): return (s.A(E) * s.B(th.reshape(-1, 1))).sum(-1)

    def hist(o, w, bins):
        bw = np.diff(bins); W = w.sum() if w is not None else len(o)
        wv = w if w is not None else np.ones(len(o))
        h, _ = np.histogram(o, bins=bins, weights=wv); h2, _ = np.histogram(o, bins=bins, weights=wv ** 2)
        return h / (W * bw), np.sqrt(h2) / (W * bw)

    out = {}
    for K in KS:
        vals = []
        svs = None
        neffs = []; squ = []
        for sd in range(2):
            rng = np.random.default_rng(sd); torch.manual_seed(sd)
            grid = list(np.linspace(-0.25, 0.25, 12))
            xref = sample(0.0, 200000, rng); xr = torch.tensor(xref).to(DEV)
            tg = [torch.tensor(sample(s, 5000, rng)).to(DEV) for s in grid]
            m = Cond(K).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
            g = torch.Generator().manual_seed(sd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000, eta_min=2e-5)
            bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
            for ep in range(8000):
                j = int(torch.randint(12, (1,), generator=g))
                ir = torch.randint(200000, (1024,), generator=g); it = torch.randint(5000, (1024,), generator=g)
                loss = bce(m.at(m.emb(torch.cat([xr[ir], tg[j][it]])), full(2048, grid[j])), lab)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            m.eval()
            # closure over held-out widths
            sq = []
            erng = np.random.default_rng(50 + sd)
            xe = sample(0.0, 100000, erng); xet = torch.tensor(xe).to(DEV)
            for s in [x for x in np.linspace(-0.2, 0.2, 7) if abs(x) > 0.02]:
                with torch.no_grad():
                    f = m.at(m.emb(xet), full(100000, s)).cpu().numpy()
                w = np.exp(f - f.max()); w /= w.sum()
                neffs.append(float(1.0/np.sum(w**2)))
                tgt = sample(s, 50000, erng)
                for fn, pp in [(lambda X: X.reshape(-1), True), (lambda X: (X ** 2).mean(1), False)]:
                    orf = fn(xe); otg = fn(tgt); wgt = np.repeat(w, NP) if pp else w
                    lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
                    pr, pre = hist(orf, wgt, bins); pt, pte = hist(otg, None, bins)
                    sig = np.sqrt(pre ** 2 + pte ** 2); nz = (pt > 0) & (pr > 0)
                    sq.append((((pr - pt) / np.where(sig > 0, sig, 1))[nz]) ** 2)
                    uw = np.ones(len(orf))/len(orf)
                    pu, pue = hist(orf, uw, bins)
                    sigu = np.sqrt(pue ** 2 + pte ** 2)
                    squ.append((((pu - pt) / np.where(sigu > 0, sigu, 1))[nz]) ** 2)
            vals.append(float(np.concatenate(sq).mean()))
            if sd == 0:
                sj = np.linspace(-0.25, 0.25, 40)
                with torch.no_grad():
                    E = m.emb(xet[:4000])
                    M = np.stack([m.at(E, full(4000, s)).cpu().numpy() for s in sj], 1)
                sv = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False); svs = (sv / sv[0])[:6].tolist()
        out[str(K)] = dict(chi2=float(np.mean(vals)), chi2_spread=float(np.std(vals)), sv=svs,
                           neff_min=float(np.min(neffs)), neff_ref=200000,
                           chi2_uniform=float(np.concatenate(squ).mean()))
        print(f'TOY   K={K:3d}: chi2/ndf={out[str(K)]["chi2"]:.2f} +- {out[str(K)]["chi2_spread"]:.2f}  sv={np.round(svs[:4],4)}')
    json.dump(out, open('output/kscan_toy.json', 'w'), indent=1)


if __name__ == '__main__':
    # the toy is the only scan here; `kscan.py` and `kscan.py toy` are the same run
    toy_scan()
    print('KSCAN DONE')
