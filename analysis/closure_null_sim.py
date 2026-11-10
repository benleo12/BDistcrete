"""Null of the closure statistic of r2_ladder.py (pulls() and width_of(), copied verbatim in
logic: pooled q from the densities, multinomial (1-q) factor, each side's own effective per-bin
count, MIN_COUNT = 5 on both sides, ndf = kept bins - 1). Two samples from the same density.
Reports E[width], E[width^2], median width, against B/(B-1)."""
import json, sys
import numpy as np
from math import lgamma, exp, sqrt

MIN_COUNT = 5
OUT = sys.argv[1]
rng = np.random.default_rng(20261110)


def widths_from(pa_q, na, Na, pb_q, nb, Nb):
    """pa_q, pb_q: in-range bin fractions (density*bw); na, nb: effective per-bin counts;
    Na, Nb: effective totals. Arrays (reps, B). Same arithmetic as r2_ladder.pulls/width_of
    (bw cancels between the density difference and sigma)."""
    qhat = (Na[:, None]*pa_q + Nb[:, None]*pb_q)/(Na + Nb)[:, None]
    sa = np.where(na > 0, na, 1.0); sb = np.where(nb > 0, nb, 1.0)
    var = qhat**2*(1.0 - qhat)*(1.0/sa + 1.0/sb)
    sig = np.sqrt(np.maximum(var, 0.0))
    keep = (sig > 0) & (na >= MIN_COUNT) & (nb >= MIN_COUNT)
    z = np.where(keep, (pa_q - pb_q)/np.where(sig > 0, sig, 1.0), 0.0)
    k = keep.sum(1); ndf = np.maximum(k - 1, 1)
    return np.sqrt((z**2).sum(1)/ndf), k


def multinomial_case(p, Na, Nb, reps):
    p = np.asarray(p, float); p = p/p.sum()
    a = rng.multinomial(Na, p, size=reps).astype(float); b = rng.multinomial(Nb, p, size=reps).astype(float)
    w, k = widths_from(a/Na, a, np.full(reps, float(Na)), b/Nb, b, np.full(reps, float(Nb)))
    return w, k


def weighted_case(Bn, reps, Nref, Ntgt, shift=0.3, scale=1.3, discrete5=False):
    """Reference from g = N(0, scale^2) with exact weights f/g, target from f = N(shift, 1). Bins: 25 equal
    bins between the 0.5 and 99.5 percentiles of the combined values (make_bins, NBIN=26), or
    5 bins with the baryon-count probabilities' structure (edges at fixed quantiles of f)."""
    out_w, out_k = [], []
    for r in range(reps):
        xr = rng.normal(0, scale, Nref); xt = rng.normal(shift, 1.0, Ntgt)
        lw = -0.5*(xr - shift)**2 + 0.5*(xr/scale)**2; w = np.exp(lw - lw.max()); w /= w.sum()
        if discrete5:
            edges = np.r_[-np.inf, shift + np.array([-0.3, 0.55, 1.2, 1.9]), np.inf]   # q ~ .38,.33,.18,.08,.03
        else:
            lo, hi = np.percentile(np.r_[xr, xt], [0.5, 99.5]); edges = np.linspace(lo, hi, Bn + 1)
        h, _ = np.histogram(xr, edges, weights=w); h2, _ = np.histogram(xr, edges, weights=w**2)
        W, W2 = h.sum(), h2.sum()
        na = np.where(h2 > 0, h**2/np.where(h2 > 0, h2, 1.0), 0.0); Na = W*W/W2
        nt, _ = np.histogram(xt, edges); Nt = nt.sum()
        ww, kk = widths_from((h/W)[None], na[None], np.array([Na]), (nt/Nt)[None], nt[None].astype(float), np.array([float(Nt)]))
        out_w.append(ww[0]); out_k.append(kk[0])
    return np.array(out_w), np.array(out_k)


def ideal(B):
    """Equal-probability asymptotics: sum z^2 = B/(B-1) * chi2_{B-1}, so width = sqrt(B/(B-1)) * sqrt(chi2_k/k)."""
    k = B - 1
    c = sqrt(2.0/k)*exp(lgamma((k + 1)/2) - lgamma(k/2))
    return dict(E_w2=B/(B - 1), E_w=sqrt(B/(B - 1))*c)


def summary(w, k, B):
    return dict(B=B, reps=int(len(w)), kept_mean=float(np.mean(k)), E_width=float(np.mean(w)), se_E_width=float(np.std(w)/np.sqrt(len(w))),
                E_width2=float(np.mean(w**2)), se_E_width2=float(np.std(w**2)/np.sqrt(len(w))), median_width=float(np.median(w)),
                sd_width=float(np.std(w)), B_over_Bm1=B/(B - 1), ideal=ideal(B))


res = {}
# 1. equal-probability bins, two unweighted samples of 40000 (the review's case) and of 20000 (the calibration halves)
for B in (5, 6, 25, 61, 64, 177):
    for (Na, Nb) in ((40000, 40000), (20000, 20000)):
        w, k = multinomial_case(np.ones(B), Na, Nb, 20000)
        res[f'equal_B{B}_{Na}v{Nb}'] = summary(w, k, B)
# 2. the actual binnings of the mixture's closure figure (fig_closure_mix.npz: target densities and edges at two held points)
d = np.load(sys.argv[2], allow_pickle=True)
for rid in d['rids']:
    for o in d['obs']:
        e = d[f'{rid}__{o}__bins']; q = d[f'{rid}__{o}__dt']*np.diff(e)
        for (Na, Nb) in ((20000, 20000), (200000, 40000)):
            w, k = multinomial_case(q, Na, Nb, 20000)
            res[f'actual_{o}_{rid}_{Na}v{Nb}'] = summary(w, k, len(q)) | dict(min_q=float(q.min()))
# 3. known weights: reweighted reference against an independent target
for B in (25,):
    w, k = weighted_case(B, 3000, 300000, 40000)
    res[f'weighted_B{B}_ref300k_tgt40k'] = summary(w, k, B)
w, k = weighted_case(5, 3000, 300000, 40000, discrete5=True)
res['weighted_B5_ref300k_tgt40k'] = summary(w, k, 5)
json.dump(res, open(OUT, 'w'), indent=1)
for key, r in res.items():
    print(f"{key:40s} B={r['B']:4d} kept={r['kept_mean']:6.1f} E[w]={r['E_width']:.4f}+-{r['se_E_width']:.4f} "
          f"E[w2]={r['E_width2']:.4f}+-{r['se_E_width2']:.4f} med={r['median_width']:.4f} B/(B-1)={r['B_over_Bm1']:.4f} "
          f"ideal E[w]={r['ideal']['E_w']:.4f}")
