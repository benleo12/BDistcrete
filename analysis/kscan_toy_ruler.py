#!/usr/bin/env python3
"""Re-evaluate the TOY K-scan closure on the PAPER's ruler (r2_ladder.pulls / width_of).

Why this exists. output/kscan_toy.json was produced by kscan.toy_scan with a private
estimator that is NOT the ruler r2_ladder uses for the generator stages, so the two curves
in the cliff panel of fig_p_head.pdf are not on one axis. The private estimator

  * takes sigma^2 = sigma_ref^2 + sigma_tgt^2 with each side's OWN observed density and no
    multinomial (1-q) factor, instead of the pooled qhat^2 (1-qhat) (1/n_a + 1/n_b) of
    r2_ladder.pulls;
  * normalizes each histogram by the TOTAL weight, including the mass outside the bin range,
    instead of renormalizing to the in-range mass;
  * keeps every bin with a nonzero entry on both sides instead of requiring MIN_COUNT = 5
    effective events on both sides;
  * divides the summed square pulls by the number of bins instead of by nbins - 1;
  * bins on the 0.3/99.7 percentiles with 30 edges instead of make_bins' 0.5/99.5 with NBIN;
  * for the per-particle observable it repeats the per-EVENT weight NP times, so the ten
    correlated coordinates of one event enter as ten independent entries. r2_ladder (and
    xp_closure for the x_p spectrum) instead uses the effective number of EVENTS in a bin,
    v_e = w_e n_e(bin), so one event counts as one unit however many of its coordinates land
    in the bin.

Training here is a byte-for-byte copy of kscan.toy_scan (same seeds, same schedule, same
architecture); ONLY the estimator changes, so the before/after difference is the estimator.

Usage:  python kscan_toy_ruler.py [--exact] [--ks 1,2,4] [--seeds 2]
        KSCAN_DEV=cpu|mps chooses the device (default cpu).
--exact skips training and uses the analytic log-ratio of the toy (a perfect model), which
measures what each ruler reads at its own null.

Writes output/kscan_toy_ruler.json with the same key structure as output/kscan_toy.json
(the 'chi2' key holds width^2, so figs_pro's `ks[k]['chi2']**0.5` keeps working).
"""
import os, sys, json, time
os.environ.setdefault('LADDER_ACT', 'silu')
import numpy as np, torch, torch.nn as nn
from r2_ladder import make_bins, pulls, width_of, hist_dens, MIN_COUNT, NBIN

DEV = os.environ.get('KSCAN_DEV', 'cpu')
NP = 10
KS = [1, 2, 4, 8, 16, 32]
SVALS = [x for x in np.linspace(-0.2, 0.2, 7) if abs(x) > 0.02]
NREF, NTGT, NTRAIN_REF = 100000, 50000, 200000


# ------------------------------------------------------------------ toy model (copy of kscan)
def full(n, v, dev=None): return torch.full((n,), float(v), device=dev or DEV)
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


# ------------------------------------------------------------------ OLD estimator (kscan.py)
def old_hist(o, w, bins):
    bw = np.diff(bins); W = w.sum() if w is not None else len(o)
    wv = w if w is not None else np.ones(len(o))
    h, _ = np.histogram(o, bins=bins, weights=wv); h2, _ = np.histogram(o, bins=bins, weights=wv ** 2)
    return h / (W * bw), np.sqrt(h2) / (W * bw)


def old_sq(orf, otg, w, per_particle):
    """The squared pulls kscan.toy_scan appends to sq (one array per observable per width)."""
    wgt = np.repeat(w, NP) if per_particle else w
    lo, hi = np.percentile(np.r_[orf, otg], [0.3, 99.7]); bins = np.linspace(lo, hi, 30)
    pr, pre = old_hist(orf, wgt, bins); pt, pte = old_hist(otg, None, bins)
    sig = np.sqrt(pre ** 2 + pte ** 2); nz = (pt > 0) & (pr > 0)
    return (((pr - pt) / np.where(sig > 0, sig, 1))[nz]) ** 2


# ------------------------------------------------------------------ event-level ruler
def counts_per_event(vals, bins):
    """(nevent, nbin) matrix of how many of an event's coordinates fall in each bin.
    Same object as xp_closure.per_event_bincontents, computed with a bincount."""
    ne, npart = vals.shape
    nb = len(bins) - 1
    flatv = np.ascontiguousarray(vals).reshape(-1)
    idx = np.searchsorted(bins, flatv, side='right') - 1
    idx = np.where(flatv == bins[-1], nb - 1, idx)          # np.histogram closes the last bin
    ok = (idx >= 0) & (idx < nb)
    ev = np.repeat(np.arange(ne), npart)[ok]
    return np.bincount(ev * nb + idx[ok], minlength=ne * nb).reshape(ne, nb).astype(float)


def hist_dens_ev(counts, w, bins):
    """Event-level analogue of r2_ladder.hist_dens for a per-particle observable.

    Identical algebra to hist_dens, except that the unit whose weight is squared is the EVENT
    contribution v_e = w_e n_e(bin), not the individual particle. For events contributing at
    most one coordinate to a bin the two coincide exactly; the difference is precisely the
    correlated multiple occupancy that the particle-level count treats as independent."""
    bw = np.diff(bins)
    h = (counts * w[:, None]).sum(0)
    h2 = ((counts * w[:, None]) ** 2).sum(0)
    tot = counts.sum(1) * w                                  # in-range weighted content per event
    W = float(h.sum()); W2 = float((tot ** 2).sum())
    if W <= 0 or W2 <= 0:
        return np.zeros(len(bw)), np.zeros(len(bw)), 0.0
    neff = np.where(h2 > 0, h ** 2 / np.where(h2 > 0, h2, 1.0), 0.0)
    return h / (W * bw), neff, W * W / W2


def pulls_ev(cref, wref, ctgt, wtgt, bins, per_bin_pool=False):
    """r2_ladder.pulls with event-level effective counts. per_bin_pool=True reproduces the
    xp_closure convention for qhat (pooled with the per-bin counts) instead of the totals."""
    bw = np.diff(bins)
    pa, na, Na = hist_dens_ev(cref, wref, bins)
    pb, nb, Nb = hist_dens_ev(ctgt, wtgt, bins)
    if Na <= 0 or Nb <= 0: return np.array([]), 0
    qa = pa * bw; qb = pb * bw
    qhat = (qa * na + qb * nb) / np.maximum(na + nb, 1e-9) if per_bin_pool \
        else (Na * qa + Nb * qb) / (Na + Nb)
    safe_a = np.where(na > 0, na, 1.0); safe_b = np.where(nb > 0, nb, 1.0)
    var = qhat ** 2 * (1.0 - qhat) * (1.0 / safe_a + 1.0 / safe_b) / bw ** 2
    sig = np.sqrt(np.maximum(var, 0.0))
    keep = (sig > 0) & (na >= MIN_COUNT) & (nb >= MIN_COUNT)
    if keep.sum() == 0: return np.array([]), 0
    return (pa[keep] - pb[keep]) / sig[keep], int(keep.sum())


def selftest():
    """pulls_ev must reduce to r2_ladder.pulls when every event carries exactly one entry."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 20000); b = rng.normal(0.05, 1.02, 15000)
    bins = make_bins(np.r_[a, b], 'x')
    wa = rng.random(len(a)); wa /= wa.sum(); wb = np.full(len(b), 1.0 / len(b))
    p0, n0 = pulls(a, wa, b, bins)
    p1, n1 = pulls_ev(counts_per_event(a[:, None], bins), wa,
                      counts_per_event(b[:, None], bins), wb, bins)
    assert n0 == n1, (n0, n1)
    assert np.allclose(p0, p1, rtol=1e-9, atol=1e-9), np.abs(p0 - p1).max()
    print(f'[selftest] pulls_ev == pulls on 1-entry events ({n0} bins, '
          f'max|dp|={np.abs(p0-p1).max():.2e}); width={width_of(p0):.4f}')


# ------------------------------------------------------------------ one evaluation
def evaluate(logit_fn, sd, want_old=True):
    """logit_fn(xe_tensor, s) -> per-event log-ratio. Returns per-width, per-observable widths
    under the new ruler plus the old kscan squared pulls."""
    erng = np.random.default_rng(50 + sd)
    xe = sample(0.0, NREF, erng)
    xet = torch.tensor(xe).to(DEV)
    orf_ms = (xe ** 2).mean(1)
    res = dict(new_ms=[], new_pp=[], new_pp_binpool=[], new_pp_particle=[],
               uni_ms=[], uni_pp=[], old_sq=[], old_sq_uni=[], neff=[], nb_ms=[], nb_pp=[])
    uni = np.full(NREF, 1.0 / NREF)
    for s in SVALS:
        f = logit_fn(xet, s)
        w = np.exp(f - f.max()); w /= w.sum()
        res['neff'].append(float(1.0 / np.sum(w ** 2)))
        tgt = sample(s, NTGT, erng)
        otg_ms = (tgt ** 2).mean(1)
        wt = np.full(NTGT, 1.0 / NTGT)

        # --- paper ruler, per-event mean square (already one number per event)
        b = make_bins(np.r_[orf_ms, otg_ms], 'mean_sq')
        p, nb = pulls(orf_ms, w, otg_ms, b); res['new_ms'].append(width_of(p)); res['nb_ms'].append(nb)
        pu, _ = pulls(orf_ms, uni, otg_ms, b); res['uni_ms'].append(width_of(pu))

        # --- paper ruler, per-particle value with EVENT-level effective counts (x_p recipe)
        flat_r = xe.reshape(-1); flat_t = tgt.reshape(-1)
        b2 = make_bins(np.r_[flat_r, flat_t], 'x')
        cr = counts_per_event(xe, b2); ct = counts_per_event(tgt, b2)
        p, nb = pulls_ev(cr, w, ct, wt, b2); res['new_pp'].append(width_of(p)); res['nb_pp'].append(nb)
        p2, _ = pulls_ev(cr, w, ct, wt, b2, per_bin_pool=True); res['new_pp_binpool'].append(width_of(p2))
        # same ruler but counting each coordinate as an independent entry (isolates the
        # event-vs-particle unit choice on its own)
        p3, _ = pulls(flat_r, np.repeat(w, NP) / NP, flat_t, b2)
        res['new_pp_particle'].append(width_of(p3))
        pu, _ = pulls_ev(cr, uni, ct, wt, b2); res['uni_pp'].append(width_of(pu))

        # --- old kscan estimator on the same weights
        if want_old:
            res['old_sq'].append(old_sq(flat_r, flat_t, w, True))
            res['old_sq'].append(old_sq(orf_ms, otg_ms, w, False))
            res['old_sq_uni'].append(old_sq(flat_r, flat_t, uni, True))
            res['old_sq_uni'].append(old_sq(orf_ms, otg_ms, uni, False))
    return res


def train(K, sd):
    rng = np.random.default_rng(sd); torch.manual_seed(sd)
    grid = list(np.linspace(-0.25, 0.25, 12))
    xref = sample(0.0, NTRAIN_REF, rng); xr = torch.tensor(xref).to(DEV)
    tg = [torch.tensor(sample(s, 5000, rng)).to(DEV) for s in grid]
    m = Cond(K).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(sd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000, eta_min=2e-5)
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    t0 = time.time()
    for ep in range(8000):
        j = int(torch.randint(12, (1,), generator=g))
        ir = torch.randint(NTRAIN_REF, (1024,), generator=g); it = torch.randint(5000, (1024,), generator=g)
        loss = bce(m.at(m.emb(torch.cat([xr[ir], tg[j][it]])), full(2048, grid[j])), lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 2000 == 0:
            print(f'   K={K} sd={sd} step {ep} loss {float(loss):.4f} ({time.time()-t0:.0f}s)', flush=True)
    m.eval()
    return m


def main():
    ks = KS
    seedlist = [0, 1]
    exact = '--exact' in sys.argv
    fn = None
    for i, a in enumerate(sys.argv):
        if a == '--ks': ks = [int(x) for x in sys.argv[i + 1].split(',')]
        if a == '--seeds': seedlist = list(range(int(sys.argv[i + 1])))
        if a == '--seedlist': seedlist = [int(x) for x in sys.argv[i + 1].split(',')]
        if a == '--out': fn = sys.argv[i + 1]
    selftest()
    out = {}
    tag = 'EXACT' if exact else 'K'
    keys = ['exact'] if exact else ks
    for K in keys:
        per_seed = {}
        svs = None
        for sd in seedlist:
            if exact:
                def lf(xet, s, _xe=None):
                    x = xet.cpu().numpy()
                    return (-0.5 * (np.exp(-2 * s) - 1.0) * (x ** 2).sum(1)).astype(np.float64)
            else:
                m = train(K, sd)
                def lf(xet, s, m=m):
                    with torch.no_grad():
                        return m.at(m.emb(xet), full(len(xet), s)).cpu().numpy().astype(np.float64)
                if sd == 0:
                    sj = np.linspace(-0.25, 0.25, 40)
                    erng = np.random.default_rng(50 + sd)
                    xe0 = torch.tensor(sample(0.0, NREF, erng)).to(DEV)
                    with torch.no_grad():
                        E = m.emb(xe0[:4000])
                        M = np.stack([m.at(E, full(4000, s)).cpu().numpy() for s in sj], 1)
                    sv = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
                    svs = (sv / sv[0])[:6].tolist()
            r = evaluate(lf, sd)
            per_seed[sd] = r
            wnew = float(np.mean(r['new_ms'] + r['new_pp']))
            wold = float(np.sqrt(np.concatenate(r['old_sq']).mean()))
            print(f'{tag}={K} sd={sd}: OLD width={wold:.3f}   PAPER width={wnew:.3f} '
                  f'(mean-sq {np.mean(r["new_ms"]):.3f}, per-particle {np.mean(r["new_pp"]):.3f})', flush=True)

        def agg(fn): return [fn(per_seed[sd]) for sd in per_seed]
        widths = agg(lambda r: float(np.mean(r['new_ms'] + r['new_pp'])))
        olds = agg(lambda r: float(np.concatenate(r['old_sq']).mean()))
        rec = dict(
            chi2=float(np.mean(widths) ** 2),                       # width^2, so **0.5 -> width
            chi2_spread=float(np.std(widths) * 2 * np.mean(widths)),
            width=float(np.mean(widths)), width_spread=float(np.std(widths)),
            sv=svs, neff_min=float(np.min(agg(lambda r: float(np.min(r['neff']))))),
            neff_ref=NREF,
            chi2_uniform=float(np.mean(agg(lambda r: float(np.mean(r['uni_ms'] + r['uni_pp'])))) ** 2),
            width_uniform=float(np.mean(agg(lambda r: float(np.mean(r['uni_ms'] + r['uni_pp']))))),
            old_chi2=float(np.mean(olds)), old_width=float(np.sqrt(np.mean(olds))),
            old_chi2_uniform=float(np.mean(agg(lambda r: float(np.concatenate(r['old_sq_uni']).mean())))),
            width_mean_square=float(np.mean(agg(lambda r: float(np.mean(r['new_ms']))))),
            width_per_particle=float(np.mean(agg(lambda r: float(np.mean(r['new_pp']))))),
            width_per_particle_binpool=float(np.mean(agg(lambda r: float(np.mean(r['new_pp_binpool']))))),
            width_per_particle_particle_units=float(np.mean(agg(lambda r: float(np.mean(r['new_pp_particle']))))),
            nbins_mean_square=float(np.mean(agg(lambda r: float(np.mean(r['nb_ms']))))),
            nbins_per_particle=float(np.mean(agg(lambda r: float(np.mean(r['nb_pp']))))),
            per_seed_width=widths, per_seed_old_chi2=olds, seeds=list(per_seed),
            per_seed_width_mean_square=agg(lambda r: float(np.mean(r['new_ms']))),
            per_seed_width_per_particle=agg(lambda r: float(np.mean(r['new_pp']))),
            per_seed_width_pp_particle_units=agg(lambda r: float(np.mean(r['new_pp_particle']))),
            per_seed_width_uniform=agg(lambda r: float(np.mean(r['uni_ms'] + r['uni_pp']))),
            per_seed_old_chi2_uniform=agg(lambda r: float(np.concatenate(r['old_sq_uni']).mean())),
            per_seed_neff_min=agg(lambda r: float(np.min(r['neff']))),
            per_width_mean_square=agg(lambda r: [float(x) for x in r['new_ms']])[0],
            per_width_per_particle=agg(lambda r: [float(x) for x in r['new_pp']])[0],
            svals=[float(s) for s in SVALS],
        )
        out[str(K)] = rec
        print(f'{tag}={K}: PAPER width={rec["width"]:.3f} +- {rec["width_spread"]:.3f}   '
              f'(old {rec["old_width"]:.3f})', flush=True)
        path = fn or ('output/kscan_toy_ruler_exact.json' if exact else 'output/kscan_toy_ruler.json')
        json.dump(out, open(path, 'w'), indent=1)
    print('KSCAN TOY RULER DONE ->', path)


if __name__ == '__main__':
    main()
