#!/usr/bin/env python3
"""Review item 3 of 10 Nov 2026: what the photon-free observables are, measured. Reads only.

For the reference sample of the seventeen-parameter mixture (the refreshed bundle MIXGEOV3S9_ref.npz
of the run folder against the original bundle of the published run, the same events) and for
held-out runs of the strict test, per generator:
  fallback   events with a beam photon (beam_photons.py, rule v3) whose stored, photon-inclusive
             event shape was kept, because the rebuild from the stored features missed it by more
             than 1e-2 (photon_free_data/README.md): a beam photon and the refreshed value equal to
             the original one
  truncated  events with more particles than the 100 stored (the hardest by energy)
  kaons      the stored masses inside the window -0.36 < log10(m/GeV) < -0.20 of the kaon fraction
Effects:
  fallback   the closure widths (r2_ladder.pulls and width_of, copied below) at held-out points,
             with the fallback events given stored - (with - without), the rebuilt photon
             contribution, on both sides and in the same bins; and the means at the central point
  truncation the charged multiplicity at the central point, as stored (the hardest 100) and with
             the dropped particles counted charged in the proportion of the ten softest stored ones
             (estimate) or all charged (bound)

    AUD_WORK=<folder> python audit_obs_defs.py [rid ...]      writes <folder>/audit_obs_defs.json
<folder>/code holds nb_core.py (photon_free_data/step3_rule_v3) and the release's gentune with
models/MIX17_head.npz; beam_photons.py is the run folder's (md5 ea7f82ed).
"""
import os, sys, json, time
import numpy as np

WORK = os.environ['AUD_WORK']
RUN = os.environ.get('AUD_RUN', '/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code')
OLD = os.environ.get('AUD_OLD', '/mnt/beegfs/bassi/reweighting/maxent_code')
RELREF = os.environ.get('AUD_RELREF', '/mnt/beegfs/bassi/reweighting/lmx_runs/release_ref/reference_MIX17.npz')
HELD_NB = f'{OLD}/data_stagePURE17_v3_h400_nb'
HELD_ORIG = f'{OLD}/data_stagePURE17_v3_h400'
os.environ.setdefault('NB_GEN', '/mnt/beegfs/bassi/reweighting/gen')
sys.path.insert(0, RUN)
sys.path.insert(0, f'{WORK}/code')
import beam_photons as BP                       # noqa: E402  rule v3
import nb_core as C                             # noqa: E402  the rebuild of step 3
from gentune import load_head, mixture, axes    # noqa: E402
SHAPES = ('1_minus_thrust', 'B_total', 'rho_heavy')
GEN = ('sherpa', 'herwig')
NCH_EDGES = np.r_[np.arange(0, 48, 2), np.inf]  # envelope_test.py
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ---- the closure statistic, as r2_ladder.py (make_bins for the continuous observables, hist_dens,
# pulls and width_of, without the FAST_HIST path)
NBIN, MIN_COUNT = 26, 5


def make_bins(v):
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, NBIN)


def hist_dens(o, w, bins):
    bw = np.diff(bins)
    if w is None:
        n, _ = np.histogram(o, bins); N = float(n.sum())
        return n/(N*bw), n.astype(float), N
    h, _ = np.histogram(o, bins, weights=w); h2, _ = np.histogram(o, bins, weights=w**2)
    W = float(h.sum()); W2 = float(h2.sum())
    neff = np.where(h2 > 0, h**2/np.where(h2 > 0, h2, 1.0), 0.0)
    return h/(W*bw), neff, W*W/W2


def pulls(oref, w, otgt, bins):
    bw = np.diff(bins)
    pa, na, Na = hist_dens(oref, w, bins)
    pb, nb, Nb = hist_dens(otgt, None, bins)
    qa = pa*bw; qb = pb*bw
    qhat = (Na*qa + Nb*qb)/(Na + Nb)
    safe_a = np.where(na > 0, na, 1.0); safe_b = np.where(nb > 0, nb, 1.0)
    var = qhat**2*(1.0 - qhat)*(1.0/safe_a + 1.0/safe_b)/bw**2
    sig = np.sqrt(np.maximum(var, 0.0))
    keep = (sig > 0) & (na >= MIN_COUNT) & (nb >= MIN_COUNT)
    return (pa[keep] - pb[keep])/sig[keep]


def width_of(p):
    return float(np.sqrt(np.sum(p**2)/max(len(p) - 1, 1)))


# ---- per-event quantities
def masks(P, M, chunk=100000):
    out = np.zeros(M.shape, bool)
    for a in range(0, len(M), chunk):
        out[a:a + chunk] = BP.beam_photon_mask(P[a:a + chunk], M[a:a + chunk])
    return out


def rebuilt_delta(P, M, B, idx):
    """(len(idx), 3): with - without the flagged photons, rebuilt from the stored features (nb_core),
    and (len(idx), 3) the rebuilt photon-inclusive values."""
    d = np.zeros((len(idx), 3)); wv = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        f = P[i][M[i]]; b = B[i][M[i]]
        w, wo, _ = C.event_delta(f, b)
        d[k] = np.array(w) - np.array(wo); wv[k] = w
    return d, wv


def kaon_window(P, M):
    lm = P[..., 3]; inw = M & (lm > -0.36) & (lm < -0.20)
    v, c = np.unique(np.round(lm[inw].astype(np.float64), 4), return_counts=True)
    ch = {float(x): sorted(set(np.round(P[..., 5][inw][np.round(lm[inw].astype(np.float64), 4) == x], 1).tolist()))
          for x in v[:20]}
    return dict(log10m=[float(x) for x in v], counts=[int(x) for x in c],
                mass_GeV=[float(10**x) for x in v], charges=ch, n_in_window=int(inw.sum()), n_particles=int(M.sum()))


def fallback_flags(new, old, nb):
    """per shape: (beam photon and value unchanged) and, for the record, (no beam photon and changed)."""
    out = {}
    for o in SHAPES:
        a = np.asarray(new[o], np.float64); b = np.asarray(old[o], np.float64)
        out[o] = dict(fb=(nb > 0) & (a == b), changed_without=(nb == 0) & (a != b))
    return out


res = dict(script='audit_obs_defs.py', beam_photons_md5=__import__('hashlib').md5(open(BP.__file__, 'rb').read()).hexdigest())

# ---- 1. the reference sample
log('loading the reference bundles')
znew = np.load(f'{RUN}/output/models/MIXGEOV3S9_ref.npz')
zold = np.load(f'{OLD}/output/models/MIXGEOV3S9_ref.npz')
rel = np.load(RELREF)
new = {k: np.asarray(znew[k]) for k in SHAPES + ('mult_total', 'strange', 'nbaryon')}
old = {k: np.asarray(zold[k]) for k in SHAPES + ('mult_total',)}
order_ok = {o: bool(np.array_equal(np.asarray(rel[o], np.float64), np.asarray(new[o], np.float64))) for o in SHAPES}
log('release reference has the order of the refreshed bundle:', order_ok)
assert all(order_ok.values())
g = np.asarray(rel['generator'])
P = znew['particles']; M = znew['mask'].astype(bool)
log('particles', P.shape)
B = masks(P, M); nb = B.sum(1); nk = M.sum(1)
trunc = nk == M.shape[1]
nch = ((P[..., 5] != 0) & M).sum(1)
assert np.array_equal(nch, np.asarray(rel['nch']).astype(nch.dtype)), 'nch of the release is not the stored charged count'
mult_orig = np.rint(np.asarray(old['mult_total'], np.float64)).astype(np.int64)
dropped = np.where(trunc, mult_orig - nk, 0)
soft_ch = (P[:, -10:, 5] != 0).mean(1)
miss_est = dropped*soft_ch
flags = fallback_flags(new, old, nb)
ref = dict(n_events=int(len(nb)))
for gi, gn in enumerate(GEN):
    s = g == gi
    r = dict(n=int(s.sum()), with_beam_photon=int((nb[s] > 0).sum()), mean_beam_photons=float(nb[s].mean()),
             truncated=int(trunc[s].sum()), mult_orig_gt_100=int((mult_orig[s] > 100).sum()),
             mean_dropped_in_truncated=float(dropped[s & trunc].mean()) if (s & trunc).any() else 0.0,
             mean_beam_photons_stored_truncated=float(nb[s & trunc].mean()) if (s & trunc).any() else 0.0,
             mean_beam_photons_stored_whole=float(nb[s & ~trunc].mean()),
             nch_mean=float(nch[s].mean()), nch_missing_est_mean=float(miss_est[s].mean()),
             nch_missing_bound_mean=float(dropped[s].mean()),
             nch_truncated_mean=float(nch[s & trunc].mean()) if (s & trunc).any() else 0.0,
             fallback={o: int(flags[o]['fb'][s].sum()) for o in SHAPES},
             changed_without_beam_photon={o: int(flags[o]['changed_without'][s].sum()) for o in SHAPES})
    # the charged-multiplicity bins of the envelope: how many events of each bin from 40 up are truncated
    hb = np.digitize(nch[s], NCH_EDGES) - 1
    r['nch_bins_from_40'] = {f'{int(NCH_EDGES[k])}+' if not np.isfinite(NCH_EDGES[k + 1]) else f'{int(NCH_EDGES[k])}-{int(NCH_EDGES[k+1])}':
                             dict(events=int((hb == k).sum()), truncated=int(((hb == k) & trunc[s]).sum()))
                             for k in range(20, len(NCH_EDGES) - 1)}
    ref[gn] = r
    log(gn, json.dumps(r)[:600])
anyfb = np.zeros(len(nb), bool)
for o in SHAPES:
    anyfb |= flags[o]['fb']
fbi = np.where(anyfb)[0]
log('fallback events in the reference:', len(fbi))
d, wv = rebuilt_delta(P, M, B, fbi)
corr = {o: np.asarray(new[o], np.float64).copy() for o in SHAPES}
for j, o in enumerate(SHAPES):
    sel = flags[o]['fb'][fbi]
    corr[o][fbi[sel]] = corr[o][fbi[sel]] - d[sel, j]
    ref[f'fallback_delta_{o}'] = dict(n=int(sel.sum()), mean=float(d[sel, j].mean()) if sel.any() else 0.0,
                                      max_abs=float(np.abs(d[sel, j]).max()) if sel.any() else 0.0,
                                      miss_mean=float(np.abs(wv[sel, j] - np.asarray(old[o], np.float64)[fbi[sel]]).mean()) if sel.any() else 0.0)
ref['kaon_window'] = kaon_window(P[:200000], M[:200000]) | dict(note='first 200000 events (Sherpa)')
ref['kaon_window_herwig'] = kaon_window(P[-200000:], M[-200000:]) | dict(note='last 200000 events (Herwig)')
res['reference'] = ref

# ---- 2. the central point: means and charged-multiplicity bins
log('the head and the event vectors')
head = load_head(f'{WORK}/code/models/MIX17_head.npz')
A = head.pack(rel['AE'], np.float64)
th0 = axes.default('MIX17')
w0 = mixture.weights(head, A, th0)
cen = dict(theta=[float(x) for x in th0])
for o in SHAPES:
    a = mixture.mean(w0, np.asarray(new[o], np.float64), th0); b = mixture.mean(w0, corr[o], th0)
    cen[o] = dict(as_used=float(a), fallback_corrected=float(b), rel_shift=float(b/a - 1))
a = mixture.mean(w0, nch.astype(np.float64), th0)
cen['nch'] = dict(as_used=float(a), est=float(mixture.mean(w0, nch + miss_est, th0)),
                  bound=float(mixture.mean(w0, (nch + dropped).astype(np.float64), th0)))
cen['nch']['rel_shift_est'] = cen['nch']['est']/a - 1; cen['nch']['rel_shift_bound'] = cen['nch']['bound']/a - 1
cen['truncated_weighted_fraction'] = float(mixture.mean(w0, trunc.astype(np.float64), th0))
# the bins of the envelope from 40 up: the bin fraction as used and with the estimate and the bound
binz = {}
for k in range(20, len(NCH_EDGES) - 1):
    lo, hi = NCH_EDGES[k], NCH_EDGES[k + 1]
    f = lambda x: mixture.mean(w0, ((x >= lo) & (x < hi)).astype(np.float64), th0)
    binz[f'{int(lo)}+' if not np.isfinite(hi) else f'{int(lo)}-{int(hi)}'] = dict(
        as_used=float(f(nch)), est=float(f(nch + np.rint(miss_est))), bound=float(f(nch + dropped)))
cen['nch_bins'] = binz
neff = {k: float(v) for k, v in mixture.n_eff(w0).items()}
cen['n_eff'] = neff
res['central'] = cen
log('central', json.dumps(cen)[:900])

# ---- 3. held-out runs: fallback counts and the closure widths with the fallback corrected
meta = json.load(open(f'{HELD_NB}/meta.json'))
held = {int(h['rid']): h for h in meta['held']}
rids = [int(x) for x in sys.argv[1:]] or [7214, 8209, 7200, 8000]
import pandas as pd                             # noqa: E402
out = {}
for rid in rids:
    h = held[rid]; th = np.asarray(h['theta'], np.float64); gn = 'sherpa' if th[-1] == 0 else 'herwig'
    z = np.load(f'{HELD_NB}/particles_full_{rid}.npz')
    p = z['particles']; m = z['mask'].astype(bool)
    cn = pd.read_csv(f'{HELD_NB}/shapes_run_{rid}.csv'); co = pd.read_csv(f'{HELD_ORIG}/shapes_run_{rid}.csv')
    assert len(cn) == len(p) == len(co)
    bb = masks(p, m); nbh = bb.sum(1); nkh = m.sum(1)
    fl = fallback_flags({o: cn[o].values for o in SHAPES}, {o: co[o].values for o in SHAPES}, nbh)
    anyh = np.zeros(len(nbh), bool)
    for o in SHAPES:
        anyh |= fl[o]['fb']
    ih = np.where(anyh)[0]
    dh, _ = rebuilt_delta(p, m, bb, ih)
    wr = mixture.weights(head, A, th)[gn]
    rec = dict(generator=gn, n=int(len(nbh)), with_beam_photon=int((nbh > 0).sum()), truncated=int((nkh == m.shape[1]).sum()),
               counts_gt_100=int((np.asarray(z['counts']) > 100).sum()) if 'counts' in z.files else None,
               fallback={o: int(fl[o]['fb'].sum()) for o in SHAPES},
               changed_without_beam_photon={o: int(fl[o]['changed_without'].sum()) for o in SHAPES}, width={})
    for j, o in enumerate(SHAPES):
        t_used = cn[o].values.astype(np.float64); t_corr = t_used.copy()
        sel = fl[o]['fb'][ih]; t_corr[ih[sel]] -= dh[sel, j]
        r_used = np.asarray(new[o], np.float64); r_corr = corr[o]
        bins = make_bins(np.r_[r_used, t_used])
        wa = width_of(pulls(r_used, wr, t_used, bins)); wb = width_of(pulls(r_corr, wr, t_corr, bins))
        half = slice(len(t_used)//2, None)
        wa2 = width_of(pulls(r_used, wr, t_used[half], bins)); wb2 = width_of(pulls(r_corr, wr, t_corr[half], bins))
        rec['width'][o] = dict(as_used_400k=wa, corrected_400k=wb, diff_400k=wb - wa,
                               as_used_200k=wa2, corrected_200k=wb2, diff_200k=wb2 - wa2)
    out[rid] = rec
    log(rid, json.dumps(rec)[:900])
res['held'] = out
json.dump(res, open(f'{WORK}/audit_obs_defs.json', 'w'), indent=1, default=float)
log('wrote', f'{WORK}/audit_obs_defs.json')
