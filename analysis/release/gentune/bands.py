"""The pieces of the paper's bands that do not depend on where the events come from. Pure numpy.

    from gentune import load_head, bands, mixture

    head = load_head('models/MIX17_head.npz')
    A = head.pack(AE, np.float64)                    # the events' a(Phi), converted once
    nodes = bands.lhs_nodes(head)                    # the paper's 2000 points, the full box, seed 11
    obs = bands.observables(F, M)                    # per-event observables, beam photons left out
    O = np.column_stack(list(obs.values()))
    means = np.array([mixture.mean(mixture.weights(head, A, t), O, t) for t in nodes])
    band = means.std(0, ddof=1)                      # the generator band of every observable

The generator band of an observable is the standard deviation of its prediction over the points of
lhs_nodes(), a flat prior on the whole box, with the fraction xi of MIX17 uniform on [0, 1]. For
MIX17 the prediction at a point is the paper's mixture of the two generators, (1 - xi) <O>_S +
xi <O>_H (gentune.mixture), and with a calculation imposed, it is imposed on each generator alone
before the mixing (mixture.impose). For a single-generator head, E or F, it is head.weights(A, t) @ O.

The statistical error of a weighted mean is mean_and_error(), and mixture.mean_and_error() for
the mixture, as in the paper: the standard deviation of the prediction over resamples of the
events, at linear order. For the generators alone its square is
sum_i w_i^2 (O_i - <O>)^2, with the weights normalized to one. With the calculation imposed, the
paper solves for the multipliers again on every resample of the events, which takes out of O its
projection on the constraint functions: pass imposed=(inwin, F, info) of the impose() that made
the weights for that error. Without it, the error is the one of the generators alone.
resample_counts() and resampled_sums() give a bootstrap over the events, which says how much of a
band is the statistics of the sample.

The theory band of a prediction with the calculation imposed comes from its displacements under
the variations of the calculation: theory_band() for one generator, by the paper's rule, which its
docstring states, and theory_band_mix() for the mixture, the same rule applied at each xi to the
displacement (1 - xi) d_S + xi d_H of the mixed prediction, its square averaged over xi flat on
[0, 1].

Two sets of per-event observables, under the same names:

  observables(particles, mask)    from the per-particle features gentune builds, so for your own
                                  HepMC sample or the 2000 events of a self-test bundle
  reference_observables(ref)      from the deposit reference_MIX17.npz (REFERENCE.md), the
                                  observables the paper itself used

Both leave out the photons radiated by the beams (gentune.beam_photons), as the paper does in every
observable; the networks still read every particle. They differ in two ways. 1 - T from the
features takes its axis from the iteration the features are built with (features.thrust_axis),
which ends in a local maximum in about 4 percent of events and so puts the mean of 1 - T above that
of the exact axis the paper uses (generators/README.md), and it sees at most the 100 hardest
particles. n and n_baryon from the features count those particles too, where the reference file
counts the full event.

The paper's thrust distribution is DELPHI's, in all twenty of its bins, PAPER_THRUST_BINS. Every
distribution of the paper is the fraction of all events in each bin over its width, with no
renormalization over a range of bins, and histogram() bins any observable this way.
"""
import hashlib
import warnings
import numpy as np

from .maxent import SCALE_PAIRS, SCHEME, DISPERSIVE

SQRT_S = 91.2          # GeV, the energy of every training sample

# The observables of observables(), in the order it returns them.
OBSERVABLES = ('1-T', 'n_ch', 'n', 'n_baryon')

# The same observables in reference_MIX17.npz, by the key the file stores them under, and the two
# event shapes the file adds. reference_observables() returns them in this order.
REFERENCE_KEYS = {'1-T': '1_minus_thrust', 'B_T': 'B_total', 'rho_H': 'rho_heavy',
                  'n_ch': 'nch', 'n': 'mult_total', 'n_baryon': 'nbaryon'}

# The sha256 of the bytes of 1_minus_thrust in the deposit of an earlier draft, whose event shapes
# and multiplicity counted the beam photons. reference_observables() warns when it reads that array.
EARLIER_DEPOSIT_TAU_SHA256 = 'dbd38e5652c614284dc5258bb178c24d9ac6f096e494441d355a3cac22c530b6'

# DELPHI's thrust distribution at 91.2 GeV in tau = 1 - T, the twenty bins the paper shows:
# 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.30 0.35 0.40 0.50.
# They hold every event of the paper's 1 - T, which lies in [0, 0.5), and histogram() over them gives
# the fraction of all events in each bin over its width, as the paper does. The edges are kept as the
# paper's code reads them from the Rivet reference table (DELPHI_1996_S3430090, d11-x01-y02, the
# measurement from charged plus neutral particles, whose edges are those of the charged-particle
# column d11-x01-y01), rounding included, so that every event falls in the paper's bin.
PAPER_THRUST_BINS = (
    np.array([0.0, 0.009999999999999998, 0.02, 0.030000000000000002, 0.04, 0.05,
              0.060000000000000005, 0.06999999999999999, 0.08, 0.09, 0.1,
              0.12000000000000001, 0.13999999999999999, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]),
    np.array([0.01, 0.02, 0.030000000000000002, 0.04, 0.049999999999999996, 0.06, 0.07, 0.08,
              0.09000000000000001, 0.1, 0.12, 0.14, 0.16, 0.18000000000000002,
              0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.5]))


def lhs_nodes(head, n=2000, seed=11):
    """The paper's points: a Latin hypercube on [-1, 1] in every parameter, mapped onto the whole
    box of the head, so for MIX17 the fraction is uniform on [0, 1]. It is written as in
    analysis/three_bands.py with BOX=full, so n=2000 and seed=11 give the paper's 2000 points to
    the last digit."""
    rng = np.random.default_rng(seed)
    U = (np.argsort(rng.random((n, head.nt)), axis=0) + rng.random((n, head.nt)))/n*2 - 1
    return head.centre + head.norm[:, 1]*U


def _thrust_axis(P3, iters=20):
    """features.thrust_axis for many events at once: P3 (E, P, 3) momenta, zero rows for the
    particles left out. Starts from the hardest particle and replaces the axis by the normalized
    sum of sign(p.n) p until it stops moving, with the stopping rules of features.thrust_axis."""
    mags = np.linalg.norm(P3, axis=2)
    r = np.arange(len(P3))
    j = np.argmax(mags, axis=1)
    n = P3[r, j]/(mags[r, j] + 1e-12)[:, None]
    run = np.ones(len(P3), bool)
    for _ in range(iters):
        a = np.flatnonzero(run)
        if len(a) == 0:
            break
        p = P3[a]
        new = np.einsum('ep,epk->ek', np.sign(np.einsum('epk,ek->ep', p, n[a])), p)
        nn = np.linalg.norm(new, axis=1)
        small = nn < 1e-12
        new = new/np.where(small, 1.0, nn)[:, None]
        stop = small | ((new*n[a]).sum(1) > 1 - 1e-12)
        n[a[~stop]] = new[~stop]
        run[a[stop]] = False
    return n


def observables(particles, mask, beam_photons='remove', chunk=20000):
    """Per-event observables from the features [z, cos_theta, phi, log10_mass, baryon, charge] of
    features.features_from_events, as a dict in the order of OBSERVABLES:

      1-T        1 - sum |p.n| / sum |p| over the particles other than the beam photons, with |p|
                 from the energy fraction at sqrt(s) = 91.2 GeV and the mass, and n the axis the
                 features are built on in an event without a beam photon, and otherwise the axis
                 the same iteration (features.thrust_axis) finds for the event without them
      n_ch       charged particles
      n          particles other than the beam photons
      n_baryon   baryons and antibaryons

    every one over the stored particles, at most the 100 hardest of the event. The beam photons
    are those of gentune.beam_photons, the paper's rule. beam_photons='keep' counts them in 1 - T
    and n, as an earlier draft did, and gives numbers that differ from the paper's."""
    if beam_photons not in ('remove', 'keep'):
        raise ValueError(f"beam_photons={beam_photons!r}: 'remove' (the paper) or 'keep'")
    m = np.asarray(mask, bool)
    particles = np.asarray(particles)
    z, cos, phi, lm = (np.asarray(particles[..., c], np.float64) for c in (0, 1, 2, 3))
    p = np.where(m, np.sqrt(np.maximum((z*SQRT_S)**2 - 10**(2*lm), 0.0)), 0.0)
    out = {'n_ch': (m & (particles[..., 5] != 0)).sum(1).astype(float),
           'n_baryon': (m & (particles[..., 4] != 0)).sum(1).astype(float)}
    if beam_photons == 'keep':
        out['1-T'] = 1 - (p*np.abs(cos)).sum(1)/np.maximum(p.sum(1), 1e-12)
        out['n'] = m.sum(1).astype(float)
        return {k: out[k] for k in OBSERVABLES}
    from .beam_photons import beam_photon_mask      # here, so that the module also runs as a script
    beam = beam_photon_mask(particles, m)
    p = np.where(beam, 0.0, p)
    num = (p*np.abs(cos)).sum(1)                   # along the axis of the features
    redo = np.flatnonzero(beam.any(1))
    for s in range(0, len(redo), chunk):
        i = redo[s:s + chunk]
        sin = np.sqrt(np.maximum(1.0 - cos[i]**2, 0.0))
        P3 = p[i, :, None]*np.stack([sin*np.cos(phi[i]), sin*np.sin(phi[i]), cos[i]], -1)
        num[i] = np.abs(np.einsum('epk,ek->ep', P3, _thrust_axis(P3))).sum(1)
    out['1-T'] = 1 - num/np.maximum(p.sum(1), 1e-12)
    out['n'] = (m & ~beam).sum(1).astype(float)
    return {k: out[k] for k in OBSERVABLES}


def reference_observables(ref):
    """The paper's per-event observables from the deposit reference_MIX17.npz, opened with np.load,
    as float64 arrays in a dict in the order of REFERENCE_KEYS: 1-T (exact thrust axis), B_T,
    rho_H, n_ch (among the 100 hardest particles, as the paper), n and n_baryon (the full event).
    The beam photons are left out of 1-T, B_T, rho_H and n, as in the paper. The deposit of an
    earlier draft counted them, and a warning says so when the file is that one."""
    tau = np.ascontiguousarray(ref['1_minus_thrust'])
    if hashlib.sha256(tau.tobytes()).hexdigest() == EARLIER_DEPOSIT_TAU_SHA256:
        warnings.warn('this reference_MIX17.npz is the deposit of an earlier draft: its 1 - T, B_T, '
                      'rho_H and n count the photons radiated by the beams, which the paper leaves '
                      'out, so predictions from it differ from the paper\'s. REFERENCE.md says '
                      'where the current deposit is.', stacklevel=2)
    return {k: np.asarray(ref[key], np.float64) for k, key in REFERENCE_KEYS.items()}


def mean_and_error(w, O, mask=None, imposed=None):
    """(mean, statistical error) of the weighted mean of O, one value per event or one column per
    observable, as in the paper (the standard deviation over resamples of the events, at linear
    order). The weights are normalized here.

    For weights without a calculation the error is sqrt(sum_i w_i^2 (O_i - <O>)^2). For weights
    from maxent.impose() or Targets.impose(), pass imposed=(inwin, F, info) of that call: the
    multipliers are then solved again on every resample, at linear order, so O_i - <O> loses its
    projection on the constraint functions, and the error is the paper's for a prediction with
    the calculation imposed. With mask, the mean is over the masked events, normalized there,
    such as the mean of 1 - T inside the window of the calculation (mask = inwin); the per-event
    term is then (O_i - <O>) Theta_i / P, with P the weight of the masked events. For the mixture
    of MIX17, gentune.mixture.mean_and_error()."""
    w = np.asarray(w, np.float64)
    w = w/w.sum()
    O = np.asarray(O, np.float64)
    if mask is None and imposed is None:
        mean = w @ O
        return mean, np.sqrt((w**2) @ (O - mean)**2)
    X = O.reshape(len(O), -1)
    if len(w) != len(X):
        raise ValueError(f'{len(w)} weights for {len(X)} rows of O')
    if mask is None:
        mean = w @ X
    else:
        wm = np.where(np.asarray(mask, bool), w, 0.0)
        mean = (wm/wm.sum()) @ X
    resp = None if imposed is None else [_response(w, imposed)]
    e = _stat_error([w], [1.0], [mean], lambda j0, j1: X[:, j0:j1], X.shape[1], mask, resp)
    return (float(mean[0]), float(e[0])) if O.ndim == 1 else (mean, e)


def _response(w, imposed):
    """maxent._Response for the imposed weights w and the (inwin, F, info) that made them."""
    from .maxent import _Response
    return _Response(w, imposed)


def _stat_error(ws, coefs, means, cols, ncols, mask=None, responses=None, factors=None, chunk=None):
    """The statistical error of sum_g coefs[g] factors[g] <X>_g, sqrt(sum_i u_i^2) per column with
    u_i = sum_g coefs[g] factors[g] w_g,i r_g,i, for ncols columns of per-event values X that
    cols(j0, j1) gives chunk columns at a time, (N, j1 - j0). ws[g] are the weights of generator g
    normalized over all N events and means[g] its means of the columns, over the masked events
    with mask. r_g is X - means[g], or (X - means[g]) Theta / P_g with mask, minus its projection
    on the constraint functions where responses[g] is given (weights with a calculation imposed).
    factors[g] is a number or one number per column."""
    m = None if mask is None else np.asarray(mask, bool)
    chunk = ncols if chunk is None else chunk
    err = np.empty(ncols)
    for j0 in range(0, ncols, chunk):
        j1 = min(j0 + chunk, ncols)
        X = cols(j0, j1)
        u = 0.0
        for g, (w, a) in enumerate(zip(ws, coefs)):
            mg = np.asarray(means[g], np.float64)[j0:j1]
            r = X - mg if m is None else np.where(m[:, None], X - mg, 0.0)/w[m].sum()
            if responses is not None and responses[g] is not None:
                r = responses[g].remove(r)[0]
            f = 1.0 if factors is None else \
                np.broadcast_to(np.asarray(factors[g], np.float64), (ncols,))[j0:j1]
            u = u + (a*f)*w[:, None]*r
        err[j0:j1] = np.sqrt(np.sum(u**2, 0))
    return err


def bin_index(x, lo, hi):
    """The bin of every event, -1 outside all of them: bin b holds lo[b] <= x < lo[b + 1], and the
    last bin lo[-1] <= x < hi[-1]. lo and hi are the bin edges, as in PAPER_THRUST_BINS; for plain
    edges e pass e[:-1] and e[1:]."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    b = np.searchsorted(np.r_[lo, hi[-1]], np.asarray(x, np.float64), side='right') - 1
    return np.where((b >= 0) & (b < len(lo)), b, -1)


def histogram(b, w, width, over_bins=False, imposed=None):
    """(value, statistical error) of every bin of a distribution: the weighted fraction of all
    events that falls in bin b, over the width of bin b, as every distribution of the paper. b is
    bin_index() of the events, w their weights, width = hi - lo. The error is that of
    mean_and_error() for the per-event value (in bin b)/(width of b): for weights from
    maxent.impose(), pass imposed=(inwin, F, info) of that call for the error with the
    calculation imposed. over_bins=True takes the fraction of the events in the bins instead, a
    distribution normalized over the bins given, which the paper does not use. For the mixture of
    MIX17, gentune.mixture.histogram()."""
    return _histogram(b, [w], [1.0], width, over_bins, None if imposed is None else [imposed])


def _histogram(b, ws, coef, width, over_bins=False, imposed=None, factors=None):
    """histogram() of sum_g coef[g] factors[g] (the histogram of the weights ws[g]), with the
    statistical error from the per-event terms of all of them together, sqrt(sum_i u_i^2),
    u_i = sum_g coef[g] factors[g] w_g,i (x_i - h_g), x_i = (in bin b)/(width of b) and h_g the
    histogram of generator g. imposed, one (inwin, F, info) or None per generator, takes the
    projection on the constraint functions out of x_i - h_g (mean_and_error())."""
    width = np.asarray(width, np.float64)
    nb = len(width)
    b = np.asarray(b)
    use = (b >= 0) if over_bins else np.ones(len(b), bool)
    ok = b >= 0
    W = []
    for w in ws:
        w = np.where(use, np.asarray(w, np.float64), 0.0)
        W.append(w/w.sum())
    H = [np.bincount(b[ok], weights=w[ok], minlength=nb)/width for w in W]
    if imposed is not None or factors is not None:
        # every event's term written out, a few bins at a time
        full = [np.asarray(w, np.float64)/np.sum(np.asarray(w, np.float64)) for w in ws]
        resp = None if imposed is None else \
            [None if imp is None else _response(w, imp) for w, imp in zip(full, imposed)]
        fac = None if factors is None else \
            [np.broadcast_to(np.asarray(f, np.float64), (nb,)) for f in factors]
        value = sum(a*(h if fac is None else fac[g]*h) for g, (a, h) in enumerate(zip(coef, H)))
        rows = np.flatnonzero(ok)

        def cols(j0, j1):
            X = np.zeros((len(b), j1 - j0))
            sel = rows[(b[rows] >= j0) & (b[rows] < j1)]
            X[sel, b[sel] - j0] = 1.0/width[b[sel]]
            return X
        err = _stat_error(full, coef, H, cols, nb, use if over_bins else None, resp, fac, chunk=4)
        return value, err
    c = sum(a*w for a, w in zip(coef, W))                     # the combined weight of each event
    value = sum(a*h for a, h in zip(coef, H))
    s_cc = np.bincount(b[ok], weights=c[ok]**2, minlength=nb)
    s_cd = sum(a*h*np.bincount(b[ok], weights=(c*w)[ok], minlength=nb)
               for a, h, w in zip(coef, H, W))
    s_dd = sum(coef[g]*coef[k]*H[g]*H[k]*float(W[g] @ W[k])
               for g in range(len(W)) for k in range(len(W)))
    var = s_cc/width**2 - 2*s_cd/width + s_dd
    return value, np.sqrt(np.maximum(var, 0.0))


def resample_counts(n, nboot, seed=0):
    """(nboot, n) uint8: how often each of n events appears in each of nboot resamples of the n
    events drawn with replacement, the bootstrap. A weighted mean of a resample is then
    (counts * w) @ O / (counts @ w), with the weights of the full sample."""
    rng = np.random.default_rng(seed)
    C = np.empty((nboot, n), np.uint8)
    for r in range(nboot):
        C[r] = np.minimum(np.bincount(rng.integers(0, n, n), minlength=n), 255)
    return C


def resampled_sums(C, X, chunk=1 << 16):
    """C @ X for counts C of resample_counts() and an (n, k) array X, in chunks of events, so that
    the floating-point copy of C stays small for a sample of a million events."""
    X = np.asarray(X, np.float64).reshape(C.shape[1], -1)
    out = np.zeros((len(C), X.shape[1]))
    for s in range(0, C.shape[1], chunk):
        out += C[:, s:s + chunk].astype(np.float64) @ X[s:s + chunk]
    return out


# -- the theory band ----------------------------------------------------------------------------
def theory_band(d, up=True, pairs=SCALE_PAIRS, scheme=SCHEME, dispersive=DISPERSIVE):
    """The paper's theory band of one generator alone: sigma_+ (up) or sigma_- from the signed
    displacements d = {label: value or array} of the prediction under the variations of the
    calculation (each imposed in place of the central one, minus the central prediction). Each
    scale pair counts once, with the larger of its two one-sided displacements, the scheme once,
    and the variations of the dispersive model as their mean square. The labels default to those
    of the paper's calculation, maxent.SCALE_PAIRS, maxent.SCHEME and maxent.DISPERSIVE."""
    f = (lambda v: np.maximum(v, 0.0)) if up else (lambda v: np.maximum(-v, 0.0))
    g = lambda lab: f(np.asarray(d[lab], np.float64))
    tot = sum(np.maximum(g(a), g(b))**2 for a, b in pairs) + g(scheme)**2
    return np.sqrt(tot + sum(g(lab)**2 for lab in dispersive)/len(dispersive))


def theory_band_mix(dS, dH, up=True, pairs=SCALE_PAIRS, scheme=SCHEME, dispersive=DISPERSIVE,
                    lo=0.0, hi=1.0):
    """The theory band of the mixture: theory_band() applied at each xi to the displacement
    (1 - xi) d_S + xi d_H of the mixed prediction, and its square averaged over xi flat on
    [lo, hi], as the paper does. dS and dH are the displacements of theory_band() for Sherpa and
    Herwig, each with the calculation imposed on that generator alone. The square is a sum of
    squares of functions linear in xi between the points where a displacement changes sign or
    the two of a scale pair cross, so Simpson's rule between those points gives the average
    exactly. Returns sigma_+ (up) or sigma_-, in the shape of the displacements."""
    labs = list(dict.fromkeys([l for p in pairs for l in p] + [scheme] + list(dispersive)))
    shape = np.shape(np.asarray(dS[scheme], np.float64))
    S = {l: np.asarray(dS[l], np.float64).ravel() for l in labs}
    H = {l: np.asarray(dH[l], np.float64).ravel() for l in labs}
    f = (lambda v: max(v, 0.0)) if up else (lambda v: max(-v, 0.0))
    out = np.empty(len(S[scheme]))
    for i in range(len(out)):
        s, h = {l: S[l][i] for l in labs}, {l: H[l][i] for l in labs}

        def sig2(x):
            dl = {l: (1 - x)*s[l] + x*h[l] for l in labs}
            t = sum(max(f(dl[a]), f(dl[b]))**2 for a, b in pairs) + f(dl[scheme])**2
            return t + sum(f(dl[l])**2 for l in dispersive)/len(dispersive)
        if hi <= lo:
            out[i] = sig2(lo)
            continue
        kinks = {lo, hi}
        for a0, a1 in [(s[l], h[l]) for l in labs] + [(s[a] - s[b], h[a] - h[b]) for a, b in pairs]:
            if a0 != a1:                                   # (1 - x) a0 + x a1 = 0
                x = a0/(a0 - a1)
                if lo < x < hi:
                    kinks.add(x)
        xs = sorted(kinks)
        out[i] = sum((x1 - x0)/6*(sig2(x0) + 4*sig2(0.5*(x0 + x1)) + sig2(x1))
                     for x0, x1 in zip(xs[:-1], xs[1:]))/(hi - lo)
    band = np.sqrt(np.maximum(out, 0.0)).reshape(shape)
    return float(band) if band.ndim == 0 else band
