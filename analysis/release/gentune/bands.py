"""The pieces of the paper's bands that do not depend on where the events come from. Pure numpy.

    from gentune import load_head, bands

    head = load_head('models/MIX17_head.npz')
    nodes = bands.lhs_nodes(head)                    # the paper's 200 points, seed 11
    obs = bands.observables(F, M)                    # per-event observables from the features
    O = np.column_stack(list(obs.values()))
    means = np.array([head.weights(A, t) @ O for t in nodes])
    band = means.std(0, ddof=1)                      # the generator band of every observable

The generator band of an observable is the standard deviation of its weighted mean over the points
of lhs_nodes(), a flat prior on the box. The statistical error of one weighted mean is
mean_and_error(); its square is sum_i w_i^2 (O_i - <O>)^2 with the weights normalized to one,
which is what the paper calls the statistical error. resample_counts() and resampled_sums() give a
bootstrap over the events, which says how much of a band is the statistics of the sample.

Two sets of per-event observables, under the same names:

  observables(particles, mask)    from the per-particle features gentune builds, so for your own
                                  HepMC sample or the 2000 events of a self-test bundle
  reference_observables(ref)      from the deposit reference_MIX17.npz (REFERENCE.md), the
                                  observables the paper itself used

They differ in two ways. 1 - T from the features is read off the frame they are built in, whose
axis comes from an iteration that puts the mean about 1.2 percent above the exact axis of the
paper (generators/README.md), and it sees at most the 100 hardest particles. n and n_baryon from
the features count those particles too, where the reference file counts the full event.

The paper's thrust distribution is DELPHI's, normalized over the bins from 1 - T = 0.05 up, the
bins of PAPER_THRUST_BINS. bin_index() and histogram() bin any observable this way.
"""
import numpy as np

SQRT_S = 91.2          # GeV, the energy of every training sample

# The observables of observables(), in the order it returns them.
OBSERVABLES = ('1-T', 'n_ch', 'n', 'n_baryon')

# The same observables in reference_MIX17.npz, by the key the file stores them under, and the two
# event shapes the file adds. reference_observables() returns them in this order.
REFERENCE_KEYS = {'1-T': '1_minus_thrust', 'B_T': 'B_total', 'rho_H': 'rho_heavy',
                  'n_ch': 'nch', 'n': 'mult_total', 'n_baryon': 'nbaryon'}

# DELPHI's thrust distribution at 91.2 GeV in tau = 1 - T, the bins the paper shows and normalizes
# over: 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.30 0.35 0.40 0.50. The edges are
# kept as the paper's code reads them from the Rivet reference table (DELPHI_1996_S3430090,
# d11-x01-y01), rounding included, so that every event falls in the paper's bin.
PAPER_THRUST_BINS = (
    np.array([0.05, 0.060000000000000005, 0.06999999999999999, 0.08, 0.09, 0.1,
              0.12000000000000001, 0.13999999999999999, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]),
    np.array([0.06, 0.07, 0.08, 0.09000000000000001, 0.1, 0.12, 0.14, 0.16, 0.18000000000000002,
              0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.5]))


def lhs_nodes(head, n=200, seed=11):
    """The paper's points: a Latin hypercube on [-1, 1] in every parameter, mapped onto the box of
    the head, so for MIX17 the fraction is uniform on [0, 1]. It is written as in
    analysis/three_bands.py, so n=200 and seed=11 give the paper's 200 points to the last digit."""
    rng = np.random.default_rng(seed)
    U = (np.argsort(rng.random((n, head.nt)), axis=0) + rng.random((n, head.nt)))/n*2 - 1
    return head.centre + head.norm[:, 1]*U


def observables(particles, mask):
    """Per-event observables from the features [z, cos_theta, phi, log10_mass, baryon, charge] of
    features.features_from_events, as a dict in the order of OBSERVABLES:

      1-T        sum |p.n| / sum |p|, with n the axis the features are built on and |p| from the
                 energy fraction at sqrt(s) = 91.2 GeV and the mass
      n_ch       charged particles
      n          particles
      n_baryon   baryons and antibaryons

    every one over the stored particles, at most the 100 hardest of the event."""
    m = np.asarray(mask, bool)
    z, cos, lm = (np.asarray(particles[..., c], np.float64) for c in (0, 1, 3))
    p = np.where(m, np.sqrt(np.maximum((z*SQRT_S)**2 - 10**(2*lm), 0.0)), 0.0)
    return {
        '1-T': 1 - (p*np.abs(cos)).sum(1)/np.maximum(p.sum(1), 1e-12),
        'n_ch': (m & (particles[..., 5] != 0)).sum(1).astype(float),
        'n': m.sum(1).astype(float),
        'n_baryon': (m & (particles[..., 4] != 0)).sum(1).astype(float),
    }


def reference_observables(ref):
    """The paper's per-event observables from the deposit reference_MIX17.npz, opened with np.load,
    as float64 arrays in a dict in the order of REFERENCE_KEYS: 1-T (exact thrust axis), B_T,
    rho_H, n_ch (among the 100 hardest particles, as the paper), n and n_baryon (the full event)."""
    return {k: np.asarray(ref[key], np.float64) for k, key in REFERENCE_KEYS.items()}


def mean_and_error(w, O):
    """(mean, statistical error) of the weighted mean of O, one value per event or one column per
    observable. The weights are normalized here, and the error is sqrt(sum_i w_i^2 (O_i - <O>)^2),
    the paper's statistical error of a reweighted mean."""
    w = np.asarray(w, np.float64)
    w = w/w.sum()
    O = np.asarray(O, np.float64)
    mean = w @ O
    return mean, np.sqrt((w**2) @ (O - mean)**2)


def bin_index(x, lo, hi):
    """The bin of every event, -1 outside all of them: bin b holds lo[b] <= x < lo[b + 1], and the
    last bin lo[-1] <= x < hi[-1]. lo and hi are the bin edges, as in PAPER_THRUST_BINS; for plain
    edges e pass e[:-1] and e[1:]."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    b = np.searchsorted(np.r_[lo, hi[-1]], np.asarray(x, np.float64), side='right') - 1
    return np.where((b >= 0) & (b < len(lo)), b, -1)


def histogram(b, w, width):
    """(value, statistical error) of every bin of a distribution normalized over the bins given:
    the weighted fraction of the events in the bins that falls in bin b, over the width of bin b.
    b is bin_index() of the events, w their weights, width = hi - lo. The error is that of
    mean_and_error() for the per-event value (in bin b)/(width of b) over the events in any bin,
    as in the paper."""
    width = np.asarray(width, np.float64)
    b = np.asarray(b)
    ok = b >= 0
    w = np.asarray(w, np.float64)[ok]
    s1 = np.bincount(b[ok], weights=w, minlength=len(width))
    s2 = np.bincount(b[ok], weights=w**2, minlength=len(width))
    W = s1.sum()
    h = s1/(width*W)
    var = (s2/width**2 - 2*h*s2/width + h**2*s2.sum())/W**2
    return h, np.sqrt(np.maximum(var, 0.0))


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
