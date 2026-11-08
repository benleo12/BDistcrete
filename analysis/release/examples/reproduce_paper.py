#!/usr/bin/env python3
"""The paper's predictions and bands, from the deposited reference sample.

    python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz --points 0
    python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz

The first command checks the central point alone. The second adds the generator band, the paper's
2000 points. Both need numpy alone, and the two files of the deposit (REFERENCE.md says where they
are).

This is the end-to-end check of the release. It reweights the events of reference_MIX17.npz with
the MIX17 head, imposes the NNLL+NLO thrust moments of targets_thrust_nlo.npz with gentune.maxent,
and compares every prediction and band with the paper's, which analysis/three_bands.py computed
with the paper's torch code.

A prediction at a point theta = (theta_S, theta_H, xi) is the paper's mixture of the generators,
(1 - xi) <O>_S + xi <O>_H. Sherpa's weights come from the head at xi = 0 and theta_S, Herwig's
from the head at xi = 1 and theta_H, and each is normalized over all events. The calculation is
imposed on each generator alone, and the two predictions are then mixed (gentune.mixture).

  central point    the default tunes of both generators and xi = 1/2, gentune.axes.default('MIX17').
                   Seven means before and after the calculation is imposed, and DELPHI's thrust
                   distribution in its twenty bins, each the fraction of all events in it over its
                   width. For each of them: the statistical error from the finite sample, the
                   learning band (models/MIX17_wifi.npz, averaged over xi flat on [0, 1]), and,
                   with the calculation imposed, the theory band: the paper's rule for combining
                   the variations of the calculation (gentune.bands.theory_band) applied at each
                   xi to (1 - xi) d_S + xi d_H, its square averaged over xi flat on [0, 1]. d_g is
                   how generator g moves when a variation of the calculation is imposed on it alone.
  generator band   the standard deviation of every prediction over 2000 points of a Latin hypercube
                   over the whole box, seed 11, xi uniform on [0, 1] (gentune.bands.lhs_nodes),
                   before and after the calculation is imposed at every point.

The seven means are 1 - T inside the window of the calculation (each generator's own mean there,
mixed), the fraction of events in that window, and the full means of 1 - T, B_T, rho_H, n_ch and
n_baryon. The photons radiated by the beams are left out of every observable, as in the deposit.

A measurement counts its particles in its own way. DELPHI decays K0S, Lambda and the other
hyperons (Sigma, Xi, Omega), which live less than 1 ns but long enough (c tau above 10 mm) that
our samples keep them, and OPAL also counts the neutrinos, which our samples leave out. When the
paper compares with a measurement, it multiplies each generator's prediction by a factor for that
measurement, after the calculation is imposed and before the two generators are mixed: a factor
per bin for DELPHI's distribution (c_over_a in the file), one for OPAL's <1-T> (cnu_over_a), and
none for L3's <n_ch>. The factors were measured with each generator at its default tune.
examples/data/convention_factors.json is a copy of the file the paper's code reads,
analysis/conventions/factors.json, which analysis/conventions/make_factors.py wrote. This script
gives <1-T> and DELPHI's distribution with these factors too, with all their bands, and compares
them with the paper's.

The paper's numbers are in examples/paper_numbers.json, copied from the output of three_bands.py.
--paper <three_bands_MIXGEOV3S9_lin.npz> compares with that output instead (it reads the .json next
to it too), which adds the prediction at each point, the percentiles of each band, and each
generator alone at the central point. It refuses an output made with another factors file. The
paper's numbers with the factors hold for the shipped file alone, so when --factors gives other
factors and --paper is not used, the predictions with those factors are printed and not compared.

The check fails when a number differs from the paper's by more than --tol, relative, 1e-6 by
default. What is left comes from the head, which the paper's code evaluates in float32 and gentune
in float64. In the statistical errors and learning bands with the calculation imposed, it also comes
from the paper's code, which inverts the matrix of the Newton step without rescaling it. The script
prints the largest difference of each comparison and its own run time.

--points N evaluates only the first N of the 2000 points. With --paper their predictions are
compared point by point; the band itself needs all 2000. --out writes every prediction to an npz.
"""
import argparse, json, os, sys, time
import numpy as np

# the release root, whether this file sits in examples/ or next to gentune/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, 'gentune')) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from gentune import load_head, axes, bands, mixture
from gentune.maxent import load_targets

NUMBERS = os.path.join(ROOT, 'examples', 'paper_numbers.json')
FACTORS = os.path.join(ROOT, 'examples', 'data', 'convention_factors.json')
OBS = ('tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')
NM = len(OBS)
LABEL = {'tau_win': '<1-T> in window', 'pwin': 'window fraction', 'thrust': '<1-T>',
         'B_total': '<B_T>', 'rho_heavy': '<rho_H>', 'nch': '<n_ch>', 'nbaryon': '<n_baryon>'}
POINTS, SEED = 2000, 11                    # the paper's generator band: bands.lhs_nodes(head)
# the variations of the calculation that the theory band combines (gentune.bands.theory_band)
VARIATIONS = [l for p in bands.SCALE_PAIRS for l in p] + [bands.SCHEME] + list(bands.DISPERSIVE)

# The factors of the comparisons with measurements, as analysis/conventions.py of the paper applies
# them: in every DELPHI bin the factor that decays K0S, Lambda and the other hyperons, on <1-T>
# (OPAL) the factor that also adds the neutrinos, and on <n_ch> (L3) none. Names as in
# convention_factors.json. The other means are not compared with a measurement and take none.
DELPHI_FACTOR = 'c_over_a'
MEAN_FACTOR = {'thrust': ('one_minus_thrust', 'cnu_over_a')}
CONV_OBS = tuple(MEAN_FACTOR)              # the means whose comparison takes a factor
MEASUREMENT = {'thrust': 'OPAL <1-T>'}

# What the paper gives, by the key of paper_numbers.json: (one entry per mean of OBS, per mean of
# CONV_OBS, or per bin), and where three_bands.py writes it: a field of central in its .json, of
# generator_band in its .json, the same two in its block conv (the predictions with the factors of
# the measurements), or an array of its .npz, which holds its first five bins under the same name
# with _low.
LAYOUT = {
    'central_before': ('means', 'central', 'generator_only'),
    'central_after': ('means', 'central', 'imposed'),
    'stat_before': ('means', 'central', 'statistical_generator_only'),
    'stat_after': ('means', 'central', 'statistical'),
    'learning_before': ('means', 'central', 'learning_band_generator_only'),
    'learning_after': ('means', 'central', 'learning_band'),
    'theory_up': ('means', 'central', 'theory_up'),
    'theory_down': ('means', 'central', 'theory_down'),
    'band_before': ('means', 'generator_band', 'before', 'sd'),
    'band_after': ('means', 'generator_band', 'after', 'sd'),
    'mean_before': ('means', 'generator_band', 'before', 'mean'),
    'mean_after': ('means', 'generator_band', 'after', 'mean'),
    'hist_before': ('bins', 'npz', 'h_central_before'),
    'hist_after': ('bins', 'npz', 'h_central'),
    'hist_stat_before': ('bins', 'npz', 'stat_hist_before'),
    'hist_stat_after': ('bins', 'npz', 'stat_hist'),
    'hist_learning_before': ('bins', 'npz', 'learn_hist_before'),
    'hist_learning_after': ('bins', 'npz', 'learn_hist'),
    'hist_theory_up': ('bins', 'npz', 'theory_hist_up'),
    'hist_theory_down': ('bins', 'npz', 'theory_hist_down'),
    'hist_band_before': ('bins', 'npz', 'gen_hist_before'),
    'hist_band_after': ('bins', 'npz', 'gen_hist'),
    'conv_central_before': ('conv', 'conv_central', 'generator_only'),
    'conv_central_after': ('conv', 'conv_central', 'imposed'),
    'conv_stat_before': ('conv', 'conv_central', 'statistical_generator_only'),
    'conv_stat_after': ('conv', 'conv_central', 'statistical'),
    'conv_learning_before': ('conv', 'conv_central', 'learning_band_generator_only'),
    'conv_learning_after': ('conv', 'conv_central', 'learning_band'),
    'conv_theory_up': ('conv', 'conv_central', 'theory_up'),
    'conv_theory_down': ('conv', 'conv_central', 'theory_down'),
    'conv_band_before': ('conv', 'conv_generator_band', 'before', 'sd'),
    'conv_band_after': ('conv', 'conv_generator_band', 'after', 'sd'),
    'conv_mean_before': ('conv', 'conv_generator_band', 'before', 'mean'),
    'conv_mean_after': ('conv', 'conv_generator_band', 'after', 'mean'),
    'conv_hist_before': ('bins', 'npz', 'h_central_before_conv'),
    'conv_hist_after': ('bins', 'npz', 'h_central_conv'),
    'conv_hist_stat_before': ('bins', 'npz', 'stat_hist_before_conv'),
    'conv_hist_stat_after': ('bins', 'npz', 'stat_hist_conv'),
    'conv_hist_learning_before': ('bins', 'npz', 'learn_hist_before_conv'),
    'conv_hist_learning_after': ('bins', 'npz', 'learn_hist_conv'),
    'conv_hist_theory_up': ('bins', 'npz', 'theory_hist_up_conv'),
    'conv_hist_theory_down': ('bins', 'npz', 'theory_hist_down_conv'),
    'conv_hist_band_before': ('bins', 'npz', 'gen_hist_before_conv'),
    'conv_hist_band_after': ('bins', 'npz', 'gen_hist_conv'),
}
CHECK_NAMES = {'central_before': 'before', 'central_after': 'after',
               'stat_before': 'statistical error before', 'stat_after': 'statistical error after',
               'learning_before': 'learning band before', 'learning_after': 'learning band after',
               'theory_up': 'theory band +', 'theory_down': 'theory band -'}


def names(kind):
    """The observables of the entries of a key of LAYOUT whose kind is means or conv."""
    return {'means': OBS, 'conv': CONV_OBS}[kind]


def labels(kind):
    """The labels of the entries of a key of LAYOUT, as they appear in the markers of
    paper_numbers.json: the means by name, the bins as bin00 to bin19."""
    if kind == 'bins':
        return [f'bin{b:02d}' for b in range(len(bands.PAPER_THRUST_BINS[0]))]
    return [k.lower() for k in names(kind)]


def factor_vectors(src=FACTORS):
    """The factors of the comparisons with measurements, from src, convention_factors.json or the
    dict it holds: ({generator: one factor per prediction}, the file's meta). The factors are laid
    out as Reference.predictions(): the seven means of OBS, then the twenty DELPHI bins. A mean of
    MEAN_FACTOR takes the factor of its measurement, every other mean 1."""
    if isinstance(src, dict):
        d, where = src, 'the factors'
    else:
        with open(src) as f:
            try:
                d = json.load(f)
            except ValueError as e:
                raise ValueError(f'{src}: not a JSON file ({e})') from None
        where = src
    lo, hi = bands.PAPER_THRUST_BINS
    out = {}
    for g in mixture.GENERATORS:
        if not isinstance(d, dict) or not isinstance(d.get(g), dict):
            raise ValueError(f'{where}: no factors of {g}, so it is not a factors file of '
                             f'analysis/conventions')
        try:
            T = d[g]['delphi_y02']
            tl, th, tv = (np.asarray(T[k], np.float64) for k in ('lo', 'hi', DELPHI_FACTOR))
            means = [float(d[g]['means'][MEAN_FACTOR[k][0]][MEAN_FACTOR[k][1]]) if k in MEAN_FACTOR
                     else 1.0 for k in OBS]
        except KeyError as e:
            raise ValueError(f'{where}: the factors of {g} have no field {e}, so it is not a factors '
                             f'file of analysis/conventions') from None
        except (TypeError, ValueError) as e:
            raise ValueError(f'{where}: cannot read the factors of {g} ({e}), so it is not a factors '
                             f'file of analysis/conventions') from None
        if not (tl.shape == th.shape == tv.shape == lo.shape
                and np.allclose(tl, lo, rtol=0, atol=1e-6) and np.allclose(th, hi, rtol=0, atol=1e-6)):
            raise ValueError(f'{where}: the DELPHI bins of {g} are not the twenty of '
                             f'gentune.bands.PAPER_THRUST_BINS')
        v = np.r_[means, tv]
        if not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError(f'{where}: a factor of {g} is not a positive number')
        out[g] = v
    return out, d.get('meta', {})


def shipped_factors(V):
    """True if V, from factor_vectors(), are the factors of the shipped convention_factors.json,
    for which the paper's numbers with the factors hold."""
    try:
        S = factor_vectors(FACTORS)[0]
    except (OSError, ValueError):
        return False
    return all(np.array_equal(V[g], S[g]) for g in mixture.GENERATORS)


class Reference:
    """The reference sample and what every prediction needs from it, computed once: the event
    vectors as one float64 matrix, the per-event observables, the window and moment functions of
    the calculation, and the DELPHI bin of every event."""

    def __init__(self, path, head, calc):
        r = np.load(path)
        obs = bands.reference_observables(r)
        self.tau = obs['1-T']
        # the five full means of OBS, one column each
        self.X = np.column_stack([obs[k] for k in ('1-T', 'B_T', 'rho_H', 'n_ch', 'n_baryon')])
        self.N = len(self.tau)
        # float64 once, here, so that every evaluation of the head is a plain float64 product; the
        # values are those of the float32 file, converted exactly
        self.A = head.pack(r['AE'], np.float64)
        self.inwin, self.F = calc.functions(self.tau)
        self.iw = np.flatnonzero(self.inwin)
        self.lo, self.hi = bands.PAPER_THRUST_BINS
        self.width = self.hi - self.lo
        self.b = bands.bin_index(self.tau, self.lo, self.hi)
        self.ok = self.b >= 0

    def predictions(self, w):
        """One generator's seven means of OBS and its DELPHI histogram, the fraction of all events
        in each bin over its width, for weights w that sum to one over all events."""
        win = w[self.iw]
        pw = win.sum()
        means = np.r_[win @ self.tau[self.iw]/pw, pw, w @ self.X]
        hist = np.bincount(self.b[self.ok], weights=w[self.ok], minlength=len(self.width))/self.width
        return np.r_[means, hist]

    def errors(self, w, theta, imposed=None, factors=None):
        """The statistical error of every prediction of the mixture: imposed=(inwin, F, info) of
        the mixture.impose() that made w for the error with the calculation imposed, and factors
        for the predictions with the factors of the measurements."""
        f = (lambda s: None) if factors is None else (lambda s: {g: v[s] for g, v in factors.items()})
        win = mixture.mean_and_error(w, self.tau, theta, mask=self.inwin, imposed=imposed,
                                     factors=f(0))[1]
        rest = mixture.mean_and_error(w, np.column_stack([self.inwin.astype(float), self.X]),
                                      theta, imposed=imposed, factors=f(slice(1, NM)))[1]
        hist = mixture.histogram(self.b, w, self.width, theta, imposed=imposed,
                                 factors=f(slice(NM, None)))[1]
        return np.r_[win, rest, hist]

    def learning(self, head, theta, wifi, imposed=None, factors=None):
        """The learning band of every prediction of the mixture, averaged over xi flat on [0, 1],
        with imposed and factors as for errors()."""
        f = (lambda s: None) if factors is None else (lambda s: {g: v[s] for g, v in factors.items()})
        win = mixture.learning_band(head, self.A, theta, self.tau, wifi, mask=self.inwin,
                                    imposed=imposed, factors=f(0))
        H = np.zeros((self.N, len(self.width)))         # (in the bin)/(bin width), all events
        H[self.ok, self.b[self.ok]] = 1.0/self.width[self.b[self.ok]]
        cols = np.column_stack([self.inwin.astype(float), self.X, H])
        del H
        return np.r_[win, mixture.learning_band(head, self.A, theta, cols, wifi, imposed=imposed,
                                                factors=f(slice(1, None)))]


def rel(mine, paper):
    """The largest relative difference, |mine - paper|/|paper|; where the paper's value is zero,
    |mine| over the largest |paper| of the same comparison."""
    m, p = np.atleast_1d(np.asarray(mine, np.float64)), np.atleast_1d(np.asarray(paper, np.float64))
    scale = np.where(p != 0, np.abs(p), max(float(np.abs(p).max()), 1e-300))
    return float(np.max(np.abs(m - p)/scale))


def read_numbers(path=NUMBERS):
    """The paper's numbers of paper_numbers.json in the layout of LAYOUT. Each is a number, or a
    marker {{fill:KEY}} until the file is filled from the paper's output; a marker stops here."""
    with open(path) as f:
        d = json.load(f)
    P, missing = {}, []
    for key, (kind, *_) in LAYOUT.items():
        vals = d.get(key)
        if not isinstance(vals, list) or len(vals) != len(labels(kind)):
            raise ValueError(f'{path}: {key} must hold {len(labels(kind))} numbers')
        out = []
        for v in vals:
            if isinstance(v, str) and v.strip().startswith('{{fill:'):
                missing.append(v.strip())
                out.append(np.nan)
            else:
                out.append(float(v))
        P[key] = np.array(out)
    if missing:
        raise ValueError(f'{path} does not hold the paper\'s numbers yet: {len(missing)} of them are '
                         f'still markers, such as {missing[0]}. They are copied from the output of '
                         f'analysis/three_bands.py when the release is built. Compare with that '
                         f'output instead: --paper <its three_bands_MIXGEOV3S9_lin.npz>.')
    P['points'] = P['hypercube'] = POINTS
    return P


def read_paper(path, factors=FACTORS):
    """The paper's numbers from the output of three_bands.py (its .npz and the .json next to it),
    in the layout of LAYOUT, plus the points and the predictions at each. The output's predictions
    with the factors of the measurements must come from the file factors."""
    base = path[:-4] if path.endswith(('.npz', '.json')) else path
    z = np.load(base + '.npz')
    with open(base + '.json') as f:
        j = json.load(f)
    if j.get('mix') != 'linear' or j.get('box') != 'full':
        raise ValueError(f'{base}: made with MIX={j.get("mix")} BOX={j.get("box")}, and the paper\'s '
                         f'bands are MIX=linear BOX=full')
    obs = [str(o) for o in z['obs']]
    if tuple(obs) != OBS:
        raise ValueError(f'{base}.npz holds the observables {obs}')
    lo, hi = np.r_[z['lo_low'], z['lo']], np.r_[z['hi_low'], z['hi']]
    if not (np.array_equal(lo, bands.PAPER_THRUST_BINS[0]) and
            np.array_equal(hi, bands.PAPER_THRUST_BINS[1])):
        raise ValueError(f'{base}.npz is not binned in the DELPHI bins of gentune.bands')
    # the predictions with the factors of the measurements: present, from the same file, the same
    # choice of factors as here
    rec = j.get('conventions') or {}
    if 'conv' not in j or 'h_central_conv' not in z.files or not rec:
        raise ValueError(f'{base}: holds no predictions with the factors of the measurements, which '
                         f'the paper compares with the data (made with CONVENTIONS=none?)')
    meta = factor_vectors(factors)[1]
    if rec.get('meta') != meta:
        raise ValueError(f'{base}: its predictions with the factors of the measurements were made with '
                         f'another factors file than {factors} (made '
                         f'{(rec.get("meta") or {}).get("made")}, against {meta.get("made")})')
    applied = {k: v for k, v in (rec.get('means') or {}).items() if v != 'none'}
    if rec.get('delphi') != f'delphi_y02 {DELPHI_FACTOR}' or \
            applied != {k: kind for k, (_, kind) in MEAN_FACTOR.items()}:
        raise ValueError(f'{base}: the paper\'s run applied the factors {rec.get("delphi")} and '
                         f'{rec.get("means")}, not those of this script')
    P = {}
    for key, (kind, where, *field) in LAYOUT.items():
        if where == 'npz':
            P[key] = np.r_[z[field[0] + '_low'], z[field[0]]]
            continue
        src = j['conv'][where[len('conv_'):]] if where.startswith('conv_') else j[where]
        if where.endswith('generator_band'):
            P[key] = np.array([src[field[0]][k][field[1]] for k in names(kind)])
        else:
            P[key] = np.array([src[field[0]][k] for k in names(kind)])
    for pre, g, ks in (('', j['generator_band'], OBS), ('conv_', j['conv']['generator_band'], CONV_OBS)):
        for s in ('before', 'after'):
            for q in ('p16', 'p84'):
                P[f'{pre}{q}_{s}'] = np.array([g[s][k][q] for k in ks])
    gens = j['central'].get('generators', {})
    if all(g in gens and f'h_central_{g}' in z.files for g in mixture.GENERATORS):
        # each generator alone at the central point, as three_bands.py also writes it
        hb = lambda key: np.r_[z[key + '_low'], z[key]]
        P['alone'] = {g: dict(before=np.r_[[gens[g]['generator_only'][k] for k in OBS],
                                           hb(f'h_central_before_{g}')],
                              after=np.r_[[gens[g]['imposed'][k] for k in OBS], hb(f'h_central_{g}')],
                              theory_up=np.r_[[gens[g]['theory_up'][k] for k in OBS],
                                              hb(f'theory_hist_up_{g}')],
                              theory_down=np.r_[[gens[g]['theory_down'][k] for k in OBS],
                                                hb(f'theory_hist_down_{g}')])
                      for g in mixture.GENERATORS}
    ic = [OBS.index(k) for k in CONV_OBS]
    P.update(theta=np.array(j['central']['theta']), nodes=z['nodes'],
             rows_before=z['rows_before'], rows_after=z['rows_after'],
             H_before=z['H_before'], H_after=z['H_after'],
             conv_rows_before=z['rows_before_conv'][:, ic], conv_rows_after=z['rows_after_conv'][:, ic],
             conv_H_before=z['H_before_conv'], conv_H_after=z['H_after_conv'],
             first_fitted=len(z['lo_low']), points=len(z['nodes']),
             hypercube=int(j.get('hypercube_size', j['nodes'])))
    return P


def table(title, rows, cols):
    """Print one table: rows of (label, values...), cols the column titles after the label."""
    print(f'\n{title}')
    print(f'{"":<17}' + ''.join(f'{c:>14}' for c in cols))
    for lab, *vals in rows:
        print(f'{lab:<17}' + ''.join(f'{v:>14.7g}' if isinstance(v, float) else f'{v:>14}'
                                     for v in vals))


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True, help='reference_MIX17.npz, the deposit')
    ap.add_argument('--targets', required=True, help='targets_thrust_nlo.npz, the deposit')
    ap.add_argument('--head', default=os.path.join(ROOT, 'models', 'MIX17_head.npz'),
                    help='the MIX17 head (default models/MIX17_head.npz)')
    ap.add_argument('--wifi', default=os.path.join(ROOT, 'models', 'MIX17_wifi.npz'),
                    help='the WiFi fit of the learning band')
    ap.add_argument('--factors', default=FACTORS,
                    help='the factors of the measurements (default examples/data/convention_factors.json). '
                         'The paper\'s numbers with the factors hold for that file alone: with other '
                         'factors and without --paper, the predictions with them are printed and not '
                         'compared')
    ap.add_argument('--paper', help="the paper's output three_bands_MIXGEOV3S9_lin.npz, to compare "
                                    "with instead of examples/paper_numbers.json")
    ap.add_argument('--points', type=int, default=POINTS,
                    help=f'evaluate the first N of the paper\'s {POINTS} points, 0 for none '
                         f'(default {POINTS})')
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='largest relative difference passed (default 1e-6)')
    ap.add_argument('--out', help='npz for every prediction this run makes')
    a = ap.parse_args()
    t0 = time.time()
    if not 0 <= a.points <= POINTS:
        print(f'ERROR: --points takes 0 to {POINTS}, not {a.points}')
        return 2
    try:
        V, fmeta = factor_vectors(a.factors)
        P = read_paper(a.paper, a.factors) if a.paper else read_numbers()
    except (OSError, ValueError, KeyError) as e:
        print(f'ERROR: {e}')
        return 2
    src = a.paper or os.path.relpath(NUMBERS)
    # the paper's numbers with the factors hold for the factors they were made with: with --paper,
    # the file its output records, which read_paper matched with --factors by its meta; without it,
    # the shipped file alone, so that other factors are printed and not compared
    conv_paper = bool(a.paper) or shipped_factors(V)

    head = load_head(a.head)
    calc = load_targets(a.targets)
    ref = Reference(a.reference, head, calc)
    print(f'{ref.N:,} reference events, {len(ref.iw):,} in the window [{calc.window[0]:.4g}, '
          f'{calc.window[1]:.4g}) of {len(calc.names)} moments, the MIX17 head at T = {head.T:g} '
          f'[{time.time() - t0:.0f} s]')
    print(f'the factors of the measurements from {os.path.relpath(a.factors)} (made '
          f'{fmeta.get("made", "?")}): DELPHI ' +
          ', '.join(f'{g} {V[g][NM:].min():.4f} to {V[g][NM:].max():.4f}' for g in V) + '; OPAL <1-T> ' +
          ', '.join(f'{g} {V[g][OBS.index("thrust")]:.5f}' for g in V))
    if not conv_paper:
        print(f'NOTE: these are not the factors of {os.path.relpath(FACTORS)}, for which the paper\'s '
              f'numbers hold, so the predictions with them are printed and not compared')
    nm = NM
    ic = [OBS.index(k) for k in CONV_OBS]
    checks = []

    # -- the central point -----------------------------------------------------------------------
    theta = axes.default('MIX17')
    w = mixture.weights(head, ref.A, theta)
    wc, info = mixture.impose(w, ref.inwin, ref.F, calc)
    imp = (ref.inwin, ref.F, info)
    alone0 = {g: ref.predictions(v) for g, v in w.items()}
    base = {g: ref.predictions(v) for g, v in wc.items()}
    y0, y1 = mixture.combine(alone0, theta), mixture.combine(base, theta)
    neff = mixture.n_eff(w)
    print('\ncentral point, the default tunes and xi = 1/2: ' + ', '.join(
        f'{g} N_eff {neff[g]:,.0f}, Newton {info[g]["iterations"]} steps'
        f'{"" if info[g]["converged"] else " NOT converged"}' for g in mixture.GENERATORS) +
        f' [{time.time() - t0:.0f} s]')
    s0, s1 = ref.errors(w, theta), ref.errors(wc, theta, imp)
    l0, l1 = ref.learning(head, theta, a.wifi), ref.learning(head, theta, a.wifi, imp)
    d = {g: {} for g in wc}
    for lab in VARIATIONS:
        wv, _ = mixture.impose(w, ref.inwin, ref.F, calc, which=lab)
        for g in wv:
            d[g][lab] = ref.predictions(wv[g]) - base[g]
    up = bands.theory_band_mix(d['sherpa'], d['herwig'], up=True)
    down = bands.theory_band_mix(d['sherpa'], d['herwig'], up=False)
    mine = {'central_before': y0, 'central_after': y1, 'stat_before': s0, 'stat_after': s1,
            'learning_before': l0, 'learning_after': l1, 'theory_up': up, 'theory_down': down}
    # the same with the factors of the measurements: each generator times its own factor, after the
    # calculation is imposed and before the mixing; so are its per-event terms, its derivatives and
    # its displacements
    dV = {g: {lab: V[g]*v for lab, v in d[g].items()} for g in d}
    conv = {'central_before': mixture.combine(alone0, theta, V),
            'central_after': mixture.combine(base, theta, V),
            'stat_before': ref.errors(w, theta, factors=V),
            'stat_after': ref.errors(wc, theta, imp, factors=V),
            'learning_before': ref.learning(head, theta, a.wifi, factors=V),
            'learning_after': ref.learning(head, theta, a.wifi, imp, factors=V),
            'theory_up': bands.theory_band_mix(dV['sherpa'], dV['herwig'], up=True),
            'theory_down': bands.theory_band_mix(dV['sherpa'], dV['herwig'], up=False)}
    print(f'statistical errors, learning and theory bands done, also with the factors of the '
          f'measurements [{time.time() - t0:.0f} s]')

    table('the seven means', [(LABEL[k], y0[j], P['central_before'][j], rel(y0[j], P['central_before'][j]),
                               y1[j], P['central_after'][j], rel(y1[j], P['central_after'][j]))
                              for j, k in enumerate(OBS)],
          ('before', 'paper', 'rel. diff', 'after', 'paper', 'rel. diff'))
    lo, hi = bands.PAPER_THRUST_BINS
    table('DELPHI thrust distribution, the fraction of all events in each bin over its width',
          [(f'{lo[b]:.2f} to {hi[b]:.2f}', y0[nm + b], P['hist_before'][b],
            rel(y0[nm + b], P['hist_before'][b]), y1[nm + b], P['hist_after'][b],
            rel(y1[nm + b], P['hist_after'][b])) for b in range(len(lo))],
          ('before', 'paper', 'rel. diff', 'after', 'paper', 'rel. diff'))
    table('uncertainties of the seven means at the central point',
          [(LABEL[k], s0[j], s1[j], l0[j], l1[j], up[j], down[j]) for j, k in enumerate(OBS)],
          ('stat. before', 'stat. after', 'learn. before', 'learn. after', 'theory +', 'theory -'))
    c0, c1 = conv['central_before'], conv['central_after']
    if conv_paper:
        table('with the factors of the measurements: OPAL <1-T>, and DELPHI\'s distribution as above',
              [(MEASUREMENT[k], c0[j], P['conv_central_before'][i], rel(c0[j], P['conv_central_before'][i]),
                c1[j], P['conv_central_after'][i], rel(c1[j], P['conv_central_after'][i]))
               for i, (j, k) in enumerate(zip(ic, CONV_OBS))] +
              [(f'{lo[b]:.2f} to {hi[b]:.2f}', c0[nm + b], P['conv_hist_before'][b],
                rel(c0[nm + b], P['conv_hist_before'][b]), c1[nm + b], P['conv_hist_after'][b],
                rel(c1[nm + b], P['conv_hist_after'][b])) for b in range(len(lo))],
              ('before', 'paper', 'rel. diff', 'after', 'paper', 'rel. diff'))
    else:
        table(f'with the factors of {os.path.relpath(a.factors)}, not compared: OPAL <1-T>, and '
              f'DELPHI\'s distribution as above',
              [(MEASUREMENT[k], c0[j], c1[j]) for j, k in zip(ic, CONV_OBS)] +
              [(f'{lo[b]:.2f} to {hi[b]:.2f}', c0[nm + b], c1[nm + b]) for b in range(len(lo))],
              ('before', 'after'))
    for key, v in mine.items():
        name = CHECK_NAMES[key]
        checks.append((f'central point, {name}, means', rel(v[:nm], P[key]), a.tol))
        checks.append((f'central point, {name}, distribution',
                       rel(v[nm:], P['hist_' + key.replace('central_', '')]), a.tol))
    for key, v in (conv.items() if conv_paper else ()):
        name = CHECK_NAMES[key]
        checks.append((f'with the factors, {name}, OPAL <1-T>', rel(v[ic], P['conv_' + key]), a.tol))
        checks.append((f'with the factors, {name}, DELPHI bins',
                       rel(v[nm:], P['conv_hist_' + key.replace('central_', '')]), a.tol))
    if 'theta' in P:
        checks.append(('central point, theta', float(np.max(np.abs(theta - P['theta']))), 0.0))
    if 'alone' in P:
        # each generator alone: its predictions before and after the calculation, and its theory band
        alone = {g: dict(before=alone0[g], after=base[g],
                         theory_up=bands.theory_band(d[g], up=True),
                         theory_down=bands.theory_band(d[g], up=False)) for g in w}
        for g in mixture.GENERATORS:
            for q in ('before', 'after', 'theory_up', 'theory_down'):
                checks.append((f'central point, {g} alone, {q.replace("_", " ")}',
                               rel(alone[g][q], P['alone'][g][q]), a.tol))

    # -- the generator band ----------------------------------------------------------------------
    nodes = bands.lhs_nodes(head, POINTS, SEED)[:a.points]
    Y = {(s, f): [] for s in ('before', 'after') for f in (False, True)}
    its, conv_ok = [], []
    for i, th in enumerate(nodes):
        wp = mixture.weights(head, ref.A, th)
        wpc, inf = mixture.impose(wp, ref.inwin, ref.F, calc)
        for s, ww in (('before', wp), ('after', wpc)):
            parts = {g: ref.predictions(v) for g, v in ww.items()}
            Y[s, False].append(mixture.combine(parts, th))
            Y[s, True].append(mixture.combine(parts, th, V))
        its += [inf[g]['iterations'] for g in inf]
        conv_ok += [inf[g]['converged'] for g in inf]
        if i % 100 == 0 or i == len(nodes) - 1:
            print(f'  point {i + 1}/{len(nodes)} [{time.time() - t0:.0f} s]', flush=True)
    Y = {k: np.array(v).reshape(len(nodes), len(y0)) for k, v in Y.items()}
    if len(nodes):
        print(f'Newton converged for {sum(conv_ok)} of {len(conv_ok)} generator solutions at '
              f'{len(nodes)} points, in {min(its)} to {max(its)} steps')
        if 'nodes' in P:
            n = min(len(nodes), P['points'])
            checks.append((f'the first {n} points', float(np.max(np.abs(nodes[:n] - P['nodes'][:n]))),
                           0.0))
            for s in ('before', 'after'):
                checks.append((f'{n} points {s}, means at every point',
                               rel(Y[s, False][:n, :nm], P['rows_' + s][:n]), a.tol))
                checks.append((f'{n} points {s}, distribution at every point',
                               rel(Y[s, False][:n, nm + P['first_fitted']:], P['H_' + s][:n]), a.tol))
                checks.append((f'{n} points {s} with the factors, OPAL <1-T>',
                               rel(Y[s, True][:n, ic], P['conv_rows_' + s][:n]), a.tol))
                checks.append((f'{n} points {s} with the factors, DELPHI bins',
                               rel(Y[s, True][:n, nm + P['first_fitted']:], P['conv_H_' + s][:n]),
                               a.tol))
    if len(nodes) == POINTS:
        band = {k: (v.std(0, ddof=1), v.mean(0)) for k, v in Y.items()}
        title = f'generator band, the standard deviation over the {POINTS} points'
        if P['points'] == POINTS and P['hypercube'] == POINTS:
            sd0, sd1 = band['before', False][0], band['after', False][0]
            cd0, cd1 = band['before', True][0], band['after', True][0]
            table(title, [(LABEL[k], sd0[j], P['band_before'][j], rel(sd0[j], P['band_before'][j]),
                           sd1[j], P['band_after'][j], rel(sd1[j], P['band_after'][j]))
                          for j, k in enumerate(OBS)] +
                  [(f'{MEASUREMENT[k]}*', cd0[j], P['conv_band_before'][i],
                    rel(cd0[j], P['conv_band_before'][i]), cd1[j], P['conv_band_after'][i],
                    rel(cd1[j], P['conv_band_after'][i])) if conv_paper else
                   (f'{MEASUREMENT[k]}*', cd0[j], '-', '-', cd1[j], '-', '-')
                   for i, (j, k) in enumerate(zip(ic, CONV_OBS))],
                  ('before', 'paper', 'rel. diff', 'after', 'paper', 'rel. diff'))
            print('* with the factor of the measurement' + ('' if conv_paper else
                  f' in {os.path.relpath(a.factors)}, not compared'))
            for s in ('before', 'after'):
                sd, mu = band[s, False]
                checks.append((f'generator band {s}, means', rel(sd[:nm], P['band_' + s]), a.tol))
                checks.append((f'generator band {s}, distribution', rel(sd[nm:], P['hist_band_' + s]),
                               a.tol))
                checks.append((f'generator band {s}, mean over the points', rel(mu[:nm], P['mean_' + s]),
                               a.tol))
                sdc, muc = band[s, True]
                if conv_paper:
                    checks.append((f'with the factors, generator band {s}, OPAL <1-T>',
                                   rel(sdc[ic], P['conv_band_' + s]), a.tol))
                    checks.append((f'with the factors, generator band {s}, DELPHI bins',
                                   rel(sdc[nm:], P['conv_hist_band_' + s]), a.tol))
                    checks.append((f'with the factors, generator band {s}, mean over points',
                                   rel(muc[ic], P['conv_mean_' + s]), a.tol))
                for pre, f, cols in (('', False, slice(0, nm)), ('conv_', True, ic))[:2 if conv_paper else 1]:
                    for q in ('p16', 'p84'):
                        if f'{pre}{q}_{s}' in P:
                            checks.append((f'{"with the factors, " if f else ""}generator band {s}, '
                                           f'{q} over the points',
                                           rel(np.percentile(Y[s, f][:, cols], int(q[1:]), axis=0),
                                               P[f'{pre}{q}_{s}']), a.tol))
        else:
            table(title, [(LABEL[k], band['before', False][0][j], band['after', False][0][j])
                          for j, k in enumerate(OBS)] +
                  [(f'{MEASUREMENT[k]}*', band['before', True][0][j], band['after', True][0][j])
                   for j, k in zip(ic, CONV_OBS)],
                  ('before', 'after'))
            print('* with the factor of the measurement')
            print(f'{src} holds {P["points"]} of the {POINTS} points, so the bands are not compared')
    else:
        print(f'\n{len(nodes)} of the {POINTS} points evaluated: the generator band needs all of them')

    # -- verdict ---------------------------------------------------------------------------------
    print(f'\nlargest relative difference from the paper ({src})')
    ok = True
    for name, dv, tol in checks:
        good = dv <= tol
        ok &= good
        print(f'  {name:<60}{dv:>10.1e}   {"ok" if good else "ABOVE " + format(tol, ".0e")}')
    if a.out:
        np.savez(a.out, theta=theta, obs=np.array(OBS), lo=lo, hi=hi, nodes=nodes,
                 rows_before=Y['before', False], rows_after=Y['after', False],
                 rows_before_conv=Y['before', True], rows_after_conv=Y['after', True],
                 **mine, **{'conv_' + k: v for k, v in conv.items()},
                 **{f'factors_{g}': V[g] for g in V},
                 **{f'theory_displacement_{g}': np.array([d[g][lab] for lab in VARIATIONS])
                    for g in d}, theory_labels=np.array(VARIATIONS))
        print(f'wrote {a.out}')
    print(f'{"PASS" if ok else "FAIL"}: the paper reproduced with numpy from the two deposited '
          f'files' + ('' if conv_paper else ', except the predictions with the factors, which use '
                      'other factors than the paper\'s and were not compared') +
          f' [{time.time() - t0:.0f} s]')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
