#!/usr/bin/env python3
r"""The generator band of an observable: its spread over the whole parameter box.

    python examples/generator_band.py --stage MIX17 --reference
    python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
    python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz \
        --targets targets_thrust_nlo.npz
    python examples/generator_band.py --stage F --hepmc examples/data/herwig_default.hepmc.gz --from herwig-default

The last line reads the Herwig example file. Your own file goes in its place.

This is the generator band of the paper, a flat prior on the box. The prior is sampled by 2000
points of a Latin hypercube over the whole box with seed 11, the fraction xi of MIX17 uniform on
[0, 1], built as analysis/three_bands.py builds them, so they are the paper's points to the last
digit (gentune.bands.lhs_nodes). The sample is reweighted to every point, and the band of an
observable is the standard deviation of its prediction over the points. For MIX17 the prediction
at a point is the paper's: (1 - xi) <O>_S + xi <O>_H, Sherpa's weights from the head at xi = 0 and
Herwig's at xi = 1 (gentune.mixture). --bins does the same for every bin of a distribution, the
fraction of all events in the bin over its width, and --bins paper gives the paper's twenty DELPHI
bins of 1 - T (gentune.bands.PAPER_THRUST_BINS).

--targets FILE imposes a calculation at every point, on each generator alone before the mixing, as
the paper does, and prints the band after it next to the band before. It needs --reference-file,
whose 1 - T is the paper's.

These predictions and bands do not carry the factors that take a prediction to the way a
measurement counts its particles, so they are not compared with data as they are.
examples/reproduce_paper.py gives <1-T> and DELPHI's distribution with those factors, with all
their bands.

The events come from one of three places.

  --reference-file FILE       the paper's reference sample, the deposit reference_MIX17.npz
                              (REFERENCE.md), 1,149,840 events with the paper's own per-event
                              observables. This gives the paper's band.
  --reference                 the 2000 reference events that ship in models/<stage>_selftest.npz,
                              drawn from that sample, so every statistical error is about
                              sqrt(1149840/2000) = 24 times larger for MIX17.
  --hepmc FILE --from POINT   your own sample, generated at POINT, reweighted with ratio weights.
                              POINT is one number per parameter, or default, sherpa-default or
                              herwig-default followed by name=value changes. For MIX17 the
                              sample of one generator sits at the fraction 0 (Sherpa) or 1 (Herwig).

Which stage. --stage F (Herwig) or E (Sherpa) gives the band of one generator's own parameters.
MIX17 also sweeps the fraction from 0 to 1, so it includes the choice between Sherpa and Herwig,
which is the paper's band. On a sample of one generator that means reweighting its events toward
the other generator, which costs effective sample size: N_eff/N, printed for each generator, says
how much.

Next to every band are the statistics that limit it. "stat. error" is the median over the points
of the statistical error of the prediction at one point, as in the paper
(gentune.mixture.mean_and_error), with the calculation imposed when it is, and then over every
tenth point, which is enough for a median and saves most of the time. --bootstrap N resamples
the events N times with replacement, reweights each resample to every point, and gives the
bootstrap sd of the prediction at each point ("boot sd", the median over the points) and of the
band itself ("boot sd of band"). It runs by default, 100 resamples, for --reference and --hepmc,
whose samples are small; not with --targets. A band is marked * and named in a warning, as not
significant, when the statistical error at a point is as large as the band or the bootstrap sd of
the band is more than a third of it. A second warning says when the median N_eff over the points
is below 1000.

--out writes an npz: the points (nodes), the predictions and their statistical errors at every
point (with --targets at every tenth, NaN elsewhere), N_eff, the histograms, the bootstrap, the
per-event observables (per_event, one column per name in observables) and the event numbers
(event_number: the number in the HepMC file, or the index in the reference sample).
--save-weights adds the weights, one row per point, each row in the order of the events and
summing to one; for MIX17 they are (1 - xi) w_S + xi w_H, whose weighted mean of any observable
is the mixed one.

The observables of --hepmc and --reference are made from the per-particle features by
gentune.bands.observables: 1 - T, the charged and the total multiplicity, and the number of
baryons and antibaryons, with the photons radiated by the beams left out of 1 - T and the total
multiplicity, as in the paper. 1 - T there takes its axis from the iteration the features are
built on, which puts its mean above that of the exact axis of the paper (generators/README.md).
--reference-file has the paper's own: 1 - T on the exact axis, B_T, rho_H, n_ch, n, n_baryon.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gentune import load_head, Trunk, features, axes, bands, mixture
from gentune.head import MIXTURE_KINDS
from gentune.maxent import load_targets

HIST_CHOICES = tuple(bands.REFERENCE_KEYS)       # 1-T B_T rho_H n_ch n n_baryon
POINTS, SEED = 2000, 11                          # the paper's points, bands.lhs_nodes(head)
# With a calculation imposed, the statistical error is taken at every STAT_EVERY-th point only:
# its projection on the constraint functions costs more than the rest of a point, and the error
# enters only through its median over the points.
STAT_EVERY = 10


class Formatter(argparse.RawDescriptionHelpFormatter):
    """--help with the default of every option that has one, and the module docstring as is."""

    def _get_help_string(self, action):
        h = action.help or ''
        if action.default not in (None, False, argparse.SUPPRESS) and '%(default)' not in h:
            h += ' (default: %(default)s)'
        return h


def add_sample_args(ap):
    """The options that say where the events come from, shared with learning_band.py."""
    ap.add_argument('--stage', default='MIX17',
                    help='the released stage: MIX17 (both generators and the fraction), E '
                         '(Sherpa) or F (Herwig)')
    ap.add_argument('--models', default=os.path.relpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')),
        help='the folder with <stage>_head.npz, _trunk.npz and _selftest.npz')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--reference-file', metavar='FILE',
                     help="the paper's reference sample, reference_MIX17.npz (REFERENCE.md)")
    src.add_argument('--reference', action='store_true',
                     help='the 2000 reference events of models/<stage>_selftest.npz')
    src.add_argument('--hepmc', metavar='FILE',
                     help='your own sample, a HepMC2 or HepMC3 ASCII file, gzip allowed')
    ap.add_argument('--from', dest='src', nargs='+', metavar='VALUE',
                    help='with --hepmc, the point the sample was generated at: one number per '
                         'parameter, or default, sherpa-default or herwig-default followed by '
                         'name=value changes')
    ap.add_argument('--fmt', choices=['auto', 'hepmc2', 'hepmc3', 'herwig'], default='auto',
                    help='with --hepmc, the format; auto reads it from the header, herwig is an old '
                         'name for hepmc2')
    ap.add_argument('--units', choices=['GEV', 'MEV'],
                    help="with --hepmc, the momentum unit of a file without a units line")
    ap.add_argument('--max-events', type=int, default=None,
                    help='with --hepmc, read at most this many events')
    ap.add_argument('--bins', nargs='+', metavar='EDGE',
                    help="bin edges for the distribution of --hist, or 'paper' for the paper's "
                         "twenty DELPHI bins of 1-T")
    ap.add_argument('--hist', default='1-T', choices=HIST_CHOICES,
                    help='the observable --bins applies to; B_T and rho_H need --reference-file')
    ap.add_argument('--targets', metavar='FILE',
                    help='a calculation to impose, such as targets_thrust_nlo.npz; needs '
                         '--reference-file')


def point(a, head, tokens, flag):
    """A point from the command line, checked against the box by head.check()."""
    try:
        t = axes.point(a.stage, tokens, head.nt, head.centre)
    except ValueError as e:
        raise ValueError(f'--{flag} {" ".join(tokens)}: {e}')
    head.check(t)
    return t


def bins_of(a):
    """(lo, hi) of the bins --bins asks for, or None."""
    if not a.bins:
        return None
    if a.bins == ['paper']:
        return bands.PAPER_THRUST_BINS
    try:
        e = np.array([float(x) for x in a.bins])
    except ValueError:
        raise ValueError(f"--bins takes increasing edges or 'paper', not {' '.join(a.bins)!r}")
    if len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ValueError(f'--bins takes at least two increasing edges, not {" ".join(a.bins)}')
    return e[:-1], e[1:]


def load_sample(a, head):
    """The events the options ask for, as a dict: A (the event vectors), obs (the per-event
    observables), event_number, and src, the point of --from, which is None for reference events
    because they are reweighted directly."""
    if a.src and not a.hepmc:
        raise ValueError('--from goes with --hepmc: reference events need no starting point')
    if a.targets and not a.reference_file:
        raise ValueError('--targets needs --reference-file: the calculation constrains 1 - T on the '
                         'exact thrust axis, which only the reference file holds')
    if a.reference_file:
        r = np.load(a.reference_file)
        got = [str(x) for x in r['theta_names']]
        if got != list(axes.names(a.stage, head.nt)):
            raise ValueError(f'{a.reference_file} is the reference sample of a head with the '
                             f'parameters {", ".join(got)}, not of {a.stage}. Pass --stage MIX17')
        # float64 once, here, so that every point is a plain float64 product, as in
        # examples/reproduce_paper.py; the values are those of the float32 file, exactly
        A = head.pack(r['AE'], np.float64)
        n = len(A)
        print(f'{n:,} events of the reference sample {a.reference_file}, with the paper\'s '
              f'per-event observables')
        return dict(A=A, obs=bands.reference_observables(r), event_number=np.arange(n), src=None)
    if a.reference:
        sb = np.load(f'{a.models}/{a.stage}_selftest.npz')
        n, nref = len(sb['mask']), int(sb['n_reference_events'])
        print(f'{n} reference events of {nref:,}: every statistical error here is about '
              f'{np.sqrt(nref/n):.0f} times that of the full reference sample')
        return dict(A=sb['AE'], obs=bands.observables(sb['particles'], sb['mask']),
                    event_number=sb['reference_index'], src=None)
    if not a.src:
        raise ValueError('--hepmc needs --from, the point the sample was generated at')
    src = point(a, head, a.src, 'from')
    if getattr(head, 'kind', None) in MIXTURE_KINDS and src[-1] not in (0.0, 1.0):
        raise ValueError(f'--from puts the fraction at {src[-1]:g}. A sample made by one generator '
                         f'sits at 0 (Sherpa) or 1 (Herwig): use sherpa-default or herwig-default')
    events = features.parse_hepmc(a.hepmc, a.max_events, a.fmt, a.units)
    F, M, _ = features.features_from_events(events)
    trunk = Trunk(f'{a.models}/{a.stage}_trunk.npz')
    head.check_trunk(trunk)
    print(f'{len(events)} events from {a.hepmc}')
    return dict(A=trunk.embed(features.trunk_features(F), M), obs=bands.observables(F, M),
                event_number=events.numbers, src=src)


class Model:
    """The prediction of a released stage at a point, for the examples. For MIX17 it is the
    paper's mixture of the two generators, each with its own weights (gentune.mixture), and for E
    or F the head's weights, under the generator name ''. Weights are dicts {generator: weights},
    each summing to one over the events."""

    def __init__(self, head, sample):
        self.head, self.s = head, sample
        self.mix = getattr(head, 'kind', None) in MIXTURE_KINDS

    def weights(self, theta):
        A, src = self.s['A'], self.s['src']
        if self.mix:
            return mixture.weights(self.head, A, theta, theta_from=src)
        return {'': self.head.weights(A, theta) if src is None else
                self.head.ratio_weights(A, theta, src)}

    def coefficients(self, theta):
        return mixture.coefficients(theta) if self.mix else {'': 1.0}

    def impose(self, w, inwin, F, calc):
        """(the weights with the calculation imposed on each generator alone, imposed), with
        imposed = (inwin, F, info) for the errors and the learning band of those weights."""
        if self.mix:
            wc, info = mixture.impose(w, inwin, F, calc)
            return wc, (inwin, F, info)
        lw, _, info = calc.impose(np.log(w['']), inwin, F)
        return {'': np.exp(lw)}, (inwin, F, info)

    def mean(self, w, O, theta):
        return mixture.mean(w, O, theta) if self.mix else w[''] @ np.asarray(O, np.float64)

    def mean_and_error(self, w, O, theta, mask=None, imposed=None):
        if self.mix:
            return mixture.mean_and_error(w, O, theta, mask=mask, imposed=imposed)
        return bands.mean_and_error(w[''], O, mask=mask, imposed=imposed)

    def histogram(self, b, w, width, theta, imposed=None):
        if self.mix:
            return mixture.histogram(b, w, width, theta, imposed=imposed)
        return bands.histogram(b, w[''], width, imposed=imposed)

    def event_weights(self, w, theta):
        """One weight per event whose weighted mean of any observable is the prediction."""
        return mixture.event_weights(w, theta) if self.mix else w['']

    def supported(self, theta):
        """Whether the training design came near theta (head.support). For MIX17 each generator
        is asked at its own fraction, with the other generator's parameters at the centre of the
        box, where every training run of the mixture had them."""
        if not self.mix:
            q = self.head.support(theta)
            return q is None or q['ok']
        ok = True
        h = self.head
        for g, sl in (('sherpa', slice(h.nS, h.nS + h.nH)), ('herwig', slice(0, h.nS))):
            t = mixture.at_fraction(theta, mixture.XI[g])
            t[sl] = h.centre[sl]
            q = h.support(t)
            ok &= q is None or q['ok']
        return ok


def resampled(C, w, O, b, width, coef):
    """The bootstrap of the predictions at one point: (resamples, k + nb) for the counts C of
    bands.resample_counts, each generator's weights w renormalized on every resample, and the
    generators mixed with coef. The histogram is the fraction of all events in each bin over its
    width, as everywhere."""
    k, nb = O.shape[1], len(width)
    inb = np.flatnonzero(b >= 0)
    out = 0.0
    for g, wg in w.items():
        X = np.zeros((len(wg), 1 + k + nb))
        X[:, 0], X[:, 1:1 + k] = wg, wg[:, None]*O
        X[inb, 1 + k + b[inb]] = wg[inb]
        S = bands.resampled_sums(C, X)
        out = out + coef[g]*np.concatenate([S[:, 1:1 + k]/S[:, :1],
                                            S[:, 1 + k:]/(width*S[:, :1])], 1)
    return out


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest, formatter_class=Formatter)
    add_sample_args(ap)
    ap.add_argument('--nodes', type=int, default=POINTS, help='points of the Latin hypercube')
    ap.add_argument('--seed', type=int, default=SEED, help='seed of the Latin hypercube')
    ap.add_argument('--bootstrap', type=int, default=None, metavar='N',
                    help='resamples of the events for the bootstrap sd of the predictions and of '
                         'the band, 0 for none (default: 100 for --reference and --hepmc, 0 for '
                         '--reference-file and with --targets)')
    ap.add_argument('--out', help='npz for the points, the predictions and their errors, N_eff, the '
                                  'histograms, the bootstrap, the per-event observables and the '
                                  'event numbers')
    ap.add_argument('--save-weights', action='store_true',
                    help='with --out, also the weights at every point, (points, events) float32')
    a = ap.parse_args()
    t0 = time.time()

    head = load_head(f'{a.models}/{a.stage}_head.npz')
    try:
        s = load_sample(a, head)
        bins = bins_of(a)
        if bins is not None and a.hist not in s['obs']:
            raise ValueError(f'--hist {a.hist} is not among the observables of these events, '
                             f'{", ".join(s["obs"])}. B_T and rho_H need --reference-file')
    except ValueError as e:
        print(f'ERROR: {e}')
        return 2
    model = Model(head, s)
    obs = s['obs']
    O = np.column_stack(list(obs.values()))
    N, k = O.shape
    calc = None
    if a.targets:
        calc = load_targets(a.targets)
        inwin, F = calc.functions(obs['1-T'])
    nodes = bands.lhs_nodes(head, a.nodes, a.seed)
    ok = np.array([model.supported(t) for t in nodes])
    lo = hi = width = np.zeros(0)
    b = np.zeros(N, int) - 1
    if bins is not None:
        lo, hi = bins
        width = hi - lo
        b = bands.bin_index(obs[a.hist], lo, hi)
    nb = len(width)
    B = a.bootstrap if a.bootstrap is not None else (0 if a.reference_file or a.targets else 100)
    if B and a.targets:
        print('--bootstrap is not taken with --targets: the calculation would have to be imposed '
              'again on every resample. The statistical error with it imposed is printed instead.')
        B = 0
    C = bands.resample_counts(N, B, seed=0) if B else None
    vals, stat, before, neff, W = [], [], [], [], []
    boot = np.zeros((B, a.nodes, k + nb))
    for p, th in enumerate(nodes):
        w = model.weights(th)
        if p == 0:
            gens = list(w)                                # the generators, '' for E and F
        neff.append([head.n_eff(w[g]) for g in gens])
        imp = None
        if calc is not None:
            h0 = model.histogram(b, w, width, th)[0] if nb else np.zeros(0)
            before.append(np.r_[model.mean(w, O, th), h0])
            w, imp = model.impose(w, inwin, F, calc)
        if imp is None or p % STAT_EVERY == 0:
            m, e = model.mean_and_error(w, O, th, imposed=imp)
            h, he = model.histogram(b, w, width, th, imposed=imp) if nb else (np.zeros(0), np.zeros(0))
            stat.append(np.r_[e, he])
        else:
            m = model.mean(w, O, th)
            h = model.histogram(b, w, width, th)[0] if nb else np.zeros(0)
            stat.append(np.full(k + nb, np.nan))
        vals.append(np.r_[m, h])
        if B:
            boot[:, p] = resampled(C, w, O, b, width, model.coefficients(th))
        if a.save_weights:
            W.append(model.event_weights(w, th).astype(np.float32))
        if a.nodes >= 500 and (p + 1) % 500 == 0:
            print(f'  point {p + 1}/{a.nodes} [{time.time() - t0:.0f} s]', flush=True)
    vals, stat, neff = np.array(vals), np.array(stat), np.array(neff)
    band = vals.std(0, ddof=1)
    band0 = np.array(before).std(0, ddof=1) if before else None
    med_err = np.nanmedian(stat, 0)
    weak = med_err >= band
    if B:
        boot_point = np.median(boot.std(0, ddof=1), 0)    # the median over the points
        boot_band = boot.std(1, ddof=1).std(0, ddof=1)
        weak |= boot_band > band/3

    print(f'\n{a.stage}: generator band over {a.nodes} points of a Latin hypercube, seed {a.seed} '
          f'[{time.time() - t0:.0f} s]')
    if (a.nodes, a.seed) != (POINTS, SEED):
        print(f"these are not the paper's points ({POINTS} with seed {SEED})")
    print(f'{ok.sum()} of {a.nodes} points lie where the training design reached (head.support)')
    for j, g in enumerate(gens):
        print(f'N_eff/N{" of " + g if g else ""} over the points: median {np.median(neff[:, j])/N:.3f}, '
              f'5th percentile {np.percentile(neff[:, j], 5)/N:.3f}, smallest '
              f'{neff[:, j].min()/N:.3f}, of {N:,} events')
    after = ' after the calculation' if calc is not None else ''
    bcols = f' {"boot sd":>10} {"boot sd of band":>16}' if B else ''
    b0col = f' {"band before":>11}' if calc is not None else ''
    print(f'\n{"observable":<10} {"mean over points":>16} {"band (sd)":>10}{b0col} {"p16":>10} '
          f'{"p84":>10} {"stat. error":>11}{bcols}' + (f'      (the predictions{after})' if after else ''))
    for j, name in enumerate(obs):
        x = vals[:, j]
        bv = f' {boot_point[j]:>10.3g} {boot_band[j]:>16.3g}' if B else ''
        b0 = f' {band0[j]:>11.4g}' if calc is not None else ''
        print(f'{name:<10} {x.mean():>16.6g} {band[j]:>10.4g}{b0} {np.percentile(x, 16):>10.5g} '
              f'{np.percentile(x, 84):>10.5g} {med_err[j]:>11.3g}{bv}{"  *" if weak[j] else ""}')
    if nb:
        print(f'\n{a.hist} distribution, the fraction of all events in each bin over its width{after}')
        print(f'{"from":>8} {"to":>8} {"mean over points":>16} {"band (sd)":>10}{b0col} '
              f'{"stat. error":>11}{bcols}')
        for i in range(nb):
            j = k + i
            bv = f' {boot_point[j]:>10.3g} {boot_band[j]:>16.3g}' if B else ''
            b0 = f' {band0[j]:>11.4g}' if calc is not None else ''
            print(f'{lo[i]:>8.4g} {hi[i]:>8.4g} {vals[:, j].mean():>16.5g} {band[j]:>10.4g}{b0} '
                  f'{med_err[j]:>11.3g}{bv}{"  *" if weak[j] else ""}')
    print('\nstat. error is the median over the points of the statistical error of the prediction'
          + (f', with the calculation imposed, over every {STAT_EVERY}th point' if calc is not None
             else '') +
          (f'. boot sd is the median over the points of the bootstrap sd of the prediction, and '
           f'boot sd of band the bootstrap sd of the band, from {B} resamples of the {N:,} events'
           if B else ''))
    if np.median(neff.min(1)) < 1000:
        print(f'WARNING the median N_eff over the points is {np.median(neff.min(1)):.0f}, below '
              f'1000, so every number here is dominated by the statistics of the sample.')
    if weak.any():
        names = np.array(list(obs) + [f'bin {lo[i]:.4g}-{hi[i]:.4g}' for i in range(nb)])
        print(f'WARNING the band is not significant for {", ".join(names[weak])} (marked *): '
              f'the statistical error at a point is as large as the band'
              + (', or the bootstrap sd of the band is more than a third of it' if B else '')
              + '. Quoting it needs a larger sample'
              + (', such as the full reference sample (--reference-file).'
                 if a.stage == 'MIX17' and not a.reference_file else '.'))
    if a.out:
        out = dict(nodes=nodes, names=np.array(axes.names(a.stage, head.nt)),
                   observables=np.array(list(obs)), generators=np.array(gens),
                   means=vals[:, :k], stat=stat[:, :k], n_eff=neff, n_events=N, support_ok=ok,
                   event_number=np.asarray(s['event_number']), per_event=O,
                   hist_observable=a.hist if nb else '', lo=lo, hi=hi, hists=vals[:, k:],
                   hist_stat=stat[:, k:], bootstrap=B, boot_means=boot[..., :k],
                   boot_hists=boot[..., k:], targets=a.targets or '')
        if before:
            before = np.array(before)
            out.update(means_before=before[:, :k], hists_before=before[:, k:])
        if a.save_weights:
            out['weights'] = np.array(W)
        np.savez(a.out, **out)
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
