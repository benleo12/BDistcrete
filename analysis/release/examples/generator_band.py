#!/usr/bin/env python3
r"""The generator band of an observable: its spread over the whole parameter box.

    python examples/generator_band.py --stage MIX17 --reference
    python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
    python examples/generator_band.py --stage F --hepmc my_herwig.hepmc.gz --from herwig-default

This is the generator band of the paper, a flat prior on the box. The prior is sampled by 200
points of a Latin hypercube with seed 11, the fraction of MIX17 uniform on [0, 1], built as
analysis/three_bands.py builds them, so they are the paper's points to the last digit
(gentune.bands.lhs_nodes). The sample is reweighted to every point, and the band of an
observable is the standard deviation of its weighted mean over the points. --bins does the same
for every bin of a distribution, normalized over the bins given, and --bins paper gives the
paper's DELPHI bins of 1 - T from 0.05 to 0.5 (gentune.bands.PAPER_THRUST_BINS).

The events come from one of three places.

  --reference-file FILE       the paper's reference sample, the deposit reference_MIX17.npz
                              (REFERENCE.md), 1,149,840 events with the paper's own per-event
                              observables, reweighted with weights. This gives the paper's band:
                              <1-T> 0.00438, <n_ch> 0.677, <n_baryon> 0.284. Memory: 3 GB.
  --reference                 the 2000 reference events that ship in models/<stage>_selftest.npz,
                              drawn from that sample, so every statistical error is about
                              sqrt(1149840/2000) = 24 times larger for MIX17.
  --hepmc FILE --from POINT   your own sample, generated at POINT, reweighted with ratio_weights.
                              POINT is one number per parameter, or default, sherpa-default or
                              herwig-default followed by name=value changes.

Which stage. --stage F (Herwig) or E (Sherpa) gives the band of one generator's own parameters.
MIX17 also sweeps the fraction from 0 to 1, so it includes the choice between Sherpa and Herwig,
which is the paper's band. On a sample of one generator that means reweighting its events toward
the other generator, which costs effective sample size: on the Herwig example file the median
N_eff/N over the points is 0.21 with MIX17 and 0.63 with F.

Next to every band are the statistics that limit it. "stat. error" is the median over the points
of the statistical error of the mean at one point, sqrt(sum_i w_i^2 (O_i - <O>)^2). --bootstrap N
(100 by default, 0 to skip) resamples the events N times with replacement, reweights each resample
to every point with its weights renormalized, and gives the bootstrap sd of the mean at each point
("boot sd", the median over the points) and of the band itself ("boot sd of band"). A band is
marked * and named in a warning, as not significant, when the statistical error at a point is as
large as the band or the bootstrap sd of the band is more than a third of it. A second warning
says when the median N_eff over the points is below 1000.

--out writes an npz: the points (nodes), the means and their statistical errors at every point,
N_eff, the histograms, the bootstrap, the per-event observables (per_event, one column per name in
observables) and the event numbers (event_number: the number in the HepMC file, or the index in
the reference sample). --save-weights adds the weights, one row per point, each row in the order
of the events and summing to one.

The observables of --hepmc and --reference are made from the per-particle features by
gentune.bands.observables: 1 - T, the charged and the total multiplicity, and the number of
baryons and antibaryons. 1 - T there is read off the iterative axis the features are built on,
which puts its mean about 1.2 percent above the exact axis of the paper (generators/README.md).
--reference-file has the paper's own: 1 - T on the exact axis, B_T, rho_H, n_ch, n, n_baryon.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gentune import load_head, Trunk, features, axes, bands

HIST_CHOICES = tuple(bands.REFERENCE_KEYS)       # 1-T B_T rho_H n_ch n n_baryon


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
                         '(Sherpa), F (Herwig) or C_1M')
    ap.add_argument('--models', default=os.path.relpath(
        os.path.join(os.path.dirname(__file__), '..', 'models')),
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
                         "DELPHI bins of 1-T from 0.05 to 0.5")
    ap.add_argument('--hist', default='1-T', choices=HIST_CHOICES,
                    help='the observable --bins applies to; B_T and rho_H need --reference-file')


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
    because weights() reweights them directly."""
    if a.src and not a.hepmc:
        raise ValueError('--from goes with --hepmc: reference events need no starting point')
    if a.reference_file:
        r = np.load(a.reference_file)
        got = [str(x) for x in r['theta_names']]
        if got != list(axes.names(a.stage, head.nt)):
            raise ValueError(f'{a.reference_file} is the reference sample of a head with the '
                             f'parameters {", ".join(got)}, not of {a.stage}. Pass --stage MIX17')
        AE = r['AE']
        n = AE.shape[1]
        # float64 once, here, so that every point is a plain float64 product, as in
        # examples/reproduce_paper.py; the values are those of the float32 file, exactly
        A = np.empty((n, head.ENS*head.K))
        for m in range(head.ENS):
            A[:, m*head.K:(m + 1)*head.K] = AE[m]
        del AE
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
    events = features.parse_hepmc(a.hepmc, a.max_events, a.fmt, a.units)
    F, M, _ = features.features_from_events(events)
    trunk = Trunk(f'{a.models}/{a.stage}_trunk.npz')
    head.check_trunk(trunk)
    print(f'{len(events)} events from {a.hepmc}')
    return dict(A=trunk.embed(features.trunk_features(F), M), obs=bands.observables(F, M),
                event_number=events.numbers, src=src)


def weights_at(head, s, theta):
    """The weights of the sample s at theta, normalized to one."""
    return head.weights(s['A'], theta) if s['src'] is None else \
        head.ratio_weights(s['A'], theta, s['src'])


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest, formatter_class=Formatter)
    add_sample_args(ap)
    ap.add_argument('--nodes', type=int, default=200, help='points of the Latin hypercube')
    ap.add_argument('--seed', type=int, default=11, help='seed of the Latin hypercube')
    ap.add_argument('--bootstrap', type=int, default=100, metavar='N',
                    help='resamples of the events for the bootstrap sd of the means and of the '
                         'band, 0 for none')
    ap.add_argument('--out', help='npz for the points, the means and their errors, N_eff, the '
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
    obs = s['obs']
    O = np.column_stack(list(obs.values()))
    N, k = O.shape
    nodes = bands.lhs_nodes(head, a.nodes, a.seed)
    ok = np.array([q is None or q['ok'] for q in map(head.support, nodes)])
    lo = hi = np.zeros(0)
    nb = 0
    if bins is not None:
        lo, hi = bins
        width = hi - lo
        nb = len(width)
        b = bands.bin_index(obs[a.hist], lo, hi)
        inb = np.flatnonzero(b >= 0)
    B = max(a.bootstrap, 0)
    C = bands.resample_counts(N, B, seed=0) if B else None
    means, stat, neff, hists, hstat, W = [], [], [], [], [], []
    boot = np.zeros((B, a.nodes, k + nb))
    for p, th in enumerate(nodes):
        w = weights_at(head, s, th)
        m, e = bands.mean_and_error(w, O)
        means.append(m)
        stat.append(e)
        neff.append(head.n_eff(w))
        if nb:
            h, he = bands.histogram(b, w, width)
            hists.append(h)
            hstat.append(he)
        if B:
            # the sums of every resample at once: the weight, weight x observable, and the weight
            # in each bin, so that a resample's mean is its sum over its total weight
            X = np.zeros((N, 1 + k + nb))
            X[:, 0], X[:, 1:1 + k] = w, w[:, None]*O
            if nb:
                X[inb, 1 + k + b[inb]] = w[inb]
            S = bands.resampled_sums(C, X)
            boot[:, p, :k] = S[:, 1:1 + k]/S[:, :1]
            if nb:
                boot[:, p, k:] = S[:, 1 + k:]/(width*S[:, 1 + k:].sum(1, keepdims=True))
        if a.save_weights:
            W.append(w.astype(np.float32))
    means, stat, neff = np.array(means), np.array(stat), np.array(neff)
    hists, hstat = np.array(hists).reshape(a.nodes, nb), np.array(hstat).reshape(a.nodes, nb)
    vals = np.concatenate([means, hists], 1)              # every prediction, one column each
    band = vals.std(0, ddof=1)
    med_err = np.median(np.concatenate([stat, hstat], 1), 0)
    weak = med_err >= band
    if B:
        boot_point = np.median(boot.std(0, ddof=1), 0)    # the median over the points
        boot_band = boot.std(1, ddof=1).std(0, ddof=1)
        weak |= boot_band > band/3

    print(f'\n{a.stage}: generator band over {a.nodes} points of a Latin hypercube, seed {a.seed} '
          f'[{time.time() - t0:.0f} s]')
    print(f'{ok.sum()} of {a.nodes} points lie where the training design reached (head.support)')
    print(f'N_eff/N over the points: median {np.median(neff)/N:.3f}, 5th percentile '
          f'{np.percentile(neff, 5)/N:.3f}, smallest {neff.min()/N:.3f}, of {N:,} events')
    bcols = f' {"boot sd":>10} {"boot sd of band":>16}' if B else ''
    print(f'\n{"observable":<10} {"mean over points":>16} {"band (sd)":>10} {"p16":>10} '
          f'{"p84":>10} {"stat. error":>11}{bcols}')
    for j, name in enumerate(obs):
        x = means[:, j]
        bv = f' {boot_point[j]:>10.3g} {boot_band[j]:>16.3g}' if B else ''
        print(f'{name:<10} {x.mean():>16.6g} {band[j]:>10.4g} {np.percentile(x, 16):>10.5g} '
              f'{np.percentile(x, 84):>10.5g} {med_err[j]:>11.3g}{bv}{"  *" if weak[j] else ""}')
    if nb:
        print(f'\n{a.hist} distribution, normalized over the bins given')
        print(f'{"from":>8} {"to":>8} {"mean over points":>16} {"band (sd)":>10} '
              f'{"stat. error":>11}{bcols}')
        for i in range(nb):
            j = k + i
            bv = f' {boot_point[j]:>10.3g} {boot_band[j]:>16.3g}' if B else ''
            print(f'{lo[i]:>8.4g} {hi[i]:>8.4g} {hists[:, i].mean():>16.5g} {band[j]:>10.4g} '
                  f'{med_err[j]:>11.3g}{bv}{"  *" if weak[j] else ""}')
    print('\nstat. error is the median over the points of sqrt(sum w^2 (O - <O>)^2)' +
          (f'. boot sd is the median over the points of the bootstrap sd of the mean, and boot '
           f'sd of band the bootstrap sd of the band, from {B} resamples of the {N:,} events'
           if B else ''))
    if np.median(neff) < 1000:
        print(f'WARNING the median N_eff over the points is {np.median(neff):.0f}, below 1000, so '
              f'every number here is dominated by the statistics of the sample.')
    if weak.any():
        labels = np.array(list(obs) + [f'bin {lo[i]:.4g}-{hi[i]:.4g}' for i in range(nb)])
        print(f'WARNING the band is not significant for {", ".join(labels[weak])} (marked *): '
              f'the statistical error at a point is as large as the band'
              + (', or the bootstrap sd of the band is more than a third of it' if B else '')
              + '. Quoting it needs a larger sample'
              + (', such as the full reference sample (--reference-file).'
                 if a.stage == 'MIX17' and not a.reference_file else '.'))
    if a.out:
        out = dict(nodes=nodes, names=np.array(axes.names(a.stage, head.nt)),
                   observables=np.array(list(obs)), means=means, stat=stat, n_eff=neff,
                   n_events=N, support_ok=ok, event_number=np.asarray(s['event_number']),
                   per_event=O, hist_observable=a.hist if nb else '', lo=lo, hi=hi,
                   hists=hists, hist_stat=hstat, bootstrap=B, boot_means=boot[..., :k],
                   boot_hists=boot[..., k:])
        if a.save_weights:
            out['weights'] = np.array(W)
        np.savez(a.out, **out)
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
