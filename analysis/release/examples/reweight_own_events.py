#!/usr/bin/env python3
r"""Weight your own event sample from the parameters it was generated at to any other point.

    python examples/reweight_own_events.py --stage E --hepmc examples/data/sherpa_default.hepmc.gz \
        --from 0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18 \
        --to   0.120 0.68 3.9 0.48 0.46 0.17 0.97 0.18 --out weights.npz

The commands here read the example files of examples/data. Your own file goes in their place.

--from is the parameter point YOUR sample was generated at and --to is where you want it. Both
must lie inside the training box, which --list prints, as it does when no --hepmc is given. The
output is one weight per event, normalized to sum to one, plus the effective sample size, which
is the number that tells you whether the shift you asked for is one this sample can support.

A point is either one number per parameter, in the order the listing prints, or a preset
followed by any parameters you want to change, by name. The command above is the same as

    python examples/reweight_own_events.py --stage E --hepmc examples/data/sherpa_default.hepmc.gz \
        --from sherpa-default --to sherpa-default alphas=0.120

sherpa-default and herwig-default are the generator's own defaults, the values a card that sets
nothing gets, and default is the paper's central point, which is the same thing for a single
generator. For MIX17 a sample made by one generator sits at an edge of the fraction xi, 0 for
Sherpa and 1 for Herwig, and --from must be such a point. The presets put the other generator's
parameters at the centre of their box, where the training runs had them; they do not enter the
weights of the sample. At --to = (theta_S, theta_H, xi) the prediction is the paper's mixture,
(1 - xi) <O>_S + xi <O>_H: Sherpa at theta_S and Herwig at theta_H each get their own weights
on your events, from the head at xi = 0 and xi = 1 against the point of the sample
(gentune.mixture.weights). So a Herwig sample at its defaults is reweighted to a larger diquark
weight with

    python examples/reweight_own_events.py --stage MIX17 --hepmc examples/data/herwig_default.hepmc.gz \
        --from herwig-default --to herwig-default H:pwtdiquark=0.40

and --to default takes the same sample to the paper's central point, where half of the
prediction is Sherpa's, made from Herwig's events.

--out writes an npz with the weights, weights[i] for the i-th event of the file, summing to one,
the event number each of those events carries in the file (event_number), both points, the
parameter names, and the per-event observables of gentune.bands.observables (per_event, one
column per name in observables), with the photons radiated by the beams left out of 1 - T and
the total multiplicity, as in the paper. Nothing here is specific to an observable: the weights
are per event, so any distribution, moment or histogram made from the sample can be remade with
them. For MIX17, weights is (1 - xi) w_S + xi w_H, whose weighted mean of any observable over all
events is the paper's mixed mean, and weights_sherpa and weights_herwig hold each generator's own,
each summing to one (a generator whose coefficient is zero is left out). A mean over part of the
events, such as that of 1 - T inside the window of a calculation, is each generator's own mean
there, mixed with (1 - xi, xi): take it from weights_sherpa and weights_herwig.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gentune import load_head, Trunk, features, axes, bands, mixture
from gentune.head import MIXTURE_KINDS


def outside(t, head, nm, tol=1e-9):
    """One line per parameter that is outside the box, with the same tolerance as in_box."""
    return [f'  [{i:>2}]  {nm[i]} = {t[i]:g} is outside {lo:.4f} to {hi:.4f}'
            for i, (lo, hi) in enumerate(head.box) if not lo - tol <= t[i] <= hi + tol]


def shown_value(t, i, nm):
    """Parameter i of t as printed, or 'unused' for a mixture block that drops out of the weights
    because the fraction sits at the other generator's edge."""
    if nm[-1] == 'fraction':
        g = axes.generator_of(nm[i])
        if (g == 'sherpa' and t[-1] == 1.0) or (g == 'herwig' and t[-1] == 0.0):
            return 'unused'
    return f'{t[i]:.6g}'


class Formatter(argparse.RawDescriptionHelpFormatter):
    """--help with the default of every option that has one, and the module docstring as is."""

    def _get_help_string(self, action):
        h = action.help or ''
        if action.default not in (None, False, argparse.SUPPRESS) and '%(default)' not in h:
            h += ' (default: %(default)s)'
        return h


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest, formatter_class=Formatter)
    ap.add_argument('--stage', default='E', help='the released stage: E (Sherpa), F (Herwig) or '
                                                 'MIX17 (both generators and the fraction)')
    ap.add_argument('--models', default=os.path.relpath(
        os.path.join(os.path.dirname(__file__), '..', 'models')),
        help='the folder with <stage>_head.npz and <stage>_trunk.npz')
    ap.add_argument('--hepmc', help='your sample, a HepMC2 or HepMC3 ASCII file, gzip allowed')
    ap.add_argument('--fmt', choices=['auto', 'hepmc2', 'hepmc3', 'herwig'], default='auto',
                    help='the HepMC format. auto reads it from the file header; herwig is an old '
                         'name for hepmc2')
    ap.add_argument('--units', choices=['GEV', 'MEV'],
                    help='the momentum unit, for a file without a units line; by default the '
                         'file\'s own U line, or GeV')
    ap.add_argument('--max-events', type=int, default=None, help='read at most this many events')
    ap.add_argument('--from', dest='src', nargs='+', metavar='VALUE',
                    help='the point your sample was generated at: one number per parameter, or '
                         'sherpa-default, herwig-default or default followed by name=value changes')
    ap.add_argument('--to', dest='dst', nargs='+', metavar='VALUE',
                    help='the point to reweight to, in the same form as --from')
    ap.add_argument('--out', help='npz to write the weights, the event numbers and the per-event '
                                  'observables to')
    ap.add_argument('--list', action='store_true',
                    help='print the parameters, their box, defaults and card lines; this is also '
                         'what happens without --hepmc')
    a = ap.parse_args()

    hp = f'{a.models}/{a.stage}_head.npz'
    kind = str(np.load(hp).get('head_kind', 'cond'))
    head = load_head(hp)
    mix = kind in MIXTURE_KINDS
    print(f'{a.stage}: ' + ('the mixture of Sherpa and Herwig, ' if mix else f'{kind} head, ') +
          f'{head.nt} parameters, {head.ENS} ensemble members, T={head.T}')
    nm = axes.names(a.stage, head.nt)
    axes.verify(a.stage, head, os.path.join(os.path.dirname(__file__), '..', 'generators'))
    if a.list or a.hepmc is None:
        print('\n'.join(axes.listing(a.stage, head)))

    pts = {}
    for name, tokens in (('from', a.src), ('to', a.dst)):
        if tokens is None:
            continue
        try:
            t = axes.point(a.stage, tokens, head.nt, head.centre)
        except ValueError as e:
            print(f'ERROR: --{name} {" ".join(tokens)}: {e}')
            return 2
        bad = outside(t, head, nm)
        if bad:
            print(f'ERROR: the --{name} point is outside the training box, where the head is not '
                  f'to be trusted:\n' + '\n'.join(bad))
            return 2
        # Inside the box is necessary and not sufficient: check() also asks whether the
        # training design came anywhere near this point.
        s = head.check(t, warn=False)
        if s is not None and not s['ok']:
            print(f'WARNING the --{name} point is inside the box but outside the region that was '
                  f'generated (score {s["score"]:.2f} against {s["threshold"]:.2f} for the '
                  f'training design). The weights are an extrapolation and the closure numbers '
                  f'quoted for this stage do not cover it.')
        pts[name] = t
    if 'from' in pts and mix and pts['from'][-1] not in (0.0, 1.0):
        print(f'ERROR: --from puts the fraction at {pts["from"][-1]:g}. A sample made by one '
              f'generator sits at 0 (Sherpa) or 1 (Herwig): use sherpa-default or herwig-default. '
              f'The head at an interior fraction is its own interpolation between the generators, '
              f'which no generator makes.')
        return 2
    if len(pts) == 2:
        shown = {k: [shown_value(t, i, nm) for i in range(head.nt)] for k, t in pts.items()}
        moved = [i for i in range(head.nt) if shown['from'][i] != shown['to'][i]]
        print('\nfrom --from to --to, the parameters that change:' if moved else
              '\n--from and --to are the same point, so every weight will be the same')
        for i in moved:
            print(f'  [{i:>2}]  {nm[i]:<20s} {shown["from"][i]:>10s}  ->  {shown["to"][i]}')
    if a.hepmc is None or len(pts) < 2:
        print('\npass --hepmc, --from and --to to compute weights')
        return 0
    src, dst = pts['from'], pts['to']

    print(f'\nreading {a.hepmc}')
    events = features.parse_hepmc(a.hepmc, a.max_events, a.fmt, a.units)
    F, M, _ = features.features_from_events(events)
    print(f'{len(events)} events, {M.sum()/len(events):.1f} particles per event on average')
    if (M.sum(1) == features.P_MAX).mean() > 0.02:
        print(f'NOTE {100*(M.sum(1) == features.P_MAX).mean():.1f}% of events hit the '
              f'{features.P_MAX} particle limit, so their softest particles were dropped. '
              f'The training samples were built the same way, so this is consistent, but a '
              f'much higher fraction than a few percent means your events are not e+e- at LEP '
              f'energies and the head is being extrapolated.')

    trunk = Trunk(f'{a.models}/{a.stage}_trunk.npz')
    # A head and a trunk are two halves of one network, and nothing in their shapes
    # distinguishes one stage's trunk from another's, so pairing an E head with an F trunk
    # returns plausible weights that are wrong. This refuses instead.
    head.check_trunk(trunk)
    A = trunk.embed(features.trunk_features(F), M)
    parts = {}
    if mix:
        parts = mixture.weights(head, A, dst, theta_from=src)
        w = mixture.event_weights(parts, dst)
        coef = mixture.coefficients(dst)
        for g, v in parts.items():
            ng = head.n_eff(v)
            print(f'\n{g} at the --to point, coefficient {coef[g]:g}: N_eff = {ng:.0f} of {len(v)} '
                  f'events, N_eff/N = {ng/len(v):.3f}, max/mean weight ratio = {v.max()*len(v):.2f}')
    else:
        w = head.ratio_weights(A, dst, src)
    neff = min([head.n_eff(w)] + [head.n_eff(v) for v in parts.values()])
    print(f'\nweights' + (', (1 - xi) w_S + xi w_H' if mix else '') +
          f': N_eff = {head.n_eff(w):.0f} of {len(w)} events, N_eff/N = {head.n_eff(w)/len(w):.3f}')
    print(f'         max/mean weight ratio = {w.max()*len(w):.2f}')
    if neff/len(w) < 0.2:
        print('NOTE the effective sample size has dropped below a fifth of the events, so this '
              'shift is large for this sample. The weights are still correct, but the '
              'statistical errors on anything you compute from them will be much larger than '
              'the raw event count suggests.')
    if a.out:
        obs = bands.observables(F, M)
        np.savez_compressed(a.out, weights=w, event_number=events.numbers, theta_from=src,
                            theta_to=dst, names=np.array(nm), observables=np.array(list(obs)),
                            per_event=np.column_stack(list(obs.values())),
                            **{f'weights_{g}': v for g, v in parts.items()})
        print(f'wrote {a.out}: weights[i], summing to one, is the weight of the i-th event of '
              f'the file, numbered event_number[i] there' +
              (', and weights_<generator> each generator\'s own' if mix else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
