#!/usr/bin/env python3
r"""Weight your own event sample from the parameters it was generated at to any other point.

    python examples/reweight_own_events.py --stage E --hepmc my_sample.hepmc.gz \
        --from 0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18 \
        --to   0.120 0.68 3.9 0.48 0.46 0.17 0.97 0.18 --out weights.npz

--from is the parameter point YOUR sample was generated at and --to is where you want it. Both
must lie inside the training box, which --list prints, as it does when no --hepmc is given. The
output is one weight per event, normalized to sum to one, plus the effective sample size, which
is the number that tells you whether the shift you asked for is one this sample can support.

A point is either one number per parameter, in the order the listing prints, or a preset
followed by any parameters you want to change, by name. The command above is the same as

    python examples/reweight_own_events.py --stage E --hepmc my_sample.hepmc.gz \
        --from sherpa-default --to sherpa-default alphas=0.120

sherpa-default and herwig-default are the generator's own defaults, the values a card that sets
nothing gets, and default is the paper's central point, which is the same thing for a single
generator. For MIX17 a sample made by one generator sits at an edge of the fraction, 0 for
Sherpa and 1 for Herwig, where the other generator's block drops out of the weights exactly.
The presets put that block at the centre of its box, where the training runs had it, so a
Herwig sample at its defaults is reweighted to a larger diquark weight with

    python examples/reweight_own_events.py --stage MIX17 --hepmc my_herwig.hepmc.gz \
        --from herwig-default --to herwig-default H:pwtdiquark=0.40

and --to default takes the same sample to the paper's central point.

--out writes an npz with the weights, weights[i] for the i-th event of the file, summing to one,
the event number each of those events carries in the file (event_number), both points, the
parameter names, and the per-event observables of gentune.bands.observables (per_event, one
column per name in observables). Nothing here is specific to an observable: the weights are per
event, so any distribution, moment or histogram made from the sample can be remade with them.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gentune import load_head, Trunk, features, axes, bands


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
    ap.add_argument('--stage', default='E', help='the released stage: C_1M, E (Sherpa), F (Herwig) '
                                                 'or MIX17 (both and the fraction)')
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
    print(f'{a.stage}: {kind} head, {head.nt} parameters, {head.ENS} ensemble members, T={head.T}')
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
    if 'from' in pts and nm[-1] == 'fraction' and 0.0 < pts['from'][-1] < 1.0:
        print(f'WARNING --from puts the fraction at {pts["from"][-1]:g}. A sample made by one '
              f'generator is at 0 (Sherpa) or 1 (Herwig), so unless yours was drawn from the '
              f'mixture itself, use sherpa-default or herwig-default.')
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
    w = head.ratio_weights(A, dst, src)
    neff = head.n_eff(w)
    print(f'\nweights: N_eff = {neff:.0f} of {len(w)} events, N_eff/N = {neff/len(w):.3f}')
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
                            per_event=np.column_stack(list(obs.values())))
        print(f'wrote {a.out}: weights[i], summing to one, is the weight of the i-th event of '
              f'the file, numbered event_number[i] there')
    return 0


if __name__ == '__main__':
    sys.exit(main())
