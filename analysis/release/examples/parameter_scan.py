#!/usr/bin/env python3
"""Move one generator parameter and watch an observable respond, by reweighting alone.

    python examples/parameter_scan.py --stage E --axis alphas
    python examples/parameter_scan.py --stage MIX17 --axis fraction

This shows what a released head does with nothing outside the release. It takes the 2000 events
that ship in the self-test bundle, walks one parameter across its trained range while holding the
others at the centre of the box, and prints the reweighted mean of a few observables computed from
the particles themselves. No generator is run: every column comes from the same events with
different weights.

For MIX17 a prediction is the paper's mixture of the two generators, (1 - xi) <O>_S + xi <O>_H,
each with its own weights from the head at xi = 0 and xi = 1 (gentune.mixture). The fraction xi
stays at the centre of its box, 1/2, unless --axis fraction moves it, and then every mean moves on
a straight line from Sherpa's to Herwig's.

The effective sample size is printed alongside, because it is what limits how far a single
sample can be pushed. It falls as the parameter moves away from the reference, and when it
falls far the means are still correct but their errors grow. For MIX17 it is the smaller of the
two generators'.

The observables are those of gentune.bands.observables: 1 - T, the charged and the total
multiplicity, and the number of baryons and antibaryons. The photons radiated by the beams are
left out of 1 - T and the total multiplicity, as in the paper. The 1 - T column is still not the
paper's thrust. It takes its axis from the single-seed iteration the per-particle features are
built on, which gentune/features.py keeps on purpose and which stops at a local maximum in a few
percent of events at the Z pole, so its mean sits above that of the exact axis of the paper and
of the closure tables. generators/README.md, under "The thrust axis, and why two versions of it
ship", says why both exist. The column also sees at most the 100 hardest particles of an event.
None of this touches the weights, so the column still shows how the head responds, but its
values are not the paper's.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gentune import load_head, Trunk, features, axes, bands, mixture
from gentune.head import MIXTURE_KINDS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--stage', default='E', help='the released stage: E (Sherpa), F (Herwig) or '
                                                 'MIX17 (both generators and the fraction)')
    ap.add_argument('--models', default=os.path.relpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')),
        help='the folder with <stage>_head.npz, _trunk.npz and _selftest.npz')
    ap.add_argument('--axis', default='0',
                    help='which parameter to move, by index or by name. Pass "list" to see them')
    ap.add_argument('--points', type=int, default=7, help='points across the range of --axis')
    a = ap.parse_args()

    head = load_head(f'{a.models}/{a.stage}_head.npz')
    mix = getattr(head, 'kind', None) in MIXTURE_KINDS
    nm = axes.names(a.stage, head.nt)
    axes.verify(a.stage, head, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                            'generators'))
    if a.axis == 'list':
        print('\n'.join(axes.listing(a.stage, head)))
        return 0
    try:
        ax = int(a.axis)
    except ValueError:
        try:
            ax = axes.find(nm, a.axis)
        except ValueError:
            print(f'{a.stage} has no parameter called {a.axis!r}. Its parameters are '
                  f'{", ".join(nm)}, and --axis list prints them with their ranges.')
            return 2
    if not 0 <= ax < head.nt:
        print(f'{a.stage} has {head.nt} parameters, so index {ax} does not exist.')
        return 2
    sb = np.load(f'{a.models}/{a.stage}_selftest.npz')
    tp = f'{a.models}/{a.stage}_trunk.npz'
    A = (Trunk(tp).embed(features.trunk_features(sb['particles']), sb['mask'])
         if os.path.exists(tp) else sb['AE'])
    obs = bands.observables(sb['particles'], sb['mask'])
    O = np.column_stack(list(obs.values()))
    lo, hi = head.box[ax]
    print(f'{a.stage}: parameter {ax}, {nm[ax]}, from {lo:.4f} to {hi:.4f}, '
          f'{len(sb["mask"])} events, others held at the centre of the box' +
          (', the generators mixed as (1 - xi) S + xi H' if mix else ''))
    hdr = f'{"value":>10} {"N_eff/N":>8} ' + ' '.join(f'{k:>14}' for k in obs)
    print(hdr); print('-'*len(hdr))
    ref = None
    for v in np.linspace(lo, hi, a.points):
        th = head.centre.copy(); th[ax] = v
        if mix:
            w = mixture.weights(head, A, th)
            vals = np.atleast_1d(mixture.mean(w, O, th))
            neff = min(head.n_eff(x) for x in w.values())
        else:
            w = head.weights(A, th)
            vals = w @ O
            neff = head.n_eff(w)
        row = f'{v:>10.4f} {neff/len(O):>8.3f} '
        if ref is None:
            ref = vals
        row += ' '.join(f'{x:>8.4f} {100*(x/r - 1):>+5.1f}%' for x, r in zip(vals, ref))
        print(row)
    print('\nEach percentage is the shift from the first row. The events are identical '
          'throughout, so every change is the head responding to the parameter.')
    print('The beam photons are left out of 1-T and n, as in the paper. 1-T is built from the '
          'stored features, on the axis of the single-seed iteration, which puts its mean above '
          'the exact-axis value of the paper. See generators/README.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
