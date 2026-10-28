#!/usr/bin/env python3
"""Move one generator parameter and watch an observable respond, by reweighting alone.

    python examples/parameter_scan.py --stage C_1M --axis 0

This is the quickest way to see what a released head actually does. It takes the events that
ship in the self-test bundle, walks one parameter across its trained range while holding the
others at the centre of the box, and prints the reweighted mean of a few observables computed
from the particles themselves. No generator is run: every column comes from the same events
with different weights.

The effective sample size is printed alongside, because it is what limits how far a single
sample can be pushed. It falls as the parameter moves away from the reference, and when it
falls far the means are still correct but their errors grow.

The 1 - T column is not the paper's thrust. It is read off the frame the per-particle features
are stored in, whose axis comes from the single-seed iteration gentune/features.py keeps on
purpose. That iteration stops at a local maximum in about 4 percent of events at the Z pole,
which biases the mean of 1 - T by about +1.2 percent against the exact axis of the paper and
of the closure tables. generators/README.md, under "The thrust axis, and why two versions of it
ship", says why both exist. The column also weights particles by energy rather than momentum
and sees at most the 100 hardest. None of this touches the weights, so the column still shows
how the head responds, but its values are not the paper's.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gentune import load_head, Trunk, features, axes


def observables(particles, mask):
    """A few event shapes and rates, from the stored per-particle features.

    Columns of particles are [z, cos_theta, phi, log10_mass, baryon, charge] in the thrust
    frame, so 1 - T follows from the energy fractions and the polar angles directly. That frame
    uses the iterative axis, so the mean of this 1 - T sits about 1.2 percent above the
    exact-axis one, as the module docstring says."""
    z, cos = particles[..., 0], particles[..., 1]
    m = mask.astype(bool)
    thrust = np.where(m, z*np.abs(cos), 0.0).sum(1)/np.maximum(np.where(m, z, 0.0).sum(1), 1e-12)
    return {
        '1 - T': 1.0 - thrust,
        'multiplicity': m.sum(1).astype(float),
        'charged mult': np.where(m, np.abs(particles[..., 5]) > 0, False).sum(1).astype(float),
        'baryons': np.where(m, np.abs(particles[..., 4]) > 0, False).sum(1).astype(float),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--stage', default='C_1M', help='the released stage: C_1M, E (Sherpa), '
                                                    'F (Herwig) or MIX17 (both and the fraction)')
    ap.add_argument('--models', default=os.path.relpath(
        os.path.join(os.path.dirname(__file__), '..', 'models')),
        help='the folder with <stage>_head.npz, _trunk.npz and _selftest.npz')
    ap.add_argument('--axis', default='0',
                    help='which parameter to move, by index or by name. Pass "list" to see them')
    ap.add_argument('--points', type=int, default=7, help='points across the range of --axis')
    a = ap.parse_args()

    hp = f'{a.models}/{a.stage}_head.npz'
    kind = str(np.load(hp).get('head_kind', 'cond'))
    head = load_head(hp)
    sb = np.load(f'{a.models}/{a.stage}_selftest.npz')
    tp = f'{a.models}/{a.stage}_trunk.npz'
    A = (Trunk(tp).embed(features.trunk_features(sb['particles']), sb['mask'])
         if os.path.exists(tp) else sb['AE'])
    obs = observables(sb['particles'], sb['mask'])

    nm = axes.names(a.stage, head.nt)
    axes.verify(a.stage, head, os.path.join(os.path.dirname(__file__), '..', 'generators'))
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
    lo, hi = head.box[ax]
    print(f'{a.stage}: parameter {ax}, {nm[ax]}, from {lo:.4f} to {hi:.4f}, '
          f'{len(sb["mask"])} events, others held at the centre of the box')
    hdr = f'{"value":>10} {"N_eff/N":>8} ' + ' '.join(f'{k:>14}' for k in obs)
    print(hdr); print('-'*len(hdr))
    ref = None
    for v in np.linspace(lo, hi, a.points):
        th = head.centre.copy(); th[ax] = v
        w = head.weights(A, th)
        row = f'{v:>10.4f} {head.n_eff(w)/len(w):>8.3f} '
        vals = [float(np.sum(w*x)) for x in obs.values()]
        if ref is None:
            ref = vals
        row += ' '.join(f'{x:>8.4f} {100*(x/r - 1):>+5.1f}%' for x, r in zip(vals, ref))
        print(row)
    print('\nEach percentage is the shift from the first row. The events are identical '
          'throughout, so every change is the head responding to the parameter.')
    print('1 - T here is built from the stored features, on the iterative thrust axis, which puts '
          'its mean about 1.2 percent above the exact-axis value of the paper. See '
          'generators/README.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
