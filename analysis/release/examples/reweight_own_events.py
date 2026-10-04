#!/usr/bin/env python3
"""Weight your own event sample from the parameters it was generated at to any other point.

    python examples/reweight_own_events.py --stage E --hepmc my_sample.hepmc.gz \
        --from 0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18 \
        --to   0.120 0.68 3.9 0.48 0.46 0.17 0.97 0.18

--from is the parameter point YOUR sample was generated at and --to is where you want it. Both
must lie inside the training box, which the head prints if you pass neither. The output is one
weight per event, normalized to sum to one, plus the effective sample size, which is the
number that tells you whether the shift you asked for is one this sample can support.

Nothing here is specific to the observable you go on to compute. The weights are per event, so
any distribution, moment or histogram made from the sample can be remade with them.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gentune import Head, MixtureHead, Trunk, features, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='E', help='stage tag, for example E or F')
    ap.add_argument('--models', default=os.path.join(os.path.dirname(__file__), '..', 'models'))
    ap.add_argument('--hepmc', help='HepMC2 ASCII file, gzip allowed')
    # 'auto' reads HepMC2 against HepMC3 from the file header, which is what you want.
    # 'herwig' is kept because it was the old name for the HepMC2 layout.
    ap.add_argument('--fmt', choices=['auto', 'hepmc2', 'hepmc3', 'herwig'],
                    default='auto')
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--from', dest='src', type=float, nargs='+')
    ap.add_argument('--to', dest='dst', type=float, nargs='+')
    ap.add_argument('--out', help='npz to write the weights to')
    a = ap.parse_args()

    hp = f'{a.models}/{a.stage}_head.npz'
    kind = str(np.load(hp).get('head_kind', 'cond'))
    head = (MixtureHead if kind == 'mixture' else Head)(hp)
    print(f'{a.stage}: {kind} head, {head.nt} parameters, {head.ENS} ensemble members, T={head.T}')
    print('training box, one row per parameter:')
    nm = axes.names(a.stage, head.nt)
    axes.verify(a.stage, head, os.path.join(os.path.dirname(__file__), '..', 'generators'))
    for i, (lo, hi) in enumerate(head.box):
        print(f'  [{i:>2}]  {nm[i]:<20s} {lo:>10.4f}  to  {hi:>10.4f}   centre {head.centre[i]:.4f}')
    for i, k in enumerate(nm):
        note = axes.NOTES.get(k.split(':')[-1])
        if note:
            print(f'       {k}: {note}')
    if a.hepmc is None or a.src is None or a.dst is None:
        print('\npass --hepmc, --from and --to to compute weights')
        return 0
    for name, t in (('from', a.src), ('to', a.dst)):
        if not head.in_box(t):
            print(f'ERROR: the --{name} point is outside the training box')
            return 2

    print(f'\nreading {a.hepmc}')
    events = features.parse_hepmc(a.hepmc, a.max_events, a.fmt)
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
    w = head.ratio_weights(A, a.dst, a.src)
    neff = head.n_eff(w)
    print(f'\nweights: N_eff = {neff:.0f} of {len(w)} events, N_eff/N = {neff/len(w):.3f}')
    print(f'         max/mean weight ratio = {w.max()*len(w):.2f}')
    if neff/len(w) < 0.2:
        print('NOTE the effective sample size has dropped below a fifth of the events, so this '
              'shift is large for this sample. The weights are still correct, but the '
              'statistical errors on anything you compute from them will be much larger than '
              'the raw event count suggests.')
    if a.out:
        np.savez_compressed(a.out, weights=w, theta_from=a.src, theta_to=a.dst)
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
