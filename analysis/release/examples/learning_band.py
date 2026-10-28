#!/usr/bin/env python3
r"""The learning band of an observable at one point: how well the network knows the log ratio.

    python examples/learning_band.py --stage MIX17 --reference
    python examples/learning_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
    python examples/learning_band.py --stage MIX17 --hepmc my_herwig.hepmc.gz \
        --from herwig-default --at herwig-default H:pwtdiquark=0.40

This is the learning band of the paper, from the WiFi fit of the ensemble (Benevedes and Thaler,
arXiv:2506.00113). head.learning_band() gives sqrt(g^T C g), with C the covariance of the fitted
basis weights in models/MIX17_wifi.npz and g the derivative of the weighted mean with respect to
them, which is the weighted covariance of the observable with the basis functions of
head.wifi_basis(). It is the band for the generator alone. With a calculation imposed the
derivative also runs through the re-solved multipliers of the maximum-entropy step, which is in
analysis/three_bands.py and is not part of this release.

--at is the point: one number per parameter, or default, sherpa-default or herwig-default
followed by name=value changes, as --to of reweight_own_events.py. By default it is the paper's
central point, both generators at their own defaults and a fraction of one half. The events come
from --reference-file, --reference or --hepmc and --from, as in generator_band.py. On the paper's
reference sample (--reference-file reference_MIX17.npz) this gives the paper's learning band:
<1-T> 0.000286, <n_ch> 0.0594, <n_baryon> 0.00954. On a sample of your own the band is that of
the log ratio between --from and --at, so it vanishes when the two are the same point.

Next to each band, "stat. error" is the statistical error of the mean, sqrt(sum_i w_i^2 (O_i -
<O>)^2). On the 2000 reference events it is about 24 times the paper's and larger than the
learning band. The band itself is a property of the network and does not shrink with more events,
but its estimate from a sample is noisy: --bootstrap N recomputes it on N resamples of the events
and prints its bootstrap sd ("boot sd of band"), and a band whose bootstrap sd is more than a
third of it is marked as not significant.

--bins gives the band of every bin of a distribution, normalized over the bins given, and --bins
paper the paper's DELPHI bins of 1 - T from 0.05. That is a mean of (in the bin)/(bin width) over
the events in any of the bins, the mask of learning_band(). A bin of the 2000 reference events
holds about a hundred of them, and the noise of g adds to g^T C g, so the band of a bin comes out
several times the paper's there. It needs the full sample.

--out writes an npz with the point, the values, bands and errors, the bootstrap, the per-event
observables (per_event, one column per name in observables) and the event numbers.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gentune import load_head, axes, bands
from generator_band import Formatter, add_sample_args, bins_of, load_sample, point, weights_at


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest, formatter_class=Formatter)
    add_sample_args(ap)
    ap.add_argument('--at', nargs='+', metavar='VALUE', default=['default'],
                    help="the point: one number per parameter, or default, sherpa-default or "
                         "herwig-default followed by name=value changes. default is the paper's "
                         "central point")
    ap.add_argument('--wifi', help='the WiFi weights, by default models/<stage>_wifi.npz')
    ap.add_argument('--bootstrap', type=int, default=0, metavar='N',
                    help='resamples of the events for the bootstrap sd of the band, 0 for none')
    ap.add_argument('--out', help='npz for the point, the values, bands and errors, the '
                                  'bootstrap, the per-event observables and the event numbers')
    a = ap.parse_args()
    t0 = time.time()

    head = load_head(f'{a.models}/{a.stage}_head.npz')
    wifi = a.wifi or f'{a.models}/{a.stage}_wifi.npz'
    if not os.path.exists(wifi):
        print(f'ERROR: {wifi} does not exist. The WiFi weights ship for MIX17 only.')
        return 2
    try:
        at = point(a, head, a.at, 'at')
        s = load_sample(a, head)
        bins = bins_of(a)
        if bins is not None and a.hist not in s['obs']:
            raise ValueError(f'--hist {a.hist} is not among the observables of these events, '
                             f'{", ".join(s["obs"])}. B_T and rho_H need --reference-file')
    except ValueError as e:
        print(f'ERROR: {e}')
        return 2
    A, src, obs = s['A'], s['src'], s['obs']
    O = np.column_stack(list(obs.values()))
    N, k = O.shape
    w = weights_at(head, s, at)
    value, stat = bands.mean_and_error(w, O)
    band = head.learning_band(A, at, O, wifi, theta_from=src)
    nb = 0
    lo = hi = np.zeros(0)
    hval = hstat = hband = np.zeros(0)
    if bins is not None:
        lo, hi = bins
        width = hi - lo
        nb = len(width)
        b = bands.bin_index(obs[a.hist], lo, hi)
        inb = b >= 0
        H = np.zeros((N, nb))                              # (in the bin)/(bin width)
        H[inb, b[inb]] = 1/width[b[inb]]
        hval, hstat = bands.histogram(b, w, width)
        hband = head.learning_band(A, at, H, wifi, mask=inb, theta_from=src)
    B = max(a.bootstrap, 0)
    boot = np.zeros((B, k + nb))
    if B:
        C = bands.resample_counts(N, B, seed=0)
        for r in range(B):
            boot[r, :k] = head.learning_band(A, at, O, wifi, theta_from=src, counts=C[r])
            if nb:
                boot[r, k:] = head.learning_band(A, at, H, wifi, mask=inb, theta_from=src,
                                                 counts=C[r])
        boot_band = boot.std(0, ddof=1)
        weak = boot_band > np.r_[band, hband]/3

    nm = axes.names(a.stage, head.nt)
    print(f'\n{a.stage} at ' + ', '.join(f'{n}={v:g}' for n, v in zip(nm, at)) +
          f' [{time.time() - t0:.0f} s]')
    print(f'N_eff/N = {head.n_eff(w)/N:.3f} of {N:,} events')
    bcol = f' {"boot sd of band":>16}' if B else ''
    print(f'\n{"observable":<10} {"value":>12} {"learning band":>14} {"stat. error":>12}{bcol}')
    for j, name in enumerate(obs):
        bv = f' {boot_band[j]:>16.3g}{"  *" if weak[j] else ""}' if B else ''
        print(f'{name:<10} {value[j]:>12.6g} {band[j]:>14.3g} {stat[j]:>12.3g}{bv}')
    if nb:
        print(f'\n{a.hist} distribution, normalized over the bins given')
        print(f'{"from":>8} {"to":>8} {"value":>12} {"learning band":>14} {"stat. error":>12}{bcol}')
        for i in range(nb):
            bv = f' {boot_band[k + i]:>16.3g}{"  *" if weak[k + i] else ""}' if B else ''
            print(f'{lo[i]:>8.4g} {hi[i]:>8.4g} {hval[i]:>12.5g} {hband[i]:>14.3g} '
                  f'{hstat[i]:>12.3g}{bv}')
    print('\nstat. error is sqrt(sum w^2 (O - <O>)^2)' +
          (f', and boot sd of band the bootstrap sd of the learning band from {B} resamples of '
           f'the {N:,} events' if B else ''))
    if B and weak.any():
        labels = np.array(list(obs) + [f'bin {lo[i]:.4g}-{hi[i]:.4g}' for i in range(nb)])
        print(f'WARNING the learning band is not significant for {", ".join(labels[weak])} '
              f'(marked *): its bootstrap sd is more than a third of it. Quoting it needs a larger '
              f'sample' + ('.' if a.reference_file else
                           ', such as the full reference sample (--reference-file).'))
    if a.out:
        np.savez(a.out, theta=at, names=np.array(nm), observables=np.array(list(obs)),
                 value=value, learning=band, stat=stat, n_eff=head.n_eff(w), n_events=N,
                 event_number=np.asarray(s['event_number']), per_event=O,
                 hist_observable=a.hist if nb else '', lo=lo, hi=hi, hist_value=hval,
                 hist_learning=hband, hist_stat=hstat, bootstrap=B, boot_learning=boot[:, :k],
                 boot_hist_learning=boot[:, k:])
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
