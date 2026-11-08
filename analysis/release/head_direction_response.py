#!/usr/bin/env python3
"""Does a head respond to parameter directions its design barely explored?

A conditional head is only informed about directions the training design actually moved. If
its parameter network responds MORE strongly along the directions the design explored least,
it is fitting noise there, and any evaluation point with a component along them picks up a
response the data never constrained.

The test rotates into the design's own principal directions, measures the norm of
db/dtheta along each by central differences, and compares the least explored directions with
the rest. A ratio below one is the healthy case, the network responding least where it knows
least. A ratio above one is the pathology.

The test is for a design that moves some directions much less than others, such as the first
Herwig design, which varied two of its eight parameters at a fraction of the amplitude of the
other six to follow published tune correlations. The designs of the released heads fill their
boxes (generators/README.md), so the spread of the design is nearly the same along every
direction, the anisotropy printed is close to one, and the ratio then says little either way:
the script says so instead of naming a verdict. It prints the ratio of each head, and of each
generator's block of MIX17.

    python head_direction_response.py                        # E and F, from models/ here
    python head_direction_response.py models E F MIX17

NWEAK (default 2) in the environment sets how many of the least explored directions are compared
with the rest, and NPTS (default 64) at how many points of the design cloud the derivatives are
taken.
"""
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gentune.head import MIXTURE_KINDS, load_head

NWEAK = int(os.environ.get('NWEAK', '2'))
NPTS = int(os.environ.get('NPTS', '64'))
EPS = 1e-3
# below this ratio of the largest to the smallest spread of the design, the least explored
# directions are not much less explored than the rest and the ratio is not read as a verdict
ISOTROPIC = 1.5


def probe(models, tag, block=None):
    """block is None for a single-generator head. A mixture head has two parameter networks
    reading two disjoint blocks of theta, so the question has to be asked of each separately,
    with that block's own slice of the design: a 17-vector fed to a network expecting 8 is a
    shape error, and averaging the two blocks together would hide whichever one misbehaves."""
    hp = f'{models}/{tag}_head.npz'
    kind = str(np.load(hp).get('head_kind', 'cond'))
    h = load_head(hp)
    if h.theta_train is None:
        print(f'{tag}: no training design shipped with this head, cannot run')
        return None
    if kind in MIXTURE_KINDS and block is None:
        # the fraction is not a direction of either generator's network, so it is left out
        rS = probe(models, tag, 'S')
        rH = probe(models, tag, 'H')
        return dict(S=rS, H=rH)
    if block == 'S':
        cols, Ws, bs, lab = slice(0, h.nS), h.W, h.b, f'{tag} Sherpa block'
    elif block == 'H':
        cols, Ws, bs, lab = slice(h.nS, h.nS + h.nH), h.WH, h.bH, f'{tag} Herwig block'
    else:
        cols, Ws, bs, lab = slice(0, h.nt), h.W, h.b, tag

    def bvec(m, x):
        return h._bvec_net(Ws[m], bs[m], x) if kind in MIXTURE_KINDS else h._bvec(m, x)

    X = h.theta_train[:, cols]
    mu = X.mean(0)
    ev, V = np.linalg.eigh(np.cov((X - mu).T))       # ascending
    sd = np.sqrt(np.maximum(ev, 1e-12))
    rng = np.random.default_rng(0)
    # sample inside the design cloud rather than the box, so the derivatives are evaluated
    # where the head is meant to be used
    nd = X.shape[1]
    pts = mu + rng.normal(0, 0.6, size=(NPTS, nd)) @ V.T
    out = []
    for i in range(nd):
        v = V[:, i]
        g = [np.linalg.norm(
                 np.concatenate([bvec(m, t + EPS*v) for m in range(h.ENS)])
                 - np.concatenate([bvec(m, t - EPS*v) for m in range(h.ENS)]))/(2*EPS)
             for t in pts]
        out.append((float(sd[i]), float(np.mean(g))))
    R = np.array(out)
    weak, rest = R[:NWEAK], R[NWEAK:]
    ratio = weak[:, 1].mean()/rest[:, 1].mean()
    anis = sd.max()/sd.min()
    print(f'== {lab}: {nd} directions, design anisotropy {anis:.2f} in spread '
          f'({anis**2:.1f} in variance), whitened={int(np.load(hp).get("whiten", 0))}')
    for i, (s, g) in enumerate(out):
        mark = '  <- least explored' if i < NWEAK else ''
        print(f'   direction {i}: design sd {s:.3f}, |db/dv| {g:.3f}{mark}')
    print(f'   least explored {NWEAK}: design sd {weak[:, 0].mean():.3f}, |db/dv| {weak[:, 1].mean():.3f}')
    print(f'   the other {nd-NWEAK}:    design sd {rest[:, 0].mean():.3f}, |db/dv| {rest[:, 1].mean():.3f}')
    if anis < ISOTROPIC:
        verdict = (f'the design is nearly isotropic (anisotropy below {ISOTROPIC:g}), so the least '
                   f'explored directions are explored almost as much as the rest and this says little')
    else:
        verdict = ('healthy, responds least where it knows least' if ratio < 1 else
                   'PATHOLOGY, responds most where it knows least')
    print(f'   RATIO {ratio:.2f}   {verdict}\n')
    return ratio


def main(argv=None):
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # This file ships inside the release, so the default is relative to it.
    ap.add_argument('models', nargs='?',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
                    help='the folder of the heads (default: models/ next to this file)')
    ap.add_argument('tags', nargs='*', default=['E', 'F'], metavar='stage',
                    help='the stages to probe, such as E, F or MIX17 (default: E F)')
    a = ap.parse_args(argv)
    missing = [p for p in (f'{a.models}/{t}_head.npz' for t in a.tags) if not os.path.exists(p)]
    if missing:
        print(f'ERROR: {", ".join(missing)} {"does" if len(missing) == 1 else "do"} not exist. The '
              f'first argument is the folder of the heads, then the stages, as in: '
              f'python head_direction_response.py models E F')
        return 2
    res = {t: probe(a.models, t) for t in a.tags}
    flat = {}
    for t, r in res.items():
        if isinstance(r, dict):
            for k, v in r.items():
                if v is not None:
                    flat[f'{t}:{k}'] = v
        elif r is not None:
            flat[t] = r
    if flat:
        print('summary: ' + ', '.join(f'{t} {r:.2f}' for t, r in flat.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
