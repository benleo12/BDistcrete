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

Measured with this script: the Sherpa head, whose eight-parameter box is isotropic, gives
0.59. The Herwig head, which varies two of its eight parameters at 30 percent of the amplitude
of the other six to follow published tune correlations, gives 2.05, and its total
multiplicity closure fails by 3.4 percent at the held points furthest off that locus.

Usage: python head_direction_response.py <release/models dir> <tag> [<tag> ...]
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'release'))
from gentune.head import Head, MixtureHead

NWEAK = int(os.environ.get('NWEAK', '2'))
NPTS = int(os.environ.get('NPTS', '64'))
EPS = 1e-3


def probe(models, tag, block=None):
    """block is None for a single-generator head. A mixture head has two parameter networks
    reading two disjoint blocks of theta, so the question has to be asked of each separately,
    with that block's own slice of the design: a 17-vector fed to a network expecting 8 is a
    shape error, and averaging the two blocks together would hide whichever one misbehaves."""
    hp = f'{models}/{tag}_head.npz'
    kind = str(np.load(hp).get('head_kind', 'cond'))
    h = (MixtureHead if kind == 'mixture' else Head)(hp)
    if h.theta_train is None:
        print(f'{tag}: no training design shipped with this head, cannot run')
        return None
    if kind == 'mixture' and block is None:
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
        return h._bvec_net(Ws[m], bs[m], x) if kind == 'mixture' else h._bvec(m, x)

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
    print(f'   RATIO {ratio:.2f}   {"healthy, responds least where it knows least" if ratio < 1 else "PATHOLOGY, responds most where it knows least"}\n')
    return ratio


if __name__ == '__main__':
    models = sys.argv[1] if len(sys.argv) > 1 else 'release/models'
    tags = sys.argv[2:] or ['E', 'F']
    res = {t: probe(models, t) for t in tags}
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
