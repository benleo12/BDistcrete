#!/usr/bin/env python3
"""Per-point closure error against how far the target sits from the pooled reference.

This is the input to fit_accuracy_law.py. Distance is the only quantity that makes different
stages comparable: how far the held run's mean sits from the reference mean, in units of the
statistical error on that mean. It replaces an earlier script that knew the two
single-generator stages by name and found their held runs through their design CSVs, which the
mixture does not have.

Held runs and the design come from the same place the ladder and the packager use, so a stage
cannot be described here with a different box than it was trained with.

Usage: python closure_vs_distance.py [<stage> ...]     (default: every stage with a closure csv)
"""
import glob, os, sys
import numpy as np
import pandas as pd

from r2_ladder import STAGES
from mixture_cfg import register as _register_mixture

REL = os.environ.get('RELEASE_DIR', 'release')
ALIAS = {'C_1M': 'C'}
OBS = ['mult_total', '1_minus_thrust', 'B_total', 'rho_heavy']


def stage_cfg(tag):
    # Export names are a stage key plus a suffix, so the design is the longest registered key
    # that prefixes the tag. The old line consulted a hand-kept alias table and then, for
    # anything it did not know, registered the MIXTURE design under that tag regardless of what
    # kind of head it was. That is how 'Faug', an eight-parameter Herwig head, came to be handed
    # the seventeen-parameter mixture design with its ten held runs, and collected no points.
    dtag = tag if tag in STAGES else ALIAS.get(tag)
    if dtag not in STAGES:
        cands = [k for k in STAGES if tag.startswith(k) and len(tag) > len(k)]
        if cands:
            dtag = max(cands, key=len)
    if dtag not in STAGES and _register_mixture(STAGES, [tag]) is not None:
        dtag = tag
    return STAGES.get(dtag)


def main(tags):
    """tags are export names, optionally EXPORT=PUBLIC_NAME.

    The displacement is computed from the head's own reference bundle at
    output/models/<tag>_ref.npz, and the error comes from release/closure_<tag>.csv, so both have
    to name the SAME head. When a retrained export is published under an older stage's name, the
    release-side files carry the public name while the export keeps its own, and asking for the
    public name alone silently paired the new head's errors with the old head's reference. That
    is not a small difference: it moved stage F's fitted floor from 0.386 to 0.554 percent and
    one point's displacement from 39 sigma to 5. Pass Fauglong=F to read the reference and the
    closure that belong together and label the result F.
    """
    rename = {}
    clean = []
    for t in (tags or []):
        if '=' in t:
            a, b = t.split('=', 1)
            rename[a] = b
            clean.append(a)
        else:
            clean.append(t)
    tags = clean or sorted(os.path.basename(p)[len('closure_'):-4]
                           for p in glob.glob(f'{REL}/closure_*.csv'))
    rows = []
    for tag in tags:
        cfg = stage_cfg(tag)
        pub = rename.get(tag, tag)
        cl = f'{REL}/closure_{pub}.csv'
        rp = f'output/models/{tag}_ref.npz'
        if cfg is None or not (os.path.exists(cl) and os.path.exists(rp)):
            print(f'{tag}: skipped (design {"yes" if cfg else "no"}, '
                  f'closure {os.path.exists(cl)}, reference {os.path.exists(rp)})')
            continue
        r = np.load(rp)
        refmean = {o: (float(r[o].astype(float).mean()) if o in r.files else
                       float(r['mask'].sum(1).astype(float).mean()) if o == 'mult_total' else None)
                   for o in OBS}
        C = pd.read_csv(cl)
        n = 0
        for rid in cfg['held']:
            f = f'{cfg["data"]}/shapes_run_{rid:04d}.csv'
            if not os.path.exists(f):
                continue
            sh = pd.read_csv(f)
            for o in OBS:
                if o not in sh.columns or refmean.get(o) is None:
                    continue
                v = sh[o].values.astype(float)
                err = v.std()/np.sqrt(len(v))
                sub = C[(C.run == rid) & (C.obs == o)]
                if not len(sub) or err == 0:
                    continue
                rows.append(dict(stage=pub, run=int(rid), obs=o,
                                 dist=abs(v.mean() - refmean[o])/err,
                                 pull=abs(float(sub.pull.iloc[0])),
                                 rel=abs(float(sub.rel.iloc[0]))))
                n += 1
        print(f'{tag}: {n} points from {len(cfg["held"])} held runs')
    t = pd.DataFrame(rows)
    # Refuse to write an empty table. Writing unconditionally replaced a good 64-row file with a
    # one-byte one, and the print below then raised AttributeError on an empty frame's missing
    # 'stage' column, so the failure looked like a crash rather than the deletion it was.
    if not len(t):
        print('no points collected, so nothing is written. The per-point closure CSV for each '
              'tag has to exist first, and the tag has to resolve to a registered design.')
        return 2
    # Overridable, because the measurement chain runs this into a scratch directory and must not
    # touch the copy the accuracy law is fitted from.
    out = os.environ.get('CLOSURE_DIST_OUT', 'output/closure_vs_distance.csv')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    t.to_csv(out, index=False)
    print(f'wrote {out}, {len(t)} rows, stages {sorted(t.stage.unique())}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
