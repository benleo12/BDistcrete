#!/usr/bin/env python3
"""Fit the accuracy of each released head, and name the points it does not describe.

A single worst-case percentage is a poor way to tell a user how accurate a head is: it is set
by one point and says nothing about where in the box that point was. What the closure data
actually supports is a floor plus a weak dependence on how far the target sits from the pooled
reference, measured in units of the held run's own statistical error, plus a small number of
named points that the law does not cover.

Fitted robustly, with soft down-weighting rather than rejection, so that the outliers are
identified by the fit instead of choosing it. Writes release/accuracy_law.json.

Input: output/closure_vs_distance.csv, one row per (stage, held run, observable).
"""
import json, os, os
import numpy as np
import pandas as pd

REL = os.environ.get('RELEASE_DIR', 'release')
OUTLIER = float(os.environ.get('OUTLIER_SIGMA', '3.0'))


def robust_line(x, y, iters=12):
    w = np.ones_like(x)
    a = b = 0.0
    s = 1.0
    for _ in range(iters):
        A = np.column_stack([np.ones_like(x), x])
        a, b = np.linalg.lstsq(w[:, None]*A, w*y, rcond=None)[0]
        r = y - (a + b*x)
        s = float(np.median(np.abs(r - np.median(r)))*1.4826) + 1e-9
        w = 1.0/np.sqrt(1.0 + (r/(2*s))**2)
    return float(a), float(b), s


def main():
    # Overridable so a measurement run can fit its own scratch table instead of the shared one.
    src = os.environ.get('CLOSURE_DIST_IN', 'output/closure_vs_distance.csv')
    t = pd.read_csv(src)
    out = {'units': {
        'displacement': 'how far the target mean sits from the pooled reference mean, in units '
                        'of the held run statistical error on that mean',
        'error': 'relative difference between the reweighted mean and the generator mean, percent'},
        'form': 'error_percent = floor + slope_per_sigma * displacement',
        'stages': {}}
    for tag, s in t.groupby('stage'):
        a, b, sc = robust_line(s.dist.values, s.rel.values)
        ex = (s.rel.values - (a + b*s.dist.values))/sc
        bad = s.assign(excess=ex).query('excess > @OUTLIER').sort_values('excess', ascending=False)
        out['stages'][tag] = dict(
            n_points=int(len(s)), floor_percent=a, slope_percent_per_sigma=b,
            robust_scatter_percent=sc,
            displacement_range=[float(s.dist.min()), float(s.dist.max())],
            expected_at=({str(d): a + b*d for d in (0, 10, 20, 30, 50)}),
            outliers=[dict(run=int(r.run), observable=r.obs, displacement_sigma=float(r.dist),
                           error_percent=float(r.rel), predicted_percent=float(a + b*r.dist),
                           excess_scatter_units=float(r.excess)) for _, r in bad.iterrows()])
        print(f'{tag}: floor {a:.3f}% + {b*100:.3f}% per 100 sigma, scatter {sc:.3f}%, '
              f'{len(bad)} outlier(s) beyond {OUTLIER:g} scatter units')
        for _, r in bad.iterrows():
            print(f'    run {int(r.run)} {r.obs}: {r.rel:.2f}% at {r.dist:.1f} sigma, '
                  f'{r.excess:.1f} units above the law')
    os.makedirs(REL, exist_ok=True)
    json.dump(out, open(f'{REL}/accuracy_law.json', 'w'), indent=1)
    print(f'wrote {REL}/accuracy_law.json')


if __name__ == '__main__':
    main()
