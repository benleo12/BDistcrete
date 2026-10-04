#!/usr/bin/env python3
"""Audit Task 2: the x_p same-run control backing the Sec. 5.2 quote "0.93 +- 0.09".

xp_closure.py computes the Stage C x_p closure width for each of the five held-out runs on
the FULL run (80k events), with XP_BINS = linspace(0, 0.35, 22), per-event bin contents, and
Kish-effective EVENT counts in the pooled-variance pull. The same-run branch here splits each
held run's events into two disjoint halves, treats one half as the "prediction" with unit
weights and the other as the "target", and pushes them through the IDENTICAL estimator
(xp_closure.width_xp with xp_closure.per_event_bincontents on the same fixed bins).

Reported: mean +- sd across the five runs (split seed 0, the primary), plus two more split
seeds to gauge the split-to-split spread. Writes output/xp_samerun.json only; does not touch
ladder_final.json.
"""
import json
import numpy as np
from xp_closure import per_event_bincontents, width_xp, XP_BINS, HELD, DATA

SPLIT_SEEDS = [0, 1, 2]


def main():
    res = {'bins': 'XP_BINS = linspace(0, 0.35, 22), 21 bins, MIN_EFF=5',
           'estimator': 'xp_closure.width_xp (pooled q from Kish-effective event counts, '
                        'sqrt(chi2/ndf) about zero, ndf = kept-1)',
           'construction': 'each held Stage C run (80k events, the same full runs the x_p '
                           'closure used) split into two disjoint halves; half 1 = prediction '
                           'with uniform weights, half 2 = target',
           'split_seeds': SPLIT_SEEDS, 'per_run': {}, 'per_seed_summary': {}}
    per_seed = {r: [] for r in SPLIT_SEEDS}
    for rid, theta in HELD.items():
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        z, m = d['particles'][:, :, 0], d['mask']
        n = len(m)
        res['per_run'][rid] = {'theta': list(theta), 'n_events': int(n), 'widths': {}}
        for r in SPLIT_SEEDS:
            pr = np.random.default_rng(1_000_000 + 1000*r + rid).permutation(n)
            h = n//2
            i1, i2 = pr[:h], pr[h:]
            c1 = per_event_bincontents(z[i1], m[i1], XP_BINS)
            c2 = per_event_bincontents(z[i2], m[i2], XP_BINS)
            w1 = np.full(len(i1), 1.0/len(i1))
            wdt, nb = width_xp(c1, w1, c2)
            res['per_run'][rid]['widths'][f'split_{r}'] = {'width': wdt, 'nbins': nb}
            per_seed[r].append(wdt)
            print(f'  run {rid} split {r}: width={wdt:.3f} ({nb} bins)')
    for r in SPLIT_SEEDS:
        v = np.array(per_seed[r])
        res['per_seed_summary'][f'split_{r}'] = {'mean': float(v.mean()),
                                                 'sd': float(v.std(ddof=1)),
                                                 'values': [float(x) for x in v]}
        print(f'split {r}: mean {v.mean():.3f} +- {v.std(ddof=1):.3f} across the 5 runs')
    allv = np.array([w for r in SPLIT_SEEDS for w in per_seed[r]])
    res['pooled_all_splits'] = {'mean': float(allv.mean()), 'sd': float(allv.std(ddof=1))}
    print(f'pooled over splits: {allv.mean():.3f} +- {allv.std(ddof=1):.3f}')
    json.dump(res, open('output/xp_samerun.json', 'w'), indent=1)
    print('XP SAMERUN DONE -> output/xp_samerun.json')


if __name__ == '__main__':
    main()
