#!/usr/bin/env python3
"""Compare a rebuilt mixture data set with the one a head was trained on.

The builder draws without replacement with one seeded stream, and the regenerated source runs
reproduce the old events bitwise, so a rebuild with the same seed and designs must give the
same events in the same order. Everything the training read (particles, mask, counts, nbaryon,
the design in meta.json) must be identical. The shapes may differ only in the columns that
depend on the thrust axis.

    python verify_mix_v2.py data_stageDM17aug data_stageDM17aug_v2
"""
import sys, json
import numpy as np, pandas as pd

old, new = sys.argv[1], sys.argv[2]
mo, mn = json.load(open(f'{old}/meta.json')), json.load(open(f'{new}/meta.json'))
for k in ('train', 'held'):
    assert len(mo[k]) == len(mn[k])
    for a, b in zip(mo[k], mn[k]):
        assert a['rid'] == b['rid'] and a['n_sherpa'] == b['n_sherpa'] and a['n_herwig'] == b['n_herwig'], (a['rid'], b['rid'])
        assert np.array_equal(np.array(a['theta']), np.array(b['theta'])), f'run {a["rid"]}: theta differs'
for k in ('seed', 's_axes', 'h_axes', 's_box', 'h_box', 'ntheta', 'n_train_events', 'n_held_events'):
    assert mo[k] == mn[k], f'meta {k} differs'
print('meta.json: design identical (thetas, fractions, counts, boxes, seed)', flush=True)

AXDEP = {'1_minus_thrust', 'thrust', 'B_total', 'B_wide', 'B_narrow', 'B_diff', 'rho_heavy', 'rho_light',
         'rho_diff', 'rho_sum', 'mult_heavy', 'mult_light'}
rows = []
for m in mo['train'] + mo['held']:
    rid = m['rid']
    zo, zn = np.load(f'{old}/particles_full_{rid:04d}.npz'), np.load(f'{new}/particles_full_{rid:04d}.npz')
    for k in zo.files:
        assert np.array_equal(zo[k], zn[k]), f'run {rid}: {k} differs'
    so, sn = pd.read_csv(f'{old}/shapes_run_{rid:04d}.csv'), pd.read_csv(f'{new}/shapes_run_{rid:04d}.csv')
    assert list(so.columns) == list(sn.columns) and len(so) == len(sn)
    for c in so.columns:
        if c in AXDEP:
            continue
        assert np.array_equal(so[c].values, sn[c].values), f'run {rid}: axis-independent column {c} differs'
    t0, t1 = so['1_minus_thrust'].values, sn['1_minus_thrust'].values
    rows.append((rid, np.mean(np.abs(t1 - t0) > 1e-9), t0.mean(), t1.mean()))
r = np.array(rows)
print(f'{len(r)} runs: every particle array identical, axis-independent shape columns identical', flush=True)
print(f'1-T changed on {100*r[:,1].mean():.2f}% of events (range {100*r[:,1].min():.2f}-{100*r[:,1].max():.2f}%), '
      f'mean old/exact - 1 = {100*(r[:,2]/r[:,3] - 1).mean():+.2f}%', flush=True)
print('MIXTURE V2 VERIFIED', flush=True)
