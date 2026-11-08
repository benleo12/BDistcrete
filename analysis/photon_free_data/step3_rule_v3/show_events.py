#!/usr/bin/env python3
"""One event at a time, by hand: its massless neutral particles near the plane of axis and beam or on
the axis, with the v2 and v3 flags, the event's visible energy, missing momentum and hemisphere sizes,
and its row in the original CSV, the current _nb CSV and (if present) WORK/test/<uid>.csv.
    python show_events.py WORK uid:row [uid:row ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beam_photons as BP                # noqa: E402  v3
import beam_photons_v2 as BP2            # noqa: E402  v2

WORK = sys.argv[1]
rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
units = {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}
COLS = ('1_minus_thrust', 'B_total', 'rho_heavy', 'mult_total')


def row(path, i):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        head = fh.readline().rstrip('\n').split(',')
        for k, l in enumerate(fh):
            if k == i:
                q = l.rstrip('\n').split(',')
                return [q[head.index(c)] for c in COLS]


for arg in sys.argv[2:]:
    uid, i = arg.split(':'); i = int(i)
    u = units[uid]
    z = np.load(u['npz'])
    P, M = z['particles'][i:i + 1], z['mask'][i:i + 1].astype(bool)
    F2, I2 = BP2.beam_photon_mask(P, M, detail=True)
    F3, I3 = BP.beam_photon_mask(P, M, detail=True)
    s, miss, _ = BP._balance(P, M)
    ct = P[0, :, 1]
    print(f'== {uid} event {i}: kept {int(M.sum())}, hemispheres +{int((M[0] & (ct > 0)).sum())} -{int((M[0] & (ct < 0)).sum())}, '
          f's {s[0]:.6f} GeV, |sum p| {miss[0]:.3g} GeV; v3 info ' + str({k: v[0].item() for k, v in I3.items()}))
    neu = M[0] & (P[0, :, 4] == 0) & (P[0, :, 5] == 0) & (np.abs(P[0, :, 3] + 3) < 1e-4)
    for j in np.where(neu)[0]:
        ph = float(P[0, j, 2]); c = float(P[0, j, 1]); e = float(P[0, j, 0]) * s[0]
        if abs(abs(ph) - np.pi / 2) < 1e-3 or abs(c) > 0.999 or F2[0, j] or F3[0, j]:
            imp = BP._implied_cos(np.float64(c), np.float64(ph), e)
            print(f'   j {j:3d}  E {e:.6g}  cos {c:.9f}  phi {ph:.9f}  implied cos(axis,beam) {imp:.7f}  v2 {int(F2[0, j])}  v3 {int(F3[0, j])}')
    for name, path in (('original', u['csv_in']), ('_nb now', u['csv_out']), ('test', f'{WORK}/test/{uid}.csv')):
        r = row(path, i)
        if r is not None:
            print(f'   {name:9s} ' + '  '.join(f'{c} {v}' for c, v in zip(COLS, r)))
