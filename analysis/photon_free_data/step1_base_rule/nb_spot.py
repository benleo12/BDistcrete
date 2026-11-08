#!/usr/bin/env python3
"""Spot check of written CSVs against their originals, text level: same header and row count,
every column other than the three byte-identical, the three changed only in events with a beam
photon, and the change equal to the delta of the unit record's rebuild on a few events.
    python nb_spot.py WORK NUNITS SEED"""
import os, sys, json, glob, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_core as C

WORK, NU, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
recs = sorted(glob.glob(f'{WORK}/units/*.json'))
random.seed(SEED)
pick = random.sample(recs, min(NU, len(recs)))
bad = 0
for f in pick:
    u = json.load(open(f))
    a = open(u['csv_in']).read().split('\n'); b = open(u['csv_out']).read().split('\n')
    ok = len(a) == len(b) and a[0] == b[0]
    head = a[0].split(','); col = [head.index(o) for o in C.OBS]
    z = np.load(u['npz']); P, M = z['particles'], z['mask'].astype(bool)
    nb = C.beam_mask(P, M).sum(1)
    other_same = changed_wo_beam = 0
    for i, (x, y) in enumerate(zip(a[1:-1], b[1:-1])):
        p, q = x.split(','), y.split(',')
        if [v for k, v in enumerate(p) if k not in col] != [v for k, v in enumerate(q) if k not in col]:
            other_same += 1
        if nb[i] == 0 and x != y:
            changed_wo_beam += 1
    # recompute a few events from scratch: new = stored - (with - without)
    idx = np.where(nb > 0)[0][:: max(1, (nb > 0).sum() // 25)][:25]
    err = 0.0
    for i in idx:
        m = M[i]
        w, wo, s = C.event_delta(P[i][m], C.beam_mask(P[i:i + 1], M[i:i + 1])[0][m])
        st = np.array([float(a[1 + i].split(',')[c]) for c in col]); nw = np.array([float(b[1 + i].split(',')[c]) for c in col])
        err = max(err, float(np.abs(nw - (st - (np.array(w) - np.array(wo)))).max()))
    good = ok and other_same == 0 and changed_wo_beam == 0 and err < 1e-12
    bad += not good
    print(f"{u['uid']:40s} rows {len(b) - 2} header/rows ok {ok}, rows with another column changed {other_same}, "
          f"rows changed without a beam photon {changed_wo_beam}, recomputed max diff {err:.1e} -> {'OK' if good else 'BAD'}")
print('spot check:', len(pick), 'units,', bad, 'bad')
