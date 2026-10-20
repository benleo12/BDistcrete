#!/usr/bin/env python3
"""The closure width of the energy-fraction spectrum between two halves of one fresh run, the
statistical expectation of that width in App. B. Its uncertainty per bin counts the events that
contribute to the bin (xp_closure.width_xp), which slightly overstates the fluctuation of a
normalized spectrum, so the expectation lies below one. As for the other observables (App. B,
Single observables), the comparison half of each held-out run of Stage C is split into two random
halves, ten times per run. Writes output/xp_null.json.

    python xp_null.py ../gen/data_abc
"""
import sys, json
import numpy as np
from xp_closure import per_event_bincontents, width_xp, XP_BINS

DATA = sys.argv[1] if len(sys.argv) > 1 else '../gen/data_abc'
RUNS, SPLITS = range(6900, 6905), 10
res = {}
for rid in RUNS:
    z = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
    C = per_event_bincontents(z['particles'][:, :, 0], z['mask'], XP_BINS)
    rep = np.random.default_rng(1000 + rid).permutation(len(C))[len(C)//2:]   # the comparison half, as r2_ladder
    C = C[rep]; ws = []
    for s in range(SPLITS):
        p = np.random.default_rng(100*rid + s).permutation(len(C)); a, b = C[p[:len(C)//2]], C[p[len(C)//2:]]
        ws.append(width_xp(a, np.full(len(a), 1.0/len(a)), b)[0])
    res[str(rid)] = ws
    print(rid, f'{len(C)} events, halves of {len(C)//2}: mean width {np.mean(ws):.3f}', flush=True)
allw = [w for v in res.values() for w in v]
out = dict(data=DATA, runs=list(res), splits=SPLITS, per_run=res, mean=float(np.mean(allw)), sd=float(np.std(allw, ddof=1)))
json.dump(out, open('output/xp_null.json', 'w'), indent=1)
print(f'mean over {len(allw)} splits: {out["mean"]:.3f} (sd {out["sd"]:.3f})')
