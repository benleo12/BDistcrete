#!/usr/bin/env python3
"""The test job's comparison of the delta output (WORK/test/<uid>.csv) with the full recomputation
(WORK/test/<uid>_full.csv), row by row. The rows whose mask changes (the changed events of the unit's
test record) are computed on this node by both paths with the same code: they must be byte-identical.
Every other row of the delta output keeps B0's text, while the full path recomputes it on this node,
whose CPU may round differently from B0's: there 1_minus_thrust, B_total and rho_heavy may differ by up
to 1e-14 (the values are below 1, one rounding step is at most 1.1e-16, B0's code differs between this Mac
and the cluster by at most 6.66e-16, and its fast and full paths on one machine by at most 7.8e-16; the
smallest real change seen is 2.4e-9), and every other column must be byte-identical.
Such rows are counted and reported; a value that is not a finite number never passes.
    python nb3_equal.py WORK UID[,UID...]        exit 1 if a unit fails"""
import sys, json, math

WORK, UIDS = sys.argv[1], sys.argv[2].split(',')
SHAPES = ('1_minus_thrust', 'B_total', 'rho_heavy')
TOL = 1e-14
fail = 0
for uid in UIDS:
    a = open(f'{WORK}/test/{uid}.csv').read().split('\n')
    b = open(f'{WORK}/test/{uid}_full.csv').read().split('\n')
    changed = {d['row'] for d in json.load(open(f'{WORK}/test/{uid}.json'))['changed_events']}
    head = a[0].split(',')
    sc = [head.index(o) for o in SHAPES]
    n_round, worst, bad = 0, 0.0, []
    if a[0] != b[0] or len(a) != len(b) or (a[-1] == '') != (b[-1] == ''):
        bad.append('header, row count or trailing newline')
    else:
        for i, (x, y) in enumerate(zip(a[1:], b[1:])):
            if x == y:
                continue
            p, q = x.split(','), y.split(',')
            if i in changed or len(p) != len(q) or any(p[k] != q[k] for k in range(len(p)) if k not in sc):
                bad.append(i); continue
            try:
                v = [(float(p[k]), float(q[k])) for k in sc]
            except ValueError:
                bad.append(i); continue
            d = max(abs(x - y) for x, y in v)
            if not all(math.isfinite(x) and math.isfinite(y) for x, y in v) or not d <= TOL:
                bad.append(i); continue
            n_round += 1; worst = max(worst, d)
    if bad:
        fail = 1
        print(f'{uid} DIFFER: {len(bad)} rows, e.g. {bad[:5]}')
    elif n_round:
        print(f'{uid} EQUAL up to rounding in {n_round} rows outside the {len(changed)} changed events (largest {worst:.3g})')
    else:
        print(f'{uid} EQUAL ({len(changed)} changed events)')
sys.exit(fail)
