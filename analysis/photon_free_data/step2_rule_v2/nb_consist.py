#!/usr/bin/env python3
"""Consistency of the recomputed CSVs of the merged and truncated data sets with those of their
sources, which were computed separately from the source particle files:
  data_stagePURE17_v3_h400_nb (48 held runs at 4e5) = run + data_ext_nb/k1..k4, as make_dataset_v3.py
  data_abc_h400_nb (16 held runs at 4e5)            = run + data_abc_ext_nb/k1..k4, as make_abc_strict.py
  data_stagePURE17_v3ev20_nb (720 runs at 2e4)      = first 20000 rows of the source run
For each, the same identity is first checked on the original CSVs.   python nb_consist.py WORK NPROC
"""
import os, sys, json
from multiprocessing import Pool

WORK, NPROC = sys.argv[1], int(sys.argv[2])
R = '/mnt/beegfs/bassi/reweighting'
MC, GEN = f'{R}/maxent_code', f'{R}/gen'


def nb(d):
    if d.startswith(GEN + '/'):
        rel = d[len(GEN) + 1:].split('/'); rel[0] += '_nb'
        return GEN + '/' + '/'.join(rel)
    return d + '_nb'


def lines(p):
    return open(p).read().split('\n')


def concat(paths, cut=0):
    cs = [lines(p) for p in paths]
    for c in cs:
        assert c[-1] == '', 'no trailing newline'
    rows = [cs[0][0]] + [x for c in cs for x in c[1:-1]]
    if cut:
        rows = rows[:cut + 1]
    return '\n'.join(rows) + '\n'


def job(a):
    kind, rid, merged, srcs, cut = a
    orig_ok = open(merged).read() == concat(srcs, cut)
    new_ok = open(nb(os.path.dirname(merged)) + '/' + os.path.basename(merged)).read() == \
        concat([nb(os.path.dirname(s)) + '/' + os.path.basename(s) for s in srcs], cut)
    return kind, rid, orig_ok, new_ok


tasks = []
for name, src_stage, ext, n in (('data_stagePURE17_v3_h400', 'data_stagePURE17_v3', 'data_ext', 4),
                                ('data_abc_h400', None, 'data_abc_ext', 4)):
    d = f'{MC}/{name}'
    for f in sorted(os.listdir(d)):
        p = f'{d}/{f}'
        if not (f.startswith('shapes_run_') and f.endswith('.csv')) or os.path.islink(p):
            continue
        rid = int(f[len('shapes_run_'):-4])
        first = os.path.realpath(f'{MC}/{src_stage}/{f}') if src_stage else f'{GEN}/data_abc/{f}'
        srcs = [first] + [f'{GEN}/{ext}/k{k}/{f}' for k in range(1, n + 1)]
        tasks.append((name, rid, p, srcs, 0))
d = f'{MC}/data_stagePURE17_v3ev20'
for f in sorted(os.listdir(d)):
    p = f'{d}/{f}'
    if f.startswith('shapes_run_') and f.endswith('.csv') and not os.path.islink(p):
        tasks.append(('data_stagePURE17_v3ev20', int(f[len('shapes_run_'):-4]), p,
                      [os.path.realpath(f'{MC}/data_stagePURE17_v3/{f}')], 20000))
with Pool(NPROC) as pool:
    res = pool.map(job, tasks, chunksize=1)
out = {}
for kind, rid, o, n in res:
    r = out.setdefault(kind, dict(runs=0, original_identity=0, new_identity=0, new_fail=[], original_fail=[]))
    r['runs'] += 1; r['original_identity'] += o; r['new_identity'] += n
    if not n: r['new_fail'].append(rid)
    if not o: r['original_fail'].append(rid)
json.dump(out, open(f'{WORK}/consistency.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v)
