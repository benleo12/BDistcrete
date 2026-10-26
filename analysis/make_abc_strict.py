"""The strict-test data directory of the ladder stages: every run of gen/data_abc linked, except the
16 held-out runs, which are replaced by real files holding the run plus its four extensions
(gen/data_abc_ext/k1..k4, seed rid + k*10^6), 400k events each, arrays and csv rows in one order.
    python make_abc_strict.py      -> maxent_code/data_abc_h400
"""
import os, glob
import numpy as np
MAXENT_DIR = os.environ.get('MAXENT_DIR', '/mnt/beegfs/bassi/reweighting/maxent_code')
R = os.path.dirname(MAXENT_DIR); SRC = f'{R}/gen/data_abc'; DST = f'{MAXENT_DIR}/data_abc_h400'
HELD = list(range(6200, 6206)) + list(range(6700, 6705)) + list(range(6900, 6905))
os.makedirs(DST, exist_ok=True)
for f in sorted(glob.glob(f'{SRC}/*')):
    b = os.path.basename(f); rid = int(b.split('_')[-1].split('.')[0])
    if rid in HELD: continue
    if os.path.lexists(f'{DST}/{b}'): os.remove(f'{DST}/{b}')
    os.symlink(f, f'{DST}/{b}')
for rid in HELD:
    srcs = [SRC] + [f'{R}/gen/data_abc_ext/k{k}' for k in range(1, 5)]
    zs = [np.load(f'{d}/particles_full_{rid}.npz') for d in srcs]
    arr = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    csvs = [open(f'{d}/shapes_run_{rid}.csv').read().splitlines() for d in srcs]
    rows = [csvs[0][0]] + [x for c in csvs for x in c[1:]]
    assert all(len(v) == len(rows) - 1 == 400000 for v in arr.values()), rid
    for b in (f'particles_full_{rid}.npz', f'shapes_run_{rid}.csv'):
        if os.path.lexists(f'{DST}/{b}'): os.remove(f'{DST}/{b}')
    np.savez_compressed(f'{DST}/particles_full_{rid}.npz', **arr)
    open(f'{DST}/shapes_run_{rid}.csv', 'w').write('\n'.join(rows) + '\n')
print(f'{DST}: {len(HELD)} held-out runs at 400k events, {len(os.listdir(DST))} files')
