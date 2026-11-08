#!/usr/bin/env python3
"""Cross-check against an independent implementation: lmx_tests/photons/photon_shapes.py (earlier
test of this workflow, read only) rebuilt the same runs of gen/data_v3 and stored per event the CSV
value ('csv'), its own rebuild with ('w') and without ('wo') the beam photons (|phi| within 2e-6 of
pi/2, E_vis from two passes of the momentum balance).
Compares, per event, the new CSV value with  csv - (w - wo)  (the same subtraction from the other
code) and with wo (its direct rebuild).    python nb_xcheck.py WORK"""
import os, sys, json
import numpy as np, pandas as pd

WORK = sys.argv[1]
T = '/mnt/beegfs/bassi/reweighting/lmx_tests/photons/out'
G = '/mnt/beegfs/bassi/reweighting/gen'
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy']
rows = []
for f in sorted(os.listdir(T)):
    if not f.endswith('.npz'):
        continue
    rid = int(f[len('photons_'):-4])
    if not os.path.exists(f'{WORK}/units/data_v3__{rid}.json'):
        continue
    t = np.load(f'{T}/{f}')
    assert list(t['obs'][:3]) == OBS
    new = pd.read_csv(f'{G}/data_v3_nb/shapes_run_{rid}.csv')[OBS].to_numpy(np.float64)
    old = pd.read_csv(f'{G}/data_v3/shapes_run_{rid}.csv')[OBS].to_numpy(np.float64)
    assert np.array_equal(old, t['csv'][:, :3]), rid
    d1 = new - (t['csv'][:, :3] - (t['w'][:, :3] - t['wo'][:, :3]))
    d2 = new - t['wo'][:, :3]
    rec = dict(rid=rid, n=len(new))
    for j, o in enumerate(OBS):
        a1, a2 = np.abs(d1[:, j]), np.abs(d2[:, j])
        rec[o] = dict(same_method_max=float(a1.max()), same_method_gt1e9=int((a1 > 1e-9).sum()),
                      same_method_gt1e6=int((a1 > 1e-6).sum()), same_method_gt1e4=int((a1 > 1e-4).sum()),
                      direct_q50=float(np.quantile(a2, .5)), direct_q99=float(np.quantile(a2, .99)),
                      direct_max=float(a2.max()),
                      mean_change_mine=float(new[:, j].mean() / old[:, j].mean() - 1),
                      mean_change_direct=float(t['wo'][:, j].mean() / old[:, j].mean() - 1))
    rows.append(rec)
    print(rid, ' '.join(f"{o[:6]}: same-method max {rec[o]['same_method_max']:.1e} (>1e-6: {rec[o]['same_method_gt1e6']}, >1e-4: {rec[o]['same_method_gt1e4']}), "
                         f"direct q50 {rec[o]['direct_q50']:.1e} q99 {rec[o]['direct_q99']:.1e}; mean change {100*rec[o]['mean_change_mine']:+.3f}% vs direct {100*rec[o]['mean_change_direct']:+.3f}%" for o in OBS))
json.dump(rows, open(f'{WORK}/xcheck_photon_test.json', 'w'), indent=1)
print(len(rows), 'runs compared')
