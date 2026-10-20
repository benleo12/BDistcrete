"""Design v3: more training runs between the existing points, and more held-out points.

Per generator, 256 new training points from a scrambled Sobol sequence (uniform over the same
box as the existing runs), the scramble chosen out of 400 to keep the new points farthest from
the existing training and held-out points, and 16 new held-out points from an independent
scrambled Sobol sequence (a typical point a user would ask for, not a worst case).
The run id is also the generator seed.

    python make_design_v3.py  ->  designs/stageE_design_v3.csv, designs/stageF_design_v3.csv
"""
import json, os, csv
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, '..', 'data_stagePURE17', 'meta.json')))
NTRAIN, NHELD, NSCRAMBLE = 256, 16, 400
# id ranges not used by any earlier design (Sherpa 7000-7095, 7200-7207, 7300-7304, 7400-7404,
# Herwig 8000-8147, 8200-8207, 8300-8304)
GEN = {
    'E': dict(axes=meta['s_axes'], box=meta['s_box'], col=slice(0, 8), pure=0, train0=7500, held0=7208),
    'F': dict(axes=meta['h_axes'], box=meta['h_box'], col=slice(8, 16), pure=1, train0=8400, held0=8208),
}
for g, c in GEN.items():
    lo, hi = np.array(c['box']).T
    old = np.array([r['theta'][c['col']] for r in meta['train'] + meta['held'] if r['theta'][-1] == c['pure']])
    u_old = (old - lo)/(hi - lo)
    assert u_old.min() > -1e-9 and u_old.max() < 1 + 1e-9, 'an existing point lies outside the box'
    best = None
    for s in range(NSCRAMBLE):
        u = qmc.Sobol(8, scramble=True, seed=1000*ord(g) + s).random(NTRAIN)
        dmin = min(cdist(u, u_old).min(), cdist(u, u)[np.triu_indices(NTRAIN, 1)].min())
        if best is None or dmin > best[0]:
            best = (dmin, s, u)
    dmin, s, u_tr = best
    u_he = qmc.Sobol(8, scramble=True, seed=777000 + ord(g)).random(NHELD)
    # nearest-neighbour distance of a held point to the training set, before and after
    nn_before = cdist(u_he, u_old).min(1); nn_after = cdist(u_he, np.r_[u_old, u_tr]).min(1)
    print(f'{g}: scramble {s}, min distance {dmin:.3f}; held points: median distance to nearest '
          f'training point {np.median(nn_before):.3f} -> {np.median(nn_after):.3f}')
    out = os.path.join(HERE, 'designs', f'stage{g}_design_v3.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['run_id', 'role'] + c['axes'])
        for i, x in enumerate(lo + u_tr*(hi - lo)): w.writerow([c['train0'] + i, 'train'] + [repr(float(v)) for v in x])
        for i, x in enumerate(lo + u_he*(hi - lo)): w.writerow([c['held0'] + i, 'held'] + [repr(float(v)) for v in x])
    print('  wrote', out)
