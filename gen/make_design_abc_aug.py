"""Densified designs for the ladder stages A, B and C: the original training grid of each stage
plus Sobol points in its box (the extent of the original grid), the scramble chosen out of 200 to
keep the new points farthest from the existing training and held-out points. Held-out points are
unchanged. Writes designs/stage{A,B,C}_design_aug.csv (original rows + new training rows).
    python make_design_abc_aug.py
"""
import csv, os
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = {'A': (64, 30000), 'B': (128, 31000), 'C': (256, 32000)}
for s, (n, rid0) in NEW.items():
    rows = list(csv.reader(open(f'{HERE}/designs/stage{s}_design.csv')))
    hdr, body = rows[0], rows[1:]
    X = np.array([[float(x) for x in r[2:]] for r in body]); tr = np.array([r[1] == 'train' for r in body])
    lo, hi = X[tr].min(0), X[tr].max(0)
    u_old = (X - lo)/(hi - lo); d = X.shape[1]
    best = None
    for k in range(200):
        u = qmc.Sobol(d, scramble=True, seed=9000*ord(s) + k).random(n)
        dm = min(cdist(u, u_old).min(), cdist(u, u)[np.triu_indices(n, 1)].min())
        if best is None or dm > best[0]: best = (dm, k, u)
    dm, k, u = best
    with open(f'{HERE}/designs/stage{s}_design_aug.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in body: w.writerow(r)
        for i, x in enumerate(lo + u*(hi - lo)): w.writerow([rid0 + i, 'train'] + [repr(float(v)) for v in x])
    print(f'{s}: {tr.sum()} grid + {n} new training runs in box {np.round(lo, 4).tolist()} - {np.round(hi, 4).tolist()}, '
          f'scramble {k}, min distance {dm:.3f}; ids {rid0}-{rid0+n-1}')
