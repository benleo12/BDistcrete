#!/usr/bin/env python3
"""Inspect the events of one unit whose rebuild disagrees most with the stored CSV, and those whose
new value is negative.   python nb_outliers.py WORK UID [K]"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_core as C
WORK, UID = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
u = json.load(open(f'{WORK}/units/{UID}.json'))
z = np.load(u['npz']); P, M = z['particles'], z['mask'].astype(bool)
old = pd.read_csv(u['csv_in'])[list(C.OBS)].to_numpy(); new = pd.read_csv(u['csv_out'])[list(C.OBS)].to_numpy()
B = C.beam_mask(P, M); nb = B.sum(1)
d = old - new                                   # = with - without for beam events
cand = set(np.where(new[:, 2] < 0)[0]) | set(np.argsort(-np.abs(d[:, 2]))[:2])
idx = np.where(nb > 0)[0]
# recompute the residual with - stored on beam events with the largest |delta| plus negatives
for i in sorted(cand)[:K]:
    m = M[i]; f = P[i][m]; b = B[i][m]
    w, wo, s = C.event_delta(f, b)
    zb = f[b, 0].sum()
    print(f'event {i}: n {m.sum()} nbeam {nb[i]} z_beam {zb:.4f} (E {zb*s:.2f} GeV) scale {s:.2f} | stored {old[i].round(5)} '
          f'rebuilt with {np.round(w, 5)} without {np.round(wo, 5)} new {new[i].round(5)}')
    # beam photons: cos, phi, z
    print('     beam photons (z, cos, phi):', [tuple(np.round(x, 6)) for x in f[b][:, :3]],
          ' hardest particles z:', np.round(np.sort(f[:, 0])[::-1][:4], 4))
