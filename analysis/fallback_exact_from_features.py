#!/usr/bin/env python3
"""Exact event shapes rebuilt from a stored particle file, for runs whose HepMC cannot be
regenerated bitwise (Stage B runs 6614 to 6617 were produced by a recipe no script records).

Momenta are rebuilt from the stored thrust-frame features with the event energy taken as the
collision energy, and the shapes are recomputed with the exact thrust axis of the release
code. Measured against exact-from-HepMC on 5000 Sherpa events: means agree to +0.11 percent on
1-T, +0.05 on the broadening and -0.30 on the heavy jet mass, and per event to about 1e-3, far
below the closure binning. The stored particle file is copied with tau_parton set to NaN, since
the parton level is not recoverable from features.

    python fallback_exact_from_features.py data_stageB_full/particles_full_6614.npz data_stageB_v2 6614
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'release/generators')
import compute_efps as CE
src, dst_dir, rid = sys.argv[1], sys.argv[2], int(sys.argv[3])
ETOT = 91.2
z = np.load(src); P, M = z['particles'], z['mask'].astype(bool)
def rebuild(f, m):
    f = f[m]; zf, ct, ph = f[:, 0].astype(float), f[:, 1].astype(float), f[:, 2].astype(float)
    mass = 10.0**f[:, 3].astype(float) if f.shape[1] > 3 else np.zeros(len(f)); mass = np.where(mass <= 1.5e-3, 0.0, mass)
    E = zf*ETOT; p = np.sqrt(np.maximum(E**2 - mass**2, 0)); st = np.sqrt(np.maximum(1 - ct**2, 0))
    return np.column_stack([E, p*st*np.cos(ph), p*st*np.sin(ph), p*ct])
rows = [CE.compute_hemisphere_shapes(rebuild(P[i], M[i])) for i in range(len(P))]
df = pd.DataFrame(rows)
os.makedirs(dst_dir, exist_ok=True)
df.to_csv(f'{dst_dir}/shapes_run_{rid:04d}.csv', index=False)
out = {k: z[k] for k in z.files}; out['tau_parton'] = np.full(len(P), np.nan, np.float32); out['n_parton'] = np.zeros(len(P), np.int16)
np.savez_compressed(f'{dst_dir}/particles_full_{rid:04d}.npz', **out)
print(f'run {rid}: {len(P)} events, <1-T> {df["1_minus_thrust"].mean():.5f}, columns {list(df.columns)[:6]}..., wrote {dst_dir}/shapes_run_{rid:04d}.csv and the particle copy (tau_parton NaN)')
