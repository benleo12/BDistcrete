#!/usr/bin/env python3
"""Rebuild a reference bundle's per-event observables from (new) shapes files, in the exact
event order the training run used, and verify the replay against the stored arrays first.

The non-LOWMEM export draws NREF_PER events per training run with ONE numpy stream seeded 0,
run after run in the STAGES train order, unsorted. Replaying that stream reproduces the order.

    python rebuild_ref_obs.py C_1M --verify                  # replay with the OLD shapes, must match
    python rebuild_ref_obs.py C_1M --shapes-dir data_stageC_v2 --anchor-dir data_stageC_1M_v2 --out output/models/C_1M_ref_v2.npz
"""
import sys, os, argparse, numpy as np, pandas as pd
sys.path.insert(0, '.')
import r2_ladder as R

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('--verify', action='store_true')
ap.add_argument('--shapes-dir', default=None); ap.add_argument('--anchor-dir', default=None)
ap.add_argument('--out', default=None); ap.add_argument('--nref-per', type=int, default=20000)
a = ap.parse_args()
grid = [(x, s, k) for x in [0.112, 0.120, 0.128] for s in [0.30, 0.46, 0.65] for k in [0.80, 1.21, 1.80]]
if a.tag == 'C_1M':
    train = dict(R.STAGES['C']['train']); train.update({7800 + i: t for i, t in enumerate(grid)})
else:
    # the ladder heads A, B, C: their own registry entries, NREF_PER of the training run
    train = dict(R.STAGES[a.tag]['train'])
order = list(train)                          # dict order: 6800..6826 then 7800..7826 for C_1M
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy']
ref = np.load(f'output/models/{a.tag}_ref.npz', mmap_mode='r')
src_dir = R.STAGES['C']['data'] if a.tag == 'C_1M' else R.STAGES[a.tag]['data']
src_dir = 'data_stageC' if src_dir.startswith('data_stageC') else src_dir
rng = np.random.default_rng(0)
idx = {}
for rid in order:
    n = len(np.load(f'{src_dir}/particles_full_{rid:04d}.npz', mmap_mode='r')['mask'])
    idx[rid] = rng.choice(n, min(a.nref_per, n), replace=False)
assert sum(len(v) for v in idx.values()) == len(ref['mask']), (sum(len(v) for v in idx.values()), len(ref['mask']))
def gather(shapes_of):
    out = {o: [] for o in OBS}
    for rid in order:
        sh = shapes_of(rid)
        for o in OBS: out[o].append(sh[o].values[idx[rid]])
    return {o: np.concatenate(v) for o, v in out.items()}
old = gather(lambda rid: pd.read_csv(f'{src_dir}/shapes_run_{rid:04d}.csv'))
for o in OBS:
    d = np.abs(old[o] - np.asarray(ref[o], float)).max()
    print(f'replay check {o:15s}: max |rebuilt - stored| = {d:.2e}')
if a.verify: sys.exit(0)
# the campaign map: 7800+i is anchor run_{i:02d} in the re-extracted directory
def new_shapes(rid):
    if rid >= 7800:
        return pd.read_csv(f'{a.anchor_dir}/shapes_run_{rid-7800:04d}.csv')
    return pd.read_csv(f'{a.shapes_dir}/shapes_run_{rid:04d}.csv')
new = gather(new_shapes)
tp = []
for rid in order:
    if rid >= 7800:
        import glob
        f = glob.glob(f'{a.anchor_dir}/particles_full_run_{rid-7800:02d}_*.npz')[0]
    else:
        f = f'{a.shapes_dir}/particles_full_{rid:04d}.npz'
    z = np.load(f); tp.append(z['tau_parton'][idx[rid]])
    # the events must be the same ones: compare the particle arrays at the drawn indices
    zo = np.load(f'{src_dir}/particles_full_{rid:04d}.npz', mmap_mode='r')
    # the stored file may carry fewer feature columns (Stage A predates the flavour columns), so
    # compare the columns both have, which are the kinematic ones and identify the events
    nc = min(z['particles'].shape[-1], zo['particles'].shape[-1])
    assert np.array_equal(np.asarray(z['particles'][idx[rid][:50], :, :nc]), np.asarray(zo['particles'][idx[rid][:50], :, :nc])), f'run {rid}: regenerated events differ from the stored ones'
tp = np.concatenate(tp)
d = {k: np.asarray(ref[k]) for k in ref.files}
for o in OBS: d[o] = new[o]
d['tau_parton'] = tp.astype(np.float32)
assert 'mult_total' in d, 'the bundle has no multiplicity array'   # nbaryon and strange exist only on the flavour stages
np.savez(a.out, **d)
for o in OBS: print(f'{o:15s} stored mean {np.asarray(ref[o]).mean():.5f}  exact mean {new[o].mean():.5f}  ({100*(np.asarray(ref[o]).mean()/new[o].mean()-1):+.2f}%)')
print(f'tau_parton mean {tp.mean():.5f}; wrote {a.out}')
