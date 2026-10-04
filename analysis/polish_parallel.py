#!/usr/bin/env python3
"""Neighbour polish of a column-wise direct profile, split across processes, then merged.

Every node whose chi^2 lies within DCHI_MAX of the minimum is re-optimized from the optima of
its four neighbours (in alpha_s and in alpha_0); an improvement replaces the node. Only that
region sets the fitted values and their errors, so polishing it is enough, and splitting the
nodes over NPART processes keeps the wall time short.

    PART=0 NPART=4 python polish_parallel.py part MIX17aug central       (one per process)
    python polish_parallel.py merge MIX17aug central                    (then the fit)

FROM_MERGED=1 starts a further pass from the merged profile, so passes can be repeated until no
node improves. NEIGH=8 adds the four diagonal neighbours as starting points.
"""
import os, sys, json, glob, time
import numpy as np, torch
sys.path.insert(0, '.')

mode, tag = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else 'central'
w_ = which.replace(':', '')
files = sorted(glob.glob(f'output/profile_{tag}_{w_}_a0*.json'))
cols = sorted((json.load(open(f)) for f in files), key=lambda d: d['alpha0'][0])
if os.environ.get('FROM_MERGED') == '1':
    # a further pass: start from the merged, already polished profile instead of the raw columns
    mg = json.load(open(f'output/profile_{tag}_{w_}.json'))
    A = np.array(mg['alphas']); A0 = np.array(mg['alpha0']); P = np.array(mg['profile']); TH = np.array(mg['profile_theta'])
else:
    A = np.array(cols[0]['alphas']); A0 = np.array([d['alpha0'][0] for d in cols])
    P = np.array([[d['profile'][i][0] if d['profile'][i][0] is not None else np.nan for d in cols] for i in range(len(A))], float)
    TH = np.array([[d['profile_theta'][i][0] for d in cols] for i in range(len(A))])
    # couplings never computed in any column (an extension that only went one way) are dropped
    keep = np.all(np.isfinite(P), axis=1)
    if not keep.all():
        print(f'dropping {int((~keep).sum())} coupling rows that no column computed: {A[~keep]}', flush=True)
    A, P, TH = A[keep], P[keep], TH[keep]
DCHI = float(os.environ.get('DCHI_MAX', '10'))
nA, n0 = P.shape

if mode == 'part':
    from directlib import Model
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '3')))
    part, npart = int(os.environ['PART']), int(os.environ['NPART'])
    M = Model(tag, export=cols[0]['export'], ref=cols[0]['ref'], first_bin=cols[0]['first_bin'],
              floor_rel=cols[0]['floor_rel'], nch_data=tuple(cols[0]['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float32'), grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid.npz'),
              last_bin=cols[0].get('last_bin'), use_nch=cols[0].get('use_nch', True), mix_form=cols[0].get('mix_form', 'additive'))
    U = (TH - M.NORM[:, 0])/M.NORM[:, 1]
    nodes = sorted([(ia, j) for ia in range(nA) for j in range(n0) if P[ia, j] <= P.min() + DCHI], key=lambda n: P[n])
    mine = nodes[part::npart]
    t0 = time.time(); out = []
    print(f'{tag} {which}: {len(nodes)} nodes within {DCHI} of the minimum, {len(mine)} in part {part}', flush=True)
    for k, (ia, j) in enumerate(mine):
        jj = int(np.where(np.isclose(M.G['alpha0'], A0[j]))[0][0])
        ia_g = int(np.where(np.isclose(M.G['alphas'], A[ia]))[0][0])      # the row's index on the theory grid
        starts = []
        nb = [(ia + 1, j), (ia - 1, j), (ia, j + 1), (ia, j - 1)]
        if os.environ.get('NEIGH') == '8':      # the diagonals too: the valley runs along alpha_s and alpha_0 together
            nb += [(ia + 1, j + 1), (ia - 1, j - 1), (ia + 1, j - 1), (ia - 1, j + 1)]
        for a, b in nb:
            if 0 <= a < nA and 0 <= b < n0 and not np.allclose(U[a, b], U[ia, j]) and not any(np.allclose(U[a, b], s) for s in starts):
                starts.append(U[a, b].copy())
        f, u, lam, ne = M.profile_node(ia_g, jj, starts, which=which)
        better = f < P[ia, j] - 1e-4
        out.append([ia, j, float(f), u.tolist(), bool(better), float(P[ia, j])])
        print(f'  node ({A[ia]:.3f}, {A0[j]:.2f}): {P[ia, j]:.3f} -> {f:.3f}{"  IMPROVED" if better else ""}  [{time.time()-t0:.0f} s]', flush=True)
    json.dump(dict(part=part, results=out), open(f'output/polish_{tag}_{w_}_part{part}.json', 'w'))
    print('done', flush=True)

elif mode == 'merge':
    import fitlib
    c = np.load(cols[0]['export']); NORM = c['norm'].astype(np.float64)
    n_imp = 0; worst = 0.0
    for f in glob.glob(f'output/polish_{tag}_{w_}_part*.json'):
        for ia, j, fv, u, better, old in json.load(open(f))['results']:
            if fv < P[ia, j] - 1e-4:
                worst = max(worst, P[ia, j] - fv); n_imp += 1
                P[ia, j] = fv; TH[ia, j] = NORM[:, 0] + NORM[:, 1]*np.array(u)
    print(f'{n_imp} nodes improved, largest improvement {worst:.3f}')
    pe = fitlib.profile_errors(P, A, A0); sp = fitlib.spline_min(P, A, A0); pb = fitlib.paraboloid_min(P, A, A0)
    ia0, j0 = np.unravel_index(np.argmin(P), P.shape)
    for nm in ('alpha_s', 'alpha_0'):
        q = pe[nm]
        print(f"profile: {nm} = {q['value']:.5f} +{q['err_hi'] if q['err_hi'] is not None else float('nan'):.5f} "
              f"-{q['err_lo'] if q['err_lo'] is not None else float('nan'):.5f}  chi2min {q['chi2min']:.2f}"
              f"{'  (UNCONSTRAINED on one side)' if q['hit_edge'] else ''}")
    if sp:
        print(f"spline: alpha_s {sp['alpha_s']:.5f} +- {sp['sa']:.5f}, alpha_0 {sp['alpha_0']:.4f} +- {sp['s0']:.4f}, rho {sp['rho']:+.3f}, chi2 {sp['chi2']:.2f}")
    d0 = cols[0]
    out = dict(tag=tag, targets=which, alphas=A.tolist(), alpha0=A0.tolist(), profile=P.tolist(), profile_theta=TH.tolist(),
               fit_profile=pe, fit_spline=sp, fit_paraboloid=pb, n_polished=n_imp,
               grid_min=dict(alpha_s=float(A[ia0]), alpha_0=float(A0[j0]), chi2=float(P[ia0, j0]), theta=TH[ia0, j0].tolist()),
               export=d0['export'], ref=d0['ref'], first_bin=d0['first_bin'], floor_rel=d0['floor_rel'],
               nch_data=d0['nch_data'], experiments=d0['experiments'], n_data=d0['n_data'], last_bin=d0.get('last_bin'), use_nch=d0.get('use_nch', True), mix_form=d0.get('mix_form', 'additive'))
    dst = f'output/profile_{tag}_{w_}.json'
    json.dump(out, open(dst, 'w'), indent=1); print('wrote', dst)
