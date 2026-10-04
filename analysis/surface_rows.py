#!/usr/bin/env python3
"""The two-parameter chi^2 surface of a fit in (alpha_s, alpha_0), for the confidence regions of
Fig. anchored_fit (c).

At every point of a grid in the coupling and alpha_0 the nuisance parameters are minimized, with the
theory targets and their covariance splined between the nodes of the theory grid
(directlib.TargetInterp), as in profile_rows.py. Each coupling row starts in the valley, from the
optimum of the continuous profile (*_rows.json) at that coupling, and walks outward in alpha_0, every
point started from its neighbour. A polish pass restarts every point from its own optimum and those of
its four neighbours.

  part   PART/NPART: a share of the coupling rows, or with POLISH_FROM a share of the points
  merge  the surface, written as *_surface.json

    PART=0 NPART=21 python surface_rows.py part output/profile_MIX17ext_central.json
    python surface_rows.py merge output/profile_MIX17ext_central.json
SURF_AS, SURF_A0: lo,hi,step of the grid (default 0.110,0.130,0.001 and 0.30,0.54,0.01).
POLISH_FROM: a merged *_surface.json.
"""
import os, sys, json, glob, time
import numpy as np

mode, src = sys.argv[1], sys.argv[2]
fit = json.load(open(src)); stem = src.replace('.json', '')
rr = json.load(open(stem + '_rows.json'))['variations']['central']['rows']
rng_ = lambda key, d: [float(x) for x in os.environ.get(key, d).split(',')]
lo, hi, st = rng_('SURF_AS', '0.110,0.130,0.001'); AS = np.round(np.arange(lo, hi + 1e-9, st), 5)
lo, hi, st = rng_('SURF_A0', '0.30,0.54,0.01'); A0 = np.round(np.arange(lo, hi + 1e-9, st), 4)
t0 = time.time()

if mode == 'part':
    import torch
    from scipy.optimize import minimize
    sys.path.insert(0, '.')
    from directlib import Model, TargetInterp, chi2_continuous
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    GRID = os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz')
    M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
              nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float32'), grid=GRID,
              last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
    TI = TargetInterp(M.G, 'central')
    part, npart = int(os.environ['PART']), int(os.environ['NPART'])

    def fit_point(a, a0, starts):
        best = None
        for u0 in starts:
            box = [None]
            def fg(u):
                f, g, lam = chi2_continuous(M, TI, np.r_[u, a, a0], box[0]); box[0] = lam
                return f, g[:M.NT]
            r = minimize(fg, np.asarray(u0, float), jac=True, method='L-BFGS-B', bounds=[(-1, 1)]*M.NT,
                         options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy())
        return best

    out = []
    if not os.environ.get('POLISH_FROM'):
        xs = np.array(rr['alphas'])
        for i in range(part, len(AS), npart):
            a = float(AS[i]); q = int(np.argmin(np.abs(xs - a)))
            y = np.asarray(rr['y'][q], float); k0 = int(np.argmin(np.abs(A0 - y[-1])))
            U = {}
            f, u = fit_point(a, float(A0[k0]), [y[:M.NT]]); U[k0] = u; out.append([a, float(A0[k0]), f, u.tolist()])
            for ks in (range(k0 + 1, len(A0)), range(k0 - 1, -1, -1)):
                prev = k0
                for k in ks:
                    f, u = fit_point(a, float(A0[k]), [U[prev]]); U[k] = u; prev = k
                    out.append([a, float(A0[k]), f, u.tolist()])
            print(f'  row {a:.3f}: minimum {min(o[2] for o in out if o[0] == a):.3f}  [{time.time()-t0:.0f} s]', flush=True)
    else:
        P = json.load(open(os.environ['POLISH_FROM'])); C = np.array(P['chi2']); Uo = np.array(P['u'])
        assert np.allclose(P['alphas'], AS) and np.allclose(P['alpha0'], A0), 'POLISH_FROM is on another grid'
        pts = [(i, k) for i in range(len(AS)) for k in range(len(A0))][part::npart]
        for i, k in pts:
            nb = [(i + di, k + dk) for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + di < len(AS) and 0 <= k + dk < len(A0)]
            # only neighbours that are lower than this point can lead it to a lower minimum quickly
            starts = [Uo[i, k]] + [Uo[p] for p in nb if C[p] < C[i, k] - 0.01]
            if len(starts) == 1:
                out.append([float(AS[i]), float(A0[k]), float(C[i, k]), Uo[i, k].tolist()]); continue
            f, u = fit_point(float(AS[i]), float(A0[k]), starts[1:])
            if f >= C[i, k]:
                f, u = float(C[i, k]), Uo[i, k]
            out.append([float(AS[i]), float(A0[k]), f, np.asarray(u).tolist()])
        print(f'  polished {len(pts)} points  [{time.time()-t0:.0f} s]', flush=True)
    json.dump(out, open(f'{stem}_surface_part{part}.json', 'w'))
    print('done', flush=True)

elif mode == 'merge':
    C = np.full((len(AS), len(A0)), np.nan); U = [[None]*len(A0) for _ in AS]
    for fpath in glob.glob(f'{stem}_surface_part*.json'):
        for a, a0, f, u in json.load(open(fpath)):
            i, k = int(np.argmin(np.abs(AS - a))), int(np.argmin(np.abs(A0 - a0)))
            if np.isnan(C[i, k]) or f < C[i, k]:
                C[i, k] = f; U[i][k] = u
    assert not np.isnan(C).any(), f'{int(np.isnan(C).sum())} points missing'
    dst = f'{stem}_surface.json'
    old = json.load(open(dst)) if os.path.exists(dst) and os.environ.get('POLISH_FROM') else None
    i, k = np.unravel_index(np.argmin(C), C.shape)
    msg = f'surface {len(AS)} x {len(A0)}: minimum {C.min():.3f} at alpha_s {AS[i]:.3f}, alpha_0 {A0[k]:.2f}'
    if old is not None:
        msg += f'; largest improvement over the previous pass {np.max(np.array(old["chi2"]) - C):.3f}'
    print(msg)
    json.dump(dict(alphas=AS.tolist(), alpha0=A0.tolist(), chi2=C.tolist(), u=U, source=src,
                   method='nuisance parameters minimized at every point, targets splined (surface_rows.py)'),
              open(dst, 'w'))
    print('wrote', dst)
