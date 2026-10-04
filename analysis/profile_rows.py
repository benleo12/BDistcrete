#!/usr/bin/env python3
"""The profile in each fit parameter with the other one treated as a continuous parameter.

The direct profile computes the chi^2 on a grid of (alpha_s, alpha_0) nodes, minimized over the
nuisance parameters at each node. Along the valley of the fit the best alpha_0 rises by about 0.03
for every step of 0.002 in the coupling, while the alpha_0 columns are 0.02 apart, so the valley
floor often falls between two columns and the minimum over a row overestimates it. A spline through
the nodes inherits this ripple, and the error on the coupling, read where the profile crosses its
minimum plus one, depends on it. Here the other fit parameter is minimized continuously together
with the nuisance parameters, with the theory targets and their covariance splined over the grid
(directlib.TargetInterp, exact at the nodes), so each one-dimensional profile is exact at its nodes.

  part   PART/NPART: tasks (variation, mode, node). 'row': alpha_s fixed at a node of the theory grid,
         L-BFGS-B over (u, alpha_0). 'col': alpha_0 fixed at a node, L-BFGS-B over (u, alpha_s). Each
         task starts from the best grid nodes of the fit along that row or column.
  merge  for every variation the profile in the coupling and (central only) in alpha_0, the values and
         errors from a cubic spline through the nodes and the crossings of minimum plus one, the
         envelopes over the variations, and the fitted point with its covariance from the curvature.

    PART=0 NPART=64 python profile_rows.py part output/profile_MIX17ext_central.json
    python profile_rows.py merge output/profile_MIX17ext_central.json
ROWS_WHICH: 'central' (default) or 'all' (central, the eleven scale and the three model variations).
ROW_STEP: spacing of the coupling rows (default: the grid nodes of the fit); the targets are splined
between the nodes of the theory grid, smooth in the coupling to a part in 10^4.
CENTRAL_ROWS: a merged *_rows.json of the central fit, whose row optima start every task as well.
POLISH_FROM: a merged file of the same tasks; every task then also starts from its own optimum and
those of its neighbours there, for passes repeated until no row or column improves.
COLS: 1 to add the alpha_0 profile of the central fit (columns every COL_STEP in alpha_0).
BOX_SHRINK: limit the generator parameters (not the mixing fraction) to this inner part of the box,
written with its own tag.
"""
import os, sys, json, glob, time
import numpy as np

mode, src = sys.argv[1], sys.argv[2]
fit = json.load(open(src)); stem = src.replace('.json', '')
SHR = float(os.environ.get('BOX_SHRINK', '1.0')); tagx = '' if SHR == 1.0 else f'_shrink{SHR:.2f}'
tagx = os.environ.get('ROWS_TAG', tagx)                 # a separate name for a further stage
ASF = np.array(fit['alphas']); A0F = np.array(fit['alpha0']); PF = np.array(fit['profile']); THF = np.array(fit['profile_theta'])
GRID = os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz')
G = np.load(GRID, allow_pickle=True)
LAB = ['central'] + [str(l) for l in G['scale_labels']] + [str(l) for l in G['np_labels']]
KIN = ['central'] + [f'scale:{v}' for v in range(len(G['scale_labels']))] + [f'np:{v}' for v in range(len(G['np_labels']))]
NSC = len(G['scale_labels'])
WHICH = KIN if os.environ.get('ROWS_WHICH', 'central') == 'all' else ['central']
COL_STEP = float(os.environ.get('COL_STEP', '0.01'))
t0 = time.time()

ROW_STEP = float(os.environ.get('ROW_STEP', '0'))       # 0: the coupling nodes of the fit; else this spacing
ROWS_AS = (ASF if ROW_STEP == 0 else
           np.round(np.arange(ASF[0], ASF[-1] + 1e-9, ROW_STEP), 5))
CENTRAL_ROWS = os.environ.get('CENTRAL_ROWS')            # a merged *_rows.json whose optima seed every task

def tasks():
    T = [(w, 'row', float(a)) for w in WHICH for a in ROWS_AS]
    if os.environ.get('COLS') == '1':
        # alpha_0 columns where the grid profile lies within 12 of its minimum, at the finer step
        near = A0F[np.any(PF <= PF.min() + 12, axis=0)]
        lo, hi = max(near.min() - 0.02, float(G['alpha0'][0])), min(near.max() + 0.02, float(G['alpha0'][-1]))
        T += [('central', 'col', float(round(x, 4))) for x in np.arange(lo, hi + 1e-9, COL_STEP)]
    return T

if mode == 'part':
    import torch
    from scipy.optimize import minimize
    sys.path.insert(0, '.')
    from directlib import Model, TargetInterp, chi2_continuous
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
              nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float32'), grid=GRID,
              last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
    lim = np.full(M.NT, SHR)
    if M.kind == 'mixture':
        lim[-1] = 1.0
    UF = (THF - M.NORM[:, 0])/M.NORM[:, 1]
    part, npart = int(os.environ['PART']), int(os.environ['NPART'])
    TIs = {}; out = []
    pol = {}
    if os.environ.get('POLISH_FROM'):
        # a polish pass: every task also starts from its own optimum and from those of its two
        # neighbouring rows (or columns) in a previous merged file of the same tasks
        PF_ = json.load(open(os.environ['POLISH_FROM']))
        for lab_, v in PF_['variations'].items():
            r_ = v['rows']; xs_ = np.round(r_['alphas'], 5)
            for k_, x_ in enumerate(xs_):
                pol[(KIN[LAB.index(lab_)], 'row', float(x_))] = [np.asarray(r_['y'][q], float) for q in (k_, k_ - 1, k_ + 1) if 0 <= q < len(xs_)]
        if 'alpha_0_profile' in PF_ and 'y' in PF_['alpha_0_profile']['cols']:
            c_ = PF_['alpha_0_profile']['cols']; xs_ = np.round(c_['alpha0'], 5)
            for k_, x_ in enumerate(xs_):
                pol[('central', 'col', float(x_))] = [np.asarray(c_['y'][q], float) for q in (k_, k_ - 1, k_ + 1) if 0 <= q < len(xs_)]
    seed = {}
    if CENTRAL_ROWS:
        cr = json.load(open(CENTRAL_ROWS))
        for a, y in zip(cr['variations']['central']['rows']['alphas'], cr['variations']['central']['rows']['y']):
            seed[round(a, 5)] = np.asarray(y, float)
    a0lo, a0hi = float(G['alpha0'][0]), float(G['alpha0'][-1]); aslo, ashi = float(G['alphas'][0]), float(G['alphas'][-1])
    for which, kind, x in tasks()[part::npart]:
        if which not in TIs:
            TIs[which] = TargetInterp(G, which)
        TI = TIs[which]
        if kind == 'row':
            i = int(np.argmin(np.abs(ASF - x))); order = np.argsort(PF[i])[:3]
            starts = [np.r_[np.clip(UF[i, j], -lim, lim), A0F[j]] for j in order]
            if seed:
                k = min(seed, key=lambda a: abs(a - x)); starts = [np.r_[np.clip(seed[k][:-1], -lim, lim), seed[k][-1]]] + starts[:2]
            to_x = lambda y: np.r_[y[:M.NT], x, y[-1]]; sel = lambda g: np.r_[g[:M.NT], g[-1]]
            bnds = [(-l, l) for l in lim] + [(a0lo, a0hi)]
        else:
            j = int(np.argmin(np.abs(A0F - x))); order = np.argsort(PF[:, j])[:3]
            starts = [np.r_[np.clip(UF[i, j], -lim, lim), ASF[i]] for i in order]
            to_x = lambda y: np.r_[y[:M.NT], y[-1], x]; sel = lambda g: np.r_[g[:M.NT], g[-2]]
            bnds = [(-l, l) for l in lim] + [(aslo, ashi)]
        key_ = (which, kind, round(x, 5))
        if key_ in pol:
            starts = [np.r_[np.clip(y[:-1], -lim, lim), y[-1]] for y in pol[key_]] + starts[:1]
        best = None; nfev = 0
        for y0 in starts:
            box = [None]
            def fg(y):
                f, g, lam = chi2_continuous(M, TI, to_x(y), box[0]); box[0] = lam
                return f, sel(g)
            r = minimize(fg, y0, jac=True, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
            nfev += int(r.nfev)
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy())
        out.append([which, kind, x, best[0], best[1].tolist(), nfev])
        print(f'  {which:10s} {kind} {x:.4f}: chi2 {best[0]:.3f} (other parameter {best[1][-1]:.4f}), {nfev} evaluations  [{time.time()-t0:.0f} s]', flush=True)
    json.dump(dict(part=part, results=out), open(f'{stem}_rows{tagx}_part{part}.json', 'w'))
    print('done', flush=True)

elif mode == 'merge':
    R = {}
    for f in glob.glob(f'{stem}_rows{tagx}_part*.json'):
        for which, kind, x, c2, y, nfev in json.load(open(f))['results']:
            R[(which, kind, x)] = (c2, y)
    missing = [t for t in tasks() if (t[0], t[1], t[2]) not in R]
    assert not missing, f'{len(missing)} tasks missing, e.g. {missing[:3]}'

    sys.path.insert(0, '.')
    from fitlib import profile_1d as prof1d
    res = dict(source=src, shrink=SHR, method='alpha_0 continuous at each coupling node (profile_rows.py)', variations={})
    for which in WHICH:
        xs = np.array(sorted(x for (w, k, x) in R if w == which and k == 'row'))
        ys = np.array([R[(which, 'row', x)][0] for x in xs]); a0s = np.array([R[(which, 'row', x)][1][-1] for x in xs])
        p = prof1d(xs, ys); p['alpha_0_at_min'] = float(np.interp(p['value'], xs, a0s))
        p['rows'] = dict(alphas=xs.tolist(), chi2=ys.tolist(), alpha0=a0s.tolist(), y=[R[(which, 'row', x)][1] for x in xs])
        res['variations'][LAB[KIN.index(which)]] = p
    c = res['variations']['central']
    print(f"central: alpha_s {c['value']:.5f} +{c['err_hi'] if c['err_hi'] is not None else float('nan'):.5f} "
          f"-{c['err_lo'] if c['err_lo'] is not None else float('nan'):.5f}, alpha_0 at the minimum {c['alpha_0_at_min']:.4f}, "
          f"chi2 {c['chi2min']:.2f}{'  EDGE' if c['hit_edge'] else ''}")
    print('   rows: ' + ' '.join(f'{a:.3f}:{v:.2f}' for a, v in zip(c['rows']['alphas'], c['rows']['chi2'])))
    if len(WHICH) > 1:
        d = {l: v['value'] - c['value'] for l, v in res['variations'].items() if l != 'central'}
        sc = [d[l] for l in LAB[1:1 + NSC]]; npv = [d[l] for l in LAB[1 + NSC:]]
        for l in LAB[1:]:
            v = res['variations'][l]
            print(f"  {l:28s} alpha_s {v['value']:.5f} ({d[l]:+.5f})  alpha_0 {v['alpha_0_at_min']:.4f}  chi2 {v['chi2min']:.2f}{'  EDGE' if v['hit_edge'] else ''}")
        res['summary'] = dict(pert_env_alpha_s=[float(min(sc)), float(max(sc))], pert_hss_alpha_s=float(np.sqrt(0.5*np.sum(np.square(sc)))),
                              np_env_alpha_s=[float(min(npv)), float(max(npv))], np_hss_alpha_s=float(np.sqrt(0.5*np.sum(np.square(npv)))),
                              n_up=int(np.sum(np.array(sc) > 0)), n_scale=NSC,
                              np_d_alpha_0=[float(res['variations'][l]['alpha_0_at_min'] - c['alpha_0_at_min']) for l in LAB[1 + NSC:]])
        s = res['summary']
        print(f"perturbative envelope [{s['pert_env_alpha_s'][0]:+.5f}, {s['pert_env_alpha_s'][1]:+.5f}], {s['n_up']} of {NSC} up; "
              f"model envelope [{s['np_env_alpha_s'][0]:+.5f}, {s['np_env_alpha_s'][1]:+.5f}], alpha_0 shifts {np.round(s['np_d_alpha_0'], 4)}")
    if os.environ.get('COLS') == '1':
        xs = np.array(sorted(x for (w, k, x) in R if w == 'central' and k == 'col'))
        ys = np.array([R[('central', 'col', x)][0] for x in xs]); a_s = np.array([R[('central', 'col', x)][1][-1] for x in xs])
        q = prof1d(xs, ys); q['alpha_s_at_min'] = float(np.interp(q['value'], xs, a_s))
        q['cols'] = dict(alpha0=xs.tolist(), chi2=ys.tolist(), alphas=a_s.tolist(), y=[R[('central', 'col', x)][1] for x in xs])
        res['alpha_0_profile'] = q
        # the correlation from the slopes of the two valley lines: d alpha_0/d alpha_s along the rows
        # (b_r) and along the columns (1/b_c); for a quadratic chi^2 rho^2 = b_r/b_c
        rr = res['variations']['central']['rows']; m = np.array(rr['chi2']) <= c['chi2min'] + 4
        b_r = np.polyfit(np.array(rr['alphas'])[m], np.array(rr['alpha0'])[m], 1)[0]
        mc = ys <= q['chi2min'] + 4; b_c = 1.0/np.polyfit(xs[mc], a_s[mc], 1)[0]
        rho = float(np.sign(b_r)*np.sqrt(max(min(b_r/b_c, 1.0), 0.0))) if b_c != 0 else float('nan')
        res['rho'] = rho
        print(f"alpha_0: {q['value']:.4f} +{q['err_hi'] if q['err_hi'] is not None else float('nan'):.4f} "
              f"-{q['err_lo'] if q['err_lo'] is not None else float('nan'):.4f} (coupling at its minimum {q['alpha_s_at_min']:.5f}); "
              f"correlation {rho:+.2f}")
    dst = f'{stem}_rows{tagx}.json'
    json.dump(res, open(dst, 'w'), indent=1); print('wrote', dst)
