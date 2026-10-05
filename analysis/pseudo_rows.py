#!/usr/bin/env python3
"""Closure of the fit on pseudo-data drawn from the anchored family itself, with the estimator of
profile_rows.py: at each coupling row alpha_0 and the nuisance parameters are minimized together,
and the coupling and its errors come from fitlib.profile_1d.

Each set takes a truth on the theory grid near the fitted point: (alpha_s, alpha_0) at a node of the
fit and the nuisance parameters at their profiled values there, so the pseudo-data resemble the
measurements. The anchored prediction at the truth, with Gaussian noise of the experiments' own
errors, replaces the thrust data and the multiplicity. The fit then starts without the truth: a
scan of the nuisance parameters and alpha_0 at the truth's coupling, rows chained outward from it,
and repeated restarts of every row from its neighbours until no row improves. It tests the
minimizer, the profile and the error convention. It does not test the theory, since the pseudo-data
are the theory.

    PSEUDO_SET=0 python pseudo_rows.py output/profile_MIX17ext_central.json
    python pseudo_rows.py summary output/pseudo_rows_sets/pseudo_rows_profile_MIX17ext_central_set*.json
"""
import os, sys, json, time, re
import numpy as np

if sys.argv[1] == 'summary':
    # every set counts whose minimum lies inside its rows and whose error crossing on the side of the
    # truth exists; the far-side crossing is not needed for the pull or the coverage. A file
    # *_wide.json (the same set rerun with longer rows) replaces the original.
    files = sorted(f for f in sys.argv[2:] if not f.endswith('_wide.json'))
    rows = []
    for f in files:
        w = f.replace('.json', '_wide.json')
        rows.append(json.load(open(w if os.path.exists(w) else f)))
    def judged(r):
        a = np.array(r['rows']['alphas']); interior = a[0] < r['alpha_s'] < a[-1] and r['chi2min'] < min(r['rows']['chi2'][0], r['rows']['chi2'][-1])
        below = r['alpha_s'] < r['truth']['alpha_s']; err = r['err_hi'] if below else r['err_lo']
        return interior and err is not None, err
    ok, pulls, cov, d, errs, dropped = [], [], [], [], [], []
    for r in rows:
        good, err = judged(r)
        if not good:
            dropped.append(r['set']); continue
        dd = r['alpha_s'] - r['truth']['alpha_s']; ok.append(r); d.append(dd); pulls.append(dd/err); cov.append(abs(dd) <= err)
        errs.append(err)
    d, pulls = np.array(d), np.array(pulls)
    print(f'{len(ok)} of {len(rows)} sets usable (minimum inside the rows, crossing on the side of the truth); dropped {dropped}')
    print(f'alpha_s: bias {d.mean():+.5f} +- {d.std(ddof=1)/np.sqrt(len(d)):.5f}, scatter {d.std(ddof=1):.5f}, median error on the side of '
          f'the truth {np.median(errs):.5f}; pull mean {pulls.mean():+.2f} rms {np.sqrt(np.mean(pulls**2)):.2f}; truth inside the error in {np.mean(cov):.0%}')
    print(f'chi2 at the minimum: mean {np.mean([r["chi2min"] for r in ok]):.1f} for {rows[0]["n_data"]} numbers')
    json.dump(dict(n=len(rows), n_ok=len(ok), dropped=dropped, bias=float(d.mean()), bias_err=float(d.std(ddof=1)/np.sqrt(len(d))),
                   scatter=float(d.std(ddof=1)), pull_mean=float(pulls.mean()), pull_rms=float(np.sqrt(np.mean(pulls**2))),
                   coverage=float(np.mean(cov)), median_error=float(np.median(errs))),
              open(re.sub(r'_set\d+\.json$', '_summary.json', files[0]), 'w'), indent=1)
    sys.exit()

import torch
from scipy.optimize import minimize
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, chi2_continuous
import fitlib

src = sys.argv[1]; fit = json.load(open(src))
kset = int(os.environ['PSEUDO_SET'])
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
GRID = os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz')
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float32'), grid=GRID,
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
TI = TargetInterp(M.G, 'central')
ASF = np.array(fit['alphas']); A0F = np.array(fit['alpha0']); THF = np.array(fit['profile_theta'])
iag = lambda a: int(np.argmin(np.abs(M.G['alphas'] - a))); jjg = lambda a0: int(np.argmin(np.abs(M.G['alpha0'] - a0)))
rng = np.random.default_rng(7000 + kset); t0 = time.time()
AS_RANGE = tuple(float(x) for x in os.environ.get('TRUTH_AS', '0.1135,0.1245').split(','))
A0_RANGE = tuple(float(x) for x in os.environ.get('TRUTH_A0', '0.345,0.515').split(','))
cand = [(i, j) for i in range(len(ASF)) for j in range(len(A0F)) if AS_RANGE[0] <= ASF[i] <= AS_RANGE[1] and A0_RANGE[0] <= A0F[j] <= A0_RANGE[1]]
i_t, j_t = cand[int(rng.integers(len(cand)))]
a_t, a0_t = float(ASF[i_t]), float(A0F[j_t]); u_t = (THF[i_t, j_t] - M.NORM[:, 0])/M.NORM[:, 1]

# pseudo-data at the truth, normalized over the fitted bins like the measurements
with torch.no_grad():
    lw, _, _, _, _ = M.weights(M.theta(torch.tensor(u_t)), iag(a_t), jjg(a0_t))
    w = torch.exp(lw)
    for x in M.exp:
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']])
        u = x['use']; d = cnt[u]/(x['wid'][u]*cnt[u].sum())
        y = x['y'].clone(); yp = d + x['e'][u]*torch.from_numpy(rng.normal(size=len(u)))
        y[u] = yp/(yp*x['wid'][u]).sum(); x['y'] = y
    M.NCH = (float(w @ M.NCHT) + M.NCH[1]*float(rng.normal()), M.NCH[1])
    c2_truth = float(M.chi2(w)[0])
print(f'set {kset}: truth alpha_s {a_t:.3f}, alpha_0 {a0_t:.2f}; chi2 of the truth on its pseudo-data {c2_truth:.1f} for {M.ndat}', flush=True)

STEP, DA = float(os.environ.get('ROW_STEP', '0.001')), float(os.environ.get('ROW_HALF', '0.010'))
lo_a, hi_a = float(M.G['alphas'][0]), float(M.G['alphas'][-1])
rows = np.round(np.arange(max(a_t - DA, lo_a), min(a_t + DA, hi_a) + 1e-9, STEP), 5); r_t = int(np.argmin(np.abs(rows - a_t)))
a0lo, a0hi = float(M.G['alpha0'][0]), float(M.G['alpha0'][-1])
bnds = [(-1, 1)]*M.NT + [(a0lo, a0hi)]
Y = [None]*len(rows); F = np.full(len(rows), np.inf); nfev = 0

def row_fit(r, starts):
    global nfev
    a = float(rows[r])
    for y0 in starts:
        box = [None]
        def fg(y):
            f, g, lam = chi2_continuous(M, TI, np.r_[y[:M.NT], a, y[-1]], box[0]); box[0] = lam
            return f, np.r_[g[:M.NT], g[-1]]
        res = minimize(fg, np.asarray(y0, float), jac=True, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
        nfev += int(res.nfev)
        if res.fun < F[r]:
            F[r] = float(res.fun); Y[r] = res.x.copy()

# a scan at the truth's coupling over the nuisance parameters and alpha_0 (theory-grid columns), no gradients
NS = int(os.environ.get('NSCAN', '192'))
U = (np.argsort(rng.random((NS, M.NT)), axis=0) + rng.random((NS, M.NT)))/NS*2 - 1
A0S = rng.choice(M.G['alpha0'][(M.G['alpha0'] >= 0.25 - 1e-9) & (M.G['alpha0'] <= 0.75 + 1e-9)], NS)
sc = []
lam = None
for u, a0 in zip(U, A0S):
    f, _, lam = M.chi2_at(u, iag(rows[r_t]), jjg(a0), None, grad=False)
    sc.append((f, np.r_[u, a0]))
sc.sort(key=lambda z: z[0])
row_fit(r_t, [y for _, y in sc[:int(os.environ.get('NBEST', '3'))]])
for r in list(range(r_t + 1, len(rows))) + list(range(r_t - 1, -1, -1)):
    row_fit(r, [Y[r - 1 if r > r_t else r + 1]])
print(f'  rows chained, minimum {F.min():.2f}  [{time.time()-t0:.0f} s]', flush=True)
for npass in range(int(os.environ.get('MAXPASS', '3'))):
    worst = 0.0
    for r in range(len(rows)):
        old = F[r]; st = [Y[q] for q in (r - 1, r + 1) if 0 <= q < len(rows) and Y[q] is not None]
        row_fit(r, st); worst = max(worst, old - F[r])
    print(f'  pass {npass + 1}: largest improvement {worst:.3f}, minimum {F.min():.2f}  [{time.time()-t0:.0f} s]', flush=True)
    if worst < 0.01:
        break
q = fitlib.profile_1d(rows, F)
res = dict(set=kset, truth=dict(alpha_s=a_t, alpha_0=a0_t, theta=THF[i_t, j_t].tolist()), alpha_s=q['value'], err_lo=q['err_lo'],
           err_hi=q['err_hi'], lo=q['lo'], hi=q['hi'], chi2min=q['chi2min'], chi2_truth=c2_truth, edge=q['hit_edge'],
           alpha_0_at_min=float(np.interp(q['value'], rows, [y[-1] for y in Y])), n_data=M.ndat, nfev=nfev,
           seconds=time.time() - t0, rows=dict(alphas=rows.tolist(), chi2=F.tolist()))
print(f'set {kset}: fit alpha_s {q["value"]:.5f} +{q["err_hi"] if q["err_hi"] is not None else float("nan"):.5f} '
      f'-{q["err_lo"] if q["err_lo"] is not None else float("nan"):.5f} against the truth {a_t:.3f}; chi2 {q["chi2min"]:.1f}'
      f'{"  EDGE" if res["edge"] else ""}  [{(time.time()-t0)/60:.1f} min, {nfev} evaluations]', flush=True)
dst = os.environ.get('PSEUDO_OUT', f'output/pseudo_rows_{fit["tag"]}_set{kset:03d}.json')
json.dump(res, open(dst, 'w'), indent=1); print('wrote', dst, flush=True)
