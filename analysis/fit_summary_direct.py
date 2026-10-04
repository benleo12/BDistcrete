#!/usr/bin/env python3
"""The numbers Sec. 6.3 and 6.4 quote about a direct-profile fit, at its fitted point.

At the fitted (alpha_s, alpha_0) of the continuous profiles (*_rows.json), with the nuisance
parameters minimized there (*_bands_point.json) and the targets splined between the theory nodes: the chi^2 of each experiment and of the multiplicity, the residuals of each
experiment inside the window in units of its errors, the KL divergence of the anchored sample from
the unanchored one, the mixing fraction, and which generator parameters lie at an edge of the box.
Along the valley of the profile (the optimum of each coupling row) it also lists the per-experiment
chi^2 and the fraction, to show what compensates a larger coupling.

    python fit_summary_direct.py output/profile_MIX17ext_central.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, logw_continuous, fitted_point

src = sys.argv[1]; fit = json.load(open(src))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float32'),
          grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
a_fit, a0_fit, u_fit, rr = fitted_point(src)
TI = TargetInterp(M.G, 'central')
try:
    sys.path.insert(0, 'release'); from gentune.axes import names
    NAMES = names({'MIX17aug': 'MIX17', 'MIX17ext': 'MIX17', 'C_1Mext': 'C_1M'}.get(fit['tag'], fit['tag']), M.NT)
except Exception:
    NAMES = [f'parameter {k}' for k in range(M.NT)]

def at(u, a, a0):
    with torch.no_grad():
        lw, lw0, _, _ = logw_continuous(M, TI, torch.tensor(np.asarray(u, float)), a, a0)
        w, w0 = torch.exp(lw), torch.exp(lw0)
        tot, per = M.chi2(w)
        kl = float((w*(lw - lw0)).sum())
        res = {}
        for x, (lo, hi, d, y, e) in zip(M.exp, M.densities(w).values()):
            res[x['name']] = dict(lo=lo.tolist(), hi=hi.tolist(), resid=((d - y)/e).tolist())
    return u, float(tot), {k: float(v) for k, v in per.items()}, kl, res

u, tot, per, kl, res = at(u_fit, a_fit, a0_fit)
TH = M.theta(torch.tensor(u)).numpy()
print(f'fitted alpha_s {a_fit:.5f}, alpha_0 {a0_fit:.4f}: chi2 {tot:.2f}')
print('  per experiment: ' + ', '.join(f'{k} {v:.1f}' + (f'/{len(M.exp[[x["name"] for x in M.exp].index(k)]["use"])}' if k != 'nch' else '') for k, v in per.items()))
print(f'  KL of the anchored sample from the unanchored one {kl:.5f} nats')
edges = [(n, float(t), float(uu)) for n, t, uu in zip(NAMES, TH, u) if abs(uu) > 0.995]
print(f'  {len(edges)} of {M.NT} nuisance parameters at an edge of the box: ' + ', '.join(f'{n} = {t:.4g} ({"upper" if uu > 0 else "lower"})' for n, t, uu in edges))
if M.kind == 'mixture':
    print(f'  mixing fraction {TH[-1]:.3f}')
for ex, r in res.items():
    inw = [(lo, hi, z) for lo, hi, z in zip(r['lo'], r['hi'], r['resid']) if hi <= 1/3 + 1e-9]
    print(f'  {ex} window residuals: ' + ' '.join(f'[{lo:.2f},{hi:.2f}] {z:+.1f}' for lo, hi, z in inw))
valley = []
R = rr['variations']['central']['rows']
for a, y in zip(R['alphas'], R['y']):
    y = np.asarray(y, float); _, t2, p2, _, _ = at(y[:M.NT], a, y[-1]); fr = float(M.theta(torch.tensor(y[:M.NT]))[-1])
    valley.append(dict(alpha_s=float(a), alpha_0=float(y[-1]), chi2=t2, per=p2, fraction=fr if M.kind == 'mixture' else None))
    print(f'  valley alpha_s {a:.3f}: alpha_0 {y[-1]:.3f}, chi2 {t2:.2f} (' + ', '.join(f'{k} {v:.1f}' for k, v in p2.items()) + ')'
          + (f', fraction {fr:.3f}' if M.kind == 'mixture' else ''))
json.dump(dict(alpha_s=a_fit, alpha_0=a0_fit, chi2=tot, per=per, kl=kl,
               edges=edges, residuals=res, valley=valley, theta=TH.tolist()),
          open(src.replace('.json', '_summary.json'), 'w'), indent=1)
print('wrote', src.replace('.json', '_summary.json'))
