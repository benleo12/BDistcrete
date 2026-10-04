#!/usr/bin/env python3
"""The numbers App. E quotes about the regularization of the tilt, at the fitted point of a
direct-profile fit.

At the fitted (alpha_s, alpha_0) of the continuous profiles, with the nuisance parameters minimized
there (*_bands_point.json) and the targets splined between the theory nodes:
  the condition number of the moment covariance C under the tilted density, in the raw
  tau^a ln^b tau coordinates, alone and with the penalty Sigma_c added (window fraction squared
  times the perturbative covariance plus the diagonal term);
  the number of numerically nonzero eigenvalues of the perturbative covariance (relative size
  above 1e-12);
  the change of the anchored means of 1-T, the charged multiplicity and the baryon count when the
  relative size of the diagonal term is varied from half to four times its nominal 0.2 percent.

    python floor_scan_direct.py output/profile_MIX17ext_central.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, fitted_point

src = sys.argv[1]; fit = json.load(open(src))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float64'),
          grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
a_fit, a0_fit, u, _ = fitted_point(src)
_c, _, _, _S, _, _ = TargetInterp(M.G, 'central')(a_fit, a0_fit)
cvec = np.asarray(_c, np.float64); Spert = np.asarray(_S, np.float64)
with torch.no_grad():
    lg = M.logit(M.theta(torch.tensor(u))); lw0 = lg - torch.logsumexp(lg, 0)
    res = {}
    for f in (0.001, 0.002, 0.004, 0.008):
        Sig = Spert + np.diag((f*np.abs(cvec))**2)
        lw, lam, Pw, its = M.tilt_on(lw0, M.IW, M.MW, cvec, Sig, torch.zeros(len(M.keys)))
        w = torch.exp(lw); o = M.observables(w)
        res[f] = {k: float(o[k]) for k in ('thrust', 'nch', 'nbaryon') if k in o}
        if f == 0.002:
            # the moment covariance under the tilted density, raw coordinates, and the penalty
            Fw = (M.MW - torch.as_tensor(cvec)[None, :]).double(); ww = w[M.IW].double()
            m = ww @ Fw; C = (Fw*ww[:, None]).T @ Fw - torch.outer(m, m)
            Sc = float(Pw)**2*Sig
            C = C.numpy(); kC = np.linalg.cond(C); kA = np.linalg.cond(C + Sc)
nom = res[0.002]
ev = np.linalg.eigvalsh(Spert); nz = int(np.sum(ev > 1e-12*ev.max()))
print(f'point ({a_fit:.5f}, {a0_fit:.4f}); condition number of C {kC:.1e}, of C + Sigma_c {kA:.1e}; '
      f'{nz} of {len(ev)} eigenvalues of the perturbative covariance above 1e-12 of the largest')
for f, r in res.items():
    print(f'  diagonal term {f:.3f}: ' + ', '.join(f'{k} {v:.6f} ({v - nom[k]:+.1e})' for k, v in r.items()))
dmax = {k: max(abs(r[k] - nom[k]) for r in res.values()) for k in nom}
print('largest change over half to four times the nominal size: ' + ', '.join(f'{k} {v:.1e}' for k, v in dmax.items()))
json.dump(dict(point=[a_fit, a0_fit], cond_C=kC, cond_C_plus_Sigma=kA, nonzero_eigen=nz,
               means={str(k): v for k, v in res.items()}, max_change=dmax),
          open(src.replace('.json', '_floorscan.json'), 'w'), indent=1)
