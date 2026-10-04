#!/usr/bin/env python3
"""The parton-level check for a direct-profile fit, and the shower-cutoff dependence.

At the fitted point (alpha_s, alpha_0, theta*), as in parton_check.py:
  1. reweight the sample at PARTON level to the purely perturbative windowed moments at alpha_s
     (the matched NNLL+NNLO calculation with no shift), so that the parton-level window shape is
     the calculation's and the hadron level is the generator's own transfer of it;
  2. compare the resulting HADRON-level windowed moments with the theory's hadron-level moments
     (dispersive shift at the fitted alpha_0), in units of the theory uncertainty;
  3. the implied alpha_0: where the theory's windowed mean equals the generator-transferred one;
  4. the coupling with alpha_0 held at the implied value, from the alpha_0 profile of the fit
     (*_rows.json: at each alpha_0 the coupling minimizing chi^2).
The fitted point is the minimum of the continuous profiles, with the nuisance parameters minimized
there (*_bands_point.json); the calculation, tabulated at the coupling nodes, is interpolated
linearly between the two nodes around it.
For a mixture head the same is repeated at a pure Herwig point (fraction one) across Herwig's
shower cutoff, since the parton level of a shower is defined by its cutoff and the calculation's
is not: the implied alpha_0 as a function of the cutoff measures that dependence.

    python parton_direct.py output/profile_MIX17aug_central.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, fitted_point
import np_shift as N, thrust_chain as TC
from anchor_targets_window import WLO, WHI, keys, FUN
from scipy.interpolate import RectBivariateSpline

src = sys.argv[1]
fit = json.load(open(src))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(fit['tag'], export=fit['export'], ref=os.environ.get('REF_PATH', fit['ref']), first_bin=fit['first_bin'],
          floor_rel=fit['floor_rel'], nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float64'),
          grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
assert 'tau_parton' in M.extra, 'the reference bundle has no parton-level thrust'
a_fit, a0_fit, u_fit, rr = fitted_point(src)
th = M.theta(torch.tensor(u_fit)).numpy()
AN = [float(x) for x in TC.ASGRID]; kb = int(np.clip(np.searchsorted(AN, a_fit) - 1, 0, len(AN) - 2))
fb = (a_fit - AN[kb])/(AN[kb + 1] - AN[kb])
TF = np.linspace(WLO, WHI, 200001); TMID = 0.5*(TF[1:] + TF[:-1]); FW = np.array([FUN[k](TMID) for k in keys])
def moments(asmz, a0=None):
    tau, m = TC.build(1.0, 1.0, 'logR', asmz, 1.0); S, _ = N.sanitize(m/m[-1])
    t = tau if a0 is None else tau + N.shift(1.0, a0, asmz=asmz)
    dS = np.diff(N.cum_eval(TF, t, S)); return (FW @ dS)/dS.sum()
def moments_fit(a0=None):
    """moments at the fitted coupling, linear between the two tabulated nodes around it"""
    return (1 - fb)*moments(TC.ASGRID[kb], a0) + fb*moments(TC.ASGRID[kb + 1], a0)
c_pert = moments_fit(); c_had = moments_fit(a0_fit)
Sig = TargetInterp(M.G, 'central')(a_fit, a0_fit)[3]; sd = np.sqrt(np.diag(Sig))
tp = M.extra['tau_parton']; inp = (tp >= WLO) & (tp < WHI)
IWp = torch.from_numpy(np.where(inp)[0]); MWp = torch.from_numpy(M.funs(tp[inp]))
a0grid = np.round(np.arange(0.15, 0.8501, 0.01), 4)                 # the alpha_0 range of the widened theory grid
means = np.array([moments_fit(a0)[1] for a0 in a0grid])
assert np.all(np.diff(means) < 0), 'the windowed mean is no longer monotonically decreasing in alpha_0'

def transfer(theta):
    """Parton-level tilt at theta; returns (hadron-level windowed moments, parton window mean, hadron
    window mean before the tilt, KL)."""
    with torch.no_grad():
        lg = M.logit(torch.tensor(theta)); lw0 = lg - torch.logsumexp(lg, 0)
        Sc = Sig + np.diag((0.002*np.abs(c_pert))**2)
        lw, lam, Pw, its = M.tilt_on(lw0, IWp, MWp, c_pert, Sc, torch.zeros(len(keys)))
        w = torch.exp(lw)
        wh = w[M.IW]; mh = ((wh @ M.MW)/wh.sum()).numpy()
        wp = w[IWp]; mp = ((wp @ MWp)/wp.sum()).numpy()
        w0h = torch.exp(lw0)[M.IW]; mh0 = ((w0h @ M.MW)/w0h.sum()).numpy()
        kl = float((w*(lw - lw0)).sum())
    return mh, mp, mh0, kl

def implied(mh_mean):
    if not (means.min() <= mh_mean <= means.max()):
        return float('nan')
    return float(np.interp(mh_mean, means[::-1], a0grid[::-1]))

mh, mp, mh0, kl = transfer(th)
pulls = (mh - c_had)/sd
a0_impl = implied(mh[1])
print(f'fitted point alpha_s {a_fit:.5f}, alpha_0 {a0_fit:.4f}; parton-level tilt KL {kl:.5f}')
print(f'windowed mean: parton tilted {mp[1]:.5f} (theory {c_pert[1]:.5f}); hadron transferred {mh[1]:.5f} against the '
      f'dispersive {c_had[1]:.5f} +- {sd[1]:.5f}, pull {pulls[1]:+.2f}')
print(f'fourteen pulls: rms {np.sqrt(np.mean(pulls**2)):.2f}, max |pull| {np.abs(pulls).max():.2f}')
print(f'window-mean shift: generator {mh[1] - mp[1]:+.5f}, dispersive {c_had[1] - c_pert[1]:+.5f}; implied alpha_0 {a0_impl:.4f}')
a_held = float('nan'); edge = None
if np.isfinite(a0_impl):
    cols = rr['alpha_0_profile']['cols']; ca0, cas = np.array(cols['alpha0']), np.array(cols['alphas'])
    edge = not (ca0.min() <= a0_impl <= ca0.max())
    a_held = float(np.interp(a0_impl, ca0, cas))
    print(f'coupling with alpha_0 held at the implied value: {a_held:.5f}'
          f'{"  BEYOND THE COMPUTED alpha_0 COLUMNS, a bound only" if edge else ""} (shift {a_held - a_fit:+.5f})')
out = dict(alpha_s_fit=a_fit, alpha_0_fit=a0_fit, theta=th.tolist(), pulls=dict(zip(keys, pulls.tolist())),
           rms_pull=float(np.sqrt(np.mean(pulls**2))), max_pull=float(np.abs(pulls).max()),
           gen_shift=float(mh[1] - mp[1]), dispersive_shift=float(c_had[1] - c_pert[1]), alpha_0_implied=a0_impl,
           alpha_s_held=a_held, held_at_edge=edge, kl=kl)
if M.kind == 'mixture':
    # pure Herwig across its shower cutoff, the other Herwig parameters at the centre of their box
    names = None
    try:
        sys.path.insert(0, 'release'); from gentune.axes import axes_of
        names = axes_of(fit['tag'])
    except Exception:
        pass
    k_pt = M.NS + 1                         # Herwig block order: alpha_fsr, ptmin, ...
    lo, hi = M.NORM[k_pt, 0] - M.NORM[k_pt, 1], M.NORM[k_pt, 0] + M.NORM[k_pt, 1]
    scan = []
    for pt in np.linspace(lo, hi, 7):
        t = M.NORM[:, 0].copy(); t[-1] = 1.0; t[k_pt] = pt
        mh_, mp_, mh0_, kl_ = transfer(t)
        scan.append(dict(ptmin=float(pt), gen_shift=float(mh_[1] - mp_[1]), alpha_0_implied=implied(mh_[1]),
                         hadron_mean=float(mh_[1]), parton_mean=float(mp_[1]), kl=kl_))
        print(f'  pure Herwig, ptmin {pt:.3f} GeV: generator window-mean shift {mh_[1] - mp_[1]:+.5f}, implied alpha_0 {implied(mh_[1]):.4f}')
    # the same cutoff scan at the fitted point itself: the other nuisance parameters and the fraction at
    # their profiled values, so the check's dependence on the cutoff is read where the fit sits
    scan_fit = []
    for pt in np.linspace(lo, hi, 7):
        t = th.copy(); t[k_pt] = pt
        mh_, mp_, mh0_, kl_ = transfer(t)
        pl_ = (mh_ - c_had)/sd
        scan_fit.append(dict(ptmin=float(pt), gen_shift=float(mh_[1] - mp_[1]), alpha_0_implied=implied(mh_[1]),
                             rms_pull=float(np.sqrt(np.mean(pl_**2))), max_pull=float(np.abs(pl_).max()), kl=kl_))
        print(f'  fitted point, Herwig ptmin {pt:.3f} GeV: generator window-mean shift {mh_[1] - mp_[1]:+.5f}, implied alpha_0 '
              f'{implied(mh_[1]):.4f}, pulls rms {np.sqrt(np.mean(pl_**2)):.2f} max {np.abs(pl_).max():.2f}')
    out['cutoff_scan_fit'] = scan_fit
    # and the pure Sherpa point at the centre of its box, for comparison
    t = M.NORM[:, 0].copy(); t[-1] = 0.0
    mh_, mp_, _, _ = transfer(t)
    out['cutoff_scan_herwig'] = scan
    out['sherpa_centre'] = dict(gen_shift=float(mh_[1] - mp_[1]), alpha_0_implied=implied(mh_[1]))
    print(f'  pure Sherpa at its box centre: generator window-mean shift {mh_[1] - mp_[1]:+.5f}, implied alpha_0 {implied(mh_[1]):.4f}')
dst = src.replace('.json', '_check.json')
json.dump(out, open(dst, 'w'), indent=1); print('wrote', dst)
