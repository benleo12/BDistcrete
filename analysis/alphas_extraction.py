#!/usr/bin/env python3
"""alpha_s extraction demo on the Stage C conditional (alpha_s, strange fraction, KT_0).

PSEUDO-DATA. The three anchored event-shape moments of the MEPS@NLO sample
(<1-T>, <B_total>, <rho_heavy>, archived in output/maxent_tilt_fixed.json and recomputed
here with the full statistical covariance) are treated as data. m(nu) and its exact
Jacobian dm_i/dnu_p = (1/T) Cov_w(m_i, g_p) come from the analytic b-network derivative of
joint_covariance.Head (verified there to ~1e-9), so every fit step is two matrix products
on the cached export -- no generator calls anywhere.

HONEST STRUCTURE OF THE RESULT. With all three moments anchored, the LO CSS + Ahadic
surrogate cannot reach the MEPS@NLO moment vector anywhere in the Stage C box: the fit
RAILS both hadronization nuisances at box edges and leaves a large chi2_min (the three
moments are ~0.95 correlated, so the full-covariance metric is very unforgiving along the
small-variance eigendirection). A profile against a railed nuisance is not a Gaussian
profile, so for the 3-moment fit the hadronization term is quoted as the BOUNDED-PROFILE
VARIATION: the spread of alpha_hat as the two nuisances scan their training box. The clean
demonstration of the machinery is the SINGLE-CONSTRAINT variant (anchor <1-T> alone):
there the two nuisances can compensate exactly, the profile has a flat bottom (the
nuisance-compensation interval) plus data wings, and with the placeholder Gaussian prior
on the nuisances (sigma = box half-widths, stated) it becomes a proper profile with an
interior minimum. Error budget quoted per variant:
 (i)   moment/data errors (Delta chi2 = 1),
 (ii)  hadronization nuisances (bounded variation, or flat-bottom half-width, or
       sqrt(prof^2 - fixed^2) where the profile is genuinely unrailed),
 (iii) shower-scale band of the pseudo-data (re-extraction on the recomputed signed
       shup/shdn moment vectors),
 plus, noted separately, the METHOD ACCURACY FLOOR from the paper's Stage C held-out
 closure (master pull width 1.13 over calibrated floor 1.00 on 80k-event runs -> residual
 per-observable bias <= sqrt(1.13^2-1) = 0.53 of the 80k-run stat error; re-extraction
 after shifting each anchored moment by that floor). An estimate, not a measured bias.

REAL DATA. OPAL, Eur. Phys. J. C 40 (2005) 287 (hep-ex/0503051), Table 7, hadron level at
91 GeV: <(1-T)> = (6.671 +- 0.017 +- 0.066)e-2, <B_T> = (1.0909 +- 0.0016 +- 0.0068)e-1,
<M_H^2> = (5.235 +- 0.014 +- 0.086)e-2 (M_H = heavy hemisphere mass / sqrt(s)). Same
treatment (3-moment fit with rail reporting + single-constraint variant). Caveats recorded
in the output: LO shower with tuned cluster hadronization (illustration of the uncertainty
machinery, not a competitive alpha_s determination), rho_heavy normalized by E_vis versus
OPAL's sqrt(s), unpublished correlations set to zero, no shower-scale band of the
reference available.

Idempotent and model-path-parametrized:
  python alphas_extraction.py --models output/models --out output/alphas_extraction_v2.json
Writes the JSON (+ target-moment cache, reused if present). CPU only.
"""
import argparse, json, os, time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from joint_covariance import Head, solve_tilt, CONSTRAINTS

EVDIR = os.environ['CSS_EVENTS_DIR']
BOUNDS = [(0.112, 0.128), (0.30, 0.65), (0.80, 1.80)]     # Stage C training box
NUIS_MU = np.array([0.475, 1.30])                          # box centres (prior mean)
NUIS_SIG = np.array([0.175, 0.50])                         # box half-widths (prior sigma)
AGRID = np.linspace(0.112, 0.128, 33)
FLOOR_EXCESS = np.sqrt(1.13**2 - 1.00**2)
NVAL = 80000

OPAL = dict(
    citation=('OPAL Collaboration (G. Abbiendi et al.), "Measurement of event shape '
              'distributions and moments in e+e- -> hadrons at 91-209 GeV and a '
              'determination of alpha_s", Eur. Phys. J. C 40 (2005) 287, '
              'arXiv:hep-ex/0503051, Table 7, sqrt(s) = 91 GeV, hadron level'),
    c=[0.06671, 0.10909, 0.05235],                        # <1-T>, <B_T>, <(M_H/sqrt(s))^2>
    stat=[0.00017, 0.00016, 0.00014],
    syst=[0.00066, 0.00068, 0.00086])


# --------------------------------------------------------------------------------------
def get_target_moments(cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    from maxent_tilt_fixed import sample_moments
    out = {}
    t0 = time.time()
    m, err, ws, V = sample_moments(f'{EVDIR}/events_nom', ret_raw=True)
    W = ws.sum(); mu = (ws[:, None]*V).sum(0)/W
    neff = W*W/np.sum(ws**2)
    D = V - mu
    cov = ((ws[:, None, None]*(D[:, :, None]*D[:, None, :])).sum(0)/W)/neff
    out['nom'] = dict(m=m.tolist(), err=err.tolist(), cov_stat=cov.tolist(),
                      neff=float(neff), nev=int(len(ws)))
    for tag in ('shup', 'shdn'):
        m2, e2 = sample_moments(f'{EVDIR}/events_{tag}')
        out[tag] = dict(m=m2.tolist(), err=e2.tolist())
    out['parse_seconds'] = time.time() - t0
    json.dump(out, open(cache, 'w'), indent=1)
    return out


class Fitter:
    def __init__(self, models='output/models'):
        self.head = Head('C', models)
        ref_path = f'{models}/C_ref.npz'
        ref = np.load(ref_path)
        self.M = np.stack([ref[o].astype(np.float64) for o in CONSTRAINTS])
        self.provenance = dict(cond=self.head.path, cond_mtime=self.head.mtime,
                               ref=ref_path, ref_mtime=os.path.getmtime(ref_path))

    def weights(self, nu):
        f = self.head.f_only(np.asarray(nu, float))
        w = np.exp((f - f.max())/self.head.T)
        return w/w.sum()

    def moments_jac(self, nu):
        f, G = self.head.f_and_g(np.asarray(nu, float))
        w = np.exp((f - f.max())/self.head.T); w /= w.sum()
        m = self.M @ w
        J = ((self.M*w) @ G - np.outer(m, w @ G))/self.head.T   # dm_i/dnu_p
        return m, J, w

    def chi2(self, nu, c, Vinv, idx=None):
        m, J, _ = self.moments_jac(nu)
        if idx is not None:
            m = m[idx]; J = J[idx]
        r = m - c
        return float(r @ Vinv @ r), 2.0*(J.T @ (Vinv @ r))

    def kl_of_tilt(self, nu, c):
        w0 = self.weights(nu)
        lam, wt, C, m, it, ok = solve_tilt(w0, self.M, c)
        return float(np.sum(wt*np.log(np.clip(wt/np.clip(w0, 1e-300, None), 1e-300, None)))), ok


def rails(x, bounds, tol=1e-6):
    return [bool(min(x[i]-lo, hi-x[i]) < tol*(hi-lo)) for i, (lo, hi) in enumerate(bounds)]


# --------------------------------------------------------------------------------------
# generic fits
# --------------------------------------------------------------------------------------
def fit_global(fun_jac, bounds, starts):
    best = None
    for s0 in starts:
        r = minimize(fun_jac, np.asarray(s0, float), jac=True, method='L-BFGS-B', bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best


def starts_3d():
    s = [[0.120, 0.475, 1.30]]
    for da in (-0.6, 0.6):
        for ds in (-0.6, 0.6):
            s.append([0.120 + da*0.008, 0.475 + ds*0.175, 1.30 - da*0.5])
    return s


def profile_alpha(fun_jac_full, agrid):
    """Profile over the 2 nuisances at each fixed alpha (multistart + warm start).
    fun_jac_full(nu3) -> (f, grad3). Returns chi2 curve, nuisance path, rail flags."""
    prof = np.empty(len(agrid)); path = np.empty((len(agrid), 2)); railed = []
    x0s = [np.array([0.475, 1.30]), np.array([0.35, 0.95]), np.array([0.60, 1.65])]
    prev = None
    for k, a in enumerate(agrid):
        starts = list(x0s) + ([prev] if prev is not None else [])
        best = None
        for s0 in starts:
            def fj(x):
                f, g = fun_jac_full(np.array([a, x[0], x[1]]))
                return f, g[1:]
            r = minimize(fj, np.asarray(s0, float), jac=True,
                         method='L-BFGS-B', bounds=BOUNDS[1:])
            if best is None or r.fun < best.fun:
                best = r
        prof[k] = best.fun; path[k] = best.x; prev = best.x
        railed.append(rails([a] + list(best.x), BOUNDS)[1:])
    return prof, path, railed


def dchi2_interval(agrid, prof, target=1.0):
    cs = CubicSpline(agrid, prof)
    afine = np.linspace(agrid[0], agrid[-1], 20001)
    pfine = cs(afine)
    k = int(np.argmin(pfine)); ahat = float(afine[k]); pmin = float(pfine[k])
    lo = hi = None
    idx = np.where(pfine <= pmin + target)[0]
    if idx[0] > 0:
        lo = float(np.interp(pmin + target, [pfine[idx[0]], pfine[idx[0]-1]],
                             [afine[idx[0]], afine[idx[0]-1]]))
    if idx[-1] < len(afine) - 1:
        hi = float(np.interp(pmin + target, [pfine[idx[-1]], pfine[idx[-1]+1]],
                             [afine[idx[-1]], afine[idx[-1]+1]]))
    return dict(alpha_hat=ahat, chi2_min=pmin, lo=lo, hi=hi,
                sigma_from_crossings=None if (lo is None or hi is None)
                else float(0.5*(hi-lo)))


def flat_bottom(agrid, prof, thresh=1e-2):
    sel = np.where(prof <= thresh)[0]
    if len(sel) == 0:
        return None
    return dict(lo=float(agrid[sel[0]]), hi=float(agrid[sel[-1]]),
                half_width=float(0.5*(agrid[sel[-1]] - agrid[sel[0]])), thresh=thresh)


def nuisance_box_variation(fit, c, Vinv, idx=None, ngrid=5):
    """alpha refit on a (strange, KT_0) grid over the training box: the honest
    hadronization term when the free fit rails. Returns the grid and the spread."""
    sf = np.linspace(*BOUNDS[1], ngrid); kt = np.linspace(*BOUNDS[2], ngrid)
    A = np.empty((ngrid, ngrid)); arail = 0
    for i, s in enumerate(sf):
        for j, k in enumerate(kt):
            def fj(x):
                f, g = fit.chi2(np.array([x[0], s, k]), c, Vinv, idx)
                return f, g[:1]
            r = fit_global(fj, BOUNDS[:1], [[0.118], [0.121]])
            A[i, j] = r.x[0]
            arail += rails(r.x, BOUNDS[:1])[0]
    return dict(strange_grid=sf.tolist(), kt_grid=kt.tolist(), alpha_hat_grid=A.tolist(),
                alpha_min=float(A.min()), alpha_max=float(A.max()),
                half_spread=float(0.5*(A.max() - A.min())),
                n_alpha_railed=int(arail))


def extract_3m(fit, c, Vinv, tag, do_curve=True):
    """3-moment fit: global minimum with rail flags, profiled + fixed-nuisance curves."""
    g = fit_global(lambda x: fit.chi2(x, c, Vinv), BOUNDS, starts_3d())
    rl = rails(g.x, BOUNDS)
    out = dict(tag=tag, alpha_hat=float(g.x[0]), strange_hat=float(g.x[1]),
               kt_hat=float(g.x[2]), chi2_min=float(g.fun),
               railed=dict(alpha=rl[0], strange=rl[1], kt=rl[2]))
    if do_curve:
        prof, path, railed = profile_alpha(lambda nu: fit.chi2(nu, c, Vinv), AGRID)
        iv = dchi2_interval(AGRID, prof)
        frac_rail = float(np.mean([any(r) for r in railed]))
        out.update(profile=dict(agrid=AGRID.tolist(), chi2=prof.tolist(),
                                nuisance_path=path.tolist(),
                                railed_along_path=railed,
                                fraction_of_grid_railed=frac_rail),
                   interval=iv)
        fixed = np.array([fit.chi2(np.array([a, g.x[1], g.x[2]]), c, Vinv)[0] for a in AGRID])
        out['interval_fixed'] = dchi2_interval(AGRID, fixed)
        sp = iv['sigma_from_crossings']; sf_ = out['interval_fixed']['sigma_from_crossings']
        out['sigma_profiled'] = sp
        out['sigma_fixed'] = sf_
        if any(rl[1:]) or frac_rail > 0.05:
            out['sigma_nuisance_component'] = None
            out['sigma_nuisance_invalid_reason'] = (
                'nuisances rail at box edges (profile against a railed nuisance is not a '
                'Gaussian profile); use nuisance_box_variation instead')
        elif sp is not None and sf_ is not None:
            out['sigma_nuisance_component'] = float(np.sqrt(max(sp**2 - sf_**2, 0.0)))
        out['nuisance_box_variation'] = nuisance_box_variation(fit, c, Vinv)
    return out


def extract_1m(fit, c1, sig1, tag, prior=False):
    """Single-constraint variant: anchor <1-T> alone. With prior=True the placeholder
    Gaussian prior on the nuisances is added (stated), giving an interior minimum."""
    idx = np.array([0]); Vinv = np.array([[1.0/sig1**2]])
    c = np.array([c1])

    def obj(nu):
        f, g = fit.chi2(nu, c, Vinv, idx)
        if prior:
            z = (nu[1:] - NUIS_MU)/NUIS_SIG
            f += float(z @ z)
            g = g.copy(); g[1:] += 2*z/NUIS_SIG
        return f, g

    g = fit_global(obj, BOUNDS, starts_3d())
    rl = rails(g.x, BOUNDS)
    prof, path, railed = profile_alpha(obj, AGRID)
    iv = dchi2_interval(AGRID, prof)
    out = dict(tag=tag, prior_on_nuisances=prior,
               alpha_hat=float(g.x[0]), strange_hat=float(g.x[1]), kt_hat=float(g.x[2]),
               chi2_min=float(g.fun), railed=dict(alpha=rl[0], strange=rl[1], kt=rl[2]),
               profile=dict(agrid=AGRID.tolist(), chi2=prof.tolist(),
                            nuisance_path=path.tolist(), railed_along_path=railed,
                            fraction_of_grid_railed=float(np.mean([any(r) for r in railed]))),
               interval=iv)
    if not prior:
        out['flat_bottom'] = flat_bottom(AGRID, prof)
    else:
        fixed = np.array([obj(np.array([a, g.x[1], g.x[2]]))[0] for a in AGRID])
        out['interval_fixed'] = dchi2_interval(AGRID, fixed)
        sp = iv['sigma_from_crossings']; sf_ = out['interval_fixed']['sigma_from_crossings']
        out['sigma_profiled'] = sp; out['sigma_fixed'] = sf_
        if sp is not None and sf_ is not None and not any(rl[1:]):
            out['sigma_nuisance_component'] = float(np.sqrt(max(sp**2 - sf_**2, 0.0)))
    return out


def alpha_of_1m(fit, c1, sig1, prior=True):
    idx = np.array([0]); Vinv = np.array([[1.0/sig1**2]]); c = np.array([c1])

    def obj(nu):
        f, g = fit.chi2(nu, c, Vinv, idx)
        if prior:
            z = (nu[1:] - NUIS_MU)/NUIS_SIG
            f += float(z @ z)
            g = g.copy(); g[1:] += 2*z/NUIS_SIG
        return f, g
    return float(fit_global(obj, BOUNDS, starts_3d()).x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='output/models')
    ap.add_argument('--out', default='output/alphas_extraction_v2.json')
    ap.add_argument('--cache', default='output/alphas_extraction_moments_cache.json')
    args = ap.parse_args()
    t00 = time.time()
    print('parsing / loading target moments ...', flush=True)
    tm = get_target_moments(args.cache)
    arc = json.load(open('output/maxent_tilt_fixed.json'))
    c0 = np.array(tm['nom']['m']); stat = np.array(tm['nom']['err'])
    cov_stat = np.array(tm['nom']['cov_stat'])
    cup = np.array(tm['shup']['m']); cdn = np.array(tm['shdn']['m'])
    corr = cov_stat/np.sqrt(np.outer(np.diag(cov_stat), np.diag(cov_stat)))
    print(f'nominal moments {np.round(c0, 5).tolist()} '
          f'(max|diff vs archived| = {np.max(np.abs(c0-np.array(arc["c"]))):.2e})')

    fit = Fitter(args.models)
    Vinv_full = np.linalg.inv(cov_stat)
    Vinv_diag = np.diag(1.0/stat**2)
    res = dict(constraints=CONSTRAINTS, stageC_box=BOUNDS, args=vars(args),
               model_provenance=fit.provenance,
               pseudo_data=dict(c=c0.tolist(), stat=stat.tolist(), cov_stat=cov_stat.tolist(),
                                corr_stat=corr.tolist(), shup=cup.tolist(), shdn=cdn.tolist(),
                                band=(0.5*np.abs(cup-cdn)).tolist()))

    # ---------------- 3-moment pseudo-data fits (rails reported honestly) ----------------
    print('pseudo-data 3-moment fit, full stat covariance ...', flush=True)
    x_full = extract_3m(fit, c0, Vinv_full, 'pseudo_3m_full_cov')
    print(f'  alpha_hat = {x_full["alpha_hat"]:.5f}  chi2_min = {x_full["chi2_min"]:.1f}  '
          f'nuisances = ({x_full["strange_hat"]:.3f}, {x_full["kt_hat"]:.3f})  '
          f'railed = {x_full["railed"]}')
    print(f'  nuisance box variation of alpha_hat: '
          f'[{x_full["nuisance_box_variation"]["alpha_min"]:.5f}, '
          f'{x_full["nuisance_box_variation"]["alpha_max"]:.5f}]  half-spread '
          f'{x_full["nuisance_box_variation"]["half_spread"]:.5f}')
    res['pseudo_3m_full_cov'] = x_full
    print('pseudo-data 3-moment fit, diagonal covariance (cross-check) ...', flush=True)
    x_diag = extract_3m(fit, c0, Vinv_diag, 'pseudo_3m_diag_cov')
    print(f'  alpha_hat = {x_diag["alpha_hat"]:.5f}  chi2_min = {x_diag["chi2_min"]:.2f}  '
          f'railed = {x_diag["railed"]}')
    res['pseudo_3m_diag_cov'] = x_diag

    # KL-of-tilt objective: same misfit measured in the sample metric
    x0 = np.array([x_diag['alpha_hat'], x_diag['strange_hat'], x_diag['kt_hat']])
    rkl = minimize(lambda x: fit.kl_of_tilt(x, c0)[0], x0, method='Nelder-Mead',
                   options=dict(xatol=1e-5, fatol=1e-12, maxfev=400))
    res['kl_objective_3m'] = dict(alpha_hat=float(rkl.x[0]), strange_hat=float(rkl.x[1]),
                                  kt_hat=float(rkl.x[2]), kl_min=float(rkl.fun),
                                  railed=dict(zip(['alpha', 'strange', 'kt'],
                                                  rails(rkl.x, BOUNDS))))
    print(f'  KL objective minimizer: alpha = {rkl.x[0]:.5f}, KL_min = {rkl.fun:.4g} nats')

    # ---------------- single-constraint variant: anchor <1-T> alone ----------------
    print('single-constraint variant (<1-T> alone) ...', flush=True)
    s1 = extract_1m(fit, c0[0], stat[0], 'pseudo_1m', prior=False)
    print(f'  free nuisances: flat bottom {s1.get("flat_bottom")}  '
          f'Dchi2=1 interval [{s1["interval"]["lo"]}, {s1["interval"]["hi"]}]  '
          f'railed at min: {s1["railed"]}')
    s1p = extract_1m(fit, c0[0], stat[0], 'pseudo_1m_prior', prior=True)
    print(f'  with placeholder prior: alpha_hat = {s1p["alpha_hat"]:.5f}  '
          f'sigma(prof) = {s1p.get("sigma_profiled")}  sigma(fixed) = {s1p.get("sigma_fixed")}  '
          f'nuisance = {s1p.get("sigma_nuisance_component")}  railed = {s1p["railed"]}')
    res['pseudo_1m'] = s1
    res['pseudo_1m_prior'] = s1p

    # shower band and method floor for the single-constraint (prior) extraction
    a_up = alpha_of_1m(fit, cup[0], stat[0]); a_dn = alpha_of_1m(fit, cdn[0], stat[0])
    band_1m = 0.5*abs(a_up - a_dn)
    print(f'  shower band (1m): alpha(shup) = {a_up:.5f} alpha(shdn) = {a_dn:.5f} '
          f'-> {band_1m:.5f}')
    _, _, wbest = fit.moments_jac(np.array([s1p['alpha_hat'], s1p['strange_hat'],
                                            s1p['kt_hat']]))
    sd = np.array([np.sqrt(max(wbest @ (fit.M[i]**2) - (wbest @ fit.M[i])**2, 0.0))
                   for i in range(3)])
    floor = FLOOR_EXCESS*sd/np.sqrt(NVAL)
    a_fp = alpha_of_1m(fit, c0[0]+floor[0], stat[0])
    a_fm = alpha_of_1m(fit, c0[0]-floor[0], stat[0])
    floor_1m = 0.5*abs(a_fp - a_fm)
    print(f'  method floor (1m): moment floor {floor[0]:.6f} -> alpha shift {floor_1m:.5f}')
    res['pseudo_1m_band_floor'] = dict(alpha_shup=a_up, alpha_shdn=a_dn,
                                       sigma_shower_band=float(band_1m),
                                       floor_per_moment=floor.tolist(),
                                       sigma_method_floor=float(floor_1m),
                                       floor_note=('paper Stage C held-out closure 1.13 over '
                                                   'floor 1.00 on 80k-event runs; estimate, '
                                                   'not a measured bias'))

    # 3-moment shower band + method floor (for the model-discrepancy variant, with rails)
    print('3-moment shower band / floor (rails possible, reported) ...', flush=True)
    xu = extract_3m(fit, cup, Vinv_full, 'shup', do_curve=False)
    xd = extract_3m(fit, cdn, Vinv_full, 'shdn', do_curve=False)
    band_3m = 0.5*abs(xu['alpha_hat'] - xd['alpha_hat'])
    shifts = []
    for i in range(3):
        aa = []
        for sgn in (+1, -1):
            cc = c0.copy(); cc[i] += sgn*floor[i]
            aa.append(extract_3m(fit, cc, Vinv_full, 'floor', do_curve=False)['alpha_hat'])
        shifts.append(0.5*abs(aa[0]-aa[1]))
    res['pseudo_3m_band_floor'] = dict(
        up=xu, dn=xd, sigma_shower_band=float(band_3m),
        alpha_shift_per_moment=[float(s) for s in shifts],
        sigma_method_floor=float(np.sqrt(np.sum(np.array(shifts)**2))),
        caveat=('3-moment band/floor shifts computed on a fit that rails its nuisances; '
                'quoted for completeness, the single-constraint numbers are the clean ones'))
    print(f'  band(3m) = {band_3m:.5f}  floor(3m, quadrature) = '
          f'{res["pseudo_3m_band_floor"]["sigma_method_floor"]:.5f}')

    # ---------------- error budget summaries ----------------
    res['error_budget_pseudo_1m_prior'] = dict(
        alpha_hat=s1p['alpha_hat'],
        sigma_data=s1p.get('sigma_fixed'),
        sigma_hadronization_nuisance_profiled=s1p.get('sigma_nuisance_component'),
        sigma_total_data_plus_nuisance=s1p.get('sigma_profiled'),
        flat_bottom_free_nuisances=s1.get('flat_bottom'),
        sigma_shower_band=float(band_1m),
        sigma_method_floor=float(floor_1m),
        note=('single anchored moment <1-T>; nuisance prior = placeholder Gaussian with '
              'sigma = training-box half-widths (stated)'))
    res['error_budget_pseudo_3m'] = dict(
        alpha_hat=x_full['alpha_hat'],
        railed=x_full['railed'], chi2_min=x_full['chi2_min'],
        sigma_data_profiled=x_full.get('sigma_profiled'),
        sigma_hadronization_bounded_variation=x_full['nuisance_box_variation']['half_spread'],
        sigma_shower_band=float(band_3m),
        sigma_method_floor=res['pseudo_3m_band_floor']['sigma_method_floor'],
        note=('3-moment fit rails both nuisances at box edges and leaves chi2_min >> ndf: '
              'the LO CSS+Ahadic surrogate cannot reach the MEPS@NLO moment vector inside '
              'the Stage C box (model discrepancy); hadronization term quoted as the '
              'bounded-profile variation of alpha_hat over the nuisance box'))

    # ---------------- real data: OPAL 91 GeV ----------------
    print('real-data extraction (OPAL, EPJ C40 (2005) 287, Table 7) ...', flush=True)
    c_op = np.array(OPAL['c'])
    e_op = np.sqrt(np.array(OPAL['stat'])**2 + np.array(OPAL['syst'])**2)
    Vinv_op = np.diag(1.0/e_op**2)
    op3 = extract_3m(fit, c_op, Vinv_op, 'opal_3m')
    print(f'  OPAL 3m: alpha_hat = {op3["alpha_hat"]:.5f}  chi2_min = {op3["chi2_min"]:.2f}  '
          f'nuisances = ({op3["strange_hat"]:.3f}, {op3["kt_hat"]:.3f})  '
          f'railed = {op3["railed"]}')
    op1 = extract_1m(fit, c_op[0], e_op[0], 'opal_1m', prior=False)
    op1p = extract_1m(fit, c_op[0], e_op[0], 'opal_1m_prior', prior=True)
    print(f'  OPAL 1m prior: alpha_hat = {op1p["alpha_hat"]:.5f}  '
          f'sigma(prof) = {op1p.get("sigma_profiled")}  railed = {op1p["railed"]}')
    a_ofp = alpha_of_1m(fit, c_op[0]+floor[0], e_op[0])
    a_ofm = alpha_of_1m(fit, c_op[0]-floor[0], e_op[0])
    res['opal'] = dict(
        OPAL, errors_combined_stat_plus_syst=e_op.tolist(),
        extraction_3m=op3, extraction_1m=op1, extraction_1m_prior=op1p,
        sigma_method_floor_1m=float(0.5*abs(a_ofp - a_ofm)),
        error_budget_1m_prior=dict(
            alpha_hat=op1p['alpha_hat'],
            sigma_data=op1p.get('sigma_fixed'),
            sigma_hadronization_nuisance_profiled=op1p.get('sigma_nuisance_component'),
            sigma_total_data_plus_nuisance=op1p.get('sigma_profiled'),
            flat_bottom_free_nuisances=op1.get('flat_bottom'),
            sigma_method_floor=float(0.5*abs(a_ofp - a_ofm))),
        caveats=[
            'the generator is a leading-order (ORDER_ALPHAS: 1) CSS shower with a tuned '
            'Ahadic cluster-hadronization model: this is an illustration of the '
            'uncertainty machinery, not a competitive alpha_s determination',
            'rho_heavy here normalizes the heavy-hemisphere mass squared by E_vis '
            '(neutrinos excluded) whereas OPAL <M_H^2> normalizes by sqrt(s); at the Z '
            'peak this shifts that one moment by ~1-2 percent',
            'correlations between the three OPAL moments are not published and are set '
            'to zero',
            'no shower-scale variation of the REFERENCE sample is available in the '
            'cache, so no reference-scale band is quoted',
            'the extracted parameter is the shower ALPHAS(MZ) knob of Sherpa CSS at '
            'leading order, not an MSbar alpha_s(MZ)'])
    json.dump(res, open(args.out, 'w'), indent=1)
    print(f'ALPHA_S EXTRACTION DONE in {time.time()-t00:.0f} s -> {args.out}')


if __name__ == '__main__':
    main()
