#!/usr/bin/env python3
"""Exact nuisance bands at the fitted point of a direct-profile fit, split over many processes.

The nuisance band of a prediction O is half its range over the nuisance parameters allowed by the
data, chi^2 within one unit of its minimum at the fitted coupling and alpha_0 (the minimum of the
continuous profiles of profile_rows.py, read from *_rows.json). Each end of the range is a
constrained minimization (SLSQP with exact gradients), as in exchange_direct.py. The quadratic model
of bands_quadratic.py is faster but underestimates the event-shape bands by about forty percent, so
here every end is found exactly: for every bin of the experiments in FIG_EXPS (the figure), as
densities normalized over the fitted bins like the fit itself, and for the scalar observables of
the transport table, each before and after anchoring.

  point  the nuisance parameters at the fitted point, minimized once, written as *_bands_point.json
  part   PART/NPART: a share of the (prediction, before/after, lower/upper end) tasks
  merge  the bands, the theory and model bands (variations re-solved, half-sum of squares), and the
         central values, written as *_exchange.json and *_exchange_dist.npz in the layout
         fig_anchored_fit.py reads

    python bands_exact.py point output/profile_MIX17ext_central.json
    PART=0 NPART=64 python bands_exact.py part output/profile_MIX17ext_central.json
    python bands_exact.py merge output/profile_MIX17ext_central.json
"""
import os, sys, json, glob, time
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, tilt_on_t

mode, src = sys.argv[1], sys.argv[2]
fit = json.load(open(src)); stem = src.replace('.json', '')
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype=os.environ.get('AE_DTYPE', 'float64'),
          grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
# the fitted point of the paper: the minimum of the continuous profiles of profile_rows.py, between
# the theory nodes, with the targets and their covariance from TargetInterp as in the fit itself
_rr = json.load(open(stem + '_rows.json'))
a_fit = _rr['variations']['central']['value']
a0_fit = _rr['alpha_0_profile']['value'] if 'alpha_0_profile' in _rr else _rr['variations']['central']['alpha_0_at_min']
FIG_EXPS = os.environ.get('FIG_EXPS', 'delphi').split(',')
OBS = [k for k in ('thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon') if k in ('thrust', 'nch') or k in M.extra]
EX = {x['name']: x for x in M.exp}
t0 = time.time()

_TI = {}
def targets(which='central'):
    """Targets and their covariance at the fitted point, with the floor of Model.sigma."""
    if which not in _TI:
        c, _, _, S, _, _ = TargetInterp(M.G, which)(a_fit, a0_fit)
        cT = torch.tensor(c)
        _TI[which] = (cT, torch.tensor(S) + torch.diag((M.floor_rel*torch.abs(cT))**2))
    return _TI[which]

def logw(u, which='central', anchored=True):
    """(log w, log w0) at the fitted point for the standardized nuisance parameters u."""
    lg = M.logit(M.theta(u)); lw0 = lg - torch.logsumexp(lg, 0)
    if not anchored:
        return lw0, lw0
    cT, Sig = targets(which)
    lw, _, _, _ = tilt_on_t(M, lw0, cT, Sig, torch.zeros(M.MW.shape[1]))
    return lw, lw0

def predictions(u, which='central', anchored=True):
    """Every prediction as a tensor of u: per experiment the densities of all bins normalized over the
    fitted bins, and the scalar observables."""
    lw, _ = logw(u, which, anchored)
    w = torch.exp(lw)
    out = {}
    for ex in FIG_EXPS:
        x = EX[ex]
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']])
        out[ex] = cnt/(cnt[x['use']].sum()*x['wid'])
    o = M.observables(w)
    for k in OBS:
        out[k] = o[k]
    return out

_cache = {}
def chi2_fg(u):
    key = np.asarray(u, float).tobytes()
    if key not in _cache:
        uu = torch.tensor(np.asarray(u, float), requires_grad=True)
        lw, _ = logw(uu)
        c2, _ = M.chi2(torch.exp(lw)); c2.backward()
        if len(_cache) > 64: _cache.clear()
        _cache[key] = (float(c2), uu.grad.numpy().copy())
    return _cache[key]

def tasks():
    T = []
    for ex in FIG_EXPS:
        for b in range(EX[ex]['nb']):
            for anch in (True, False):
                for sgn in (+1, -1):
                    T.append((ex, b, anch, sgn))
    for k in OBS:
        for anch in (True, False):
            for sgn in (+1, -1):
                T.append((k, -1, anch, sgn))
    return T

PT = stem + '_bands_point.json'
if mode == 'point':
    # the nuisance parameters at the fitted point, minimized from the two coupling rows around it
    R = _rr['variations']['central']['rows']; xs = np.array(R['alphas']); k = int(np.searchsorted(xs, a_fit))
    best = None
    for q in (k - 1, k):
        if 0 <= q < len(xs):
            r = minimize(chi2_fg, np.asarray(R['y'][q], float)[:M.NT], jac=True, method='L-BFGS-B', bounds=[(-1, 1)]*M.NT,
                         options=dict(maxiter=2000, ftol=1e-13, gtol=1e-8))
            print(f'  from the row at {xs[q]:.3f} (chi2 {R["chi2"][q]:.3f}): {r.fun:.4f} after {r.nit} iterations', flush=True)
            if best is None or r.fun < best.fun:
                best = r
    json.dump(dict(alpha_s=a_fit, alpha_0=a0_fit, u=best.x.tolist(), chi2=float(best.fun)), open(PT, 'w'), indent=1)
    print(f'fitted point alpha_s {a_fit:.5f} alpha_0 {a0_fit:.4f}: chi2 {best.fun:.3f}; wrote {PT}', flush=True)
    sys.exit()
_pt = json.load(open(PT))
assert abs(_pt['alpha_s'] - a_fit) < 1e-9 and abs(_pt['alpha_0'] - a0_fit) < 1e-9, 'the point file is stale, rerun the point mode'
u_star = np.array(_pt['u']); th = M.theta(torch.tensor(u_star)).numpy()

if mode == 'part':
    part, npart = int(os.environ['PART']), int(os.environ['NPART'])
    c2min = chi2_fg(u_star)[0]
    out = []
    for key, b, anch, sgn in tasks()[part::npart]:
        def f(u):
            uu = torch.tensor(u, requires_grad=True)
            p = predictions(uu, anchored=anch)[key]; v = p[b] if b >= 0 else p
            (sgn*v).backward()
            return sgn*float(v), uu.grad.numpy().copy()
        cons = {'type': 'ineq', 'fun': lambda u: c2min + 1.0 - chi2_fg(u)[0], 'jac': lambda u: -chi2_fg(u)[1]}
        r = minimize(lambda u: f(u)[0], u_star, jac=lambda u: f(u)[1], method='SLSQP', bounds=[(-1, 1)]*M.NT,
                     constraints=[cons], options=dict(maxiter=int(os.environ.get('MAXITER', '200')), ftol=1e-10))
        ok = chi2_fg(r.x)[0] <= c2min + 1.0 + 1e-3
        out.append([key, b, anch, sgn, sgn*float(r.fun), bool(ok), int(r.nit)])
        print(f'  {key} bin {b} {"after" if anch else "before"} {"min" if sgn > 0 else "max"}: {sgn*float(r.fun):.6g}'
              f'{"" if ok else "  (constraint not met)"}, {r.nit} iterations  [{time.time()-t0:.0f} s]', flush=True)
    json.dump(dict(part=part, c2min=c2min, results=out), open(f'{stem}_bands_part{part}.json', 'w'))
    print('done', flush=True)

elif mode == 'merge':
    ends = {}
    for fpath in glob.glob(f'{stem}_bands_part*.json'):
        for key, b, anch, sgn, val, ok, nit in json.load(open(fpath))['results']:
            ends[(key, b, anch, sgn)] = (val, ok)
    missing = [t for t in tasks() if t not in ends]
    assert not missing, f'{len(missing)} band ends missing, e.g. {missing[:3]}'
    bad = [k for k, (v, ok) in ends.items() if not ok]
    if bad:
        print(f'WARNING: the constraint was not met for {len(bad)} ends, e.g. {bad[:3]}')
    with torch.no_grad():
        u0 = torch.tensor(u_star)
        cen_a, cen_u = predictions(u0), predictions(u0, anchored=False)
        nsc, nnp = M.G['scale'].shape[2], M.G['np_vars'].shape[2]
        var = {kind: [predictions(u0, f'{kind}:{v}') for v in range(n)] for kind, n in (('scale', nsc), ('np', nnp))}
    from theory_cov import band as tband
    SL = [str(l) for l in M.G['scale_labels']]
    def hss(kind, key):
        d = [(p[key] - cen_a[key]).numpy() for p in var[kind]]
        if kind == 'scale':
            return tband(d, SL)                       # Eq. (sigmac): pairs at half weight, scheme difference once
        # the model variations: the Milan-factor pair at half weight, the third-order subtraction once
        return np.sqrt(0.5*(np.square(d[0]) + np.square(d[1])) + np.square(d[2]))
    half = lambda key, b, anch: 0.5*(ends[(key, b, anch, -1)][0] - ends[(key, b, anch, +1)][0])
    dist = {}
    for ex in FIG_EXPS:
        x = EX[ex]; wid = x['wid'].numpy(); usen = x['use'].numpy()
        # the figure divides by the normalization over the fitted bins of densities normalized over all
        # bins, so every density and band is stored on that scale: multiplied by S = sum_used(dens*wid)
        # of the all-bins normalization, which here equals the fraction of the sample in the fitted bins
        a_all = cen_a[ex].numpy(); u_all = cen_u[ex].numpy()
        fa = 1.0/np.sum(a_all*wid); fu = 1.0/np.sum(u_all*wid)        # used-bins to all-bins rescaling
        dist[f'{ex}_lo'], dist[f'{ex}_hi'] = x['lo'], x['hi']
        dist[f'{ex}_data'], dist[f'{ex}_err'] = x['y_raw'], x['e_raw']
        dist[f'{ex}_anchored'], dist[f'{ex}_unanchored'] = a_all*fa, u_all*fu
        dist[f'{ex}_knobs_after'] = np.array([half(ex, b, True) for b in range(x['nb'])])*fa
        dist[f'{ex}_knobs_before'] = np.array([half(ex, b, False) for b in range(x['nb'])])*fu
        dist[f'{ex}_theory'] = hss('scale', ex)*fa; dist[f'{ex}_np_model'] = hss('np', ex)*fa
    np.savez(f'{stem}_exchange_dist.npz', **dist, alpha_s=a_fit, alpha_0=a0_fit, theta=th, experiments=np.array(FIG_EXPS))
    rows = {}
    for k in OBS:
        rows[k] = dict(unanchored=float(cen_u[k]), anchored=float(cen_a[k]), theory=float(hss('scale', k)), np_model=float(hss('np', k)),
                       knobs_before=half(k, -1, False), knobs_after=half(k, -1, True))
    with torch.no_grad():
        lw, lw0 = logw(torch.tensor(u_star))
        o = M.observables(torch.exp(lw)); dsc = [M.observables(torch.exp(logw(torch.tensor(u_star), f'scale:{v}')[0]))['tau_win'] for v in range(nsc)]
        dnp = [M.observables(torch.exp(logw(torch.tensor(u_star), f'np:{v}')[0]))['tau_win'] for v in range(nnp)]
        rows['tau_win'] = dict(unanchored=float(M.observables(torch.exp(lw0))['tau_win']), anchored=float(o['tau_win']),
                               theory=float(tband([float(d) - float(o['tau_win']) for d in dsc], SL)),
                               np_model=float(np.sqrt(0.5*((float(dnp[0]) - float(o['tau_win']))**2 + (float(dnp[1]) - float(o['tau_win']))**2) + (float(dnp[2]) - float(o['tau_win']))**2)),
                               knobs_before=None, knobs_after=None)
    print(f'{"observable":10s} {"unanchored":>11s} {"anchored":>10s} {"theory":>9s} {"NP model":>9s} {"params before":>14s} {"params after":>13s}')
    for k, r in rows.items():
        kb = f'{r["knobs_before"]:14.5f}' if r['knobs_before'] is not None else f'{"":14s}'
        ka = f'{r["knobs_after"]:13.5f}' if r['knobs_after'] is not None else f'{"":13s}'
        print(f'{k:10s} {r["unanchored"]:11.5f} {r["anchored"]:10.5f} {r["theory"]:9.5f} {r["np_model"]:9.5f} {kb} {ka}')
    json.dump(dict(alpha_s=a_fit, alpha_0=a0_fit, chi2=float(chi2_fg(u_star)[0]), theta=th.tolist(), rows=rows,
                   method='exact constrained extremes (bands_exact.py)'), open(f'{stem}_exchange.json', 'w'), indent=1)
    print('wrote', f'{stem}_exchange.json', f'{stem}_exchange_dist.npz')
