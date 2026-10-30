#!/usr/bin/env python3
"""The envelope of NDRAW parameter variations against the linear covariance, for the main section of the
paper (prior only: generators and learning) and for the application (with the calculation imposed in the window).

Prior. Every parameter of the box, the mixing fraction included, is flat on its range, written as a standard
Gaussian z through u = erfc(-z/sqrt(2))/2, theta = lo + (hi - lo) u, so the middle 68% of each range is one sigma
and no draw leaves the trained box. BOX picks the ranges:
  full          the trained box, linear at its centre (xi = 1/2)
  tune          every range shrunk to TUNE_SCALE of its width around the central point (the default tunes,
                xi = 1/2), cut to the trained box
  tune_xifull   the same with xi over [0, 1]
  tune_sherpa   the Sherpa ranges shrunk as in tune, xi = 0 (pure Sherpa)
  tune_herwig   the Herwig ranges shrunk as in tune, xi = 1 (pure Herwig)
The learning weights are omega = w_at + N(0, C) and, after only, the calculation c = c0 + N(0, Sigma_c).

Predictions (labels in the json): the seven means of three_bands.py, the thrust histogram of TEST (its bins
below the window first, normalized over the fitted bins), the charged-multiplicity histogram (bins of two,
fractions summing to one) and the heavy-jet-mass histogram (density over [0, 0.4)), before and after.

Exact: the sample covariance over the NDRAW draws (Latin hypercube in u, so iid Gaussian in z up to the
stratification), each reweighted exactly and re-solved after, jointly and one source at a time (the others at
the centre); bootstrap SEs of the SDs; NREF more joint draws as a high-statistics reference. Linear, at the centre
m of the box (omega = w_at, c = c0), with J = dy/dtheta through the re-solved multipliers:
  lin_z     J diag((hi - lo) phi(0))^2 J^T     slopes in z at z = 0
  lin_flat  J diag((hi - lo)^2/12) J^T         slopes in the parameter with the flat width
  xi_quad   xi integrated by XI_NODES-point Gauss-Legendre, the other parameters lin_flat at each node:
            Var_xi[y(m, xi)] + E_xi[J(xi) diag(V) J(xi)^T]
plus J_omega C J_omega^T for learning and J_c Sigma_c J_c^T for the calculation (after), at the centre or averaged
over the xi nodes. Writes OUT.json (bands, ratios, SEs, correlation and whole-matrix comparisons) and OUT.npz
(draws, covariances, central values, bin edges, the test data).

    BOX=full python envelope_test.py MIXGEOV3S9 output/models/MIXGEOV3S9_cond.npz \\
        output/models/MIXGEOV3S9_ref_v2_slim.npz output/thrust_targets_nlo_point.npz output/wifi_MIXGEOV3S9_fit.npz
"""
import os, sys, json, time, copy, resource
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, tilt_on_t

tag, export, ref, grid, wfit = sys.argv[1:6]
E = os.environ.get
BOX = E('BOX', 'full'); NDRAW = int(E('NDRAW', '400')); NREF = int(E('NREF', '0')); SEED = int(E('SEED', '11'))
TUNE = float(E('TUNE_SCALE', str(1/3))); XI_NODES = int(E('XI_NODES', '8')); NBOOT = int(E('NBOOT', '300'))
TEST = E('TEST_DATA', 'delphi'); OUT = E('OUT', f'output/envelope_test_{tag}_{BOX}'); RES_TOL = float(E('RES_TOL', '1e-8'))
MC_NP = E('MC_NP', '0') == '1'
CENTRAL = E('CENTRAL')
torch.set_num_threads(int(E('TORCH_THREADS', '8'))); np.seterr(divide='ignore', invalid='ignore')
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
H = copy.copy(M); H.AE = M.AE.to(torch.float64); H.wifi = None    # float64 head for the derivatives, M (float32) for the draws
G = M.G; assert len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'a single-point target file is expected'
W = np.load(wfit); C = np.asarray(W['C'], np.float64); W_AT = np.asarray(W['w_published'], np.float64)
NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy(); is_mix = M.kind == 'mixture'; XI = NT - 1 if is_mix else None
assert is_mix and abs(cen[-1] - 0.5) < 1e-9 and abs(hw[-1] - 0.5) < 1e-9, ('xi box', cen[-1], hw[-1])
NS = M.NS                                                           # theta = (theta_S [NS], theta_H, xi)

# ---- predictions ------------------------------------------------------------------------------------------
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy()
low = np.where(x['lo'] < x['lo'][use].min() - 1e-12)[0]; ext = np.r_[low, use]
EXT = torch.as_tensor(ext); WEXT = x['wid'][EXT]
NCH_EDGES = np.r_[np.arange(0, 48, 2), np.inf]                     # [0,2), ..., [44,46), [46, inf)
nidx = np.clip(np.searchsorted(NCH_EDGES, M.nch, side='right') - 1, 0, len(NCH_EDGES) - 2)
NIDX = torch.as_tensor(nidx); NNB = len(NCH_EDGES) - 1
RHO_EDGES = np.r_[np.arange(0, 0.1, 0.01), 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.4]
rho = M.extra['rho_heavy']; rin = (rho >= 0) & (rho < RHO_EDGES[-1])
RIN = torch.as_tensor(np.where(rin)[0]); RIDX = torch.as_tensor(np.searchsorted(RHO_EDGES, rho[rin], side='right') - 1)
NRB = len(RHO_EDGES) - 1; RWID = torch.as_tensor(np.diff(RHO_EDGES))
LAB = (OBK + [f'thr_{x["lo"][b]:.3f}_{x["hi"][b]:.3f}' for b in ext]
       + [f'nch_{int(NCH_EDGES[i])}_{"inf" if np.isinf(NCH_EDGES[i + 1]) else int(NCH_EDGES[i + 1])}' for i in range(NNB)]
       + [f'rho_{RHO_EDGES[i]:.2f}_{RHO_EDGES[i + 1]:.2f}' for i in range(NRB)])
NM, NTH = len(OBK), len(ext); NO = len(LAB)
GROUPS = dict(means=np.arange(NM), thrust=np.arange(NM, NM + NTH), nch=np.arange(NM + NTH, NM + NTH + NNB),
              rho=np.arange(NM + NTH + NNB, NO))


def outv(lw):
    """The predictions as one differentiable vector: the means, then the thrust, multiplicity and rho histograms."""
    w = torch.exp(lw); o = M.observables(w)
    cnt = torch.zeros(x['nb'], dtype=w.dtype).index_add(0, x['idx'], w[x['ins']])
    nh = torch.zeros(NNB, dtype=w.dtype).index_add(0, NIDX, w)
    rh = torch.zeros(NRB, dtype=w.dtype).index_add(0, RIDX, w[RIN])/RWID
    return torch.cat([torch.stack([o[k] for k in OBK]), cnt[EXT]/(WEXT*cnt[x['use']].sum()), nh, rh])


# ---- the calculation and its variations (as strategy_test.py) ----------------------------------------------
C0 = np.asarray(M.target(0, 0, 'central'), np.float64); SIGP = torch.as_tensor(np.asarray(G['sigma_pert'][0, 0], np.float64))
labels = [str(l) for l in G['scale_labels']]; NPL = [str(l) for l in G['np_labels']]
VC = {**{l: np.asarray(G['scale'][0, 0, v], np.float64) for v, l in enumerate(labels)},
      **{l: np.asarray(G['np_vars'][0, 0, v], np.float64) for v, l in enumerate(NPL)}}
PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"
WT = {**{l: 0.5 for p in PAIRS for l in p}, SCHEME: 1.0, **({l: 1.0/len(NPL) for l in NPL} if MC_NP else {})}
D = np.stack([np.sqrt(wt)*(VC[l] - C0) for l, wt in WT.items()], 1); SIGC = D @ D.T
Dp = D[:, :2*len(PAIRS) + 1]; sig_rec = float(np.abs(Dp @ Dp.T - SIGP.numpy()).max()/np.abs(SIGP.numpy()).max())
assert sig_rec < 1e-8, f'sigma_pert of {grid} is not Eq. (sigmac) of its scale variations ({sig_rec:.1e})'
NEWTON = dict(solves=0, max_iterations=0, max_residual=0.0, unconverged=0, residual_tol=RES_TOL)


def prior(th, om, head=None):
    """log w0 at theta (tensor) with the learning weights om, with the float64 head H unless head is given."""
    hd = H if head is None else head; hd.set_wifi(om)
    try:
        lg = hd.logit(th)
    finally:
        hd.set_wifi(None)
    return lg - torch.logsumexp(lg, 0)


def tilt(lw0, c):
    """The maximum-entropy weight for the calculated values c (a tensor keeps the gradient), with the residual check."""
    cT = c if torch.is_tensor(c) else torch.as_tensor(np.asarray(c, np.float64))
    Sig = SIGP + torch.diag((M.floor_rel*torch.abs(cT))**2)
    lw, lam, _, its = tilt_on_t(M, lw0, cT, Sig, torch.zeros(len(M.keys)))
    with torch.no_grad():
        l0 = lw0.detach(); Fw = M.MW - cT.detach()[None, :]; Pw = torch.exp(torch.logsumexp(l0[M.IW], 0))
        zw = l0[M.IW] + Fw @ lam; lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
        r = float(torch.linalg.norm(torch.exp(zw - lZ) @ Fw + Pw**2*(Sig.detach() @ lam)))
    NEWTON['solves'] += 1; NEWTON['max_iterations'] = max(NEWTON['max_iterations'], its)
    NEWTON['max_residual'] = max(NEWTON['max_residual'], r); NEWTON['unconverged'] += int(not r < RES_TOL)
    return lw


def evaluate(theta, om, c, lw0=None):
    """y = (before, after) of length 2 NO, with the float32 head (as three_bands.py); lw0 skips the head."""
    with torch.no_grad():
        lw0 = prior(torch.as_tensor(np.asarray(theta, np.float64)), om, M) if lw0 is None else lw0
        return np.r_[outv(lw0).numpy(), outv(tilt(lw0, c)).numpy()]


def jac(theta, om, c):
    """(y, {theta, omega, c: dy/d.}) through the re-solved multipliers (y = before, after)."""
    leaf = dict(theta=torch.tensor(np.asarray(theta, np.float64), requires_grad=True),
                omega=torch.tensor(np.asarray(om, np.float64), requires_grad=True),
                c=torch.tensor(np.asarray(c, np.float64), requires_grad=True))
    ins = [leaf['theta'], leaf['omega'], leaf['c']]
    lw0 = prior(leaf['theta'], leaf['omega'])
    y = torch.cat([outv(lw0), outv(tilt(lw0, leaf['c']))])
    rows = [torch.autograd.grad(y[k], ins, retain_graph=True, allow_unused=True) if y[k].requires_grad else [None]*3
            for k in range(len(y))]
    J = {n: np.stack([np.zeros(len(ins[i])) if r[i] is None else r[i].numpy() for r in rows])
         for i, n in enumerate(('theta', 'omega', 'c'))}
    return y.detach().numpy(), J


sd = lambda Y: np.asarray(Y).std(0, ddof=1)
dg = lambda A: np.sqrt(np.maximum(np.diag(A), 0))
rat = lambda a, b: np.asarray(a, float)/np.where(np.asarray(b) == 0, np.nan, np.asarray(b, float))
per = lambda v: {l: float(a) for l, a in zip(LAB, np.asarray(v, float))}
corr = lambda A: np.nan_to_num(A/np.outer(dg(A), dg(A)))


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---- the box ---------------------------------------------------------------------------------------------
central = cen.copy(); central[XI] = 0.5
if CENTRAL:
    central = np.array([float(v) for v in CENTRAL.split(',')], np.float64); assert len(central) == NT
P = cen.copy() if BOX == 'full' else central.copy(); g = np.ones(NT)
if BOX != 'full':
    g[:] = TUNE
    if BOX == 'tune_xifull':
        g[XI] = 1.0
    if BOX in ('tune_sherpa', 'tune_herwig'):
        P[XI] = 0.0 if BOX == 'tune_sherpa' else 1.0; g[XI] = 0.0
        g[NS:XI] = 0.0 if BOX == 'tune_sherpa' else TUNE          # the other generator's parameters do not enter
        g[:NS] = TUNE if BOX == 'tune_sherpa' else 0.0
assert BOX in ('full', 'tune', 'tune_xifull', 'tune_sherpa', 'tune_herwig'), BOX
lo, hi = np.maximum(P - g*hw, cen - hw), np.minimum(P + g*hw, cen + hw)
m = (lo + hi)/2; R = hi - lo
PHI0 = 1/np.sqrt(2*np.pi)
print(f'[{tag}] {BOX}: {M.N} events, d={NT}, {len(W_AT)} learning weights, {NO} predictions; box centre {np.round(m, 4).tolist()}, '
      f'widths {np.round(R, 4).tolist()} [{time.time()-t0:.0f} s]', flush=True)
res = dict(tag=tag, box=BOX, tune_scale=TUNE, ndraw=NDRAW, nref=NREF, seed=SEED, xi_nodes=XI_NODES, labels=LAB,
           groups={k: v.tolist() for k, v in GROUPS.items()}, lo=lo.tolist(), hi=hi.tolist(), centre=m.tolist(),
           nch_edges=[float(e) if np.isfinite(e) else None for e in NCH_EDGES], rho_edges=RHO_EDGES.tolist(),
           thrust_edges=[float(x['lo'][b]) for b in ext] + [float(x['hi'][ext[-1]])], newton=NEWTON, timings={})
tick = [time.time()]


def done(stage):
    res['timings'][stage] = round(time.time() - tick[0], 1); tick[0] = time.time()
    res['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)
    json.dump(clean(res), open(OUT + '.json', 'w'), indent=1, allow_nan=False)
    print(f'[{tag}] {BOX} {stage} done in {res["timings"][stage]:.0f} s, Newton {NEWTON["solves"]} solves, '
          f'{NEWTON["unconverged"]} above {RES_TOL:g} [{time.time()-t0:.0f} s]', flush=True)


# ---- linear at the centre, and xi by quadrature ---------------------------------------------------------
yc, J = jac(m, W_AT, C0); Jt, Jw, Jc = J['theta'], J['omega'], J['c']
VFLAT = R**2/12; VZ = (R*PHI0)**2
LIN = dict(lin_z=(Jt*VZ) @ Jt.T, lin_flat=(Jt*VFLAT) @ Jt.T, learn=Jw @ C @ Jw.T, theory=Jc @ SIGC @ Jc.T)
if R[XI] > 0 and XI_NODES > 0:
    uq, wq = np.polynomial.legendre.leggauss(XI_NODES); uq, wq = (uq + 1)/2, wq/2
    V0 = VFLAT.copy(); V0[XI] = 0.0; YQ = []; Cq = np.zeros((2*NO, 2*NO)); Lq = np.zeros_like(Cq); Tq = np.zeros_like(Cq)
    for u, wk in zip(uq, wq):
        mk = m.copy(); mk[XI] = lo[XI] + R[XI]*u; yk, Jk = jac(mk, W_AT, C0); YQ.append(yk)
        Cq += wk*(Jk['theta']*V0) @ Jk['theta'].T; Lq += wk*Jk['omega'] @ C @ Jk['omega'].T; Tq += wk*Jk['c'] @ SIGC @ Jk['c'].T
        print(f'  xi node {mk[XI]:.4f} [{time.time()-t0:.0f} s]', flush=True)
    YQ = np.array(YQ); yq = wq @ YQ; Cxi = ((YQ - yq)*wq[:, None]).T @ (YQ - yq)
    LIN.update(xi_quad=Cq + Cxi, xi_only=Cxi, learn_xq=Lq, theory_xq=Tq); res['xi_nodes_values'] = (lo[XI] + R[XI]*uq).tolist()
    res['xi_quad_mean_shift_in_sd'] = per(rat((yq - yc)[NO:], dg(LIN['xi_quad'])[NO:]))
res['central'] = dict(before=per(yc[:NO]), after=per(yc[NO:]))
done('linear')

# ---- the draws ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
lhs = lambda n, d: (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d)))/n
try:
    ndtri = lambda U: torch.special.ndtri(torch.as_tensor(np.clip(U, 1e-12, 1 - 1e-12))).numpy()   # z of u = erfc(-z/sqrt2)/2
    ndtri(np.array([0.5]))
except AttributeError:
    from statistics import NormalDist
    ndtri = np.vectorize(NormalDist().inv_cdf)
ew, Vw = np.linalg.eigh(C); BW = Vw*np.sqrt(np.maximum(ew, 0))     # omega = w_at + BW z, c = c0 + D z


def draws(n, tagn):
    U = lhs(n, NT); TH = lo + R*U; ZW = rng.standard_normal((n, len(W_AT))); ZC = rng.standard_normal((n, D.shape[1]))
    OM = W_AT + ZW @ BW.T; CS = C0 + ZC @ D.T
    return dict(U=U, Z=ndtri(U), TH=TH, OM=OM, CS=CS)


DR = draws(NDRAW, 'main'); SETS = dict(joint=('TH', 'OM', 'CS'), theta=('TH',), omega=('OM',), c=('CS',))
with torch.no_grad():
    LW_M = prior(torch.as_tensor(m), W_AT, M)                      # the c-only set: the head at the centre, once
Y = {}
for name, vary in SETS.items():
    rows = []
    for i in range(NDRAW):
        th = DR['TH'][i] if 'TH' in vary else m; om = DR['OM'][i] if 'OM' in vary else W_AT; cc = DR['CS'][i] if 'CS' in vary else C0
        rows.append(evaluate(th, om, cc, LW_M if name == 'c' else None))
        if i % 100 == 0:
            print(f'  {name} draw {i+1}/{NDRAW} [{time.time()-t0:.0f} s]', flush=True)
    Y[name] = np.array(rows)
done('draws')

# ---- comparisons -----------------------------------------------------------------------------------------
BOOT = {}                                                           # bootstrap resamples per number of draws


def boots(n):
    if n not in BOOT:
        BOOT[n] = np.random.default_rng(SEED + n).integers(0, n, (NBOOT, n))
    return BOOT[n]


def compare(Cm, Ys, side):
    """Model covariance Cm (2NO x 2NO) against the draws Ys on one side: SD ratios with bootstrap SEs, correlation
    differences per group, and the generalized eigenvalues of the model in the principal subspace of the draws."""
    s = slice(0, NO) if side == 'before' else slice(NO, 2*NO)
    A, Yd = Cm[s, s], Ys[:, s]; Ce = np.cov(Yd, rowvar=False); se, sm = dg(Ce), dg(A); boot = boots(len(Yd))
    sb = np.array([sd(Yd[b]) for b in boot]); r = rat(sm, se); rse = r*rat(sb.std(0, ddof=1), se)
    out = dict(sd_exact=per(se), sd_model=per(sm), ratio=per(r), ratio_se=per(rse))
    for gname, idx in GROUPS.items():
        ok = idx[(se[idx] > 0) & np.isfinite(se[idx])]
        if len(ok) < 2:
            continue
        dc = corr(A[np.ix_(ok, ok)]) - corr(Ce[np.ix_(ok, ok)]); iu = np.triu_indices(len(ok), 1)
        cb = np.array([corr(np.cov(Yd[b][:, ok], rowvar=False))[iu] for b in boot[:100]])
        # whitened by the draws' SDs: generalized eigenvalues of (model, draws) in the subspace holding 99.9% of the draws' variance
        Sn = 1/se[ok]; Ce_n, A_n = Ce[np.ix_(ok, ok)]*np.outer(Sn, Sn), A[np.ix_(ok, ok)]*np.outer(Sn, Sn)
        ev, V = np.linalg.eigh(Ce_n); keep = np.argsort(ev)[::-1]; cum = np.cumsum(ev[keep])/ev.sum()
        k = int(np.searchsorted(cum, 0.999) + 1); Vk = V[:, keep[:k]]*(1/np.sqrt(ev[keep[:k]]))
        gev = np.linalg.eigvalsh(Vk.T @ A_n @ Vk)
        out[f'corr_{gname}'] = dict(max_abs_diff=float(np.abs(dc[iu]).max()), rms_diff=float(np.sqrt((dc[iu]**2).mean())),
                                    corr_se_median=float(np.median(cb.std(0, ddof=1))), n=int(len(ok)),
                                    gen_eig_min=float(gev.min()), gen_eig_max=float(gev.max()), gen_eig_k=k)
    return out


models = {'lin_z': ('lin_z', 'learn', 'theory'), 'lin_flat': ('lin_flat', 'learn', 'theory')}
if 'xi_quad' in LIN:
    models['xi_quad'] = ('xi_quad', 'learn_xq', 'theory_xq')
res['compare'] = {}
for side in ('before', 'after'):
    for setname in ('joint', 'theta', 'omega', 'c'):
        if side == 'before' and setname == 'c':
            continue
        for mname, (kt, kl, kc) in models.items():
            Cm = (LIN[kt] if setname in ('joint', 'theta') else 0) + (LIN[kl] if setname in ('joint', 'omega') else 0) \
                 + (LIN[kc] if side == 'after' and setname in ('joint', 'c') else 0)
            if setname in ('omega', 'c') and mname != 'lin_flat':
                continue                                           # one linear model for learning and the calculation
            res['compare'][f'{side}/{setname}/{mname}'] = compare(Cm, Y[setname], side)
done('compare')
for side in ('before', 'after'):
    for mname in models:
        c = res['compare'][f'{side}/joint/{mname}']; r = np.array([v for v in c['ratio'].values() if v is not None])
        print(f'  {side} joint {mname}: SD ratio {np.nanmin(r):.2f} to {np.nanmax(r):.2f}; '
              + ', '.join(f'{gname} corr max diff {c[f"corr_{gname}"]["max_abs_diff"]:.2f} (eig {c[f"corr_{gname}"]["gen_eig_min"]:.2f}-'
                          f'{c[f"corr_{gname}"]["gen_eig_max"]:.2f})' for gname in GROUPS if f'corr_{gname}' in c), flush=True)
save_npz = lambda: np.savez_compressed(
    OUT + '.npz', labels=np.array(LAB), lo=lo, hi=hi, centre=m, y_central=yc, **{f'Y_{k}': v for k, v in Y.items()},
    **{f'C_{k}': v for k, v in LIN.items()}, U=DR['U'], Z=DR['Z'], TH=DR['TH'],
    thrust_edges=np.array(res['thrust_edges']), nch_edges=NCH_EDGES, rho_edges=RHO_EDGES,
    data_lo=x['lo'], data_hi=x['hi'], data_y=x['y'].numpy(), data_e=x['e'].numpy(), data_use=use, ext=ext)
save_npz()
if NREF:                                                            # the sampling error of NDRAW draws, against NREF more
    DRR = draws(NREF, 'ref'); Y['ref'] = np.array([evaluate(DRR['TH'][i], DRR['OM'][i], DRR['CS'][i]) for i in range(NREF)])
    for side in ('before', 'after'):
        res['compare'][f'{side}/draws{NDRAW}_vs_ref'] = compare(np.cov(Y['joint'], rowvar=False), Y['ref'], side)
    done('reference'); save_npz()
print(f'ENVELOPE TEST {tag} {BOX} DONE [{time.time()-t0:.0f} s]', flush=True)
