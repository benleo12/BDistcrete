#!/usr/bin/env python3
"""Strategies for the combined uncertainty of one reweighted sample, against the re-solved bands of three_bands.py.

Same model, reference, calculation, wifi fit, test data and seven means as three_bands.py. A prediction y is
the vector of the seven means and the test histogram, the bins below FIRST_BIN first (n_low of them, the _low
bins of three_bands.py), all normalized over the bins above it. Every band is given before (generators alone)
and after the calculation is imposed, where meaningful. S0 uses the float32 head of three_bands.py; S1-S6 use
a float64 copy of it (LIN_DTYPE), so that derivatives and finite differences are free of rounding in the head.
  S0  re-solved generator band: the SD over the NODES Latin-hypercube nodes of three_bands.py (same SEED),
      checked against the stored H_before/H_after/rows (max relative difference of the SDs)
  S1  linear generator band at the box centre: sigma^2 = diag(J diag(hw^2/3) J^T), J = dy/dtheta for the 17
      parameters with xi, by autograd through the differentiable Newton step at the converged multipliers
      (so with the multipliers' response dlam = -A^-1 B dtheta of App. D), as jac_wifi does for omega; J is
      checked against central finite differences with full re-solves for FD_PARAMS
  S2  the same at the central point of three_bands.py (CENTRAL, the default tunes with xi = 1/2)
  LJ  the linear joint covariance of Jesse's proposal, J_th Cov_th J_th^T + J_w C J_w^T + J_c Sigma_c J_c^T,
      at the box centre (against S5) and at the central point (against the r8 quadrature), with its correlations
  S3  width scan: the S0 hypercube shrunk by s around the box centre, re-solved SD against s times the S1
      band and against the S1 tangent on the same nodes; R^2 of a least-squares linear fit over the nodes
  S4  theory: J_c = dy/dc through the solve, with the explicit dependence of the moment functions m - c and
      of the floor of Sigma on c (directlib.tilt_on_t), so J_c (c_v - c) is the linear displacement of each
      variation against the re-solved one; the Gaussian form sqrt(diag(J_c Sigma_c J_c^T)) against the
      symmetric re-solved band (th_hist of three_bands.py); Sigma_c is sigma_pert of the targets file
      (Eq. sigmac, built by targets_nlo_point.py), written as D D^T over the variations. Also Jesse's literal
      form, h = m - c held fixed and c only the target (the terms App. D calls not small are then missing)
  S5  joint Monte Carlo: MC_N draws theta uniform on the box, omega = w_at + N(0, C), c = c + N(0, Sigma_c),
      each re-solved; SD and correlation matrix against the r8 quadrature generator (S0) + symmetric theory
      (S4) + learning (no statistical error), against the linear joint (LJ), and against three one-source MCs
      on the same draws (the other sources at the central point): their quadrature isolates the cross terms
      (var_joint - sum var_i), their linear sum is reported as well
  LL  log-linear propagation: log w of every event linear in the inputs at the box centre (the derivatives of D1,
      before and after), the predictions exact in those weights, on the S5 draws (joint, and theta alone) and on
      the S3 nodes at each width, against the re-solved SDs; also the log-linear prior with the multipliers
      re-solved at the drawn c (band_resolved, ratio_joint_resolved: no network evaluation, one Newton solve per draw)
  S6  hybrid: at the first HYB_N nodes of S0 the linear learning and theory covariances, J_w C J_w^T and
      J_c Sigma_c J_c^T (and the re-solved theory covariance), Cov = Cov_theta(y) over the S0 nodes +
      E_theta[learning + theory], against the quadrature and S5
When each strategy fails: per width of S3, per variation of S4 and per draw set of S5, against the linearisation
point (box centre for S3 and the joint and theta sets, the central point for S4 and the omega and c sets):
  D1  sigma_lnw, the SD over the events (weights of the linearisation point) of the change of log w, RMS over the
      nodes or draws; also its generator part (log w0, f/kappa normalized), its maximum-entropy part (lam.m - log Z)
      and the generator part under the weights before. PREDICTED from one point alone: sqrt(E Var_w(J_lnw dz)) with
      J_lnw the event-level derivatives (central differences of float64 re-solves) along unit-variance directions dz
      (sqrt(VTH) e_p, C^1/2, D), as tr Cov_w(J_lnw) for the prior (cross terms between parameters vanish in
      expectation, independent sources) and as the mean of dz^T Cov_w(J_lnw) dz over the actual nodes or draws
  D2  Kish N_eff/N before and after, absolute and relative to the linearisation point, min and median
  D3  cond(A) of the Newton matrix (raw and with unit diagonal) at the point and its range; |lam - lam*|/|lam*|
  D4  per prediction, the RMS over the nodes of (re-solved - tangent) over the linear band (S3)
failure_rule collects them per width with the worst |linear/re-solved - 1| over the predictions and the fraction
outside 10%, the first width at which linear fails by more than 10%, the theory variations (failure measure: the
fraction of predictions whose linear displacement is off by more than 10% of the re-solved band), and for the
quadrature the measured cross-term fraction of S5 split exactly on the shared theta draws into A, the change of the
learning and theory variances across the box, against E_theta[learning + theory] - their central values (S6), and
B = 2 Cov_theta(y, b), the mean shift b of omega and c with theta (not predicted by S6, of the same order).
checks.lnw_chain_rule_in_band: the finite-difference log w derivatives of D1 against the autograd Jacobians.
MC_NP=1 adds the three dispersive-model variations to Sigma_c at weight 1/3 each (their root mean square,
as in the signed bands of three_bands.py); by default Sigma_c is that of Eq. (sigmac) alone.
MC_CRN=1 draws S5 on common random numbers: theta at the S0 nodes in turn (MC_N a multiple of NODES, so the generator
part of the joint and of the r8 quadrature is the same sample, and the bootstrap resamples the nodes with the draws),
omega and c made orthogonal to the nodes and whitened together (sample covariance exactly C and Sigma_c, no sample
correlation between the sources), so that joint against quadrature carries no sampling difference to linear order:
what remains is the cross terms and the one-source differences (Gaussian Sigma_c against the re-solved variations,
re-solved against linear learning), ratio_to_one_source and cross_fraction separating the two. The bootstrap SEs
resample an iid design and are then upper bounds. S5 also gives the tangent at the box centre on the same draws
(linear_same_draws): joint against it is linear against re-solved free of the sampling of the draws.

    python strategy_test.py MIXGEOV3S9 output/models/MIXGEOV3S9_cond.npz output/models/MIXGEOV3S9_ref_v2_slim.npz \\
        output/thrust_targets_nlo_point.npz output/wifi_MIXGEOV3S9_fit.npz
Writes output/strategy_test_<tag>.json (strict JSON, nan written as null), again after every stage.
"""
import os, sys, json, time, copy, resource
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, tilt_on_t

tag, export, ref, grid, wfit = sys.argv[1:6]
E = os.environ.get
NODES = int(E('NODES', '200')); SEED = int(E('SEED', '11')); TEST = E('TEST_DATA', 'delphi')
OUT = E('OUT', f'output/strategy_test_{tag}'); STORED = E('STORED', f'output/three_bands_{tag}')
WIDTHS = [float(s) for s in E('WIDTHS', '0.1,0.2,0.3,0.5,0.75,1.0').split(',')]
MC_N = int(E('MC_N', '400')); HYB_N = int(E('HYB_N', '50')); MC_NP = E('MC_NP', '0') == '1'; NBOOT = int(E('NBOOT', '300'))
MC_CRN = E('MC_CRN', '0') == '1'
MIX_FORM = E('MIX_FORM', 'geometric'); assert MIX_FORM in ('geometric', 'additive'), MIX_FORM
FD_PARAMS = [int(p) for p in E('FD_PARAMS', '0,8,16').split(',')]; FD_STEPS = [float(h) for h in E('FD_STEPS', '1e-2,1e-3').split(',')]
LIN_DTYPE = E('LIN_DTYPE', 'float64'); RES_TOL = float(E('RES_TOL', '1e-8'))
torch.set_num_threads(int(E('TORCH_THREADS', '8'))); np.seterr(divide='ignore', invalid='ignore')     # empty bins give nan ratios
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
# the head for S1-S6: the same model with the event functions in LIN_DTYPE (a shallow copy, one more AE array)
M64 = M if LIN_DTYPE == 'float32' else copy.copy(M)
if M64 is not M:
    M64.AE = M.AE.to(getattr(torch, LIN_DTYPE)); M64.wifi = None
HEAD = [M]                                                           # the head prior() uses; M64 after the checks
G = M.G; assert len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'a single-point target file is expected'
W = np.load(wfit); w_wifi = W['w']; C = np.asarray(W['C'], np.float64)
CENW = w_wifi if E('WIFI_CENTRAL', '0') == '1' else None              # the central log ratio, as in three_bands.py
W_AT = np.asarray(w_wifi if CENW is not None else W['w_published'], np.float64)   # the same function in the wifi basis
NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy(); is_mix = M.kind == 'mixture'
if is_mix:                                                          # Cov(theta) below takes xi uniform on [0, 1]
    assert abs(cen[-1] - 0.5) < 1e-9 and abs(hw[-1] - 0.5) < 1e-9, ('xi box', cen[-1], hw[-1])
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy(); wid = x['wid'].numpy()
low = np.where(x['lo'] < x['lo'][use].min() - 1e-12)[0]; ext = np.r_[low, use]; nlow = len(low)
LAB = OBK + [f'bin_{x["lo"][b]:.3f}_{x["hi"][b]:.3f}' for b in ext]; NM, NO = len(OBK), len(OBK) + len(ext)
EXT = torch.as_tensor(ext); WEXT = x['wid'][EXT]
gb = lambda t: t.element_size()*t.nelement()/2**30
print(f'[{tag}] {M.N} reference events, d={NT}, head {M.kind} ({M.mix_form}), {len(W_AT)} wifi weights, {NO} predictions '
      f'({nlow} bins below the fitted ones), box centre xi {cen[-1]:.3f} half-width {hw[-1]:.3f}; AE {gb(M.AE):.1f} GB float32'
      f'{"" if M64 is M else f" + {gb(M64.AE):.1f} GB {LIN_DTYPE}"} [{time.time()-t0:.0f} s]', flush=True)
_DEFAULT = object()


def predict(theta, which='central', anchored=True, wifi=_DEFAULT):
    """three_bands.predict, verbatim (S0 and the API checks)."""
    M.set_wifi(CENW if wifi is _DEFAULT else wifi)
    with torch.no_grad():
        lw, lw0, lam, Pw, its = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, which=which, anchored=anchored)
        w = torch.exp(lw); o = M.observables(w); o = {k: float(o[k]) for k in OBK if k in o}
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
        h = cnt[ext]/(wid[ext]*cnt[use].sum())
    M.set_wifi(None)
    return o, h, float(torch.exp(torch.logsumexp(lw0[M.IW], 0))) if anchored else None


def py(theta, anchored=True, which='central'):
    o, h, _ = predict(theta, which=which, anchored=anchored)
    return np.r_[[o[k] for k in OBK], h]


# ---- the calculation, its variations and Sigma_c ---------------------------------------------------
C0 = np.asarray(M.target(0, 0, 'central'), np.float64); SIGP = torch.as_tensor(np.asarray(G['sigma_pert'][0, 0], np.float64))
labels = [str(l) for l in G['scale_labels']]; NPL = [str(l) for l in G['np_labels']]
VC = {**{l: np.asarray(G['scale'][0, 0, v], np.float64) for v, l in enumerate(labels)},
      **{l: np.asarray(G['np_vars'][0, 0, v], np.float64) for v, l in enumerate(NPL)}}
PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"
WT = {**{l: 0.5 for p in PAIRS for l in p}, SCHEME: 1.0, **({l: 1.0/len(NPL) for l in NPL} if MC_NP else {})}
D = np.stack([np.sqrt(wt)*(VC[l] - C0) for l, wt in WT.items()], 1)          # Sigma_c = D D^T
SIGC = D @ D.T
Dp = D[:, :2*len(PAIRS) + 1]; sig_rec = float(np.abs(Dp @ Dp.T - SIGP.numpy()).max()/np.abs(SIGP.numpy()).max())
assert sig_rec < 1e-8, f'sigma_pert of {grid} is not Eq. (sigmac) of its scale variations ({sig_rec:.1e})'
NEWTON = dict(solves=0, max_iterations=0, max_residual=0.0, unconverged=0, residual_tol=RES_TOL)
LAST = {}                                                           # tilt(): multipliers and cond(A) of the last solve


def prior1(th, om):
    """log w0 at theta (tensor) for the wifi weights om (None: the published network), with the head HEAD[0]."""
    H = HEAD[0]; H.set_wifi(om)
    try:
        lg = H.logit(th)
    finally:
        H.set_wifi(None)
    return lg - torch.logsumexp(lg, 0)


def prior(th, om):
    """prior1, or with MIX_FORM=additive the population mixture (1 - xi) w(xi=0) + xi w(xi=1) of the two
    normalized endpoints (the same networks; the predictions before the tilt are then linear in xi). S0 and
    the API checks (predict) keep the geometric head."""
    if MIX_FORM == 'geometric' or not is_mix:
        return prior1(th, om)
    f, z = th[-1], torch.zeros(1, dtype=th.dtype)
    a, b = prior1(torch.cat([th[:-1], z]), om), prior1(torch.cat([th[:-1], z + 1]), om); big = torch.maximum(a, b)
    return big + torch.log(torch.clamp((1 - f)*torch.exp(a - big) + f*torch.exp(b - big), min=1e-300))


def tilt_fixed_h(lw0, cT, Sig):
    """tilt_on_t with h = m - c held at the value of c (detached): c enters only as the target of E_w[h] = c - c."""
    IW, cd = M.IW, cT.detach(); Fd = M.MW - cd[None, :]
    Pw = torch.exp(torch.logsumexp(lw0[IW], 0)); Sc = Pw**2*Sig
    with torch.no_grad():
        lam, ww, m, its = Model.newton(lw0[IW].detach(), torch.log1p(-Pw.detach()), Fd, Sc.detach(), torch.zeros(len(M.keys)))
        J = (Fd*ww[:, None]).T @ Fd - torch.outer(m, m) + Sc.detach()
    zw = lw0[IW] + Fd @ lam
    lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
    g = torch.exp(zw - lZ) @ Fd - (cT - cd) + Sc @ lam
    z = lw0.index_add(0, IW, Fd @ (lam - torch.linalg.solve(J, g)))
    return z - torch.logsumexp(z, 0), lam, int(its)


def tilt(lw0, c, fixed_h=False):
    """Model.weights for any calculated values c (a tensor keeps the gradient), Sigma = Model.sigma(c); records
    the residual |E_w[F] + Pw^2 Sigma lam| of the Newton solve at its multipliers."""
    cT = c if torch.is_tensor(c) else torch.as_tensor(np.asarray(c, np.float64))
    Sig = SIGP + torch.diag((M.floor_rel*torch.abs(cT))**2)
    if fixed_h:
        lw, lam, its = tilt_fixed_h(lw0, cT, Sig)
    else:
        lw, lam, _, its = tilt_on_t(M, lw0, cT, Sig, torch.zeros(len(M.keys)))
    with torch.no_grad():
        l0 = lw0.detach(); Fw = M.MW - cT.detach()[None, :]; Pw = torch.exp(torch.logsumexp(l0[M.IW], 0))
        zw = l0[M.IW] + Fw @ lam; lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
        ww = torch.exp(zw - lZ); r = float(torch.linalg.norm(ww @ Fw + Pw**2*(Sig.detach() @ lam)))
        mm = ww @ Fw; A = (Fw*ww[:, None]).T @ Fw - torch.outer(mm, mm) + Pw**2*Sig.detach(); dA = torch.sqrt(torch.diag(A))
        LAST.update(lam=lam.detach().numpy().copy(), cond_A=float(torch.linalg.cond(A)),
                    cond_A_scaled=float(torch.linalg.cond(A/torch.outer(dA, dA))))
    NEWTON['solves'] += 1; NEWTON['max_iterations'] = max(NEWTON['max_iterations'], its)
    NEWTON['max_residual'] = max(NEWTON['max_residual'], r); NEWTON['unconverged'] += int(not r < RES_TOL)
    return lw


def outv(lw):
    """The predictions as one differentiable vector: the means of OBK, then the histogram over ext."""
    w = torch.exp(lw); o = M.observables(w)
    cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']])
    return torch.cat([torch.stack([o[k] for k in OBK]), cnt[EXT]/(WEXT*cnt[x['use']].sum())])


def evaluate(theta, om, cs=()):
    """y before, and after for every c of cs, from one pass of the head."""
    with torch.no_grad():
        lw0 = prior(torch.as_tensor(np.asarray(theta, np.float64)), om)
        return outv(lw0).numpy(), [outv(tilt(lw0, c)).numpy() for c in cs]


def jac(theta, om, c, wrt, fixed_h=False):
    """(y, {name: dy/dname}, log w0) with y = (before, after) of length 2 NO, through the re-solved multipliers."""
    leaf = dict(theta=torch.tensor(np.asarray(theta, np.float64), requires_grad='theta' in wrt),
                omega=None if om is None else torch.tensor(np.asarray(om, np.float64), requires_grad='omega' in wrt),
                c=torch.tensor(np.asarray(c, np.float64), requires_grad='c' in wrt))
    ins = [leaf[n] for n in wrt]; assert all(t is not None for t in ins)
    lw0 = prior(leaf['theta'], leaf['omega'])
    y = torch.cat([outv(lw0), outv(tilt(lw0, leaf['c'], fixed_h))])
    rows = [torch.autograd.grad(y[k], ins, retain_graph=True, allow_unused=True) if y[k].requires_grad else [None]*len(ins)
            for k in range(len(y))]
    J = {n: np.stack([np.zeros(len(ins[i])) if r[i] is None else r[i].numpy() for r in rows]) for i, n in enumerate(wrt)}
    return y.detach().numpy(), J, lw0.detach()


sd = lambda Y: np.asarray(Y).std(0, ddof=1)
rat = lambda a, b: np.asarray(a, float)/np.where(np.asarray(b) == 0, np.nan, np.asarray(b, float))
per = lambda v: {l: float(a) for l, a in zip(LAB, np.asarray(v, float))}
ba = lambda v0, v1: dict(before=None if v0 is None else per(v0), after=per(v1))
relmax = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))/np.maximum(np.abs(np.asarray(b)), 1e-300)))
qv = lambda J, Cv: np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', J, Cv, J), 0))
dg = lambda A: np.sqrt(np.maximum(np.diag(A), 0))
corr = lambda A: np.nan_to_num(A/np.outer(dg(A), dg(A)))


def r2(Y, X):
    """R^2 and adjusted R^2 of the least-squares fit of every column of Y on X with an intercept."""
    A = np.c_[np.ones(len(X)), X]; res = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    R = 1 - (res**2).sum(0)/((Y - Y.mean(0))**2).sum(0); n, p = X.shape
    return R, 1 - (1 - R)*(n - 1)/(n - p - 1)


# ---- D1-D3: the weights at a node against the linearisation point R -------------------------------------
DK = ('sigma_lnw', 'sigma_lnw_generator', 'sigma_lnw_maxent', 'sigma_lnw_before', 'kish_before', 'kish_after', 'kish_rel_before',
      'kish_rel_after', 'cond_A', 'cond_A_scaled', 'dlam_rel')
wsd = lambda w, d: float(torch.sqrt(torch.clamp(w @ (d - w @ d)**2, min=0)))           # SD over the events under w


def kish(l, w=None):
    """Kish N_eff/N of the weights exp(l); with w, of exp(l) relative to the weights w (1/(1 + CV^2) under w)."""
    r = torch.exp(l - l.max())
    return float(r.sum()**2/(len(r)*(r**2).sum())) if w is None else float((w @ r)**2/(w @ r**2))


def evalw(theta, om, c, R=None):
    """evaluate() for one c, with the D1-D3 row of the node against R; R None returns the point itself as R."""
    with torch.no_grad():
        lw0 = prior(torch.as_tensor(np.asarray(theta, np.float64)), om); lw = tilt(lw0, c)
        return outv(lw0).numpy(), outv(lw).numpy(), mkref(lw0, lw) if R is None else dstat(R, lw0, lw)


def mkref(lw0, lw):
    return dict(lw0=lw0, lw=lw, w0=torch.exp(lw0), w=torch.exp(lw), kish_before=kish(lw0), kish_after=kish(lw), **LAST)


def dstat(R, lw0, lw):
    """One row of DK (the last solve is that of lw)."""
    d0, d = lw0 - R['lw0'], lw - R['lw']
    return [wsd(R['w'], d), wsd(R['w'], d0), wsd(R['w'], d - d0), wsd(R['w0'], d0), kish(lw0), kish(lw), kish(d0, R['w0']),
            kish(d, R['w']), LAST['cond_A'], LAST['cond_A_scaled'],
            float(np.linalg.norm(LAST['lam'] - R['lam'])/max(np.linalg.norm(R['lam']), 1e-300))]


def dsum(rows, R):
    """D1 RMS over the rows, D2 min and median, D3 at R and range, |dlam|/|lam*| median and max."""
    a = np.array(rows); o = {k: float(np.sqrt(np.mean(a[:, i]**2))) for i, k in enumerate(DK[:4])}
    o.update({k: dict(at_linearisation=R.get(k), min=float(a[:, i].min()), median=float(np.median(a[:, i]))) for i, k in enumerate(DK[4:8], 4)})
    o.update({k: dict(at_linearisation=R[k], min=float(a[:, i].min()), max=float(a[:, i].max())) for i, k in enumerate(DK[8:10], 8)})
    o['dlam_rel'] = dict(median=float(np.median(a[:, 10])), max=float(a[:, 10].max()))
    return o


def gcov(w, X):
    """Cov_w over the events (rows) of the columns of X."""
    Xc = X - w @ X
    return ((Xc*w[:, None]).T @ Xc).numpy()


def lnw_cov(theta, om, c, srcs, Jy, h=1e-3, e=1e-4):
    """(Cov_w of the event-level derivatives of log w, of log w0, of log w - log w0, Cov_w0 of those of log w0; the columns
    of each source) along the unit-variance directions of srcs (theta: sqrt(VTH) e_p, omega: C^1/2, c: D), by central
    differences of float64 re-solves at h such units: all that the predicted sigma_lnw needs, from one point. Checked by
    the chain rule against the autograd Jacobians Jy of jac() at the same point: (outv(lw + e J_k) - outv(lw - e J_k))/2e,
    log w normalized, against Jy B, as the max error in units of each source's linear band (before and after)."""
    base = dict(theta=np.asarray(theta, np.float64), omega=np.asarray(om, np.float64), c=np.asarray(c, np.float64))
    cols, ix, n, BS = [], {}, 0, {}
    for s in srcs:
        B = BS[s] = dict(theta=np.diag(np.sqrt(VTH)), omega=BW, c=D)[s]; ix[s] = np.arange(n, n + B.shape[1]); n += B.shape[1]
        for b in B.T:
            pm = []
            for sg in (1, -1):
                a = {**base, s: base[s] + sg*h*b}
                with torch.no_grad():
                    l0 = prior(torch.as_tensor(a['theta']), a['omega']); pm.append((tilt(l0, a['c']), l0))
            cols.append([(p - m)/(2*h) for p, m in zip(*pm)])
    _, _, R = evalw(theta, om, c)
    J, J0 = torch.stack([u for u, _ in cols], 1), torch.stack([v for _, v in cols], 1)
    nl, chk = lambda z: z - torch.logsumexp(z, 0), {}
    with torch.no_grad():
        for side, l, Jl, a in (('before', R['lw0'], J0, 0), ('after', R['lw'], J, NO)):
            dy = np.stack([(outv(nl(l + e*Jl[:, k])) - outv(nl(l - e*Jl[:, k]))).numpy()/(2*e) for k in range(n)], 1)
            for s in srcs:
                Bm = Jy[s][a:a + NO] @ BS[s]; band = np.sqrt((Bm**2).sum(1)); ok = band > 0
                chk[f'{s}_{side}'] = float(np.max((np.abs(dy[:, ix[s]] - Bm).max(1)/np.where(ok, band, 1))[ok])) if ok.any() else None
    R.update(J=J, J0=J0)                                            # kept for the log-linear propagation (LL)
    return [gcov(R['w'], J), gcov(R['w'], J0), gcov(R['w'], J - J0), gcov(R['w0'], J0)], ix, R, chk


def pred(Gs, sel, Z=None):
    """Predicted D1 (the four parts): sqrt(tr G) for dz of unit covariance on the columns sel, or sqrt(mean dz^T G dz) over
    the rows dz of Z (the actual nodes or draws: the cross terms between parameters kept, their fraction reported)."""
    o = {}
    for k, G in zip(DK[:4], Gs):
        G = G[np.ix_(sel, sel)]
        o[k] = float(np.sqrt(max(np.trace(G) if Z is None else np.mean(np.einsum('ni,ij,nj->n', Z, G, Z)), 0)))
    if Z is not None:
        o['cross_parameter_fraction'] = float(1 - np.mean((Z**2) @ np.diag(Gs[0][np.ix_(sel, sel)]))/max(o['sigma_lnw']**2, 1e-300))
    return o


def fails(rl, idx=slice(None)):
    """Worst |ratio - 1| and the fraction outside [0.9, 1.1] over the finite ratios rl[idx]."""
    v = np.abs(np.asarray(rl, float)[idx] - 1); v = v[np.isfinite(v)]
    return (float(v.max()), float(np.mean(v > 0.1))) if len(v) else (None, None)


def clean(o):
    """nan and inf as null, for strict JSON readers."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o


res = dict(tag=tag, export=export, grid=grid, wifi=wfit, wifi_central=CENW is not None, test=TEST, labels=LAB, n_low=int(nlow),
           observables=OBK, nodes=NODES, seed=SEED, widths=WIDTHS, mc_n=MC_N, hyb_n=HYB_N, mc_np=MC_NP, mc_crn=MC_CRN, mix_form=MIX_FORM, lin_dtype=LIN_DTYPE,
           sigma_c_variations={l: wt for l, wt in WT.items()}, checks=dict(sigma_pert_rebuilt=sig_rec), timings={}, newton=NEWTON)
tick = [time.time()]


def done(stage):
    res['timings'][stage] = round(time.time() - tick[0], 1); tick[0] = time.time()
    res.setdefault('peak_rss_gb', {})[stage] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)   # Linux: kB
    json.dump(clean(res), open(OUT + '.json', 'w'), indent=1, allow_nan=False)
    print(f'[{tag}] {stage} done in {res["timings"][stage]:.0f} s, peak RSS {res["peak_rss_gb"][stage]:.1f} GB, '
          f'Newton {NEWTON["solves"]} solves, {NEWTON["unconverged"]} above {RES_TOL:g} [{time.time()-t0:.0f} s]', flush=True)


# ---- the points and the API checks -------------------------------------------------------------------
central = cen.copy()
if is_mix:
    central[-1] = 0.5
if E('CENTRAL'):
    central = np.array([float(v) for v in E('CENTRAL').split(',')], np.float64); assert len(central) == NT, (len(central), NT)
    assert np.all(np.abs((central - cen)/hw) <= 1 + 1e-9), 'the central point must lie inside the box'
BC = cen.copy()                                                     # the box centre (xi at the centre of its range)
yp0, yp1, yps = py(central, False), py(central), py(central, which='scale:0')
ye0, (ye1, yes) = evaluate(central, CENW, [C0, VC[labels[0]]]); ya0, (ya1,) = evaluate(central, W_AT, [C0])
HEAD[0] = M64; yd0, (yd1,) = evaluate(central, W_AT, [C0])         # from here on the LIN_DTYPE head
res['checks'].update(evaluate_vs_predict=dict(before=relmax(ye0, yp0), after=relmax(ye1, yp1), scale0=relmax(yes, yps)),
                     wifi_basis_vs_central=dict(before=relmax(ya0, ye0), after=relmax(ya1, ye1)),
                     lin_dtype_vs_float32=dict(before=relmax(yd0, ya0), after=relmax(yd1, ya1)))
res['central'] = dict(theta=central.tolist(), box_centre=BC.tolist(), before=per(ye0), after=per(ye1))
print(f'[{tag}] checks: helper against three_bands.predict {res["checks"]["evaluate_vs_predict"]}, '
      f'wifi basis against the central network {res["checks"]["wifi_basis_vs_central"]}, '
      f'{LIN_DTYPE} head against float32 {res["checks"]["lin_dtype_vs_float32"]}', flush=True)
done('checks')

# ---- S0: the re-solved generator band (float32, as three_bands.py) ----------------------------------------
rng = np.random.default_rng(SEED)
U = (np.argsort(rng.random((NODES, NT)), axis=0) + rng.random((NODES, NT)))/NODES*2 - 1      # as three_bands.py
nodes = cen + hw*U
Y0, Y1 = [], []
for i, th in enumerate(nodes):
    Y0.append(py(th, False)); Y1.append(py(th))
    if i % 25 == 0:
        print(f'  S0 node {i+1}/{NODES} [{time.time()-t0:.0f} s]', flush=True)
Y0, Y1 = np.array(Y0), np.array(Y1); gen0, gen1 = sd(Y0), sd(Y1)
res['S0'] = dict(band=ba(gen0, gen1))
St = np.load(STORED + '.npz') if os.path.exists(STORED + '.npz') else None
Sj = json.load(open(STORED + '.json'))['central'] if os.path.exists(STORED + '.json') else None
if St is not None:
    fit = np.arange(NM + nlow, NO); chk = dict(nodes_max_abs_diff=float(np.abs(nodes - St['nodes']).max()))
    for side, Y, rows, H, glow in ((0, Y0, 'rows_before', 'H_before', 'gen_hist_before_low'), (1, Y1, 'rows_after', 'H_after', 'gen_hist_low')):
        s_new = np.r_[sd(Y)[:NM], sd(Y)[fit]]; s_old = np.r_[sd(St[rows]), sd(St[H])]
        if glow in St.files and nlow:
            s_new = np.r_[s_new, sd(Y)[NM:NM + nlow]]; s_old = np.r_[s_old, St[glow]]
        chk['before' if side == 0 else 'after'] = dict(sd_max_rel_diff=relmax(s_new, s_old),
                                                       values_max_rel_diff=relmax(np.c_[Y[:, :NM], Y[:, fit]], np.c_[St[rows], St[H]]))
    res['S0']['stored_check'] = chk
    print(f'[{tag}] S0 against {STORED}.npz: {chk}', flush=True)
done('S0')

# ---- S1, S2: the linear generator band; the linear joint covariance ---------------------------------------
VTH = hw**2/3                                                      # the variance of the flat prior


def linear(theta, wrt, fd):
    y, J, lw0 = jac(theta, W_AT, C0, wrt); Jt = J['theta']
    s = [np.sqrt((Jt[a:a + NO]**2) @ VTH) for a in (0, NO)]
    out = dict(theta=np.asarray(theta).tolist(), band=ba(*s), ratio_to_S0=ba(rat(s[0], gen0), rat(s[1], gen1)),
               jacobian_after=Jt[NO:].tolist(), jacobian_before=Jt[:NO].tolist())
    if fd:
        chk = {}
        for p in FD_PARAMS:
            for h in FD_STEPS:
                e = np.zeros(NT); e[p] = h*hw[p]
                (a0, (a1,)), (b0, (b1,)) = evaluate(theta + e, W_AT, [C0]), evaluate(theta - e, W_AT, [C0])
                f = np.r_[a0 - b0, a1 - b1]/(2*e[p]); dif = np.abs(f - Jt[:, p])
                band = np.r_[s[0], s[1]]
                chk[f'{p}_h{h:g}'] = dict(before_in_band=float(np.max(dif[:NO]*np.sqrt(VTH[p])/np.maximum(band[:NO], 1e-300))),
                                          after_in_band=float(np.max(dif[NO:]*np.sqrt(VTH[p])/np.maximum(band[NO:], 1e-300))),
                                          median_rel=float(np.median(dif/np.maximum(np.abs(Jt[:, p]), 1e-300))))
        out['fd_check'] = chk
        print(f'[{tag}] finite differences (error of the parameter\'s band contribution, in units of the band): '
              + '; '.join(f'{k} {v["before_in_band"]:.1e}/{v["after_in_band"]:.1e}' for k, v in chk.items()), flush=True)
    return out, y, J, lw0, s


def joint_linear(J):
    """Jesse's Gaussian joint covariance (before: theta and omega; after: theta, omega and c)."""
    Jt, Jw, Jq = J['theta'], J['omega'], J['c']
    return ((Jt[:NO]*VTH) @ Jt[:NO].T + Jw[:NO] @ C @ Jw[:NO].T,
            (Jt[NO:]*VTH) @ Jt[NO:].T + Jw[NO:] @ C @ Jw[NO:].T + Jq[NO:] @ SIGC @ Jq[NO:].T)


ALL = ('theta', 'omega', 'c')
res['S1'], ybc, J1, _, s1 = linear(BC, ALL, True)
print(f'[{tag}] S1 box centre: <1-T> {s1[1][2]:.5f} against S0 {gen1[2]:.5f}', flush=True)
done('S1')
res['S2'], yc, Jc, lw0c, s2 = linear(central, ALL, True)
yc0, yc1 = yc[:NO], yc[NO:]
print(f'[{tag}] S2 central: <1-T> {s2[1][2]:.5f} against S0 {gen1[2]:.5f}', flush=True)
# the learning band at the central point (three_bands.py), for the quadrature
Jw0, Jw1 = Jc['omega'][:NO], Jc['omega'][NO:]; learn0, learn1 = qv(Jw0, C), qv(Jw1, C)
res['learning'] = dict(band=ba(learn0, learn1))
if St is not None:                                                  # float32 there, LIN_DTYPE here: agree to ~1e-5
    lh = np.r_[St['learn_hist_low'], St['learn_hist']] if nlow and 'learn_hist_low' in St.files else St['learn_hist']
    res['checks']['learning_vs_stored'] = relmax(learn1[NO - len(lh):], lh)
if Sj is not None:
    res['checks']['learning_means_vs_stored'] = relmax(learn1[:NM], [Sj['learning_band'][k] for k in OBK])
LJ = {k: joint_linear(J) for k, J in (('box_centre', J1), ('central', Jc))}
done('S2')

# ---- S3X: the width scan with a subset shrunk (dims@point: all, params or xi, around box or central, or
# around central with xi fixed at 0 or 1: sherpa, herwig), the shrunk box cut to the full one, linear at its own
# centre against re-solved on the S0 hypercube mapped into it. Where xi varies over [a, b], also the two-generator
# formula: E[(1-xi)^2] J_S V J_S^T + E[xi^2] J_H V J_H^T + Var(xi) (y_H - y_S)^2, the generators linear at xi = 0
# and 1 (exact for the additive mixture before the tilt when each generator is linear in its parameters)
if E('S3X'):
    XI = NT - 1 if is_mix else -1; res['S3X'] = {}
    for mode in E('S3X').split(','):
        dims, at = mode.split('@'); P = (BC if at == 'box' else central).copy()
        if at in ('sherpa', 'herwig'):
            P[XI] = 0.0 if at == 'sherpa' else 1.0
        msk = dict(all=np.ones(NT, bool), params=np.arange(NT) != XI, xi=np.arange(NT) == XI)[dims]; out = res['S3X'][mode] = {}
        for s in WIDTHS:
            g = np.where(msk, s, 1.0)
            if at in ('sherpa', 'herwig'):
                g[XI] = 0.0
            lo, hi = np.maximum(P - g*hw, cen - hw), np.minimum(P + g*hw, cen + hw); m, h = (lo + hi)/2, (hi - lo)/2
            _, J, _ = jac(m, W_AT, C0, ALL); Jt = J['theta']; lin = [np.sqrt((Jt[a:a + NO]**2) @ (h**2/3)) for a in (0, NO)]
            Z = [evaluate(th, W_AT, [C0]) for th in m + h*U]; Z0, Z1 = np.array([z[0] for z in Z]), np.array([z[1][0] for z in Z])
            form = [None, None]
            if is_mix and h[XI] > 0:
                a, b = lo[XI], hi[XI]; ht = h.copy(); ht[XI] = 0.0; ends = []
                for e in (0.0, 1.0):
                    me = m.copy(); me[XI] = e; ye, Je, _ = jac(me, W_AT, C0, ALL); ends.append((ye, Je['theta']))
                Ex2, E1x2, Vx = (a*a + a*b + b*b)/3, ((1 - a)**2 + (1 - a)*(1 - b) + (1 - b)**2)/3, (b - a)**2/12
                form = [np.sqrt(E1x2*(ends[0][1][k:k + NO]**2) @ (ht**2/3) + Ex2*(ends[1][1][k:k + NO]**2) @ (ht**2/3)
                                + Vx*(ends[1][0][k:k + NO] - ends[0][0][k:k + NO])**2) for k in (0, NO)]
            out[f'{s:g}'] = dict(centre=m.tolist(), half_width=h.tolist(),
                                 **{side: dict(sd=per(sd(Zs)), linear=per(l), ratio_linear=per(rat(l, sd(Zs))),
                                               **({} if f is None else dict(formula=per(f), ratio_formula=per(rat(f, sd(Zs))))))
                                    for side, Zs, l, f in (('before', Z0, lin[0], form[0]), ('after', Z1, lin[1], form[1]))})
            if mode == 'all@box' and abs(s - 1) < 1e-12:
                res['checks']['S3X_s1_vs_S0_sd'] = dict(before=relmax(sd(Z0), gen0), after=relmax(sd(Z1), gen1))
            r = np.array(list(out[f'{s:g}']['after']['ratio_linear'].values()), float)
            rf = out[f'{s:g}']['before'].get('ratio_formula'); rf = None if rf is None else np.array(list(rf.values()), float)
            print(f'  S3X {mode} s={s:g}: linear/re-solved after {np.nanmin(r):.2f} to {np.nanmax(r):.2f}'
                  + ('' if rf is None else f', formula/re-solved before {np.nanmin(rf):.2f} to {np.nanmax(rf):.2f}') + f' [{time.time()-t0:.0f} s]', flush=True)
    done('S3X')
    if E('STOP_AFTER_S3X', '0') == '1':
        sys.exit(0)

# ---- the predicted sigma_lnw (D1) at the two linearisation points, before any re-solve at a node ----------------
ew, Vw = np.linalg.eigh(C); BW = Vw*np.sqrt(np.maximum(ew, 0))     # omega = w_at + BW z, c = C0 + D z (S5)
GB, IB, RB, XB = lnw_cov(BC, W_AT, C0, ALL, J1); GC, IC, RC, XC = lnw_cov(central, W_AT, C0, ('omega', 'c'), Jc)
del RC['J'], RC['J0']                                               # LL needs the box centre's alone
res['checks']['lnw_chain_rule_in_band'] = dict(box_centre=XB, central=XC)
print(f'[{tag}] chain-rule check of the log w derivatives (max error in units of the band): box centre {XB}, central {XC}', flush=True)
res['predicted_sigma_lnw'] = dict(box_centre={k: pred(GB, IB[k]) for k in ALL}, central={k: pred(GC, IC[k]) for k in IC}, fd_step=1e-3)
res['predicted_sigma_lnw']['box_centre']['joint'] = pred(GB, np.arange(len(GB[0])))
res['linearisation_points'] = {k: {q: R[q] for q in ('kish_before', 'kish_after', 'cond_A', 'cond_A_scaled')} for k, R in (('box_centre', RB), ('central', RC))}
print(f'[{tag}] predicted sigma_lnw at the box centre: ' + ', '.join(f'{k} {v["sigma_lnw"]:.3f}' for k, v in res['predicted_sigma_lnw']['box_centre'].items())
      + f'; cond(A) {RB["cond_A"]:.2e} (unit diagonal {RB["cond_A_scaled"]:.2e})', flush=True)
done('pred')

# ---- S3: the width scan (LIN_DTYPE, all widths re-solved; s = 1 is the S0 hypercube) -------------------------
res['S3'] = {}; FR = res['failure_rule'] = dict(widths={})
PP, PN = res['predicted_sigma_lnw']['box_centre']['theta'], pred(GB, IB['theta'], np.sqrt(3)*U)   # nodes in unit-variance units
res['predicted_sigma_lnw']['box_centre']['theta_nodes'] = PN
for s in WIDTHS:
    Z0, Z1, DR = [], [], []
    for th in BC + s*hw*U:
        a, b, d = evalw(th, W_AT, C0, RB); Z0.append(a); Z1.append(b); DR.append(d)
    Z0, Z1 = np.array(Z0), np.array(Z1); dm = dsum(DR, RB); fw = FR['widths'][f'{s:g}'] = {}
    if abs(s - 1) < 1e-12:
        Y0d, Y1d = Z0, Z1
        res['checks']['S3_s1_vs_S0_sd'] = dict(before=relmax(sd(Z0), gen0), after=relmax(sd(Z1), gen1))
    L = (s*hw*U) @ J1['theta'][:NO].T, (s*hw*U) @ J1['theta'][NO:].T
    r = {}
    for side, Z, Lz, sl, yb in (('before', Z0, L[0], s1[0], ybc[:NO]), ('after', Z1, L[1], s1[1], ybc[NO:])):
        R, Ra = r2(Z, U); Zc, Lc = Z - Z.mean(0), Lz - Lz.mean(0)
        rl, d4 = rat(s*sl, sd(Z)), rat(np.sqrt(((Z - yb - Lz)**2).mean(0)), s*sl)       # D4: (re-solved - tangent)/linear band
        r[side] = dict(sd=per(sd(Z)), linear=per(s*sl), ratio_linear=per(rl), ratio_tangent_same_nodes=per(rat(sd(Lz), sd(Z))),
                       r2=per(R), r2_adjusted=per(Ra), r2_tangent=per(1 - ((Zc - Lc)**2).sum(0)/(Zc**2).sum(0)), second_order=per(d4))
        k = 'sigma_lnw' if side == 'after' else 'sigma_lnw_before'; (wa, fo), (wm, fm) = fails(rl), fails(rl, slice(0, NM))
        (wt, ft), ns = fails(rat(sd(Lz), sd(Z))), fails(rat(s*sl, sd(Lz)))[0]   # ratio_linear = node sampling x tangent on the nodes
        fw[side] = dict(sigma_lnw=dm[k], sigma_lnw_predicted=s*PP[k], sigma_lnw_predicted_nodes=s*PN[k], worst_abs_ratio_minus_1=wa,
                        frac_outside_10pc=fo, worst_abs_ratio_minus_1_means=wm, frac_outside_10pc_means=fm,
                        worst_abs_ratio_minus_1_tangent=wt, frac_outside_10pc_tangent=ft, node_sampling_worst_abs=ns,
                        second_order_median=float(np.nanmedian(d4)), second_order_max=float(np.nanmax(d4)),
                        kish_min=dm[f'kish_{side}']['min'], kish_rel_min=dm[f'kish_rel_{side}']['min'],
                        cond_A_range=[dm['cond_A']['min'], dm['cond_A']['max']], cond_A_scaled_range=[dm['cond_A_scaled']['min'], dm['cond_A_scaled']['max']])
    r['diagnostics'] = dm; res['S3'][f'{s:g}'] = r; fa = fw['after']
    print(f'  S3 s={s:g}: <1-T> after re-solved {sd(Z1)[2]:.2e}, linear/re-solved {r["after"]["ratio_linear"]["thrust"]:.3f}, '
          f'R^2 {r["after"]["r2"]["thrust"]:.3f}; sigma_lnw {fa["sigma_lnw"]:.3f} (predicted {fa["sigma_lnw_predicted"]:.3f}), worst '
          f'|ratio-1| {fa["worst_abs_ratio_minus_1"]:.3f}, {fa["frac_outside_10pc"]:.2f} outside 10%, Kish min {fa["kish_min"]:.3f} '
          f'[{time.time()-t0:.0f} s]', flush=True)


def first_fail(side, key):
    """The smallest width at which linear is off by more than 10% (key over the predictions), with the width before it."""
    ws = sorted(WIDTHS); f = [s for s in ws if (FR['widths'][f'{s:g}'][side][key] or 0) > 0.1]
    if not f:
        return None
    pick = lambda s: None if s is None else dict(width=s, **{q: FR['widths'][f'{s:g}'][side][q] for q in ('sigma_lnw', 'sigma_lnw_predicted', key)})
    i = ws.index(f[0]); return dict(**pick(f[0]), last_pass=pick(ws[i - 1] if i else None))


FR['first_fail'] = {side: dict(any=first_fail(side, 'worst_abs_ratio_minus_1'), means=first_fail(side, 'worst_abs_ratio_minus_1_means'),
                               tangent=first_fail(side, 'worst_abs_ratio_minus_1_tangent'))
                    for side in ('before', 'after')}
print(f'[{tag}] linear first off by >10% (after): {FR["first_fail"]["after"]["any"]}', flush=True)
if 'Y0d' not in globals():                                         # WIDTHS without 1: S6 takes the float32 nodes
    Y0d, Y1d = Y0, Y1
done('S3')

# ---- S4: the theory band, linear against re-solved -------------------------------------------------------
Jcc = Jc['c'][NO:]
_, Jh, _ = jac(central, W_AT, C0, ('c',), fixed_h=True); Jch = Jh['c'][NO:]
DV, PV, dres = {}, {}, {}
with torch.no_grad():
    for l in VC:
        lw = tilt(lw0c, VC[l]); dres[l] = outv(lw).numpy() - yc1; DV[l] = dict(zip(DK, dstat(RC, lw0c, lw))); dv = VC[l] - C0
        PV[l] = wsd(RC['w'], (tilt(lw0c, C0 + 1e-3*dv) - tilt(lw0c, C0 - 1e-3*dv))/2e-3)   # D1 predicted along c_v - c
    dl = Dp[:, -1]; tt = 1e-2                                         # the directional derivative along the scheme
    fd = (outv(tilt(lw0c, C0 + tt*dl)).numpy() - outv(tilt(lw0c, C0 - tt*dl)).numpy())/(2*tt)
dlin = {l: Jcc @ (VC[l] - C0) for l in VC}; dlh = {l: Jch @ (VC[l] - C0) for l in VC}
th_res = np.sqrt(sum(wt*dres[l]**2 for l, wt in WT.items()))
th_gauss, th_gauss_h = qv(Jcc, SIGC), qv(Jch, SIGC)
inband = lambda a, b: float(np.max(np.abs(a - b)/np.maximum(th_res, 1e-300)))
res['S4'] = dict(band_resolved=per(th_res), band_gaussian=per(th_gauss), ratio_gaussian=per(rat(th_gauss, th_res)),
                 band_gaussian_fixed_h=per(th_gauss_h), ratio_gaussian_fixed_h=per(rat(th_gauss_h, th_res)),
                 variations={l: dict(resolved=per(dres[l]), linear=per(dlin[l]), ratio=per(rat(dlin[l], dres[l])),
                                     max_error_in_band=inband(dlin[l], dres[l]), linear_fixed_h=per(dlh[l]),
                                     max_error_in_band_fixed_h=inband(dlh[l], dres[l]),
                                     diagnostics={k: DV[l][k] for k in ('sigma_lnw', 'kish_after', 'kish_rel_after', 'cond_A', 'cond_A_scaled', 'dlam_rel')},
                                     sigma_lnw_predicted=PV[l]) for l in VC})
efr = lambda a, b: float(np.mean((np.abs(a - b)/th_res)[th_res > 0] > 0.1))       # fraction of predictions off by >10% of the band
FR['theory'] = dict(failure_measure='frac_error_above_10pc_of_band (the per-variation ratios divide by displacements that can be ~0)',
                    variations={l: dict(sigma_lnw=DV[l]['sigma_lnw'], sigma_lnw_predicted=PV[l], max_error_in_band=inband(dlin[l], dres[l]),
                                        frac_error_above_10pc_of_band=efr(dlin[l], dres[l]), worst_abs_ratio_minus_1=fails(rat(dlin[l], dres[l]))[0],
                                        frac_ratio_outside_10pc=fails(rat(dlin[l], dres[l]))[1], kish_rel_after=DV[l]['kish_rel_after'],
                                        cond_A=DV[l]['cond_A'], dlam_rel=DV[l]['dlam_rel']) for l in VC},
                    band=dict(sigma_lnw=float(np.sqrt(sum(wt*DV[l]['sigma_lnw']**2 for l, wt in WT.items()))),
                              sigma_lnw_predicted=float(np.sqrt(sum(wt*PV[l]**2 for l, wt in WT.items()))),
                              sigma_lnw_predicted_gaussian=res['predicted_sigma_lnw']['central']['c']['sigma_lnw'],
                              worst_abs_ratio_minus_1=fails(rat(th_gauss, th_res))[0], frac_outside_10pc=fails(rat(th_gauss, th_res))[1],
                              cond_A_at_central=RC['cond_A'], cond_A_range=[min(v['cond_A'] for v in DV.values()), max(v['cond_A'] for v in DV.values())]))
res['checks']['jc_fd_scheme_direction_in_band'] = inband(fd, Jcc @ dl)
if St is not None and not MC_NP:
    thh = np.r_[St['theory_hist_low'], St['theory_hist']] if nlow and 'theory_hist_low' in St.files else St['theory_hist']
    res['checks']['theory_vs_stored'] = relmax(th_res[NO - len(thh):], thh)
if Sj is not None and not MC_NP:
    res['checks']['theory_means_vs_stored'] = relmax(th_res[:NM], [Sj['theory_band'][k] for k in OBK])
print(f'[{tag}] S4: <1-T> theory re-solved {th_res[2]:.2e}, Gaussian {th_gauss[2]:.2e} (h fixed {th_gauss_h[2]:.2e}); worst '
      f'variation error in units of the band {max(v["max_error_in_band"] for v in res["S4"]["variations"].values()):.2f} '
      f'(h fixed {max(v["max_error_in_band_fixed_h"] for v in res["S4"]["variations"].values()):.2f})', flush=True)
q0, q1 = np.sqrt(gen0**2 + learn0**2), np.sqrt(gen1**2 + th_res**2 + learn1**2)      # the r8 quadrature, no statistical error
res['linear_joint'] = {k: dict(band=ba(dg(A0), dg(A1)), correlation_after=corr(A1).tolist(), correlation_before=corr(A0).tolist())
                       for k, (A0, A1) in LJ.items()}
res['linear_joint']['central']['ratio_to_r8'] = ba(rat(dg(LJ['central'][0]), q0), rat(dg(LJ['central'][1]), q1))
done('S4')

# ---- S5: the joint Monte Carlo ---------------------------------------------------------------------------
rng5 = np.random.default_rng(SEED + 5)
Ut = rng5.uniform(-1, 1, (MC_N, NT)); Zw = rng5.standard_normal((MC_N, len(W_AT))); Zc = rng5.standard_normal((MC_N, D.shape[1]))
if MC_CRN:                                                          # theta at the S0 nodes, omega and c at their exact covariances
    assert MC_N % NODES == 0 and MC_N > NT + len(W_AT) + D.shape[1] + 1, 'MC_CRN needs MC_N a multiple of NODES, above the dimensions'
    white = lambda Z: (Z - Z.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(Z - Z.mean(0), rowvar=False))).T
    Ut = U[np.arange(MC_N) % NODES]; Uc = Ut - Ut.mean(0); Z = np.c_[Zw, Zc]
    Z = white(Z - Uc @ np.linalg.lstsq(Uc, Z, rcond=None)[0]); Zw, Zc = Z[:, :len(W_AT)], Z[:, len(W_AT):]
OM, CC = W_AT + Zw @ BW.T, C0 + Zc @ D.T                            # the same calls of rng5 as before; MC_CRN replaces the draws
Q = {k: ([], []) for k in ('joint', 'theta', 'omega', 'c')}; DS = {k: [] for k in Q}
for i in range(MC_N):
    for k, (th, om, c, R) in (('joint', (cen + hw*Ut[i], OM[i], CC[i], RB)), ('theta', (cen + hw*Ut[i], W_AT, C0, RB)),
                              ('omega', (central, OM[i], C0, RC))):
        a, b, d = evalw(th, om, c, R); Q[k][0].append(a); Q[k][1].append(b); DS[k].append(d)
    with torch.no_grad():
        lw = tilt(lw0c, CC[i]); Q['c'][0].append(yc0); Q['c'][1].append(outv(lw).numpy()); DS['c'].append(dstat(RC, lw0c, lw))
    if i % 50 == 0:
        print(f'  S5 draw {i+1}/{MC_N} [{time.time()-t0:.0f} s]', flush=True)
Q = {k: (np.array(a), np.array(b)) for k, (a, b) in Q.items()}
ybc0, (ybc1,) = evaluate(BC, W_AT, [C0])
# bootstrap: the draws, and the S0 nodes behind the generator part of the quadrature
rb = np.random.default_rng(SEED + 6); bi = rb.integers(0, MC_N, (NBOOT, MC_N)); bn = np.random.default_rng(SEED + 7).integers(0, NODES, (NBOOT, NODES))
if MC_CRN:
    bn = bi % NODES                                                 # the nodes of the resampled draws
res['S5'] = {}
vr = lambda A: A.var(0, ddof=1)
lt = (hw*Ut) @ J1['theta'].T + (OM - W_AT) @ J1['omega'].T          # the tangent at the box centre on the draws
LD = (lt[:, :NO], (lt + (CC - C0) @ J1['c'].T)[:, NO:])


def xsplit(j, ix=slice(None)):
    """The cross-term fraction split exactly on the shared theta draws, Dl = joint - theta (var_J = var_T + var_Dl + 2 cov):
    A = (var Dl - var_omega - var_c)/var_J, the change of the learning and theory variances across the box (against S6),
    B = 2 cov(T, Dl)/var_J, the mean shift of omega and c with theta (the curvature in omega, c) and the interaction."""
    Jj, T = Q['joint'][j][ix], Q['theta'][j][ix]; Dl = Jj - T; vJ = vr(Jj)
    cv = ((T - T.mean(0))*(Dl - Dl.mean(0))).sum(0)/(len(T) - 1)
    return rat(vr(Dl) - vr(Q['omega'][j][ix]) - vr(Q['c'][j][ix]), vJ), rat(2*cv, vJ)


for j, side, q, Yg, rest in ((0, 'before', q0, Y0, learn0**2), (1, 'after', q1, Y1, th_res**2 + learn1**2)):
    if MC_CRN:                                                      # the generator SD over the nodes as the draws repeat them (ddof)
        q = np.sqrt(sd(Yg[np.arange(MC_N) % NODES])**2 + rest)
    S = {k: sd(Q[k][j]) for k in Q}; qmc = np.sqrt(S['theta']**2 + S['omega']**2 + S['c']**2); smc = S['theta'] + S['omega'] + S['c']
    bq = np.array([[sd(Q['joint'][j][ix]), np.sqrt(sum(sd(Q[k][j][ix])**2 for k in ('theta', 'omega', 'c'))),
                    np.sqrt(sd(Yg[ixn])**2 + rest)] for ix, ixn in zip(bi, bn)])
    (xa, xb), bx = xsplit(j), np.array([xsplit(j, ix) for ix in bi])                 # bx: (NBOOT, 2, NO)
    xse = np.nanstd(bx, 0, ddof=1)
    cr = np.nan_to_num(np.corrcoef(Q['joint'][j], rowvar=False)); A = LJ['box_centre'][j]
    res['S5'][side] = dict(sd={k: per(v) for k, v in S.items()}, quadrature_one_source=per(qmc), sum_one_source=per(smc), quadrature_r8=per(q),
                           ratio_to_r8=per(rat(S['joint'], q)), ratio_to_r8_se=per((bq[:, 0]/bq[:, 2]).std(0, ddof=1)),
                           ratio_to_one_source=per(rat(S['joint'], qmc)), ratio_to_one_source_se=per((bq[:, 0]/bq[:, 1]).std(0, ddof=1)),
                           ratio_to_sum_one_source=per(rat(S['joint'], smc)),
                           cross_fraction=per(rat(S['joint']**2 - qmc**2, S['joint']**2)),
                           cross_fraction_se=per(np.nanstd(bx.sum(1), 0, ddof=1)), cross_A_bands=per(xa), cross_A_bands_se=per(xse[0]),
                           cross_B_mean_shift=per(xb), cross_B_mean_shift_se=per(xse[1]),
                           ratio_to_linear_joint=per(rat(S['joint'], dg(A))), correlation_max_abs_diff_linear_joint=float(np.abs(cr - corr(A)).max()),
                           linear_same_draws=per(sd(LD[j])), ratio_to_linear_same_draws=per(rat(S['joint'], sd(LD[j]))),
                           mean_shift_vs_central_in_sd=per(rat(Q['joint'][j].mean(0) - (yc1 if j else yc0), S['joint'])),
                           mean_shift_vs_box_centre_in_sd=per(rat(Q['joint'][j].mean(0) - (ybc1 if j else ybc0), S['joint'])),
                           mean_shift_vs_theta_only_in_sd=per(rat(Q['joint'][j].mean(0) - Q['theta'][j].mean(0), S['joint'])),
                           correlation=cr.tolist())
ZJ = dict(joint=(GB, np.arange(len(GB[0])), np.c_[np.sqrt(3)*Ut, Zw, Zc], RB), theta=(GB, IB['theta'], np.sqrt(3)*Ut, RB),
          omega=(GC, IC['omega'], Zw, RC), c=(GC, IC['c'], Zc, RC))
res['S5']['diagnostics'] = {k: dict(measured=dsum(DS[k], R), predicted=pred(G, sel), predicted_draws=pred(G, sel, Z)) for k, (G, sel, Z, R) in ZJ.items()}
s5j0, s5j1 = sd(Q['joint'][0]), sd(Q['joint'][1])
print(f'[{tag}] S5: <1-T> joint {s5j1[2]:.2e}, r8 quadrature {q1[2]:.2e}, linear joint {dg(LJ["box_centre"][1])[2]:.2e}, '
      f'one-source quadrature {res["S5"]["after"]["quadrature_one_source"]["thrust"]:.2e}', flush=True)
done('S5')

# ---- LL: log-linear propagation (log w linear in the inputs, the predictions exact in w) ---------------------
nl = lambda z: z - torch.logsumexp(z, 0)
loglin = lambda z: (outv(nl(RB['lw0'] + RB['J0'] @ z)).numpy(), outv(nl(RB['lw'] + RB['J'] @ z)).numpy())
ZS = torch.as_tensor(np.c_[np.sqrt(3)*Ut, Zw, Zc])                  # the S5 draws in the unit-variance directions of lnw_cov
LL = {k: ([], []) for k in ('joint', 'theta')}; LLr = []
with torch.no_grad():
    for i in range(MC_N):
        zt = ZS[i].clone(); zt[NT:] = 0
        for k, z in (('joint', ZS[i]), ('theta', zt)):
            a, b = loglin(z); LL[k][0].append(a); LL[k][1].append(b)
        LLr.append(outv(tilt(nl(RB['lw0'] + RB['J0'] @ ZS[i]), CC[i])).numpy())
    LW = {}
    for s in WIDTHS:                                                # the S3 nodes, theta alone
        a, b = zip(*[loglin(torch.as_tensor(np.r_[s*np.sqrt(3)*u, np.zeros(len(ZS[0]) - NT)])) for u in U]); LW[s] = (sd(a), sd(b))
SL = {k: (sd(a), sd(b)) for k, (a, b) in LL.items()}
arr27 = lambda r: np.array([r[l] for l in LAB], float)
res['LL'] = dict(band=ba(*SL['joint']), band_resolved=per(sd(LLr)),
                 ratio_joint=ba(rat(SL['joint'][0], sd(Q['joint'][0])), rat(SL['joint'][1], sd(Q['joint'][1]))),
                 ratio_joint_resolved=per(rat(sd(LLr), sd(Q['joint'][1]))),
                 ratio_theta=ba(rat(SL['theta'][0], sd(Q['theta'][0])), rat(SL['theta'][1], sd(Q['theta'][1]))),
                 widths={f'{s:g}': {side: dict(ratio=per(r), worst_abs_ratio_minus_1=fails(r)[0], frac_outside_10pc=fails(r)[1],
                                               worst_abs_ratio_minus_1_means=fails(r, slice(0, NM))[0])
                                    for side, r in (('before', rat(LW[s][0], arr27(res['S3'][f'{s:g}']['before']['sd']))),
                                                    ('after', rat(LW[s][1], arr27(res['S3'][f'{s:g}']['after']['sd']))))}
                         for s in WIDTHS})
for k in ('ratio_joint', 'ratio_theta'):
    res['LL'][k + '_worst'] = {side: dict(zip(('worst_abs_ratio_minus_1', 'frac_outside_10pc'), fails(arr27(res['LL'][k][side]))))
                               for side in ('before', 'after')}
res['LL']['ratio_joint_resolved_worst'] = dict(zip(('worst_abs_ratio_minus_1', 'frac_outside_10pc'), fails(arr27(res['LL']['ratio_joint_resolved']))))
print(f'[{tag}] LL: log-linear/re-solved joint, worst |ratio-1| before {res["LL"]["ratio_joint_worst"]["before"]["worst_abs_ratio_minus_1"]:.3f}, '
      f'after {res["LL"]["ratio_joint_worst"]["after"]["worst_abs_ratio_minus_1"]:.3f} (multipliers re-solved '
      f'{res["LL"]["ratio_joint_resolved_worst"]["worst_abs_ratio_minus_1"]:.3f}); per width (after): '
      + ', '.join(f'{w} {v["after"]["worst_abs_ratio_minus_1"]:.3f}' for w, v in res['LL']['widths'].items()), flush=True)
done('LL')

# ---- S6: the hybrid -------------------------------------------------------------------------------------
EL0 = np.zeros((NO, NO)); EL1 = np.zeros((NO, NO)); ETL = np.zeros((NO, NO)); ETR = np.zeros((NO, NO)); nh = min(HYB_N, NODES)
BN = {k: [] for k in ('learn0', 'learn1', 'theory_linear', 'theory_resolved')}          # the bands at each node
for i in range(nh):
    y, J, lw0 = jac(nodes[i], W_AT, C0, ('omega', 'c'))
    a, b, jc = J['omega'][:NO], J['omega'][NO:], J['c'][NO:]
    la, lb, lt, tr = a @ C @ a.T, b @ C @ b.T, jc @ SIGC @ jc.T, np.zeros((NO, NO))
    with torch.no_grad():
        for l, wt in WT.items():
            d = outv(tilt(lw0, VC[l])).numpy() - y[NO:]; tr += wt*np.outer(d, d)
    EL0 += la/nh; EL1 += lb/nh; ETL += lt/nh; ETR += tr/nh
    for k, A in zip(BN, (la, lb, lt, tr)):
        BN[k].append(dg(A))
    if i % 10 == 0:
        print(f'  S6 node {i+1}/{nh} [{time.time()-t0:.0f} s]', flush=True)
CV0, CV1 = np.cov(Y0d, rowvar=False), np.cov(Y1d, rowvar=False)
T0, T1, T1r = CV0 + EL0, CV1 + EL1 + ETL, CV1 + EL1 + ETR
res['S6'] = dict(nodes=nh, band_linear_theory=ba(dg(T0), dg(T1)), band_resolved_theory=per(dg(T1r)),
                 ratio_to_r8=ba(rat(dg(T0), q0), rat(dg(T1), q1)), ratio_resolved_theory_to_r8=per(rat(dg(T1r), q1)),
                 ratio_to_S5=ba(rat(dg(T0), s5j0), rat(dg(T1), s5j1)),
                 learning_mean_over_central=ba(rat(dg(EL0), learn0), rat(dg(EL1), learn1)),
                 theory_mean_over_central=per(rat(dg(ETR), th_res)), theory_linear_over_resolved=per(rat(dg(ETL), dg(ETR))),
                 correlation=corr(T1r).tolist())
# the quadrature: the cross terms of S5 against the variation of the learning and theory bands across the box. With
# b(theta) = E[y|theta] - y(theta, w_at, c), var_joint - sum var_i = (E_theta[learning + theory] - their central values)
# + 2 Cov_theta(y, b) + Var_theta(b): S6 predicts the first term, which is part A of S5 to leading order; the second,
# of the same order (VTH C, VTH Sigma_c), is part B, measured on the shared theta draws and not predicted here
BN = {k: np.array(v) for k, v in BN.items()}; arr = lambda d: np.array([d[l] for l in LAB], float)
cvb = lambda B: per(rat(B.std(0, ddof=1), B.mean(0)))
xp = dict(before=(rat(np.diag(EL0) - learn0**2, s5j0**2), None),
          after=(rat(np.diag(EL1) + np.diag(ETL) - learn1**2 - th_gauss**2, s5j1**2), rat(np.diag(EL1) + np.diag(ETR) - learn1**2 - th_res**2, s5j1**2)))
FR['quadrature'] = dict(sigma_lnw={k: dict(measured=v['measured']['sigma_lnw'], predicted=v['predicted']['sigma_lnw'])
                                   for k, v in res['S5']['diagnostics'].items()})


def cor(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None


for side, (xl, xr) in xp.items():
    r5 = res['S5'][side]; xm, xa, xb = arr(r5['cross_fraction']), arr(r5['cross_A_bands']), arr(r5['cross_B_mean_shift'])
    sa, sb = arr(r5['cross_A_bands_se']), arr(r5['cross_B_mean_shift_se'])
    FR['quadrature'][side] = dict(cross_fraction_measured=per(xm), cross_fraction_se=r5['cross_fraction_se'],
                                  cross_A_bands=per(xa), cross_A_bands_se=per(sa), cross_B_mean_shift=per(xb), cross_B_mean_shift_se=per(sb),
                                  cross_fraction_predicted=per(xl), cross_fraction_predicted_resolved_theory=None if xr is None else per(xr),
                                  median_abs_measured=float(np.nanmedian(np.abs(xm))), median_abs_predicted=float(np.nanmedian(np.abs(xl))),
                                  median_abs_A=float(np.nanmedian(np.abs(xa))), median_abs_B=float(np.nanmedian(np.abs(xb))),
                                  median_se_total=float(np.nanmedian(arr(r5['cross_fraction_se']))), median_se_A=float(np.nanmedian(sa)),
                                  median_se_B=float(np.nanmedian(sb)), frac_B_above_2se=float(np.mean((np.abs(xb) > 2*sb)[np.isfinite(xb)])),
                                  correlation_A_predicted=cor(xa, xl), correlation_A_predicted_resolved_theory=None if xr is None else cor(xa, xr),
                                  correlation_measured_predicted=cor(xm, xl), worst_abs_ratio_to_r8_minus_1=fails(arr(r5['ratio_to_r8']))[0],
                                  frac_ratio_to_r8_outside_10pc=fails(arr(r5['ratio_to_r8']))[1],
                                  learning_band_cv=cvb(BN['learn0' if side == 'before' else 'learn1']),
                                  theory_band_cv=None if side == 'before' else cvb(BN['theory_resolved']))
fq = FR['quadrature']['after']
print(f'[{tag}] quadrature: <1-T> cross fraction measured {fq["cross_fraction_measured"]["thrust"]:.3f} = A {fq["cross_A_bands"]["thrust"]:.3f}'
      f'+-{fq["cross_A_bands_se"]["thrust"]:.3f} + B {fq["cross_B_mean_shift"]["thrust"]:.3f}+-{fq["cross_B_mean_shift_se"]["thrust"]:.3f}, '
      f'A predicted from S6 {fq["cross_fraction_predicted"]["thrust"]:.3f}; median |A| {fq["median_abs_A"]:.3f} against predicted '
      f'{fq["median_abs_predicted"]:.3f}, correlation {fq["correlation_A_predicted"]}; median |B| {fq["median_abs_B"]:.3f}, '
      f'{fq["frac_B_above_2se"]:.2f} of the predictions with |B| > 2 SE; median SE total/A/B {fq["median_se_total"]:.3f}/'
      f'{fq["median_se_A"]:.3f}/{fq["median_se_B"]:.3f}', flush=True)
done('S6')

# ---- the table --------------------------------------------------------------------------------------------
lj1 = dg(LJ['box_centre'][1])
print(f'\n[{tag}] after the calculation is imposed (ratios are band/S0 for S1-S3, Gaussian/re-solved for S4, joint/quadrature for S5, '
      f'linear joint at the box centre/S5 joint for LJ)')
print(f'{"":20s} {"S0 gen":>9s} {"S1/S0":>6s} {"S2/S0":>6s} {"S3 s=" + f"{WIDTHS[0]:g}":>8s} {"theory":>9s} {"S4 G/r":>6s} '
      f'{"learn":>9s} {"S5 joint":>9s} {"quad":>9s} {"S5/q":>6s} {"cross":>6s} {"LJ/S5":>6s} {"S6/q":>6s}')
r3 = res['S3'][f'{WIDTHS[0]:g}']['after']['ratio_linear']; r5 = res['S5']['after']
for k, l in enumerate(LAB):
    print(f'{l:20s} {gen1[k]:9.2e} {s1[1][k]/gen1[k]:6.2f} {s2[1][k]/gen1[k]:6.2f} {r3[l]:8.2f} {th_res[k]:9.2e} {th_gauss[k]/th_res[k]:6.2f} '
          f'{learn1[k]:9.2e} {s5j1[k]:9.2e} {q1[k]:9.2e} {s5j1[k]/q1[k]:6.2f} {r5["cross_fraction"][l]:6.2f} {lj1[k]/s5j1[k]:6.2f} '
          f'{dg(T1r)[k]/q1[k]:6.2f}')
print(f'generators alone: <1-T> S0 {gen0[2]:.2e}, S1/S0 {s1[0][2]/gen0[2]:.2f}, S5 joint/quadrature {s5j0[2]/q0[2]:.2f}; '
      f'Newton {NEWTON}; timings {res["timings"]}')
print(f'\n[{tag}] failure_rule (after; before in parentheses): sigma_lnw measured / predicted, worst |linear/re-solved - 1|, '
      f'(tangent on the same nodes), fraction outside 10%, min Kish N_eff/N absolute/relative, cond(A) range')
for sw, fw in FR['widths'].items():
    a_, b_ = fw['after'], fw['before']
    print(f'  s={sw:5s} {a_["sigma_lnw"]:.3f}/{a_["sigma_lnw_predicted"]:.3f} ({b_["sigma_lnw"]:.3f}/{b_["sigma_lnw_predicted"]:.3f})  '
          f'{a_["worst_abs_ratio_minus_1"]:.3f} ({b_["worst_abs_ratio_minus_1"]:.3f}) [{a_["worst_abs_ratio_minus_1_tangent"]:.3f}]  '
          f'{a_["frac_outside_10pc"]:.2f} ({b_["frac_outside_10pc"]:.2f}) [{a_["frac_outside_10pc_tangent"]:.2f}]  '
          f'{a_["kish_min"]:.3f}/{a_["kish_rel_min"]:.3f} ({b_["kish_min"]:.3f}/{b_["kish_rel_min"]:.3f})  {a_["cond_A_range"][0]:.2e}-{a_["cond_A_range"][1]:.2e}')
print(f'  first fail: {FR["first_fail"]}\n  theory band: {FR["theory"]["band"]}\n  theory variations, fraction off by >10% of the band: '
      + ', '.join(f'{l} {v["frac_error_above_10pc_of_band"]:.2f}' for l, v in FR['theory']['variations'].items())
      + f'\n  quadrature sigma_lnw: {FR["quadrature"]["sigma_lnw"]}\n  quadrature cross (after): median |A| {fq["median_abs_A"]:.3f} '
      f'(predicted {fq["median_abs_predicted"]:.3f}, correlation {fq["correlation_A_predicted"]}), median |B| {fq["median_abs_B"]:.3f}')
print(f'wrote {OUT}.json [{time.time()-t0:.0f} s]')
