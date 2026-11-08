#!/usr/bin/env python3
"""The envelope of NDRAW parameter variations against the linear covariance, for the main section of the
paper (prior only: generators and learning) and for the application (with the calculation imposed in the window).

MIX sets how the two generators combine:
  linear      (default) the prediction is (1 - xi) <O>_S(theta_S) + xi <O>_H(theta_H): each generator from its own
              weights, the head at xi = 0 (Sherpa) and at xi = 1 (Herwig), the calculation imposed on each one
              alone, the two predictions mixed with (1 - xi, xi). The thrust histogram is normalized to the total
              rate, so every prediction is linear in xi. NDRAW defaults to 2000 and the outputs end in _lin.
  geometric   the weight of the geometric head at the drawn xi, the thrust histogram normalized over the fitted
              bins: the results of the earlier draft, unchanged.

Linear. BOX is tune or full, alone or with _sherpa or _herwig (directlib.lin_box):
  full          each generator's parameters over the trained box, xi over [0, 1], slopes at the box centre
  tune          each generator range TUNE_SCALE of its width, centred on the default tunes (CENTRAL) and cut
                symmetrically where the box edge is nearer, so the default stays the centre; xi over [0, 1]
  <box>_sherpa, <box>_herwig   one generator alone (xi = 0 or 1) over its tune-sized or full ranges
Per generator g, at its parameters at the centre (xi = 0 or 1), the slopes J_g = dy_g/d(theta_g, omega, c) through
the re-solved multipliers (saved as J_theta_S, J_omega_S, J_c_S, ... with y_central_S, y_central_H), and
  lin_flat    Var(xi) D D^T + sum_g E[xi_g^2] J_g diag((hi - lo)^2/12) J_g^T, D = y_H - y_S at the centre
  lin_z       the same with every flat variance replaced by ((hi - lo) phi(0))^2, slopes in z (= 6/pi lin_flat)
  learn       sum_gh E[xi_g xi_h] J_g^omega C J_h^omega^T, the shared learning weights
  theory      the same with Sigma_c for the shared calculation (after only)
  learn_mid, theory_mid   the same with xi at the centre, for the draws that vary the learning weights or the
              calculation alone and keep theta and xi at the centre
with E[xi_g xi_h] = 1/3 on the diagonal and 1/6 off it for xi over [0, 1] (directlib.mix_cov). The draws are
exact: both generators reweighted (and re-solved after) at each draw and mixed. No quadrature in xi.

Geometric. Every parameter of the box, the mixing fraction included, is flat on its range, written as a standard
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
below the window first, normalized over the fitted bins, or to the total rate when linear), the charged-
multiplicity histogram (bins of two, fractions summing to one) and the heavy-jet-mass histogram (density over
[0, 0.4) as a fraction of all events), before and after. When linear, the window mean tau_win is each
generator's mean of tau inside the window, mixed with (1 - xi, xi) like every other prediction.

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
(draws, covariances, central values, bin edges, the test data, and mix). Linear: the expansion point of the formula,
the centre of the prior (the defaults for tune, D6; the box centre for full, D10), is recorded as expansion_point.
The learning weights are expanded at the published network of the export's temperature (directlib.wifi_point).
DRAW_LIMIT (tests) evaluates only the first DRAW_LIMIT draws of the NDRAW of each set.

    BOX=tune python envelope_test.py MIXGEOV3S9 output/models/MIXGEOV3S9_cond.npz \\
        output/models/MIXGEOV3S9_ref_v2_slim.npz output/thrust_targets_nlo_point.npz output/wifi_MIXGEOV3S9_fit.npz
"""
import os, sys, json, time, copy, resource
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, tilt_on_t, at_fraction, xi_moments, mix_cov, lin_box, wifi_point

tag, export, ref, grid, wfit = sys.argv[1:6]
E = os.environ.get
MIX = E('MIX', 'linear'); assert MIX in ('linear', 'geometric'), MIX
LINEAR = MIX == 'linear'; SFX = '_lin' if LINEAR else ''
BOX = E('BOX', 'full'); NDRAW = int(E('NDRAW', '2000' if LINEAR else '400')); NREF = int(E('NREF', '0')); SEED = int(E('SEED', '11'))
TUNE = float(E('TUNE_SCALE', str(1/3))); XI_NODES = int(E('XI_NODES', '8')); NBOOT = int(E('NBOOT', '300'))
DRAW_LIMIT = int(E('DRAW_LIMIT', '0')); NEVAL = min(NDRAW, DRAW_LIMIT) if DRAW_LIMIT else NDRAW
TEST = E('TEST_DATA', 'delphi'); OUT = E('OUT', f'output/envelope_test_{tag}_{BOX}{SFX}'); RES_TOL = float(E('RES_TOL', '1e-8'))
MC_NP = E('MC_NP', '0') == '1'
CENTRAL = E('CENTRAL')
torch.set_num_threads(int(E('TORCH_THREADS', '8'))); np.seterr(divide='ignore', invalid='ignore')
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
H = copy.copy(M); H.AE = M.AE.to(torch.float64); H.wifi = None    # float64 head for the derivatives, M (float32) for the draws
G = M.G; assert len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'a single-point target file is expected'
W = np.load(wfit); C = np.asarray(W['C'], np.float64); W_AT, W_AT_NOTE = wifi_point(W, M)
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
    """The predictions as one differentiable vector: the means, then the thrust, multiplicity and rho histograms.
    lw is normalized over all events; the thrust histogram is normalized over the fitted bins (geometric) or to
    the total rate (linear)."""
    w = torch.exp(lw); o = M.observables(w)
    cnt = torch.zeros(x['nb'], dtype=w.dtype).index_add(0, x['idx'], w[x['ins']])
    nh = torch.zeros(NNB, dtype=w.dtype).index_add(0, NIDX, w)
    rh = torch.zeros(NRB, dtype=w.dtype).index_add(0, RIDX, w[RIN])/RWID
    th = cnt[EXT]/WEXT if LINEAR else cnt[EXT]/(WEXT*cnt[x['use']].sum())
    return torch.cat([torch.stack([o[k] for k in OBK]), th, nh, rh])


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


GENS = (('S', 0.0), ('H', 1.0))                                    # the generators and their xi in the head


def evaluate_lin(theta, om, c, lw0s=None):
    """The linear mixture at theta = (theta_S, theta_H, xi): ({g: y_g}, (1 - xi) y_S + xi y_H), each y_g from
    evaluate() at xi = 0 or 1, so with its own tilt; a generator with no weight is skipped. lw0s = {g: log w0_g}
    skips the head."""
    xi = float(theta[XI]); ys = {}
    for g, xg in GENS:
        if (1 - xi if g == 'S' else xi) != 0:
            ys[g] = evaluate(at_fraction(theta, xg), om, c, None if lw0s is None else lw0s[g])
    return ys, sum((1 - xi if g == 'S' else xi)*y for g, y in ys.items())


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
if LINEAR:                                                          # D6: tune-sized ranges centred on the defaults
    lo, hi = lin_box(cen, hw, central, BOX, TUNE, NS)
    XE, XVAR = xi_moments(lo[XI], hi[XI])                          # E[xi_g xi_h] and Var(xi) of the flat prior
    A0 = dict(S=1 - (lo[XI] + hi[XI])/2, H=(lo[XI] + hi[XI])/2)    # (1 - xi, xi) at the centre
    XE_MID = np.outer([A0['S'], A0['H']], [A0['S'], A0['H']])
    ACTIVE = [g for i, (g, _) in enumerate(GENS) if XE[i, i] > 0]  # the generators that enter
    OWN = dict(S=np.arange(NS), H=np.arange(NS, XI))               # each generator's own parameters in theta
else:
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
print(f'[{tag}] {BOX} ({MIX}): {M.N} events, d={NT}, {len(W_AT)} learning weights, {NO} predictions; box centre {np.round(m, 6).tolist()}, '
      f'widths {np.round(R, 6).tolist()} [{time.time()-t0:.0f} s]', flush=True)
res = dict(tag=tag, mix=MIX, box=BOX, tune_scale=TUNE, ndraw=NEVAL, nref=NREF, seed=SEED, xi_nodes=0 if LINEAR else XI_NODES, labels=LAB,
           groups={k: v.tolist() for k, v in GROUPS.items()}, lo=lo.tolist(), hi=hi.tolist(), centre=m.tolist(),
           nch_edges=[float(e) if np.isfinite(e) else None for e in NCH_EDGES], rho_edges=RHO_EDGES.tolist(),
           thrust_edges=[float(x['lo'][b]) for b in ext] + [float(x['hi'][ext[-1]])], newton=NEWTON, timings={})
tick = [time.time()]
if NEVAL < NDRAW:
    res['draws_generated'] = NDRAW                                 # tests: only the first NEVAL of them evaluated
if LINEAR:
    res['expansion_point'] = m.tolist(); res['learning_point'] = W_AT_NOTE


def done(stage):
    res['timings'][stage] = round(time.time() - tick[0], 1); tick[0] = time.time()
    res['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)
    json.dump(clean(res), open(OUT + '.json', 'w'), indent=1, allow_nan=False)
    print(f'[{tag}] {BOX} {stage} done in {res["timings"][stage]:.0f} s, Newton {NEWTON["solves"]} solves, '
          f'{NEWTON["unconverged"]} above {RES_TOL:g} [{time.time()-t0:.0f} s]', flush=True)


# ---- linear at the centre, and xi by quadrature ---------------------------------------------------------
VFLAT = R**2/12; VZ = (R*PHI0)**2
if LINEAR:
    # each generator at the centre of its ranges: its predictions and their slopes in its own parameters, the
    # learning weights and the calculation, then the covariance of the mixture (directlib.mix_cov)
    YG0, JG = {}, {}
    for g, xg in GENS:
        if g in ACTIVE:
            YG0[g], JG[g] = jac(at_fraction(m, xg), W_AT, C0)
            print(f'  slopes of {g} at the centre [{time.time()-t0:.0f} s]', flush=True)
    yc = sum(A0[g]*YG0[g] for g in ACTIVE)
    own = lambda V: [(JG[g]['theta'][:, OWN[g]] if g in JG else None, V[OWN[g]]) for g, _ in GENS]
    shared = lambda k, Cov: [(JG['S'][k] if 'S' in JG else None, JG['H'][k] if 'H' in JG else None, Cov)]
    delta = YG0['H'] - YG0['S'] if len(ACTIVE) == 2 else None
    LIN = dict(lin_flat=mix_cov(XE, XVAR, delta, own(VFLAT)), lin_z=mix_cov(XE, (R[XI]*PHI0)**2, delta, own(VZ)),
               learn=mix_cov(XE, 0, shared=shared('omega', C)), theory=mix_cov(XE, 0, shared=shared('c', SIGC)),
               learn_mid=mix_cov(XE_MID, 0, shared=shared('omega', C)), theory_mid=mix_cov(XE_MID, 0, shared=shared('c', SIGC)))
    res['xi'] = dict(E=XE.tolist(), var=XVAR, E_mid=XE_MID.tolist(), generators=ACTIVE)
    res['central_generators'] = {g: dict(before=per(YG0[g][:NO]), after=per(YG0[g][NO:])) for g in ACTIVE}
else:
    yc, J = jac(m, W_AT, C0); Jt, Jw, Jc = J['theta'], J['omega'], J['c']
    LIN = dict(lin_z=(Jt*VZ) @ Jt.T, lin_flat=(Jt*VFLAT) @ Jt.T, learn=Jw @ C @ Jw.T, theory=Jc @ SIGC @ Jc.T)
if not LINEAR and R[XI] > 0 and XI_NODES > 0:
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
with torch.no_grad():                                               # the c-only set: the head at the centre, once
    LW_M = ({g: prior(torch.as_tensor(at_fraction(m, xg)), W_AT, M) for g, xg in GENS if g in ACTIVE} if LINEAR
            else prior(torch.as_tensor(m), W_AT, M))
Y, YG = {}, {}
for name, vary in SETS.items():
    rows, rows_g = [], []
    for i in range(NEVAL):
        th = DR['TH'][i] if 'TH' in vary else m; om = DR['OM'][i] if 'OM' in vary else W_AT; cc = DR['CS'][i] if 'CS' in vary else C0
        if LINEAR:
            ys, y = evaluate_lin(th, om, cc, LW_M if name == 'c' else None); rows.append(y); rows_g.append(ys)
        else:
            rows.append(evaluate(th, om, cc, LW_M if name == 'c' else None))
        if i % 100 == 0:
            print(f'  {name} draw {i+1}/{NEVAL} [{time.time()-t0:.0f} s]', flush=True)
    Y[name] = np.array(rows)
    if LINEAR:
        YG[name] = {g: np.array([r[g] for r in rows_g]) for g in ACTIVE}
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
            if LINEAR and setname in ('omega', 'c'):
                kl, kc = kl + '_mid', kc + '_mid'                  # these draws keep theta and xi at the centre
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
# linear: each generator's central prediction and slopes (rows: the predictions before, then after; columns: its own
# parameters, the learning weights, the calculated moments), E[xi_g xi_h], the learning weights and calculation of
# every draw, and each generator's predictions at the joint draws (float32)
extra = ({**{f'y_central_{g}': YG0[g] for g in ACTIVE}, **{f'J_theta_{g}': JG[g]['theta'][:, OWN[g]] for g in ACTIVE},
          **{f'J_omega_{g}': JG[g]['omega'] for g in ACTIVE}, **{f'J_c_{g}': JG[g]['c'] for g in ACTIVE},
          **{f'Y_joint_{g}': YG['joint'][g].astype(np.float32) for g in ACTIVE},
          'xi_E': XE, 'xi_var': np.float64(XVAR), 'xi_E_mid': XE_MID, 'OM': DR['OM'], 'CS': DR['CS'], 'w_at': W_AT, 'c0': C0}
         if LINEAR else {})
save_npz = lambda: np.savez_compressed(
    OUT + '.npz', labels=np.array(LAB), lo=lo, hi=hi, centre=m, y_central=yc, **{f'Y_{k}': v for k, v in Y.items()},
    **{f'C_{k}': v for k, v in LIN.items()}, U=DR['U'], Z=DR['Z'], TH=DR['TH'],
    thrust_edges=np.array(res['thrust_edges']), nch_edges=NCH_EDGES, rho_edges=RHO_EDGES,
    data_lo=x['lo'], data_hi=x['hi'], data_y=x['y'].numpy(), data_e=x['e'].numpy(), data_use=use, ext=ext,
    **({'data_y_raw': x['y_raw'], 'data_e_raw': x['e_raw'], 'expansion_point': m} if LINEAR else {}), mix=np.array(MIX), **extra)
save_npz()
if NREF:                                                            # the sampling error of NDRAW draws, against NREF more
    DRR = draws(NREF, 'ref')
    Y['ref'] = np.array([evaluate_lin(DRR['TH'][i], DRR['OM'][i], DRR['CS'][i])[1] if LINEAR else
                         evaluate(DRR['TH'][i], DRR['OM'][i], DRR['CS'][i]) for i in range(NREF)])
    for side in ('before', 'after'):
        res['compare'][f'{side}/draws{NEVAL}_vs_ref'] = compare(np.cov(Y['joint'], rowvar=False), Y['ref'], side)
    done('reference'); save_npz()
print(f'ENVELOPE TEST {tag} {BOX} DONE [{time.time()-t0:.0f} s]', flush=True)
