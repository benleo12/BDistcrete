#!/usr/bin/env python3
"""Three uncertainties on one reweighted sample, with the calculation imposed at a fixed point.

For a model (export, reference bundle), its wifi weights (wifi_fit.py) and the calculated moments
of one point (targets_nlo_point.py):
  central          the mean of the members is the log ratio everywhere (the prediction of the
                   paper); WIFI_CENTRAL=1 instead uses the fitted basis weights w_0 + sum_ij w_ij
                   <a_i, b_j>, which the paper uses for the learning band only
  generator band   the spread of every prediction over a flat prior on the parameter box, NODES
                   points of a Latin hypercube (the mixing fraction uniform on [0, 1]), before and
                   after the calculation is imposed
  learning band    at the central point, the propagated covariance of the wifi weights:
                   sigma^2(<O>) = g^T C g with g = d<O>/dw through the re-solved multipliers
  theory band      at the central point, the multipliers solved again for every variation of the
                   calculation, signed as Eq. (thband): each scale pair once per side with the larger of
                   its two displacements, the scheme once, the dispersive-model variations as their root
                   mean square
  statistical      at the central point, the statistical error of every prediction from the finite
                   reference sample, which no band contains
  combined         the three bands and the statistical error added in quadrature; theory_up/down and
                   combined_up/down keep the sign of each theory variation, with the
                   dispersive-model variations in the theory band as their root mean square
Observables: the mean of tau inside the window and the window fraction, the full means of 1-T,
B_tot, rho_H, the mean charged multiplicity and the mean baryon count, and the thrust distribution
in the bins of TEST_DATA (DELPHI by default; the all-particle table d11-x01-y02, fit_alpha0.LEP). The
bins below FIRST_BIN = 0.05, outside the window, are evaluated too and written under keys ending in _low.

MIX sets how the generators combine. linear (the default): every prediction is (1 - xi) <O>_S + xi <O>_H,
each generator from its own weights (the head at xi = 0 and at xi = 1) with the calculation imposed on each
one alone (directlib, the linear mixture), every distribution the fraction of all events in each bin over its
width (no renormalization over a range of bins), the data as published (DELPHI integrates to 0.990), the window
mean each generator's own, mixed; the central point has xi = 1/2, the mean of the flat prior on [0, 1]. Then
  generator band   from NODES (default 2000) draws over BOX (full, the trained box, or tune, every range
                   TUNE_SCALE of its width centred on the default tunes, directlib.lin_box), each with xi, and
                   from the formula Var(xi) D D^T + sum_g E[xi_g^2] J_g diag(V_g) J_g^T, D = y_H - y_S, J_g the
                   slopes of generator g in its own parameters, V_g = (hi - lo)^2/12 (generator_band_formula),
                   expanded at the centre of the prior: the defaults for BOX=tune, the box centre for BOX=full
                   (D10; the json records the point); combined_* use the draws, combined_*_formula the formula
  learning band    sum_gh E[xi_g xi_h] g_g^T C g_h with each generator's slopes g_g in the wifi weights
  theory band      D9: Eq. (thband) applied at each xi to the displacement (1 - xi) d_S + xi d_H of the mixed
                   prediction, sigma_+^2 and sigma_-^2 averaged over xi flat on [0, 1] (directlib.thband_mix,
                   exact); theory_band (symmetric, without the dispersive model) is the same average of its
                   quadratic rule. Each generator alone keeps Eq. (thband) as it stands (central/generators), and
                   every displacement of each generator is stored
  statistical      of the mixed prediction at xi = 1/2, from the per-event terms of both generators together
with E[xi_g xi_h] = 1/3 on the diagonal and 1/6 off it.
Comparisons with the measurements (D8): every prediction is also given at the hadron level of the measurement it
is compared with, each generator's prediction multiplied by its own factors after imposing and before mixing
(conventions.py: c/a in the DELPHI bins, cnu/a on <1-T> for OPAL, no factor on <n_ch> for L3), with every band
of the corrected prediction (keys *_conv; central/conv in the json); the uncorrected values keep their keys.
CONVENTIONS=none leaves this out. The comparison normalized over the window, the twelve bins wholly inside
[0.05, 1/3), is stored apart (keys *_win, with the generator band from the draws), next to the comparison as
published. The outputs end in _lin. MIX=geometric is the earlier draft's head at xi = 1/2, unchanged.

    python three_bands.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz \\
        output/thrust_targets_nlo_point.npz output/wifi_MIXGEO_fit.npz
Writes output/three_bands_<tag>.json and .npz (with _lin when linear). NODE_LIMIT (tests) evaluates only the
first NODE_LIMIT points of the hypercube.
"""
import os, sys, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, at_fraction, xi_moments, mix_cov, lin_box, thband, thband_mix, stat_parts, wifi_point

tag, export, ref, grid, wfit = sys.argv[1:6]
MIX = os.environ.get('MIX', 'linear'); assert MIX in ('linear', 'geometric'), MIX
LINEAR = MIX == 'linear'
NODES = int(os.environ.get('NODES', '2000' if LINEAR else '200')); SEED = int(os.environ.get('SEED', '11'))
NODE_LIMIT = int(os.environ.get('NODE_LIMIT', '0'))
BOX = os.environ.get('BOX', 'full'); TUNE = float(os.environ.get('TUNE_SCALE', str(1/3)))
assert not LINEAR or BOX in ('full', 'tune'), f'BOX={BOX}: three_bands.py takes full or tune (xi flat on [0, 1] in both)'
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('OUT', f'output/three_bands_{tag}' + ('_lin' if LINEAR else ''))
CONV = os.environ.get('CONVENTIONS', 'factors'); assert CONV in ('factors', 'none'), CONV
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
t0 = time.time()
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
G = M.G; assert len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'a single-point target file is expected'
if os.environ.get('NO_WIFI') == '1':
    W, w_wifi, C = None, None, None
else:
    assert os.path.exists(wfit), f'wifi fit file {wfit} not found (set NO_WIFI=1 to skip the learning band)'
    W = np.load(wfit); w_wifi = W['w']; C = W['C']
# the central log ratio: the wifi weights unless told otherwise
CENW = w_wifi if (w_wifi is not None and os.environ.get('WIFI_CENTRAL', '0') == '1') else None
NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy()
is_mix = M.kind == 'mixture'
print(f'[{tag}] {M.N} reference events, d={NT}, head {M.kind} ({M.mix_form}), generators mixed {MIX}, wifi weights '
      f'{len(w_wifi) if w_wifi is not None else 0} ({"the central estimate" if CENW is not None else "learning band only"}), '
      f'calculation at alpha_s={float(G["alphas"][0]):.4f} alpha_0={float(G["alpha0"][0]):.4f} [{time.time()-t0:.0f} s]', flush=True)
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; edges = np.r_[x['lo'], x['hi'][-1]]; use = x['use'].numpy(); wid = x['wid'].numpy()
# every histogram runs over the bins below the fitted ones as well (ext), still normalized over the
# fitted bins (use), so the fitted bins come out exactly as before; nlow of them lie below. Linear: normalized
# to the total rate instead (the weights sum to one over all events)
low = np.where(x['lo'] < x['lo'][use].min() - 1e-12)[0]; ext = np.r_[low, use]; nlow = len(low)
_DEFAULT = object()
GENS = (('S', 0.0), ('H', 1.0))                                    # the generators and their xi in the head
GNAME = dict(S='sherpa', H='herwig')
assert not LINEAR or is_mix, 'the linear mixture needs a mixture head'


def predict1(theta, which='central', anchored=True, wifi=_DEFAULT):
    """Observable means and the test histogram (normalized over the fitted bins, or to the total rate when
    linear) at theta, with the head as it is (geometric: at the xi of theta)."""
    M.set_wifi(CENW if wifi is _DEFAULT else wifi)
    with torch.no_grad():
        lw, lw0, lam, Pw, its = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, which=which, anchored=anchored)
        w = torch.exp(lw); o = M.observables(w); o = {k: float(o[k]) for k in OBK if k in o}
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
        h = cnt[ext]/wid[ext] if LINEAR else cnt[ext]/(wid[ext]*cnt[use].sum())
    M.set_wifi(None)
    return o, h, float(torch.exp(torch.logsumexp(lw0[M.IW], 0))) if anchored else None


def predict(theta, which='central', anchored=True, wifi=_DEFAULT, parts=None):
    """predict1, or when linear the mixture (1 - xi) S + xi H of the predictions of the two generators at xi = 0
    and 1, each with its own tilt; parts, a dict, receives each generator's (o, h, window fraction before)."""
    if not LINEAR:
        return predict1(theta, which, anchored, wifi)
    xi = float(theta[-1]); acc = None
    for g, xg in GENS:
        a = 1 - xi if g == 'S' else xi
        if a == 0:
            continue
        o, h, p = predict1(at_fraction(theta, xg), which, anchored, wifi)
        if parts is not None:
            parts[g] = (o, h, p)
        acc = ({k: a*v for k, v in o.items()}, a*h, None if p is None else a*p) if acc is None else \
              ({k: acc[0][k] + a*o[k] for k in o}, acc[1] + a*h, None if p is None else acc[2] + a*p)
    return acc


def jac_wifi(theta, w, anchored=True):
    """d<O>/dw and dh/dw at theta for the wifi weight vector w, through the re-solved multipliers."""
    wt = torch.tensor(np.asarray(w, np.float64), requires_grad=True); M.set_wifi(wt)
    lw, _, _, _, _ = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, anchored=anchored)
    wv = torch.exp(lw); o = M.observables(wv)
    cnt = torch.zeros(x['nb']).index_add(0, x['idx'], wv[x['ins']]); h = cnt[ext]/(torch.as_tensor(wid[ext])*cnt[use].sum())
    J = {}
    for k in OBK:
        if k in o:
            J[k] = torch.autograd.grad(o[k], wt, retain_graph=True)[0].numpy()
    Jh = np.stack([torch.autograd.grad(h[b], wt, retain_graph=True)[0].numpy() for b in range(len(h))])
    M.set_wifi(None)
    return J, Jh


HF = []                                                             # the float64 head of the slopes, made once


def jac_lin(theta, w, anchored=True):
    """Linear: (y, dy/dtheta, dy/dw) at theta of one generator (xi = 0 or 1) for y = (the means of OBK, the
    histogram normalized to the total rate), with the float64 head and the wifi weights w (None: the published
    network, no dy/dw), through the re-solved multipliers."""
    if not HF:
        HF.append(copy.copy(M)); HF[0].AE = M.AE.to(torch.float64); HF[0].wifi = None
    hd = HF[0]
    th = torch.tensor(np.asarray(theta, np.float64), requires_grad=True)
    wt = None if w is None else torch.tensor(np.asarray(w, np.float64), requires_grad=True)
    hd.set_wifi(wt)
    try:
        lw, _, _, _, _ = hd.weights(th, 0, 0, anchored=anchored)
    finally:
        hd.set_wifi(None)
    wv = torch.exp(lw); o = hd.observables(wv)
    cnt = torch.zeros(x['nb'], dtype=wv.dtype).index_add(0, x['idx'], wv[x['ins']])
    y = torch.cat([torch.stack([o[k] for k in OBK]), cnt[torch.as_tensor(ext)]/torch.as_tensor(wid[ext])])
    leaves = (th,) if wt is None else (th, wt)
    rows = [torch.autograd.grad(y[k], leaves, retain_graph=True) for k in range(len(y))]
    return (y.detach().numpy(), np.stack([r[0].numpy() for r in rows]),
            None if wt is None else np.stack([r[1].numpy() for r in rows]))


def quad(*a):
    return float(np.sqrt(sum(float(v)**2 for v in a)))


PER_EVENT = {'thrust': M.thr, 'nch': M.nch, **{k: M.extra[k] for k in ('B_total', 'rho_heavy', 'nbaryon')}}
INWIN = np.zeros(M.N, bool); INWIN[M.IW.numpy()] = True


def stat_errors(theta, anchored=True):
    """The statistical error of the means and of the test histogram from the finite reference sample, at
    fixed classifier weights. Each prediction is written as a weighted mean <O> of a centred per-event
    function O (for a ratio such as the window mean or a bin of the distribution normalized over the
    fitted bins, its linearized form), and var = sum_i w_i^2 r_i^2 with weights normalized to one. With
    the calculation imposed the multipliers are solved again on every resampled sample, so r_i is O_i
    minus its projection on the constraint functions, r = O - (F - <F>) J^-1 Cov(F, O) with
    J = Cov(F, F) + P_win^2 Sigma, the matrix of the Newton step of tilt_on (the response of P_win itself is
    neglected). Without it r = O."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        lw, lw0, lam, Pw, _ = M.weights(th, 0, 0, anchored=anchored)
        w = torch.exp(lw).numpy()
    w = w/w.sum(); w2 = w**2
    if anchored:
        cvec = np.asarray(M.target(0, 0, 'central'), np.float64)
        F = np.zeros((M.N, len(cvec))); F[INWIN] = M.MW.numpy() - cvec[None, :]
        Fc = F - w @ F
        J = (Fc*w[:, None]).T @ Fc + float(Pw)**2*np.asarray(M.sigma(0, 0, cvec), np.float64)
        Jinv = np.linalg.inv(J)
    def err_of(O):                       # O centred under w
        r = O - Fc @ (Jinv @ (Fc.T @ (w*O))) if anchored else O
        return float(np.sqrt(w2 @ r**2))
    o = {k: float(w @ v) for k, v in PER_EVENT.items()}
    err = {k: err_of(v - o[k]) for k, v in PER_EVENT.items()}
    pwin = float(w[INWIN].sum()); twin = float(w[INWIN] @ M.thr[INWIN])/pwin
    err['pwin'] = err_of(INWIN - pwin)
    err['tau_win'] = err_of(np.where(INWIN, M.thr - twin, 0.0)/pwin)
    ins = x['ins'].numpy(); idx = x['idx'].numpy(); ub = np.asarray(use)          # use is an index array of bins
    binof = np.full(M.N, -1); binof[ins] = idx
    inuse = np.isin(binof, ub); Wt = float(w[inuse].sum()); hb = []
    for bb in ext:
        ind = (binof == bb).astype(float); pb = float(w @ ind)/Wt
        hb.append(err_of((ind - pb*inuse)/Wt)/wid[bb])
    return err, np.array(hb)


def stat_lin(theta, anchored, vsets):
    """Linear: the statistical error of every prediction y (the means of OBK, then the histogram over ext) of the
    mixture at theta, for each factor set of vsets ({S, H: factor vector over y}, or None for no factors). The same
    reference events carry both generators, so the error of sum_g a_g f_g <O>_g is sqrt(sum_i (sum_g a_g f_g u_g,i)^2)
    with u_g the per-event term of directlib.stat_parts, every per-event function centred on its own generator's
    prediction (a bin: the fraction of all events in it over its width)."""
    xi = float(theta[-1])
    parts = {g: (1 - xi if g == 'S' else xi, *stat_parts(M, at_fraction(theta, xg), anchored)) for g, xg in GENS
             if (1 - xi if g == 'S' else xi) != 0}
    binof = np.full(M.N, -1); binof[x['ins'].numpy()] = x['idx'].numpy()

    def per_event(j, w):
        if j >= len(OBK):
            ind = (binof == ext[j - len(OBK)]).astype(float)
            return (ind - w @ ind)/wid[ext[j - len(OBK)]]
        k = OBK[j]
        if k == 'tau_win':
            return np.where(INWIN, M.thr - (w[INWIN] @ M.thr[INWIN])/w[INWIN].sum(), 0.0)/w[INWIN].sum()
        if k == 'pwin':
            return INWIN - w[INWIN].sum()
        return PER_EVENT[k] - w @ PER_EVENT[k]
    out = [np.empty(len(OBK) + len(ext)) for _ in vsets]
    for j in range(len(out[0])):
        us = {g: a*u(per_event(j, w)) for g, (a, w, u) in parts.items()}
        for s, V in enumerate(vsets):
            out[s][j] = float(np.sqrt(np.sum(sum((1.0 if V is None else V[g][j])*us[g] for g in us)**2)))
    return out


# ---- the central point and its theory and learning bands --------------------------------------
central = cen.copy()
if is_mix:
    central[-1] = 0.5
if os.environ.get('CENTRAL'):
    # the central point as a comma-separated list of physical parameter values (for example the
    # default tunes of both generators and a fraction of one half)
    central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64); assert len(central) == NT, (len(central), NT)
    assert np.all(np.abs((central - cen)/hw) <= 1 + 1e-9), 'the central point must lie inside the box'
P0, P1 = {}, {}                                                     # linear: each generator's central predictions
o_c0, h_c0, pw0 = predict(central, anchored=False, parts=P0)
o_c, h_c, pw = predict(central, parts=P1)
o_p0, h_p0, _ = predict(central, anchored=False, wifi=None)        # the published network, as a baseline
VSETS = {'': None}                                                 # linear: the factor sets of the comparisons (D8)
CONV_REC = None
if LINEAR and CONV == 'factors':
    import conventions
    assert TEST == 'delphi', 'the factors of D8 are given for DELPHI (TEST_DATA=delphi); set CONVENTIONS=none otherwise'
    FAC = conventions.load()
    VSETS['_conv'], CONV_REC = FAC.vectors(OBK, x['lo'][ext], x['hi'][ext])
    print(f'[{tag}] hadron-level conventions (D8) from {FAC.path}: DELPHI {conventions.DELPHI_KIND}, <1-T> '
          f'{conventions.MEAN_KIND["thrust"]} [{time.time()-t0:.0f} s]', flush=True)
if LINEAR:
    SL = {s: v for s, v in zip(VSETS, stat_lin(central, True, list(VSETS.values())))}
    SL0 = {s: v for s, v in zip(VSETS, stat_lin(central, False, list(VSETS.values())))}
    NM = len(OBK)
    stat, stat_hist = {k: float(SL[''][j]) for j, k in enumerate(OBK)}, SL[''][NM:]
    stat0, stat_hist0 = {k: float(SL0[''][j]) for j, k in enumerate(OBK)}, SL0[''][NM:]
else:
    stat, stat_hist = stat_errors(central); stat0, stat_hist0 = stat_errors(central, anchored=False)
o_p, h_p, _ = predict(central, wifi=None)
labels = [str(l) for l in G['scale_labels']]
PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"
dv = {}; dh = {}; dvg = {g: {} for g in P1}; dhg = {g: {} for g in P1}   # linear: also per generator


def _vary(which, lab, store_o, store_h):
    """The displacement of the prediction at the central point for one variation of the calculation (linear: mixed
    at xi = 1/2, and each generator's own in dvg, dhg)."""
    pp = {}; ov, hv, _ = predict(central, which=which, parts=pp)
    store_o[lab] = {k: ov[k] - o_c[k] for k in o_c}; store_h(lab, hv - h_c)
    for g, (og, hg, _) in pp.items():
        dvg[g][lab] = {k: og[k] - P1[g][0][k] for k in o_c}; dhg[g][lab] = hg - P1[g][1]


for v, lab in enumerate(labels):
    _vary(f'scale:{v}', lab, dv, dh.__setitem__)
np_band = {}; np_hist = []
for v, lab in enumerate([str(l) for l in G['np_labels']]):
    _vary(f'np:{v}', lab, np_band, lambda lab, h: np_hist.append(h))
np_hist = np.array(np_hist)
NPL = [str(l) for l in G['np_labels']]
npi = {lab: np_hist[i] for i, lab in enumerate(NPL)}
if LINEAR:
    # E[xi_g xi_h] of the flat prior on [0, 1]
    XE, XVAR = xi_moments(0.0, 1.0)
    vec = lambda o, h: np.r_[[o[k] for k in OBK], h]               # a prediction as one vector: the means, then the bins
    DV = {g: {lab: vec(dvg[g][lab], dhg[g][lab]) for lab in dvg[g]} for g in ('S', 'H')}   # every displacement, per generator

    def theory(V):
        """The theory bands of y = (means, bins) for the factor set V (None: no factors): the symmetric band (the
        scale pairs and the scheme, a quadratic rule, so its average over xi is the E-weighted sum), the signed
        bands of D9 (directlib.thband_mix) and Eq. (thband) for each generator alone."""
        d = {g: {lab: (1.0 if V is None else V[g])*v for lab, v in DV[g].items()} for g in ('S', 'H')}
        q = lambda lab: XE[0, 0]*d['S'][lab]**2 + 2*XE[0, 1]*d['S'][lab]*d['H'][lab] + XE[1, 1]*d['H'][lab]**2
        sym = np.sqrt(sum(0.5*q(a) + 0.5*q(b) for a, b in PAIRS) + q(SCHEME))
        up, dn = (thband_mix(d['S'], d['H'], PAIRS, SCHEME, NPL, s) for s in (True, False))
        one = {g: (thband(d[g], PAIRS, SCHEME, NPL, True), thband(d[g], PAIRS, SCHEME, NPL, False)) for g in ('S', 'H')}
        return sym, up, dn, one
    TH = {s: theory(V) for s, V in VSETS.items()}
    sym, up, dn, _ = TH['']
    th_band = {k: float(sym[j]) for j, k in enumerate(OBK)}; th_hist = sym[NM:]
    th_up = {k: float(up[j]) for j, k in enumerate(OBK)}; th_dn = {k: float(dn[j]) for j, k in enumerate(OBK)}
    th_hist_up, th_hist_dn = up[NM:], dn[NM:]
else:
    th_band = {k: float(np.sqrt(sum(0.5*dv[a][k]**2 + 0.5*dv[b][k]**2 for a, b in PAIRS) + dv[SCHEME][k]**2)) for k in o_c}
    th_hist = np.sqrt(sum(0.5*dh[a]**2 + 0.5*dh[b]**2 for a, b in PAIRS) + dh[SCHEME]**2)
# signed theory bands (Eq. thband): each variation moves the prediction one way, so the band above the
# prediction collects the upward displacements and the band below the downward ones. Each scale pair counts
# once on each side, with the larger of its two displacements there, so a pair that moves the prediction by
# +-delta gives delta on both sides, as one standard deviation of Sigma_c does; the change of scheme counts
# once, and the three variations of the dispersive model enter as their root mean square. th_band and th_hist above are
# the symmetric band without the dispersive model, kept for reference.


def _signed(get, up, sq=lambda v: v**2):
    """sq(m) of every term's one-sided magnitude m (the geometric draft)."""
    f = (lambda v: np.maximum(v, 0.0)) if up else (lambda v: np.maximum(-v, 0.0))
    tot = sum(sq(np.maximum(f(get(a)), f(get(b)))) for a, b in PAIRS) + sq(f(get(SCHEME)))
    return np.sqrt(tot + sum(sq(f(get(l))) for l in NPL)/len(NPL))


if not LINEAR:
    th_up = {k: float(_signed(lambda lab: np.asarray(dv[lab][k] if lab in dv else np_band[lab][k]), True)) for k in o_c}
    th_dn = {k: float(_signed(lambda lab: np.asarray(dv[lab][k] if lab in dv else np_band[lab][k]), False)) for k in o_c}
    th_hist_up = _signed(lambda lab: dh[lab] if lab in dh else npi[lab], True)
    th_hist_dn = _signed(lambda lab: dh[lab] if lab in dh else npi[lab], False)
print(f'[{tag}] central: generator only <1-T> {o_c0["thrust"]:.5f}, imposed {o_c["thrust"]:.5f}, theory band {th_band["thrust"]:.5f}; '
      f'published network {o_p0["thrust"]:.5f} and {o_p["thrust"]:.5f} [{time.time()-t0:.0f} s]', flush=True)
# learning band: the covariance of the wifi weights at the central point, with and without the
# calculation (NO_WIFI=1 skips it, for heads without one)
WPT_NOTE = None
if LINEAR:
    # each generator at its central parameters (xi = 0, 1): the predictions y_g and their slopes in theta and in
    # the wifi weights, with and without the calculation, for the learning band
    if w_wifi is None:
        w_at = None
    else:
        w_at, WPT_NOTE = (w_wifi, 'the fitted wifi weights') if CENW is not None else wifi_point(W, M)
    JL = {(g, anc): jac_lin(at_fraction(central, xg), w_at, anc) for g, xg in GENS for anc in (False, True)}
    split = lambda v: ({k: float(np.sqrt(max(v[j], 0))) for j, k in enumerate(OBK)}, np.sqrt(np.maximum(v[NM:], 0)))
    if w_at is not None:
        learn_gen = {nm: {s: split(np.einsum('ri,ij,rj->r', JL[g, anc][2], C, JL[g, anc][2]))[0] for s, anc in
                          (('generator_only', False), ('imposed', True))} for g, nm in (('S', 'sherpa'), ('H', 'herwig'))}
    # the generator band from the formula, expanded at the centre of the prior over BOX (D10): the defaults for
    # BOX=tune, where the tune-sized ranges are centred (D6), and the centre of the box for BOX=full
    blo, bhi = lin_box(cen, hw, central, BOX, TUNE, M.NS); VG = (bhi - blo)**2/12
    XPT = (blo + bhi)/2
    XPT_AT = 'box centre' if BOX == 'full' else 'defaults (the centre of the tune-sized ranges)'
    if np.allclose(XPT, central, rtol=0, atol=1e-12):              # the slopes of the learning band serve
        JX = {k: v[:2] for k, v in JL.items()}
    else:
        JX = {(g, anc): jac_lin(at_fraction(XPT, xg), None, anc)[:2] for g, xg in GENS for anc in (False, True)}
    OWN = dict(S=np.arange(M.NS), H=np.arange(M.NS, NT - 1))
    sc = lambda V, g: 1.0 if V is None else V[g]
    LC = {s: {anc: (None if w_at is None else mix_cov(XE, 0, shared=[(sc(V, 'S')[:, None]*JL['S', anc][2] if V is not None else JL['S', anc][2],
                                                                      sc(V, 'H')[:, None]*JL['H', anc][2] if V is not None else JL['H', anc][2], C)]))
              for anc in (False, True)} for s, V in VSETS.items()}
    CGS = {s: {anc: mix_cov(XE, XVAR, sc(V, 'H')*JX['H', anc][0] - sc(V, 'S')*JX['S', anc][0],
                            own=[((JX[g, anc][1] if V is None else V[g][:, None]*JX[g, anc][1])[:, OWN[g]], VG[OWN[g]]) for g in ('S', 'H')])
               for anc in (False, True)} for s, V in VSETS.items()}
    nanv = np.full(NM + len(ext), np.nan)
    if w_at is None:
        learn_band = learn_band0 = {k: float('nan') for k in o_c}; learn_hist = learn_hist0 = np.full(len(h_c), np.nan)
    else:
        (learn_band0, learn_hist0), (learn_band, learn_hist) = split(np.diag(LC[''][False])), split(np.diag(LC[''][True]))
    CL = LC['']; CG = CGS['']
    (genf0, genf0_hist), (genf1, genf1_hist) = split(np.diag(CG[False])), split(np.diag(CG[True]))
    print(f'[{tag}] learning band: <1-T> {learn_band["thrust"]:.2e} (generator only {learn_band0["thrust"]:.2e}), '
          f'n_ch {learn_band["nch"]:.3f}; generator band from the formula (at the {XPT_AT}) on <1-T> {genf0["thrust"]:.5f} before, '
          f'{genf1["thrust"]:.5f} after [{time.time()-t0:.0f} s]', flush=True)
elif w_wifi is None:
    learn_band = learn_band0 = {k: float('nan') for k in o_c}; learn_hist = learn_hist0 = np.full(len(h_c), np.nan)
else:
    # the derivative is taken at the weights that define the central prediction: the wifi weights
    # when they are the central estimate, otherwise the mean of members written in the same basis
    w_at = w_wifi if CENW is not None else wifi_point(W, M)[0]
    J, Jh = jac_wifi(central, w_at)
    learn_band = {k: float(np.sqrt(max(J[k] @ C @ J[k], 0))) for k in J}
    learn_hist = np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', Jh, C, Jh), 0))
    J0, Jh0 = jac_wifi(central, w_at, anchored=False)
    learn_band0 = {k: float(np.sqrt(max(J0[k] @ C @ J0[k], 0))) for k in J0}
    learn_hist0 = np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', Jh0, C, Jh0), 0))
    print(f'[{tag}] learning band: <1-T> {learn_band["thrust"]:.2e} (generator only {learn_band0["thrust"]:.2e}), '
          f'n_ch {learn_band["nch"]:.3f} [{time.time()-t0:.0f} s]', flush=True)

# ---- the generator band: a flat prior over the box ---------------------------------------------
rng = np.random.default_rng(SEED)
U = (np.argsort(rng.random((NODES, NT)), axis=0) + rng.random((NODES, NT)))/NODES*2 - 1      # Latin hypercube in [-1, 1]
nodes = (blo + bhi)/2 + (bhi - blo)/2*U if LINEAR and BOX != 'full' else cen + hw*U    # xi uniform on [0, 1]
if NODE_LIMIT:
    nodes = nodes[:NODE_LIMIT]                                     # tests: the first points of the same hypercube
rows0, rows1, H0, H1 = [], [], [], []
YG = {(g, s): [] for g in ('S', 'H') for s in (0, 1)}             # linear: each generator's predictions at the points
for i, th in enumerate(nodes):
    if LINEAR:
        p0, p1 = {}, {}
        o0, h0, _ = predict(th, anchored=False, parts=p0); o1, h1, _ = predict(th, parts=p1)
        for g in ('S', 'H'):
            for s, pp in ((0, p0), (1, p1)):
                YG[g, s].append(vec(pp[g][0], pp[g][1]) if g in pp else np.zeros(NM + len(ext)))
    else:
        o0, h0, _ = predict(th, anchored=False); o1, h1, _ = predict(th)
    rows0.append([o0[k] for k in OBK]); rows1.append([o1[k] for k in OBK]); H0.append(h0); H1.append(h1)
    if i % 25 == 0:
        print(f'  node {i+1}/{len(nodes)} [{time.time()-t0:.0f} s]', flush=True)
rows0, rows1, H0, H1 = map(np.array, (rows0, rows1, H0, H1))
gen = {'before': {k: dict(mean=float(rows0[:, j].mean()), sd=float(rows0[:, j].std(ddof=1)),
                           p16=float(np.percentile(rows0[:, j], 16)), p84=float(np.percentile(rows0[:, j], 84))) for j, k in enumerate(OBK)},
       'after': {k: dict(mean=float(rows1[:, j].mean()), sd=float(rows1[:, j].std(ddof=1)),
                          p16=float(np.percentile(rows1[:, j], 16)), p84=float(np.percentile(rows1[:, j], 84))) for j, k in enumerate(OBK)}}
g0_hist = H0.std(0, ddof=1); g1_hist = H1.std(0, ddof=1)
# the combination: generator, theory and learning in quadrature (generator only: generator and learning)
# every combination includes the statistical error of the prediction at the central point
comb = {k: quad(gen['after'][k]['sd'], th_band[k], learn_band[k] if np.isfinite(learn_band[k]) else 0, stat[k]) for k in OBK}
comb0 = {k: quad(gen['before'][k]['sd'], learn_band0[k] if np.isfinite(learn_band0[k]) else 0, stat0[k]) for k in OBK}
comb_hist = np.sqrt(g1_hist**2 + th_hist**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2)
comb_hist0 = np.sqrt(g0_hist**2 + np.nan_to_num(learn_hist0)**2 + stat_hist0**2)
lb = lambda k: learn_band[k] if np.isfinite(learn_band[k]) else 0
comb_up = {k: quad(gen['after'][k]['sd'], th_up[k], lb(k), stat[k]) for k in OBK}
comb_dn = {k: quad(gen['after'][k]['sd'], th_dn[k], lb(k), stat[k]) for k in OBK}
comb_hist_up = np.sqrt(g1_hist**2 + th_hist_up**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2)
comb_hist_dn = np.sqrt(g1_hist**2 + th_hist_dn**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2)
print(f'[{tag}] generator band on <1-T>: {gen["before"]["thrust"]["sd"]:.5f} before, {gen["after"]["thrust"]["sd"]:.5f} after; '
      f'on the window mean: {gen["before"]["tau_win"]["sd"]:.5f} before, {gen["after"]["tau_win"]["sd"]:.2e} after; '
      f'combined on <1-T> {comb0["thrust"]:.5f} before, {comb["thrust"]:.5f} after [{time.time()-t0:.0f} s]', flush=True)
out = dict(tag=tag, export=export, grid=grid, wifi=wfit, wifi_central=bool(CENW is not None), nodes=len(nodes), test=TEST, observables=OBK,
           calc=dict(alpha_s=float(G['alphas'][0]), alpha_0=float(G['alpha0'][0]), window=G['window'].tolist()),
           central=dict(theta=central.tolist(), generator_only=o_c0, imposed=o_c, imposed_wifi=o_c,
                        published_generator_only=o_p0, published_imposed=o_p,
                        theory_band=th_band, dispersive_model=np_band, learning_band=learn_band, learning_band_generator_only=learn_band0,
                        combined=comb, combined_generator_only=comb0, window_fraction_before=pw0,
                        theory_up=th_up, theory_down=th_dn, combined_up=comb_up, combined_down=comb_dn,
                        statistical=stat, statistical_generator_only=stat0,
                        theory_variations={lab: dv[lab] for lab in dv}),
           generator_band=gen)
if NODE_LIMIT:
    out['hypercube_size'] = NODES                                  # only the first len(nodes) points were evaluated
S = lambda a: np.asarray(a)[..., nlow:]          # the fitted bins, exactly as before
L = lambda a: np.asarray(a)[..., :nlow]          # the bins below them
hists = dict(h_central_before=h_c0, h_central=h_c, h_central_wifi=h_c, h_published_before=h_p0, h_published=h_p,
             theory_hist=th_hist, np_hist=np_hist, learn_hist=learn_hist, learn_hist_before=learn_hist0,
             gen_hist_before=g0_hist, gen_hist=g1_hist, comb_hist=comb_hist, comb_hist_before=comb_hist0,
             theory_hist_up=th_hist_up, theory_hist_down=th_hist_dn, comb_hist_up=comb_hist_up, comb_hist_down=comb_hist_dn,
             stat_hist=stat_hist, stat_hist_before=stat_hist0)
extra = {}
# the data as the predictions: renormalized over the fitted bins, or as published (to the total rate) when linear
dy, de = (x['y_raw'], x['e_raw']) if LINEAR else (x['y'].numpy(), x['e'].numpy())
if LINEAR:
    # ---- the factor sets of the comparisons (D8): every band again for the corrected prediction ----------------
    XI_N = nodes[:, -1]
    A_N = dict(S=1 - XI_N, H=XI_N)                                 # (1 - xi, xi) at every point
    AC = dict(S=1 - central[-1], H=central[-1])
    WINB = (x['lo'][ext] >= M.WLO - 1e-9) & (x['hi'][ext] <= M.WHI + 1e-9)   # the bins wholly inside the window
    WW = wid[ext][WINB]

    def winnorm(h):
        """The bins wholly inside the window, normalized over them (the last axis of h holds the bins over ext)."""
        hw_ = np.asarray(h)[..., WINB]
        return hw_/np.sum(hw_*WW, axis=-1, keepdims=True)

    def band_set(s, V):
        """Every output of one factor set: the central predictions, the bands, their combinations, as vectors over
        y = (means, bins), the draws at every point, the per-generator theory bands and the window-normalized
        comparison."""
        f = lambda g: 1.0 if V is None else V[g]
        if V is None:
            c0, c1 = vec(o_c0, h_c0), vec(o_c, h_c)
            Y0, Y1 = np.c_[rows0, H0], np.c_[rows1, H1]
        else:
            c0 = sum(AC[g]*f(g)*vec(P0[g][0], P0[g][1]) for g in P0)
            c1 = sum(AC[g]*f(g)*vec(P1[g][0], P1[g][1]) for g in P1)
            Y0 = sum(A_N[g][:, None]*f(g)*np.array(YG[g, 0]) for g in ('S', 'H'))
            Y1 = sum(A_N[g][:, None]*f(g)*np.array(YG[g, 1]) for g in ('S', 'H'))
        g0, g1 = Y0.std(0, ddof=1), Y1.std(0, ddof=1)
        sym, up, dn, one = TH[s]
        l0 = nanv if LC[s][False] is None else np.sqrt(np.maximum(np.diag(LC[s][False]), 0))
        l1 = nanv if LC[s][True] is None else np.sqrt(np.maximum(np.diag(LC[s][True]), 0))
        f0, f1 = np.sqrt(np.maximum(np.diag(CGS[s][False]), 0)), np.sqrt(np.maximum(np.diag(CGS[s][True]), 0))
        s0, s1 = SL0[s], SL[s]
        Q = lambda *v: np.sqrt(sum(np.nan_to_num(a)**2 for a in v))
        return dict(c0=c0, c1=c1, Y0=Y0, Y1=Y1, g0=g0, g1=g1, l0=l0, l1=l1, f0=f0, f1=f1, s0=s0, s1=s1, sym=sym, up=up, dn=dn, one=one,
                    comb=Q(g1, sym, l1, s1), comb0=Q(g0, l0, s0), comb_up=Q(g1, up, l1, s1), comb_dn=Q(g1, dn, l1, s1),
                    combf=Q(f1, sym, l1, s1), combf0=Q(f0, l0, s0), combf_up=Q(f1, up, l1, s1), combf_dn=Q(f1, dn, l1, s1),
                    win0=winnorm(c0[NM:]), win1=winnorm(c1[NM:]), gwin0=winnorm(Y0[:, NM:]).std(0, ddof=1),
                    gwin1=winnorm(Y1[:, NM:]).std(0, ddof=1))
    BS = {s: band_set(s, V) for s, V in VSETS.items()}
    md = lambda v: {k: float(v[j]) for j, k in enumerate(OBK)}
    # the same combinations with the generator band of the formula, each generator at the central point, and the
    # covariances of the formula over (the means of OBK, the histogram including the bins below the fitted ones)
    cf = lambda k: quad(genf1[k], th_band[k], lb(k), stat[k])
    out['central'].update(
        combined_formula={k: cf(k) for k in OBK},
        combined_generator_only_formula={k: quad(genf0[k], learn_band0[k] if np.isfinite(learn_band0[k]) else 0, stat0[k]) for k in OBK},
        combined_up_formula={k: quad(genf1[k], th_up[k], lb(k), stat[k]) for k in OBK},
        combined_down_formula={k: quad(genf1[k], th_dn[k], lb(k), stat[k]) for k in OBK},
        generators={nm: dict(generator_only=P0[g][0], imposed=P1[g][0], window_fraction_before=P1[g][2],
                             theta=at_fraction(central, xg).tolist(),
                             theory_variations={lab: dvg[g][lab] for lab in dvg[g]},
                             theory_up=md(TH[''][3][g][0]), theory_down=md(TH[''][3][g][1]),
                             **({'learning_band': learn_gen[nm]['imposed'], 'learning_band_generator_only': learn_gen[nm]['generator_only']}
                                if w_at is not None else {}))
                    for g, xg, nm in (('S', 0.0, 'sherpa'), ('H', 1.0, 'herwig'))})
    out.update(mix=MIX, box=BOX, tune_scale=TUNE, box_lo=blo.tolist(), box_hi=bhi.tolist(), xi_E=XE.tolist(), xi_var=XVAR,
               normalization='total rate (every bin the fraction of all events in it over its width)',
               data=f'{TEST} as published', learning_point=WPT_NOTE,
               theory_rule='D9: Eq. (thband) at each xi on (1 - xi) d_S + xi d_H, sigma^2 averaged over xi flat on [0, 1]; '
                           'each generator alone: Eq. (thband)',
               generator_band_formula={'before': {k: dict(sd=genf0[k]) for k in OBK}, 'after': {k: dict(sd=genf1[k]) for k in OBK},
                                       'expansion_point': XPT.tolist(), 'expanded_at': XPT_AT},
               window_normalized=dict(bins=[[float(a), float(b)] for a, b in zip(x['lo'][ext][WINB], x['hi'][ext][WINB])],
                                      note='prediction and data normalized over the bins wholly inside the window; '
                                           'the band is the generator band from the draws'),
               conventions=CONV_REC)
    if '_conv' in BS:
        b = BS['_conv']
        gconv = {side: {k: dict(mean=float(Y[:, j].mean()), sd=float(Y[:, j].std(ddof=1)), p16=float(np.percentile(Y[:, j], 16)),
                                p84=float(np.percentile(Y[:, j], 84))) for j, k in enumerate(OBK)}
                 for side, Y in (('before', b['Y0']), ('after', b['Y1']))}
        out['conv'] = dict(note='each generator multiplied by its own factors after imposing and before mixing (D8); only the '
                                'means named in conventions.compared take a factor, the others equal the uncorrected ones',
                           central=dict(generator_only=md(b['c0']), imposed=md(b['c1']), theory_band=md(b['sym']), theory_up=md(b['up']),
                                        theory_down=md(b['dn']), learning_band=md(b['l1']), learning_band_generator_only=md(b['l0']),
                                        statistical=md(b['s1']), statistical_generator_only=md(b['s0']),
                                        combined=md(b['comb']), combined_generator_only=md(b['comb0']),
                                        combined_up=md(b['comb_up']), combined_down=md(b['comb_dn']),
                                        combined_up_formula=md(b['combf_up']), combined_down_formula=md(b['combf_dn']),
                                        generators={GNAME[g]: dict(theory_up=md(b['one'][g][0]), theory_down=md(b['one'][g][1]))
                                                    for g in ('S', 'H')}),
                           generator_band=gconv, generator_band_formula={'before': {k: dict(sd=float(b['f0'][j])) for j, k in enumerate(OBK)},
                                                                         'after': {k: dict(sd=float(b['f1'][j])) for j, k in enumerate(OBK)}})
    hists.update(gen_hist_formula_before=genf0_hist, gen_hist_formula=genf1_hist,
                 comb_hist_formula=np.sqrt(genf1_hist**2 + th_hist**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2),
                 comb_hist_formula_before=np.sqrt(genf0_hist**2 + np.nan_to_num(learn_hist0)**2 + stat_hist0**2),
                 comb_hist_up_formula=np.sqrt(genf1_hist**2 + th_hist_up**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2),
                 comb_hist_down_formula=np.sqrt(genf1_hist**2 + th_hist_dn**2 + np.nan_to_num(learn_hist)**2 + stat_hist**2),
                 **{f'h_central_{nm}': P1[g][1] for g, nm in (('S', 'sherpa'), ('H', 'herwig'))},
                 **{f'h_central_before_{nm}': P0[g][1] for g, nm in (('S', 'sherpa'), ('H', 'herwig'))},
                 **{f'theory_hist_up_{GNAME[g]}': TH[''][3][g][0][NM:] for g in ('S', 'H')},
                 **{f'theory_hist_down_{GNAME[g]}': TH[''][3][g][1][NM:] for g in ('S', 'H')})
    for s, b in BS.items():
        if s:
            hists.update({k + s: v[NM:] for k, v in dict(
                h_central_before=b['c0'], h_central=b['c1'], gen_hist_before=b['g0'], gen_hist=b['g1'], learn_hist=b['l1'],
                learn_hist_before=b['l0'], theory_hist=b['sym'], theory_hist_up=b['up'], theory_hist_down=b['dn'],
                stat_hist=b['s1'], stat_hist_before=b['s0'], comb_hist=b['comb'], comb_hist_before=b['comb0'],
                comb_hist_up=b['comb_up'], comb_hist_down=b['comb_dn'], gen_hist_formula_before=b['f0'], gen_hist_formula=b['f1'],
                comb_hist_up_formula=b['combf_up'], comb_hist_down_formula=b['combf_dn']).items()})
            hists.update({f'theory_hist_up_{GNAME[g]}{s}': b['one'][g][0][NM:] for g in ('S', 'H')})
            hists.update({f'theory_hist_down_{GNAME[g]}{s}': b['one'][g][1][NM:] for g in ('S', 'H')})
            extra.update({f'H_before{s}': S(b['Y0'][:, NM:]), f'H_after{s}': S(b['Y1'][:, NM:]),
                          f'rows_before{s}': b['Y0'][:, :NM], f'rows_after{s}': b['Y1'][:, :NM]})
        # the comparison normalized over the window, apart: the twelve bins, prediction and data, the draws' band
        dw = winnorm(dy[ext]); ew = de[ext][WINB]/np.sum(dy[ext][WINB]*WW)
        extra.update({f'h_central_win{s}': b['win1'], f'h_central_before_win{s}': b['win0'], f'gen_hist_win{s}': b['gwin1'],
                      f'gen_hist_before_win{s}': b['gwin0']})
        if not s:
            extra.update(lo_win=x['lo'][ext][WINB], hi_win=x['hi'][ext][WINB], data_win=dw, err_win=ew)
    # every displacement of each generator (the scale variations of theory_var_labels and the dispersive model of np_labels)
    for g in ('S', 'H'):
        hists[f'theory_var_hist_{GNAME[g]}'] = np.array([dhg[g][lab] for lab in labels])
        hists[f'np_hist_{GNAME[g]}'] = np.array([dhg[g][lab] for lab in NPL])
    extra.update(C_gen_formula_before=CG[False], C_gen_formula=CG[True], **({'C_learn_before': CL[False], 'C_learn': CL[True]} if w_at is not None else {}),
                 **{f'J_theta_{g}{"" if anc else "_before"}': JX[g, anc][1][:, OWN[g]] for g in ('S', 'H') for anc in (False, True)},
                 **{f'y_slopes_{g}{"" if anc else "_before"}': JX[g, anc][0] for g in ('S', 'H') for anc in (False, True)},
                 **{f'C_gen_formula{"" if anc else "_before"}{s}': CGS[s][anc] for s in VSETS if s for anc in (False, True)},
                 formula_expansion_point=XPT, np_labels=np.array(NPL), mix=np.array(MIX),
                 cov_labels=np.array(OBK + [f'thr_{x["lo"][b]:.3f}_{x["hi"][b]:.3f}' for b in ext]), box_lo=blo, box_hi=bhi, xi_E=XE)
json.dump(out, open(OUT + '.json', 'w'), indent=1)
np.savez(OUT + '.npz', lo=x['lo'][use], hi=x['hi'][use], data=dy[use], err=de[use],
         **{k: S(v) for k, v in hists.items()},
         theory_var_hist=S(np.array([dh[lab] for lab in labels])), theory_var_labels=np.array(labels),
         H_before=S(H0), H_after=S(H1), rows_before=rows0, rows_after=rows1, nodes=nodes, obs=np.array(OBK),
         lo_low=x['lo'][low], hi_low=x['hi'][low], data_low=dy[low], err_low=de[low],
         **{k + '_low': L(v) for k, v in hists.items()}, **extra)
print(f'wrote {OUT}.json and {OUT}.npz [{time.time()-t0:.0f} s]')
