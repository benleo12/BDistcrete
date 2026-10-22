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
                   calculation, combined as Eq. (sigmac): the scale pairs at half weight, the scheme
                   once; the dispersive-model variations kept separate
  statistical      at the central point, the statistical error of every prediction from the finite
                   reference sample (paper r8), which no band contains
  combined         the three bands and the statistical error added in quadrature; theory_up/down and
                   combined_up/down keep the sign of each theory variation (paper r8), with the
                   dispersive-model variations in the theory band as their root mean square
Observables: the mean of tau inside the window and the window fraction, the full means of 1-T,
B_tot, rho_H, the mean charged multiplicity and the mean baryon count, and the thrust distribution
in the bins of TEST_DATA (DELPHI by default), normalized over the bins above FIRST_BIN as in the
paper's figure. The bins below FIRST_BIN, outside the window, are evaluated too with the same
normalization and written under keys ending in _low, so the bins above it are unchanged.

    python three_bands.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz \\
        output/thrust_targets_nlo_point.npz output/wifi_MIXGEO_fit.npz
Writes output/three_bands_<tag>.json and .npz.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model

tag, export, ref, grid, wfit = sys.argv[1:6]
NODES = int(os.environ.get('NODES', '200')); SEED = int(os.environ.get('SEED', '11'))
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('OUT', f'output/three_bands_{tag}')
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
print(f'[{tag}] {M.N} reference events, d={NT}, head {M.kind} ({M.mix_form}), wifi weights {len(w_wifi) if w_wifi is not None else 0} '
      f'({"the central estimate" if CENW is not None else "learning band only"}), '
      f'calculation at alpha_s={float(G["alphas"][0]):.4f} alpha_0={float(G["alpha0"][0]):.4f} [{time.time()-t0:.0f} s]', flush=True)
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; edges = np.r_[x['lo'], x['hi'][-1]]; use = x['use'].numpy(); wid = x['wid'].numpy()
# every histogram runs over the bins below the fitted ones as well (ext), still normalized over the
# fitted bins (use), so the fitted bins come out exactly as before; nlow of them lie below
low = np.where(x['lo'] < x['lo'][use].min() - 1e-12)[0]; ext = np.r_[low, use]; nlow = len(low)
_DEFAULT = object()


def predict(theta, which='central', anchored=True, wifi=_DEFAULT):
    """Observable means and the test histogram (normalized over the fitted bins) at theta."""
    M.set_wifi(CENW if wifi is _DEFAULT else wifi)
    with torch.no_grad():
        lw, lw0, lam, Pw, its = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, which=which, anchored=anchored)
        w = torch.exp(lw); o = M.observables(w); o = {k: float(o[k]) for k in OBK if k in o}
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
        h = cnt[ext]/(wid[ext]*cnt[use].sum())
    M.set_wifi(None)
    return o, h, float(torch.exp(torch.logsumexp(lw0[M.IW], 0))) if anchored else None


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


# ---- the central point and its theory and learning bands --------------------------------------
central = cen.copy()
if is_mix:
    central[-1] = 0.5
if os.environ.get('CENTRAL'):
    # the central point as a comma-separated list of physical parameter values (for example the
    # default tunes of both generators and a fraction of one half)
    central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64); assert len(central) == NT, (len(central), NT)
    assert np.all(np.abs((central - cen)/hw) <= 1 + 1e-9), 'the central point must lie inside the box'
o_c0, h_c0, pw0 = predict(central, anchored=False)
o_c, h_c, pw = predict(central)
o_p0, h_p0, _ = predict(central, anchored=False, wifi=None)        # the published network, as a baseline
stat, stat_hist = stat_errors(central); stat0, stat_hist0 = stat_errors(central, anchored=False)
o_p, h_p, _ = predict(central, wifi=None)
labels = [str(l) for l in G['scale_labels']]
PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"
dv = {}; dh = {}
for v, lab in enumerate(labels):
    ov, hv, _ = predict(central, which=f'scale:{v}'); dv[lab] = {k: ov[k] - o_c[k] for k in o_c}; dh[lab] = hv - h_c
th_band = {k: float(np.sqrt(sum(0.5*dv[a][k]**2 + 0.5*dv[b][k]**2 for a, b in PAIRS) + dv[SCHEME][k]**2)) for k in o_c}
th_hist = np.sqrt(sum(0.5*dh[a]**2 + 0.5*dh[b]**2 for a, b in PAIRS) + dh[SCHEME]**2)
np_band = {}; np_hist = []
for v, lab in enumerate([str(l) for l in G['np_labels']]):
    ov, hv, _ = predict(central, which=f'np:{v}'); np_band[lab] = {k: ov[k] - o_c[k] for k in o_c}; np_hist.append(hv - h_c)
np_hist = np.array(np_hist)
# signed theory bands (paper r8): each variation moves the prediction one way, so the band above the
# prediction collects the upward displacements and the band below the downward ones, with the weights of
# Eq. (sigmac), the scale pairs at half weight and the change of scheme once, and the three variations
# of the dispersive model enter the theory band as their root mean square. th_band and th_hist above are
# the symmetric band without the dispersive model, kept for reference.
NPL = [str(l) for l in G['np_labels']]
npi = {lab: np_hist[i] for i, lab in enumerate(NPL)}


def _signed(get, up):
    f = (lambda v: np.maximum(v, 0.0)) if up else (lambda v: np.maximum(-v, 0.0))
    tot = sum(0.5*f(get(a))**2 + 0.5*f(get(b))**2 for a, b in PAIRS) + f(get(SCHEME))**2
    return np.sqrt(tot + sum(f(get(l))**2 for l in NPL)/len(NPL))


th_up = {k: float(_signed(lambda lab: np.asarray(dv[lab][k] if lab in dv else np_band[lab][k]), True)) for k in o_c}
th_dn = {k: float(_signed(lambda lab: np.asarray(dv[lab][k] if lab in dv else np_band[lab][k]), False)) for k in o_c}
th_hist_up = _signed(lambda lab: dh[lab] if lab in dh else npi[lab], True)
th_hist_dn = _signed(lambda lab: dh[lab] if lab in dh else npi[lab], False)
print(f'[{tag}] central: generator only <1-T> {o_c0["thrust"]:.5f}, imposed {o_c["thrust"]:.5f}, theory band {th_band["thrust"]:.5f}; '
      f'published network {o_p0["thrust"]:.5f} and {o_p["thrust"]:.5f} [{time.time()-t0:.0f} s]', flush=True)
# learning band: the covariance of the wifi weights at the central point, with and without the
# calculation (NO_WIFI=1 skips it, for heads without one)
if w_wifi is None:
    learn_band = learn_band0 = {k: float('nan') for k in o_c}; learn_hist = learn_hist0 = np.full(len(h_c), np.nan)
else:
    # the derivative is taken at the weights that define the central prediction: the wifi weights
    # when they are the central estimate, otherwise the mean of members written in the same basis
    w_at = w_wifi if CENW is not None else W['w_published']
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
nodes = cen + hw*U
rows0, rows1, H0, H1 = [], [], [], []
for i, th in enumerate(nodes):
    o0, h0, _ = predict(th, anchored=False); o1, h1, _ = predict(th)
    rows0.append([o0[k] for k in OBK]); rows1.append([o1[k] for k in OBK]); H0.append(h0); H1.append(h1)
    if i % 25 == 0:
        print(f'  node {i+1}/{NODES} [{time.time()-t0:.0f} s]', flush=True)
rows0, rows1, H0, H1 = map(np.array, (rows0, rows1, H0, H1))
gen = {'before': {k: dict(mean=float(rows0[:, j].mean()), sd=float(rows0[:, j].std(ddof=1)),
                           p16=float(np.percentile(rows0[:, j], 16)), p84=float(np.percentile(rows0[:, j], 84))) for j, k in enumerate(OBK)},
       'after': {k: dict(mean=float(rows1[:, j].mean()), sd=float(rows1[:, j].std(ddof=1)),
                          p16=float(np.percentile(rows1[:, j], 16)), p84=float(np.percentile(rows1[:, j], 84))) for j, k in enumerate(OBK)}}
g0_hist = H0.std(0, ddof=1); g1_hist = H1.std(0, ddof=1)
# the combination: generator, theory and learning in quadrature (generator only: generator and learning)
# every combination includes the statistical error of the prediction at the central point (paper r8)
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
out = dict(tag=tag, export=export, grid=grid, wifi=wfit, wifi_central=bool(CENW is not None), nodes=NODES, test=TEST, observables=OBK,
           calc=dict(alpha_s=float(G['alphas'][0]), alpha_0=float(G['alpha0'][0]), window=G['window'].tolist()),
           central=dict(theta=central.tolist(), generator_only=o_c0, imposed=o_c, imposed_wifi=o_c,
                        published_generator_only=o_p0, published_imposed=o_p,
                        theory_band=th_band, dispersive_model=np_band, learning_band=learn_band, learning_band_generator_only=learn_band0,
                        combined=comb, combined_generator_only=comb0, window_fraction_before=pw0,
                        theory_up=th_up, theory_down=th_dn, combined_up=comb_up, combined_down=comb_dn,
                        statistical=stat, statistical_generator_only=stat0,
                        theory_variations={lab: dv[lab] for lab in dv}),
           generator_band=gen)
json.dump(out, open(OUT + '.json', 'w'), indent=1)
S = lambda a: np.asarray(a)[..., nlow:]          # the fitted bins, exactly as before
L = lambda a: np.asarray(a)[..., :nlow]          # the bins below them
hists = dict(h_central_before=h_c0, h_central=h_c, h_central_wifi=h_c, h_published_before=h_p0, h_published=h_p,
             theory_hist=th_hist, np_hist=np_hist, learn_hist=learn_hist, learn_hist_before=learn_hist0,
             gen_hist_before=g0_hist, gen_hist=g1_hist, comb_hist=comb_hist, comb_hist_before=comb_hist0,
             theory_hist_up=th_hist_up, theory_hist_down=th_hist_dn, comb_hist_up=comb_hist_up, comb_hist_down=comb_hist_dn,
             stat_hist=stat_hist, stat_hist_before=stat_hist0)
np.savez(OUT + '.npz', lo=x['lo'][use], hi=x['hi'][use], data=x['y'].numpy()[use], err=x['e'].numpy()[use],
         **{k: S(v) for k, v in hists.items()},
         theory_var_hist=S(np.array([dh[lab] for lab in labels])), theory_var_labels=np.array(labels),
         H_before=S(H0), H_after=S(H1), rows_before=rows0, rows_after=rows1, nodes=nodes, obs=np.array(OBK),
         lo_low=x['lo'][low], hi_low=x['hi'][low], data_low=x['y'].numpy()[low], err_low=x['e'].numpy()[low],
         **{k + '_low': L(v) for k, v in hists.items()})
print(f'wrote {OUT}.json and {OUT}.npz [{time.time()-t0:.0f} s]')
