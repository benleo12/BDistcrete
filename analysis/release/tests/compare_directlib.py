"""gentune against the paper's torch code, analysis/directlib.py, both in float64.

NOT NUMPY ONLY, and not needed to use gentune: tests/test_mixture.py and tests/test_maxent.py are the
checks that need numpy alone. This one needs torch and scipy, and the paper's analysis code next to
the release folder, analysis/directlib.py with the modules it imports (analysis/fit_alpha0.py,
analysis/np_shift.py, analysis/bundle_rule.py), so it runs in a full clone of the repository, from
analysis/release, and not in a copy of the release folder alone. It runs the same problems through
both codes and prints the largest differences. Two parts.

The maximum-entropy step, gentune.maxent.impose against Model.tilt_on on the same log weights:

  synthetic   50,000 made-up events with a thrust-like tau, imposed with the paper's calculation
              and all its variations, and with a made-up second calculation (six other moments in
              another order, another window, a rank-deficient covariance of its own)
  selftest    the 2000 reference events of models/MIX17_selftest.npz with the weights of each
              generator at its default tune (the MIX17 head at the fraction 0 for Sherpa and 1 for
              Herwig), with the same calculations. The bundle carries no tau. With --reference the
              tau of those events is read from the reference sample at the bundle's
              reference_index, with --tau from a file of their 2000 values, and without either a
              synthetic tau is drawn.

  Printed per case: dlogw, the largest difference of the normalized log weights, which is the
  largest relative difference of any weight; dlam, the largest difference of the multipliers over
  the largest multiplier, both taken after the final Newton step Model.tilt_on takes; dlam_J, the
  size of that multiplier difference as a change of log w, sqrt(dlam . J dlam); cond, the
  condition number of the Newton matrix J; dF, the largest difference of the moment functions.

The paper's mixture of the two generators, gentune.mixture against Model.weights_lin and the
predictions of analysis/three_bands.py (MIX=linear): each generator's weights from the head at the
fraction 0 (Sherpa) and 1 (Herwig), normalized over the reference events, the calculation imposed
on each alone, the predictions mixed with (1 - xi, xi). A real directlib.Model is built on the same
events from three temporary files: an export made of the released MIX17 head and the events' AE,
named after the export the head comes from (MIXGEOV3S9_cond.npz, so that with T_DIR set the Model
checks the head's temperature against the refit), a reference bundle with their tau and n_ch, and
the calculation of the targets file in the layout of the paper's grid files. The events are those
of the self-test bundle, or with --reference and --linear-events N the first N events of the
reference sample (0 for all of them). The points: the paper's central point (the default tunes,
xi = 1/2), the same at xi = 0, 1/4 and 1, and the first --points points of the paper's Latin
hypercube over the full box (gentune.bands.lhs_nodes, 2000 points, seed 11).

  Printed per point: dlogw0 and dlogw, the largest difference of each generator's normalized log
  weights before and after the calculation, over both generators; dw0 and dw, the largest relative
  difference of the mixed per-event weights (1 - xi) w_S + xi w_H; dpred, the largest relative
  difference of the mixed predictions, before and after: <1 - T>, <n_ch>, the mean of 1 - T in the
  window and the window fraction (each generator's own, mixed), and the DELPHI thrust histogram as
  fractions of all events over the bin widths; dstat0 and dstat, that of their statistical errors
  before and after the calculation (directlib.stat_parts with anchored False and True, combined as
  three_bands.py stat_lin does; after it, gentune.mixture.mean_and_error and histogram with
  imposed). At the central point also: the theory band of <1 - T> and of the histogram, for each
  generator alone and its average over xi for the mixture (gentune.bands.theory_band and
  theory_band_mix against directlib.thband and thband_mix, on the displacements of each code),
  and the learning band of the mixture before and after the calculation, of <1 - T>, <n_ch>, the
  window fraction, the histogram and the window mean (gentune.mixture.learning_band, with imposed
  after it, against the derivatives of Model in the WiFi weights by autograd through
  Model.tilt_on at directlib.wifi_point, combined with directlib.mix_cov, as three_bands.py does).

    python tests/compare_directlib.py targets_thrust_nlo.npz [--reference reference_MIX17.npz]
        [--linear-events N] [--points K]
"""
import argparse, os, sys, tempfile, time, warnings
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                       # gentune
sys.path.insert(1, os.path.dirname(os.path.dirname(HERE)))      # analysis/, for directlib
# The temporary reference bundle of the mixture part is not one refreshed under the beam-photon
# rule, and only weights and predictions on the same inputs are compared, so the rerun's bundle
# check is not wanted here.
os.environ['BEAM_PHOTONS'] = 'keep'
try:
    import torch
    import directlib
    from directlib import Model
except ImportError as e:
    if __name__ == '__main__' and ('-h' in sys.argv[1:] or '--help' in sys.argv[1:]):
        print(__doc__)              # the help needs none of them
        sys.exit(0)
    sys.exit(f'compare_directlib.py needs torch, scipy and analysis/directlib.py of a full clone '
             f'of the repository ({e}). tests/test_mixture.py and tests/test_maxent.py are the '
             f'checks that need numpy alone.')
from gentune import load_head, axes, bands, mixture
from gentune.maxent import Targets, load_targets, impose, logsumexp

torch.set_default_dtype(torch.float64)
CENTRAL = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18, 0.102337, 0.654714, 3.528693, 1.849375,
           0.914156, 0.374094, 0.33107, 0.78, 0.5]                # three_bands.sbatch CENTRAL


# -- the maximum-entropy step ------------------------------------------------------------------
def torch_tilt(lw0, tau, names, window, c, sigma, floor_rel):
    """Model.tilt_on on these events, called through a stand-in for the Model instance that
    carries only what tilt_on, funs and sigma read. Returns its log weights, the multipliers after
    its final Newton step, the Newton matrix there, and its moment functions."""
    ns = SimpleNamespace(keys=list(names), G={'sigma_pert': np.asarray(sigma)[None, None]},
                         floor_rel=floor_rel, newton=Model.newton)
    inwin = (tau >= window[0]) & (tau < window[1])
    IW = torch.from_numpy(np.flatnonzero(inwin))
    MW = torch.from_numpy(Model.funs(ns, tau[inwin]))
    lw0t = torch.from_numpy(lw0 - logsumexp(lw0))
    Sig = Model.sigma(ns, 0, 0, c)
    lw, lam, Pw, its = Model.tilt_on(ns, lw0t, IW, MW, c, Sig, torch.zeros(len(names)))
    # the multipliers its weights use, lam minus one Newton step, as tilt_on forms them
    Fw = MW - torch.as_tensor(c)[None, :]
    Sc = Pw**2*torch.as_tensor(Sig)
    zw = lw0t[IW] + Fw @ lam
    lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
    ww = torch.exp(zw - lZ); m = ww @ Fw
    J = (Fw*ww[:, None]).T @ Fw - torch.outer(m, m) + Sc
    lam1 = lam - torch.linalg.solve(J, m + Sc @ lam)
    return lw.numpy(), lam1.numpy(), J.numpy(), MW.numpy(), int(its)


def compare(label, lw0, tau, T, which='central', floor_rel=0.002):
    c = T[which]
    lw_t, lam_t, J, MW, its_t = torch_tilt(lw0, tau, T.names, T.window, c, T.sigma, floor_rel)
    inwin, F = T.functions(tau)
    lw, lam, info = impose(lw0, inwin, F, c, T.sigma, floor_rel=floor_rel)
    d = lam - lam_t
    r = dict(dlogw=float(np.max(np.abs(lw - lw_t))),
             dlam=float(np.max(np.abs(d))/np.max(np.abs(lam_t))),
             dlam_J=float(np.sqrt(max(d @ J @ d, 0.0))),
             cond=float(np.linalg.cond(J)), dF=float(np.max(np.abs(F - MW))),
             its=(info['iterations'], its_t), res=info['residual'], maxpull=float(np.max(np.abs(info['pulls']))))
    print(f'{label:<56} dlogw={r["dlogw"]:.1e} dlam={r["dlam"]:.1e} dlam_J={r["dlam_J"]:.1e} '
          f'cond={r["cond"]:.1e} dF={r["dF"]:.0e} newton={r["its"][0]}/{r["its"][1]} '
          f'residual={r["res"]:.0e} maxpull={r["maxpull"]:.2f}', flush=True)
    return r


def second_calculation(lw0, tau, rng):
    """A made-up calculation: six other moments in another order, the window [0.04, 0.30), c the
    window means of the sample shifted by a few percent, and a covariance of rank four from four
    made-up variations plus a small diagonal, with two of those variations kept as named ones."""
    names = ('tau', 'log2t', 'tlogt', 'tau3', 'log3t', 't2logt')
    probe = Targets(names, np.zeros(6), np.eye(6), (0.04, 0.30))
    inwin, F = probe.functions(tau)
    w = np.exp(lw0[inwin] - logsumexp(lw0[inwin]))
    mean = w @ F
    c = mean*(1 + 0.03*rng.standard_normal(6))
    D = 0.02*np.abs(c)[:, None]*rng.standard_normal((6, 4))
    sigma = D @ D.T + np.diag((0.005*np.abs(c))**2)
    return Targets(names, c, sigma, (0.04, 0.30), {'up': c + D[:, 0], 'down': c - D[:, 1]})


def run(label, lw0, tau, paper, rng):
    out = [compare(f'{label}, paper central', lw0, tau, paper)]
    for v in paper.variations:
        out.append(compare(f'{label}, paper {v}', lw0, tau, paper, v))
    T2 = second_calculation(lw0, tau, rng)
    with warnings.catch_warnings():
        # made-up targets with no distribution behind them: that no tilt of the window events
        # reaches them, and impose() says so, is expected, and both codes still solve the problem
        warnings.filterwarnings('ignore', message='impose: the window held')
        for v in ('central', 'up', 'down'):
            out.append(compare(f'{label}, second calculation {v}', lw0, tau, T2, v))
    # not part of the verdict: the paper's sigma_pert has rank five, so without the floor nine
    # combinations of the fourteen moments would be imposed exactly, J has a condition number
    # near 1e21 and neither code is expected to converge
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        compare(f'{label}, paper central, NO FLOOR (illustration)', lw0, tau, paper, floor_rel=0.0)
    return out


# -- the mixture of the two generators ---------------------------------------------------------
def paper_model(head_path, AE, tau, nch, targets_path, tmp):
    """A directlib.Model on these events, from temporary files (see the module docstring), and the
    name Model.target() takes for each variation of the targets file."""
    h = np.load(head_path)
    export = os.path.join(tmp, 'MIXGEOV3S9_cond.npz')
    np.savez(export, AE=np.asarray(AE, np.float32), **{k: h[k] for k in h.files})
    ref = os.path.join(tmp, 'reference.npz')
    np.savez(ref, **{'1_minus_thrust': np.asarray(tau, np.float64), 'nch': np.asarray(nch, np.float64)})
    t = np.load(targets_path)
    grid = os.path.join(tmp, 'grid.npz')
    np.savez(grid, keys=t['moment_names'], alphas=np.atleast_1d(t['alpha_s']),
             alpha0=np.atleast_1d(t['alpha_0']), window=t['window'], central=t['central'][None, None],
             sigma_pert=t['sigma_pert'][None, None], scale=t['scale_variations'][None, None],
             np_vars=t['np_variations'][None, None], scale_labels=t['scale_labels'],
             np_labels=t['np_labels'])
    M = Model('MIX17', export=export, ref=ref, exps=(), first_bin=0.05,
              floor_rel=float(t['floor_rel']), a0_stride=1, grid=grid)
    which = {**{str(l): f'scale:{v}' for v, l in enumerate(t['scale_labels'])},
             **{str(l): f'np:{v}' for v, l in enumerate(t['np_labels'])}}
    return M, which


def rel(a, b):
    """The largest relative difference of a from b, element by element."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)/np.maximum(np.abs(b), 1e-300)))


class Paper:
    """The paper's predictions on the events of a Model: each generator from Model.weights at its
    fraction, mixed, as three_bands.py predict() and stat_lin() form them."""

    def __init__(self, M, tau, nch, inwin, b, width):
        self.M, self.tau, self.nch, self.inwin, self.b, self.width = M, tau, nch, inwin, b, width

    def logw(self, theta, g, anchored, which='central'):
        th = torch.as_tensor(directlib.at_fraction(np.asarray(theta, np.float64), mixture.XI[g]))
        with torch.no_grad():
            return self.M.weights(th, 0, 0, which=which, anchored=anchored)[0].numpy()

    def one(self, w):
        """<1-T>, <n_ch>, the window mean of 1 - T, the window fraction, the histogram."""
        win = w[self.inwin]
        ok = self.b >= 0
        h = np.bincount(self.b[ok], weights=w[ok], minlength=len(self.width))/self.width
        return np.r_[w @ self.tau, w @ self.nch, win @ self.tau[self.inwin]/win.sum(), win.sum(), h]

    def stat(self, theta, xi, anchored=False):
        """The statistical errors of <1-T>, <n_ch>, the window mean and the histogram, before the
        calculation or with it (anchored), sqrt(sum_i (sum_g a_g u_g,i)^2) with u_g of
        directlib.stat_parts, as three_bands.py stat_lin."""
        a = {'sherpa': 1 - xi, 'herwig': xi}
        parts = {g: directlib.stat_parts(self.M, directlib.at_fraction(np.asarray(theta, np.float64),
                                                                       mixture.XI[g]), anchored=anchored)
                 for g in mixture.GENERATORS if a[g] != 0}

        def per_event(j, w):
            if j == 0:
                return self.tau - w @ self.tau
            if j == 1:
                return self.nch - w @ self.nch
            if j == 2:
                pw = w[self.inwin].sum()
                return np.where(self.inwin, self.tau - (w[self.inwin] @ self.tau[self.inwin])/pw, 0.0)/pw
            ind = (self.b == j - 3).astype(float)
            return (ind - w @ ind)/self.width[j - 3]
        out = []
        for j in range(3 + len(self.width)):
            s = sum(a[g]*u(per_event(j, w)) for g, (w, u) in parts.items())
            out.append(float(np.sqrt(np.sum(s**2))))
        return np.array(out)


def gentune_preds(w, theta, tau, nch, inwin, b, width):
    """What Paper.one() gives, mixed, from gentune.mixture."""
    m = mixture.mean(w, np.c_[tau, nch], theta)
    h, _ = mixture.histogram(b, w, width, theta)
    return np.r_[m, mixture.mean(w, tau, theta, mask=inwin), mixture.mean(w, inwin.astype(float), theta), h]


def gentune_stat(w, theta, tau, nch, inwin, b, width, imposed=None):
    """What Paper.stat() gives, from gentune.mixture; with imposed, the error with the calculation
    imposed, which Paper.stat(anchored=True) gives."""
    _, e = mixture.mean_and_error(w, np.c_[tau, nch], theta, imposed=imposed)
    _, ew = mixture.mean_and_error(w, tau, theta, mask=inwin, imposed=imposed)
    _, he = mixture.histogram(b, w, width, theta, imposed=imposed)
    return np.r_[e, ew, he]


def linear_part(a, head, paper, AE, tau, nch, label):
    """The mixture part on these events; the temporary files stay until it ends, since the Model
    reads its grid file lazily."""
    tmp = tempfile.TemporaryDirectory()
    try:
        return _linear_part(a, head, paper, AE, tau, nch, label, tmp.name)
    finally:
        tmp.cleanup()


def _linear_part(a, head, paper, AE, tau, nch, label, tmp):
    t0 = time.time()
    try:
        M, which = paper_model(a.head, AE, tau, nch, a.targets, tmp)
    except RuntimeError as e:
        if not os.environ.get('T_DIR'):
            raise
        sys.exit(f'{a.head} carries the temperature T = {head.T!r}, and directlib.Model, with T_DIR set, '
                 f'refuses it: {e}. The released head must carry the refitted temperature.')
    N = len(tau)
    Ap = np.empty((N, head.ENS*head.K))                            # float64 once, as the examples do
    for m in range(head.ENS):
        Ap[:, m*head.K:(m + 1)*head.K] = AE[m]
    inwin, F = paper.functions(tau)
    lo, hi = bands.PAPER_THRUST_BINS
    b, width = bands.bin_index(tau, lo, hi), hi - lo
    P = Paper(M, tau, nch, inwin, b, width)
    print(f'\nthe mixture of the two generators on {label}: {N:,} events, {int(inwin.sum()):,} in the '
          f'window; directlib.Model built in {time.time() - t0:.0f} s, T = {M.T!r} (gentune {head.T!r})'
          + (', checked against T_DIR' if os.environ.get('T_DIR') else ''), flush=True)
    at_ok = all(np.array_equal(directlib.at_fraction(np.array(CENTRAL), x), mixture.at_fraction(CENTRAL, x))
                for x in (0.0, 0.3, 1.0))
    theta_c = axes.default('MIX17')
    assert np.array_equal(theta_c, CENTRAL), 'axes.default is not the CENTRAL of three_bands.sbatch'
    pts = [('central, xi = 1/2', theta_c)] + [(f'central, xi = {x:g}', mixture.at_fraction(theta_c, x))
                                              for x in (0.0, 0.25, 1.0)]
    nodes = bands.lhs_nodes(head)[:a.points]
    pts += [(f'hypercube point {i}, xi = {t[-1]:.3f}', t) for i, t in enumerate(nodes)]
    rows = []
    for name, th in pts:
        xi = float(th[-1])
        lw = mixture.log_weights(head, Ap, th)
        w = {g: np.exp(v) for g, v in lw.items()}
        wc, info = mixture.impose(w, inwin, F, paper)
        d0 = {g: P.logw(th, g, False) for g in mixture.GENERATORS}
        d1 = {g: P.logw(th, g, True) for g in mixture.GENERATORS}
        tt = torch.as_tensor(np.asarray(th, np.float64))
        with torch.no_grad():
            wl0 = M.weights_lin(tt, 0, 0, anchored=False)[0].numpy()
            wl1 = M.weights_lin(tt, 0, 0, anchored=True)[0].numpy()
        co = mixture.coefficients(th)
        pd0 = sum(co[g]*P.one(np.exp(d0[g])) for g in mixture.GENERATORS if co[g] != 0)
        pd1 = sum(co[g]*P.one(np.exp(d1[g])) for g in mixture.GENERATORS if co[g] != 0)
        r = dict(dlogw0=max(float(np.max(np.abs(lw[g] - d0[g]))) for g in lw),
                 dlogw=max(float(np.max(np.abs(np.log(wc[g]) - d1[g]))) for g in wc),
                 dw0=rel(mixture.event_weights(w, th), wl0), dw=rel(mixture.event_weights(wc, th), wl1),
                 dpred=max(rel(gentune_preds(w, th, tau, nch, inwin, b, width), pd0),
                           rel(gentune_preds(wc, th, tau, nch, inwin, b, width), pd1)),
                 dstat0=rel(gentune_stat(w, th, tau, nch, inwin, b, width), P.stat(th, xi)),
                 dstat=rel(gentune_stat(wc, th, tau, nch, inwin, b, width, (inwin, F, info)),
                           P.stat(th, xi, anchored=True)),
                 newton=max(i['iterations'] for i in info.values()))
        rows.append(r)
        print(f'  {name:<34} dlogw0={r["dlogw0"]:.1e} dlogw={r["dlogw"]:.1e} dw0={r["dw0"]:.1e} '
              f'dw={r["dw"]:.1e} dpred={r["dpred"]:.1e} dstat0={r["dstat0"]:.1e} dstat={r["dstat"]:.1e} '
              f'newton={r["newton"]} [{time.time() - t0:.0f} s]', flush=True)

    # the theory band at the central point, from the displacements of each code
    w0c = mixture.weights(head, Ap, theta_c)
    base, base_info = mixture.impose(w0c, inwin, F, paper)
    ok_b = b >= 0
    one = lambda wg: np.r_[wg @ tau, np.bincount(b[ok_b], weights=wg[ok_b], minlength=len(width))/width]
    dG = {g: {} for g in mixture.GENERATORS}
    dD = {g: {} for g in mixture.GENERATORS}
    for lab in which:
        wv = mixture.impose(w0c, inwin, F, paper, which=lab)[0]
        for g in mixture.GENERATORS:
            dG[g][lab] = one(wv[g]) - one(base[g])
            dD[g][lab] = one(np.exp(P.logw(theta_c, g, True, which[lab]))) - one(np.exp(P.logw(theta_c, g, True)))
    PAIRS, SCHEME, NPL = [tuple(p) for p in bands.SCALE_PAIRS], bands.SCHEME, list(bands.DISPERSIVE)
    th_rows = {}
    for up in (True, False):
        side = 'up' if up else 'down'
        th_rows[f'mixture {side}'] = rel(bands.theory_band_mix(dG['sherpa'], dG['herwig'], up),
                                         directlib.thband_mix(dD['sherpa'], dD['herwig'], PAIRS, SCHEME, NPL, up))
        for g in mixture.GENERATORS:
            th_rows[f'{g} {side}'] = rel(bands.theory_band(dG[g], up), directlib.thband(dD[g], PAIRS, SCHEME, NPL, up))
    th_disp = max(rel(dG[g][lab], dD[g][lab]) for g in dG for lab in which)
    th_same = max(max(float(np.max(np.abs(bands.theory_band_mix(dD['sherpa'], dD['herwig'], up)
                                          - directlib.thband_mix(dD['sherpa'], dD['herwig'], PAIRS, SCHEME, NPL, up)))),
                      max(float(np.max(np.abs(bands.theory_band(dD[g], up) - directlib.thband(dD[g], PAIRS, SCHEME, NPL, up))))
                          for g in mixture.GENERATORS))
                  for up in (True, False))
    band_tau = (bands.theory_band_mix(dG['sherpa'], dG['herwig'], True)[0],
                bands.theory_band_mix(dG['sherpa'], dG['herwig'], False)[0])
    print(f'  theory band at the central point, <1-T> and the histogram: the displacements of the two codes '
          f'differ by {th_disp:.1e} relative, the bands by ' + ', '.join(f'{k} {v:.1e}' for k, v in th_rows.items())
          + f'; on the same displacements the two functions differ by {th_same:.1e} '
          f'(gentune: <1-T> +{band_tau[0]:.6e} -{band_tau[1]:.6e})', flush=True)

    # the learning band of the mixture at the central point, before and after the calculation:
    # <1-T>, <n_ch>, the window fraction, the histogram, then the window mean
    wifi = os.path.join(a.models, 'MIX17_wifi.npz')
    W = np.load(wifi)
    C = np.asarray(W['C'], np.float64)
    w_at = directlib.wifi_point(W, M, log=lambda *x: None)[0]
    X = np.zeros((N, len(width)))
    X[ok_b, b[ok_b]] = 1.0/width[b[ok_b]]
    O = np.c_[tau, nch, inwin.astype(np.float64), X]
    imp = (inwin, F, base_info)
    lb_g = {anc: np.r_[mixture.learning_band(head, Ap, theta_c, O, wifi, imposed=imp if anc else None),
                       mixture.learning_band(head, Ap, theta_c, tau, wifi, mask=inwin, imposed=imp if anc else None)]
            for anc in (False, True)}
    Ot, wint, taut = torch.from_numpy(O), torch.from_numpy(inwin.astype(np.float64)), torch.from_numpy(tau)
    J = {}
    for g in mixture.GENERATORS:
        th_g = torch.as_tensor(directlib.at_fraction(theta_c, mixture.XI[g]))
        for anc in (False, True):
            wt = torch.tensor(w_at, requires_grad=True)
            M.set_wifi(wt)
            try:
                wv = torch.exp(M.weights(th_g, 0, 0, anchored=anc)[0])
            finally:
                M.set_wifi(None)
            y = torch.cat([wv @ Ot, ((wv @ (taut*wint))/(wv @ wint))[None]])
            J[g, anc] = np.stack([torch.autograd.grad(y[k], wt, retain_graph=True)[0].numpy()
                                  for k in range(len(y))])
    XE = directlib.xi_moments(0.0, 1.0)[0]
    lb_d = {anc: np.sqrt(np.diag(directlib.mix_cov(XE, 0, shared=[(J['sherpa', anc], J['herwig', anc], C)])))
            for anc in (False, True)}
    lb0, lb1 = rel(lb_g[False], lb_d[False]), rel(lb_g[True], lb_d[True])
    xe = float(np.max(np.abs(mixture.xi_moments(0.0, 1.0)[0] - XE)))
    wp_same = bool(np.array_equal(w_at, head._w_published(W)))
    print(f'  learning band of the mixture at the central point, <1-T> and <n_ch>: before the calculation '
          f'gentune {lb_g[False][0]:.6e} {lb_g[False][1]:.6e}, directlib {lb_d[False][0]:.6e} {lb_d[False][1]:.6e}; '
          f'with it gentune {lb_g[True][0]:.6e} {lb_g[True][1]:.6e}, directlib {lb_d[True][0]:.6e} '
          f'{lb_d[True][1]:.6e}; largest relative difference over the {len(lb_d[True])} predictions '
          f'{lb0:.1e} before and {lb1:.1e} after; xi_moments differ by {xe:.0e}; the point of the band '
          f'{"is" if wp_same else "IS NOT"} that of directlib.wifi_point', flush=True)
    worst = {k: max(r[k] for r in rows) for k in ('dlogw0', 'dlogw', 'dw0', 'dw', 'dpred', 'dstat0', 'dstat')}
    worst.update(theory=max(th_rows.values()), theory_same_input=th_same, learning0=lb0, learning=lb1)
    print(f'  worst over {len(rows)} points: ' + '  '.join(f'{k}={v:.1e}' for k, v in worst.items())
          + f'; at_fraction identical: {at_ok} [{time.time() - t0:.0f} s]', flush=True)
    return worst, at_ok and wp_same and xe == 0


def main():
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('targets', help='targets_thrust_nlo.npz, the deposit')
    ap.add_argument('--reference', help='reference_MIX17.npz, for the tau of the self-test events')
    ap.add_argument('--tau', help='tau of the 2000 MIX17 self-test reference events (.npy)')
    ap.add_argument('--models', default=os.path.join(os.path.dirname(HERE), 'models'))
    ap.add_argument('--linear-events', type=int, default=None, metavar='N',
                    help='with --reference, the mixture part on the first N events of the reference '
                         'sample, 0 for all of them, instead of the self-test events')
    ap.add_argument('--points', type=int, default=24,
                    help="points of the paper's Latin hypercube for the mixture part")
    a = ap.parse_args()
    if a.linear_events is not None and not a.reference:
        ap.error('--linear-events needs --reference')
    a.head = os.path.join(a.models, 'MIX17_head.npz')
    rng = np.random.default_rng(5)
    paper = load_targets(a.targets)
    print(f'paper targets: {paper}')
    res = []
    tau = 0.42*rng.beta(0.9, 5.0, 50000)
    lw0 = 0.3*rng.standard_normal(50000) + 2.0*tau
    res += run('synthetic 50000', lw0, tau, paper, rng)
    st = np.load(os.path.join(a.models, 'MIX17_selftest.npz'))
    head = load_head(a.head)
    ref = np.load(a.reference) if a.reference else None
    if ref is not None:
        tau = ref['1_minus_thrust'][st['reference_index']].astype(np.float64)
        lab = 'selftest 2000, real tau'
    elif a.tau:
        tau, lab = np.load(a.tau).astype(np.float64), 'selftest 2000, real tau'
        assert tau.shape == (len(st['mask']),), (tau.shape, len(st['mask']))
    else:
        tau, lab = 0.42*rng.beta(0.9, 5.0, len(st['mask'])), 'selftest 2000, synthetic tau'
    for g, lw0 in mixture.log_weights(head, st['AE'], CENTRAL).items():
        res += run(f'{lab}, {g}', lw0, tau, paper, rng)
    worst = {k: max(r[k] for r in res) for k in ('dlogw', 'dlam', 'dlam_J', 'dF')}
    print('worst over all cases of the maximum-entropy step: '
          + '  '.join(f'{k}={v:.1e}' for k, v in worst.items()))
    ok = worst['dlogw'] < 1e-10 and worst['dlam_J'] < 1e-10 and worst['dF'] == 0

    if a.linear_events is not None:
        n = len(ref['1_minus_thrust']) if a.linear_events == 0 else a.linear_events
        lw, lin_ok = linear_part(a, head, paper, ref['AE'][:, :n], ref['1_minus_thrust'][:n].astype(np.float64),
                                 ref['nch'][:n].astype(np.float64), f'the first {n:,} events of {a.reference}')
    else:
        nch = ((st['particles'][..., 5] != 0) & st['mask'].astype(bool)).sum(1).astype(np.float64)
        lw, lin_ok = linear_part(a, head, paper, st['AE'], tau, nch, f'the self-test events ({lab})')
    # The errors and bands with the calculation imposed go through the inverse of the Newton matrix,
    # whose condition number is near 1e13; directlib inverts it as it stands and gentune with its
    # diagonal scaled to one, so they agree less closely than the predictions do.
    ok_lin = (lin_ok and lw['dlogw0'] < 1e-10 and lw['dlogw'] < 1e-10 and lw['dw0'] < 1e-9
              and lw['dw'] < 1e-9 and lw['dpred'] < 1e-9 and lw['dstat0'] < 1e-8 and lw['dstat'] < 1e-5
              and lw['theory'] < 1e-6 and lw['theory_same_input'] == 0 and lw['learning0'] < 1e-8
              and lw['learning'] < 1e-5)
    print(f'\nthe maximum-entropy step: {"PASS" if ok else "FAIL"}; the mixture of the generators: '
          f'{"PASS" if ok_lin else "FAIL"}')
    print('PASS' if ok and ok_lin else 'FAIL')
    return 0 if ok and ok_lin else 1


if __name__ == '__main__':
    sys.exit(main())
