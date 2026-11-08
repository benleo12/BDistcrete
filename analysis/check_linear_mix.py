#!/usr/bin/env python3
"""An independent check of the linear mixture (MIX=linear) of envelope_test.py, three_bands.py, band_checks.py and
theory_variations.py, with the public numpy code alone: the MIX17 head of gentune (release/models), the reference
sample reference_MIX17.npz and gentune.maxent. Nothing of directlib or of those scripts is imported.

From scratch, in numpy: each generator's log weights at xi = 0 and 1 (head.bcat, with the learning weights in the
basis of the wifi fit), the maximum-entropy tilt of each generator alone (gentune.maxent.impose), the predictions
normalized to the total rate, their slopes by central finite differences in each generator's own parameters, the
learning weights and the calculated moments, and from those the covariance of the mixture

    Var(xi) D D^T + sum_g E[xi_g^2] J_g V_g J_g^T + sum_gh E[xi_g xi_h] J_g C J_h^T

with E[xi_g xi_h] integrated over the xi range by Gauss-Legendre. Each number is compared with the script's: the
boxes (D6: tune-sized ranges centred on the defaults of gentune.axes), the central predictions of each generator,
a few draws of every set and the per-generator draws, the slopes, the covariances (once assembled from the
script's own slopes, exact, and once from the finite differences), the central point of three_bands.py with its
learning, theory, formula generator and statistical bands and a few of its points, the checks of band_checks.py
and the variations of theory_variations.py. A difference is measured against the spread it would change (the SD
of the draws, the band of the output, the diagonal of the covariance), and every check prints PASS or FAIL.
Also, independently of the scripts' own code: the data as published (DELPHI d11-x01-y02 read from the Rivet table,
D3 and D5), the theory band of D9 (Eq. (thband) at each xi on (1 - xi) d_S + xi d_H, averaged over xi by a dense
Gauss-Legendre rule) and of each generator alone, the expansion point of the formula band (D10: the defaults on
the tune box, the box centre on the full box), the predictions at the hadron level of DELPHI and OPAL (D8, with the
factors read from the factors file directly) and the comparison normalized over the window. Every file must hold
mix = linear. The envelope check runs on every box given (all six of a rerun). In the rerun, with T_DIR set the
head must be an export that carries the refitted temperature of its model (score_obs.check_export_T, E2), and
under BEAM_PHOTONS=remove the observables must come from a bundle of the beam-photon rule in use (bundle_rule.py).

    python check_linear_mix.py --reference reference_MIX17.npz --targets output/thrust_targets_nlo_point.npz \\
        --envelope output/envelope_test_MIXGEOV3S9_tune_lin.npz --bands output/three_bands_MIXGEOV3S9_lin \\
        --checks output/band_checks_MIXGEOV3S9_lin.json --variations output/theory_variations_MIX_lin.json
"""
import argparse, json, os, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'release'))
from gentune import load_head, axes                                  # noqa: E402
from gentune.maxent import load_targets, impose                     # noqa: E402
import bundle_rule                                                  # noqa: E402

OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"
T0 = time.time()
RESULTS = []


def report(name, value, tol):
    ok = bool(np.isfinite(value) and value <= tol)
    RESULTS.append(dict(check=name, value=float(value), tol=tol, ok=ok))
    print(f'  {"PASS" if ok else "FAIL"}  {name}: {value:.2e} (tolerance {tol:g}) [{time.time() - T0:.0f} s]', flush=True)


def rel(a, b, scale):
    """max |a - b| / scale over the entries, scale floored at 1e-300."""
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))/np.maximum(np.asarray(scale, float), 1e-300)))


def xi_E(lo, hi):
    """E[xi_g xi_h] of (1 - xi, xi) and Var(xi) for xi flat on [lo, hi], by 5-point Gauss-Legendre (exact here)."""
    if hi - lo <= 0:
        a = np.array([1 - lo, lo]); return np.outer(a, a), 0.0
    u, wq = np.polynomial.legendre.leggauss(5); xi = lo + (hi - lo)*(u + 1)/2; wq = wq/2
    a = np.stack([1 - xi, xi])
    m = wq @ xi
    return (a*wq) @ a.T, float(wq @ (xi - m)**2)


class Ref:
    """The reference sample with one head: per-generator log weights, the tilt, the predictions."""

    def __init__(self, path, head, targets, observables=None):
        bundle_rule.check(observables or path)      # under BEAM_PHOTONS=remove: a bundle of the beam-photon rule in use
        r = np.load(path)
        AE = r['AE']
        self.N = AE.shape[1]; self.head = head; self.E, self.K = head.ENS, head.K
        self.A = np.ascontiguousarray(AE.transpose(1, 0, 2).reshape(self.N, self.E*self.K), dtype=np.float64)
        del AE
        ro = r if observables is None else np.load(observables)           # the per-event observables, same event order
        self.o = {k: np.asarray(ro[k], np.float64) for k in ('1_minus_thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')}
        assert all(len(v) == self.N for v in self.o.values()), 'the observables do not match the reference events'
        self.tau = self.o['1_minus_thrust']
        self.t = targets
        self.inwin, self.F = targets.functions(self.tau)
        self.win = self.inwin.astype(float)
        print(f'reference: {self.N} events, window [{targets.window[0]:.4f}, {targets.window[1]:.4f}) holds '
              f'{int(self.inwin.sum())} [{time.time() - T0:.0f} s]', flush=True)

    def logw0(self, theta, omega=None):
        """log w0 at theta (xi = 0 or 1 for one generator); omega the learning weights (w_0, W_ij)."""
        b = self.head.bcat(theta).reshape(self.E, self.K)
        if omega is None:
            return self.A @ (b.ravel()/self.E)/self.head.T
        om = np.asarray(omega, np.float64)
        return om[0] + self.A @ (om[1:].reshape(self.E, self.E) @ b).ravel()

    def tilt(self, lw0, c):
        lw, _, info = impose(lw0, self.inwin, self.F, c, self.t.sigma, floor_rel=self.t.floor_rel)
        assert info['converged'], info['residual']
        return lw

    @staticmethod
    def norm(lw):
        w = np.exp(lw - lw.max()); return w/w.sum()

    def means(self, w):
        pw = float(w @ self.win)
        return np.array([float(w @ (self.tau*self.win))/pw, pw] + [float(w @ self.o[k]) for k in
                                                                     ('1_minus_thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')])

    def bins(self, x, edges, density=True, clip_last=False):
        """Per-event bin index for edges (lo <= x < hi), and the widths."""
        e = np.asarray(edges, np.float64)
        b = np.searchsorted(e, x, side='right') - 1
        if clip_last:
            b = np.clip(b, 0, len(e) - 2)
        ok = (b >= 0) & (b < len(e) - 1)
        return np.where(ok, b, -1), np.diff(e)

    def hist(self, w, b, width, nb, density=True):
        ok = b >= 0
        h = np.bincount(b[ok], weights=w[ok], minlength=nb)[:nb]
        return h/width if density else h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--head', default=os.path.join(HERE, 'release', 'models', 'MIX17_head.npz'))
    ap.add_argument('--wifi', default=os.path.join(HERE, 'release', 'models', 'MIX17_wifi.npz'))
    ap.add_argument('--targets', default=os.path.join(HERE, 'output', 'thrust_targets_nlo_point.npz'))
    ap.add_argument('--observables', default=None, help='npz with the per-event observables the scripts used (their '
                    'REF bundle), if not those of the reference file')
    ap.add_argument('--envelope', nargs='*', default=[], help='envelope_test npz files (MIX=linear)')
    ap.add_argument('--bands', default=None, help='three_bands output, without .json/.npz (MIX=linear)')
    ap.add_argument('--checks', default=None, help='band_checks json (MIX=linear)')
    ap.add_argument('--variations', default=None, help='theory_variations json (MIX=linear)')
    ap.add_argument('--ndraw', type=int, default=3)
    ap.add_argument('--nnodes', type=int, default=3)
    ap.add_argument('--tune-scale', type=float, default=1/3)
    ap.add_argument('--out', default=None, help='write the results as json here')
    ap.add_argument('--rivet', default=os.path.join(HERE, 'rivet_ref'), help='the Rivet reference tables (DELPHI d11-x01-y02)')
    ap.add_argument('--factors', default=None, help='the factors file of D8 (default: the one the three_bands json names)')
    a = ap.parse_args()

    head = load_head(a.head); assert head.kind == 'geometric', head.kind
    if os.environ.get('T_DIR'):        # the rerun (D11, E2): the head must carry the refitted temperature of its model
        import score_obs
        try:
            score_obs.check_export_T(a.head, head.T)
        except ValueError as e:
            raise SystemExit(f'T_DIR is set, so --head must be an export, output/models/<tag>_cond.npz: {e}')
    t = load_targets(a.targets)
    W = np.load(a.wifi); C = np.asarray(W['C'], np.float64)
    # the published network of the head's temperature (a refitted temperature keeps C but not the file's w_published)
    W_AT = np.zeros(head.ENS*head.ENS + 1); W_AT[1:] = np.eye(head.ENS).ravel()/(head.ENS*head.T)
    if not np.array_equal(W_AT, np.asarray(W['w_published'], np.float64)):
        print(f'NOTE the wifi file holds w_published of another temperature than the head (T = {head.T}); the head\'s is used')
    R = Ref(a.reference, head, t, a.observables)
    cen, hw = head.norm[:, 0], head.norm[:, 1]; NT = head.nt; NS = head.nS; XI = NT - 1
    OWN = dict(S=np.arange(NS), H=np.arange(NS, XI)); XG = dict(S=0.0, H=1.0)
    DEF = np.array(axes.default('MIX17'), np.float64)
    C0 = t.c
    # Sigma_c = D D^T, D the five scale and scheme variations (each pair at weight 1/2), as envelope_test.py builds it
    D = np.stack([np.sqrt(0.5)*(t[l] - C0) for pr in PAIRS for l in pr] + [t[SCHEME] - C0], 1); SIGC = D @ D.T
    NPL = [str(l) for l in np.load(a.targets)['np_labels']]           # the dispersive-model variations
    at = lambda th, g: np.r_[np.asarray(th, np.float64)[:-1], XG[g]]
    cache = {}

    # ---- the predictions of one generator, as the scripts lay them out --------------------------------------
    def layout_env(d):
        e_t = np.asarray(d['thrust_edges'], float); e_n = np.asarray(d['nch_edges'], float); e_r = np.asarray(d['rho_edges'], float)
        bt, wt = R.bins(R.tau, e_t); bn, _ = R.bins(R.o['nch'], e_n, clip_last=True); br, wr = R.bins(R.o['rho_heavy'], e_r)
        def y(w):
            return np.r_[R.means(w), R.hist(w, bt, wt, len(wt)), R.hist(w, bn, None, len(e_n) - 1, density=False),
                         R.hist(w, br, wr, len(wr))]
        return y

    def layout_bands(z):
        e = np.r_[z['lo_low'], z['lo'], z['hi'][-1]]
        bt, wt = R.bins(R.tau, e)
        return lambda w: np.r_[R.means(w), R.hist(w, bt, wt, len(wt))], bt, wt

    def gen(theta, g, y, omega=None, c=None):
        """(before, after) of generator g at theta (its xi set), omega the learning weights, c the calculation."""
        lw0 = R.logw0(at(theta, g), W_AT if omega is None else omega)
        return np.r_[y(R.norm(lw0)), y(R.norm(R.tilt(lw0, C0 if c is None else c)))]

    def slopes(theta, g, y, key, theta_only=False):
        """Central finite differences of gen() in g's own parameters, the learning weights and the calculation
        (theta_only: the parameters alone)."""
        if (key, g) in cache:
            return cache[key, g]
        th = at(theta, g); y0 = gen(th, g, y); J = {}
        cols = []
        for p in OWN[g]:
            h = 1e-4*hw[p]; tp, tm = th.copy(), th.copy(); tp[p] += h; tm[p] -= h
            cols.append((gen(tp, g, y) - gen(tm, g, y))/(2*h))
        J['theta'] = np.stack(cols, 1)
        if theta_only:
            cache[key, g] = (y0, J); print(f'  slopes of {g} in its parameters by finite differences [{time.time() - T0:.0f} s]', flush=True)
            return y0, J
        cols = []
        for k in range(len(W_AT)):
            h = 1e-2*np.sqrt(max(C[k, k], 1e-12)); op, om = W_AT.copy(), W_AT.copy(); op[k] += h; om[k] -= h
            cols.append((gen(th, g, y, op) - gen(th, g, y, om))/(2*h))
        J['omega'] = np.stack(cols, 1)
        # the calculation along the five directions of D, which is all that Sigma_c = D D^T sees: J_c D. (A step in one
        # moment alone moves the target off the moments of any distribution, where only the floor of Sigma holds it, and
        # needs a step far below the floor, 0.002 |c|, to stay linear.)
        lw = R.logw0(th, W_AT); cols = []
        for j in range(D.shape[1]):
            dy = (y(R.norm(R.tilt(lw, C0 + 1e-3*D[:, j]))) - y(R.norm(R.tilt(lw, C0 - 1e-3*D[:, j]))))/2e-3
            cols.append(np.r_[np.zeros_like(dy), dy])                  # the calculation leaves the generators alone
        J['cD'] = np.stack(cols, 1)
        print(f'  slopes of {g} by finite differences [{time.time() - T0:.0f} s]', flush=True)
        cache[key, g] = (y0, J)
        return y0, J

    def assemble(E, var, ys, Js, V, active):
        """The covariance pieces of the mixture from per-generator predictions ys and slopes Js."""
        n = len(next(iter(ys.values())))
        gen_ = var*np.outer(ys['H'] - ys['S'], ys['H'] - ys['S']) if len(active) == 2 else np.zeros((n, n))
        ix = dict(S=0, H=1)
        for g in active:
            gen_ = gen_ + E[ix[g], ix[g]]*(Js[g]['theta']*V[OWN[g]]) @ Js[g]['theta'].T
        sh = lambda key, Cov, EE: sum(EE[ix[g], ix[h]]*Js[g][key] @ Cov @ Js[h][key].T for g in active for h in active)
        return gen_, sh

    # ---- envelope_test.py -----------------------------------------------------------------------------------
    for f in a.envelope:
        d = np.load(f); js = json.load(open(f.replace('.npz', '.json')))
        box = js['box']; print(f'\nenvelope_test {box}: {f}', flush=True)
        report(f'{box}: the file holds the linear mixture', 0.0 if (js.get('mix') == 'linear' and 'mix' in d.files and str(d['mix']) == 'linear') else 1.0, 0.5)
        lo, hi, m = d['lo'], d['hi'], d['centre']
        kind, _, only = box.partition('_')
        P = (DEF if kind == 'tune' else cen).copy(); h = hw*(a.tune_scale if kind == 'tune' else 1.0)
        h = np.minimum(h, np.minimum(P - (cen - hw), cen + hw - P)); P[XI], h[XI] = 0.5, 0.5
        if only:
            keep = np.zeros(NT, bool); keep[OWN['S' if only == 'sherpa' else 'H']] = True
            h = np.where(keep, h, 0.0); P[XI] = 0.0 if only == 'sherpa' else 1.0
        report(f'{box}: box ranges (D6, centred on the defaults)', max(np.max(np.abs(lo - (P - h))), np.max(np.abs(hi - (P + h)))), 1e-12)
        report(f'{box}: slopes at the centre of the prior', np.max(np.abs(m - P)/hw), 1e-12)
        report(f'{box}: expansion point recorded (D10)', np.max(np.abs(np.asarray(js['expansion_point']) - m)/hw), 1e-12)
        E, var = xi_E(lo[XI], hi[XI]); active = [g for i, g in enumerate('SH') if E[i, i] > 0]
        report(f'{box}: E[xi_g xi_h] and Var(xi) as stored', max(np.max(np.abs(E - d['xi_E'])), abs(var - float(d['xi_var']))), 1e-12)
        y = layout_env(d); V = (hi - lo)**2/12
        sd_draws = d['Y_joint'].std(0, ddof=1) if len(d['Y_joint']) > 2 else np.abs(d['y_central'])
        yc, Jfd = {}, {}
        for g in active:
            yc[g], Jfd[g] = slopes(m, g, y, ('env', tuple(m)))
            report(f'{box}: central prediction of {g} (in SD of the draws)', rel(yc[g], d[f'y_central_{g}'], np.maximum(sd_draws, 1e-9*np.abs(yc[g]))), 1e-4)
        Js = {g: dict(theta=d[f'J_theta_{g}'], omega=d[f'J_omega_{g}'], c=d[f'J_c_{g}']) for g in active}
        # the covariance from the script's own slopes, assembled here: must agree to rounding
        Em = np.outer([1 - m[XI], m[XI]], [1 - m[XI], m[XI]])
        Cg, sh = assemble(E, var, {g: d[f'y_central_{g}'] for g in active}, Js, V, active)
        scale = lambda Cx: np.sqrt(np.outer(np.diag(Cx), np.diag(Cx))) + 1e-300
        for key, mine in (('lin_flat', Cg), ('learn', sh('omega', C, E)), ('theory', sh('c', SIGC, E)),
                          ('learn_mid', sh('omega', C, Em)), ('theory_mid', sh('c', SIGC, Em)),
                          ('lin_z', Cg*6/np.pi)):
            report(f'{box}: C_{key} from the stored slopes', rel(mine, d[f'C_{key}'], scale(d[f'C_{key}'])), 1e-9)
        # the slopes themselves, against finite differences, measured by the band they make
        tot = d['C_lin_flat'] + d['C_learn'] + d['C_theory']; band = np.sqrt(np.maximum(np.diag(tot), 0))
        for g in active:
            for key, Cov in (('theta', np.diag(V[OWN[g]])), ('omega', C), ('c', None)):
                dJ = Jfd[g]['cD'] - Js[g]['c'] @ D if key == 'c' else Jfd[g][key] - Js[g][key]
                err = np.sqrt(np.maximum(np.einsum('ri,ij,rj->r', dJ, np.eye(dJ.shape[1]) if Cov is None else Cov, dJ), 0))
                report(f'{box}: slopes of {g} in {key} (finite differences, in units of the band)', np.max(err/np.maximum(band, 1e-12*np.abs(yc[g]) + 1e-300)), 1e-3)
        Cg2, sh2 = assemble(E, var, yc, Jfd, V, active)
        tot2 = Cg2 + sh2('omega', C, E) + sh2('cD', np.eye(D.shape[1]), E)
        report(f'{box}: whole covariance from finite differences (relative to its diagonal)', rel(tot2, tot, scale(tot)), 2e-3)
        # draws: the joint set (mixed and per generator) and the first draw of every other set
        TH, OM, CS = d['TH'], d['OM'], d['CS']
        for i in range(min(a.ndraw, len(d['Y_joint']))):
            xi = TH[i, XI]; ys = {g: gen(TH[i], g, y, OM[i], CS[i]) for g in active}
            mix = sum((1 - xi if g == 'S' else xi)*ys[g] for g in active)
            report(f'{box}: joint draw {i} (in SD of the draws)', rel(mix, d['Y_joint'][i], np.maximum(sd_draws, 1e-9*np.abs(mix))), 1e-3)
            for g in active:
                report(f'{box}: joint draw {i}, {g} alone', rel(ys[g], d[f'Y_joint_{g}'][i], np.maximum(sd_draws, 1e-9*np.abs(ys[g]))), 1e-3)
        for setname, (th, om, cc) in (('theta', (TH[0], W_AT, C0)), ('omega', (m, OM[0], C0)), ('c', (m, W_AT, CS[0]))):
            xi = th[XI]; mix = sum((1 - xi if g == 'S' else xi)*gen(th, g, y, om, cc) for g in active)
            report(f'{box}: {setname} draw 0', rel(mix, d[f'Y_{setname}'][0], np.maximum(sd_draws, 1e-9*np.abs(mix))), 1e-3)
        report(f'{box}: draws inside the prior', float(np.max(np.maximum(lo - TH, 0) + np.maximum(TH - hi, 0))), 1e-12)

    # ---- the data as published, read here from the Rivet table (D3, D5) and the factors of D8 -----------------
    def delphi_y02():
        txt = open(os.path.join(a.rivet, 'DELPHI_1996_S3430090.yoda')).read()
        m_ = re.search(r'/REF/DELPHI_1996_S3430090/d11-x01-y02\n(.*?)\nEND YODA', txt, re.S)
        rows = []
        for ln in m_.group(1).split('\n'):
            f_ = ln.split()
            if len(f_) == 6 and not ln.startswith('#'):
                try:
                    rows.append([float(v) for v in f_])
                except ValueError:
                    pass
        xx, xm, xp, yy, em, ep = np.array(rows).T; o_ = np.argsort(xx)
        return (xx - xm)[o_], (xx + xp)[o_], yy[o_], (0.5*(np.abs(em) + np.abs(ep)))[o_]
    DLO, DHI, DY, DE = delphi_y02()

    def published(lo_, y_, e_, what):
        """The stored data against DELPHI's all-particle table as published, bin by bin."""
        j = np.array([int(np.argmin(np.abs(DLO - l))) for l in lo_])
        report(f'{what}: data are DELPHI d11-x01-y02 as published (relative)', max(rel(y_, DY[j], DY[j]), rel(e_, DE[j], DE[j])), 1e-9)

    def factors(path):
        """{S, H: (c/a over DELPHI's bins by lower edge, cnu/a of <1-T>)}, read from the factors file directly."""
        F = json.load(open(path))
        out = {}
        for g, nm in (('S', 'sherpa'), ('H', 'herwig')):
            T = F[nm]['delphi_y02']; ca = dict(zip(np.round(np.asarray(T['lo'], float), 6), np.asarray(T['c_over_a'], float)))
            out[g] = (lambda lo_, ca=ca: np.array([ca[round(float(l), 6)] for l in lo_]), float(F[nm]['means']['one_minus_thrust']['cnu_over_a']))
        return out

    def mix_ok(what, ok):
        report(f'{what}: the file holds the linear mixture', 0.0 if ok else 1.0, 0.5)

    # the theory band of D9, by a dense Gauss-Legendre rule over xi (512 panels of 4 nodes), and Eq. (thband) alone
    _u, _w = np.polynomial.legendre.leggauss(4); _e = np.linspace(0, 1, 513)
    XQ = np.concatenate([(_e[k] + _e[k + 1])/2 + (_e[k + 1] - _e[k])/2*_u for k in range(512)]); WQ = np.tile(_w/2/512, 512)

    def thb(d, up):
        f_ = (lambda v: np.maximum(v, 0)) if up else (lambda v: np.maximum(-v, 0))
        t_ = sum(np.maximum(f_(d[x_]), f_(d[y_]))**2 for x_, y_ in PAIRS) + f_(d[SCHEME])**2
        return t_ + sum(f_(d[l])**2 for l in NPL)/len(NPL)

    def thb_mix(dS, dH, up):
        return np.sqrt(sum(wq_*thb({l: (1 - xq_)*dS[l] + xq_*dH[l] for l in dS}, up) for xq_, wq_ in zip(XQ, WQ)))

    # ---- three_bands.py -------------------------------------------------------------------------------------
    if a.bands:
        js = json.load(open(a.bands + '.json')); z = np.load(a.bands + '.npz'); print(f'\nthree_bands: {a.bands}', flush=True)
        mix_ok('three_bands', js.get('mix') == 'linear' and 'mix' in z.files and str(z['mix']) == 'linear')
        c3 = js['central']; th0 = np.asarray(c3['theta']); assert abs(th0[XI] - 0.5) < 1e-12
        report('three_bands: central point at the defaults, xi = 1/2', np.max(np.abs(th0 - DEF)), 1e-12)
        y, bt, wt = layout_bands(z); nm = len(OBK); nlow = len(z['lo_low'])
        E, var = xi_E(0.0, 1.0)
        yc, Jfd = {}, {}
        for g in ('S', 'H'):
            yc[g], Jfd[g] = slopes(th0, g, y, ('bands', tuple(th0)))
        mixc = 0.5*(yc['S'] + yc['H']); n1 = len(mixc)//2
        h_all = lambda k: np.r_[z[k + '_low'], z[k]]
        comb_scale = np.r_[[max(c3['combined_up'][o], c3['combined_down'][o]) for o in OBK],
                           np.maximum(h_all('comb_hist_up'), h_all('comb_hist_down'))]
        for side, (key_o, key_h), sl in (('before', ('generator_only', 'h_central_before'), slice(0, n1)),
                                         ('after', ('imposed', 'h_central'), slice(n1, 2*n1))):
            got = np.r_[[c3[key_o][o] for o in OBK], h_all(key_h)]
            report(f'three_bands: central prediction {side} (in units of the combined band)', rel(mixc[sl], got, comb_scale), 1e-4)
        for g, gname in (('S', 'sherpa'), ('H', 'herwig')):
            got = [c3['generators'][gname]['imposed'][o] for o in OBK]
            report(f'three_bands: {gname} alone, imposed means', rel(yc[g][n1:n1 + nm], got, comb_scale[:nm]), 1e-4)
        # D10: the formula is expanded at the centre of the prior, the defaults on the tune box and the box centre on the full one
        gbf = js['generator_band_formula']; xpt = np.asarray(gbf['expansion_point'])
        want = (cen if js['box'] == 'full' else DEF).copy(); want[XI] = 0.5
        report(f'three_bands: formula expanded at the centre of the prior ({js["box"]} box, D10)', np.max(np.abs(xpt - want)/hw), 1e-12)
        if np.allclose(xpt, th0, rtol=0, atol=1e-12):
            yX, JX = yc, Jfd
        else:
            yX, JX = {}, {}
            for g in ('S', 'H'):
                yX[g], JX[g] = slopes(xpt, g, y, ('bandsx', tuple(xpt)), theta_only=True)
        # learning band (at the central point) and the generator band of the formula (at its expansion point)
        for side, sl, lb_key, lh_key, gb_key, gh_key in (('before', slice(0, n1), 'learning_band_generator_only', 'learn_hist_before', 'before', 'gen_hist_formula_before'),
                                                          ('after', slice(n1, 2*n1), 'learning_band', 'learn_hist', 'after', 'gen_hist_formula')):
            Jw = {g: Jfd[g]['omega'][sl] for g in ('S', 'H')}
            Lc = sum(E[i, j]*Jw[g] @ C @ Jw[h].T for i, g in enumerate('SH') for j, h in enumerate('SH'))
            got = np.r_[[c3[lb_key][o] for o in OBK], h_all(lh_key)]
            report(f'three_bands: learning band {side} (relative)', rel(np.sqrt(np.diag(Lc)), got, np.maximum(got, 1e-12)), 2e-3)
            lo3, hi3 = np.asarray(js['box_lo']), np.asarray(js['box_hi']); V3 = (hi3 - lo3)**2/12
            dl = yX['H'][sl] - yX['S'][sl]
            Gc = var*np.outer(dl, dl) + sum(E[i, i]*(JX[g]['theta'][sl]*V3[OWN[g]]) @ JX[g]['theta'][sl].T for i, g in enumerate('SH'))
            got = np.r_[[gbf[gb_key][o]['sd'] for o in OBK], h_all(gh_key)]
            report(f'three_bands: generator band of the formula {side} (relative)', rel(np.sqrt(np.diag(Gc)), got, np.maximum(got, 1e-12)), 2e-3)
        # theory band: every variation per generator, D9 for the mixture, Eq. (thband) for each generator alone
        q = lambda v: E[0, 0]*v[0]**2 + 2*E[0, 1]*v[0]*v[1] + E[1, 1]*v[1]**2
        ds = {}
        for lab in list(c3['theory_variations']) + NPL:                 # every scale variation and the dispersive model
            ds[lab] = np.array([y(R.norm(R.tilt(R.logw0(at(th0, g), W_AT), t[lab]))) - yc[g][n1:] for g in ('S', 'H')])
        dS, dH = {l: v[0] for l, v in ds.items()}, {l: v[1] for l, v in ds.items()}
        for up, ko, kh in ((True, 'theory_up', 'theory_hist_up'), (False, 'theory_down', 'theory_hist_down')):
            got = np.r_[[c3[ko][o] for o in OBK], h_all(kh)]
            report(f'three_bands: {ko}, D9 (in units of the combined band)', rel(thb_mix(dS, dH, up), got, comb_scale), 1e-4)
            for g, gname, dg_ in (('S', 'sherpa', dS), ('H', 'herwig', dH)):
                got = np.r_[[c3['generators'][gname][ko][o] for o in OBK], h_all(f'{kh}_{gname}')]
                report(f'three_bands: {gname} alone, {ko}, Eq. (thband) (in units of the combined band)', rel(np.sqrt(thb(dg_, up)), got, comb_scale), 1e-4)
        sym = np.sqrt(sum(0.5*q(ds[x_]) + 0.5*q(ds[y_]) for x_, y_ in PAIRS) + q(ds[SCHEME]))
        report('three_bands: symmetric theory band (in units of the combined band)',
               rel(sym, np.r_[[c3['theory_band'][o] for o in OBK], h_all('theory_hist')], comb_scale), 1e-3)
        mixd = {lab: 0.5*(v[0] + v[1]) for lab, v in ds.items()}
        report('three_bands: theory variations at xi = 1/2 (means)', max(rel(mixd[lab][:nm], [c3['theory_variations'][lab][o] for o in OBK], comb_scale[:nm])
                                                                     for lab in c3['theory_variations']), 1e-3)
        report('three_bands: dispersive-model variations at xi = 1/2 (means)', max(rel(mixd[lab][:nm], [c3['dispersive_model'][lab][o] for o in OBK], comb_scale[:nm])
                                                                                for lab in NPL), 1e-3)
        for g, gname in (('S', 'sherpa'), ('H', 'herwig')):          # every displacement of each generator, stored
            labs = [str(l) for l in z['theory_var_labels']]
            got = np.r_[[np.r_[z[f'theory_var_hist_{gname}_low'][i], z[f'theory_var_hist_{gname}'][i]] for i in range(len(labs))]]
            report(f'three_bands: {gname} displacements of the scale variations, stored (in units of the combined band)',
                   max(rel(ds[l][0 if g == 'S' else 1][nm:], got[i], comb_scale[nm:]) for i, l in enumerate(labs)), 1e-3)
        # the factors of D8 (when the file has the corrected predictions)
        fpath = a.factors or (js.get('conventions') or {}).get('file')
        FG = factors(fpath) if ('h_central_conv' in z.files and fpath and os.path.exists(fpath)) else None
        if 'h_central_conv' in z.files and FG is None:
            report('three_bands: the factors file of the corrected predictions is readable', 1.0, 0.5)
        lo_all = np.r_[z['lo_low'], z['lo']]
        Vf = None if FG is None else {g: np.r_[[FG[g][1] if o == 'thrust' else 1.0 for o in OBK], FG[g][0](lo_all)] for g in ('S', 'H')}
        # statistical error of the mixture at xi = 1/2, both generators' per-event terms together
        F = np.zeros((R.N, R.F.shape[1])); F[R.inwin] = R.F - C0[None, :]
        Sig = SIGC + np.diag((t.floor_rel*C0)**2)
        def stat(anchored, Vs=None):
            us = []
            for g in ('S', 'H'):
                lw0 = R.logw0(at(th0, g), W_AT); w0 = R.norm(lw0); P = float(w0 @ R.win)
                w = R.norm(R.tilt(lw0, C0)) if anchored else w0
                if anchored:
                    Fc = F - w @ F; Ji = np.linalg.inv((Fc*w[:, None]).T @ Fc + P**2*Sig)
                    u = lambda O, w=w, Fc=Fc, Ji=Ji: w*(O - Fc @ (Ji @ (Fc.T @ (w*O))))
                else:
                    u = lambda O, w=w: w*O
                pw = float(w @ R.win); tw = float(w @ (R.tau*R.win))/pw
                cols = [R.win*(R.tau - tw)/pw, R.win - pw] + [R.o[k] - w @ R.o[k] for k in ('1_minus_thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')]
                for b in range(len(wt)):
                    ind = (bt == b).astype(float); cols.append((ind - w @ ind)/wt[b])
                us.append([u(O) for O in cols])
            fS = np.ones(len(us[0])) if Vs is None else Vs['S']; fH = np.ones(len(us[0])) if Vs is None else Vs['H']
            return np.array([np.sqrt(np.sum((0.5*fS[k]*us[0][k] + 0.5*fH[k]*us[1][k])**2)) for k in range(len(us[0]))])
        for anchored, ko, kh in ((True, 'statistical', 'stat_hist'), (False, 'statistical_generator_only', 'stat_hist_before')):
            got = np.r_[[c3[ko][o] for o in OBK], h_all(kh)]
            report(f'three_bands: {ko} (relative)', rel(stat(anchored), got, got), 1e-4)
        # a few points of the generator band from draws (mixed, and at the hadron level of D8)
        for i in range(min(a.nnodes, len(z['nodes']))):
            th = z['nodes'][i]; xi = th[XI]
            vg = {g: gen(th, g, y) for g in ('S', 'H')}
            v = sum((1 - xi if g == 'S' else xi)*vg[g] for g in ('S', 'H'))
            got = np.r_[z['rows_before'][i], z['H_before'][i], z['rows_after'][i], z['H_after'][i]]
            mine = np.r_[v[:nm], v[nm + nlow:n1], v[n1:n1 + nm], v[n1 + nm + nlow:]]
            sc = np.r_[comb_scale[:nm], comb_scale[nm + nlow:], comb_scale[:nm], comb_scale[nm + nlow:]]
            report(f'three_bands: point {i} of the draws (in units of the combined band)', rel(mine, got, sc), 1e-3)
            if Vf is not None:
                vc = sum((1 - xi if g == 'S' else xi)*np.r_[Vf[g], Vf[g]]*vg[g] for g in ('S', 'H'))
                got = np.r_[z['rows_after_conv'][i], z['H_after_conv'][i]]
                report(f'three_bands: point {i} of the draws at the hadron level of D8', rel(np.r_[vc[n1:n1 + nm], vc[n1 + nm + nlow:]], got,
                                                                                              np.r_[comb_scale[:nm], comb_scale[nm + nlow:]]), 1e-3)
        if Vf is not None:
            for side, sl, key_o, key_h in (('before', slice(0, n1), 'generator_only', 'h_central_before_conv'), ('after', slice(n1, 2*n1), 'imposed', 'h_central_conv')):
                mine = 0.5*Vf['S']*yc['S'][sl] + 0.5*Vf['H']*yc['H'][sl]
                got = np.r_[[js['conv']['central'][key_o][o] for o in OBK], h_all(key_h)]
                report(f'three_bands: central prediction {side} at the hadron level of D8 (in units of the combined band)', rel(mine, got, comb_scale), 1e-4)
            dSc, dHc = {l: Vf['S']*v for l, v in dS.items()}, {l: Vf['H']*v for l, v in dH.items()}
            for up, ko, kh in ((True, 'theory_up', 'theory_hist_up_conv'), (False, 'theory_down', 'theory_hist_down_conv')):
                got = np.r_[[js['conv']['central'][ko][o] for o in OBK], h_all(kh)]
                report(f'three_bands: {ko} at the hadron level of D8, D9 (in units of the combined band)', rel(thb_mix(dSc, dHc, up), got, comb_scale), 1e-4)
            got = np.r_[[js['conv']['central']['statistical'][o] for o in OBK], h_all('stat_hist_conv')]
            report('three_bands: statistical error at the hadron level of D8 (relative)', rel(stat(True, Vf), got, got), 1e-4)
        # the comparison normalized over the window (the bins wholly inside it), and the data as published
        data = np.r_[z['data_low'], z['data']]; err_ = np.r_[z['err_low'], z['err']]
        published(lo_all, data, err_, 'three_bands')
        report('three_bands: integral of the data as published (1 - 0.9897)', abs(float(data @ wt) - 0.989699), 1e-5)
        hi_all = np.r_[z['hi_low'], z['hi']]; wb = (lo_all >= t.window[0] - 1e-9) & (hi_all <= t.window[1] + 1e-9)
        nw = lambda v: v[..., wb]/np.sum(v[..., wb]*wt[wb], axis=-1, keepdims=True)
        report('three_bands: window-normalized prediction and data (relative)',
               max(rel(nw(h_all('h_central')), z['h_central_win'], z['h_central_win']), rel(nw(data), z['data_win'], z['data_win'])), 1e-12)
        sw = (z['lo'] >= t.window[0] - 1e-9) & (z['hi'] <= t.window[1] + 1e-9); Hn = z['H_after'][:, sw]
        Hn = Hn/np.sum(Hn*(z['hi'] - z['lo'])[sw], axis=1, keepdims=True)
        report('three_bands: window-normalized band from the draws (relative)', rel(Hn.std(0, ddof=1), z['gen_hist_win'], z['gen_hist_win']), 1e-9)

    # ---- band_checks.py ---------------------------------------------------------------------------------------
    if a.checks:
        bc = json.load(open(a.checks)); print(f'\nband_checks: {a.checks}', flush=True)
        mix_ok('band_checks', bc.get('mix') == 'linear')
        th0 = np.asarray(bc['central']); ws = {}
        F = np.zeros((R.N, R.F.shape[1])); F[R.inwin] = R.F - C0[None, :]
        Sig = SIGC + np.diag((t.floor_rel*C0)**2)
        for anchored, name in ((False, 'generators_alone'), (True, 'imposed')):
            us_ = {}
            for g in ('S', 'H'):
                lw0 = R.logw0(at(th0, g)); w0 = R.norm(lw0); ws[g] = R.norm(R.tilt(lw0, C0)) if anchored else w0
                if anchored:                                           # the multipliers solved again: the projection removed
                    w = ws[g]; Fc = F - w @ F; Ji = np.linalg.inv((Fc*w[:, None]).T @ Fc + float(w0 @ R.win)**2*Sig)
                    us_[g] = lambda O, w=w, Fc=Fc, Ji=Ji: w*(O - Fc @ (Ji @ (Fc.T @ (w*O))))
                else:
                    us_[g] = lambda O, w=w0: w*O
            mixm = 0.5*(R.means(ws['S']) + R.means(ws['H']))
            got = np.array([bc[f'stat_{name}']['means'][o] for o in OBK])
            report(f'band_checks: {name} means (relative)', rel(mixm, got, np.abs(got)), 1e-5)
            err = []
            for k, O in enumerate(['tau_win', 'pwin'] + ['1_minus_thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']):
                tot_ = 0
                for g in ('S', 'H'):
                    w = ws[g]; pw = float(w @ R.win)
                    Oc = (R.win*(R.tau - float(w @ (R.tau*R.win))/pw)/pw if O == 'tau_win' else R.win - pw if O == 'pwin'
                          else R.o[O] - w @ R.o[O])
                    tot_ = tot_ + 0.5*us_[g](Oc)
                err.append(np.sqrt(np.sum(tot_**2)))
            got = np.array([bc[f'stat_{name}']['mean_err'][o] for o in OBK])
            report(f'band_checks: {name} statistical errors (relative)', rel(err, got, got), 1e-4)
            wm = 0.5*(ws['S'] + ws['H'])
            report(f'band_checks: {name} largest weight of the mixture (relative)', abs(wm.max()*R.N/bc[f'stat_{name}']['max_weight_over_mean'] - 1), 1e-4)
        if a.bands and bc.get('box') == js.get('box') and bc.get('nodes') == js.get('hypercube_size', js.get('nodes')) \
                and bc.get('nodes_evaluated', bc.get('nodes')) == len(z['nodes']):      # the same points as three_bands
            xs = z['nodes'][:, XI]; dl = yc['H'][n1:n1 + nm] - yc['S'][n1:n1 + nm]
            got = np.array([bc['xi_only']['after']['sd'][o] for o in OBK])
            report('band_checks: xi alone, after (relative)', rel(np.abs(dl)*xs.std(ddof=1), got, got), 1e-4)

    # ---- theory_variations.py ---------------------------------------------------------------------------------
    if a.variations:
        tv = json.load(open(a.variations)); print(f'\ntheory_variations: {a.variations}', flush=True)
        mix_ok('theory_variations', tv.get('mix') == 'linear')
        th0 = np.asarray(tv['central'])
        e = np.r_[tv['lo_low'], tv['lo'], tv['hi'][-1]]; bt, wt = R.bins(R.tau, e); nlow = len(tv['lo_low'])
        y = lambda w: R.hist(w, bt, wt, len(wt))
        hc = {}
        for g in ('S', 'H'):
            hc[g] = y(R.norm(R.tilt(R.logw0(at(th0, g)), C0)))
        got = np.r_[tv['h_imposed_low'], tv['h_imposed']]; sc = np.abs(got)
        report('theory_variations: imposed distribution at xi = 1/2 (relative)', rel(0.5*(hc['S'] + hc['H']), got, sc), 1e-5)
        for g, gname, xi in (('S', 'sherpa', 0), ('H', 'herwig', 1)):
            report(f'theory_variations: {gname} alone, imposed (relative)', rel(hc[g], np.r_[tv[f'xi{xi}']['h_imposed_low'], tv[f'xi{xi}']['h_imposed']], sc), 1e-5)
        worst = 0.0
        for key, v in tv['variations'].items():
            lab = key.split(':', 1)[1]
            for g, gname in (('S', 'sherpa'), ('H', 'herwig')):
                dh = y(R.norm(R.tilt(R.logw0(at(th0, g)), t[lab]))) - hc[g]
                worst = max(worst, rel(dh, np.r_[v[f'dh_{gname}_low'], v[f'dh_{gname}']], sc))
        report('theory_variations: every variation of each generator (relative to the prediction)', worst, 1e-5)
        published(np.r_[tv['lo_low'], tv['lo']], np.r_[tv['data_low'], tv['data']], np.r_[tv['err_low'], tv['err']], 'theory_variations')
        fpath = a.factors or (tv.get('conventions') or {}).get('file')
        if 'h_imposed_conv' in tv and fpath and os.path.exists(fpath):
            FG = factors(fpath); lo_tv = np.r_[tv['lo_low'], tv['lo']]
            mine = 0.5*FG['S'][0](lo_tv)*hc['S'] + 0.5*FG['H'][0](lo_tv)*hc['H']
            report('theory_variations: imposed distribution at the hadron level of D8 (relative)', rel(mine, np.r_[tv['h_imposed_conv_low'], tv['h_imposed_conv']], sc), 1e-5)
            for g, xi in (('S', 0), ('H', 1)):
                report(f'theory_variations: {g} alone at the hadron level of D8 (relative)',
                       rel(FG[g][0](lo_tv)*hc[g], np.r_[tv[f'xi{xi}']['h_imposed_conv_low'], tv[f'xi{xi}']['h_imposed_conv']], sc), 1e-5)

    bad = [r for r in RESULTS if not r['ok']]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks pass [{time.time() - T0:.0f} s]')
    for r in bad:
        print(f'  FAILED {r["check"]}: {r["value"]:.3e} > {r["tol"]:g}')
    if a.out:
        json.dump(dict(results=RESULTS, passed=len(RESULTS) - len(bad), total=len(RESULTS)), open(a.out, 'w'), indent=1)
    print('CHECK LINEAR MIX', 'PASS' if not bad else 'FAIL')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
