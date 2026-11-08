"""Checks of gentune.mixture, gentune.bands and gentune.beam_photons that need nothing but numpy.

Each check states a property the code must have and tests it on the 2000 reference events of
models/MIX17_selftest.npz, or on made-up numbers: each generator's weights are the head at the
fraction 0 or 1, normalized; a generator whose coefficient is zero is left out and its parameters
are not read; theta_from must be one generator alone; a mixed mean, window mean, histogram and
statistical error equal the same quantities written out event by event, with and without the
factors of a measurement's convention; the calculation imposed on each generator alone gives
normalized weights; the statistical error with the calculation imposed is the paper's projected
one, written out event by event, and refuses weights that are not the solution; the learning band
of the mixture reduces to that of one generator at xi = 0 or 1 and averages its square over xi as
stated; with the calculation imposed its derivative equals finite differences of the re-solved
predictions; a WiFi file of another temperature is taken at the head's temperature, with a
warning; the paper's points are the Latin hypercube of analysis/three_bands.py; the theory band of
the mixture is the average over xi it claims to be; the observables leave the beam photons out and
change nothing else; the beam-photon rule passes its own checks and, in the features gentune
builds from the example HepMC files, flags exactly the photons that travel along the beam in the
generator record, and its command line says FAIL for a file it cannot use and prints its
docstring for --help; the factors file of the measurements, examples/data/convention_factors.json,
is a byte-identical copy of the paper's analysis/conventions/factors.json when the checkout has
that file (SKIP otherwise), and examples/reproduce_paper.py takes from it the factor of each
comparison; a weight of the network at an interior fraction warns. Given a deposit
reference_MIX17.npz, it also checks that reading it warns if and only if it is the deposit of an
earlier draft, and that the theory band's variations are those of the targets_thrust_nlo.npz next
to it.

    python tests/test_mixture.py [reference_MIX17.npz]

It prints PASS or FAIL for each check and exits 1 if any fails.
"""
import contextlib, io, json, os, sys, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from gentune import load_head, axes, bands, mixture, beam_photons, features, GeometricFractionWarning
from gentune.maxent import load_targets, Targets

warnings.filterwarnings('ignore', message='impose: the window held')
MODELS = os.path.join(ROOT, 'models')
FAILED = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')
    if not ok:
        FAILED.append(name)


def close(a, b, tol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(np.max(np.abs(a - b)/np.maximum(np.abs(b), 1e-300)))
    return d <= tol, f'(largest relative difference {d:.1e})'


def refused(f, *args, **kw):
    """True if f(*args, **kw) raises ValueError."""
    try:
        f(*args, **kw)
    except ValueError:
        return True
    return False


def made_up_targets(tau, w):
    """A calculation the events can reach: the window means of the weights w, moved by 1 percent of
    their spread, with a covariance of rank three."""
    names = ('tau', 'log2t', 'tlogt', 'tau3')
    probe = Targets(names, np.zeros(4), np.eye(4), (0.05, 1/3))
    inwin, F = probe.functions(tau)
    ww = w[inwin]/w[inwin].sum()
    c = ww @ F + 0.01*np.sqrt(ww @ (F - ww @ F)**2)
    D = 0.005*np.abs(c)[:, None]*np.random.default_rng(1).standard_normal((4, 3))
    return Targets(names, c, D @ D.T, (0.05, 1/3)), inwin, F


def usage(argv):
    """None for a good command line, else the exit code: 0 after -h or --help, which print the
    docstring, and 2 after a usage error, which is printed."""
    args = argv[1:]
    if '-h' in args or '--help' in args:
        print(__doc__)
        return 0
    if len(args) > 1 or any(a.startswith('-') for a in args):
        print('usage: python tests/test_mixture.py [reference_MIX17.npz] (--help says more)')
        return 2
    if args and not os.path.isfile(args[0]):
        print(f'ERROR: {args[0]}: no such file')
        return 2
    return None


def main(argv):
    code = usage(argv)
    if code is not None:
        return code
    head = load_head(os.path.join(MODELS, 'MIX17_head.npz'))
    sb = np.load(os.path.join(MODELS, 'MIX17_selftest.npz'))
    A, P, M = sb['AE'], sb['particles'], sb['mask'].astype(bool)
    N = len(M)
    obs = bands.observables(P, M)
    tau = np.clip(obs['1-T'], 1e-6, 0.4999)
    O = np.column_stack(list(obs.values()))
    rng = np.random.default_rng(3)
    lo, hi = bands.PAPER_THRUST_BINS
    b, width = bands.bin_index(tau, lo, hi), hi - lo
    Xbins = np.zeros((N, len(width)))
    Xbins[b >= 0, b[b >= 0]] = 1/width[b[b >= 0]]

    # -- each generator's weights ---------------------------------------------------------------
    theta = axes.default('MIX17')
    w = mixture.weights(head, A, theta)
    with warnings.catch_warnings():
        warnings.simplefilter('error', GeometricFractionWarning)       # none at xi = 0 and 1
        same = [close(w[g], head.weights(A, mixture.at_fraction(theta, mixture.XI[g])), 1e-12)[0]
                for g in mixture.GENERATORS]
    check('each generator is the head at xi = 0 (Sherpa) or 1 (Herwig)', all(same))
    check('each set of weights sums to one', all(abs(v.sum() - 1) < 1e-12 for v in w.values()))
    t2 = mixture.at_fraction(theta, 0.3)
    check('at_fraction copies theta and sets the fraction', theta[-1] == 0.5 and t2[-1] == 0.3
          and np.array_equal(t2[:-1], theta[:-1]))
    w2 = mixture.weights(head, A, t2)
    check('the fraction of theta does not enter the weights',
          all(np.array_equal(w[g], w2[g]) for g in w))
    src = mixture.at_fraction(theta, 0.0)
    lr = mixture.log_weights(head, A, theta, theta_from=src)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GeometricFractionWarning)
        ok = [close(np.exp(lr[g]), head.ratio_weights(A, mixture.at_fraction(theta, mixture.XI[g]), src),
                    1e-12)[0] for g in mixture.GENERATORS]
    check('with theta_from, the weights of ratio_weights', all(ok))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        bad = [refused(mixture.weights, head, A, theta, theta_from=t) for t in (theta, t2)]
        bad.append(refused(mixture.learning_band, head, A, theta, tau, os.path.join(MODELS, 'MIX17_wifi.npz'),
                           theta_from=theta))
    check('theta_from at an interior fraction is refused, as a sample no generator makes',
          all(bad) and not rec)
    check('a single-generator head is refused',
          refused(mixture.weights, load_head(os.path.join(MODELS, 'E_head.npz')), A, theta))

    # -- a generator with coefficient zero ------------------------------------------------------
    nS, hw = head.nS, head.norm[:, 1]
    far = theta.copy()
    far[nS] = head.centre[nS] + 1.5*hw[nS]                         # Herwig's first parameter out of the box
    w0 = mixture.weights(head, A, mixture.at_fraction(far, 0.0))
    check('at xi = 0 Herwig is left out, and its parameters may lie outside the box',
          list(w0) == ['sherpa'] and np.array_equal(w0['sherpa'], w['sherpa']))
    far_s = theta.copy()
    far_s[0] = head.centre[0] + 1.5*hw[0]                          # Sherpa's first parameter out of the box
    check('... while Sherpa\'s own parameters outside the box are refused there, and not at xi = 1',
          refused(mixture.weights, head, A, mixture.at_fraction(far_s, 0.0))
          and list(mixture.weights(head, A, mixture.at_fraction(far_s, 1.0))) == ['herwig'])
    check('the head at the fraction 0 or 1 does not read the other generator\'s parameters',
          np.array_equal(head.logit(A, mixture.at_fraction(far, 0.0), check_box=False),
                         head.logit(A, mixture.at_fraction(theta, 0.0)))
          and np.array_equal(head.logit(A, mixture.at_fraction(far_s, 1.0), check_box=False),
                             head.logit(A, mixture.at_fraction(theta, 1.0))))

    # -- mixing ---------------------------------------------------------------------------------
    inwin = (tau >= 0.05) & (tau < 1/3)
    for xi in (0.0, 0.3, 1.0):
        t = mixture.at_fraction(theta, xi)
        m = mixture.mean(w, O, t)
        by_hand = (1 - xi)*(w['sherpa'] @ O) + xi*(w['herwig'] @ O)
        ok1 = close(m, by_hand, 1e-13)
        ok2 = close(mixture.event_weights(w, t) @ O, by_hand, 1e-13)
        check(f'xi = {xi}: the mean is (1 - xi) <O>_S + xi <O>_H, also from the per-event weights',
              ok1[0] and ok2[0], ok1[1])
        wm = sum(c*(w[g][inwin] @ tau[inwin])/w[g][inwin].sum() for g, c in mixture.coefficients(t).items())
        check(f'xi = {xi}: the window mean is each generator\'s own, mixed',
              *close(mixture.mean(w, tau, t, mask=inwin), wm, 1e-13))
        # the statistical error, event by event
        u = sum(c*w[g][:, None]/w[g].sum()*(O - (w[g]/w[g].sum()) @ O)
                for g, c in mixture.coefficients(t).items())
        check(f'xi = {xi}: the statistical error, sqrt(sum_i (sum_g a_g w_g,i (O_i - <O>_g))^2)',
              *close(mixture.mean_and_error(w, O, t)[1], np.sqrt((u**2).sum(0)), 1e-12))
        for over in (False, True):
            keep = (b >= 0) if over else np.ones(N, bool)
            ws = {g: np.where(keep, w[g], 0.0)/np.where(keep, w[g], 0.0).sum() for g in w}
            hv = sum(c*(ws[g] @ Xbins) for g, c in mixture.coefficients(t).items())
            uu = sum(c*ws[g][:, None]*(Xbins - ws[g] @ Xbins) for g, c in mixture.coefficients(t).items())
            h, e = mixture.histogram(b, w, width, t, over_bins=over)
            ok_v, d_v = close(h, hv, 1e-12)
            ok_e, d_e = close(e, np.sqrt((uu**2).sum(0)), 1e-9)
            check(f'xi = {xi}: the histogram{" over the bins" if over else ", fractions of all events"}, '
                  f'values and errors event by event', ok_v and ok_e, d_v + ' ' + d_e)
    check('a single histogram is the mixture at xi = 0',
          np.allclose(bands.histogram(b, w['sherpa'], width)[0],
                      mixture.histogram(b, w, width, mixture.at_fraction(theta, 0.0))[0], rtol=1e-14, atol=0))
    # the factors of a measurement's convention, one per generator and observable or bin
    t = mixture.at_fraction(theta, 0.3)
    co = mixture.coefficients(t)
    fo = {g: rng.uniform(0.9, 1.1, O.shape[1]) for g in w}
    fb = {g: rng.uniform(0.9, 1.1, len(width)) for g in w}
    mf, ef = mixture.mean_and_error(w, O, t, factors=fo)
    uf = sum(c*fo[g]*w[g][:, None]*(O - w[g] @ O) for g, c in co.items())
    ok_m = close(mf, sum(c*fo[g]*(w[g] @ O) for g, c in co.items()), 1e-13)
    ok_e = close(ef, np.sqrt((uf**2).sum(0)), 1e-12)
    hf, hef = mixture.histogram(b, w, width, t, factors=fb)
    uh = sum(c*fb[g]*w[g][:, None]*(Xbins - w[g] @ Xbins) for g, c in co.items())
    ok_h = close(hf, sum(c*fb[g]*(w[g] @ Xbins) for g, c in co.items()), 1e-12)
    ok_he = close(hef, np.sqrt((uh**2).sum(0)), 1e-12)
    check('factors: each generator\'s mean, bin and per-event term times its factor before the mixing',
          ok_m[0] and ok_e[0] and ok_h[0] and ok_he[0], ' '.join(x[1] for x in (ok_m, ok_e, ok_h, ok_he)))
    part = mixture.combine({'sherpa': {'a': 1.0, 'b': np.array([1.0, 2.0])}, 'herwig': {'a': 3.0, 'b': np.array([3.0, 4.0])}},
                           0.25, factors={'sherpa': {'a': 2.0, 'b': 1.0}, 'herwig': {'a': 1.0, 'b': np.array([0.5, 1.0])}})
    check('combine mixes dicts, each generator times its factor first',
          part['a'] == 0.75*2.0 + 0.25*3.0 and np.allclose(part['b'], [0.75*1 + 0.25*1.5, 0.75*2 + 0.25*4]))
    check('combine leaves out a generator with no weight', mixture.combine({'sherpa': 2.0}, 0.0) == 2.0)
    tp = mixture.combine({'sherpa': (np.array([1.0, 2.0]), 3.0), 'herwig': (np.array([3.0, 4.0]), 5.0)}, 0.5)
    check('combine mixes tuples element by element', isinstance(tp, tuple) and np.allclose(tp[0], [2.0, 3.0])
          and tp[1] == 4.0)

    # -- the calculation imposed on each generator ------------------------------------------------
    calc, inwin, F = made_up_targets(tau, 0.5*(w['sherpa'] + w['herwig']))
    wc, info = mixture.impose(w, inwin, F, calc)
    check('impose: each generator converged, its weights sum to one',
          all(info[g]['converged'] and abs(wc[g].sum() - 1) < 1e-12 for g in wc))
    single = {g: np.exp(calc.impose(np.log(w[g]), inwin, F)[0]) for g in w}
    check('impose: each generator alone, as maxent.impose', all(close(wc[g], single[g], 1e-12)[0] for g in w))
    wc2, _ = mixture.impose(w, inwin, F, (calc.c, calc.sigma), floor_rel=calc.floor_rel)
    check('impose: a pair (c, Sigma) gives what the Targets gives',
          all(close(wc2[g], wc[g], 1e-12)[0] for g in w))
    ia = mixture.impose(w, inwin, F, calc, lam0={g: info[g]['lam'] for g in w})[1]
    check('impose: starting from the solution takes no Newton step',
          all(i['iterations'] == 0 for i in ia.values()))

    # -- the statistical error with the calculation imposed ----------------------------------------
    imp = (inwin, F, info)
    Fa = np.zeros((N, F.shape[1]))
    Fa[inwin] = F - calc.c

    def projected(g, x):
        """x minus its projection, written out: x - Fc J^-1 Cov(Fc, x), J = Cov(Fc, Fc) + P^2 Sigma."""
        wg = wc[g]
        Fc = Fa - wg @ Fa
        J = (Fc*wg[:, None]).T @ Fc + info[g]['pwin_before']**2*info[g]['sigma_used']
        d = 1/np.sqrt(np.diag(J))
        return x - Fc @ (d[:, None]*np.linalg.solve(J*np.outer(d, d), d[:, None]*(Fc.T @ (wg[:, None]*x))))
    ok_all, ratio = [], None
    for xi in (0.0, 0.3, 1.0):
        t = mixture.at_fraction(theta, xi)
        co = {g: c for g, c in mixture.coefficients(t).items() if c}
        u = sum(c*wc[g][:, None]*projected(g, O - wc[g] @ O) for g, c in co.items())
        pw = {g: wc[g][inwin].sum() for g in co}
        uw = sum(c*wc[g][:, None]*projected(g, (np.where(inwin, tau - (wc[g][inwin] @ tau[inwin])/pw[g], 0.0)
                                                 /pw[g])[:, None]) for g, c in co.items())
        uh = sum(c*wc[g][:, None]*projected(g, Xbins - wc[g] @ Xbins) for g, c in co.items())
        ok_all += [close(mixture.mean_and_error(wc, O, t, imposed=imp)[1], np.sqrt((u**2).sum(0)), 1e-9),
                   close(mixture.mean_and_error(wc, tau, t, mask=inwin, imposed=imp)[1], np.sqrt((uw**2).sum()), 1e-9),
                   close(mixture.histogram(b, wc, width, t, imposed=imp)[1], np.sqrt((uh**2).sum(0)), 1e-9)]
        if xi == 0.3:
            ratio = (mixture.mean_and_error(wc, tau, t, mask=inwin, imposed=imp)[1]
                     / mixture.mean_and_error(wc, tau, t, mask=inwin)[1])
    check('the error with the calculation imposed, the per-event terms minus their projection on the '
          'constraint functions, event by event (means, window mean, histogram; xi = 0, 0.3, 1)',
          all(x[0] for x in ok_all), max(x[1] for x in ok_all))
    check('... which takes most of the error of the window mean away, as the calculation fixes it',
          ratio < 0.1, f'(imposed over without: {ratio:.3f})')
    check('... and for one generator, bands.mean_and_error and bands.histogram with imposed give it too',
          np.array_equal(bands.mean_and_error(wc['sherpa'], O, imposed=(inwin, F, info['sherpa']))[1],
                         mixture.mean_and_error(wc, O, 0.0, imposed=imp)[1])
          and np.array_equal(bands.histogram(b, wc['sherpa'], width, imposed=(inwin, F, info['sherpa']))[1],
                             mixture.histogram(b, wc, width, 0.0, imposed=imp)[1]))
    check('... and weights that are not the solution, or an info without a generator, are refused',
          refused(mixture.mean_and_error, w, O, theta, imposed=imp)
          and refused(mixture.mean_and_error, wc, O, theta, imposed=(inwin, F, {'sherpa': info['sherpa']}))
          and refused(bands.mean_and_error, wc['herwig'], O, imposed=(inwin, F, info['sherpa'])))

    # -- the learning band ----------------------------------------------------------------------
    wifi = os.path.join(MODELS, 'MIX17_wifi.npz')
    for g, xi in mixture.XI.items():
        t = mixture.at_fraction(theta, xi)
        ok = close(mixture.learning_band(head, A, t, O, wifi, xi_prior=None), head.learning_band(A, t, O, wifi), 1e-12)
        check(f'learning band at xi = {xi:g} is that of {g} alone', *ok)
    W = np.load(wifi)
    G = {g: head._learning_gradient(A, mixture.at_fraction(theta, mixture.XI[g]), O, W) for g in mixture.GENERATORS}
    C = W['C']
    q = lambda x, y: np.einsum('ak,kl,al->a', x, C, y)
    flat = np.sqrt((q(G['sherpa'], G['sherpa']) + q(G['herwig'], G['herwig']) + q(G['sherpa'], G['herwig']))/3)
    check('learning band of the mixture, xi flat on [0, 1]: (g_S C g_S + g_H C g_H + g_S C g_H)/3',
          *close(mixture.learning_band(head, A, theta, O, wifi), flat, 1e-12))
    xs = np.linspace(0, 1, 2001)
    num = np.array([q((1 - x)*G['sherpa'] + x*G['herwig'], (1 - x)*G['sherpa'] + x*G['herwig']) for x in xs])
    avg = np.sqrt(((num[:-1] + num[1:])/2).mean(0))
    check('... which is the average over xi of the square of the band at each xi', *close(flat, avg, 1e-6))
    # with the calculation imposed: the derivative through the re-solved multipliers
    Ol = np.c_[tau, inwin.astype(float), Xbins[:, 3:9]]
    r2 = np.random.default_rng(9)
    worst = 0.0
    for g in mixture.GENERATORS:
        t = mixture.at_fraction(theta, mixture.XI[g])
        Gi = head._learning_gradient(A, t, Ol, W, imposed=(inwin, F, info[g]))
        Fb, wp = head.wifi_basis(A, t), head._w_published(W)

        def means(om):
            lw = calc.impose(Fb @ om, inwin, F, lam0=info[g]['lam'], tol=1e-13)[0]
            return np.exp(lw) @ Ol
        for k in range(3):
            d = 0.01*r2.standard_normal(len(wp))
            fd = (means(wp + 1e-3*d) - means(wp - 1e-3*d))/2e-3
            worst = max(worst, float(np.max(np.abs(fd - Gi @ d)/np.abs(fd))))
    check('learning band with the calculation imposed: its derivative equals finite differences of the '
          'predictions with the multipliers solved again (means and bins, both generators)',
          worst < 1e-4, f'(largest relative difference {worst:.1e})')
    ok = [close(mixture.learning_band(head, A, mixture.at_fraction(theta, x), Ol, wifi, xi_prior=None,
                                      imposed=(inwin, F, {g: info[g]})),
                head.learning_band(A, mixture.at_fraction(theta, x), Ol, wifi, imposed=(inwin, F, info[g])), 1e-12)
          for g, x in mixture.XI.items()]
    check('... and the mixture\'s at xi = 0 or 1 is that of the one generator', all(o[0] for o in ok))
    check('... and counts are not taken with imposed',
          refused(head.learning_band, A, mixture.at_fraction(theta, 0.0), tau, wifi, counts=np.ones(N),
                  imposed=(inwin, F, info['sherpa'])))
    # a WiFi file written at another temperature
    h3 = load_head(os.path.join(MODELS, 'MIX17_head.npz'))
    h3.T = 1.025
    t = mixture.at_fraction(theta, 0.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        b3 = h3.learning_band(A, t, tau, wifi)
        h3.learning_band(A, t, tau, wifi)
    w3 = h3.weights(A, t)
    g3 = (w3*(tau - w3 @ tau)) @ h3.wifi_basis(A, t)
    n_warn = sum('temperature' in str(x.message) for x in rec)
    check('a WiFi file of another temperature: the band at the temperature of the head, warned once',
          n_warn == 1 and close(b3, np.sqrt(g3 @ C @ g3), 1e-10)[0], close(b3, np.sqrt(g3 @ C @ g3), 1e-10)[1])

    # -- the paper's points -----------------------------------------------------------------------
    nodes = bands.lhs_nodes(head)
    r = np.random.default_rng(11)
    U = (np.argsort(r.random((2000, 17)), axis=0) + r.random((2000, 17)))/2000*2 - 1
    check("the paper's points: 2000, seed 11, the full box, as three_bands.py (BOX=full)",
          nodes.shape == (2000, 17) and np.array_equal(nodes, head.centre + head.norm[:, 1]*U)
          and nodes[:, -1].min() >= 0 and nodes[:, -1].max() <= 1
          and np.histogram(nodes[:, -1], 10, (0, 1))[0].tolist() == [200]*10)

    # -- the theory band ------------------------------------------------------------------------
    labs = [l for p in bands.SCALE_PAIRS for l in p] + [bands.SCHEME] + list(bands.DISPERSIVE)
    dS = {l: rng.normal(0, 1, 7) for l in labs}
    dH = {l: rng.normal(0, 1, 7) for l in labs}
    xs = np.linspace(0, 1, 20001)
    for up in (True, False):
        sq = np.array([bands.theory_band({l: (1 - x)*dS[l] + x*dH[l] for l in labs}, up)**2 for x in xs])
        brute = np.sqrt(((sq[:-1] + sq[1:])/2).mean(0))
        check(f'theory band of the mixture ({"up" if up else "down"}): the average over xi of the square',
              *close(bands.theory_band_mix(dS, dH, up), brute, 1e-6))
        check(f'theory band of the mixture ({"up" if up else "down"}) of one generator twice is its own',
              *close(bands.theory_band_mix(dS, dS, up), bands.theory_band(dS, up), 1e-12))
        check(f'theory band of the mixture ({"up" if up else "down"}) at one fixed xi',
              *close(bands.theory_band_mix(dS, dH, up, lo=0.3, hi=0.3),
                     bands.theory_band({l: 0.7*dS[l] + 0.3*dH[l] for l in labs}, up), 1e-12))
    # the targets file of the deposit, next to its reference sample
    tp = (os.path.join(os.path.dirname(os.path.abspath(argv[1])), 'targets_thrust_nlo.npz')
          if len(argv) > 1 else None)
    if tp and os.path.isfile(tp):
        check('the labels of the theory band are those of the targets file',
              set(labs) <= set(load_targets(tp).variations))
    elif tp:
        print(f'SKIP  the labels of the theory band against {tp}, which does not exist')

    # -- the observables and the beam photons ------------------------------------------------------
    keep = bands.observables(P, M, beam_photons='keep')
    bp = beam_photons.beam_photon_mask(P, M)
    has = bp.any(1)
    check('n leaves out the beam photons, n_ch and n_baryon are unchanged',
          np.array_equal(keep['n'] - obs['n'], bp.sum(1)) and np.array_equal(keep['n_ch'], obs['n_ch'])
          and np.array_equal(keep['n_baryon'], obs['n_baryon']))
    check('1 - T is unchanged in an event without a beam photon',
          np.array_equal(obs['1-T'][~has], keep['1-T'][~has]), f'({int((~has).sum())} such events)')
    z, cos, phi, lm = (P[..., c].astype(np.float64) for c in (0, 1, 2, 3))
    pm = np.where(M & ~bp, np.sqrt(np.maximum((z*bands.SQRT_S)**2 - 10**(2*lm), 0.0)), 0.0)
    sin = np.sqrt(np.maximum(1 - cos**2, 0))
    P3 = pm[..., None]*np.stack([sin*np.cos(phi), sin*np.sin(phi), cos], -1)
    one = []
    for i in np.flatnonzero(has)[:300]:
        p3 = P3[i][pm[i] > 0]
        n = features.thrust_axis(p3)
        one.append(1 - np.abs(p3 @ n).sum()/pm[i].sum())
    check('1 - T of an event with beam photons: the axis features.thrust_axis finds without them',
          *close(obs['1-T'][np.flatnonzero(has)[:300]], one, 1e-12))
    check('the beam-photon rule: version 3, and its own checks pass on these events',
          beam_photons.RULE_VERSION == 3 and beam_photons.check(P, M)['photons'] == int(bp.sum()))
    for f in ('sherpa_default.hepmc.gz', 'herwig_default.hepmc.gz'):
        ev = features.parse_hepmc(os.path.join(ROOT, 'examples', 'data', f))
        Fx, Mx, _ = features.features_from_events(ev)
        along = np.array([int(((e[:, 4] == 22) & (np.hypot(e[:, 1], e[:, 2]) < 1e-6*e[:, 0])).sum()) for e in ev])
        nbx = beam_photons.n_beam(Fx, Mx)
        check(f'{f}: the rule flags, in the features gentune builds, the photons along the beam in the record',
              np.array_equal(nbx, along) and along.sum() > 0, f'({int(along.sum())} photons in {len(ev)} events)')
    check('an unknown beam_photons option is refused', refused(bands.observables, P, M, beam_photons='drop'))
    outs = []
    for p in (os.path.join(ROOT, 'examples', 'data', 'expected.npz'), os.path.join(HERE, 'no_such_file.npz'),
              '--help', '--no-such-option'):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = beam_photons.main(['beam_photons', p])
        outs.append((code, buf.getvalue()))
    check('python -m gentune.beam_photons: a file without particles and mask, or none at all, prints '
          'FAIL and exits 1; --help prints the docstring and exits 0, another option exits 2',
          outs[0][0] == 1 and 'FAIL, needs the arrays particles and mask' in outs[0][1]
          and outs[1][0] == 1 and 'FAIL, cannot read it' in outs[1][1]
          and outs[2] == (0, beam_photons.__doc__ + '\n') and outs[3][0] == 2 and 'usage:' in outs[3][1])

    # -- the factors of the measurements: the file, and what reproduce_paper.py takes from it ---------
    sys.path.insert(0, os.path.join(ROOT, 'examples'))
    import reproduce_paper as rp
    original = os.path.normpath(os.path.join(ROOT, '..', 'conventions', 'factors.json'))
    if os.path.exists(original):
        with open(rp.FACTORS, 'rb') as f1, open(original, 'rb') as f2:
            same = f1.read() == f2.read()
        check('examples/data/convention_factors.json is a byte-identical copy of '
              'analysis/conventions/factors.json, the file the paper\'s code reads', same)
    else:
        print(f'SKIP  examples/data/convention_factors.json against {original}, which this checkout '
              f'does not have')
    with open(rp.FACTORS) as f:
        fd = json.load(f)
    V, meta = rp.factor_vectors(rp.FACTORS)
    th = rp.OBS.index('thrust')
    ok = all(np.array_equal(V[g][rp.NM:], fd[g]['delphi_y02']['c_over_a'])
             and V[g][th] == fd[g]['means']['one_minus_thrust']['cnu_over_a']
             and all(V[g][j] == 1.0 for j in range(rp.NM) if j != th) for g in mixture.GENERATORS)
    check('reproduce_paper.py takes, per generator, the DELPHI factor of every bin, the OPAL factor of '
          '<1-T> and none for the other means', ok and meta == fd['meta'])
    shifted, gone = json.loads(json.dumps(fd)), json.loads(json.dumps(fd))
    shifted['herwig']['delphi_y02']['lo'][3] += 0.005
    del gone['sherpa']
    check('... and refuses factors in other bins, or without a generator',
          refused(rp.factor_vectors, shifted) and refused(rp.factor_vectors, gone))

    # -- the network at an interior fraction --------------------------------------------------------
    h2 = load_head(os.path.join(MODELS, 'MIX17_head.npz'))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        h2.weights(A, mixture.at_fraction(theta, 1.0))
        none_at_edge = not any(issubclass(x.category, GeometricFractionWarning) for x in rec)
        h2.weights(A, theta)
        h2.weights(A, theta)
        n_warn = sum(issubclass(x.category, GeometricFractionWarning) for x in rec)
    check('MixtureHead.weights warns at an interior fraction, once, and not at xi = 1',
          none_at_edge and n_warn == 1)

    if len(argv) > 1:
        ref = np.load(argv[1])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            bands.reference_observables(ref)
        earlier = bands.EARLIER_DEPOSIT_TAU_SHA256
        import hashlib
        is_earlier = hashlib.sha256(np.ascontiguousarray(ref['1_minus_thrust']).tobytes()).hexdigest() == earlier
        warned = any('earlier draft' in str(x.message) for x in rec)
        check(f'reading the deposit warns if and only if it is the earlier draft\'s ({"it is" if is_earlier else "it is not"})',
              warned == is_earlier)

    print(f'\n{"FAILED: " + ", ".join(FAILED) if FAILED else "all checks pass"}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
