"""Imposing a calculation on the reweighted reference sample. Pure numpy, no torch.

The head carries the reference sample to a parameter point theta and gives every reference event
a weight w0. A calculation, such as the NNLL+NLO thrust moments of the paper, predicts the means
of a few functions m_j(tau) over the events inside a window of tau, together with the covariance
of those means. Imposing it means finding the weights w closest to w0 in relative entropy whose
window means agree with the calculation within that covariance. This module solves that problem
as the paper's code does (analysis/directlib.py, Model.tilt_on) and reproduces its weights to
float64 rounding.

The problem. Normalize w0 to one over all events. Let Theta_i be one for an event inside the
window and zero outside, c the calculated window means, F_i = m(tau_i) - c on the window events,
Sigma the covariance of c, and P = sum over the window of w0_i, the window fraction before the
tilt. The weights are

    w_i = w0_i exp(Theta_i F_i . lam) / Z(lam),
    Z(lam) = (1 - P) + sum over the window of w0_i exp(F_i . lam),

so an event outside the window keeps its weight relative to every other event outside, and only
the normalization moves it. The multipliers lam minimize the convex dual

    D(lam) = log Z(lam) + 1/2 lam . (P^2 Sigma) lam ,

whose gradient is E_w[Theta F] + P^2 Sigma lam, so at the solution

    E_w[Theta F] = - P^2 Sigma lam .

E_w[Theta F] is P_w (<m>_w - c), the window fraction after the tilt times the miss of the
window-conditional means. The same weights minimize the relative entropy KL(w | w0) plus
1/2 (P_w/P)^2 (<m>_w - c) . Sigma^-1 (<m>_w - c), a chi^2 of the window means against the
calculation. With Sigma = 0 the window means come out equal to c, and with Sigma > 0 they are
pulled toward c only as far as that chi^2 is worth in relative entropy. The factor P^2 is what
turns the full-sample expectation of Theta F into the window-conditional mean the calculation
predicts.

The covariance used is Sigma + diag((floor_rel c)^2), with floor_rel = 0.002 as in the paper, so
no moment is held to better than 0.2 percent of its value whatever the covariance supplied says.
This matters for the paper's own Sigma, which is built from five variations of the calculation
and so has rank five: without the floor, nine combinations of the fourteen moments would be
imposed exactly and the Newton matrix has a condition number near 1e21. Pass floor_rel=0 to use
Sigma as given.

The fourteen thrust moments of the paper are the window means of tau^a ln^b tau, tau = 1 - T,
named as in THRUST_MOMENTS ('log2t' is ln^2 tau, 'tlogt' is tau ln tau, 'tau3' is tau^3). Any
name of that pattern is understood, and the same functions of any other positive per-event
observable are formed the same way. For functions of any other form, compute their per-event
values on the reference events yourself and pass them to impose() as F; nothing in impose() is
specific to thrust.

    targets = load_targets('targets_thrust_nlo.npz')
    inwin, F = targets.functions(tau)                 # tau of the reference events
    logw, lam, info = targets.impose(logw0, inwin, F)
"""
import json
import re
import warnings
import numpy as np


# The paper's constraint functions, in the order its targets file lists them.
THRUST_MOMENTS = ('logt', 'tau', 'tlogt', 'log2t', 'tau2', 'tlog2t', 't2logt', 'log3t', 'tau3',
                  'tlog3t', 't3logt', 't2log2t', 'log4t', 'tau4')
FLOOR_REL = 0.002      # the floor on the calculation's uncertainty, relative to each moment
TAU_MIN = 1e-6         # tau is clipped here before the logarithm, as in the paper's code

_NAME = re.compile(r'^(?:(t|tau)(\d*))?(?:log(\d*)t)?$')


def logsumexp(x):
    """log sum exp(x) with the largest term pulled out, so nothing overflows."""
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return -np.inf
    top = float(np.max(x))
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.sum(np.exp(x - top))))


# -- the moment functions ---------------------------------------------------------------------
def moment_powers(name):
    """(a, b) for a moment named for tau^a ln^b tau: 'tau2' is (2, 0), 'log3t' is (0, 3),
    'tlogt' is (1, 1), 't2log2t' is (2, 2). Raises on a name of any other form."""
    g = _NAME.match(str(name))
    if g is None or not (g.group(1) or g.group(3) is not None):
        raise ValueError(f'{name!r} is not a moment name of the form tau^a ln^b tau, such as '
                         f'tau2, log3t or t2logt')
    a = (int(g.group(2)) if g.group(2) else 1) if g.group(1) else 0
    b = (int(g.group(3)) if g.group(3) else 1) if g.group(3) is not None else 0
    if a == 0 and b == 0:
        raise ValueError(f'{name!r} is the constant function, which constrains nothing')
    return a, b


def moment_functions(tau, names=THRUST_MOMENTS, powers=None):
    """tau^a ln^b tau per event and per function, shape (len(tau), k). The powers (a, b) are read
    from the names, or given directly as powers, one pair per function. tau is clipped at TAU_MIN
    first, as the paper's code does, so an event at tau = 0 has a finite logarithm, and the
    products are formed in the same order as there, so the values agree to the last bit."""
    pw = [moment_powers(n) for n in names] if powers is None else [tuple(int(v) for v in p) for p in powers]
    t = np.clip(np.asarray(tau, np.float64), TAU_MIN, None)
    L = np.log(t)
    cols = []
    for a, b in pw:
        if a == 0 and b == 0:
            raise ValueError('the power (0, 0) is the constant function, which constrains nothing')
        x = (t if a == 1 else t**a) if a else None
        y = (L if b == 1 else L**b) if b else None
        cols.append(x*y if (x is not None and y is not None) else (x if x is not None else y))
    return np.column_stack(cols)


def in_window(tau, window):
    """True for the events with window[0] <= tau < window[1], the half-open window of the paper."""
    t = np.asarray(tau, np.float64)
    return (t >= float(window[0])) & (t < float(window[1]))


def thrust_moments(tau, window, names=THRUST_MOMENTS, powers=None):
    """(inwin, F) for impose(): inwin is the boolean window mask over all events and F the moment
    functions on the window events only, shape (inwin.sum(), k), in event order."""
    inwin = in_window(tau, window)
    return inwin, moment_functions(np.asarray(tau, np.float64)[inwin], names, powers)


# -- the maximum-entropy step -----------------------------------------------------------------
def impose(logw0, inwin, F, c, Sigma, floor_rel=FLOOR_REL, lam0=None, tol=1e-10, maxiter=60):
    """Impose calculated window means on a weighted sample. Returns (logw, lam, info).

    logw0     log weights of all N events before the tilt, any normalization (the head's logit
              divided by its temperature will do)
    inwin     the window, a boolean mask over the N events or the indices of the events inside
    F         the constraint functions on the window events, shape (n_window, k), rows in the
              order of the window events; an (N, k) array is accepted too and cut to the window
    c         the calculated window-conditional means of the k functions
    Sigma     their k by k covariance, to which diag((floor_rel c)^2) is added
    lam0      starting multipliers, for example the solution at a nearby parameter point
    tol       the solve stops when |E_w[Theta F] + P^2 Sigma lam| falls below this, as the
              paper's code does

    The multipliers minimize the dual D(lam) of the module docstring by Newton's method, each
    step shortened until D decreases by a fixed fraction of the Newton decrement, or, once that
    decrease is below what float64 can resolve in D, until the gradient shrinks. One more full
    Newton step is then taken at the converged multipliers, as Model.tilt_on does, and kept unless
    it leaves a larger residual. The returned lam is the one the weights are built from, so
    logw = logw0 + Theta F . lam - log Z(lam) holds exactly for the returned pair.

    logw is normalized, sum exp(logw) = 1 over all N events. info holds the number of Newton
    steps, the final residual and whether it is below tol, the window fraction before and after,
    the window means after the tilt and their pulls against c in units of the floored Sigma, the
    relative entropy KL(w | w0) and the effective sample size 1/sum w^2, and the covariance used.
    """
    lw0 = np.asarray(logw0, np.float64).ravel()
    N = lw0.shape[0]
    lw0 = lw0 - logsumexp(lw0)
    iw = np.asarray(inwin)
    if iw.dtype == bool:
        assert iw.shape == (N,), f'the window mask has shape {iw.shape} for {N} events'
        iw = np.flatnonzero(iw)
    iw = iw.astype(np.int64).ravel()
    if len(iw) == 0:
        raise ValueError('no event lies inside the window')
    assert len(np.unique(iw)) == len(iw), 'the window lists an event twice'
    c = np.asarray(c, np.float64).ravel()
    k = len(c)
    F = np.asarray(F, np.float64)
    if F.ndim == 1:
        F = F[:, None]
    if F.shape[0] == N and len(iw) != N:
        F = F[iw]
    assert F.shape == (len(iw), k), \
        f'F has shape {F.shape}, expected ({len(iw)} window events, {k} functions)'
    S = np.asarray(Sigma, np.float64).reshape(k, k) + np.diag((float(floor_rel)*np.abs(c))**2)
    out = np.ones(N, bool)
    out[iw] = False
    lw = lw0[iw]
    logP = logsumexp(lw)
    P = float(np.exp(logP))
    log_out = logsumexp(lw0[out])          # log(1 - P), -inf when every event is in the window
    Fw = F - c[None, :]
    Sc = P**2*S

    def state(l):
        z = lw + Fw @ l
        lZ = float(np.logaddexp(log_out, logsumexp(z)))
        ww = np.exp(z - lZ)                # the tilted weights of the window events
        m = ww @ Fw                        # E_w[Theta F]
        return lZ + 0.5*float(l @ Sc @ l), ww, m, m + Sc @ l

    def newton_step(ww, m, g):
        # J = Cov_w(Theta F) + P^2 Sigma, solved with its diagonal scaled to one: the moments
        # differ in size by orders of magnitude, and for the paper's fourteen the scaling takes the
        # condition number from about 1e13 to about 1e7
        J = (Fw*ww[:, None]).T @ Fw - np.outer(m, m) + Sc
        d = 1.0/np.sqrt(np.diag(J))
        return d*np.linalg.solve(J*np.outer(d, d), d*g)

    lam = np.zeros(k) if lam0 is None else np.asarray(lam0, np.float64).ravel().copy()
    assert lam.shape == (k,), f'lam0 has {lam.shape[0]} entries for {k} functions'
    D, ww, m, g = state(lam)
    gn = float(np.linalg.norm(g))
    its = 0
    for its in range(maxiter):
        if gn < tol:
            break
        step = newton_step(ww, m, g)
        dec = float(g @ step)              # the squared Newton decrement, twice the predicted drop
        s, accepted = 1.0, False
        for _ in range(30):
            lt = lam - s*step
            Dt, wt, mt, gt = state(lt)
            gnt = float(np.linalg.norm(gt))
            if Dt <= D - 1e-4*s*dec:
                accepted = True
                break
            # a drop this small is below the rounding of D, so judge the step by the gradient
            if s*dec <= 1e-13*max(1.0, abs(D)) and gnt < gn:
                accepted = True
                break
            s *= 0.5
        if not accepted:
            break
        lam, D, ww, m, g, gn = lt, Dt, wt, mt, gt, gnt
    else:
        its = maxiter
    converged = gn < tol
    # one more Newton step at the converged multipliers, the one Model.tilt_on takes, kept unless
    # it leaves a larger residual, which happens only when the Newton matrix is near singular
    l1 = lam - newton_step(ww, m, g)
    _, w1, m1, g1 = state(l1)
    if float(np.linalg.norm(g1)) <= max(gn, 1e-15):
        lam, ww, m, g = l1, w1, m1, g1
    z = lw0.copy()
    z[iw] += Fw @ lam
    logw = z - logsumexp(z)
    if not converged:
        warnings.warn(f'impose: the Newton solve stopped after {its} steps with residual '
                      f'{gn:.2e} above tol={tol:.0e}; the weights are not the solution')
    w = np.exp(logw)
    pw = float(ww.sum())
    mom = (ww @ F)/pw
    info = dict(iterations=int(its), residual=float(np.linalg.norm(g)), converged=bool(converged),
                pwin_before=P, pwin_after=pw, moments=mom, pulls=(mom - c)/np.sqrt(np.diag(S)),
                kl=float(w @ (logw - lw0)), n_eff=float(1.0/np.sum(w**2)), sigma_used=S)
    if abs(pw/P - 1) > 0.1:
        # The penalty of the problem is weighted by the square of the window fraction, so when no
        # tilt of the window events comes near c, moving weight out of the window is the cheaper
        # answer, and the pulls then look small. The paper's calculation and its variations move
        # the fraction by less than 1e-3, so a change of more than 10 percent flags a calculation
        # that these events cannot reach.
        warnings.warn(f'impose: the window held {P:.3g} of the weight before and holds {pw:.3g} '
                      f'after. No reweighting of these events comes near the calculated means '
                      f'within Sigma, and the solution moves weight across the window edge instead, '
                      f'because the penalty scales with the square of the window fraction; small '
                      f'pulls do not mean agreement here. Check the calculation: predictions from '
                      f'these weights are not a result.',
                      stacklevel=2)
    return logw, lam, info


# -- targets ----------------------------------------------------------------------------------
class Targets:
    """A calculation to impose: the window, the k functions, their calculated window means c, the
    covariance sigma of c, the relative floor added to sigma, and any number of named variations,
    each a replacement for c (scale choices, schemes, model variations). A variation keeps the
    same sigma, and the floor is taken on the varied c, as the paper's code does.

    names label the functions. Unless powers are given, every name must be of the tau^a ln^b tau
    pattern of moment_powers and the powers are read from it; with powers given, the names are
    free labels. See load_targets for the file format and save() to write one."""

    def __init__(self, names, c, sigma, window, variations=None, floor_rel=FLOOR_REL, powers=None,
                 meta=None):
        self.names = [str(n) for n in np.atleast_1d(names)]
        k = len(self.names)
        if powers is None:
            self.powers = [moment_powers(n) for n in self.names]
        else:
            self.powers = [tuple(int(v) for v in p) for p in np.asarray(powers).reshape(-1, 2)]
            assert len(self.powers) == k, f'{k} names but {len(self.powers)} pairs of powers'
            for n, p in zip(self.names, self.powers):
                try:
                    q = moment_powers(n)
                except ValueError:
                    continue
                assert q == p, f'the name {n!r} says tau^{q[0]} ln^{q[1]} tau, its powers say {p}'
        self.c = np.asarray(c, np.float64).ravel()
        assert self.c.shape == (k,), f'{k} names but {self.c.shape[0]} values of c'
        self.sigma = np.asarray(sigma, np.float64)
        assert self.sigma.shape == (k, k), f'sigma has shape {self.sigma.shape}, expected ({k}, {k})'
        top = max(float(np.abs(self.sigma).max()), 1e-300)
        assert np.allclose(self.sigma, self.sigma.T, rtol=0, atol=1e-12*top), 'sigma is not symmetric'
        ev = np.linalg.eigvalsh(self.sigma)
        assert ev[0] >= -1e-10*max(ev[-1], 0.0), \
            f'sigma is not a covariance, its smallest eigenvalue is {ev[0]:.3e}'
        self.window = (float(np.ravel(window)[0]), float(np.ravel(window)[1]))
        assert self.window[0] < self.window[1], f'the window {self.window} is empty'
        self.floor_rel = float(floor_rel)
        self.variations = {}
        for name, v in (variations or {}).items():
            v = np.asarray(v, np.float64).ravel()
            assert v.shape == (k,), f'variation {name!r} has {v.shape[0]} values for {k} functions'
            self.variations[str(name)] = v
        self.meta = dict(meta or {})

    def __repr__(self):
        return (f'Targets({len(self.names)} moments, window [{self.window[0]:.4g}, '
                f'{self.window[1]:.4g}), floor {self.floor_rel:g}, {len(self.variations)} variations)')

    def __getitem__(self, which):
        """c for 'central', or the c of a variation by its name."""
        return self.c if which == 'central' else self.variations[which]

    def functions(self, tau):
        """(inwin, F) for impose(), from the per-event values tau of the reference events."""
        return thrust_moments(tau, self.window, self.names, self.powers)

    def impose(self, logw0, inwin, F, which='central', **kw):
        """impose() with this calculation's c (or a variation's, by name), sigma and floor."""
        return impose(logw0, inwin, F, self[which], self.sigma, floor_rel=self.floor_rel, **kw)

    def save(self, path):
        """Write the targets in the format of load_targets, as .json or .npz by the extension."""
        if str(path).endswith('.json'):
            d = dict(moment_names=self.names, moment_powers=[list(p) for p in self.powers],
                     window=list(self.window), central=self.c.tolist(), sigma=self.sigma.tolist(),
                     floor_rel=self.floor_rel,
                     variations={n: v.tolist() for n, v in self.variations.items()})
            with open(path, 'w') as f:
                json.dump(d, f, indent=1)
            return
        vn = list(self.variations)
        np.savez(path, moment_names=np.array(self.names), moment_powers=np.array(self.powers, np.int64),
                 window=np.array(self.window), central=self.c, sigma=self.sigma,
                 floor_rel=np.float64(self.floor_rel), variation_names=np.array(vn, dtype=str),
                 variation_c=np.array([self.variations[n] for n in vn]).reshape(len(vn), len(self.c)))


def _pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f'none of {keys} in the targets file')


def load_targets(path, point=(0, 0)):
    """Read a calculation to impose, from .npz or .json.

    The fields:
        moment_names     the k functions; for thrust tau^a ln^b tau, named as in THRUST_MOMENTS
        moment_powers    optional, (k, 2): the (a, b) of each function, which then makes the
                         names free labels
        window           [low, high), in tau = 1 - T for thrust
        central          the k calculated window-conditional means c
        sigma            their k by k covariance, diag(sd^2) for uncorrelated errors
        floor_rel        optional, the relative floor impose() adds to sigma, 0.002 if absent
        variations       optional, named replacements for central; in .json {name: [k values]},
                         in .npz variation_names (V,) and variation_c (V, k)

    The paper's deposit, targets_thrust_nlo.npz, is such a file. It spells the covariance
    sigma_pert and stores its variations as scale_labels with scale_variations and np_labels with
    np_variations; those are read as variations named by their labels, for example
    "(2.0, 1.0, 'logR', 1.0)" or 'milan_up'. The training code's own grid file
    (thrust_targets_nlo_point.npz: keys, central and sigma_pert on a grid of alpha_s and alpha_0)
    is read too, at the grid node point."""
    if str(path).endswith('.json'):
        with open(path) as f:
            d = json.load(f)
    else:
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
    names = _pick(d, 'moment_names', 'names', 'keys')
    c = np.asarray(_pick(d, 'central', 'c'), np.float64)
    sigma = np.asarray(_pick(d, 'sigma', 'sigma_pert'), np.float64)
    grid = c.ndim == 3
    at = (lambda a: np.asarray(a)[point[0], point[1]]) if grid else (lambda a: np.asarray(a))
    var = {}
    if isinstance(d.get('variations'), dict):
        var.update(d['variations'])
    if 'variation_names' in d:
        var.update({str(n): d['variation_c'][v] for v, n in enumerate(d['variation_names'])})
    for labels, values in (('scale_labels', ('scale_variations', 'scale')),
                           ('np_labels', ('np_variations', 'np_vars'))):
        if labels in d:
            vals = at(_pick(d, *values))
            var.update({str(lab): vals[v] for v, lab in enumerate(d[labels])})
    meta = {}
    for out, keys in (('alpha_s', ('alpha_s', 'alphas')), ('alpha_0', ('alpha_0', 'alpha0'))):
        for k in keys:
            if k in d:
                v = np.ravel(d[k])
                meta[out] = float(v[point[0] if out == 'alpha_s' else point[1]] if grid else v[0])
                break
    if 'pwin' in d:
        meta['pwin'] = float(at(d['pwin'])) if grid else float(np.ravel(d['pwin'])[0])
    return Targets(names, at(c), at(sigma), d['window'], var,
                   floor_rel=float(np.ravel(d['floor_rel'])[0]) if 'floor_rel' in d else FLOOR_REL,
                   powers=d.get('moment_powers'), meta=meta)
