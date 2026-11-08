"""Readers for the exported reweighting heads. Pure numpy, no torch.

An exported head is a factorized log-ratio,

    f(Phi; theta) = < a(Phi), b(theta) > ,

averaged over the ensemble members. a(Phi) is the event-side vector produced by the trunk
(shipped precomputed for the reference sample in the export, or computed for your own events
with gentune.Trunk.embed), and b(theta) is a small network of the generator parameters, which is
what these classes evaluate.

Two operations are useful, and they are not the same thing.

  weights(A, theta)
      Reweights the REFERENCE sample to theta. The reference is the pooled union of all
      training runs, so this is only correct on reference events, which means the events
      shipped with the export or events you generated with the same pooled configuration.

  ratio_weights(A, theta_to, theta_from)
      Reweights a sample generated at theta_from to theta_to. Both logits share the same
      reference, so their difference is log[q(Phi; theta_to)/q(Phi; theta_from)] with the
      reference cancelled exactly. This is the operation you want for your own sample: pass
      the parameter point your sample was generated at as theta_from. It is exact for any
      pair of points inside the training box and needs no reference events at all.

Weights are returned normalized to sum to one over the events you passed in.

MIX17 holds both generators. The paper evaluates its head only at the fraction xi = 0 (Sherpa
alone) and xi = 1 (Herwig alone) and mixes the two predictions, (1 - xi) <O>_S + xi <O>_H:
gentune.mixture does this, and gives the paper's prediction at every point of the box.

The learning band of the paper, the uncertainty of the log ratio itself, comes from the WiFi fit
of the ensemble (Benevedes and Thaler, arXiv:2506.00113), whose weights and covariance ship for
MIX17 as models/MIX17_wifi.npz. wifi_basis(A, theta) gives its basis functions and
learning_band(A, theta, O, wifi_path) the band of a weighted mean of one generator, with
imposed=(inwin, F, info) of gentune.maxent.impose that of a prediction with the calculation
imposed; gentune.mixture.learning_band() gives that of the mixture.
"""
import os
import warnings
import numpy as np


# head_kind of the two kinds of mixture export, by how the network combines the generators at
# an interior fraction f: 'geometric', q_S^(1-f) q_H^f, is MIX17; 'mixture', the logarithm of
# (1 - f) q_S + f q_H, is an export of an earlier draft. At f = 0 and f = 1 both are one generator
# alone, which is all the paper uses (gentune.mixture). Every other export is a single-generator head.
MIXTURE_KINDS = ('mixture', 'geometric')


class GeometricFractionWarning(UserWarning):
    """MixtureHead.weights or ratio_weights was called at a fraction strictly between 0 and 1."""


def sigm(z):
    return 0.5*(1.0 + np.tanh(0.5*z))         # logistic, written so large |z| cannot overflow


class Head:
    """A single-generator head, such as `E` (Sherpa) or `F` (Herwig)."""

    def __init__(self, path):
        c = np.load(path)
        # The training code names the single-generator head 'cond'. Exports made before that
        # field existed carry no head_kind at all. Both are this class, and only the mixture is not,
        # so test for what is excluded rather than for one accepted spelling.
        kind = str(c.get('head_kind', 'cond'))
        assert kind not in MIXTURE_KINDS, f'{path} is a mixture export, use MixtureHead or load_head'
        self.path = path
        self.mtime = os.path.getmtime(path)
        self.K = int(c['K'])                   # contraction width, includes any additive columns
        self.K_bilinear = int(c.get('K_bilinear', c['K']))
        self.nt = int(c['ntheta'])
        self.ENS = int(c['ens'])
        self.T = float(c['temperature'])       # calibration temperature, applied to every logit
        self.norm = c['norm'].astype(np.float64)      # (nt, 2) columns [centre, half width]
        self.act = str(c['act'])
        assert self.act in ('silu', 'relu'), f'unknown activation {self.act}'
        L = int(c['nlayers'])
        self.W = [[c[f'B{m}_W{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.b = [[c[f'B{m}_b{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.feat_mu = c['feat_mu'].astype(np.float32)
        self.feat_sd = c['feat_sd'].astype(np.float32)
        self.AE = c['AE'] if 'AE' in c.files else None    # (ENS, Nref, K), the reference sample
        self.checksum = c['f_checksum'] if 'f_checksum' in c.files else None
        self.checksum_theta = c['f_checksum_theta'] if 'f_checksum_theta' in c.files else None
        # The training design, standardized. A box is not the same thing as the region that
        # was sampled, and support() measures the difference against these points.
        self.theta_train = c['theta_train'].astype(np.float64) if 'theta_train' in c.files else None
        self._spacing = None

    # -- the box the head was trained in -----------------------------------------------------
    @property
    def centre(self):
        """Centre of the training box, one entry per parameter."""
        return self.norm[:, 0].copy()

    @property
    def box(self):
        """(low, high) per parameter. The head is an interpolant and is not to be trusted
        outside this box, which is why in_box() is checked on every evaluation."""
        return np.stack([self.norm[:, 0] - self.norm[:, 1], self.norm[:, 0] + self.norm[:, 1]], 1)

    def in_box(self, theta, tol=1e-9):
        t = self._vec(theta)
        lo, hi = self.box[:, 0], self.box[:, 1]
        return bool(np.all(t >= lo - tol) and np.all(t <= hi + tol))

    def tn(self, theta):
        t = self._vec(theta)
        return (t - self.norm[:, 0])/self.norm[:, 1]

    def _names(self):
        """The parameter names gentune.axes holds for this file's stage, read off the file name,
        or None when the file is not a released stage."""
        tag = os.path.basename(str(getattr(self, 'path', ''))).replace('_head.npz', '')
        try:
            from . import axes
            return axes.names(tag, self.nt) if tag in getattr(axes, 'NAMES', {}) else None
        except (ImportError, ValueError):
            return None

    def _vec(self, theta):
        """theta as float64, after checking it holds one value per parameter. A wrong count used
        to fail inside in_box() as a numpy broadcasting error, and a single number was compared
        with every row of the box and could pass."""
        t = np.asarray(theta, np.float64)
        if t.ndim == 0 or t.shape[-1] != self.nt:
            nm = self._names()
            order = f', in the order {", ".join(nm)}' if nm else ''
            n = 1 if t.ndim == 0 else t.shape[-1]
            raise ValueError(f'theta has {n} value{"" if n == 1 else "s"} and '
                             f'{getattr(self, "path", "this head")} takes {self.nt}{order}')
        return t

    def _box_error(self, theta, tol=1e-9):
        """The ValueError for a point outside the box, naming each parameter that is out."""
        rows = self._vec(theta).reshape(-1, self.nt)
        lo, hi = self.box[:, 0], self.box[:, 1]
        out = ~((rows >= lo - tol) & (rows <= hi + tol))          # also catches nan
        r = int(np.argmax(out.any(1)))
        nm = self._names() or tuple(f'parameter {i}' for i in range(self.nt))
        lines = [f'  [{i:>2}] {nm[i]} = {rows[r, i]:.6g}, the box runs from {lo[i]:.6g} to '
                 f'{hi[i]:.6g}' for i in np.where(out[r])[0]]
        which = f'point {r} of theta' if len(rows) > 1 else 'theta'
        return ValueError(f'{which} is outside the training box of {self.path}, where the head is '
                          f'not to be trusted:\n' + '\n'.join(lines))

    def support(self, theta):
        """Was this point sampled, or is the head extrapolating?

        Being inside the box is necessary and not sufficient. Both eight-parameter designs of
        this release fill their boxes: each holds the points of the earlier design of its
        generator and 256 points of a scrambled Sobol sequence over the whole box. Only the
        first Herwig design followed the published retunes, which move the shower coupling with
        the shower cutoff and the cluster fission mass with its power, and so covered a
        six-dimensional locus. The Sobol points fill the rest. The training runs of MIX17 are
        pure runs of one generator, at a fraction of 0 or 1, with the other generator's
        parameters at the centre of their box. About two points in a thousand drawn uniformly
        in the box of E or MIX17 fail, and fewer for F, a corner further from the design than any
        training point. A head whose design does not fill its box fails far more often, which is the
        case this test is for.

        The test is the design's own covariance. Rotating into its principal directions and
        dividing by the spread along each one gives a score, so a point far out along a
        direction the design covers thinly scores high. The score is compared with the
        largest score any TRAINING point reaches, which is the model's own definition of how
        far out it has seen.

        Returns a dict with score, threshold, ok, and the nearest training point distance.
        Returns None if the design was not shipped with this head.
        """
        if self.theta_train is None:
            return None
        t = self.tn(theta)
        if self._spacing is None:
            X = self.theta_train
            mu = X.mean(0)
            ev, V = np.linalg.eigh(np.cov((X - mu).T))
            sd = np.sqrt(np.maximum(ev, 1e-12))
            self._spacing = (mu, V, sd, float(np.max([self._score(x, mu, V, sd) for x in X])))
        mu, V, sd, thr = self._spacing
        sc = self._score(t, mu, V, sd)
        d = float(np.min(np.linalg.norm(self.theta_train - t, axis=-1)))
        return dict(score=sc, threshold=thr, ok=bool(sc <= thr), nearest=d)

    @staticmethod
    def _score(t, mu, V, sd):
        return float(np.max(np.abs(V.T @ (np.asarray(t, np.float64) - mu))/sd))

    def check(self, theta, warn=True):
        """in_box plus support, as one call. Raises if outside the box, returns the support
        dict and prints a warning if the point is further out than anything trained on."""
        if not self.in_box(theta):
            raise self._box_error(theta)
        s = self.support(theta)
        if s is not None and not s['ok'] and warn:
            print(f'WARNING this point is inside the box but outside the region that was '
                  f'generated (score {s["score"]:.2f} against {s["threshold"]:.2f} for the '
                  f'training design). The weights are an extrapolation and the closure '
                  f'numbers quoted for this stage do not cover it.')
        return s

    # -- b(theta) ----------------------------------------------------------------------------
    def _bvec(self, m, thn):
        x = np.asarray(thn, np.float64)
        Ws, bs = self.W[m], self.b[m]
        L = len(Ws)
        for l in range(L):
            z = Ws[l] @ x + bs[l]
            if l < L - 1:
                x = z*sigm(z) if self.act == 'silu' else np.maximum(z, 0.0)
            else:
                x = z
        return x

    def bcat(self, theta):
        """The ensemble's b vectors concatenated, so one matrix product covers all members."""
        thn = self.tn(theta)
        return np.concatenate([self._bvec(m, thn) for m in range(self.ENS)])

    # -- logits and weights ------------------------------------------------------------------
    def pack(self, A, dtype=None):
        """Put event-side vectors in the (Nev, ENS*K) layout the matrix product expects.
        Accepts either that layout or the (ENS, Nev, K) layout the export stores. dtype, for
        example np.float64, also converts them, member by member, so that a loop over many points
        does not convert the float32 file at every evaluation: A = head.pack(r['AE'], np.float64)
        once, then head.weights(A, theta) or gentune.mixture.weights(head, A, theta) at each."""
        A = np.asarray(A)
        if A.ndim == 3:
            assert A.shape[0] == self.ENS and A.shape[2] == self.K, \
                f'expected (ENS={self.ENS}, Nev, K={self.K}), got {A.shape}'
            if dtype is None:
                return np.ascontiguousarray(A.transpose(1, 0, 2).reshape(A.shape[1], self.ENS*self.K))
            out = np.empty((A.shape[1], self.ENS*self.K), dtype)
            for m in range(self.ENS):
                out[:, m*self.K:(m + 1)*self.K] = A[m]
            return out
        assert A.ndim == 2 and A.shape[1] == self.ENS*self.K, \
            f'expected (Nev, {self.ENS*self.K}), got {A.shape}'
        return A if dtype is None else np.asarray(A, dtype)

    def logit(self, A, theta, check_box=True):
        """f(Phi; theta) per event, ensemble mean of the member logits, before temperature."""
        if check_box and not self.in_box(theta):
            raise self._box_error(theta)
        return self.pack(A) @ (self.bcat(theta)/self.ENS)

    def weights(self, A, theta, check_box=True):
        """Weights that take the REFERENCE sample to theta. See the module docstring."""
        f = self.logit(A, theta, check_box)/self.T
        w = np.exp(f - f.max())
        return w/w.sum()

    def ratio_weights(self, A, theta_to, theta_from, check_box=True):
        """Weights that take a sample generated at theta_from to theta_to. The reference
        cancels between the two logits, so this needs no reference events."""
        f = (self.logit(A, theta_to, check_box) - self.logit(A, theta_from, check_box))/self.T
        w = np.exp(f - f.max())
        return w/w.sum()

    def n_eff(self, w):
        """Effective sample size of a normalized weight set. A number far below len(w) means
        the reweighting is being carried by a few events and its errors will be large."""
        w = np.asarray(w, np.float64)
        return float(1.0/np.sum((w/w.sum())**2))

    # -- the learning band -------------------------------------------------------------------
    def wifi_basis(self, A, theta, check_box=True):
        """The basis functions of the WiFi fit at theta, one row per event, (Nev, 1 + ENS^2).

        Column 0 is the constant and column 1 + i*ENS + j is < a_i(Phi), b_j(theta) >, the event
        vector of member i against the parameter vector of member j. That is the order of the
        weights in models/<stage>_wifi.npz, w_0 first and then w_ij row by row, so the log ratio
        of the fit is basis @ w. The published network is the diagonal, w_ij = delta_ij/(ENS T),
        so basis @ w_published is logit(A, theta)/T and gives the weights of weights(A, theta).
        """
        if check_box and not self.in_box(theta):
            raise self._box_error(theta)
        a = self.pack(A).reshape(-1, self.ENS, self.K)
        f = np.einsum('nik,jk->nij', a, self.bcat(theta).reshape(self.ENS, self.K))
        return np.concatenate([np.ones((len(a), 1)), f.reshape(len(a), self.ENS**2)], 1)

    def learning_band(self, A, theta, O, wifi_path, mask=None, theta_from=None, check_box=True,
                      counts=None, imposed=None):
        """The learning band of the weighted mean of O at theta, sqrt(g^T C g).

        C is the covariance of the WiFi weights in wifi_path, and g_k, the derivative of the
        mean with respect to w_k at w_published, is the weighted covariance of O with the basis
        function k of wifi_basis(). w_published is the published network at this head's
        temperature, w_0 = 0 and w_ij = delta_ij/(ENS T), as in the paper's code; a WiFi file
        written at another temperature keeps its covariance, and a warning says its own
        w_published is not used. The temperature sits inside w_published, so no separate factor
        of 1/T enters g, and the constant has no covariance with anything. This is the band of one
        generator; for MIX17 the paper quotes that of its mixture of the two generators, which
        gentune.mixture.learning_band() gives.

        The weights are those of weights(A, theta), so A is the reference sample. For a sample
        generated at theta_from pass it, which takes the weights of ratio_weights() and the
        difference of the two bases. O is one value per event, or (Nev, n) for n observables at
        once, which returns n bands. A bin of a distribution, the fraction of all events in it over
        its width, is O = (in the bin)/(bin width). mask restricts the mean to a subset of the
        events, normalized over that subset: the mean of 1 - T inside a window is O = 1 - T with
        the window as mask. counts, one number per event, multiplies the weights, which for the
        counts of gentune.bands.resample_counts() gives the band on a bootstrap resample of the
        events.

        Without imposed, the band of the generator alone. With imposed=(inwin, F, info), the
        (inwin, F, info) of the maxent.impose() that imposed a calculation on these weights, g is
        taken through the multipliers solved again, as the paper does: a change of
        the WiFi weights moves the multipliers by -J^-1 times its change of the calculation's
        condition, which includes the change of the window fraction before the tilt in P^2 Sigma,
        as in the paper's code (analysis/three_bands.py with directlib.tilt_on), so
        g = Cov_w(r, f) - 2 P (J^-1 Cov_w(F, O)) . (Sigma lam) Cov_w0(Theta, f), with r the
        projected per-event term of gentune.bands.mean_and_error. counts is not taken with imposed.
        """
        W = np.load(wifi_path)
        self._check_wifi(W, wifi_path)
        g = self._learning_gradient(A, theta, O, W, mask, theta_from, check_box, counts, imposed,
                                    wifi_path)
        band = np.sqrt(np.maximum(np.einsum('ak,kl,al->a', g, W['C'], g), 0.0))
        return float(band[0]) if np.asarray(O).ndim == 1 else band

    def _check_wifi(self, W, wifi_path):
        """Raise unless the WiFi file W was fitted to a head of this shape and kind."""
        want = (self.ENS, self.K, getattr(self, 'kind', 'cond'))
        got = (int(W['ens']), int(W['K']), str(W['kind']))
        if got != want:
            raise ValueError(f'{wifi_path} was fitted to a head with (members, K, kind) = {got} '
                             f'and {self.path} has {want}, so its weights do not belong to this '
                             f'head')

    def _w_published(self, W, wifi_path=None):
        """The WiFi weights of the published network at this head's temperature, w_0 = 0 and
        w_ij = delta_ij/(ENS T), where the learning band is taken (directlib.wifi_point). A WiFi
        file written at another temperature, before a refit of the temperature, holds another
        w_published; its covariance does not depend on the temperature and is kept, and a warning
        says that the file's w_published is not used."""
        w = np.zeros(self.ENS*self.ENS + 1)
        w[1:] = np.eye(self.ENS).ravel()/(self.ENS*self.T)
        wf = np.asarray(W['w_published'], np.float64) if 'w_published' in W.files else None
        if wf is None or wf.shape != w.shape or not np.array_equal(wf, w):
            key = str(wifi_path)
            seen = self.__dict__.setdefault('_wifi_warned', set())
            if key not in seen:
                seen.add(key)
                t = (1.0/(self.ENS*wf[1]) if wf is not None and wf.shape == w.shape and wf[1] != 0
                     else float('nan'))
                warnings.warn(f'{wifi_path} holds the published network at the temperature '
                              f'T = {t!r} and {self.path} carries T = {self.T!r}: the learning band '
                              f'is taken at the temperature of the head, as in the paper, with the '
                              f'covariance of the file', stacklevel=4)
        return w

    def _learning_gradient(self, A, theta, O, W, mask=None, theta_from=None, check_box=True,
                           counts=None, imposed=None, wifi_path=None):
        """g, (n, 1 + ENS^2): the derivative of the weighted mean of each of the n columns of O
        with respect to the WiFi weights of W, at w_published (see learning_band), with imposed
        through the re-solved multipliers."""
        F = self.wifi_basis(A, theta, check_box)
        if theta_from is not None:
            F = F - self.wifi_basis(A, theta_from, check_box)
        O = np.asarray(O, np.float64)
        if O.ndim not in (1, 2) or len(O) != len(F):
            raise ValueError(f'O has shape {O.shape} and A has {len(F)} events, so O needs one row '
                             f'per event')
        wp = self._w_published(W, wifi_path)
        if imposed is None:
            m = np.ones(len(F), bool) if mask is None else np.asarray(mask, bool)
            z = F[m] @ wp
            w = np.exp(z - z.max())
            if counts is not None:
                w = w*np.asarray(counts, np.float64)[m]
            w /= w.sum()
            X = O[m].reshape(len(w), -1)
            return (w[:, None]*(X - w @ X)).T @ F[m]             # (n, 1 + ENS^2), Cov_w(O, f_k)
        if counts is not None:
            raise ValueError('counts and imposed together are not taken: on a resample the '
                             'multipliers would have to be solved again first')
        from .maxent import _Response, _tilted
        z = F @ wp
        w0 = np.exp(z - z.max())
        w0 /= w0.sum()
        w = np.exp(_tilted(z, imposed))                         # the imposed weights
        R = _Response(w, imposed)
        X = O.reshape(len(F), -1)
        if mask is None:
            r = X - w @ X
        else:
            m = np.asarray(mask, bool)
            P = w[m].sum()
            r = np.where(m[:, None], X - (w[m] @ X[m])/P, 0.0)/P
        r, v = R.remove(r)
        th = np.zeros(len(F))
        th[R.iw] = 1.0
        dP = (w0*(th - w0[R.iw].sum())) @ F                      # Cov_w0(Theta, f_k)
        return (w[:, None]*r).T @ F - 2.0*R.P*np.outer(v.T @ (R.S @ R.lam), dP)

    def check_trunk(self, trunk, tol=1e-6):
        """Raise unless this head and that trunk come from the same trained stage.

        A head and a trunk are two halves of one network. Nothing in their shapes distinguishes
        one stage's trunk from another's when the rank and the ensemble size agree, which they do
        for the two eight-parameter stages, so pairing an E head with an F trunk used to sail
        through every assertion and return plausible weights that were wrong by several times the
        quoted accuracy floor. The per-feature standardization is fitted per stage, so comparing
        it is enough to tell them apart.
        """
        for name in ('feat_mu', 'feat_sd'):
            a = getattr(self, name, None)
            b = getattr(trunk, name, None)
            if a is None or b is None:
                raise ValueError(f'cannot pair head and trunk: {name} is missing from one of them')
            a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=tol):
                raise ValueError(
                    f'this head and this trunk are from different trained stages: their {name} '
                    f'differ by up to {np.abs(a - b).max() if a.shape == b.shape else float("nan"):.3e}. '
                    f'Load the trunk whose name matches the head, for example E_head.npz with '
                    f'E_trunk.npz.')
        return True

    def replay_checksum(self):
        """Reproduce the 256 logits recorded at export time. This is the one check that says
        the reader and the trained model agree. Any deviation above float32 rounding means
        the export and this code have drifted apart."""
        if self.checksum is None or self.AE is None:
            # A released head ships without AE, the cached a(Phi) of the reference sample, since
            # that is hundreds of megabytes and is only needed to reweight the reference itself.
            # Guarding on checksum alone left this raising TypeError on every shipped head.
            return None
        f = self.logit(self.AE[:, :256, :], self.checksum_theta, check_box=False)
        return float(np.max(np.abs(f - self.checksum)))


class MixtureHead(Head):
    """The mixture head: two generators, their own parameters, and the fraction between them.

    theta = (theta_S, theta_H, f), where theta_S are Sherpa's parameters, theta_H Herwig's, and
    f in [0, 1] the fraction of Herwig. At f = 0 the head is Sherpa alone at theta_S, and at
    f = 1 Herwig alone at theta_H. The paper uses it there and nowhere else: its prediction at
    theta is (1 - f) <O>_S + f <O>_H, each generator with its own weights, and gentune.mixture
    computes it (mixture.weights, mixture.mean).

    At an interior f the network has a value of its own, the interpolation built into it; it was
    trained on runs at f = 0 and f = 1 only. Per ensemble member

        l_S = < a(Phi), b_S(theta_S) > ,   l_H = < a(Phi), b_H(theta_H) > ,

    and the member's logit is

        f_m = (1 - f) l_S + f l_H                       (head_kind 'geometric', MIX17)
        f_m = log[ (1 - f) exp(l_S) + f exp(l_H) ]      (head_kind 'mixture')

    which reweight to q_S^(1-f) q_H^f and to (1 - f) q_S + f q_H as single densities. The
    ensemble logit is the mean of f_m. Both forms are exact at the edges: the derivative with
    respect to theta_H vanishes identically at f = 0 and the one with respect to theta_S at
    f = 1, so a fraction of zero is the first generator alone and not an approximation to it.
    There logit() evaluates only that generator's network; with check_box it still wants the other
    generator's parameters inside the box, which gentune.mixture does not ask. weights() and
    ratio_weights() return this interpolation at an interior f, with a
    GeometricFractionWarning, since it is not the paper's prediction there; logit() returns it
    without one, as the self-test and the WiFi basis need it.

    The two parameter blocks are read from the shapes of the exported networks rather than
    assumed to be equal in size, so a mixture of generators with different parameter counts
    needs no change here.
    """

    def __init__(self, path):
        c = np.load(path)
        self.kind = str(c.get('head_kind', 'cond'))
        assert self.kind in ('mixture', 'geometric'), \
            f'{path} is not a mixture export, use Head'
        self.path = path
        self.mtime = os.path.getmtime(path)
        self.K = int(c['K'])
        self.K_bilinear = int(c.get('K_bilinear', c['K']))
        self.nt = int(c['ntheta'])
        self.ENS = int(c['ens'])
        self.T = float(c['temperature'])
        self.norm = c['norm'].astype(np.float64)
        self.act = str(c['act'])
        assert self.act in ('silu', 'relu'), f'unknown activation {self.act}'
        L = int(c['nlayers'])
        self.W = [[c[f'B{m}_W{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.b = [[c[f'B{m}_b{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.WH = [[c[f'BH{m}_W{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.bH = [[c[f'BH{m}_b{l}'].astype(np.float64) for l in range(L)] for m in range(self.ENS)]
        self.nS = int(self.W[0][0].shape[1])
        self.nH = int(self.WH[0][0].shape[1])
        assert self.nS + self.nH + 1 == self.nt, \
            f'{self.nS} + {self.nH} + 1 does not make {self.nt} parameters'
        self.feat_mu = c['feat_mu'].astype(np.float32)
        self.feat_sd = c['feat_sd'].astype(np.float32)
        self.AE = c['AE'] if 'AE' in c.files else None
        self.checksum = c['f_checksum'] if 'f_checksum' in c.files else None
        self.checksum_theta = c['f_checksum_theta'] if 'f_checksum_theta' in c.files else None
        self.theta_train = c['theta_train'].astype(np.float64) if 'theta_train' in c.files else None
        self._spacing = None

    def _warn_interior(self, *thetas):
        """Warn once per head when weights come from the network at an interior fraction."""
        if getattr(self, '_warned', False):
            return
        fr = [float(np.asarray(t, np.float64).ravel()[-1]) for t in thetas]
        if any(0.0 < f < 1.0 for f in fr):
            self._warned = True
            warnings.warn(f'{self.path}: weights at the fraction {", ".join(f"{f:g}" for f in fr)} '
                          f'come from the network\'s own interpolation between the generators, which '
                          f'is not the paper\'s prediction. The paper mixes the two generators\' '
                          f'predictions: gentune.mixture.weights(head, A, theta) and '
                          f'gentune.mixture.mean().', GeometricFractionWarning, stacklevel=3)

    def weights(self, A, theta, check_box=True):
        """Weights that take the REFERENCE sample to theta, the network's own value. At an interior
        fraction that is the interpolation of the class docstring, not the paper's prediction,
        which gentune.mixture.weights() gives."""
        self._warn_interior(theta)
        return super().weights(A, theta, check_box)

    def ratio_weights(self, A, theta_to, theta_from, check_box=True):
        """ratio_weights of Head, the network's own value; see weights() for an interior fraction."""
        self._warn_interior(theta_to, theta_from)
        return super().ratio_weights(A, theta_to, theta_from, check_box)

    def split(self, theta):
        """(theta_S, theta_H, f) from one physical parameter vector.

        theta_S and theta_H come back STANDARDIZED, which is what the two parameter networks
        read. f comes back PHYSICAL, on [0, 1], because that is what f means everywhere else in
        this class and in the paper. It used to be returned standardized along with the rest, so
        a pure second-generator point read back as +1 and a half-and-half mixture as 0, which is
        wrong for the one component a caller is most likely to want.
        """
        t = self.tn(theta)
        f_phys = float(np.asarray(theta, np.float64).ravel()[-1])
        return t[:self.nS], t[self.nS:self.nS + self.nH], f_phys

    def _bvec_net(self, Ws, bs, x):
        L = len(Ws)
        for l in range(L):
            z = Ws[l] @ np.asarray(x, np.float64) + bs[l]
            x = (z*sigm(z) if self.act == 'silu' else np.maximum(z, 0.0)) if l < L - 1 else z
        return x

    def bcat(self, theta):
        """The members' b vectors concatenated, as for a single generator. The geometric mixture
        is an inner product in theta, with b_m = (1 - f) b_S,m(theta_S) + f b_H,m(theta_H) per
        member, and the additive one is not, so it has no such vector."""
        if self.kind != 'geometric':
            raise ValueError(f'{self.path} is an additive mixture, whose log ratio is not an inner '
                             f'product < a, b(theta) >, so it has no b vector and no WiFi basis')
        tS, tH, fr = self.split(theta)
        return np.concatenate([(1.0 - fr)*self._bvec_net(self.W[m], self.b[m], tS)
                               + fr*self._bvec_net(self.WH[m], self.bH[m], tH)
                               for m in range(self.ENS)])

    def logit(self, A, theta, check_box=True):
        if check_box and not self.in_box(theta):
            raise self._box_error(theta)
        Ap = self.pack(A)
        tS, tH, _ = self.split(theta)
        # the fraction is a physical number in [0, 1] and enters the mixture unstandardized,
        # so it is taken from theta itself rather than from the standardized vector
        fr = float(np.asarray(theta, np.float64)[-1])
        acc = np.zeros(Ap.shape[0])
        for m in range(self.ENS):
            sl = slice(m*self.K, (m + 1)*self.K)
            # at the fraction 0 or 1 both forms are one generator alone, exactly, so only its
            # network is evaluated (for MIX17 the same numbers as the general form below)
            if fr == 0.0:
                acc += Ap[:, sl] @ self._bvec_net(self.W[m], self.b[m], tS)
                continue
            if fr == 1.0:
                acc += Ap[:, sl] @ self._bvec_net(self.WH[m], self.bH[m], tH)
                continue
            bS = self._bvec_net(self.W[m], self.b[m], tS)
            bH = self._bvec_net(self.WH[m], self.bH[m], tH)
            lS = Ap[:, sl] @ bS
            lH = Ap[:, sl] @ bH
            if self.kind == 'geometric':
                # the multiplicative mixture q_S^(1-f) q_H^f: the log ratio is linear in f
                acc += (1.0 - fr)*lS + fr*lH
                continue
            # log-sum-exp with the larger exponent pulled out, so neither term can overflow
            big = np.maximum(lS, lH)
            acc += big + np.log(np.clip((1 - fr)*np.exp(lS - big) + fr*np.exp(lH - big), 1e-300, None))
        return acc/self.ENS


def load_head(path):
    """The head class an export needs, read from its head_kind."""
    kind = str(np.load(path).get('head_kind', 'cond'))
    return (MixtureHead if kind in MIXTURE_KINDS else Head)(path)
