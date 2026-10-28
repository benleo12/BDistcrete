"""Readers for the exported reweighting heads. Pure numpy, no torch.

An exported head is a factorized log-ratio,

    f(Phi; theta) = < a(Phi), b(theta) > ,

averaged over the ensemble members. a(Phi) is the event-side vector produced by the trunk
(shipped precomputed for the reference sample in the export, or computed for your own events
with gentune.embed), and b(theta) is a small network of the generator parameters, which is
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

The learning band of the paper, the uncertainty of the log ratio itself, comes from the WiFi fit
of the ensemble (Benevedes and Thaler, arXiv:2506.00113), whose weights and covariance ship for
MIX17 as models/MIX17_wifi.npz. wifi_basis(A, theta) gives its basis functions and
learning_band(A, theta, O, wifi_path) the band of a weighted mean.
"""
import os
import numpy as np


# head_kind of the two mixture exports: 'mixture' is the additive (1 - f) q_S + f q_H of the
# earlier release, 'geometric' the multiplicative q_S^(1-f) q_H^f of the paper's MIX17. Every
# other export is a single-generator head.
MIXTURE_KINDS = ('mixture', 'geometric')

def sigm(z):
    return 0.5*(1.0 + np.tanh(0.5*z))         # logistic, written so large |z| cannot overflow


class Head:
    """A single-generator head: Stage C, Stage E (Sherpa) or Stage F (Herwig)."""

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
        parameters at the centre of their box. About one point in a thousand drawn uniformly
        in the box of E, F or MIX17 fails, a corner further from the design than any training
        point. A head whose design does not fill its box fails far more often, which is the
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
    def pack(self, A):
        """Put event-side vectors in the (Nev, ENS*K) layout the matrix product expects.
        Accepts either that layout or the (ENS, Nev, K) layout the export stores."""
        A = np.asarray(A)
        if A.ndim == 3:
            assert A.shape[0] == self.ENS and A.shape[2] == self.K, \
                f'expected (ENS={self.ENS}, Nev, K={self.K}), got {A.shape}'
            return np.ascontiguousarray(A.transpose(1, 0, 2).reshape(A.shape[1], self.ENS*self.K))
        assert A.ndim == 2 and A.shape[1] == self.ENS*self.K, \
            f'expected (Nev, {self.ENS*self.K}), got {A.shape}'
        return A

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
                      counts=None):
        """The learning band of the weighted mean of O at theta, sqrt(g^T C g).

        C is the covariance of the WiFi weights in wifi_path, and g_k, the derivative of the
        mean with respect to w_k at w_published, is the weighted covariance of O with the basis
        function k of wifi_basis(). The temperature sits inside w_published, so no separate
        factor of 1/T enters g, and the constant has no covariance with anything. This is the
        band the paper quotes for the generator alone, before a calculation is imposed.

        The weights are those of weights(A, theta), so A is the reference sample. For a sample
        generated at theta_from pass it, which takes the weights of ratio_weights() and the
        difference of the two bases. O is one value per event, or (Nev, n) for n observables at
        once, which returns n bands. mask restricts the mean to a subset of the events,
        normalized over that subset: the mean of 1 - T inside a window is O = 1 - T with the
        window as mask, and a bin of a distribution normalized over a set of bins is
        O = (in the bin)/(bin width) with mask = (in any of the bins). counts, one number per
        event, multiplies the weights, which for the counts of gentune.bands.resample_counts()
        gives the band on a bootstrap resample of the events.
        """
        W = np.load(wifi_path)
        want = (self.ENS, self.K, getattr(self, 'kind', 'cond'))
        got = (int(W['ens']), int(W['K']), str(W['kind']))
        if got != want:
            raise ValueError(f'{wifi_path} was fitted to a head with (members, K, kind) = {got} '
                             f'and {self.path} has {want}, so its weights do not belong to this '
                             f'head')
        F = self.wifi_basis(A, theta, check_box)
        if theta_from is not None:
            F = F - self.wifi_basis(A, theta_from, check_box)
        O = np.asarray(O, np.float64)
        if O.ndim not in (1, 2) or len(O) != len(F):
            raise ValueError(f'O has shape {O.shape} and A has {len(F)} events, so O needs one row '
                             f'per event')
        m = np.ones(len(F), bool) if mask is None else np.asarray(mask, bool)
        z = F[m] @ np.asarray(W['w_published'], np.float64)
        w = np.exp(z - z.max())
        if counts is not None:
            w = w*np.asarray(counts, np.float64)[m]
        w /= w.sum()
        X = O[m].reshape(len(w), -1)
        g = (w[:, None]*(X - w @ X)).T @ F[m]                 # (n, 1 + ENS^2), Cov_w(O, f_k)
        band = np.sqrt(np.maximum(np.einsum('ak,kl,al->a', g, W['C'], g), 0.0))
        return float(band[0]) if O.ndim == 1 else band

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

    theta = (theta_S, theta_H, f), where theta_S are the first generator's parameters, theta_H
    the second generator's, and f in [0, 1] the fraction of the second. Per ensemble member

        l_S = < a(Phi), b_S(theta_S) > ,   l_H = < a(Phi), b_H(theta_H) > ,

    and the member's logit is

        f_m = (1 - f) l_S + f l_H                       (head_kind 'geometric', MIX17)
        f_m = log[ (1 - f) exp(l_S) + f exp(l_H) ]      (head_kind 'mixture', additive)

    for the multiplicative mixture q_S^(1-f) q_H^f and the additive one (1 - f) q_S + f q_H.
    The ensemble logit is the mean of f_m. Both forms are exact at the edges: the derivative
    with respect to theta_H vanishes identically at f = 0 and the one with respect to theta_S
    at f = 1, so a fraction of zero is the first generator alone and not an approximation to it.

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
            bS = self._bvec_net(self.W[m], self.b[m], tS)
            bH = self._bvec_net(self.WH[m], self.bH[m], tH)
            sl = slice(m*self.K, (m + 1)*self.K)
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
