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
"""
import os
import numpy as np


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
        assert kind != 'mixture', f'{path} is a mixture export, use MixtureHead'
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
        # was sampled: the Herwig stage varies eight parameters along a six-dimensional locus
        # taken from published retunes, so the corners of its bounding box were never
        # generated and a point there is an extrapolation even though in_box() passes.
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
        t = np.asarray(theta, np.float64)
        lo, hi = self.box[:, 0], self.box[:, 1]
        return bool(np.all(t >= lo - tol) and np.all(t <= hi + tol))

    def tn(self, theta):
        t = np.asarray(theta, np.float64)
        assert t.shape[-1] == self.nt, f'expected {self.nt} parameters, got {t.shape[-1]}'
        return (t - self.norm[:, 0])/self.norm[:, 1]

    def support(self, theta):
        """Was this point sampled, or is the head extrapolating?

        Being inside the box is necessary and not sufficient. The Sherpa stage varies eight
        parameters independently, so its box really is the region that was generated. The
        Herwig stage varies eight parameters along a six-dimensional locus taken from
        published retunes, because the shower coupling and the shower cutoff rise together in
        that family and so do the cluster fission mass and power. Its bounding box therefore
        contains large regions no run ever visited, and a point there passes in_box() while
        being a genuine extrapolation.

        The test that sees this is the design's own covariance. Rotating into its principal
        directions and dividing by the spread along each one gives a score, and the two
        directions across the locus have a small spread, so a point off the locus scores
        high. The score is compared with the largest score any TRAINING point reaches, which
        is the model's own definition of how far out it has seen.

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
            raise ValueError(f'theta {np.asarray(theta)} is outside the training box\n{self.box}')
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
            raise ValueError(f'theta {np.asarray(theta)} is outside the training box\n{self.box}')
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
        f_m = log[ (1 - f) exp(l_S) + f exp(l_H) ] ,

    and the ensemble logit is the mean of f_m. Written this way the head is exact at the
    edges: the derivative with respect to theta_H vanishes identically at f = 0 and the one
    with respect to theta_S at f = 1, so a fraction of zero is the first generator alone and
    not an approximation to it.

    The two parameter blocks are read from the shapes of the exported networks rather than
    assumed to be equal in size, so a mixture of generators with different parameter counts
    needs no change here.
    """

    def __init__(self, path):
        c = np.load(path)
        assert str(c.get('head_kind', 'cond')) == 'mixture', \
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

    def logit(self, A, theta, check_box=True):
        if check_box and not self.in_box(theta):
            raise ValueError(f'theta {np.asarray(theta)} is outside the training box\n{self.box}')
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
            # log-sum-exp with the larger exponent pulled out, so neither term can overflow
            big = np.maximum(lS, lH)
            acc += big + np.log(np.clip((1 - fr)*np.exp(lS - big) + fr*np.exp(lH - big), 1e-300, None))
        return acc/self.ENS
