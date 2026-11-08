#!/usr/bin/env python3
"""The anchored family as a differentiable function of the generator's knobs.

One class, Model, holds everything the coupling fit evaluates: the head (single-generator or
mixture), the reference sample's thrust and multiplicity, the theory grid, and the LEP data. Its
forward pass is the one of anchored_family.py and fitlib.py, step for step:

    w0(theta)   the reference carried to theta by the head (ensemble-mean logit / T)
    tilt        w = w0 exp(F lam)/Z, F = (m(tau) - c(as, a0)) Theta(window), with lam solving
                E_w[F] + Pw^2 Sigma lam = 0, Sigma = Sigma_pert + diag(FLOOR_REL c)^2, Pw = the
                window fraction of w0
    chi^2       thrust bins with lo >= FIRST_BIN of each experiment, prediction and data both
                renormalized over those bins, plus the L3 charged multiplicity

Everything is torch float64 and differentiable in theta. The tilt's dependence on theta is
carried by one Newton step taken at the converged multipliers, whose derivative is the implicit-
function derivative exactly at convergence (checked against finite differences to 1e-6).

Used by profile_direct.py (the fit), variations_direct.py, exchange_direct.py, parton_direct.py.
"""
import os, sys
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_alpha0 import lepdata
import bundle_rule

torch.set_default_dtype(torch.float64)


class Model:
    def __init__(self, tag, export=None, ref=None, exps=('aleph', 'delphi', 'opal'), nch_data=(18.63, 0.11),
                 first_bin=0.05, floor_rel=0.002, a0_stride=2, ae_dtype='float64', grid='output/thrust_targets_grid.npz',
                 last_bin=None, use_nch=True, mix_form='additive'):
        self.tag = tag
        self.export = export or f'output/models/{tag}_cond.npz'
        self.refpath = ref or f'output/models/{tag}_ref_v2.npz'
        # the rerun (BEAM_PHOTONS=remove, D4): the bundle must be one refreshed under the beam-photon rule in use
        bundle_rule.check(self.refpath)
        self.exps, self.NCH, self.first_bin, self.floor_rel = list(exps), tuple(nch_data), float(first_bin), float(floor_rel)
        # a fit variant may keep only bins below last_bin (the window) and drop the multiplicity
        self.last_bin = None if last_bin is None else float(last_bin); self.use_nch = bool(use_nch)
        # how the mixing fraction combines the two generators: 'additive', the population mixture the
        # head was trained as, (1 - f) q_S + f q_H; or 'geometric', q_S^(1 - f) q_H^f normalized, which
        # uses the same two trained parameter networks and serves as a check of that choice
        assert mix_form in ('additive', 'geometric'); self.mix_form = mix_form
        c = np.load(self.export)
        self.kind = str(c['head_kind']) if 'head_kind' in c.files else 'cond'
        if self.kind == 'geometric':
            # the multiplicative mixture head: the two parameter networks of the mixture kind,
            # combined linearly in the fraction, which is the geometric form below
            self.kind = 'mixture'; self.mix_form = 'geometric'
        # wifi weights (wifi_fit.py): log w = w_0 + sum_ij W_ij <a_i, b_j>; None is the published
        # network, w_ij = delta_ij/(ENS T)
        self.wifi = None
        self.ENS, self.K, self.L, self.T = int(c['ens']), int(c['K']), int(c['nlayers']), float(c['temperature'])
        if os.environ.get('T_DIR'):         # the rerun (D11, E2): the export must carry the refitted temperature of T_DIR
            from score_obs import check_export_T
            self.T = check_export_T(self.export, self.T)
        self.NORM = c['norm'].astype(np.float64); self.NT = int(c['ntheta']); self.ACT = str(c['act'])
        assert int(c['additive']) == 0, 'additive heads are not handled here'
        net = lambda p: [[(torch.tensor(c[f'{p}{m}_W{l}'], dtype=torch.float64),
                           torch.tensor(c[f'{p}{m}_b{l}'], dtype=torch.float64)) for l in range(self.L)] for m in range(self.ENS)]
        self.NET_S = net('B'); self.NET_H = net('BH') if self.kind == 'mixture' else None
        self.NS = self.NET_S[0][0][0].shape[1]; self.NH = self.NET_H[0][0][0].shape[1] if self.NET_H else 0
        assert (self.NS + self.NH + 1 == self.NT) if self.kind == 'mixture' else (self.NS == self.NT)
        # float64 by default: a float32 product leaves ~1e-6 noise in chi^2, enough to stall a line search
        self.AE = torch.from_numpy(np.ascontiguousarray(c['AE']).astype(ae_dtype))      # (ENS, N, K)
        self.N = self.AE.shape[1]
        self.CEN = torch.tensor(self.NORM[:, 0]); self.HW = torch.tensor(self.NORM[:, 1])
        # reference observables
        r = np.load(self.refpath)
        self.thr = r['1_minus_thrust'].astype(np.float64); assert len(self.thr) == self.N, (len(self.thr), self.N)
        self.nch = (r['nch'].astype(np.float64) if 'nch' in r.files else
                    ((r['particles'][..., 5] != 0) & r['mask'].astype(bool)).sum(1).astype(np.float64))
        self.extra = {k: r[k].astype(np.float64) for k in ('B_total', 'rho_heavy', 'nbaryon', 'strange', 'mult_total', 'tau_parton')
                      if k in r.files}
        # theory grid
        self.G = np.load(grid, allow_pickle=True)
        self.keys = [str(k) for k in self.G['keys']]
        self.ASG = self.G['alphas']; self.A0G = self.G['alpha0'][::a0_stride]
        self.j_of = [int(np.where(np.isclose(self.G['alpha0'], a0))[0][0]) for a0 in self.A0G]
        self.WLO, self.WHI = float(self.G['window'][0]), float(self.G['window'][1])
        self.inwin = (self.thr >= self.WLO) & (self.thr < self.WHI)
        self.IW = torch.from_numpy(np.where(self.inwin)[0])
        self.MW = torch.from_numpy(self.funs(self.thr[self.inwin]))                  # (Nw, 14)
        self.NCHT = torch.from_numpy(self.nch); self.THRT = torch.from_numpy(self.thr)
        self.exp = []
        for ex in self.exps:
            lo, hi, y, e = lepdata(ex)
            edges = np.r_[lo, hi[-1]]; assert np.allclose(edges[1:-1], hi[:-1])
            idx = np.searchsorted(edges, self.thr, side='right') - 1
            wid = hi - lo; use = lo >= self.first_bin - 1e-12
            if self.last_bin is not None:
                use = use & (hi <= self.last_bin + 1e-9)
            Sd = float((y[use]*wid[use]).sum())
            ins = (idx >= 0) & (idx < len(lo))
            self.exp.append(dict(name=ex, nb=len(lo), ins=torch.from_numpy(np.where(ins)[0]), idx=torch.from_numpy(idx[ins]),
                                 use=torch.from_numpy(np.where(use)[0]), wid=torch.from_numpy(wid), y=torch.from_numpy(y/Sd),
                                 e=torch.from_numpy(e/Sd), lo=lo, hi=hi, y_raw=y, e_raw=e))
        self.ndat = sum(len(x['use']) for x in self.exp) + (1 if self.use_nch else 0)

    # -- moments --------------------------------------------------------------------------------
    def funs(self, t):
        t = np.clip(t, 1e-6, None); L = np.log(t)
        F = {'logt': L, 'tau': t, 'tlogt': t*L, 'log2t': L**2, 'tau2': t**2, 'tlog2t': t*L**2, 't2logt': t**2*L,
             'log3t': L**3, 'tau3': t**3, 'tlog3t': t*L**3, 't3logt': t**3*L, 't2log2t': t**2*L**2, 'log4t': L**4, 'tau4': t**4}
        return np.column_stack([F[k] for k in self.keys])

    # -- the head -------------------------------------------------------------------------------
    def _mlp(self, params, x):
        for l, (W, b) in enumerate(params):
            z = W @ x + b
            x = (z*torch.sigmoid(z) if self.ACT == 'silu' else torch.relu(z)) if l < len(params) - 1 else z
        return x

    def set_wifi(self, w):
        """Use wifi weights: w = (w_0, W_11, W_12, ..., W_MM) row-major, a numpy array or a tensor
        (with requires_grad for the learning band). None restores the published network."""
        if w is None:
            self.wifi = None; return
        w = torch.as_tensor(np.asarray(w, np.float64)) if not torch.is_tensor(w) else w
        assert len(w) == self.ENS*self.ENS + 1, (len(w), self.ENS)
        self.wifi = (w[0], w[1:].reshape(self.ENS, self.ENS))

    def logit(self, theta):
        """Log ratio per reference event. The published network: the ensemble-mean logit divided
        by the temperature. With wifi weights: w_0 + sum_ij W_ij <a_i, b_j>. theta physical."""
        tn = (theta - self.CEN)/self.HW
        if self.kind == 'mixture' and self.mix_form == 'additive':
            assert self.wifi is None, 'wifi weights need an inner-product head (cond or geometric)'
            acc = torch.zeros(self.N)
            for m in range(self.ENS):
                fr = theta[-1]                                   # the fraction enters physical
                bS = self._mlp(self.NET_S[m], tn[:self.NS]); bH = self._mlp(self.NET_H[m], tn[self.NS:self.NS + self.NH])
                # two matrix-vector products: a single (N,K)x(K,2) product takes a slow path in torch
                lS = (self.AE[m] @ bS.to(self.AE.dtype)).double(); lH = (self.AE[m] @ bH.to(self.AE.dtype)).double()
                big = torch.maximum(lS, lH)
                acc = acc + big + torch.log(torch.clamp((1 - fr)*torch.exp(lS - big) + fr*torch.exp(lH - big), min=1e-300))
            return acc/self.ENS/self.T
        bt = []
        for m in range(self.ENS):
            if self.kind == 'mixture':                            # geometric: linear in the fraction
                fr = theta[-1]
                bS = self._mlp(self.NET_S[m], tn[:self.NS]); bH = self._mlp(self.NET_H[m], tn[self.NS:self.NS + self.NH])
                bt.append((1 - fr)*bS + fr*bH)
            else:
                bt.append(self._mlp(self.NET_S[m], tn))
        if self.wifi is None:
            acc = torch.zeros(self.N)
            for m in range(self.ENS):
                acc = acc + (self.AE[m] @ bt[m].to(self.AE.dtype)).double()
            return acc/self.ENS/self.T
        w0, W = self.wifi
        acc = torch.zeros(self.N, dtype=torch.float64) + w0
        # sum_ij W_ij <a_i, b_j> = sum_i <a_i, sum_j W_ij b_j>: M products instead of M^2
        for i in range(self.ENS):
            bi = sum(W[i, j]*bt[j] for j in range(self.ENS))
            acc = acc + (self.AE[i] @ bi.to(self.AE.dtype)).double()
        return acc

    def theta(self, u):
        """Physical parameters from standardized ones on [-1, 1]."""
        return self.CEN + self.HW*u

    # -- targets ------------------------------------------------------------------------------
    def target(self, ia, jj, which='central'):
        G = self.G
        if which == 'central':
            return G['central'][ia, jj]
        kind, v = which.split(':'); v = int(v)
        return G['scale'][ia, jj, v] if kind == 'scale' else G['np_vars'][ia, jj, v]

    def sigma(self, ia, jj, cvec):
        return self.G['sigma_pert'][ia, jj] + np.diag((self.floor_rel*np.abs(cvec))**2)

    # -- the tilt -----------------------------------------------------------------------------
    @staticmethod
    def newton(lw0w, logPout, Fw, Sc, lam, itmax=60, tol=1e-10):
        """Solve E_w[F] + Sc lam = 0 on the window events (no gradient). w0 normalized over all events."""
        def state(l):
            zw = lw0w + Fw @ l
            lZ = torch.logaddexp(logPout, torch.logsumexp(zw, 0))
            ww = torch.exp(zw - lZ)
            m = ww @ Fw
            return ww, m, m + Sc @ l
        ww, m, g = state(lam)
        it = 0
        for it in range(itmax):
            if torch.linalg.norm(g) < tol:
                break
            J = (Fw*ww[:, None]).T @ Fw - torch.outer(m, m) + Sc
            step = torch.linalg.solve(J, g)
            gn0 = torch.linalg.norm(g); s = 1.0
            for _ in range(30):
                lt = lam - s*step; wt, mt, gt = state(lt)
                if torch.linalg.norm(gt) < gn0:
                    break
                s *= 0.5
            lam, ww, m, g = lt, wt, mt, gt
        return lam, ww, m, it

    def tilt_on(self, lw0, IW, MW, cvec, Sig, lam0):
        """Tilted log-weights over all events for constraint functions MW on the events IW."""
        cc = torch.as_tensor(np.asarray(cvec, np.float64))
        SigT = torch.as_tensor(np.asarray(Sig, np.float64))
        Pw = torch.exp(torch.logsumexp(lw0[IW], 0))
        Fw = MW - cc[None, :]
        Sc = Pw**2*SigT
        with torch.no_grad():
            lam, ww, m, its = self.newton(lw0[IW].detach(), torch.log1p(-Pw.detach()), Fw, Sc.detach(), lam0)
            J = (Fw*ww[:, None]).T @ Fw - torch.outer(m, m) + Sc.detach()
        zw = lw0[IW] + Fw @ lam
        lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
        g = torch.exp(zw - lZ) @ Fw + Sc @ lam
        lam1 = lam - torch.linalg.solve(J, g)                  # one differentiable Newton step
        z = lw0.index_add(0, IW, Fw @ lam1)
        return z - torch.logsumexp(z, 0), lam.detach(), Pw, int(its)

    def weights(self, theta, ia, jj, lam0=None, which='central', anchored=True):
        """(log w, log w0, lam, Pw, newton iterations) at physical theta and theory node (ia, jj)."""
        lg = self.logit(theta)
        lw0 = lg - torch.logsumexp(lg, 0)
        if not anchored:
            return lw0, lw0, None, torch.exp(torch.logsumexp(lw0[self.IW], 0)), 0
        cvec = self.target(ia, jj, which)
        lw, lam, Pw, its = self.tilt_on(lw0, self.IW, self.MW, cvec, self.sigma(ia, jj, cvec),
                                        torch.zeros(len(self.keys)) if lam0 is None else lam0)
        return lw, lw0, lam, Pw, its

    # -- data -----------------------------------------------------------------------------------
    def chi2(self, w, with_nch=True):
        """(total, per-experiment dict) for normalized weights w."""
        with_nch = with_nch and self.use_nch
        tot = ((w @ self.NCHT - self.NCH[0])/self.NCH[1])**2 if with_nch else torch.zeros(())
        per = {}
        for x in self.exp:
            cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']])
            u = x['use']; d = cnt[u]/(x['wid'][u]*cnt[u].sum())
            c2 = (((d - x['y'][u])/x['e'][u])**2).sum()
            tot = tot + c2; per[x['name']] = c2
        if with_nch:
            per['nch'] = ((w @ self.NCHT - self.NCH[0])/self.NCH[1])**2
        return tot, per

    def densities(self, w):
        """Per experiment: (lo, hi, prediction, data, error), both sides renormalized over the fitted bins."""
        out = {}
        with torch.no_grad():
            for x in self.exp:
                cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
                u = x['use'].numpy(); d = cnt[u]/(x['wid'].numpy()[u]*cnt[u].sum())
                out[x['name']] = (x['lo'][u], x['hi'][u], d, x['y'].numpy()[u], x['e'].numpy()[u])
        return out

    def observables(self, w):
        """Derived quantities as differentiable tensors."""
        win = torch.zeros(self.N).index_fill(0, self.IW, 1.0)
        pw = w @ win
        o = {'pwin': pw, 'tau_win': (w @ (self.THRT*win))/pw, 'thrust': w @ self.THRT, 'nch': w @ self.NCHT}
        for k, v in self.extra.items():
            if k != 'tau_parton':
                o[k] = w @ torch.from_numpy(v)
        return o

    # -- the profile at one theory node --------------------------------------------------------
    def chi2_at(self, u, ia, jj, lam0, which='central', grad=True, with_nch=True):
        if grad:
            uu = torch.tensor(np.asarray(u, float), requires_grad=True)
            lw, _, lam, _, _ = self.weights(self.theta(uu), ia, jj, lam0, which)
            c2, _ = self.chi2(torch.exp(lw), with_nch)
            c2.backward()
            return float(c2), uu.grad.numpy().copy(), lam
        with torch.no_grad():
            lw, _, lam, _, _ = self.weights(self.theta(torch.tensor(np.asarray(u, float))), ia, jj, lam0, which)
            c2, _ = self.chi2(torch.exp(lw), with_nch)
        return float(c2), None, lam

    def extras(self, u, ia, jj, which='central', lam0=None):
        """Diagnostics at a profiled point: chi^2 per experiment, window fraction before and after
        the tilt, observables, relative entropy, effective sample fraction, largest moment pull."""
        with torch.no_grad():
            lw, lw0, lam, Pw, its = self.weights(self.theta(torch.tensor(np.asarray(u, float))), ia, jj, lam0, which)
            w = torch.exp(lw); c2, per = self.chi2(w)
            cvec = self.target(ia, jj, which); sd = np.sqrt(np.diag(self.sigma(ia, jj, cvec)))
            ww = w[self.IW]; mom = ((ww @ self.MW)/ww.sum()).numpy()
            o = {k: float(v) for k, v in self.observables(w).items()}
            o0 = {k + '_unanchored': float(v) for k, v in self.observables(torch.exp(lw0)).items()}
            return dict(chi2=float(c2), chi2_per={k: float(v) for k, v in per.items()}, pwin0=float(Pw),
                        kl=float((w*(lw - lw0)).sum()), neff=float(1/(w**2).sum()/self.N),
                        maxpull=float(np.max(np.abs(mom - cvec)/sd)), newton_iters=int(its), **o, **o0)

    # -- the linear mixture of the two generators (D1, D2), for the cost benchmark ------------------
    def weights_lin(self, theta, ia, jj, lam0=None, which='central', anchored=True):
        """The linear mixture at physical theta = (theta_S, theta_H, xi): each generator's weights from the head at
        xi = 0 and at xi = 1 (at_fraction), normalized over all events and tilted alone, and the mixed event weights
        (1 - xi) w_S + xi w_H, whose means are (1 - xi) <O>_S + xi <O>_H. lam0 = (lam_S, lam_H) or None.
        Returns (w, [lam_S, lam_H])."""
        xi = theta[-1]; w, lams = 0, []
        for g, xg in enumerate((0.0, 1.0)):
            lw, _, lam, _, _ = self.weights(at_fraction(theta, xg), ia, jj, None if lam0 is None else lam0[g], which, anchored)
            w = w + ((1 - xi) if g == 0 else xi)*torch.exp(lw); lams.append(lam)
        return w, lams

    def chi2_lin_at(self, u, ia, jj, lam0=None, which='central', grad=True, with_nch=True):
        """chi2_at for the linear mixture: the chi^2 of the mixed weights of weights_lin at standardized u, and its
        gradient in u (both generators' parameters and the fraction)."""
        if grad:
            uu = torch.tensor(np.asarray(u, float), requires_grad=True)
            w, lams = self.weights_lin(self.theta(uu), ia, jj, lam0, which)
            c2, _ = self.chi2(w, with_nch)
            c2.backward()
            return float(c2), uu.grad.numpy().copy(), lams
        with torch.no_grad():
            w, lams = self.weights_lin(self.theta(torch.tensor(np.asarray(u, float))), ia, jj, lam0, which)
            c2, _ = self.chi2(w, with_nch)
        return float(c2), None, lams

    def scan(self, ia, jj, n, rng, which='central'):
        """chi^2 without gradients on a Latin hypercube of n points in the standardized box,
        returned sorted, for choosing starting points cheaply."""
        P = (np.argsort(rng.random((n, self.NT)), axis=0) + rng.random((n, self.NT)))/n*2 - 1
        res, lam = [], None
        for u in P:
            f, _, lam = self.chi2_at(u, ia, jj, lam, which, grad=False)
            res.append((f, u))
        res.sort(key=lambda r: r[0])
        return res

    def profile_node(self, ia, jj, starts, lam0=None, which='central', with_nch=True, bounds=None):
        """Best L-BFGS-B result over the starts: (chi2, u, lam, evaluations summed over all starts)."""
        best = None; nfev = 0
        lam_c = torch.zeros(len(self.keys)) if lam0 is None else lam0
        bnds = bounds or [(-1, 1)]*self.NT
        for u0 in starts:
            box = [lam_c]
            def fg(u):
                f, g, lam = self.chi2_at(u, ia, jj, box[0], which, True, with_nch)
                box[0] = lam
                return f, g
            r = minimize(fg, np.asarray(u0, float), jac=True, method='L-BFGS-B', bounds=bnds,
                         options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
            nfev += int(r.nfev)
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy(), box[0])
        return best + (nfev,)


class TargetInterp:
    """Smooth theory targets in (alpha_s, alpha_0): bicubic splines of the fourteen moments and of
    the covariance over the grid, for the central calculation or any variation, with first
    derivatives, so the coupling and alpha_0 can be fitted continuously together with the nuisance
    parameters."""

    def __init__(self, G, which='central', a0_stride=1):
        from scipy.interpolate import RectBivariateSpline
        A, A0 = np.asarray(G['alphas'], float), np.asarray(G['alpha0'], float)[::a0_stride]
        if which == 'central':
            C = np.asarray(G['central'])[:, ::a0_stride]
        else:
            kind, v = which.split(':'); v = int(v)
            C = np.asarray(G['scale'] if kind == 'scale' else G['np_vars'])[:, ::a0_stride, v]
        S = np.asarray(G['sigma_pert'])[:, ::a0_stride]
        self.lim = (A[0], A[-1], A0[0], A0[-1])
        self.sc = [RectBivariateSpline(A, A0, C[..., k], kx=3, ky=3, s=0) for k in range(C.shape[-1])]
        nk = C.shape[-1]; self.nk = nk
        self.ss = {(i, j): RectBivariateSpline(A, A0, S[..., i, j], kx=3, ky=3, s=0) for i in range(nk) for j in range(i, nk)}

    def __call__(self, a, a0):
        """(c, dc/da, dc/da0, Sigma, dSigma/da, dSigma/da0) at a point."""
        c = np.array([float(s(a, a0)) for s in self.sc])
        ca = np.array([float(s(a, a0, dx=1)) for s in self.sc]); c0 = np.array([float(s(a, a0, dy=1)) for s in self.sc])
        S = np.zeros((self.nk, self.nk)); Sa = np.zeros_like(S); S0 = np.zeros_like(S)
        for (i, j), s in self.ss.items():
            S[i, j] = S[j, i] = float(s(a, a0)); Sa[i, j] = Sa[j, i] = float(s(a, a0, dx=1)); S0[i, j] = S0[j, i] = float(s(a, a0, dy=1))
        return c, ca, c0, S, Sa, S0


def chi2_continuous(M, TI, x, lam0=None):
    """chi^2 and its gradient at x = (u_1..u_NT, alpha_s, alpha_0), with the targets from TI and the
    floor of Sigma as in Model.sigma. The gradient in (alpha_s, alpha_0) is the chain rule through the
    spline derivatives of the targets and their covariance."""
    u = np.asarray(x[:M.NT], float); a, a0 = float(x[-2]), float(x[-1])
    c, ca, c0, S, Sa, S0 = TI(a, a0)
    cT = torch.tensor(c, requires_grad=True); ST = torch.tensor(S, requires_grad=True)
    uu = torch.tensor(u, requires_grad=True)
    lg = M.logit(M.theta(uu)); lw0 = lg - torch.logsumexp(lg, 0)
    Sig = ST + torch.diag((M.floor_rel*torch.abs(cT))**2)
    lw, lam, Pw, its = tilt_on_t(M, lw0, cT, Sig, torch.zeros(M.MW.shape[1]) if lam0 is None else lam0)
    c2, _ = M.chi2(torch.exp(lw)); c2.backward()
    gc, gS = cT.grad.numpy(), ST.grad.numpy()
    ga = float(gc @ ca + np.sum(gS*Sa)); g0 = float(gc @ c0 + np.sum(gS*S0))
    return float(c2), np.r_[uu.grad.numpy(), ga, g0], lam


def tilt_on_t(M, lw0, cT, SigT, lam0):
    """Model.tilt_on with the targets and their covariance as differentiable tensors."""
    IW, MW = M.IW, M.MW
    Pw = torch.exp(torch.logsumexp(lw0[IW], 0))
    Fw = MW - cT[None, :]
    Sc = Pw**2*SigT
    with torch.no_grad():
        lam, ww, m, its = Model.newton(lw0[IW].detach(), torch.log1p(-Pw.detach()), Fw.detach(), Sc.detach(), lam0)
        J = (Fw.detach()*ww[:, None]).T @ Fw.detach() - torch.outer(m, m) + Sc.detach()
    zw = lw0[IW] + Fw @ lam
    lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
    g = torch.exp(zw - lZ) @ Fw + Sc @ lam
    lam1 = lam - torch.linalg.solve(J, g)
    z = lw0.index_add(0, IW, Fw @ lam1)
    return z - torch.logsumexp(z, 0), lam.detach(), Pw, int(its)


def logw_continuous(M, TI, u, a, a0, which_TI=None, lam0=None):
    """(log w, log w0, lam, Pw) at the standardized nuisance parameters u (a tensor) and a point
    (a, a0) between the theory nodes, the targets and their covariance from TI with the floor of
    Model.sigma, as in chi2_continuous."""
    c, _, _, S, _, _ = TI(a, a0)
    cT = torch.tensor(c); Sig = torch.tensor(S) + torch.diag((M.floor_rel*torch.abs(cT))**2)
    lg = M.logit(M.theta(u)); lw0 = lg - torch.logsumexp(lg, 0)
    lw, lam, Pw, its = tilt_on_t(M, lw0, cT, Sig, torch.zeros(M.MW.shape[1]) if lam0 is None else lam0)
    return lw, lw0, lam, Pw


# -- the linear mixture of the two generators ------------------------------------------------------
# The prediction of the linear mixture is (1 - xi) <O>_S(theta_S) + xi <O>_H(theta_H). Each generator's
# prediction comes from its own weights, the mixture head at xi = 0 (Sherpa's log ratio at theta_S) or at
# xi = 1 (Herwig's at theta_H), normalized over all events and, with a calculation, tilted on its own; the two
# predictions are then mixed with (1 - xi, xi). Distributions normalized to the total rate and means over the
# full sample make every prediction linear in xi. The geometric head is only evaluated at xi = 0 and 1.
def at_fraction(theta, xi):
    """theta with the mixing fraction set to xi: 0 for Sherpa at theta_S, 1 for Herwig at theta_H. A tensor keeps
    its graph, so derivatives in the generator's own parameters go through."""
    if torch.is_tensor(theta):
        return torch.cat([theta[:-1], torch.full((1,), float(xi), dtype=theta.dtype)])
    t = np.array(theta, np.float64)
    t[-1] = xi
    return t


def xi_moments(lo, hi):
    """(E, var) for xi flat on [lo, hi]: E[xi_g xi_h] of (xi_S, xi_H) = (1 - xi, xi) and Var(xi). On [0, 1],
    E = [[1/3, 1/6], [1/6, 1/3]] and Var = 1/12; for a fixed xi, E = outer((1 - xi, xi)) and Var = 0."""
    m, v = 0.5*(lo + hi), (hi - lo)**2/12
    e2 = v + m*m
    return np.array([[1 - 2*m + e2, m - e2], [m - e2, e2]]), float(v)


def mix_cov(E, var, delta=None, own=(), shared=()):
    """The covariance of y = (1 - xi) y_S + xi y_H with each y_g linear in its inputs and xi flat and independent
    of them:
        Var(xi) delta delta^T + sum_g E_gg J_g V_g J_g^T + sum_gh E_gh J_g C J_h^T ,
    which on [0, 1] is delta delta^T/12 + (V_SS + V_HH)/3 + (V_SH + V_HS)/6. delta = y_H - y_S at the central
    point; own holds (J_g, V_g) for g = S, H, the slopes in the generator's own parameters and their variances
    (a vector) or covariance; shared holds triples (J_S, J_H, C), the slopes of both generators in one shared
    input (the learning weights, the calculation) and its covariance. A generator whose E is zero, or whose J is
    None, does not enter."""
    out = 0.0
    if delta is not None and var > 0:
        out = out + var*np.outer(delta, delta)
    for g, (J, V) in enumerate(own):
        if J is not None and E[g, g] != 0:
            V = np.asarray(V, np.float64)
            out = out + E[g, g]*((J*V) @ J.T if V.ndim == 1 else J @ V @ J.T)
    for JS, JH, C in shared:
        Js = (JS, JH)
        for g in range(2):
            for h in range(2):
                if E[g, h] != 0 and Js[g] is not None and Js[h] is not None:
                    out = out + E[g, h]*(Js[g] @ C @ Js[h].T)
    return out


def lin_box(cen, hw, central, box, scale=1/3, ns=None):
    """(lo, hi) of the flat prior of the linear mixture. box is tune or full, alone or with _sherpa or _herwig:
    full     each generator's parameters over the trained box, centred on the box
    tune     each range scale of its width, centred on the default tunes in central and cut symmetrically
             where the box edge is nearer, so that the default stays the centre and the slopes at the central
             prediction are taken at the centre of the prior (D6)
    The fraction is flat on [0, 1]; with _sherpa it is 0 and with _herwig 1, and the other generator's
    parameters stay where they are (they do not enter). ns is the number of Sherpa parameters."""
    kind, _, gen = box.partition('_')
    assert kind in ('tune', 'full') and gen in ('', 'sherpa', 'herwig'), box
    cen, hw = np.asarray(cen, np.float64), np.asarray(hw, np.float64)
    P = (np.asarray(central, np.float64) if kind == 'tune' else cen).copy()
    h = hw*(scale if kind == 'tune' else 1.0)
    h = np.maximum(np.minimum(h, np.minimum(P - (cen - hw), (cen + hw) - P)), 0.0)
    xi = len(cen) - 1
    P[xi], h[xi] = 0.5, 0.5
    if gen:
        own = np.zeros(len(cen), bool)
        own[:ns] = gen == 'sherpa'
        own[ns:xi] = gen == 'herwig'
        h = np.where(own, h, 0.0)
        P[xi] = 0.0 if gen == 'sherpa' else 1.0
    return P - h, P + h


def w_published(M):
    """The wifi weights of the published network for the export's temperature: w_0 = 0, w_ij = delta_ij/(N T), as
    wifi_fit.py writes them (w_published), the point at which the learning band is taken."""
    w = np.zeros(M.ENS*M.ENS + 1); w[1:] = np.eye(M.ENS).ravel()/(M.ENS*M.T)
    return w


def wifi_point(W, M, log=print):
    """(w_at, note): w_published(M), the published network with the export's temperature. A wifi fit file written
    with another temperature (a refitted calibration constant, D11) keeps its covariance C, which does not depend
    on the temperature, but its w_published is stale, so the export's is used and the note says so."""
    w = w_published(M); wf = np.asarray(W['w_published'], np.float64)
    if wf.shape == w.shape and np.array_equal(wf, w):
        return w, 'the wifi fit file (same temperature as the export)'
    note = (f'the export temperature T = {M.T!r}: the wifi fit file holds w_published of another temperature '
            f'(T = {1/(M.ENS*wf[1]) if wf.shape == w.shape and wf[1] else float("nan")!r})')
    log(f'NOTE learning band taken at {note}')
    return w, note


def thband(d, pairs, scheme, npl, up):
    """Eq. (thband) of the paper for one prediction (one generator alone): sigma_+ (up) or sigma_- (down) from the
    signed displacements d = {label: array} of the variations, each scale pair once with the larger of its two
    one-sided displacements, the scheme once, the dispersive-model variations npl as their mean square."""
    f = (lambda v: np.maximum(v, 0.0)) if up else (lambda v: np.maximum(-v, 0.0))
    tot = sum(np.maximum(f(np.asarray(d[a], float)), f(np.asarray(d[b], float)))**2 for a, b in pairs) + f(np.asarray(d[scheme], float))**2
    return np.sqrt(tot + sum(f(np.asarray(d[l], float))**2 for l in npl)/len(npl))


def thband_mix(dS, dH, pairs, scheme, npl, up, lo=0.0, hi=1.0):
    """The theory band of the linear mixture (D9): Eq. (thband) applied at each xi to the displacement
    (1 - xi) d_S + xi d_H of the mixed prediction, and sigma_+^2 (up) or sigma_-^2 averaged over xi flat on [lo, hi].
    The integrand is a sum of squares of piecewise-linear functions of xi, with kinks where a displacement changes
    sign or the two of a scale pair cross, so Simpson's rule between consecutive kinks gives the average exactly.
    dS, dH = {label: array of the same shape} for Sherpa and Herwig; returns sigma_+ or sigma_-, of that shape."""
    labs = list(dict.fromkeys([l for p in pairs for l in p] + [scheme] + list(npl)))
    shape = np.shape(np.asarray(dS[scheme], float))
    S = {l: np.asarray(dS[l], float).ravel() for l in labs}; H = {l: np.asarray(dH[l], float).ravel() for l in labs}
    f = (lambda v: max(v, 0.0)) if up else (lambda v: max(-v, 0.0))
    out = np.empty(len(S[scheme]))
    for i in range(len(out)):
        s, h = {l: S[l][i] for l in labs}, {l: H[l][i] for l in labs}

        def sig2(x):
            dl = {l: (1 - x)*s[l] + x*h[l] for l in labs}
            t = sum(max(f(dl[a]), f(dl[b]))**2 for a, b in pairs) + f(dl[scheme])**2
            return t + sum(f(dl[l])**2 for l in npl)/len(npl)
        if hi <= lo:
            out[i] = sig2(lo); continue
        kinks = {lo, hi}
        for a0, a1 in [(s[l], h[l]) for l in labs] + [(s[a] - s[b], h[a] - h[b]) for a, b in pairs]:
            if a0 != a1:                                   # (1 - x) a0 + x a1 = 0
                x = a0/(a0 - a1)
                if lo < x < hi:
                    kinks.add(x)
        xs = sorted(kinks)
        out[i] = sum((x1 - x0)/6*(sig2(x0) + 4*sig2(0.5*(x0 + x1)) + sig2(x1)) for x0, x1 in zip(xs[:-1], xs[1:]))/(hi - lo)
    return np.sqrt(np.maximum(out, 0.0)).reshape(shape)


def stat_parts(M, theta, anchored=True):
    """One generator at physical theta (the head at its xi): its weights normalized to one, and u(O) = w r, the
    per-event term of the statistical error of the weighted mean of a centred per-event function O. With the
    calculation imposed the multipliers are solved again on every resampled sample, so r = O minus its projection
    on the constraint functions, r = O - (F - <F>) J^-1 Cov(F, O), J = Cov(F, F) + P_win^2 Sigma; without it r = O.
    The error of a mean over several generators on the same events is sqrt(sum_i (sum_g a_g u_g,i)^2)."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        lw, lw0, lam, Pw, _ = M.weights(th, 0, 0, anchored=anchored)
        w = torch.exp(lw).numpy()
    w = w/w.sum()
    if not anchored:
        return w, lambda O: w*O
    inwin = np.zeros(M.N, bool); inwin[M.IW.numpy()] = True
    cvec = np.asarray(M.target(0, 0, 'central'), np.float64)
    F = np.zeros((M.N, len(cvec))); F[inwin] = M.MW.numpy() - cvec[None, :]
    Fc = F - w @ F
    Jinv = np.linalg.inv((Fc*w[:, None]).T @ Fc + float(Pw)**2*np.asarray(M.sigma(0, 0, cvec), np.float64))
    return w, lambda O: w*(O - Fc @ (Jinv @ (Fc.T @ (w*O))))


def fitted_point(src):
    """The fitted point of a direct-profile fit: (alpha_s, alpha_0, u) from the continuous profiles
    (*_rows.json) and the nuisance parameters minimized there (*_bands_point.json, bands_exact.py)."""
    import json
    stem = src.replace('.json', '')
    rr = json.load(open(stem + '_rows.json')); pt = json.load(open(stem + '_bands_point.json'))
    a = rr['variations']['central']['value']
    a0 = rr['alpha_0_profile']['value'] if 'alpha_0_profile' in rr else rr['variations']['central']['alpha_0_at_min']
    assert abs(pt['alpha_s'] - a) < 1e-9 and abs(pt['alpha_0'] - a0) < 1e-9, 'stale *_bands_point.json: rerun bands_exact.py point'
    return a, a0, np.array(pt['u']), rr
