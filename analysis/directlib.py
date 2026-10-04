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

torch.set_default_dtype(torch.float64)


class Model:
    def __init__(self, tag, export=None, ref=None, exps=('aleph', 'delphi', 'opal'), nch_data=(18.63, 0.11),
                 first_bin=0.05, floor_rel=0.002, a0_stride=2, ae_dtype='float64', grid='output/thrust_targets_grid.npz',
                 last_bin=None, use_nch=True, mix_form='additive'):
        self.tag = tag
        self.export = export or f'output/models/{tag}_cond.npz'
        self.refpath = ref or f'output/models/{tag}_ref_v2.npz'
        self.exps, self.NCH, self.first_bin, self.floor_rel = list(exps), tuple(nch_data), float(first_bin), float(floor_rel)
        # a fit variant may keep only bins below last_bin (the window) and drop the multiplicity
        self.last_bin = None if last_bin is None else float(last_bin); self.use_nch = bool(use_nch)
        # how the mixing fraction combines the two generators: 'additive', the population mixture the
        # head was trained as, (1 - f) q_S + f q_H; or 'geometric', q_S^(1 - f) q_H^f normalized, which
        # uses the same two trained parameter networks and serves as a check of that choice
        assert mix_form in ('additive', 'geometric'); self.mix_form = mix_form
        c = np.load(self.export)
        self.kind = str(c['head_kind']) if 'head_kind' in c.files else 'cond'
        self.ENS, self.K, self.L, self.T = int(c['ens']), int(c['K']), int(c['nlayers']), float(c['temperature'])
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

    def logit(self, theta):
        """Ensemble-mean logit per reference event, divided by the temperature. theta physical."""
        tn = (theta - self.CEN)/self.HW
        acc = torch.zeros(self.N)
        for m in range(self.ENS):
            if self.kind == 'mixture':
                fr = theta[-1]                                   # the fraction enters physical
                bS = self._mlp(self.NET_S[m], tn[:self.NS]); bH = self._mlp(self.NET_H[m], tn[self.NS:self.NS + self.NH])
                # two matrix-vector products: a single (N,K)x(K,2) product takes a slow path in torch
                lS = (self.AE[m] @ bS.to(self.AE.dtype)).double(); lH = (self.AE[m] @ bH.to(self.AE.dtype)).double()
                if self.mix_form == 'geometric':
                    acc = acc + (1 - fr)*lS + fr*lH
                else:
                    big = torch.maximum(lS, lH)
                    acc = acc + big + torch.log(torch.clamp((1 - fr)*torch.exp(lS - big) + fr*torch.exp(lH - big), min=1e-300))
            else:
                acc = acc + (self.AE[m] @ self._mlp(self.NET_S[m], tn).to(self.AE.dtype)).double()
        return acc/self.ENS/self.T

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
