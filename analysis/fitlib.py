#!/usr/bin/env python3
"""The fit's core, shared by fit_coupling.py and fit_closure_pseudo.py so the closure test and
the fit cannot drift apart.

Conventions, in one place:
  bins below FIRST_BIN are dropped, and both prediction and data are renormalized over the bins
  that remain, so the excluded region enters nowhere. Data errors are scaled by the same factor,
  which ignores the correlation the renormalization induces, the usual treatment.
  predictions are interpolated across the nuisance grid and the chi^2 is formed from them, not
  the other way round, because the predictions are nearly linear in the nuisances and the chi^2
  is quadratic in the predictions.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.optimize import minimize


def stack(fam, exps, first_bin, nch):
    """(VEC, DY, DE, ndat) with VEC of shape (nth, nA, n0, ndat): every fitted number, in order."""
    done = int(fam['done_nodes'])
    VEC, DY, DE = [], [], []
    for ex in exps:
        lo, hi, y, e = fam[f'bins_{ex}_lo'], fam[f'bins_{ex}_hi'], fam[f'data_{ex}_y'], fam[f'data_{ex}_e']
        w = hi - lo; use = lo >= first_bin - 1e-12; Sd = float((y[use]*w[use]).sum())
        P = fam[f'D_{ex}'][:done][..., use]
        VEC.append(P/(P*w[use]).sum(-1)[..., None]); DY.append((y/Sd)[use]); DE.append((e/Sd)[use])
    VEC.append(fam['S_nch'][:done][..., None]); DY.append(np.array([nch[0]])); DE.append(np.array([nch[1]]))
    VEC = np.concatenate(VEC, axis=-1); DY = np.concatenate(DY); DE = np.concatenate(DE)
    return VEC, DY, DE, len(DY)


def window_mask(fam, exps, first_bin):
    """Which entries of the stacked vector are thrust bins inside the anchoring window."""
    w0, w1 = float(fam['window'][0]), float(fam['window'][1]); m = []
    for ex in exps:
        lo, hi = fam[f'bins_{ex}_lo'], fam[f'bins_{ex}_hi']; use = lo >= first_bin - 1e-12
        m.append(((lo >= w0 - 1e-9) & (hi <= w1 + 1e-9))[use])
    return np.concatenate(m + [np.array([False])])


def profile(VEC, DY, DE, axes, TH, shape, ia, j, nstart=6):
    """min over the nuisances of chi^2 at one theory node, and where it sits."""
    lo = np.array([a[0] for a in axes]); hi = np.array([a[-1] for a in axes])
    meth = 'cubic' if min(len(a) for a in axes) >= 4 else 'linear'
    f = RegularGridInterpolator(axes, VEC[:, ia, j].reshape(shape + (VEC.shape[-1],)),
                                method=meth, bounds_error=False, fill_value=None)
    def c2(x):
        return float(np.sum(((f(np.clip(np.atleast_2d(x), lo, hi))[0] - DY)/DE)**2))
    flat = (((VEC[:, ia, j] - DY)/DE)**2).sum(-1)
    best = None
    for x0 in [TH[k] for k in np.argsort(flat)[:nstart]]:
        r = minimize(c2, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun: best = r
    return float(best.fun), best.x


def surface(VEC, DY, DE, axes, TH, shape, nA, n0):
    prof = np.zeros((nA, n0)); prof_th = np.zeros((nA, n0, TH.shape[1]))
    for ia in range(nA):
        for j in range(n0):
            prof[ia, j], prof_th[ia, j] = profile(VEC, DY, DE, axes, TH, shape, ia, j)
    return prof, prof_th


def paraboloid_min(prof, ASG, A0G):
    """The minimum and the Delta chi^2 = 1 covariance from a paraboloid through the 3x3 around it."""
    nA, n0 = prof.shape
    ia0, j0 = np.unravel_index(np.argmin(prof), prof.shape)
    ia1, j1 = int(np.clip(ia0, 1, nA-2)), int(np.clip(j0, 1, n0-2))
    X, Y, Z = [], [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            X.append(ASG[ia1+di]); Y.append(A0G[j1+dj]); Z.append(prof[ia1+di, j1+dj])
    X, Y, Z = map(np.array, (X, Y, Z))
    coef = np.linalg.lstsq(np.column_stack([np.ones(9), X, Y, X*X, Y*Y, X*Y]), Z, rcond=None)[0]
    c0, cx, cy, cxx, cyy, cxy = coef
    H = np.array([[2*cxx, cxy], [cxy, 2*cyy]]); g = np.array([cx, cy])
    if np.linalg.det(H) <= 0:
        return None
    x = -np.linalg.solve(H, g); cov = 2*np.linalg.inv(H)
    return dict(alpha_s=float(x[0]), alpha_0=float(x[1]), sa=float(np.sqrt(cov[0, 0])), s0=float(np.sqrt(cov[1, 1])),
                rho=float(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])), chi2=float(c0 + g @ x + 0.5*x @ H @ x),
                edge=bool(ia0 in (0, nA-1) or j0 in (0, n0-1)), node=(int(ia0), int(j0)))


def spline_min(prof, ASG, A0G, smooth=0.0):
    """Minimum and Delta chi^2 = 1 covariance from a smooth surface through the profiled grid.

    The profiled chi^2 is smooth in the two theory parameters (verified: along a fixed alpha_0 it
    is a clean parabola and the nuisance minimiser agrees with a dense scan to 0.02), but taking a
    discrete minimum over the other parameter's grid puts a sawtooth into any one-dimensional
    profile. Interpolating the surface with a bicubic spline and reading the minimum and the
    curvature off it avoids both that sawtooth and the choice of how wide a parabola to fit.
    Returns the same keys as paraboloid_min so the two can be compared directly."""
    sp = RectBivariateSpline(ASG, A0G, prof, kx=3, ky=3, s=smooth)
    fa = np.linspace(ASG[0], ASG[-1], 801); f0 = np.linspace(A0G[0], A0G[-1], 801)
    Z = sp(fa, f0); ia, j0 = np.unravel_index(np.argmin(Z), Z.shape)
    x, y = float(fa[ia]), float(f0[j0])
    H = np.array([[float(sp(x, y, dx=2, dy=0)), float(sp(x, y, dx=1, dy=1))],
                  [float(sp(x, y, dx=1, dy=1)), float(sp(x, y, dx=0, dy=2))]])
    if np.linalg.det(H) <= 0 or H[0, 0] <= 0:
        return None
    cov = 2*np.linalg.inv(H)
    edge = ia in (0, len(fa)-1) or j0 in (0, len(f0)-1)
    return dict(alpha_s=x, alpha_0=y, sa=float(np.sqrt(cov[0, 0])), s0=float(np.sqrt(cov[1, 1])),
                rho=float(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])), chi2=float(Z[ia, j0]),
                edge=bool(edge), node=(int(np.argmin(np.abs(ASG - x))), int(np.argmin(np.abs(A0G - y)))))


def profile_errors(prof, ASG, A0G, smooth=0.0, dchi2=1.0):
    """The minimum and the one-parameter errors from where the PROFILED chi^2 crosses min + 1.

    This is the definition of the error rather than a curvature, so it needs neither a choice of
    how wide a parabola to fit nor a second derivative of an interpolant. The surface is splined
    first, one parameter is profiled out continuously, and the crossing is found by bisection on
    the resulting one-dimensional curve. Asymmetric errors are returned as they come out.
    """
    sp = RectBivariateSpline(ASG, A0G, prof, kx=3, ky=3, s=smooth)
    fa = np.linspace(ASG[0], ASG[-1], 2001); f0 = np.linspace(A0G[0], A0G[-1], 2001)
    pa = sp(fa, f0).min(axis=1)          # alpha_0 profiled out
    p0 = sp(fa, f0).min(axis=0)          # alpha_s profiled out
    out = {}
    for name, grid, curve in (('alpha_s', fa, pa), ('alpha_0', f0, p0)):
        k = int(np.argmin(curve)); m = curve[k]; tgt = m + dchi2
        def cross(lo, hi_):
            if curve[lo] < tgt or curve[hi_] < tgt: return None
            return None
        def find(side):
            idx = range(k, -1, -1) if side < 0 else range(k, len(grid))
            prev = k
            for i in idx:
                if curve[i] >= tgt:
                    x0, x1 = grid[prev], grid[i]; y0, y1 = curve[prev], curve[i]
                    return float(x0 + (tgt - y0)*(x1 - x0)/(y1 - y0)) if y1 != y0 else float(x1)
                prev = i
            return None
        lo_, hi_ = find(-1), find(+1)
        out[name] = dict(value=float(grid[k]), lo=lo_, hi=hi_,
                         err_lo=None if lo_ is None else float(grid[k] - lo_),
                         err_hi=None if hi_ is None else float(hi_ - grid[k]),
                         hit_edge=bool(lo_ is None or hi_ is None), chi2min=float(m))
    return out


def profile_1d(xs, ys, dchi2=1.0):
    """Value and errors from a one-dimensional profile known at the points xs: a cubic spline through
    them, its minimum, and the crossings of minimum + dchi2 on either side (None when the curve does
    not reach it inside the range). The estimator of profile_rows.py and pseudo_direct.py."""
    from scipy.interpolate import CubicSpline
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    cs = CubicSpline(xs, ys); fx = np.linspace(xs[0], xs[-1], 20001); fy = cs(fx)
    k = int(np.argmin(fy)); m = fy[k]; tgt = m + dchi2
    def find(rng):
        prev = k
        for i in rng:
            if fy[i] >= tgt:
                return float(fx[prev] + (tgt - fy[prev])*(fx[i] - fx[prev])/(fy[i] - fy[prev]))
            prev = i
        return None
    lo, hi = find(range(k, -1, -1)), find(range(k, len(fx)))
    return dict(value=float(fx[k]), chi2min=float(m), lo=lo, hi=hi, err_lo=None if lo is None else float(fx[k] - lo),
                err_hi=None if hi is None else float(hi - fx[k]), hit_edge=bool(lo is None or hi is None or k in (0, len(fx) - 1)))
