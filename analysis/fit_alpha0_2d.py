#!/usr/bin/env python3
"""Joint (alpha_s, alpha_0) fit of the matched NNLL+NNLO + dispersive-shift prediction to
the ALEPH thrust distribution.

Needs ares_recovered/ares_prod_asgrid.csv: the vmax production script re-run with, per row,
Sigma_res evaluated at alpha_s(MZ) in ASGRID (each run to muR with the package's own
two-loop running).  Everything else -- EERAD3 cumulants, series matching, Milan shift with
the alpha_s-dependent subtraction -- is rebuilt per alpha_s hypothesis with the SAME code
paths as the 1-parameter fit (match_v3 / np_shift / fit_alpha0), so at alpha_s = 0.118 this
must reproduce alpha_0 = 0.330 exactly (regression printed).

chi2 is the same diagonal form as fit_alpha0 (window 0.05 < tau < 0.33), so the two fits
differ ONLY in floating alpha_s.  The alpha_s direction is profiled on the 9-point grid and
interpolated with a parabola through the minimum.
"""
import numpy as np, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np_shift as N
from match_v3 import fo_cumulants, fo_shift, logR
from fit_alpha0 import aleph

ASGRID = [0.108, 0.110, 0.112, 0.114, 0.116, 0.118, 0.120, 0.122, 0.124]
CSV = 'ares_recovered/ares_prod_asgrid.csv'
# ARES_EXT=1 selects the widened table (alpha_s 0.100 to 0.136 in steps of 0.002), whose columns at
# the nine original couplings reproduce the original table
if os.environ.get('ARES_EXT', '') == '1':
    ASGRID = [round(0.100 + 0.002*k, 3) for k in range(19)]
    CSV = 'ares_recovered/ares_prod_asgrid_ext.csv'
MZ = 91.1876


def load_grid(path):
    def mf(x):
        x = x.strip('"').replace('*^', 'e')
        return float('nan') if 'I' in x else float(x)
    A = {}
    for r in csv.reader(open(path)):
        A.setdefault((mf(r[1]), mf(r[2])), []).append([mf(r[0])] + [mf(x) for x in r[3:]])
    for k in A: A[k] = np.array(sorted(A[k]))
    bad = np.zeros(len(next(iter(A.values()))), bool)
    for M in A.values(): bad |= ~np.isfinite(M).all(1)
    for k in A: A[k] = A[k][~bad]
    return A


def matched(M, mu, asmz):
    """log-R matched cumulant at alpha_s(MZ) = asmz for scale point (mu, xv=1)."""
    tau = M[:, 0]; c = M[:, 2:6]                       # c0..c3 (col 1 is Sigma(0.118))
    j = ASGRID.index(asmz)
    sig = M[:, 8 + j]                                   # Sigma_res at this asmz (cols 8..16)
    fo = fo_cumulants()
    t_fo, cumA = fo['LO']; _, cumB = fo['NLO']; _, cumC = fo['NNLO']
    i = np.searchsorted(t_fo, tau[0]); sl = slice(i, i + len(tau))
    assert np.allclose(t_fo[sl], tau), 'grid mismatch'
    Ash, Bsh, Csh = fo_shift(cumA[sl], cumB[sl], cumC[sl], np.log(mu**2))
    asmu = N.alpha_s(mu*MZ, asmz)
    ab = asmu/(2*np.pi)
    S = [ab*Ash, ab**2*Bsh, ab**3*Csh]
    R = [c[:, 1]*asmu, c[:, 2]*asmu**2, c[:, 3]*asmu**3]
    return tau, logR(sig, R, S, 3)


def chi2_profile(M, mu, asmz, lo, hi, y, e, msk):
    tau, m = matched(M, mu, asmz)
    S, _ = N.sanitize(m/m[-1])
    best = (np.inf, None)
    for a0 in np.arange(0.20, 0.85, 0.0025):
        t = tau + N.shift(mu, a0, asmz=asmz)
        d = (N.cum_eval(hi, t, S) - N.cum_eval(lo, t, S))/(hi - lo)
        c2 = float(np.sum(((d - y)[msk]/e[msk])**2))
        if c2 < best[0]:
            best = (c2, a0)
    return best


def main():
    A = load_grid(CSV)
    ncol = next(iter(A.values())).shape[1]
    print(f'grid CSV: {len(A)} scale points, {ncol} columns '
          f'(expect 17 = tau..Sigma(asmu) + {len(ASGRID)} grid evaluations)')
    lo, hi, y, e = aleph()
    ctr = 0.5*(lo + hi)
    msk = (ctr > 0.05) & (ctr < 0.33)
    M = A[(1.0, 1.0)]
    rows = []
    for asmz in ASGRID:
        c2, a0 = chi2_profile(M, 1.0, asmz, lo, hi, y, e, msk)
        rows.append((asmz, a0, c2))
        print(f'  alpha_s(MZ)={asmz:.3f}  ->  alpha_0={a0:.4f}  chi2={c2:.2f}/{msk.sum()-2}')
    # parabola through the three points around the minimum
    rows = np.array(rows)
    i = np.argmin(rows[:, 2])
    i = min(max(i, 1), len(rows) - 2)
    x, f = rows[i-1:i+2, 0], rows[i-1:i+2, 2]
    den = (f[0] - 2*f[1] + f[2])
    xmin = x[1] - 0.5*(x[2] - x[0])*(f[2] - f[0])/(2*den) if den > 0 else x[1]
    sig = (x[1] - x[0])/np.sqrt(den) if den > 0 else float('nan')
    a0min = np.interp(xmin, rows[:, 0], rows[:, 1])
    c2min = f[1] - ((f[2]-f[0])/2)**2/(8*den/2) if den > 0 else f[1]
    print(f'\nJOINT FIT:  alpha_s(MZ) = {xmin:.4f} +- {sig:.4f} (exp, Dchi2=1 in alpha_s)')
    print(f'            alpha_0(2 GeV) = {a0min:.3f}   chi2_min ~ {c2min:.1f}/{msk.sum()-2}')
    print(f'REGRESSION: at alpha_s=0.118 the profiled alpha_0 must equal the 1-parameter '
          f'fit value 0.330 (up to the finer 0.0025 scan grid).')

if __name__ == '__main__':
    main()
