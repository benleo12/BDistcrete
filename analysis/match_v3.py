"""Matched NNLL+NNLO thrust, with the scale variation done correctly.

What changes relative to match_v2:

(1) The running coupling. The ARES driver sets refscale = muR, so its expansion coefficients
    c_k multiply powers of alpha_s(muR), and match_v2 then evaluated everything at
    alpha_s(muR) = 0.118 for every muR. That holds the coupling fixed while the logarithms
    move, which is a change of alpha_s(MZ), not a scale variation. The regenerated CSV
    carries alpha_s(muR) from the package's own running (columns 8 and 9: alpha_s(muR) and
    Sigma at that value), and both the resummed and the fixed-order pieces use it.

(2) Third order. The fixed-order cumulant now carries the NNLO coefficient C when
    output/eerad_NNLO.npy exists, the scale transformation of (A, B, C) is the one already in
    match_v2.fo_shift, and both matching schemes are built from truncated power series in
    alpha_s rather than hand-expanded formulas, which is how the third-order cross terms
    stay right:
        log-R : Sigma = Sigma_res * exp[ ln Sigma_FO - ln Sigma_res ]_3
        mod-R : Sigma = Sigma_res^Z * T_3 exp[ ln Sigma_FO - Z ln Sigma_res ]_3
    where [.]_3 truncates the logarithm at alpha_s^3 and T_3 truncates the exponential.
    With Z = 1 and second-order truncation both reduce to the match_v2 expressions, which is
    the regression test run at the bottom.

Variations: {muR x2, muR /2, xV x2, xV /2} one at a time, times {log-R, mod-R}.
Writes output/matched_v3.pkl in the same layout as matched_v2.pkl.
"""
import numpy as np, csv, pickle, os, sys
from np_shift import alpha_s

NF = 5
B0 = (33 - 2*NF)/(12*np.pi)
B1 = (153 - 19*NF)/(24*np.pi**2)
CSV = './ares_recovered/ares_prod_vmax.csv'


def fo_shift(A, B, C, Lmu):
    """(A, B, C) in alpha_s(Q) -> coefficients of alpha_s(muR)/2pi, L = ln(muR^2/Q^2)."""
    u = 2*np.pi*B0*Lmu
    v = (2*np.pi)**2*(B0**2*Lmu**2 + B1*Lmu)
    return A, B + u*A, C + 2*u*B + v*A


# ---- truncated power-series algebra in alpha_s; a series is [s1, s2, s3] (s0 = 1 implied)
def ser_log(s):
    s1, s2, s3 = s
    return [s1, s2 - s1**2/2, s3 - s1*s2 + s1**3/3]


def ser_exp(e):                      # exp of a series with e0 = 0 -> [1, x1, x2, x3]
    e1, e2, e3 = e
    return [e1, e2 + e1**2/2, e3 + e1*e2 + e1**3/6]


def logR(sig, R, S, order=3):
    lf, lr = ser_log(S), ser_log(R)
    E = sum((lf[k] - lr[k]) for k in range(order))
    m = sig*np.exp(E)
    return m/m[-1]


def modR(tau, sig, R, S, order=3, v0=0.5, u=1.0, h=3.0):
    Z = np.where(tau < v0, (1 - (tau/v0)**u)**h, 0.0)
    lf, lr = ser_log(S), ser_log(R)
    E = [lf[k] - Z*lr[k] for k in range(3)]
    if order == 2:
        E[2] = 0*E[2]
    x = ser_exp(E)
    m = 1 + sum(x[:order])
    m = np.where(sig > 0, np.clip(sig, 0, None)**Z, 0.0)*m
    return m/m[-1]


def load(path):
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
    return A, int(bad.sum())


def fo_cumulants():
    out = {}
    for tag in ('LO', 'NLO', 'NNLO'):
        # A rank-trimmed CUMULANT file takes precedence (eerad_cumulant_combine.py): for NNLO the
        # bin-by-bin trimmed histogram in eerad_NNLO.npy integrates to a cumulant that is biased
        # by the local-subtraction pairs in adjacent bins (10-25% in the fit window). Delete or
        # rename output/eerad_NNLO_cum.npy to reproduce the pre-2026-08-25 numbers.
        fc = f'output/eerad_{tag}_cum.npy'
        if os.path.exists(fc):
            tc, cum, _ = np.load(fc)
            out[tag] = (tc, cum); continue
        f = f'output/eerad_{tag}.npy'
        if not os.path.exists(f):
            out[tag] = None; continue
        tc, val, err, _, _ = np.load(f)
        w = tc[1]-tc[0]
        above = np.concatenate([np.cumsum((val*w)[::-1])[::-1][1:], [0.0]])
        out[tag] = (tc, -(above + 0.5*val*w))
    return out


def build(csv_path, order=3, use_nnlo=True):
    A, ndrop = load(csv_path)
    fo = fo_cumulants()
    t_fo, cumA = fo['LO']; _, cumB = fo['NLO']
    have_C = use_nnlo and fo['NNLO'] is not None
    cumC = fo['NNLO'][1] if have_C else np.zeros_like(cumA)
    res = {}
    MZ = 91.1876
    for (mu, xv), M in sorted(A.items()):
        if M.shape[1] == 8:                              # tau, Sigma118, c0..c3, asmu, sig
            tau, sig118, c0, c1, c2, c3, asmu, sig = M.T
        else:                                            # 8-column CSV: rebuild asmu and sig
            tau, sig118, c0, c1, c2, c3 = M.T
            asmu = alpha_s(mu*MZ)*np.ones_like(tau)
            sig = 1 + c1*asmu + c2*asmu**2 + c3*asmu**3
        i = np.searchsorted(t_fo, tau[0]); sl = slice(i, i+len(tau))
        assert np.allclose(t_fo[sl], tau), 'grid mismatch'
        Ash, Bsh, Csh = fo_shift(cumA[sl], cumB[sl], cumC[sl], np.log(mu**2))
        ab = asmu/(2*np.pi)                          # alpha_s(muR)/2pi, really
        S = [ab*Ash, ab**2*Bsh, ab**3*Csh]
        R = [c1*asmu, c2*asmu**2, c3*asmu**3]
        if order == 2:
            S[2] = 0*S[2]; R[2] = 0*R[2]
        res[(mu, xv, 'logR')] = (tau, sig, logR(sig, R, S, order))
        res[(mu, xv, 'modR')] = (tau, sig, modR(tau, sig, R, S, order))
        # fixed-order numerical uncertainty: at central scales, rescale the NNLO cumulant by
        # +-5% (the top of the 3-5% cross-run error band) as one-at-a-time variations, so the
        # integrator error propagates into the envelope instead of being quoted beside it.
        if (mu, xv) == (1.0, 1.0) and have_C and order == 3:
            for tag, fac in (('logR_Cp', 1.05), ('logR_Cm', 0.95)):
                Sv = [S[0], S[1], S[2]*fac]
                res[(mu, xv, tag)] = (tau, sig, logR(sig, R, Sv, order))
    return res, ndrop, have_C


if __name__ == '__main__':
    order = 2 if '--order2' in sys.argv else 3
    res, nd, have_C = build(CSV, order=order, use_nnlo='--no-nnlo' not in sys.argv)
    print(f'order {order}, NNLO coefficient {"included" if have_C else "ABSENT (C=0)"}, '
          f'{nd} Landau point(s) dropped, {len(res)} variations')
    tag = 'v3' if order == 3 else 'v3o2'
    pickle.dump(res, open(f'output/matched_{tag}.pkl', 'wb'))
    # regression: at muR = Q the coupling is 0.118 either way, so the second-order build must
    # reproduce match_v2 there (it differs only in the muR-varied entries)
    if os.path.exists('output/matched_v2.pkl') and order == 2:
        old = pickle.load(open('output/matched_v2.pkl', 'rb'))
        for k in ((1.0, 1.0, 'logR'), (1.0, 1.0, 'modR'), (1.0, 2.0, 'logR')):
            if k in old and k in res:
                a, b = old[k][2], res[k][2]
                n = min(len(a), len(b))
                print(f'  regression {k}: max |v3(order2) - v2| = {np.abs(a[:n]-b[:n]).max():.2e}')
    tau = res[(1.0, 1.0, 'logR')][0]
    print(f'\n{"tau":>7s} {"central":>9s} {"muR band":>20s} {"full envelope":>20s}')
    for t in (0.02, 0.05, 0.10, 0.20, 0.30):
        i = np.argmin(abs(tau-t))
        cen = res[(1.0, 1.0, 'logR')][2][i]
        mus = [res[(m, 1.0, 'logR')][2][i] for m in (0.5, 1.0, 2.0)]
        allv = [v[2][i] for v in res.values()]
        print(f'{tau[i]:7.4f} {cen:9.5f}   [{min(mus):.5f}, {max(mus):.5f}]   '
              f'[{min(allv):.5f}, {max(allv):.5f}]')
