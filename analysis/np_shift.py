#!/usr/bin/env python3
"""Dispersive (Dokshitzer-Webber / Milan) power correction for thrust.

The perturbative thrust distribution cannot be integrated down to tau = 0: the coupling
hits its Landau pole and the resummed Sigma(tau) turns over and goes negative.  Cutting the
integral off by hand is what the earlier moment tables did, and it is exactly what we do not
want.  The physical regulator is the leading power correction, which for an additive
observable like thrust is a rigid SHIFT of the distribution,

    dsigma/dtau (tau)  ->  dsigma/dtau (tau - a_tau P) ,      a_tau = 2 for 1 - T,

    P = (4 C_F / pi^2) M (mu_I / Q)
        [ alpha_0(mu_I) - alpha_s(muR) - (beta_0/2pi)(ln(muR/mu_I) + K/beta_0 + 1) alpha_s^2(muR) ]

with the Milan factor M = 1.490, the infrared matching scale mu_I = 2 GeV, and alpha_0(mu_I)
the one nonperturbative parameter.  The subtraction inside the bracket removes the part of
the low-scale coupling that the perturbative series has already counted, which is what makes
this a genuine cure rather than a second cutoff.

Implemented as a shift of the tau grid itself, (tau, Sigma) -> (tau + delta, Sigma), which is
exact and preserves the normalisation identically.  Below tau = delta the distribution then
has no support, so ln tau is bounded and every log moment converges with NO bounds imposed.

Writes output/np_shift.json.
"""
import os
import numpy as np, pickle, json, sys

NF = 5
CF = 4/3.0
CA = 3.0
Q = 91.1876
MUI = 2.0                                   # infrared matching scale, GeV
MILAN = 1.490
A_TAU = 2.0                                 # the observable's shift coefficient, Delta = 2
ALPHA0 = 0.50                               # alpha_0(2 GeV), the one NP parameter
ALPHA0_SD = 0.04
ASMZ = 0.118
B0 = 11 - 2*NF/3.0                          # beta_0 in the (alpha_s/4pi -free) normalisation
B1 = 102 - 38*NF/3.0
KCMW = CA*(67/18.0 - np.pi**2/6) - 5*NF/9.0
# NP_SUB_ORDER=3 adds the O(alpha_s^3) renormalon-subtraction term (GLM 2013); default 2
# reproduces every published number bit for bit.
SUB_ORDER = int(os.environ.get('NP_SUB_ORDER', '2'))
A3_OVER_A1 = -12.46      # GLM eq. (A.6b) at nf=5 (cusp + two-loop soft piece), as read from the PDF


def alpha_s(mu, asmz=ASMZ):
    """Exact two-loop running from alpha_s(MZ). With b0 = B0/4pi, b1 = B1/(4pi)^2 and
    t = ln(mu^2/MZ^2), integrating da/dt = -b0 a^2 - b1 a^3 gives
        1/a + (b1/b0) ln[ a/(1 + b1 a/b0) ] = 1/a0 + (b1/b0) ln[ a0/(1 + b1 a0/b0) ] + b0 t,
    solved by fixed point. Validated against ARES's own running (AlphaSFixedNF):
    0.106837 at 2 MZ and 0.131847 at MZ/2 from 0.118."""
    b0 = B0/(4*np.pi); b1 = B1/(4*np.pi)**2
    t = np.log(mu**2/Q**2)
    g = lambda x: (b1/b0)*np.log(x/(1 + b1*x/b0))
    rhs = 1/asmz + g(asmz) + b0*t
    a = asmz/(1 + asmz*b0*t)                    # one-loop start
    for _ in range(200):
        an = 1.0/(rhs - g(a))
        if abs(an - a) < 1e-15:
            a = an
            break
        a = an
    return a


def shift(mur_mult=1.0, alpha0=ALPHA0, q=Q, verbose=False, asmz=ASMZ):
    """delta tau = a_tau * P, the rigid shift of the thrust distribution."""
    muR = mur_mult*q
    aS = alpha_s(muR, asmz)
    br = (alpha0 - aS
          - (B0/(2*np.pi))*(np.log(muR/MUI) + KCMW/B0 + 1)*aS**2)
    if SUB_ORDER >= 3:
        # O(alpha_s^3) term of the perturbative average coupling below mu_I, the NNLO
        # renormalon subtraction of Gehrmann-Luisoni-Monni (EPJC 73 (2013) 2265, eq. 40).
        # Derivation: alpha_0^PT(mu_I) = (1/mu_I) int_0^mu_I dk Gamma(alpha_s(k^2)) with the
        # physical coupling Gamma(a) = a (1 + a/pi a2 + a^2/pi^2 a3), a2 = A2/A1 = K/2, and the
        # two-loop running a(k) = a - a^2/pi b0 l + a^3/pi^2 (b0^2 l^2 - b1 l), l = ln(k^2/muR^2),
        # b0 = B0/4, b1 = B1/16 (GLM normalisation). With L = ln(muR/mu_I) the k-averages are
        # <l> = -2L-2 and <l^2> = 4L^2+8L+8. The a^2 term of this expansion is exactly the line
        # above; the a^3 term is what follows. Its size is dominated by the b0^2 <l^2> running
        # term (L = 3.8 at Q = MZ); A3/A1 itself, read from GLM eq. (A.6b), enters at < 0.002.
        b0 = B0/4.0; b1 = B1/16.0; a2 = KCMW/2.0; L = np.log(muR/MUI)
        c3 = (A3_OVER_A1 + 2*b0*a2*(2*L + 2) + b0**2*(4*L**2 + 8*L + 8) + b1*(2*L + 2))/np.pi**2
        br = br - c3*aS**3
    P = (4*CF/np.pi**2)*MILAN*(MUI/q)*br
    if verbose:
        print(f'    muR={muR:7.3f}  alpha_s(muR)={aS:.5f}  bracket={br:.5f}  '
              f'P={P:.6f}  delta={A_TAU*P:.6f}')
    return A_TAU*P


SCET = {'logt':-3.046,'tau':0.06638,'tlogt':-0.1584,'log2t':9.934,'tau2':0.00798,
        'tlog2t':0.4184,'t2logt':-0.0151,'log3t':-34.13,'tau3':0.001402,'tlog3t':-1.205,
        't3logt':-0.002263,'t2log2t':0.03154,'log4t':122.1,'tau4':0.0003034}
G = {'logt':lambda t:np.log(t),'tau':lambda t:t,'tlogt':lambda t:t*np.log(t),
     'log2t':lambda t:np.log(t)**2,'tau2':lambda t:t**2,'tlog2t':lambda t:t*np.log(t)**2,
     't2logt':lambda t:t**2*np.log(t),'log3t':lambda t:np.log(t)**3,'tau3':lambda t:t**3,
     'tlog3t':lambda t:t*np.log(t)**3,'t3logt':lambda t:t**3*np.log(t),
     't2log2t':lambda t:t**2*np.log(t)**2,'log4t':lambda t:np.log(t)**4,'tau4':lambda t:t**4}


def sanitize(S):
    """Sigma is a cumulative distribution, so it must be non-negative and non-decreasing.
    The resummed logR result violates both just above the Landau region.  Clipping at zero
    and taking the running maximum imposes exactly the defining property and nothing more.
    Returns the repaired Sigma and the total weight moved, as a systematic to quote."""
    Sc = np.maximum.accumulate(np.clip(S, 0.0, None))
    moved = float(np.abs(Sc - S).max())
    return Sc, moved


def moments(tau, S, clean=True, lo=0.0):
    # dS[i] is the probability mass between grid points tau[i-1] and tau[i] (with a
    # zeroth edge at lo), so the weight must be evaluated at the MIDPOINT of that
    # interval.  Evaluating at tau[i] itself -- the right endpoint -- biases every
    # increasing weight up by half a grid spacing (found in the 2026-08 audit: it
    # moved <1-T> by h/2 = 0.00125, half the EERAD3 bin width).  Midpoint is O(h^2).
    #
    # lo is the left edge of the SUPPORT.  A caller that has already displaced the grid by
    # the dispersive shift, tau -> tau + delta, must pass lo=delta: the shifted distribution
    # has no support below delta, so the first cell runs from delta, not from zero.  Leaving
    # it at zero puts the first cell's midpoint delta/2 too low, and since that cell carries
    # the steeply logarithmic small-tau end it biased <ln^4 tau> by 0.4 per cent (2026-09-07).
    if clean:
        S, _ = sanitize(S)
    S = S/S[-1]
    dS = np.diff(np.concatenate([[0.0], S]))
    edges = np.concatenate([[lo], tau])
    tmid = np.maximum(0.5*(edges[:-1] + edges[1:]), 1e-12)
    return {n: float(np.sum(f(tmid)*dS)) for n, f in G.items()}


def main():
    print('=== the shift, and how it moves with muR and alpha_0 ===')
    for mm in (0.5, 1.0, 2.0):
        shift(mm, verbose=True)
    print(f'    alpha_0 = {ALPHA0-ALPHA0_SD:.2f} -> delta={shift(1.0, ALPHA0-ALPHA0_SD):.6f}')
    print(f'    alpha_0 = {ALPHA0+ALPHA0_SD:.2f} -> delta={shift(1.0, ALPHA0+ALPHA0_SD):.6f}')
    d0 = shift(1.0)
    print(f'\n  central delta = {d0:.5f}  (this is also the shift induced in <1-T>)')

    src = next((a for a in sys.argv[1:] if a.endswith('.pkl')), 'output/matched_v2.pkl')
    print(f'  matched input: {src}')
    res = pickle.load(open(src, 'rb'))
    allv, cen = {}, None
    for (mu, xv, sch), (tau, sig, m) in res.items():
        for a0 in (ALPHA0, ALPHA0-ALPHA0_SD, ALPHA0+ALPHA0_SD):
            d = shift(mu, a0)
            mo = moments(tau + d, m)
            for n, v in mo.items():
                allv.setdefault(n, []).append(v)
            if (mu, xv, sch, a0) == (1.0, 1.0, 'logR', ALPHA0):
                cen = mo
    print(f'\n=== 14 moments, FULL range, NO bounds imposed '
          f'({len(res)*3} variations) ===')
    print(f'{"moment":9s} {"shifted":>11s} {"SCET":>11s} {"ratio":>8s} '
          f'{"envelope":>26s} {"covers":>7s}')
    nin = 0
    out = {}
    for n in G:
        lo, hi = min(allv[n]), max(allv[n])
        ok = lo <= SCET[n] <= hi; nin += ok
        out[n] = dict(central=cen[n], scet=SCET[n], lo=lo, hi=hi, covers=bool(ok))
        print(f'{n:9s} {cen[n]:11.5g} {SCET[n]:11.5g} {cen[n]/SCET[n]:8.3f} '
              f'[{lo:11.4g},{hi:11.4g}] {"yes" if ok else "NO":>6s}')
    print(f'\nSCET central inside the envelope for {nin}/{len(G)} moments')

    print('\n=== cost of imposing the cumulative-distribution property ===')
    print('  the resummed logR Sigma dips negative just above the Landau region; modR does not')
    worst = 0.0
    for k in sorted(res, key=str):
        tau, sig, m = res[k]
        _, moved = sanitize(m/m[-1])
        worst = max(worst, moved)
    print(f'  largest weight moved by the clip, over all variations: {worst:.3e}')
    print('  effect on the central moments (clip on vs off):')
    tau, sig, m = res[(1.0, 1.0, 'logR')]
    d = shift(1.0)
    a = moments(tau+d, m, clean=True); b = moments(tau+d, m, clean=False)
    for n in ('logt', 'tau', 'log2t', 'log4t'):
        rel = abs(a[n]-b[n])/abs(a[n])
        env = (max(allv[n])-min(allv[n]))/2/abs(a[n])
        print(f'    {n:7s} clip {a[n]:11.5g}  raw {b[n]:11.5g}   shift {rel*100:6.2f}%'
              f'   vs envelope half-width {env*100:6.2f}%')
    json.dump(dict(delta=d0, alpha0=ALPHA0, milan=MILAN, mu_I=MUI, a_tau=A_TAU,
                   moments=out, n_variations=len(res)*3),
              open('output/np_shift' + ('_v3' if 'v3' in src else '') + '.json', 'w'), indent=1)
    print('wrote output/np_shift.json')


if __name__ == '__main__':
    main()


def cum_eval(x, t, S):
    """The cumulant S, tabulated at the points t, evaluated at x by monotone cubic (PCHIP)
    interpolation, zero below the first point and one above the last. Its derivative, the
    distribution, is continuous, so binned and windowed quantities vary smoothly when the shift
    moves the grid. Linear interpolation of the tabulated cumulant (spacing 0.0025 in tau) makes the
    distribution piecewise constant and leaves a kink in every bin or window integral whenever a grid
    point crosses its edge, which put ripples of a few tenths into the chi^2 of the fits."""
    from scipy.interpolate import PchipInterpolator
    x = np.asarray(x, float)
    y = PchipInterpolator(t, S, extrapolate=False)(x)
    return np.where(x < t[0], 0.0, np.where(x > t[-1], 1.0, y))
