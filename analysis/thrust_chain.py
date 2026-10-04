#!/usr/bin/env python3
"""The NNLL+NNLO+dispersive thrust chain after the matching, in one re-runnable script.

Consolidates three inline steps of 2026-08-23/24 (recovered from the session transcript):
  1. joint (alpha_s, alpha_0) profile fit to ALEPH 91.2 GeV over 0.05<tau<0.33 for the central
     scales and for each of the 12 perturbative variations (muR, x_V one at a time in both
     schemes, NNLO cumulant +-5%), each variation REFITTED so the pair stays correlated;
  2. the 14 moments <tau^a ln^b tau> at every refitted pair, plus a correlated experimental
     pair (+-1 sigma_as along the error ellipse) -> output/moments_joint.json;
  3. the anchor targets: central moments at the joint fit, Sigma_c = (1/2) sum_v d_v d_v^T over
     the 12 variations and the experimental pair -> output/thrust_anchor_targets.npz.
New here: the experimental errors and correlation come from a quadratic fit to the chi^2
surface around the minimum instead of being typed in, and a chain_summary.json is written.
Inputs: ares_recovered/ares_prod_asgrid.csv (resummation on the alpha_s grid), the EERAD3
cumulants through match_v3.fo_cumulants (rank-trimmed eerad_NNLO_cum.npy when present), ALEPH.
"""
import os
import numpy as np, sys, json
sys.path.insert(0, '.')
import np_shift as N
from fit_alpha0_2d import load_grid, ASGRID, CSV
from match_v3 import fo_cumulants, fo_shift, logR, modR
from fit_alpha0 import aleph

A = load_grid(CSV); MZ = 91.1876
lo, hi, y, e = aleph(); ctr = 0.5*(lo+hi); msk = (ctr > 0.05) & (ctr < 0.33)
fo = fo_cumulants(); t_fo, cumA = fo['LO']; _, cumB = fo['NLO']; _, cumC = fo['NNLO']
# EERAD3 normalizes its coefficients to the Born cross section sigma_0, while the resummed cumulant
# and the measurements are normalized to the total hadronic cross section sigma. With
# sigma/sigma_0 = 1 + 2 abar + 4 K2 abar^2 (abar = alpha_s/2pi, R_had = 1 + a + K2 a^2, a = alpha_s/pi),
# the sigma-normalized coefficients are A, B - 2A and C - 2B + (4 - 4 K2) A. FO_NORM=born reproduces
# the unconverted chain.
NF_R = 5; K2_R = 1.9857 - 0.1153*NF_R
FO_NORM = os.environ.get('FO_NORM', 'total')
def fo_total(cA, cB, cC):
    if FO_NORM == 'born':
        return cA, cB, cC
    return cA, cB - 2*cA, cC - 2*cB + (4 - 4*K2_R)*cA
VARS = [(1.0,1.0,'logR',1.0),(2.0,1.0,'logR',1.0),(0.5,1.0,'logR',1.0),
        (1.0,2.0,'logR',1.0),(1.0,0.5,'logR',1.0),
        (1.0,1.0,'modR',1.0),(2.0,1.0,'modR',1.0),(0.5,1.0,'modR',1.0),
        (1.0,2.0,'modR',1.0),(1.0,0.5,'modR',1.0),
        (1.0,1.0,'logR',1.05),(1.0,1.0,'logR',0.95)]
A0GRID = np.arange(0.20, 0.90, 0.0025)

def build(mu, xv, scheme, asmz, cfac=1.0):
    M = A[(mu, xv)]; tau = M[:, 0]; c = M[:, 2:6]; sig = M[:, 8+ASGRID.index(asmz)]
    i = np.searchsorted(t_fo, tau[0]); sl = slice(i, i+len(tau))
    Ash, Bsh, Csh = fo_shift(*fo_total(cumA[sl], cumB[sl], cumC[sl]*cfac), np.log(mu**2))
    asmu = N.alpha_s(mu*MZ, asmz); ab = asmu/(2*np.pi)
    S = [ab*Ash, ab**2*Bsh, ab**3*Csh]; R = [c[:, 1]*asmu, c[:, 2]*asmu**2, c[:, 3]*asmu**3]
    return tau, (logR(sig, R, S, 3) if scheme == 'logR' else modR(tau, sig, R, S, 3))

def chi2_of(tau, S, mu, asmz, a0):
    t = tau + N.shift(mu, a0, asmz=asmz)
    d = (N.cum_eval(hi, t, S) - N.cum_eval(lo, t, S))/(hi-lo)
    return float(np.sum(((d-y)[msk]/e[msk])**2))

def profile(mu, xv, scheme, cfac=1.0):
    """2-parameter profile on the alpha_s grid; parabola in alpha_s through the minimum."""
    rows = []; surf = {}
    for asmz in ASGRID:
        tau, m = build(mu, xv, scheme, asmz, cfac); S, _ = N.sanitize(m/m[-1])
        c2 = np.array([chi2_of(tau, S, mu, asmz, a0) for a0 in A0GRID]); surf[asmz] = c2
        j = int(np.argmin(c2)); rows.append((asmz, A0GRID[j], c2[j]))
    r = np.array(rows); i = int(np.clip(np.argmin(r[:, 2]), 1, len(r)-2))
    x, f = r[i-1:i+2, 0], r[i-1:i+2, 2]; den = f[0]-2*f[1]+f[2]
    xm = x[1]-0.5*(x[2]-x[0])*(f[2]-f[0])/(2*den) if den > 0 else x[1]
    xm = float(np.clip(xm, ASGRID[0], ASGRID[-1])); a0m = float(np.interp(xm, r[:, 0], r[:, 1]))
    return xm, a0m, float(f[1]), (r, surf, i)

def hessian_errors(r, surf, i):
    """Quadratic fit of chi^2(alpha_s, alpha_0) on the 3 grid couplings around the minimum and
    the alpha_0 scan points within +-0.06 of the profiled minimum: Cov = 2 H^-1."""
    x0, y0 = r[i, 0], r[i, 1]; X = []; F = []
    for asmz in r[i-1:i+2, 0]:
        c2 = surf[asmz]; sel = np.abs(A0GRID - y0) <= 0.06
        for a0, v in zip(A0GRID[sel], c2[sel]):
            dx, dy = asmz-x0, a0-y0; X.append([1, dx, dy, dx*dx, dx*dy, dy*dy]); F.append(v)
    co = np.linalg.lstsq(np.array(X), np.array(F), rcond=None)[0]
    H = np.array([[2*co[3], co[4]], [co[4], 2*co[5]]]); C = 2*np.linalg.inv(H)
    sa, s0 = np.sqrt(C[0, 0]), np.sqrt(C[1, 1]); rho = C[0, 1]/(sa*s0)
    return float(sa), float(s0), float(rho)

def mom_interp(mu, xv, scheme, asmz, a0, cfac=1.0):
    ia = int(np.clip(np.searchsorted(ASGRID, asmz)-1, 0, len(ASGRID)-2))
    a1, a2 = ASGRID[ia], ASGRID[ia+1]; w = (asmz-a1)/(a2-a1); out = {}
    for aa, ww in ((a1, 1-w), (a2, w)):
        _sh = N.shift(mu, a0, asmz=aa)
        tau, m = build(mu, xv, scheme, aa, cfac); mo = N.moments(tau+_sh, m, lo=_sh)
        for k, v in mo.items(): out[k] = out.get(k, 0)+ww*v
    return out

def main():
    fits = {}; allm = {}; cen_pair = None; errs = None
    for mu, xv, sch, cf in VARS:
        asm, a0, c2, aux = profile(mu, xv, sch, cf); fits[(mu, xv, sch, cf)] = (asm, a0, c2)
        if (mu, xv, sch, cf) == (1.0, 1.0, 'logR', 1.0):
            cen_pair = (asm, a0); errs = hessian_errors(*aux)
        mo = mom_interp(mu, xv, sch, asm, a0, cf)
        for k, v in mo.items(): allm.setdefault(k, []).append(v)
        print(f'mu={mu:3} xv={xv:3} {sch} C x{cf:4}: alpha_s={asm:.4f} alpha_0={a0:.4f} chi2={c2:.1f}', flush=True)
    asc, a0c = cen_pair; sa, s0, rho = errs
    print(f'\nJOINT FIT central: alpha_s = {asc:.4f} +- {sa:.4f}, alpha_0 = {a0c:.3f} +- {s0:.3f}, rho = {rho:.2f}, '
          f'chi2 = {fits[(1.0,1.0,"logR",1.0)][2]:.1f}/{int(msk.sum())-2}, shift = {N.shift(1.0, a0c, asmz=asc):.4f}')
    cen = mom_interp(1.0, 1.0, 'logR', asc, a0c)
    out = {}; nin = 0
    for k in N.G:
        l, h = min(allm[k]), max(allm[k]); ok = l <= N.SCET[k] <= h; nin += ok
        out[k] = dict(central=cen[k], scet=N.SCET[k], lo=l, hi=h, covers=bool(ok))
    # correlated experimental pair along the error ellipse's major axis (+-1 sigma_as)
    allx = {k: [v['lo'], v['hi']] for k, v in out.items()}; expv = []
    for s in (+1, -1):
        asz, a0 = asc + s*sa, a0c + s*rho*s0
        mo = mom_interp(1.0, 1.0, 'logR', asz, a0); expv.append(mo)
        for k, v in mo.items(): allx[k].append(v)
        print(f'exp pair {"+" if s > 0 else "-"}: alpha_s={asz:.4f} alpha_0={a0:.4f} <1-T>={mo["tau"]:.5f}')
    outx = {}; ninx = 0
    for k in N.G:
        l, h = min(allx[k]), max(allx[k]); ok = l <= N.SCET[k] <= h; ninx += ok
        outx[k] = dict(central=cen[k], scet=N.SCET[k], lo=l, hi=h, covers=bool(ok))
    print(f'\n{"moment":9s} {"central":>11s} {"NNLLp":>10s} {"ratio":>7s} {"hw%":>6s} {"covers":>6s}')
    for k in N.G:
        hw = (outx[k]['hi']-outx[k]['lo'])/2/abs(cen[k])*100
        print(f'{k:9s} {cen[k]:11.5g} {N.SCET[k]:10.5g} {cen[k]/N.SCET[k]:7.3f} {hw:6.2f} {"yes" if outx[k]["covers"] else "NO":>5s}')
    print(f'NNLLp inside envelope: {nin}/14 (perturbative), {ninx}/14 (with exp pair)')
    asr = [v[0] for v in fits.values()]; a0r = [v[1] for v in fits.values()]
    print(f'alpha_s range over variations: {min(asr):.4f}-{max(asr):.4f};  alpha_0 range: {min(a0r):.4f}-{max(a0r):.4f}')
    json.dump(dict(fits={str(k): v for k, v in fits.items()}, moments=out, moments_with_exp=outx),
              open('output/moments_joint.json', 'w'), indent=1)
    # anchor targets
    keys = list(N.G); central = np.array([cen[k] for k in keys]); vecs = []; labels = []
    for k, (asm, a0, c2) in fits.items():
        vecs.append(np.array([mom_interp(*k[:3], asm, a0, k[3])[kk] for kk in keys])); labels.append(str(k))
    for s, mo in zip(('exp+1', 'exp-1'), expv):
        vecs.append(np.array([mo[kk] for kk in keys])); labels.append(s)
    V = np.array(vecs); D = V-central; Sig = (D.T@D)/2.0
    sd = np.sqrt(np.diag(Sig))
    np.savez('output/thrust_anchor_targets.npz', keys=np.array(keys), central=central, Sigma_c=Sig,
             variations=V, labels=np.array(labels))
    summ = dict(alpha_s=asc, alpha_s_err=sa, alpha_0=a0c, alpha_0_err=s0, rho=rho,
                chi2=fits[(1.0,1.0,'logR',1.0)][2], ndf=int(msk.sum())-2, shift=float(N.shift(1.0, a0c, asmz=asc)),
                alpha_s_range=[min(asr), max(asr)], alpha_0_range=[min(a0r), max(a0r)],
                mean_thrust=cen['tau'], coverage_pert=nin, coverage_with_exp=ninx,
                sigma_rel={k: float(s/abs(c)) for k, s, c in zip(keys, sd, central)},
                ratio_to_scet={k: cen[k]/N.SCET[k] for k in keys})
    json.dump(summ, open('output/chain_summary.json', 'w'), indent=1)
    print('wrote output/moments_joint.json, output/thrust_anchor_targets.npz, output/chain_summary.json')

if __name__ == '__main__':
    main()
