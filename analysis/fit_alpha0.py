#!/usr/bin/env python3
"""Fit the nonperturbative parameter to LEP thrust data, the way event-shape fits do it.

The matched NNLL+NNLO cumulant plus the dispersive shift is compared bin by bin with the
ALEPH 91.2 GeV thrust distribution, and alpha_0(mu_I) is fitted by chi^2 over a standard
fit range. This is the Dokshitzer-Webber programme as used in the LEP power-correction
analyses: the perturbative part is fixed, and the single NP parameter absorbs the leading
1/Q correction.

alpha_0 enters only through the shift delta = a_tau * P(alpha_0), so the scan is cheap: it
is a rigid translation of the tau grid, applied exactly as in np_shift.

Fit range follows the resummation literature for thrust at the Z pole, where the fixed-order
description is reliable and the shift is the whole NP effect: the default is
0.05 < tau < 0.33, with 0.06-0.30 reported as a stability check.
"""
import os
import numpy as np, pickle, re, sys, json
import np_shift as N

MATCH = sys.argv[1] if len(sys.argv) > 1 else 'output/matched_v3.pkl'
RIVET = os.environ.get('RIVET_REF', 'rivet_ref')   # the Rivet reference tables
YODA = f'{RIVET}/ALEPH_2004_S5765862.yoda'


def aleph():
    txt = open(YODA).read()
    m = re.search(r'/REF/ALEPH_2004_S5765862/d54-x01-y01\n(.*?)\nEND YODA', txt, re.S)
    rows = []
    for ln in m.group(1).split('\n'):
        f = ln.split()
        if len(f) == 6 and not ln.startswith('#'):
            try: rows.append([float(x) for x in f])
            except ValueError: pass
    T, dTm, dTp, y, em, ep = np.array(rows).T
    lo, hi = 1-(T+dTp), 1-(T-dTm)                 # tau bin edges
    err = 0.5*(np.abs(em)+np.abs(ep))
    o = np.argsort(lo)
    return lo[o], hi[o], y[o], err[o]


LEP = {'aleph': ('ALEPH_2004_S5765862', 'd54-x01-y01', True),    # table in T, converted to tau = 1-T
       'delphi': ('DELPHI_1996_S3430090', 'd11-x01-y01', False),  # table in 1-T
       'opal': ('OPAL_2004_S6132243', 'd01-x01-y01', False)}      # table in 1-T, 91 GeV


def lepdata(name='aleph'):
    """(lo, hi, y, err) of the 91 GeV thrust distribution of one LEP experiment, in tau = 1-T,
    from the Rivet reference tables (HEPData values). The calculation is fitted to ALEPH only;
    the other two never enter a fit and serve as blind tests."""
    ana, tab, in_T = LEP[name]
    txt = open(f'{RIVET}/{ana}.yoda').read()
    m = re.search(rf'/REF/{ana}/{tab}\n(.*?)\nEND YODA', txt, re.S)
    rows = []
    for ln in m.group(1).split('\n'):
        f = ln.split()
        if len(f) == 6 and not ln.startswith('#'):
            try: rows.append([float(x) for x in f])
            except ValueError: pass
    x, dxm, dxp, y, em, ep = np.array(rows).T
    lo, hi = (1-(x+dxp), 1-(x-dxm)) if in_T else (x-dxm, x+dxp)
    err = 0.5*(np.abs(em)+np.abs(ep)); o = np.argsort(lo)
    return lo[o], hi[o], y[o], err[o]


def binned(res, key, a0, lo, hi):
    """Theory bin density with the shift applied, same construction as np_shift.moments."""
    tau, sig, m = res[key]
    S, _ = N.sanitize(m/m[-1])
    t = tau + N.shift(key[0], a0)
    slo = N.cum_eval(lo, t, S)
    shi = N.cum_eval(hi, t, S)
    return (shi-slo)/(hi-lo)


def main():
    res = pickle.load(open(MATCH, 'rb'))
    lo, hi, y, e = aleph()
    ctr = 0.5*(lo+hi)
    out = {}
    for label, (a, b) in (('0.05-0.33', (0.05, 0.33)), ('0.06-0.30', (0.06, 0.30))):
        msk = (ctr > a) & (ctr < b)
        n = int(msk.sum())
        scan = []
        for a0 in np.arange(0.20, 0.85, 0.005):
            d = binned(res, (1.0, 1.0, 'logR'), a0, lo, hi)
            chi2 = float(np.sum(((d[msk]-y[msk])/e[msk])**2))
            scan.append((chi2, a0))
        scan.sort()
        chi2b, a0b = scan[0]
        # delta chi2 = 1 interval
        arr = np.array([(a0, c) for c, a0 in sorted(scan, key=lambda x: x[1])])
        inside = arr[arr[:, 1] <= chi2b+1]
        lo68, hi68 = inside[:, 0].min(), inside[:, 0].max()
        # theory envelope on alpha_0: refit under each variation
        var = []
        for key in res:
            s2 = sorted((float(np.sum(((binned(res, key, a0, lo, hi)[msk]-y[msk])/e[msk])**2)), a0)
                        for a0 in np.arange(0.20, 0.85, 0.01))
            var.append(s2[0][1])
        d = binned(res, (1.0, 1.0, 'logR'), a0b, lo, hi)
        out[label] = dict(alpha0=a0b, chi2=chi2b, ndf=n-1, chi2_ndf=chi2b/(n-1),
                          stat=[lo68, hi68], theory=[min(var), max(var)], nbins=n)
        print(f'fit range tau in ({a:.2f},{b:.2f}):  {n} bins')
        print(f'  alpha_0(2 GeV) = {a0b:.3f}  +{hi68-a0b:.3f}/-{a0b-lo68:.3f} (exp)  '
              f'[{min(var):.3f},{max(var):.3f}] (theory envelope)')
        print(f'  chi2/ndf = {chi2b:.1f}/{n-1} = {chi2b/(n-1):.2f}')
        print(f'  shift at best fit: delta = {N.shift(1.0, a0b):.5f}   '
              f'(was {N.shift(1.0, 0.50):.5f} at the assumed alpha_0=0.50)')
        # the mean at the fitted alpha_0
        _sh = N.shift(1.0, a0b)
        mo = N.moments(res[(1.0, 1.0, 'logR')][0] + _sh, res[(1.0, 1.0, 'logR')][2], lo=_sh)
        print(f'  resulting <1-T> = {mo["tau"]:.5f}   (data 0.0667, SCET NNLL\' 0.06638)')
        print()
    json.dump(out, open('output/fit_alpha0.json', 'w'), indent=1)
    print('wrote output/fit_alpha0.json')


if __name__ == '__main__':
    main()
