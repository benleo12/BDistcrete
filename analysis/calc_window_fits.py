#!/usr/bin/env python3
"""Fits of the thrust calculation alone to the LEP thrust data inside the window, no generator.

The numbers behind Sec. 6.4 of the paper, on the widened coupling grid (ARES_EXT, alpha_s 0.100 to
0.136). A fit whose minimum or error crossing reaches the end of that range is flagged EDGE, and its
value is then a bound within the range only. For a range of bins the
calculation's bin densities at (alpha_s, alpha_0) are compared with the data of the three experiments
in two normalizations:
  own          the experiments' published densities, normalized to the total cross section, against
               the calculation normalized to its own total, so the fraction of events in the range
               enters the fit;
  conditional  prediction and data both normalized over the bins used, so only the shape inside the
               range enters, as in the anchored fit.
For each range the coupling and its errors come from profile_errors (fitlib.py) on a grid of
alpha_s (the ARES nodes) and alpha_0 (0.15 to 0.85 in steps of 0.005).

Every chi^2 quoted is the spline-refined minimum of profile_errors, also for the common pair, so
differences are taken on one convention. Reported: the two halves of the window split at tau = 0.10
and the whole window in both normalizations, the cost of a common pair, the window fits of the three experiments together and of
ALEPH alone, and the DELPHI window chi^2 at its best pair for the central calculation and for each
of the eleven variations, refitted.

    python calc_window_fits.py
"""
import os, sys, json
os.environ['ARES_EXT'] = '1'
sys.argv = sys.argv[:1]
import numpy as np
sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
from fit_alpha0 import lepdata
import fitlib

EXPS = ('aleph', 'delphi', 'opal')
DATA = {ex: lepdata(ex) for ex in EXPS}
ASG = list(TC.ASGRID); A0G = np.round(np.arange(0.15, 0.8501, 0.005), 4)
VARS = [(1.0, 1.0, 'logR', 1.0), (2.0, 1.0, 'logR', 1.0), (0.5, 1.0, 'logR', 1.0), (1.0, 2.0, 'logR', 1.0),
        (1.0, 0.5, 'logR', 1.0), (1.0, 1.0, 'modR', 1.0), (2.0, 1.0, 'modR', 1.0), (0.5, 1.0, 'modR', 1.0),
        (1.0, 2.0, 'modR', 1.0), (1.0, 0.5, 'modR', 1.0), (1.0, 1.0, 'logR', 1.05), (1.0, 1.0, 'logR', 0.95)]
assert len(ASG) == 19, 'the widened ARES table is not selected'

def densities(var):
    """Bin densities of every experiment on the (alpha_s, alpha_0) grid: dict ex -> (nA, n0, nbins)."""
    mu, xv, sch, cf = var
    out = {ex: np.zeros((len(ASG), len(A0G), len(DATA[ex][0]))) for ex in EXPS}
    for ia, a in enumerate(ASG):
        tau, m = TC.build(mu, xv, sch, a, cf); S, _ = N.sanitize(m/m[-1])
        for j, a0 in enumerate(A0G):
            t = tau + N.shift(mu, a0, asmz=a)
            for ex in EXPS:
                lo, hi = DATA[ex][0], DATA[ex][1]
                out[ex][ia, j] = (N.cum_eval(hi, t, S) - N.cum_eval(lo, t, S))/(hi - lo)
    return out

def chi2_surface(D, exps, lo_cut, hi_cut, mode):
    """chi^2 on the grid over the bins of exps inside [lo_cut, hi_cut], and the number of bins."""
    c2 = np.zeros((len(ASG), len(A0G))); n = 0
    for ex in exps:
        lo, hi, y, e = DATA[ex]
        sel = (lo >= lo_cut - 1e-9) & (hi <= hi_cut + 1e-9); n += int(sel.sum())
        d = D[ex][..., sel]; yy, ee, wid = y[sel], e[sel], (hi - lo)[sel]
        if mode == 'conditional':
            d = d/np.sum(d*wid, -1, keepdims=True); s = np.sum(yy*wid); yy, ee = yy/s, ee/s
        c2 += np.sum(((d - yy)/ee)**2, -1)
    return c2, n

def fit(D, exps, lo_cut, hi_cut, mode):
    c2, n = chi2_surface(D, exps, lo_cut, hi_cut, mode)
    pe = fitlib.profile_errors(c2, np.array(ASG), A0G)
    a, a0 = pe['alpha_s'], pe['alpha_0']
    return dict(n=n, alpha_s=a['value'], err_lo=a['err_lo'], err_hi=a['err_hi'], alpha_0=a0['value'],
                chi2=float(a['chi2min']), grid_min=float(c2.min()), edge=bool(a['hit_edge'] or a0['hit_edge']))

def show(label, r):
    eh = f"+{r['err_hi']:.4f}" if r['err_hi'] is not None else '+(off grid)'
    el = f"-{r['err_lo']:.4f}" if r['err_lo'] is not None else '-(off grid)'
    print(f"  {label:44s} {r['n']:3d} bins  alpha_s {r['alpha_s']:.4f} {eh} {el}  alpha_0 {r['alpha_0']:.3f}  "
          f"chi2 {r['chi2']:.1f}{'  EDGE' if r['edge'] else ''}", flush=True)

res = {}
Dc = densities(VARS[0])
W0, W1, SPLIT = 0.05, 1/3, 0.10
for mode in ('own', 'conditional'):
    print(f'{mode} normalization:')
    lo_ = fit(Dc, EXPS, W0, SPLIT, mode); up_ = fit(Dc, EXPS, SPLIT, W1, mode); wh_ = fit(Dc, EXPS, W0, W1, mode)
    show('window below 0.10, three experiments', lo_); show('window above 0.10, three experiments', up_)
    show('whole window, three experiments', wh_)
    # the common pair: both halves at one (alpha_s, alpha_0), the straddling bins left out as in the halves
    c_lo, _ = chi2_surface(Dc, EXPS, W0, SPLIT, mode); c_up, _ = chi2_surface(Dc, EXPS, SPLIT, W1, mode)
    pc = fitlib.profile_errors(c_lo + c_up, np.array(ASG), A0G)
    common = float(pc['alpha_s']['chi2min']); dchi = common - lo_['chi2'] - up_['chi2']
    print(f'  one pair for both halves: chi2 {common:.1f}, cost of the common pair {dchi:.1f} for two parameters')
    al = fit(Dc, ('aleph',), W0, W1, mode); show('whole window, ALEPH alone', al)
    res[mode] = dict(lower=lo_, upper=up_, whole=wh_, common_chi2=common, delta_chi2=dchi, aleph=al)
print('DELPHI window bins, conditional normalization, and ALEPH window bins, own normalization (App. D), under each variation:')
dl, al = [], []
for var in VARS:
    D = Dc if var == VARS[0] else densities(var)
    r = fit(D, ('delphi',), W0, W1, 'conditional'); r['variation'] = list(var); dl.append(r)
    show(f'DELPHI, {var}', r)
    r = fit(D, ('aleph',), W0, W1, 'own'); r['variation'] = list(var); al.append(r)
    show(f'ALEPH own, {var}', r)
res['delphi_window'] = dl; res['aleph_own_variations'] = al
print(f"DELPHI window chi2: central {dl[0]['grid_min']:.1f}, best over the variations {min(r['grid_min'] for r in dl[1:]):.1f}")
json.dump(res, open('output/calc_window_fits.json', 'w'), indent=1)
print('wrote output/calc_window_fits.json')
