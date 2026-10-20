#!/usr/bin/env python3
"""Why the prediction of Sec. 4 lies below DELPHI above tau = 0.2: the NNLL+NLO calculation at the paper's
fixed point (chain_summary_nlo.json), ALEPH merged into DELPHI's bins, and the prediction with the
calculation imposed (three_bands npz), each against DELPHI, in the twelve DELPHI bins 0.05 <= tau < 0.30
that lie wholly inside the window, all normalized over these twelve bins. The calculation is binned with
the functions of fig_thrust_vs_aleph.py (the head of that script, before its plotting, is executed here
so that the two use the same code). Writes output/calc_vs_delphi_<tag>.json.

    NPZ=output/three_bands_MIXGEOV3S9.npz python calc_vs_delphi.py MIXGEOV3S9
"""
import os, sys, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else 'MIXGEOV3S9'
os.environ['THRUST_ORDER'] = '2'; os.environ['CHAIN_SUFFIX'] = '_nlo'; os.environ.pop('FIXED_POINT', None)
head = open('fig_thrust_vs_aleph.py').read().split('lo, hi, y, e = aleph()')[0]
assert 'def binned' in head and 'def build' in head
exec(compile(head.replace('setup(usetex=True)', ''), 'fig_thrust_vs_aleph.py', 'exec'))
sys.path.insert(0, 'lepcov')
from lepdata_cov import lepdata_split
CS = json.load(open('output/chain_summary_nlo.json')); ASC, A0C = CS['alpha_s'], CS['alpha_0']
la, ha, ya, sta, sya = lepdata_split('aleph'); ld, hd, yd, std, syd = lepdata_split('delphi')
Z = np.load(os.environ.get('NPZ', f'output/three_bands_{tag}.npz'))
assert abs(ASC - 0.11350) < 1e-4 and abs(A0C - 0.4312) < 1e-3, (ASC, A0C)
k12 = (ld >= 0.05 - 1e-9) & (hd <= 0.30 + 1e-9); lo, hi = ld[k12], hd[k12]; wid = hi - lo
D, De = yd[k12], np.hypot(std[k12], syd[k12])
AL, ALe = [], []
for l, h in zip(lo, hi):                       # ALEPH's bins merged into DELPHI's, systematics linear (conservative)
    k = (la >= l - 1e-9) & (ha <= h + 1e-9); w = ha[k] - la[k]; assert np.isclose(w.sum(), h - l)
    AL.append((ya[k]*w).sum()/(h - l)); ALe.append(np.hypot(np.sqrt(((sta[k]*w)**2).sum()), (sya[k]*w).sum())/(h - l))
AL, ALe = np.array(AL), np.array(ALe)   # not A, which the executed head uses for the grid
kb = (Z['lo'] >= 0.05 - 1e-9) & (Z['hi'] <= 0.30 + 1e-9)
assert np.allclose(Z['lo'][kb], lo) and np.allclose(Z['hi'][kb], hi)
n = lambda v: v/(v*wid).sum()
# the DELPHI values of the npz are those of lepdata_split, up to the normalization
assert np.allclose(n(Z['data'][kb]), n(D), rtol=1e-6), 'DELPHI values differ'
calc = binned(1.0, 1.0, 'logR', ASC, A0C, lo, hi); modr = binned(1.0, 1.0, 'modR', ASC, A0C, lo, hi)
imp = Z['h_central'][kb]
out = dict(tag=tag, alpha_s=ASC, alpha_0=A0C, lo=lo.tolist(), hi=hi.tolist(),
           calc_over_delphi=(n(calc)/n(D)).tolist(), modR_over_delphi=(n(modr)/n(D)).tolist(),
           aleph_over_delphi=(n(AL)/n(D)).tolist(), aleph_rel_err=(ALe/AL).tolist(), delphi_rel_err=(De/D).tolist(),
           imposed_over_calc=(n(imp)/n(calc)).tolist(), imposed_over_delphi=(n(imp)/n(D)).tolist())
json.dump(out, open(f'output/calc_vs_delphi_{tag}.json', 'w'), indent=1)
print(f"{'bin':10s} {'calc/D':>7s} {'modR/D':>7s} {'ALEPH/D':>7s} {'imp/D':>7s} {'imp/calc':>8s} {'D err%':>6s} {'A err%':>6s}")
for i in range(len(lo)):
    print(f"{lo[i]:.2f}-{hi[i]:.2f} {out['calc_over_delphi'][i]:7.3f} {out['modR_over_delphi'][i]:7.3f} {out['aleph_over_delphi'][i]:7.3f} "
          f"{out['imposed_over_delphi'][i]:7.3f} {out['imposed_over_calc'][i]:8.3f} {100*De[i]/D[i]:6.2f} {100*ALe[i]/AL[i]:6.2f}")
print('wrote', f'output/calc_vs_delphi_{tag}.json')
