#!/usr/bin/env python3
"""Sec. 6.5: why the full mean thrust of the anchored sample lies below the measurements. The fit
normalizes data and prediction over the bins above tau = 0.05, so the fraction of events below
0.05 is left to the generators. At the fitted point: the fractions of the sample below 0.05, in the
window and above 1/3, before and after anchoring, against those of the three measured
distributions, and the part of the mean that the window fraction accounts for,
(P_win(data) - P_win(model)) * (<tau>_window - <tau>_below), with the data fraction averaged over the
three experiments.

    python peak_fraction.py output/profile_MIX17ext_central.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, TargetInterp, logw_continuous, fitted_point
from fit_alpha0 import lepdata

src = sys.argv[1]; fit = json.load(open(src))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype='float64', grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
a, a0, u, _ = fitted_point(src)
with torch.no_grad():
    lw, lw0, _, _ = logw_continuous(M, TargetInterp(M.G, 'central'), torch.tensor(u), a, a0)
thr = M.THRT.numpy(); WLO, WHI = 0.05, 1/3
regions = dict(below=thr < WLO, window=(thr >= WLO) & (thr < WHI), above=thr >= WHI)
out = {}
for name, w in (('unanchored', torch.exp(lw0).numpy()), ('anchored', torch.exp(lw).numpy())):
    out[name] = {r: float(w[m].sum()) for r, m in regions.items()}
    out[name].update({f'mean_{r}': float((w*thr)[m].sum()/w[m].sum()) for r, m in regions.items()}, mean=float(w @ thr))
for ex in ('aleph', 'delphi', 'opal'):
    lo, hi, y, e = lepdata(ex); p = y*(hi - lo); p = p/p.sum()
    out[ex] = dict(below=float(p[hi <= WLO + 1e-9].sum()), window=float(p[(lo >= WLO - 1e-9) & (hi <= WHI + 1e-3)].sum()))
an = out['anchored']; dwin = np.mean([out[ex]['window'] for ex in ('aleph', 'delphi', 'opal')]) - an['window']
out['window_shift'] = float(dwin*(an['mean_window'] - an['mean_below']))
out['data_below'] = float(np.mean([out[ex]['below'] for ex in ('aleph', 'delphi', 'opal')]))
print(json.dumps(out, indent=1))
json.dump(out, open(src.replace('.json', '_peak.json'), 'w'), indent=1)
