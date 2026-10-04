#!/usr/bin/env python3
"""Tail stress test: importance weights classically fail first in the tails. For each of the
12 validation runs, define the tail as the top 2.5 percent of the combined distribution in
multiplicity and in 1-T, then test three things per tail:
  rate     : reweighted tail probability against the fresh run's, as a z-score,
  N_eff    : the effective number of REFERENCE events inside the tail (weight concentration
             localized where it hurts),
  wmax     : the largest single-event share of the total tail weight.
Also the same for the LOW-multiplicity tail (bottom 2.5 percent), which is where hard
3-jet-like events live at the other end. Writes output/tail_stress.json."""
import json
import numpy as np
import pandas as pd
from ab_analysis import Cond
from r2_ladder import strange_frac

DATA = 'data_stageC'
POINTS = {
    6950: (0.1130, 0.33, 0.90), 6951: (0.1150, 0.52, 1.65), 6952: (0.1170, 0.62, 1.10),
    6953: (0.1190, 0.36, 1.40), 6954: (0.1210, 0.58, 0.85), 6955: (0.1230, 0.42, 1.70),
    6956: (0.1250, 0.50, 1.05), 6957: (0.1270, 0.32, 1.30), 6958: (0.1135, 0.63, 1.75),
    6959: (0.1275, 0.64, 0.82), 6960: (0.1180, 0.48, 0.95), 6961: (0.1220, 0.40, 1.55),
}
TAILS = [('mult_total', 'high', 0.975), ('mult_total', 'low', 0.025),
         ('1_minus_thrust', 'high', 0.975)]


def tail_stats(rv, w, tv, obs, side, q):
    thr = np.quantile(np.r_[rv, tv], q)
    if side == 'high':
        mr = rv >= thr; mt = tv >= thr
    else:
        mr = rv <= thr; mt = tv <= thr
    pw = w[mr].sum()                       # reweighted tail probability
    vw = (w[mr]**2).sum()                   # its variance contribution (weighted binomial)
    pt = mt.mean(); vt = pt*(1-pt)/len(tv)
    z = (pw-pt)/np.sqrt(max(vw + vt, 1e-300))
    wt = w[mr]
    neff = (wt.sum()**2/np.maximum((wt**2).sum(), 1e-300)) if wt.size else 0.0
    wmax = float(wt.max()/wt.sum()) if wt.size and wt.sum() > 0 else np.nan
    return dict(thr=float(thr), p_reweighted=float(pw), p_fresh=float(pt), z=float(z),
                neff_tail=float(neff), n_fresh_tail=int(mt.sum()), wmax_share=wmax)


def main():
    cond = Cond('C'); ref = np.load('output/models/C_ref.npz')
    robs = {'mult_total': ref['mult_total'].astype(float),
            '1_minus_thrust': ref['1_minus_thrust'].astype(float)}
    out = {}
    for rid, th in POINTS.items():
        sh = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        f = cond.f_at(np.array(th)); w = np.exp((f-f.max())/cond.T); w /= w.sum()
        row = {}
        for obs, side, q in TAILS:
            tv = sh[obs].values
            row[f'{obs}_{side}'] = tail_stats(robs[obs], w, tv, obs, side, q)
        out[rid] = row
    # aggregate
    agg = {}
    for obs, side, q in TAILS:
        key = f'{obs}_{side}'
        zs = [abs(out[r][key]['z']) for r in POINTS]
        ne = [out[r][key]['neff_tail'] for r in POINTS]
        wm = [out[r][key]['wmax_share'] for r in POINTS]
        rel = [abs(out[r][key]['p_reweighted']/out[r][key]['p_fresh']-1) for r in POINTS]
        agg[key] = dict(absz_mean=round(float(np.mean(zs)), 2), absz_max=round(float(np.max(zs)), 2),
                        rel_err_mean=round(float(np.mean(rel)), 4), rel_err_max=round(float(np.max(rel)), 4),
                        neff_tail_min=round(float(np.min(ne)), 0),
                        wmax_share_max=round(float(np.max(wm)), 5))
        print(f'{key:>22s}: |z| mean {agg[key]["absz_mean"]:.2f} max {agg[key]["absz_max"]:.2f}  '
              f'rel.err mean {100*agg[key]["rel_err_mean"]:.1f}% max {100*agg[key]["rel_err_max"]:.1f}%  '
              f'min tail N_eff {agg[key]["neff_tail_min"]:.0f}  max single-event share {agg[key]["wmax_share_max"]:.4f}')
    out['aggregate'] = agg
    json.dump(out, open('output/tail_stress.json', 'w'), indent=1)
    print('TAIL STRESS DONE')


if __name__ == '__main__':
    main()
