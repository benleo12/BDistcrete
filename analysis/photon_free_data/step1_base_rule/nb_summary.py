#!/usr/bin/env python3
"""Aggregate the unit records of nb_units.py.   python nb_summary.py WORK  ->  WORK/summary.json"""
import os, sys, json, glob
import numpy as np

WORK = sys.argv[1]
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
THRUST_EDGES = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .14, .16, .18, .20, .25, .30, .35, .40, .50]
RES_EDGES = np.arange(-12, 0.01, 0.25)
U = [json.load(open(f)) for f in sorted(glob.glob(f'{WORK}/units/*.json'))]
print(len(U), 'unit records')


def agg(units):
    out = dict(runs=len(units), events=int(sum(u['n_events'] for u in units)),
               beam_events=int(sum(u['n_beam_events'] for u in units)),
               nflag_hist=np.sum([u['nflag_hist'] for u in units], 0).tolist(),
               mean_zbeam=float(sum(u['sum_zbeam'] for u in units) / max(1, sum(u['n_events'] for u in units))),
               frac_ebeam_gt1GeV=float(sum(u['n_ebeam_gt1'] for u in units) / max(1, sum(u['n_events'] for u in units))),
               frac_ebeam_gt5GeV=float(sum(u['n_ebeam_gt5'] for u in units) / max(1, sum(u['n_events'] for u in units))),
               truncated=int(sum(u['n_truncated'] for u in units)),
               axis_on_beam=int(sum(u['corner']['axis_on_beam'] for u in units)),
               inconsistent=int(sum(u['corner']['inconsistent'] for u in units)),
               scale_lt91_1=int(sum(u['scale']['lt91_1'] for u in units)), scale_lt80=int(sum(u['scale']['lt80'] for u in units)),
               scale_clamped=int(sum(u['scale']['clamped'] for u in units)),
               unit_seconds=float(sum(u['time']['total'] for u in units)),
               max_unit_seconds=float(max(u['time']['total'] for u in units)), obs={})
    N = out['events']
    for o in OBS:
        x = [u['obs'][o] for u in units]
        ms, mn = sum(v['sum_stored'] for v in x) / N, sum(v['sum_new'] for v in x) / N
        hs, hn = np.sum([v['hist_stored'] for v in x], 0), np.sum([v['hist_new'] for v in x], 0)
        rec = dict(mean_stored=ms, mean_new=mn, rel_change_mean=mn / ms - 1,
                   per_run_rel_change_mean=dict(zip(('min', 'q05', 'median', 'q95', 'max'),
                       np.quantile([v['sum_new'] / v['sum_stored'] - 1 for v in x], [0, .05, .5, .95, 1]).tolist())),
                   n_nan_stored=int(sum(v['n_nan_stored'] for v in x)), n_neg_new=int(sum(v['n_neg_new'] for v in x)),
                   min_new=float(min(v['min_new'] for v in x)),
                   frac_stored=(hs / N).tolist(), frac_new=(hn / N).tolist())
        for key in ('residual_with', 'residual_control', 'fixed_vs_balance'):
            r = [v[key] for v in x if v.get(key) and v[key]['n'] > 0]
            if not r:
                continue
            n = sum(q['n'] for q in r); lh = np.sum([q['loghist'] for q in r], 0)
            cz = sum(q['n_zero'] + q['log_under'] for q in r)
            cdf = (cz + np.cumsum(lh)) / n

            def qq(p):
                k = int(np.searchsorted(cdf, p))
                return float(10 ** RES_EDGES[min(k + 1, len(RES_EDGES) - 1)])      # upper edge of the bin
            rec[key] = dict(n=n, max=float(max(q['max'] for q in r)),
                            frac_gt={t: sum(q['gt'][t] for q in r) / n for t in r[0]['gt']},
                            q50_upper=qq(.5), q90_upper=qq(.9), q99_upper=qq(.99), q999_upper=qq(.999))
        out['obs'][o] = rec
    # thrust bins: relative change of the fraction of all events, aggregated and per run
    x = [u['obs']['1_minus_thrust'] for u in units]
    hs = np.array([v['hist_stored'] for v in x], float); hn = np.array([v['hist_new'] for v in x], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_run = hn / hs - 1
    agg_rel = hn.sum(0) / hs.sum(0) - 1
    out['thrust_bins'] = [dict(lo=lo, hi=hi, frac_stored=float(hs.sum(0)[k + 1] / N), frac_new=float(hn.sum(0)[k + 1] / N),
                               rel_change=float(agg_rel[k + 1]),
                               per_run_q05=float(np.nanquantile(per_run[:, k + 1], .05)),
                               per_run_median=float(np.nanmedian(per_run[:, k + 1])),
                               per_run_q95=float(np.nanquantile(per_run[:, k + 1], .95)))
                          for k, (lo, hi) in enumerate(zip(THRUST_EDGES[:-1], THRUST_EDGES[1:]))]
    out['thrust_under_over'] = dict(stored=[int(hs.sum(0)[0]), int(hs.sum(0)[-1])], new=[int(hn.sum(0)[0]), int(hn.sum(0)[-1])])
    return out


groups = sorted({u['group'] for u in U})
S = dict(groups={g: agg([u for u in U if u['group'] == g]) for g in groups})
v3 = [u for u in U if u['group'] in ('data_release', 'data_v3')]
S['mixture_runs'] = {gen: agg([u for u in v3 if u['gen'] == gen]) for gen in ('S', 'H')}
S['all'] = agg(U)
S['sherpa_all'] = agg([u for u in U if u['gen'] == 'S'])
S['herwig_all'] = agg([u for u in U if u['gen'] == 'H'])
json.dump(S, open(f'{WORK}/summary.json', 'w'), indent=1)


def line(name, a):
    o = a['obs']
    return (f"{name:26s} runs {a['runs']:5d} events {a['events']:10d} with beam photons {a['beam_events'] / a['events']:.4f} | "
            + ' '.join(f"{k}: {o[k]['mean_stored']:.5f}->{o[k]['mean_new']:.5f} ({100 * o[k]['rel_change_mean']:+.2f}%)" for k in OBS))


for g in groups:
    print(line(g, S['groups'][g]))
for gen in ('S', 'H'):
    print(line(f'mixture runs {gen}', S['mixture_runs'][gen]))
print(line('all', S['all']))
for gen in ('S', 'H'):
    print(f'thrust bins, mixture runs {gen} (rel. change of the fraction of all events, aggregated; per run q05/median/q95):')
    for b in S['mixture_runs'][gen]['thrust_bins']:
        print(f"   [{b['lo']:.2f},{b['hi']:.2f})  {b['frac_stored']:.5f} -> {b['frac_new']:.5f}  {100 * b['rel_change']:+6.2f}%   "
              f"({100 * b['per_run_q05']:+.2f}, {100 * b['per_run_median']:+.2f}, {100 * b['per_run_q95']:+.2f})")
a = S['all']
for o in OBS:
    for key in ('residual_with', 'residual_control', 'fixed_vs_balance'):
        r = a['obs'][o].get(key)
        if r:
            print(f"{o:15s} {key:17s} n {r['n']:10d} max {r['max']:.1e} q50<{r['q50_upper']:.0e} q90<{r['q90_upper']:.0e} "
                  f"q99<{r['q99_upper']:.0e} q999<{r['q999_upper']:.0e} frac>1e-4 {r['frac_gt']['0.0001']:.2e} >1e-3 {r['frac_gt']['0.001']:.2e} >1e-2 {r['frac_gt']['0.01']:.2e}")
print('corner cases (events): axis_on_beam', a['axis_on_beam'], 'inconsistent', a['inconsistent'], 'truncated', a['truncated'],
      'neg new', {o: a['obs'][o]['n_neg_new'] for o in OBS}, 'nan stored', {o: a['obs'][o]['n_nan_stored'] for o in OBS})
print('unit process-seconds', round(a['unit_seconds']), 'max unit', round(a['max_unit_seconds']))
