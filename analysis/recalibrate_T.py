#!/usr/bin/env python3
"""Re-fit a released head's calibration temperature with the SAME objective the training run
used, |master closure width - 1| on the stop half of the held runs, over a fine grid.

The training run searched [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0], which has no
point between 1.0 and 1.2. This scans 0.85 to 1.60 in steps of 0.025 with the coarse points
kept at both ends, and reports the width on the stop half (the selection set), on the disjoint
report half (the paper's reported number), and the mean-closure chi2 of the observable means on
both, so the choice can be seen rather than trusted. Writes output/recal_T_<tag>.json and does
not touch any export: applying the value is a separate, recorded step.

    python recalibrate_T.py C_1M
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head, MixtureHead
import r2_ladder as R
from make_stage_table import strange_frac, ALIAS, design_key

tag = sys.argv[1] if len(sys.argv) > 1 else 'C_1M'
dtag = design_key(tag, R.STAGES, ALIAS)
if dtag not in R.STAGES:
    # a mixture head registers its design from its data set's meta.json at run time
    from mixture_cfg import register as _register_mixture
    if _register_mixture(R.STAGES, [tag]) is not None:
        dtag = tag
if dtag not in R.STAGES and os.path.exists(f"{os.environ.get('DM_DATA', '')}/meta.json"):
    # the paper's seven-parameter Stage D mixture predates the box being recorded in meta.json,
    # so its box is the literal that stageDM.py carries, reproduced here verbatim
    import json as _json
    _meta = _json.load(open(f"{os.environ['DM_DATA']}/meta.json"))
    if 's_box' not in _meta and len(_meta['train'][0]['theta']) == 7:
        R.STAGES[tag] = dict(
            data=os.environ['DM_DATA'], flavor=True, ntheta=7,
            train={m['rid']: tuple(m['theta']) for m in _meta['train']},
            held={m['rid']: tuple(m['theta']) for m in _meta['held']},
            norm=[(0.120, 0.008), (0.475, 0.175), (1.30, 0.50), (0.1186, 0.008), (0.325, 0.175), (3.80, 1.00), (0.5, 0.5)],
            obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'], flav_obs=['nbaryon', 'strange'])
        dtag = tag
assert dtag in R.STAGES, f'no design for {tag}'
cfg = R.STAGES[dtag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
# held runs that duplicate a training run (Sherpa's default seed) test nothing and are left out,
# as in ladder_widths_dedup.py: DROP=9932 for the mixture stage
for _r in [int(x) for x in os.environ.get('DROP', '').split(',') if x]:
    cfg['held'].pop(_r, None)
# the ladder heads A, B, C are not in the release, so fall back to the raw export, which the
# reader accepts and which carries the same temperature field
hp = f'release/models/{tag}_head.npz'
if not os.path.exists(hp):
    hp = f'output/models/{tag}_cond.npz'
kind = str(np.load(hp).get('head_kind', 'cond'))
h = (MixtureHead if kind == 'mixture' else Head)(hp)
A = h.pack(np.asarray(np.load(f'output/models/{tag}_cond.npz', mmap_mode='r')['AE']))
ref = np.load(f'output/models/{tag}_ref.npz', mmap_mode='r')
robs = {o: np.asarray(ref[o], np.float64) for o in OBS}
print(f'{tag}: shipped T = {h.T}, {len(robs[OBS[0]]):,} reference events, {len(cfg["held"])} held runs')

def target(rid, o):
    if o in ('nbaryon', 'strange'):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        return (d['nbaryon'].astype(float) if o == 'nbaryon'
                else strange_frac(d['particles'], d['mask']).astype(float))
    return pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')[o].values.astype(float)

TG, SPLIT, BINS, LOG = {}, {}, {}, {}
for rid, th in cfg['held'].items():
    n = None
    for o in OBS:
        t = target(rid, o); TG[rid, o] = t; n = len(t)
    perm = np.random.default_rng(1000 + rid).permutation(n)
    SPLIT[rid] = dict(stop=perm[:n//2], report=perm[n//2:])
    for o in OBS:   # bins fixed from reference + REPORT half, as in run_stage
        BINS[rid, o] = R.make_bins(np.r_[robs[o], TG[rid, o][SPLIT[rid]['report']]], o)
    LOG[rid] = h.logit(A, np.array(th))

def widths(T, which):
    out = {}
    for rid in cfg['held']:
        f = LOG[rid]/T; w = np.exp(f - f.max()); w /= w.sum()
        po = {}
        for o in OBS:
            p, nb = R.pulls(robs[o], w, TG[rid, o][SPLIT[rid][which]], BINS[rid, o])
            po[o] = R.width_of(p)
        out[rid] = po
    return out

def master(cl): return float(np.mean([np.mean(list(po.values())) for po in cl.values()]))

def mean_chi2(T, which, obs):
    c = 0.0
    for rid in cfg['held']:
        f = LOG[rid]/T; w = np.exp(f - f.max()); w /= w.sum(); neff = 1/np.sum(w**2)
        for o in obs:
            t = TG[rid, o][SPLIT[rid][which]]; mt, et = t.mean(), t.std(ddof=1)/np.sqrt(len(t))
            mw = np.sum(w*robs[o]); ew = np.sqrt(np.sum(w*(robs[o] - mw)**2)/neff)
            c += ((mw - mt)/np.hypot(et, ew))**2
    return c/(len(cfg['held'])*len(obs))

grid = sorted(set([0.5, 0.6, 0.7, 0.8] + [round(x, 3) for x in np.arange(0.85, 1.60, 0.025)] + [1.75, 2.0, 2.5, 3.0]))
rows = []
print(f'{"T":>6} {"width stop":>11} {"width report":>13} {"chi2 stop(4)":>13} {"chi2 report(4)":>15}')
for T in grid:
    ws, wr = master(widths(T, 'stop')), master(widths(T, 'report'))
    cs, cr = mean_chi2(T, 'stop', cfg['obs']), mean_chi2(T, 'report', cfg['obs'])
    rows.append(dict(T=T, width_stop=ws, width_report=wr, chi2_stop=cs, chi2_report=cr))
    print(f'{T:6.3f} {ws:11.3f} {wr:13.3f} {cs:13.2f} {cr:15.2f}')
best = min(rows, key=lambda r: abs(r['width_stop'] - 1.0))
coarse = [r for r in rows if r['T'] in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0)]
best_coarse = min(coarse, key=lambda r: abs(r['width_stop'] - 1.0))
print(f'\nfine grid:   T = {best["T"]:.3f}  (stop width {best["width_stop"]:.3f}, report width {best["width_report"]:.3f})')
print(f'coarse grid: T = {best_coarse["T"]:.3f}  (stop width {best_coarse["width_stop"]:.3f}, report width {best_coarse["width_report"]:.3f})  <- what the training run could choose')
os.makedirs('output', exist_ok=True)
json.dump(dict(tag=tag, shipped_T=h.T, objective='|master width - 1| on the stop half, the training objective',
               best_fine=best, best_coarse=best_coarse, scan=rows), open(f'output/recal_T_{tag}.json', 'w'), indent=1)
print(f'wrote output/recal_T_{tag}.json')
