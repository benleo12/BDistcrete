#!/usr/bin/env python3
"""Re-fit a head's calibration temperature with the SAME objective the training run used,
|master closure width - 1| on the stop half of the held runs, over the grid the training run
chose from (r2_ladder.T_GRID: 0.85 to 1.60 in steps of 0.025, with 0.5 to 0.8 and 1.75 to 3.0 at
the ends). It reports the width on the stop half (the selection set), on the disjoint report half
(the paper's reported number), and the mean-closure chi2 of the observable means on both, so the
choice can be seen rather than trusted. The best value on the older grid [0.5, 0.6, 0.7, 0.8, 0.9,
1.0, 1.2, 1.5, 2.0, 3.0] is reported too ('best_coarse'); the trainings of the paper chose from
r2_ladder.T_GRID ('best_fine', also written as 'T'). Writes output/recal_T_<tag>.json
(RECAL_OUT overrides it) and does not touch any export: applying the value is a separate,
recorded step (apply_T.py).

    python recalibrate_T.py C_1M

Inputs, all explicit for the rerun:
  RECAL_EXPORT   the export (default output/models/<tag>_cond.npz)
  RECAL_REF      the reference bundle of its events (default output/models/<tag>_ref.npz)
  DM_DATA        the data set of a mixture (its meta.json gives the held-out runs and their points);
                 STAGE<S>_DATA (and STAGEE_CSV/STAGEF_CSV, LADDER_ABC_DESIGN) for a single stage
D4 (score_obs.py, BEAM_PHOTONS=remove): photon-free shapes files, the multiplicity and the kaon
fraction without the beam photons on both sides, the reference's from RECAL_REF_PARTICLES (default
RECAL_REF), its shapes checked against the old bundle RECAL_REF_ORIG.
D11: with T_DIR set the result is also written to T_DIR/<model>.json, for every later step.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head, MixtureHead
import r2_ladder as R
from make_stage_table import ALIAS, design_key
import score_obs as SO

tag = sys.argv[1] if len(sys.argv) > 1 else 'C_1M'
dtag = design_key(tag, R.STAGES, ALIAS)
if dtag not in R.STAGES:
    # a mixture head registers its design from its data set's meta.json at run time
    SO.need(f"{os.environ['DM_DATA']}/meta.json")
    from mixture_cfg import register as _register_mixture
    if _register_mixture(R.STAGES, [tag], os.environ['DM_DATA']) is not None:
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
if SO.REMOVE and dtag in 'ABCEF':
    assert os.environ.get(f'STAGE{dtag}_DATA'), f'set STAGE{dtag}_DATA: the stage has a default data folder'
cfg = R.STAGES[dtag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
# held runs that duplicate a training run (Sherpa's default seed) test nothing and are left out,
# as in ladder_widths_dedup.py: DROP=9932 for the mixture stage. The trainings of the paper kept
# every held run, so a refit of their temperature (D11) leaves DROP unset.
for _r in [int(x) for x in os.environ.get('DROP', '').split(',') if x]:
    cfg['held'].pop(_r, None)
hp = SO.need(os.environ.get('RECAL_EXPORT', f'output/models/{tag}_cond.npz'))
kind = str(np.load(hp).get('head_kind', 'cond'))
# the multiplicative mixture (geometric) is a mixture head as well: Head would read its first block only
h = (MixtureHead if kind in ('mixture', 'geometric') else Head)(hp)
A = h.pack(np.asarray(np.load(hp, mmap_mode='r')['AE']))
REF = SO.need(os.environ.get('RECAL_REF', f'output/models/{tag}_ref.npz'))
REF_PARTICLES = os.environ.get('RECAL_REF_PARTICLES', REF); REF_ORIG = os.environ.get('RECAL_REF_ORIG', '')
robs = SO.reference_obs(REF, OBS, particles=REF_PARTICLES, orig=REF_ORIG or None)
assert len(robs[OBS[0]]) == A.shape[0], f'{REF}: {len(robs[OBS[0]])} reference events, the export {A.shape[0]}'
for _o in OBS:
    R.register_sorted(robs[_o])        # LADDER_FAST_HIST=1: the same histograms, without sorting them again per T
print(f'{tag}: shipped T = {h.T}, {len(robs[OBS[0]]):,} reference events, {len(cfg["held"])} held runs, '
      f'data {DATA}, beam photons {SO.MODE}')

def targets(rid):
    sh = SO.read_shapes(f'{DATA}/shapes_run_{rid:04d}.csv')
    out = {o: sh[o].values.astype(float) for o in OBS if o in cfg['obs']}
    if set(OBS) & {'nbaryon', 'strange'} or (SO.REMOVE and 'mult_total' in out):
        d = np.load(SO.need(f'{DATA}/particles_full_{rid:04d}.npz'))
        P, M = d['particles'], d['mask']
        assert len(M) == len(sh), f'{rid}: shapes {len(sh)} events, particles {len(M)}'
        nb = SO.n_beam(P, M) if SO.REMOVE else None
        if 'nbaryon' in OBS:
            out['nbaryon'] = d['nbaryon'].astype(float)
        if 'strange' in OBS:
            out['strange'] = SO.strange_frac(P, M, nb=nb).astype(float)
        if 'mult_total' in out:
            out['mult_total'] = SO.mult_total(out['mult_total'], P, M, where=f'{DATA} run {rid}', nb=nb).astype(float)
    return out

TG, SPLIT, BINS, LOG = {}, {}, {}, {}
for rid, th in cfg['held'].items():
    t = targets(rid); n = len(t[OBS[0]])
    for o in OBS:
        TG[rid, o] = t[o]
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

grid = R.T_GRID
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
at_shipped = [r for r in rows if abs(r['T'] - h.T) < 1e-12]
print(f'\ntraining grid: T = {best["T"]:.3f}  (stop width {best["width_stop"]:.3f}, report width {best["width_report"]:.3f})'
      f'  <- what the training run chose from; shipped T = {h.T}')
print(f'older grid:    T = {best_coarse["T"]:.3f}  (stop width {best_coarse["width_stop"]:.3f}, report width {best_coarse["width_report"]:.3f})')
OUT = os.environ.get('RECAL_OUT', f'output/recal_T_{tag}.json')
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
rec = dict(tag=tag, shipped_T=h.T, T=best['T'], changed=bool(abs(best['T'] - h.T) > 1e-12),
           objective='|master width - 1| on the stop half, the training objective (r2_ladder.fit_temperature)',
           grid='r2_ladder.T_GRID', best_fine=best, best_coarse=best_coarse,
           at_shipped=at_shipped[0] if at_shipped else None, scan=rows,
           export=hp, ref=REF, ref_particles=REF_PARTICLES, ref_orig=REF_ORIG or None, data=DATA,
           data_realpath=os.path.realpath(DATA), **SO.describe())
json.dump(rec, open(OUT, 'w'), indent=1)
print(f'wrote {OUT}')
if SO.T_DIR:
    SO.write_temperature(SO.model_key(hp), dict(
        T=best['T'], T_trained=h.T, stop_width=best['width_stop'],
        stop_width_at_trained_T=at_shipped[0]['width_stop'] if at_shipped else None,
        objective=rec['objective'], grid=list(grid), data=DATA, data_realpath=rec['data_realpath'],
        beam_photons=SO.MODE, export=hp, ref=REF, tool='recalibrate_T.py'))
