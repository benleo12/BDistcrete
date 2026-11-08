#!/usr/bin/env python3
"""Binned closure widths (the statistic of Sec. 3.2, the convention of Table 1) for the
eight-parameter Sherpa and Herwig stages and the seventeen-parameter mixture, on the corrected
shapes. Same estimator as ladder_widths_v2.py: bins from reference plus the report half of each
held run, per-bin pulls with both sides' multinomial errors, width = sqrt(chi2/ndf) about zero,
master width = mean over held runs and observables, at the shipped temperature.

    python ladder_widths_big.py E output/models/E_cond.npz output/models/E_ref.npz
    STAGEF_CSV=stageF_design_aug.csv python ladder_widths_big.py F output/models/Fauglong_cond.npz output/models/Fauglong_ref.npz
    DM_DATA=data_stageDM17aug_v2 python ladder_widths_big.py MIX output/models/MIX17aug_cond.npz output/models/MIX17aug_ref.npz

The rerun (score_obs.py): BEAM_PHOTONS=remove takes the beam photons out of the multiplicity and the
kaon fraction on both sides, reads only photon-free shapes files, and checks that the reference
bundle holds the photon-free shapes against the old bundle REF_ORIG; the multiplicity and kaon
fraction of the reference come from the particles of the same events, REF_PARTICLES (cached in
POBS_DIR when set). With T_DIR set the export must carry the refitted temperature of its model.
WIDTHS_OUT names the output file.
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head, MixtureHead
import r2_ladder as R
import score_obs as SO
tag, export, refpath = sys.argv[1], sys.argv[2], sys.argv[3]
SO.need(export, refpath)
if tag == 'MIX':
    from mixture_cfg import mixture_cfg
    SO.need(f"{os.environ['DM_DATA']}/meta.json")
    cfg, _ = mixture_cfg(os.environ['DM_DATA'])
else:
    if SO.REMOVE:
        assert os.environ.get(f'STAGE{tag}_DATA'), f'set STAGE{tag}_DATA: the stage has a default data folder'
    cfg = R.STAGES[tag]
DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
kind = str(np.load(export).get('head_kind', 'cond'))
h = (MixtureHead if kind in ('mixture', 'geometric') else Head)(export)
SO.check_export_T(export, h.T)
A = h.pack(np.asarray(np.load(export, mmap_mode='r')['AE']))
REF_PARTICLES, REF_ORIG = os.environ.get('REF_PARTICLES', ''), os.environ.get('REF_ORIG', '')
POBS = os.environ.get('POBS_DIR', '')
if SO.REMOVE:
    assert REF_PARTICLES and REF_ORIG, 'BEAM_PHOTONS=remove: set REF_PARTICLES and REF_ORIG'
cache = (os.path.join(POBS, os.path.basename(os.path.realpath(REF_PARTICLES)).replace('.npz', '_pobs.npz'))
         if POBS and REF_PARTICLES else None)
ref = np.load(refpath); robs = SO.reference_obs(refpath, OBS, particles=REF_PARTICLES or None, orig=REF_ORIG or None, cache=cache)
assert len(robs[OBS[0]]) == A.shape[0]
# REF_PER_RUN=k scores against the first k stored reference events of every training run, a smaller
# reference with the same density (the stored reference holds nper random events of each run, run by
# run in the order of 'order'). The closure width grows with the size of the reference, so models
# trained on different numbers of runs are compared at equal reference size this way.
KPER = int(os.environ.get('REF_PER_RUN', '0'))
if KPER:
    nper, nrun = int(ref['nper']), len(ref['order'])
    assert nper*nrun == A.shape[0] and KPER <= nper, (nper, nrun, A.shape)
    keep = (np.arange(nrun)[:, None]*nper + np.arange(KPER)[None, :]).ravel()
    A = A[keep]; robs = {o: v[keep] for o, v in robs.items()}
    print(f'reference: {KPER} of {nper} events of each of {nrun} runs, {len(keep)} in all', flush=True)
def targets(rid):
    """the observables of one held-out run: the shapes file, and the particle file for the baryon
    count, the kaon fraction and (under BEAM_PHOTONS=remove) the multiplicity"""
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
        if SO.REMOVE and 'mult_total' in out:
            out['mult_total'] = SO.mult_total(out['mult_total'], P, M, where=f'{DATA} run {rid}', nb=nb).astype(float)
    return {o: out[o] for o in OBS}
uni = np.full(len(robs[OBS[0]]), 1.0/len(robs[OBS[0]]))
per = {}; uper = {}; neffs = {}; nrep = {}
for rid, th in cfg['held'].items():
    T_ = targets(rid); n = len(T_[OBS[0]])
    perm = np.random.default_rng(1000 + rid).permutation(n); rep = perm[n//2:]
    w = h.weights(A, np.array(th)); neffs[rid] = h.n_eff(w)/len(w); nrep[rid] = len(rep)
    for o in OBS:
        b = R.make_bins(np.r_[robs[o], T_[o][rep]], o)
        p, nb = R.pulls(robs[o], w, T_[o][rep], b); per[rid, o] = R.width_of(p)
        pu, _ = R.pulls(robs[o], uni, T_[o][rep], b); uper[rid, o] = R.width_of(pu)
master = float(np.mean([per[k] for k in per])); umaster = float(np.mean([uper[k] for k in uper]))
print(f'{tag} ({os.path.basename(export)}) on {DATA}, T={h.T}: report-split width {master:.3f} '
      f'(uniform control {umaster:.2f}), N_eff/N {min(neffs.values()):.2f} to {max(neffs.values()):.2f}, {len(cfg["held"])} held runs', flush=True)
res = dict(tag=tag, export=export, T=h.T, master=master, uniform=umaster, neff=[min(neffs.values()), max(neffs.values())],
           per_obs={o: float(np.mean([per[r, o] for r in cfg['held']])) for o in OBS},
           per_obs_uniform={o: float(np.mean([uper[r, o] for r in cfg['held']])) for o in OBS},
           per_run={str(r): {o: float(per[r, o]) for o in OBS} for r in cfg['held']},
           # per held run: the effective reference size and the report-split size, which set how
           # sharp the test is (phase2/bias_scale.py turns the widths into a size-free bias)
           n_eff_ref={str(r): float(neffs[r]*len(robs[OBS[0]])) for r in cfg['held']},
           n_report={str(r): int(nrep[r]) for r in cfg['held']})
for o in OBS: print(f'   {o:16s} width {res["per_obs"][o]:.3f}   uniform {res["per_obs_uniform"][o]:.2f}', flush=True)
if KPER:
    res['ref_per_run'] = KPER
if SO.REMOVE or SO.T_DIR:
    res.update(SO.describe(), ref=refpath, ref_realpath=os.path.realpath(refpath), ref_particles=REF_PARTICLES or None,
               ref_orig=REF_ORIG or None, data=DATA, data_realpath=os.path.realpath(DATA))
out = os.environ.get('WIDTHS_OUT') or (f'output/widths_v2_{os.path.basename(export).replace("_cond.npz", "")}'
                                       f'{os.environ.get("WIDTHS_SUFFIX", "")}.json')
json.dump(res, open(out, 'w'), indent=1)
print(f'wrote {out}', flush=True)
