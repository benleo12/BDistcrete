#!/usr/bin/env python3
"""The ladder closure of tab:ladder recomputed without held-out runs that duplicate training runs.

Stage B held-out run 6704 is event for event the training run 6613, and Stage C held-out run 6902
the training run 6813: the ladder runs set no random seed, so Sherpa's default seed reproduced the
events at the coinciding parameter points. Those two runs test nothing and are dropped here.

Same estimator as r2_ladder.py and ladder_widths_v2.py: bins from the reference plus the report
half of each held run, per-bin pulls with both sides' multinomial errors, width = sqrt(chi2/ndf)
about zero, at the export's temperature. Reported per observable: the closure width (mean over the
held runs), the uniform-weight width (mean over the held runs), and the same-run check (median over
the held runs and three random halvings of the width between two halves of one held run), and the
mean over observables of the closure width, the number tab:ladder quotes.

    python ladder_widths_dedup.py B output/models/B_ref.npz [DROP=6704]
    STAGEC_DATA=data_stageC_v2 python ladder_widths_dedup.py C output/models/C_ref_v2.npz
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'release'); sys.path.insert(0, '.')
from gentune.head import Head
import r2_ladder as R
from make_stage_table import strange_frac

DUPLICATES = {'B': 6704, 'C': 6902, 'MIXSTAGE': 9932}
tag, refpath = sys.argv[1], sys.argv[2]
if tag == 'MIXSTAGE':
    # the seven-parameter mixture stage of Sec. 5.6: held points from its data set's meta.json,
    # export given by EXPORT (the two trainings at their published temperature of 1.2)
    from gentune.head import MixtureHead
    DATA = os.environ.get('DM_DATA', 'data_stageDM2')
    meta = json.load(open(f'{DATA}/meta.json'))
    cfg = dict(held={int(m['rid']): tuple(m['theta']) for m in meta['held']})
    OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy', 'nbaryon', 'strange']
    export = os.environ['EXPORT']; h = MixtureHead(export)
else:
    cfg = R.STAGES[tag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
    export = os.environ.get('EXPORT', f'output/models/{tag}_cond.npz'); h = Head(export)
drop = {int(x) for x in os.environ.get('DROP', str(DUPLICATES.get(tag, ''))).split(',') if x}
held = {rid: th for rid, th in cfg['held'].items() if rid not in drop}
A = h.pack(np.asarray(np.load(export, mmap_mode='r')['AE']))
ref = np.load(refpath, mmap_mode='r'); robs = {o: np.asarray(ref[o], np.float64) for o in OBS}

def target(rid, o):
    if o in ('nbaryon', 'strange'):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        return d['nbaryon'].astype(float) if o == 'nbaryon' else strange_frac(d['particles'], d['mask']).astype(float)
    return pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')[o].values.astype(float)

uni = np.full(len(robs[OBS[0]]), 1.0/len(robs[OBS[0]]))
per, uper, calib = {}, {}, {o: [] for o in OBS}
for rid, th in held.items():
    T_ = {o: target(rid, o) for o in OBS}; n = len(T_[OBS[0]])
    perm = np.random.default_rng(1000 + rid).permutation(n); rep = perm[n//2:]
    w = h.weights(A, np.array(th))
    for o in OBS:
        ot = T_[o][rep]
        b = R.make_bins(np.r_[robs[o], ot], o)
        p, _ = R.pulls(robs[o], w, ot, b); per[rid, o] = R.width_of(p)
        pu, _ = R.pulls(robs[o], uni, ot, b); uper[rid, o] = R.width_of(pu)
        for r in range(3):
            pr = np.random.default_rng(1_000_000 + 1000*r + rid).permutation(len(ot)); hh = len(ot)//2
            o1, o2 = ot[pr[:hh]], ot[pr[hh:]]
            pc, _ = R.pulls(o1, np.full(len(o1), 1.0/len(o1)), o2, b); calib[o].append(R.width_of(pc))
per_obs = {o: float(np.mean([per[r, o] for r in held])) for o in OBS}
uni_obs = {o: float(np.mean([uper[r, o] for r in held])) for o in OBS}
cal_obs = {o: float(np.nanmedian(calib[o])) for o in OBS}
master = float(np.mean(list(per_obs.values())))
print(f'{tag} on {DATA} with {os.path.basename(refpath)}, T={h.T}, dropped {sorted(drop)}, {len(held)} held runs: '
      f'closure width {master:.3f}')
for o in OBS:
    print(f'   {o:16s} width {per_obs[o]:.3f}   same-run {cal_obs[o]:.3f}   uniform {uni_obs[o]:.2f}')
print(f'   same-run range {min(cal_obs.values()):.2f} to {max(cal_obs.values()):.2f}; uniform range {min(uni_obs.values()):.1f} to {max(uni_obs.values()):.1f}')
json.dump(dict(tag=tag, export=export, data=DATA, ref=refpath, T=h.T, dropped=sorted(drop), held=[int(r) for r in held], master=master,
               per_obs=per_obs, uniform=uni_obs, same_run=cal_obs, per_point={str(r): {o: per[r, o] for o in OBS} for r in held}),
          open(f'output/ladder_dedup_{tag}_{os.path.basename(refpath).replace(".npz", "")}{"_all" if not drop else ""}.json', 'w'), indent=1)
