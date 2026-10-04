#!/usr/bin/env python3
"""Dense validation of the Stage C conditional: does the ONE exported network reproduce a
fresh generator run at ANY point of the 3-parameter box, judged many ways?

Points: 12 new runs (6950-6961, interior + near-edge) plus the 5 original held-out runs
(6900-6904) re-evaluated through the identical code path as an anchor.
Ways: the six scalar observables of the ladder (event shapes, multiplicity, baryon number,
strangeness), the x_p fragmentation spectrum, and a JOINT 2D closure in (1-T, mult) which is
sensitive to correlations that every 1D histogram integrates out. Controls: N_eff of the
weights and the uniform-weight width per point.
Writes output/valid_map.json."""
import json, os
import numpy as np
from ab_analysis import Cond
from r2_ladder import make_bins, pulls, width_of, strange_frac
from xp_closure import per_event_bincontents, width_xp, XP_BINS

DATA = 'data_stageC'
POINTS = {
    6950: (0.1130, 0.33, 0.90), 6951: (0.1150, 0.52, 1.65), 6952: (0.1170, 0.62, 1.10),
    6953: (0.1190, 0.36, 1.40), 6954: (0.1210, 0.58, 0.85), 6955: (0.1230, 0.42, 1.70),
    6956: (0.1250, 0.50, 1.05), 6957: (0.1270, 0.32, 1.30), 6958: (0.1135, 0.63, 1.75),
    6959: (0.1275, 0.64, 0.82), 6960: (0.1180, 0.48, 0.95), 6961: (0.1220, 0.40, 1.55),
}
ANCHOR = {6900: (0.116, 0.38, 1.00), 6901: (0.124, 0.55, 1.50), 6902: (0.120, 0.46, 1.21),
          6903: (0.114, 0.60, 0.95), 6904: (0.126, 0.35, 1.60)}
CSV_OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy']
N2D = 7          # per-axis quantile bins for the joint test


def joint_width(x_ref, y_ref, w, x_t, y_t):
    """2D closure in (x,y): quantile bins per axis from the combined sample, flattened pulls
    with the same pooled-multinomial variance as the 1D ruler."""
    bx = np.unique(np.quantile(np.r_[x_ref, x_t], np.linspace(0, 1, N2D+1)))
    by = np.unique(np.quantile(np.r_[y_ref, y_t], np.linspace(0, 1, N2D+1)))
    bx[0], bx[-1] = bx[0]-1e-9, bx[-1]+1e-9
    by[0], by[-1] = by[0]-1e-9, by[-1]+1e-9
    ha, _, _ = np.histogram2d(x_ref, y_ref, bins=[bx, by], weights=w)
    h2, _, _ = np.histogram2d(x_ref, y_ref, bins=[bx, by], weights=w**2)
    hb, _, _ = np.histogram2d(x_t, y_t, bins=[bx, by])
    Wa = ha.sum(); Nb = hb.sum()
    na = np.where(h2 > 0, ha**2/np.where(h2 > 0, h2, 1), 0)      # effective counts
    Na = Wa*Wa/max(h2.sum(), 1e-300)
    qa = ha/Wa; qb = hb/Nb
    qhat = (Na*qa + Nb*qb)/(Na+Nb)
    safe_a = np.where(na > 0, na, 1.0); safe_b = np.where(hb > 0, hb, 1.0)
    var = qhat**2*(1-qhat)*(1/safe_a + 1/safe_b)
    sig = np.sqrt(np.maximum(var, 0))
    keep = (sig > 0) & (na >= 5) & (hb >= 5)
    if keep.sum() < 2: return float('nan'), 0
    p = (qa[keep]-qb[keep])/sig[keep]
    return float(np.sqrt(np.sum(p**2)/max(keep.sum()-1, 1))), int(keep.sum())


def main():
    cond = Cond('C'); ref = np.load('output/models/C_ref.npz')
    robs = {o: ref[o].astype(float) for o in CSV_OBS + ['nbaryon', 'strange']}
    ref_counts = per_event_bincontents(ref['particles'][:, :, 0], ref['mask'], XP_BINS)
    out = {}
    for rid, th in {**ANCHOR, **POINTS}.items():
        fnpz = f'{DATA}/particles_full_{rid:04d}.npz'
        fcsv = f'{DATA}/shapes_run_{rid:04d}.csv'
        if not (os.path.exists(fnpz) and os.path.exists(fcsv)):
            continue
        import pandas as pd
        d = np.load(fnpz); sh = pd.read_csv(fcsv)
        tobs = {o: sh[o].values for o in CSV_OBS}
        tobs['nbaryon'] = d['nbaryon'].astype(float)
        tobs['strange'] = strange_frac(d['particles'], d['mask']).astype(float)
        f = cond.f_at(np.array(th))
        w = np.exp((f-f.max())/cond.T); w /= w.sum()
        neff = 1.0/np.sum(w**2)
        uni = np.full(cond.Nref, 1.0/cond.Nref)
        row = {'theta': list(th), 'N_eff': float(neff), 'kind': 'anchor' if rid in ANCHOR else 'valid'}
        for o in tobs:
            b = make_bins(np.r_[robs[o], tobs[o]], o)
            p, nb = pulls(robs[o], w, tobs[o], b)
            pu, _ = pulls(robs[o], uni, tobs[o], b)
            row[o] = round(width_of(p, nb), 3); row[f'{o}_uniform'] = round(width_of(pu, nb), 3)
            # the sharp rate test: mismatch of the MEAN in units of its combined stat error.
            # For few-bin yield observables the shape width has little power and this is the
            # instrument that actually discriminates (cf. the baryon accounting in the paper).
            mu_r = (w*robs[o]).sum(); var_r = (w**2*(robs[o]-mu_r)**2).sum()
            mu_t = tobs[o].mean();    var_t = tobs[o].var()/len(tobs[o])
            row[f'{o}_z'] = round(float((mu_r-mu_t)/np.sqrt(var_r+var_t)), 2)
        # x_p spectrum
        tgt_counts = per_event_bincontents(d['particles'][:, :, 0], d['mask'], XP_BINS)
        row['x_p'], _ = width_xp(ref_counts, w, tgt_counts)
        row['x_p'] = round(row['x_p'], 3)
        # joint 2D (1-T, mult)
        jw, nb2 = joint_width(robs['1_minus_thrust'], robs['mult_total'], w,
                              tobs['1_minus_thrust'], tobs['mult_total'])
        row['joint_thrust_mult'] = round(jw, 3); row['joint_nbins'] = nb2
        out[rid] = row
        obs7 = [row[o] for o in list(tobs)+['x_p']]
        print(f'{rid} {th}: mean={np.nanmean(obs7):.3f} obs={ {o: row[o] for o in list(tobs)+["x_p"]} } '
              f'joint={row["joint_thrust_mult"]:.3f} N_eff={neff:.0f}')
    # summary
    val = [r for r in out.values() if r['kind'] == 'valid']
    if val:
        allobs = CSV_OBS + ['nbaryon', 'strange', 'x_p']
        summ = {o: round(float(np.nanmean([r[o] for r in val])), 3) for o in allobs}
        summ['joint_thrust_mult'] = round(float(np.nanmean([r['joint_thrust_mult'] for r in val])), 3)
        summ['n_points'] = len(val)
        summ['worst_single'] = round(float(np.nanmax([[r[o] for o in allobs] for r in val])), 3)
        for o in ['nbaryon', 'strange', 'mult_total']:
            zz = [abs(r[f'{o}_z']) for r in val if f'{o}_z' in r]
            summ[f'{o}_absz_mean'] = round(float(np.mean(zz)), 2)
            summ[f'{o}_absz_max'] = round(float(np.max(zz)), 2)
        print('SUMMARY over validation points:', summ)
        out['summary'] = summ
    json.dump(out, open('output/valid_map.json', 'w'), indent=1)
    print('VALID MAP DONE')


if __name__ == '__main__':
    main()
