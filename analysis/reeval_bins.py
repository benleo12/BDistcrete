#!/usr/bin/env python3
"""Re-evaluate the exported ladder models under the corrected discrete binning, and attach an
uncertainty to every width.

Two things were wrong with the first reading of the baryon rung. The binning assumed unit
spacing for an observable whose support is spaced by two (baryons come in pairs), so half the
bins were empty by construction and were dropped. And the width was quoted with no error even
though it rests on very few degrees of freedom: for chi2/ndf the sampling spread of
sqrt(chi2/ndf) about the null is ~1/sqrt(2*ndf), which at ndf=4 is 18%.

No retraining happens here: the conditional is rebuilt from the exported A-vectors and B-net
weights (verified against the stored checksum), so this isolates the change in the RULER.
Writes output/reeval_bins.json."""
import json
import numpy as np, pandas as pd
from r2_ladder import STAGES, pulls, width_of, make_bins, strange_frac


def silu(x): return x/(1+np.exp(-x))


def load_model(tag):
    d = np.load(f'output/models/{tag}_cond.npz')
    assert str(d['combine']) == 'mean_logits'
    return d


def f_at(d, theta):
    norm = d['norm']; nth = int(d['ntheta']); ens = int(d['ens']); AE = d['AE']
    tn = ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32).reshape(1, nth)
    act = (lambda z: np.maximum(z, 0.0)) if str(d['act']) == 'relu' else silu
    nl = sum(1 for k in d.files if k.startswith('B0_W'))
    fs = []
    for mi in range(ens):
        x = tn
        for li in range(nl):
            x = x @ d[f'B{mi}_W{li}'].T + d[f'B{mi}_b{li}']
            if li < nl-1: x = act(x)
        fs.append(AE[mi] @ x.ravel())
    return np.mean(fs, 0)


def width_err(nbins):
    ndf = max(nbins-1, 1)
    return 1.0/np.sqrt(2.0*ndf)


def mean_z(vref, w, vtgt):
    """Closure of the MEAN, as a z-score.

    A binned shape test has almost no power on a rate-like observable whose support holds only
    a handful of values: the baryon count has 4 populated values, so ndf=3 and any width it
    returns carries a ~40% uncertainty. The mean is the quantity the corresponding parameter
    actually controls and it is measured to high precision, so it is the sharp test. Errors
    use the weighted (effective) statistics on the reference side."""
    mr = float(np.sum(w*vref)); mt = float(np.mean(vtgt))
    var_r = float(np.sum(w*(vref-mr)**2))
    neff = 1.0/float(np.sum(w**2))
    se = np.sqrt(var_r/neff + np.var(vtgt)/len(vtgt))
    return (mr-mt)/se if se > 0 else float('nan'), mr, mt, se


def run(tag):
    cfg = STAGES[tag]; OBS = cfg['obs'] + cfg['flav_obs']; DATA = cfg['data']
    d = load_model(tag)
    # checksum round-trip
    chk = d['f_checksum']; f0 = f_at(d, np.atleast_1d(d['f_checksum_theta']).astype(float))
    err = float(np.max(np.abs(f0[:len(chk)]-chk)))
    assert err < 1e-4, f'{tag}: export checksum mismatch {err}'
    ref = np.load(f'output/models/{tag}_ref.npz')
    robs = {o: ref[o] for o in OBS}
    out = {'checksum_err': err, 'per_obs': {}, 'per_obs_err': {}, 'nbins': {},
           'calib': {}, 'per_obs_old_bins': {}, 'nbins_old': {},
           'mean_z': {}, 'mean_z_uniform': {}, 'mean_rel_err': {}}
    acc = {o: [] for o in OBS}; accn = {o: [] for o in OBS}
    acco = {o: [] for o in OBS}; accno = {o: [] for o in OBS}
    calib = {o: [] for o in OBS}
    mz = {o: [] for o in OBS}; mzu = {o: [] for o in OBS}; mre = {o: [] for o in OBS}
    for rid, theta in cfg['held'].items():
        dd = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        SH = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        NB = dd['nbaryon'].astype(np.float32) if 'nbaryon' in cfg['flav_obs'] else None
        STR = strange_frac(dd['particles'], dd['mask']).astype(np.float32) if 'strange' in cfg['flav_obs'] else None
        p = np.random.default_rng(1000+rid).permutation(len(SH)); rep = p[len(p)//2:]
        T = float(d['temperature']) if 'temperature' in d.files else 1.0
        f = f_at(d, theta); w = np.exp((f - f.max())/T); w /= w.sum()
        for o in OBS:
            v = (SH[o].values if o in cfg['obs'] else (NB if o == 'nbaryon' else STR))
            ot = v[rep]
            b = make_bins(np.r_[robs[o], ot], o)
            pl, nb = pulls(robs[o], w, ot, b)
            acc[o].append(width_of(pl, nb)); accn[o].append(nb)
            z, mr, mt, se = mean_z(robs[o], w, ot)
            zu, _, _, _ = mean_z(robs[o], np.full(len(robs[o]), 1.0/len(robs[o])), ot)
            mz[o].append(z); mzu[o].append(zu)
            mre[o].append(abs(mr-mt)/abs(mt) if mt != 0 else np.nan)
            # the OLD unit-spaced discrete grid, for comparison
            if o in ('nbaryon',):
                lo, hi = np.percentile(np.r_[robs[o], ot], [0.5, 99.5])
                bo = np.arange(np.floor(lo)-0.5, np.ceil(hi)+1.5, 1.0)
                plo, nbo = pulls(robs[o], w, ot, bo)
                acco[o].append(width_of(plo, nbo)); accno[o].append(nbo)
            # calibration: two disjoint halves of the same held run, same bins
            for r in range(3):
                pr = np.random.default_rng(1_000_000+1000*r+rid).permutation(len(ot)); h = len(ot)//2
                o1, o2 = ot[pr[:h]], ot[pr[h:]]
                pc, nc = pulls(o1, np.full(len(o1), 1.0/len(o1)), o2, b)
                calib[o].append(width_of(pc, nc))
    for o in OBS:
        out['per_obs'][o] = float(np.mean(acc[o]))
        out['nbins'][o] = int(np.mean(accn[o]))
        out['per_obs_err'][o] = float(width_err(np.mean(accn[o]))/np.sqrt(len(acc[o])))
        out['calib'][o] = float(np.nanmedian(calib[o]))
        out['mean_z'][o] = float(np.mean(np.abs(mz[o])))
        out['mean_z_uniform'][o] = float(np.mean(np.abs(mzu[o])))
        out['mean_rel_err'][o] = float(np.nanmean(mre[o]))
        if acco[o]:
            out['per_obs_old_bins'][o] = float(np.mean(acco[o]))
            out['nbins_old'][o] = int(np.mean(accno[o]))
    out['master'] = float(np.mean(list(out['per_obs'].values())))
    return out


def main():
    res = {}
    for tag in ['A', 'B', 'C']:
        try:
            res[tag] = run(tag)
        except FileNotFoundError:
            print(f'{tag}: no export, skipped'); continue
        r = res[tag]
        print(f'--- Stage {tag}: master {r["master"]:.3f} (checksum {r["checksum_err"]:.1e}) ---')
        print(f'  {"obs":16} {"width":>16} {"nbins":>6} {"calib":>7}')
        for o in r['per_obs']:
            e = r['per_obs_err'][o]
            print(f'  {o:16} {r["per_obs"][o]:7.2f} +- {e:5.2f} {r["nbins"][o]:6d} {r["calib"][o]:7.2f}'
                  f'   mean|z|={r["mean_z"][o]:7.2f} (uniform {r["mean_z_uniform"][o]:8.1f})'
                  f'  rel.err={r["mean_rel_err"][o]*100:.2f}%')
        for o, v in r.get('per_obs_old_bins', {}).items():
            print(f'  [{o}] unit-spaced bins would give {v:.2f} on {r["nbins_old"][o]} bins '
                  f'vs {r["per_obs"][o]:.2f} on {r["nbins"][o]} bins')
    json.dump(res, open('output/reeval_bins.json', 'w'), indent=1)
    print('REEVAL DONE')


if __name__ == '__main__':
    main()
