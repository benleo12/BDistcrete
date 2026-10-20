#!/usr/bin/env python3
"""The numbers behind the Stage C closure figure (figs_pro.py hero), without plotting, so they can
be computed where the model and the data are: the reference of one trained Stage C model reweighted
to a held-out point, the unweighted reference and the full held-out run, binned as in the figure,
and the closure width of each panel on the full run (from the validation map of the same model).

    python fig_closure_data.py <model tag> <data dir> <validation map json>
e.g. VALID_HALF=report python fig_closure_data.py C_P2augfullfac0 ../gen/data_abc output/valid_map_C_P2augfullfac0_report.json
(VALID_HALF=report draws the report half of the run, the paper's closure test, as the map does)
Writes output/fig_closure_<model tag>_<held run>.npz, which figs_pro.py hero draws (HERO_FROM)."""
import json, os, sys
import numpy as np
import pandas as pd
from ab_analysis import Cond
from r2_ladder import make_bins
from xp_closure import per_event_bincontents, XP_BINS

HELD = 6901; THETA = (0.124, 0.55, 1.5)      # off-centre in two of three parameters
PANELS = ['1_minus_thrust', 'mult_total', 'nbaryon', 'x_p']


def _hist_w(vals, w, bins):
    h, _ = np.histogram(vals, bins, weights=w)
    h2, _ = np.histogram(vals, bins, weights=w**2)
    W = h.sum(); bw = np.diff(bins)
    return h/(W*bw), np.sqrt(h2)/(W*bw)


def _hist_u(vals, bins):
    n, _ = np.histogram(vals, bins)
    N = n.sum(); bw = np.diff(bins)
    return n/(N*bw), np.sqrt(n)/(N*bw)


def hero_panels(model, data, valid_map):
    """{obs: dict(bins, dw, ew, du, dt, et, width)} for the four panels of the figure."""
    cond = Cond(model); ref = np.load(f'output/models/{model}_ref.npz')
    c = np.load(f'output/models/{model}_cond.npz', mmap_mode='r')
    if 'f_checksum' in c.files:                  # the loader must reproduce the trained model
        fc = cond.f_at(np.asarray(c['f_checksum_theta'], float))[:len(c['f_checksum'])]
        err = float(np.max(np.abs(fc - c['f_checksum'])))
        assert err < 1e-3, f'the export does not reproduce its checksum (max err {err:.2e})'   # the export's own test
    d = np.load(f'{data}/particles_full_{HELD}.npz')
    sh = pd.read_csv(f'{data}/shapes_run_{HELD:04d}.csv')
    # VALID_HALF=report: the report half of the run, the paper's closure test (as in eval_valid.py)
    sel = (np.random.default_rng(1000 + HELD).permutation(len(sh))[len(sh)//2:]
           if os.environ.get('VALID_HALF', '') == 'report' else np.arange(len(sh)))
    d = {k: d[k][sel] for k in ('particles', 'mask', 'nbaryon')}; sh = sh.iloc[sel]
    f = cond.f_at(np.array(THETA)); w = np.exp((f - f.max())/cond.T); w /= w.sum()
    uni = np.full(cond.Nref, 1.0/cond.Nref)
    widths = json.load(open(valid_map))[str(HELD)]
    out = {}
    for obs in PANELS:
        if obs == 'x_p':
            # per-particle spectrum with event-level effective counts for the pulls
            rC = per_event_bincontents(ref['particles'][:, :, 0], ref['mask'], XP_BINS)
            tC = per_event_bincontents(d['particles'][:, :, 0], d['mask'], XP_BINS)
            bins = XP_BINS; bw = np.diff(bins)
            def dens_eff(C, wts):
                v = C*wts[:, None]; cont = v.sum(0)
                s2 = ((C**2)*(wts[:, None]**2)).sum(0)
                W = cont.sum()
                neff = np.where(s2 > 0, cont**2/np.where(s2 > 0, s2, 1), 0)
                dens = cont/(W*bw)
                return dens, np.where(neff > 0, dens/np.sqrt(np.maximum(neff, 1)), 0)
            dw, ew = dens_eff(rC, w); du, _ = dens_eff(rC, uni)
            dt, et = dens_eff(tC, np.full(len(tC), 1.0/len(tC)))
        else:
            if obs == 'nbaryon':
                rv = ref['nbaryon'].astype(float); tv = d['nbaryon'].astype(float)
            else:
                rv = ref[obs].astype(float); tv = sh[obs].values
            bins = make_bins(np.r_[rv, tv], obs)
            dw, ew = _hist_w(rv, w, bins); du, _ = _hist_w(rv, uni, bins); dt, et = _hist_u(tv, bins)
        out[obs] = dict(bins=np.asarray(bins, float), dw=dw, ew=ew, du=du, dt=dt, et=et, width=float(widths[obs]))
    return out


if __name__ == '__main__':
    model, data, vmap = sys.argv[1:4]
    P = hero_panels(model, data, vmap)
    path = f'output/fig_closure_{model}_{HELD}' + ('_report' if os.environ.get('VALID_HALF', '') == 'report' else '') + '.npz'
    np.savez(path, model=model, held=HELD, theta=np.array(THETA), panels=np.array(PANELS),
             **{f'{o}__{k}': v for o, p in P.items() for k, v in p.items()})
    print(f'wrote {path}: widths ' + ', '.join(f'{o} {p["width"]:.3f}' for o, p in P.items()))
