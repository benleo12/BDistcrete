#!/usr/bin/env python3
"""Merge the per-(K,seed) kscan_toy_ruler outputs into output/kscan_toy_ruler.json."""
import json, glob, os
import numpy as np

KS = [1, 2, 4, 8, 16, 32]
D = './output/_kruler'
out = {}
# the published column comes from the older kscan_toy.json. It reads nan when that file is
# missing or empty (it is 0 bytes in the release). The merge itself does not need it.
try:
    old = json.load(open('./output/kscan_toy.json'))
except (FileNotFoundError, json.JSONDecodeError):
    old = {}
rows = []
for K in KS:
    fs = sorted(glob.glob(f'{D}/K{K}_s*.json'))
    if not fs:
        print(f'K={K}: MISSING'); continue
    recs = [json.load(open(f))[str(K)] for f in fs]
    w = [r['width'] for r in recs]
    wms = [r['width_mean_square'] for r in recs]
    wpp = [r['width_per_particle'] for r in recs]
    wppp = [r['width_per_particle_particle_units'] for r in recs]
    ow = [np.sqrt(r['old_chi2']) for r in recs]
    # r3_rank-style aggregation: pool the two observables into one chi2 per held-out width,
    # ndf = total bins - 1, then average over the six held-out widths (what rank_scan.json does)
    r3 = []
    for r in recs:
        nms, npp = r['nbins_mean_square'], r['nbins_per_particle']
        per = []
        for a, b in zip(r['per_width_mean_square'], r['per_width_per_particle']):
            chi = a**2*(nms-1) + b**2*(npp-1)
            per.append(np.sqrt(chi/(nms+npp-1)))
        r3.append(float(np.mean(per)))
    sv = next((r['sv'] for r in recs if r['sv']), None)
    rec = dict(chi2=float(np.mean(w))**2, chi2_spread=float(np.std(w)*2*np.mean(w)),
               width=float(np.mean(w)), width_spread=float(np.std(w)),
               sv=sv,
               neff_min=float(min(r['neff_min'] for r in recs)), neff_ref=recs[0]['neff_ref'],
               chi2_uniform=float(np.mean([r['width_uniform'] for r in recs]))**2,
               width_uniform=float(np.mean([r['width_uniform'] for r in recs])),
               width_r3agg=float(np.mean(r3)), width_r3agg_spread=float(np.std(r3)),
               width_mean_square=float(np.mean(wms)), width_per_particle=float(np.mean(wpp)),
               width_per_particle_particle_units=float(np.mean(wppp)),
               nbins_mean_square=float(np.mean([r['nbins_mean_square'] for r in recs])),
               nbins_per_particle=float(np.mean([r['nbins_per_particle'] for r in recs])),
               old_width=float(np.mean(ow)), old_chi2=float(np.mean([r['old_chi2'] for r in recs])),
               seeds=[int(s) for r in recs for s in r['seeds']],
               n_seeds=len(recs))
    out[str(K)] = rec
    pub = old[str(K)]['chi2']**0.5 if str(K) in old else float('nan')
    rows.append((K, pub, rec['old_width'], rec['width'], rec['width_spread'], rec['width_r3agg'],
                 rec['width_mean_square'], rec['width_per_particle'], rec['width_per_particle_particle_units'],
                 rec['neff_min'], rec['nbins_mean_square'], rec['nbins_per_particle']))

hdr = ('  K | published | old-est |  PAPER  |  +-   | PAPER  | mean-sq | per-part | per-part | neff_min | nb_ms | nb_pp')
hdr2 = ('    |  width    | reprod  |  ruler  |       | (r3agg)|         | (event)  | (particle)|         |       |')
print(hdr); print(hdr2); print('-'*118)
for r in rows:
    print(f'{r[0]:3d} | {r[1]:9.3f} | {r[2]:7.3f} | {r[3]:7.3f} | {r[4]:5.3f} | {r[5]:6.3f} | '
          f'{r[6]:7.3f} | {r[7]:8.3f} | {r[8]:9.3f} | {r[9]:8.0f} | {r[10]:5.1f} | {r[11]:5.1f}')
json.dump(out, open('./output/kscan_toy_ruler.json', 'w'), indent=1)
print('\n-> output/kscan_toy_ruler.json')
