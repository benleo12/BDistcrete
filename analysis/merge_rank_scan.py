#!/usr/bin/env python3
"""Join the per-rank files written by an r3_rank.py job array into one scan.

Each array task trains a single K and writes output/rank_scan_<stage>_K<k>.json, so a task
that dies costs one rank rather than the whole scan. This merges whatever is present, reports
which ranks are missing rather than silently interpolating over them, and locates the largest
drop in closure width, which is the cliff the score expansion predicts at K = d+1.
Usage: python merge_rank_scan.py <stage> [<stage> ...]"""
import glob, json, os, sys
import numpy as np

# RANK_DIR points at the per-rank files, so a scan repeated under a different recipe (a
# different seed count, say) can be kept in its own directory and merged separately rather
# than overwriting the first. Outputs land beside the inputs.
RDIR = os.environ.get('RANK_DIR', 'output')

for tag in (sys.argv[1:] or ['E']):
    files = sorted(glob.glob(f'{RDIR}/rank_scan_{tag}_K*.json'))
    if not files:
        print(f'{tag}: no per-rank files'); continue
    d = None; predicted = None; scan = {}; recipes = {}
    for fn in files:
        r = json.load(open(fn))[tag]
        d, predicted = r['d'], r['predicted_cliff']
        # Every rank must have been trained under the SAME recipe or the comparison across
        # ranks is meaningless, and a scan assembled from separate jobs is exactly where a
        # stray point at a different step count gets in. A file written before the recipe was
        # recorded reads as unknown, which is also a mismatch.
        rec = tuple(r.get(kk, 'UNKNOWN') for kk in ('steps', 'ens', 'lr', 'act', 'emb_scale', 'nref'))
        recipes.setdefault(rec, []).append(os.path.basename(fn))
        for k, v in r['scan'].items():
            if k in scan and scan[k] != v:
                print(f'{tag}: WARNING K={k} appears twice with different widths, keeping the later file')
            scan[k] = v
    if len(recipes) > 1:
        print(f'{tag}: REFUSING to merge, the per-rank files were not trained the same way.')
        for rec, fns in sorted(recipes.items(), key=lambda t: -len(t[1])):
            print(f'   steps={rec[0]} ens={rec[1]} lr={rec[2]} act={rec[3]} '
                  f'emb={rec[4]} nref={rec[5]}: {len(fns)} file(s) {sorted(fns)}')
        print('   Delete or rerun the odd ones out, then merge again. A rank trained on a '
              'different budget is not comparable with the rest and would move the cliff.')
        continue
    recipe = next(iter(recipes))
    ks = sorted(int(k) for k in scan)
    want = [int(k) for k in os.environ.get('RANK_KS', '').split(',') if k.strip()]
    missing = [k for k in want if k not in ks]
    if missing: print(f'{tag}: MISSING ranks {missing} (these tasks did not finish)')
    K2 = 1 + d + d*(d+1)//2
    # The claim is SATURATION at K = d+1, not a largest drop. Reporting the largest drop is
    # actively misleading here: the ranks are unevenly spaced and the width is farthest from
    # one at small K, so the biggest absolute fall always lands at the bottom of the scan
    # whatever the curve does afterwards. The test that matches the claim compares K = d+1
    # with the asymptote the scan reaches at and above K2, and with the rank below it.
    W = {k: scan[str(k)]['width'] for k in ks}
    E = {k: scan[str(k)]['spread'] for k in ks}
    # The asymptote is the mean over the ranks at and above K2, where the expansion says
    # nothing is left to gain. If the scan never reached K2, say so instead of implying it
    # did: the fallback is the highest rank present and it is a weaker statement.
    tail = [k for k in ks if k >= K2] or [ks[-1]]
    tail_is_k2 = tail[0] >= K2
    asym = float(np.mean([W[k] for k in tail]))
    asym_err = float(np.mean([E[k] for k in tail])) or 1e-9
    def z(k):
        return (W[k] - asym)/np.hypot(E[k], asym_err)
    # the smallest rank that is statistically indistinguishable from the asymptote
    sat = next((k for k in ks if abs(z(k)) <= 1.0), None)
    at, below = predicted, max([k for k in ks if k < predicted], default=None)
    verdict = {}
    if at in W:
        verdict['z_at_predicted'] = float(z(at))
        verdict['saturated_at_predicted'] = bool(abs(z(at)) <= 1.0)
    if below in W and at in W:
        gain = W[below] - W[at]
        verdict['gain_from_below_to_predicted'] = float(gain)
        verdict['gain_significance'] = float(gain/np.hypot(E[below], E[at]))
    out = dict(d=d, predicted_cliff=predicted, predicted_exhausted=K2,
               asymptote=asym, asymptote_err=asym_err, saturation_rank=sat,
               asymptote_from=list(tail), asymptote_reached_k2=bool(tail_is_k2),
               recipe=dict(zip(('steps', 'ens', 'lr', 'act', 'emb_scale', 'nref'), recipe)),
               ranks=ks, missing=missing, verdict=verdict,
               scan={str(k): scan[str(k)] for k in ks})
    json.dump(out, open(f'{RDIR}/rank_scan_{tag}.json', 'w'), indent=1)
    # Also update the combined file the paper figure reads, keyed by stage, WITHOUT touching
    # the other stages in it. The figure loops over explicit stage keys, so adding one is
    # safe, and rewriting the file wholesale would silently drop the A, B and C scans.
    comb = f'{RDIR}/rank_scan.json'
    allsc = json.load(open(comb)) if os.path.exists(comb) else {}
    prev = allsc.get(tag, {}).get('scan')
    allsc[tag] = out
    json.dump(allsc, open(comb, 'w'), indent=1)
    note = '' if prev is None else f' (replaced a previous {len(prev)} rank entry)'
    print(f'   updated {comb} key {tag}{note}; it now holds {sorted(allsc)}')
    print(f'== {tag}: d={d}, cliff predicted at K={predicted}, exhausted at K2={K2}')
    print(f'   every rank trained identically: steps={recipe[0]} ens={recipe[1]} '
          f'act={recipe[3]} emb={recipe[4]} ref={recipe[5]}')
    for k in ks:
        mark = ' <- predicted cliff' if k == predicted else (' <- K2' if k == K2 else '')
        print(f'   K={k:>3}: width={W[k]:.3f} +- {E[k]:.3f}  '
              f'(asymptote {z(k):+.1f} sigma){mark}')
    src = (f'K >= {K2}' if tail_is_k2 else
           f'K = {tail[0]} only, the scan never reached K2 = {K2}, so this is a FLOOR on the '
           f'asymptote and the saturation rank below may be too high')
    print(f'   asymptote {asym:.3f} +- {asym_err:.3f} from {src}')
    print(f'   saturation rank (first within 1 sigma of it): K={sat}, predicted {predicted}')
    if 'gain_significance' in verdict:
        print(f'   going from K={below} to K={at} gains '
              f'{verdict["gain_from_below_to_predicted"]:+.3f} '
              f'({verdict["gain_significance"]:+.1f} sigma)')
    print(f'   wrote {RDIR}/rank_scan_{tag}.json')
