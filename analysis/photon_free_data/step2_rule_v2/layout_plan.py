#!/usr/bin/env python3
"""Step 4 of B0-observables (D12): the files under maxent_code/output that the rerun reads, and how
each enters RUN/output (RUN = /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code):
  refreshed   step 3 wrote a copy with the photon-free observables (refresh_bundles.py)
  copied      a byte copy (files under 1 GB, and the exports that D11 rewrites in place, see below)
  symlinked   an absolute link to the original: read-only inputs of 1 GB or more that no rerun step
              writes, or a relative link inside RUN where Symmetry has the same relative link
Exports (<tag>_cond.npz) of the models whose readers call score_obs.check_export_T (directlib and its
users, ladder_widths_big, fig_closure_mix, wifi_*, eval_valid, single_members, the release) must carry
the refitted temperature, which apply_T.py --lmx writes into the export (D11): those are copied
whatever their size, so that no rerun step ever writes through a link.

    python layout_plan.py OUT_TSV        (reads directory listings and sizes only)
"""
import os, sys, glob

R = '/mnt/beegfs/bassi/reweighting'
SRC = f'{R}/maxent_code/output'
GB = 1e9
rows = []


def add(rel, kind, why, src=None):
    rows.append(dict(path=rel, kind=kind, source=src or f'{SRC}/{rel}', why=why))


def model(tag, why, files=('_trunk.pt', '_cond.npz'), d11=False):
    for suf in files:
        rel = f'models/{tag}{suf}'
        p = f'{SRC}/{rel}'
        if not os.path.lexists(p):
            add(rel, 'MISSING', why); continue
        if os.path.islink(p):
            t = os.readlink(p)
            add(rel, 'symlinked', why + ' (relative link, as on Symmetry)', src=t); continue
        big = os.path.getsize(p) >= GB
        if suf == '_cond.npz' and d11:
            add(rel, 'copied', why + (' (>=1 GB, copied: D11 rewrites the export in place)' if big else ''))
        else:
            add(rel, 'symlinked' if big else 'copied', why)


def record(name, why):
    p = f'{SRC}/{name}'
    add(name, 'copied' if os.path.exists(p) else 'MISSING', why)


# ---- the factorized mixture (Table 1, Figs. 3, 5, 8-12, Cov, Wifi, Tb, Env, Cost, release)
for t in ('MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2'):
    w = f'factorized mixture {t}: three_bands, envelope_test, theory_variations, band_checks, closure_mix, wifi_cov, ladder_widths_big, recalibrate_T'
    model(t, w, files=('_trunk.pt', '_cond.npz', '_fitsplit.npz'), d11=True)
    for suf in ('_ref_v2_slim.npz', '_ref.npz'):
        add(f'models/{t}{suf}', 'refreshed', w)
    add(f'models/{t}h400_cond.npz', 'symlinked', 'strict_closure.sbatch (relative link, as on Symmetry; the script makes it again)', src=f'{t}_cond.npz')
    add(f'wifi_{t}_basis.npz', 'refreshed', 'wifi_fit/wifi_diag/wifi_gensplit (held-out observables)')
    add(f'wifi_{t}_fit.npz', 'copied', 'three_bands, envelope_test, wifi_diag, wifi_coverage (wifi_cov.sbatch skips wifi_fit while it exists)')
    record(f'stage_mixture_{t}.json', 'the training record of the standard test (Table 1), make_numbers')
for k, t in enumerate(('MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2')):
    for suf in ('_cond.npz', '_trunk.pt'):
        add(f'models/MIXP2v3fullfac{k}{suf}', 'symlinked', 'strict test of Table 1, score_p2_array MIX:fac (relative link, as on Symmetry)',
            src=f'{t}{suf}')
# ---- the learning curve (Sec. 3.1, LcRuns*, LcEv*, LcSteps*, LcMembers, CovEight)
for t in ('MIXGEOV3E20S9', 'MIXGEOV3E160S9', 'MIXGEOV3E320S9', 'MIXGEOV3S27F', 'MIXGEOV3M8S9', 'MIXGEOV4S9', 'MIXGEOV4L'):
    w = f'learning curve {t}: ladder_widths_big (strict_closure, rescore_v4), recalibrate_T'
    model(t, w, files=('_trunk.pt', '_cond.npz'), d11=True)
    for suf in ('_ref_v2_slim.npz', '_ref.npz'):
        add(f'models/{t}{suf}', 'refreshed', w)
    add(f'models/{t}h400_cond.npz', 'symlinked', 'strict_closure/rescore_v4 (relative link, as on Symmetry; the script makes it again)', src=f'{t}_cond.npz')
add('wifi_MIXGEOV3M8S9_basis.npz', 'refreshed', 'wifi_cov.sbatch TAG=MIXGEOV3M8S9 (CovEight)')
add('wifi_MIXGEOV3M8S9_fit.npz', 'copied', 'wifi_cov.sbatch TAG=MIXGEOV3M8S9 (CovEight)')
# ---- the 208-run row of the learning curve
for k in range(3):
    model(f'MIXP2origfullfac{k}', '208-run row: score_p2 at its 16 and at the 48 held-out points (LcRunsThree...)')
    for n in (f'p2origfull_MIX_fac_s{k}.json', f'p2origfull48_MIX_fac_s{k}.json'):
        record(n, 'training record of the 208-run row (LADDER_TRAINED_RESULT, make_numbers)')
# ---- Stages A-C (Table 1, Calib, SvdRank, K = 48 of Fig. 6), D/E (Table 1), the DCTR mixture
abc = {'A': [('augfull', 'fac', k) for k in range(3)] + [('augfull', 'cat', k) for k in range(3)],
       'B': [('aug', 'fac', k) for k in range(3)] + [('augfull', 'cat', k) for k in range(3)],
       'C': [('augfull', 'fac', 0), ('augfull', 'fac', 1), ('aug', 'fac', 2),
             ('augfull', 'cat', 0), ('augfull', 'cat', 1), ('aug', 'cat', 2)]}
for S, specs in abc.items():
    for dt, net, k in specs:
        tag = f'{S}_P2{dt}{net}{k}'
        w = f'Stage {S} {net} s{k}: score_p2 (standard and strict test of Table 1)'
        model(tag, w, files=('_trunk.pt', '_cond.npz') if net == 'fac' else ('_concat.pt',),
              d11=(tag == 'C_P2augfullfac0'))
        record(f'p2{dt}_{S}_{net}_s{k}.json', f'training record of {tag} (LADDER_TRAINED_RESULT, make_numbers)')
add('models/C_P2augfullfac0_ref.npz', 'refreshed', 'eval_valid (Fig. 4, Map), single_members (Single)')
for S in ('E', 'F'):
    for net in ('fac', 'cat'):
        for k in range(3):
            tag = f'{S}_P2v3full{net}{k}'
            model(tag, f'Stage {"D" if S == "E" else "E"} {net} s{k}: score_p2 (Table 1)' + ('; the release' if (net, k) == ('fac', 0) else ''),
                  files=('_trunk.pt', '_cond.npz') if net == 'fac' else ('_concat.pt',), d11=(net, k) == ('fac', 0))
            record(f'p2v3full_{S}_{net}_s{k}.json', f'training record of {tag} (LADDER_TRAINED_RESULT, make_numbers)')
    add(f'models/{S}_P2v3fullfac0_ref.npz', 'refreshed', 'the release: make_stage_table, precision_gap, closure_vs_distance, package_release')
for k in range(3):
    model(f'MIXP2v3fullcat{k}', f'DCTR network of the mixture s{k}: score_p2 (Table 1)', files=('_concat.pt',))
    record(f'p2v3full_MIX_cat_s{k}.json', f'training record of MIXP2v3fullcat{k} (LADDER_TRAINED_RESULT, make_numbers)')
# ---- the K scan (Fig. 6)
for S, Ks in (('A', (1, 2, 3, 4, 6, 8, 16)), ('B', (1, 2, 3, 4, 6, 8, 16)), ('C', (1, 2, 3, 4, 6, 8, 16))):
    for K in Ks:
        model(f'{S}_P2augK{K}fac0', f'K scan Stage {S} K={K}: score_p2 (Fig. 6)')
        record(f'p2augK{K}_{S}_fac_s0.json', f'training record of {S}_P2augK{K}fac0 (kscan_collect)')
for K in (1, 2, 4, 6, 8, 9, 10, 12, 16, 24, 32, 45):
    model(f'E_P2v3K{K}fac0', f'K scan Stage D K={K}: score_p2 (Fig. 6)')
    record(f'p2v3K{K}_E_fac_s0.json', f'training record of E_P2v3K{K}fac0 (kscan_collect)')
# ---- the wide coupling scan (Wide)
for n in ['single', 'pool'] + [f'local{r}' for r in range(9100, 9106)]:
    model(f'W_P2wb18{n}fac0', f'wide scan {n}: score_p2 stage W (Wide)')
    record(f'p2wb18{n}_W_fac_s0.json', f'training record of W_P2wb18{n}fac0 (widebox_collect)')
for n in ('W_P2wb18singlefac0_w_9100.npy', 'W_P2wb18local9100fac0_w_9100.npy'):
    add(f'models/{n}', 'copied', 'widebox_collect.py')
# ---- the calculation
add('thrust_targets_nlo_point.npz', 'copied', 'three_bands, envelope_test, theory_variations, band_checks, wifi_coverage, bench, calc_vs_delphi, check_linear_mix')
add('calc_point.json', 'copied', 'targets_nlo_point, calc_bins, calc_vs_delphi (the fixed point of Eq. (C.9))')

# sizes, and a check that nothing is listed twice
seen = set()
for r in rows:
    assert r['path'] not in seen, r['path']
    seen.add(r['path'])
    s = r['source']
    r['size'] = os.path.getsize(s) if (r['kind'] in ('copied', 'refreshed') and os.path.exists(s)) else 0
with open(sys.argv[1], 'w') as fh:
    fh.write('path\tkind\tsource\tsize\twhy\n')
    for r in rows:
        fh.write(f"{r['path']}\t{r['kind']}\t{r['source']}\t{r['size']}\t{r['why']}\n")
tot = {}
for r in rows:
    tot.setdefault(r['kind'], [0, 0]); tot[r['kind']][0] += 1; tot[r['kind']][1] += r['size']
for k, (n, s) in sorted(tot.items()):
    print(f'{k:10s} {n:4d} files {s / 1e9:8.2f} GB')
big = [r for r in rows if r['kind'] == 'copied' and r['size'] >= GB]
print('copied >= 1 GB:', [(r['path'], round(r['size'] / 1e9, 2)) for r in big])
print('MISSING:', [r['path'] for r in rows if r['kind'] == 'MISSING'])
