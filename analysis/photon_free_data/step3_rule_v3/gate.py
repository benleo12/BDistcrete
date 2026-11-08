#!/usr/bin/env python3
"""The checks of nb3_finish.sbatch must all pass before the caches are refreshed: Phase A's link check
(WORK/verify.json), the merged and truncated sets (WORK/consistency.json), the spot check
(WORK/results/spot3_2026.json), the independent recomputation (WORK/results/indep_check3_11.json), the
folder check (WORK/results/nb_folders.json, written again by the finish job), the unit records (every unit, none failed in input state,
no negative value).
    python gate.py WORK          exit 1 on any failure"""
import os, sys, json, glob

W = sys.argv[1]
bad = []
v = json.load(open(f'{W}/verify.json'))
if v['bad']:
    bad.append(f"nb_links verify: {len(v['bad'])} problems")
for k, r in json.load(open(f'{W}/consistency.json')).items():
    if r['new_identity'] != r['runs'] or r['original_identity'] != r['runs']:
        bad.append(f'consistency {k}: {r}')
sp = json.load(open(f'{W}/results/spot3_2026.json'))
if not all(r['ok'] for r in sp):
    bad.append(f"spot check: {sum(not r['ok'] for r in sp)} bad units")
t = json.load(open(f'{W}/results/indep_check3_11.json'))['total']
for k in ('rows_changed_without_beam', 'beam_events_unchanged', 'mult_mismatch'):
    if t[k]:
        bad.append(f'indep_check3 {k} {t[k]}')
if t['v3_differs_v2'] != t['v3_differs_v2_record']:
    bad.append(f"indep_check3: independent v3/v2 masks differ in {t['v3_differs_v2']} events, the records say {t['v3_differs_v2_record']}")
if t['fallback_decision_mismatch'] > max(2, 0.01 * t['fallback_in_csv']):   # B0's two codes: 2 of 228 differed
    bad.append(f"indep_check3: {t['fallback_decision_mismatch']} fallback decisions differ")
if max(t['max_abs_diff'].values()) > 1e-4:
    bad.append(f"indep_check3: shapes differ by up to {max(t['max_abs_diff'].values())}")
if not os.path.exists(f'{W}/results/nb_folders.json'):           # the finish job removes the test job's first
    bad.append('folder check: no results/nb_folders.json (nb3_folders.py did not finish)')
elif json.load(open(f'{W}/results/nb_folders.json'))['n_problems']:
    bad.append(f"folder check: {json.load(open(f'{W}/results/nb_folders.json'))['n_problems']} problems")
want = sum(1 for _ in open(f'{W}/units.tsv')) - 1
recs = [json.load(open(f)) for f in glob.glob(f'{W}/units/*.json')]
if len(recs) != want:
    bad.append(f'unit records {len(recs)} of {want}')
neg = sum(r['obs'][o]['n_neg_new'] for r in recs for o in r['obs'])
if neg:
    bad.append(f'{neg} negative values in the new CSVs')
print('GATE:', 'ok' if not bad else bad)
sys.exit(1 if bad else 0)
