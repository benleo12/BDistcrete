#!/usr/bin/env python3
"""Merge the per-job jsons and collect final deliverables into results/."""
import json, os, glob, shutil

BASE = os.environ['SCRATCH'] + '/energyflower'
J = os.path.join(BASE, 'jobs')
R = os.path.join(BASE, 'results')
os.makedirs(os.path.join(R, 'models'), exist_ok=True)

# 1. rank_scan.json: merge the 21 single-K scans (stages A, B, C at RANK_ENS=4)
merged = {}
for stage in 'ABC':
    for grp in ('K1', 'K2', 'K3', 'K4', 'K6', 'K8', 'K16'):
        p = os.path.join(J, f'rank_{stage}_{grp}', 'output', 'rank_scan.json')
        if not os.path.exists(p):
            print(f'MISSING {p}'); continue
        part = json.load(open(p))
        for st, r in part.items():
            if st not in merged:
                merged[st] = dict(d=r['d'], predicted_cliff=r['predicted_cliff'], scan={})
            assert merged[st]['d'] == r['d']
            merged[st]['scan'].update(r['scan'])
for stage, r in sorted(merged.items()):
    w = {int(k): v['width'] for k, v in r['scan'].items()}
    ks = sorted(w)
    drops = [(k, w[ks[i-1]] - w[k]) for i, k in enumerate(ks) if i > 0]
    cliff = max(drops, key=lambda t: t[1])[0] if drops else None
    r['largest_drop_at_K'] = cliff
    r['rank_ens'] = 4
    print(f'{stage}: d={r["d"]} predicted cliff K={r["predicted_cliff"]}, largest drop at K={cliff}, Ks={ks}')
json.dump(merged, open(os.path.join(R, 'rank_scan.json'), 'w'), indent=1)

# 2. rank_production_v2.json: merge the three physical-rank ladder runs by stage key
prod = {}
for job, fn in (('k3_A', 'rank_v2_A.json'), ('k6_B', 'rank_v2_B.json'), ('k10_C', 'rank_v2_C.json')):
    p = os.path.join(J, job, 'output', fn)
    if not os.path.exists(p):
        print(f'MISSING {p}'); continue
    prod.update(json.load(open(p)))
json.dump(prod, open(os.path.join(R, 'rank_production_v2.json'), 'w'), indent=1)
print('rank_production_v2 stages:', sorted(prod), {s: round(prod[s]['master'], 3) for s in prod})

# 2b. concat_baseline.json: merge the per-stage runs
cb = {}
for job in ('concat_A', 'concat_B'):
    p = os.path.join(J, job, 'output', 'concat_baseline.json')
    if not os.path.exists(p):
        print(f'MISSING {p}'); continue
    cb.update(json.load(open(p)))
json.dump(cb, open(os.path.join(R, 'concat_baseline.json'), 'w'), indent=1)
print('concat_baseline stages:', sorted(cb), {s: round(cb[s]['master'], 3) for s in cb})

# 3. copy the satellite/production jsons
for job, names in (('widebox', ['widebox_final.json']),
                   ('widebox_ctrl', ['widebox_ctrl.json']),
                   ('showers', ['stageD3.json']),
                   ('disjoint', ['disjoint_score.json', 'disjoint_scores.npz']),
                   ('b_r200', ['ladder_B_r200.json']),
                   ('mass_B', ['ladder_B_mass.json']),
                   ('mass_C', ['ladder_C_mass.json'])):
    for n in names:
        p = os.path.join(J, job, 'output', n)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(R, n)); print('collected', n)
        else:
            print(f'MISSING {p}')

# 4. cond npz exports from the production-recipe ladder jobs
for job in ('k3_A', 'k6_B', 'k10_C', 'b_r200', 'mass_B', 'mass_C'):
    for p in glob.glob(os.path.join(J, job, 'output', 'models', '*_cond.npz')):
        shutil.copy2(p, os.path.join(R, 'models', os.path.basename(p)))
        print('collected model', os.path.basename(p))
print('MERGE/COLLECT DONE')
