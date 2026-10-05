#!/usr/bin/env python3
"""Table 4 with training-seed spreads, and the mixture-head control, read with the rules fixed in
TEST_PLAN_2026-10-04.md before the runs returned.

Per cell (design x stage): mean and standard deviation of the closure width over the trainings that
differ only in the seed, the same for the baryon count, and the number of trainings. The factorized
network is retrained five times under identical inputs (perlmutter/seedstudy.sbatch, design fact);
the concatenation and late-fusion networks count their published run as seed 0. A difference
between two designs is called significant when |m1 - m2| > 2 sqrt(s1^2/n1 + s2^2/n2).
The control: the concatenation and late-fusion networks with the exact mixture head (mixe, mixl) are
compared with the factorized network and with the plain concatenation network at seventeen parameters.

    python seedstudy_summary.py        # prints the table, writes output/seedstudy_summary.json
"""
import json, glob, os, re
import numpy as np

STAGES = ['A', 'B', 'C', 'E', 'MIX']


def load(f):
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return None


def first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


PUBLISHED_CONCAT = {
    ('concat', 'A'): 'output/concat_baseline_A_prod.json', ('concat', 'B'): 'output/concat_baseline_B_prod.json',
    ('concat', 'C'): first('output/concat_baseline_C_prod.json', 'from_perlmutter/concat_baseline_C_prod.json'),
    ('concat', 'E'): 'output/concat_baseline_E_silu.json', ('concat', 'MIX'): 'output/concat_baseline_MIX_silu.json',
    ('late', 'C'): 'output/concat_baseline_C_late.json', ('late', 'MIX'): 'output/concat_baseline_MIX_late.json'}
PUBLISHED_FACT = {'A': 'output/ladder_dedup_A_A_ref_all.json', 'B': 'output/ladder_dedup_B_B_ref.json',
                  'C': 'output/ladder_dedup_C_C_ref.json', 'E': 'output/widths_v2_E.json',
                  'MIX': 'output/widths_v2_MIX17aug.json'}


def fact_file(stage, seed):
    return {'A': f'output/ladder_dedup_A_A_seed{seed}_ref_all.json', 'B': f'output/ladder_dedup_B_B_seed{seed}_ref.json',
            'C': f'output/ladder_dedup_C_C_seed{seed}_ref.json', 'E': f'output/widths_v2_E_seed{seed}.json',
            'MIX': f'output/widths_v2_MIX17aug_seed{seed}.json'}[stage]


def runs(design, stage):
    """[(seed, closure width, baryon width or None)] for one cell"""
    out = []
    if design == 'fact':
        for seed in (0, 10, 20, 30, 40):
            d = load(fact_file(stage, seed))
            if d:
                out.append((seed, d['master'], d['per_obs'].get('nbaryon')))
        return out
    pub = PUBLISHED_CONCAT.get((design, stage))
    d = load(pub) if pub else None
    if d and stage in d:
        out.append((0, d[stage]['master'], d[stage]['per_obs'].get('nbaryon')))
    for f in sorted(glob.glob(f'output/seedstudy/{design}_{stage}_s*.json')):
        m = re.search(r'_s(\d+)\.json$', f)
        if not m:
            continue                                   # smoke runs end in "smoke.json"
        seed = int(m.group(1))
        if seed == 0 and design in ('concat', 'late'):
            continue                                   # the published run is seed 0
        d = load(f)
        if d and stage in d:
            out.append((seed, d[stage]['master'], d[stage]['per_obs'].get('nbaryon')))
    return out


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=float(np.mean(xs)), sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else None)


def differ(a, b):
    """the significance rule of the test plan, None when either side lacks a spread"""
    if not a or not b or a['sd'] is None or b['sd'] is None:
        return None
    return abs(a['mean'] - b['mean']) > 2*np.sqrt(a['sd']**2/a['n'] + b['sd']**2/b['n'])


res = {'cells': {}, 'published_factorized': {}, 'comparisons': {}, 'control': {}}
for design in ('fact', 'concat', 'late', 'mixe', 'mixl'):
    for stage in STAGES:
        r = runs(design, stage)
        if r:
            res['cells'][f'{design}:{stage}'] = dict(runs=r, width=stats([x[1] for x in r]), baryon=stats([x[2] for x in r]))
for stage, f in PUBLISHED_FACT.items():
    d = load(f)
    if d:
        res['published_factorized'][stage] = dict(width=d['master'], baryon=d['per_obs'].get('nbaryon'))
cell = lambda key, q='width': (res['cells'].get(key) or {}).get(q)
for stage in STAGES:
    for other in ('concat', 'late'):
        for q in ('width', 'baryon'):
            a, b = cell(f'fact:{stage}', q), cell(f'{other}:{stage}', q)
            if a and b:
                res['comparisons'][f'fact-{other}:{stage}:{q}'] = dict(
                    fact=a['mean'], other=b['mean'], difference=a['mean'] - b['mean'], significant=differ(a, b))
for ctrl in ('mixe', 'mixl'):
    c = cell(f'{ctrl}:MIX')
    if not c:
        continue
    like_fact = differ(c, cell('fact:MIX')) is False
    like_plain = differ(c, cell('concat:MIX' if ctrl == 'mixe' else 'late:MIX')) is False
    verdict = ('gain belongs to the mixture head' if like_fact and not like_plain else
               'gain belongs to the factorized form' if like_plain and not like_fact else
               'in between, report both' if not like_fact and not like_plain else 'not resolved by the spreads')
    res['control'][ctrl] = dict(width=c, like_factorized=like_fact, like_plain=like_plain, verdict=verdict)

fmt = lambda s: '      --      ' if not s else (f"{s['mean']:.3f} +- {s['sd']:.3f} ({s['n']})" if s['sd'] is not None
                                                 else f"{s['mean']:.3f}         (1)")
print(f"{'stage':6s} {'factorized':>20s} {'concatenation':>20s} {'late fusion':>20s} {'mix head, concat':>20s} {'mix head, late':>20s}")
for stage in STAGES:
    print(f"{stage:6s} " + ' '.join(f'{fmt(cell(f"{d}:{stage}")):>20s}' for d in ('fact', 'concat', 'late', 'mixe', 'mixl')))
print('baryon count:')
for stage in STAGES:
    print(f"{stage:6s} " + ' '.join(f'{fmt(cell(f"{d}:{stage}", "baryon")):>20s}' for d in ('fact', 'concat', 'late', 'mixe', 'mixl')))
for k, v in res['comparisons'].items():
    if v['significant'] is not None:
        print(f"  {k:28s} difference {v['difference']:+.3f}  {'significant' if v['significant'] else 'within the spread'}")
for k, v in res['control'].items():
    print(f"  control {k}: {v['verdict']}")
json.dump(res, open('output/seedstudy_summary.json', 'w'), indent=1)
print('wrote output/seedstudy_summary.json')
