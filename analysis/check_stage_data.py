#!/usr/bin/env python3
"""Refuse to train unless every run in the design has its data on disk.

A design CSV that lists 112 training runs while only 64 have events is the failure this
blocks. It does not crash the trainer in an obvious way, it just trains on whatever loaded,
and the resulting head looks entirely normal. Exit 3 means a real shortfall; exit 0 means the
stage is complete and training may proceed.

usage: check_stage_data.py <stage>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1]
os.environ.setdefault('LADDER_STAGES', tag)
import r2_ladder as R

# A mixture stage is not in the static registry. Its design lives in its data set's meta.json and
# is registered at training time by stage_mixture.py, so looking it up here raised KeyError and,
# because this script is a gate, refused a perfectly good run. Register it the same way.
if tag not in R.STAGES:
    try:
        from mixture_cfg import register as _register_mixture
        _register_mixture(R.STAGES, [tag], data=os.environ.get('DM_DATA'))
    except Exception as e:
        print(f'stage {tag} is not in the registry and no mixture config could be built: {e}')
        sys.exit(2)
if tag not in R.STAGES:
    print(f'stage {tag} is not in the registry. For a mixture, set DM_DATA to its data set.')
    sys.exit(2)

S = R.STAGES[tag]
data = S['data']
missing = []
for kind, rids in (('train', sorted(S['train'])), ('held', sorted(S['held']))):
    for rid in rids:
        for f in (f'{data}/particles_full_{rid:04d}.npz', f'{data}/shapes_run_{rid:04d}.csv'):
            if not os.path.exists(f):
                missing.append((kind, rid, f))
n_tr, n_hd = len(S['train']), len(S['held'])
print(f'stage {tag}: design lists {n_tr} training and {n_hd} held runs in {data}')
if missing:
    rids = sorted({r for _, r, _ in missing})
    print(f'  MISSING data for {len(rids)} run(s): {rids}')
    for k, r, f in missing[:10]:
        print(f'    {k} {r}: {f}')
    if len(missing) > 10:
        print(f'    ... and {len(missing)-10} more files')
    sys.exit(3)
print(f'  all {n_tr + n_hd} runs have both particles and shapes. complete.')
