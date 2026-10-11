#!/usr/bin/env python3
"""Train the mixture stage on any mixture data set.

The default is the MULTIPLICATIVE (geometric) mixture of the two generators,

    q(Phi; theta_S, theta_H, f) ~ q_S(Phi; theta_S)^(1 - f) q_H(Phi; theta_H)^f ,

whose log ratio to the pooled reference is linear in the fraction, (1 - f) l_S + f l_H, so the
whole model stays one exponential family in theta and the head stays an inner product
<a(Phi), (1 - f) b_S(theta_S) + f b_H(theta_H)>. That is what the wifi basis (wifi_embed.py)
and the maximum-entropy tilt need. No generator samples the interpolation between the ends,
so the classifier targets are the pure runs of each generator (LADDER_TARGET_PURE=1), and the
wifi protocol holds a fifth of each run out of training for the weight fit and trains each
member on a bootstrap resample (LADDER_FIT_FRAC=0.2, LADDER_BOOTSTRAP=1). Set
LADDER_HEAD_KIND=mixture for the additive mixture (1 - f) q_S + f q_H of the earlier drafts.

This replaces stageDM.py, which knew that the mixture had seven parameters and carried their
box as a literal list. Everything here comes from the data set's own meta.json, which
make_mixture_data.py (or make_pure_dataset.py) writes with the axes and the box of each side,
so the eight-parameter Sherpa and Herwig designs give a seventeen-parameter mixture with
nothing to edit. The block sizes LADDER_MIX_SPLIT follow from the same file.

The head is the EXACT mixture form, not a generic function of seventeen inputs. That matters
at the edges: at f = 0 the derivative with respect to theta_H vanishes identically, so a
fraction of zero is the first generator alone rather than an approximation to it.

The rank deserves a word, because it is the one number here that is a choice. The score
expansion needs 1 + d directions to first order and 1 + d + d(d+1)/2 to second. At d = 17
those are 18 and 171, and 171 is a large head to train. LADDER_K defaults to first order plus
a margin, because the single-generator rank scans measure where the descent actually stops,
and the honest procedure is to take that measurement rather than to assume the second-order
count is needed. Set LADDER_K explicitly to override.

    DM_DATA=data_stagePURE17 LADDER_K=48 python stage_mixture.py
"""
import json, os

DATA = os.environ.get('DM_DATA', 'data_stagePURE17' if os.path.exists('data_stagePURE17/meta.json') else 'data_stageDM17')
meta = json.load(open(f'{DATA}/meta.json'))
train = {m['rid']: tuple(m['theta']) for m in meta['train']}
held = {m['rid']: tuple(m['theta']) for m in meta['held']}
s_box, h_box = meta['s_box'], meta['h_box']
d = meta['ntheta']
assert len(s_box) + len(h_box) + 1 == d, f'{len(s_box)} + {len(h_box)} + 1 is not {d}'
assert all(len(t) == d for t in train.values()), 'a training theta has the wrong length'
assert all(len(t) == d for t in held.values()), 'a held theta has the wrong length'

K1 = 1 + d
K2 = 1 + d + d*(d + 1)//2
os.environ.setdefault('LADDER_HEAD_KIND', 'geometric')
os.environ.setdefault('LADDER_MIX_SPLIT', f'{len(s_box)},{len(h_box)}')
if os.environ['LADDER_HEAD_KIND'] == 'geometric':
    # the wifi protocol: pure-run targets, a fit set held out of training, bootstrap members.
    # The fit split needs the per-run reference draws of the low-memory path.
    for _k, _v in (('LADDER_TARGET_PURE', '1'), ('LADDER_FIT_FRAC', '0.2'),
                   ('LADDER_BOOTSTRAP', '1'), ('LADDER_LOWMEM', '1')):
        os.environ.setdefault(_k, _v)
    _pure = [t for t in train.values() if t[-1] in (0, 1)]
    assert _pure, 'the geometric mixture trains on pure runs, and this data set has none at f = 0 or 1'
# The export filename is the stage key followed by LADDER_SUFFIX, so a suffix of 'MIX17' on the
# stage key 'MIX' writes MIXMIX17_cond.npz. DM_TAG is the tag you WANT, and the suffix is
# derived from it. Getting this wrong once produced a correct model under a name nothing else
# looked for, which is the same failure as a doubled filename: harmless to the physics and
# fatal to every script downstream.
STAGE_KEY = 'MIX'
TAG = os.environ.get('DM_TAG', 'MIX17')
# setdefault is not enough here. train_stage.sbatch does `export LADDER_SUFFIX=${LADDER_SUFFIX:-}`,
# which leaves the name PRESENT and EMPTY, and setdefault does not replace a key that exists. The
# suffix would stay empty, the export would be written as MIX_cond.npz, and the assertion at the
# end of this file would look for exactly that and find it, so a run under the wrong name would
# look entirely successful. Treat empty as unset, and refuse to run with no suffix at all.
_suf = TAG[len(STAGE_KEY):] if TAG.startswith(STAGE_KEY) else TAG
if not os.environ.get('LADDER_SUFFIX', ''):
    os.environ['LADDER_SUFFIX'] = _suf
assert os.environ['LADDER_SUFFIX'], (
    f'LADDER_SUFFIX is empty and DM_TAG={TAG!r} yields no suffix, so the export would be written '
    f'as {STAGE_KEY}_cond.npz with nothing to distinguish it from another mixture run.')
os.environ.setdefault('LADDER_K', str(K1 + 6))
# A dead ensemble member on a seventeen-parameter head costs hours, so revive it and report
# rather than discarding the run. Both are recorded in the export either way.
os.environ.setdefault('LADDER_DEAD', 'report')
os.environ.setdefault('LADDER_REVIVE', '1')

import r2_ladder as R

# norm is (centre, half range) per parameter, the convention every stage uses. The fraction is
# a physical number on [0, 1], so it standardizes to (0.5, 0.5).
norm = [(float((lo + hi)/2), float((hi - lo)/2)) for lo, hi in s_box + h_box] + [(0.5, 0.5)]
for i, (c, h) in enumerate(norm):
    assert h > 0, f'parameter {i} has a zero range, so the design never moved it'

R.STAGES[STAGE_KEY] = dict(
    data=DATA, flavor=True, ntheta=d, train=train, held=held, norm=norm,
    obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'],
    flav_obs=['nbaryon', 'strange'])
# Keep the pooled reference near the single-generator stages' 1.15M rather than letting it
# scale with the number of runs, and never ask a run for more events than it holds.
target = int(os.environ.get('DM_REF_TOTAL', '1150000'))
R.NREF_PER = min(meta['n_train_events'], max(1000, target//max(len(train), 1)))

print(f'MIXTURE STAGE: d={d} ({len(s_box)} + {len(h_box)} + 1), '
      f'{len(train)} train x {meta["n_train_events"]}, {len(held)} held, '
      f'K={R.K} (first order {K1}, second order {K2}), '
      f'head={os.environ["LADDER_HEAD_KIND"]}, '
      f'ref {len(train)*R.NREF_PER} at {R.NREF_PER} per run, device {R.DEV}', flush=True)
print(f'  Sherpa axes {meta["s_axes"]}')
print(f'  Herwig axes {meta["h_axes"]}')

if os.environ.get('DM_DRYRUN', '') == '1':
    # Check the configuration without touching a GPU or an event file. Worth a minute before
    # a queued job, because every mistake below is one that produces a plausible model.
    import numpy as _np
    T = _np.array(list(train.values()))
    N = _np.array(norm)
    Z = (T - N[:, 0])/N[:, 1]
    print(f'  standardized training thetas: min {Z.min():.3f}, max {Z.max():.3f} '
          f'(inside [-1, 1] means the box covers the design)')
    bad = [i for i in range(d) if abs(Z[:, i]).max() > 1 + 1e-9]
    print(f'  axes whose design leaves the box: {bad if bad else "none"}')
    f = T[:, -1]
    print(f'  fraction f: {len(f[f == 0])} runs at 0, {len(f[f == 1])} at 1, '
          f'{len(f[(f > 0) & (f < 1)])} interior, values {sorted(set(_np.round(f, 4)))}')
    print('DRY RUN: configuration only, nothing trained')
    raise SystemExit(0)

result = R.run_stage(STAGE_KEY)
# The export must be where the rest of the pipeline looks for it.
_want = f'output/models/{STAGE_KEY}{os.environ["LADDER_SUFFIX"]}_cond.npz'
assert os.path.exists(_want), (
    f'the run finished but wrote no export at {_want}. LADDER_SUFFIX is '
    f'{os.environ["LADDER_SUFFIX"]!r} and the stage key is {STAGE_KEY!r}, so check DM_TAG.')
print(f'export written to {_want}')
out = (os.environ.get('STAGED_OUT') or
       f'output/stage_mixture_{os.environ["LADDER_SUFFIX"]}.json')
json.dump(result, open(out, 'w'), indent=1)
print(f'MIXTURE STAGE DONE, wrote {out}')
