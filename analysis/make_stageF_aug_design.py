#!/usr/bin/env python3
"""Combine the original 64-point Herwig design with the 48 off-locus augmentation points.

Writes stageF_design_aug.csv, which r2_ladder reads when STAGEF_CSV points at it. The original
file is untouched, so the 64-point head stays reproducible and the two can be compared.

The check that matters here is the BOX. _stage_from_csv takes each parameter's normalization
from the design's own min and max, so a combined design whose extremes differ from the original
would give the head a different standardization, and the retrained head would then differ for
two reasons at once instead of one. The augmentation points were drawn inside the original box
by construction, so the box must come out identical, and this asserts it rather than trusting it.
"""
import sys
import pandas as pd
import numpy as np

AX = ['alpha_fsr', 'ptmin', 'clmax', 'clpow', 'psplit', 'pwtsquark', 'pwtdiquark', 'clsmr']
BASE, AUG, OUT = 'stageF_design.csv', 'stageF_augment_design.csv', 'stageF_design_aug.csv'

b = pd.read_csv(BASE)
a = pd.read_csv(AUG)
assert list(a.columns) == list(b.columns), f'columns differ:\n{list(b.columns)}\n{list(a.columns)}'
assert set(a.run_id) & set(b.run_id) == set(), 'the augmentation reuses a run id'
assert (a.role == 'train').all(), 'the augmentation must add training points only'

out = pd.concat([b, a], ignore_index=True)
lo_b, hi_b = b[AX].min(), b[AX].max()
lo_o, hi_o = out[AX].min(), out[AX].max()
bad = [x for x in AX if abs(lo_b[x] - lo_o[x]) > 1e-12 or abs(hi_b[x] - hi_o[x]) > 1e-12]
assert not bad, (f'the combined design moves the box on {bad}. The head would then be '
                 f'standardized differently and the retrain would not be a controlled comparison.')

for r in ('train', 'held'):
    print(f'  {r}: {(b.role == r).sum()} -> {(out.role == r).sum()}')


def stats(d):
    Z = (d[AX].values - (lo_b.values + hi_b.values)/2)/((hi_b.values - lo_b.values)/2)
    ev = np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T))
    return float(np.sqrt(ev.max()/ev.min())), float(ev.sum()**2/(ev**2).sum())


for lab, d in (('original 64', b[b.role == 'train']), ('combined 112', out[out.role == 'train'])):
    an, ed = stats(d)
    print(f'  {lab}: anisotropy {an:.2f}, effective dimensions {ed:.2f}')

if '--write' in sys.argv:
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT}: {len(out)} rows, box unchanged on all {len(AX)} axes')
else:
    print('checks pass. rerun with --write to produce the file')
