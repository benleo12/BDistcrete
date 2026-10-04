#!/usr/bin/env python3
"""App. D: the third-order cumulant of the production run (698 jobs, infrared cutoff y0 = 1e-8)
against an independent production of 504 jobs with y0 = 1e-7, both rank-trimmed
(eerad_cumulant_combine.py). Prints the largest difference over the values of tau below 0.33 in
units of the combined bootstrap error.

    python eerad_y7_check.py
"""
import numpy as np
g, C, S = np.load('output/eerad_NNLO_cum.npy')
g2, C2, S2 = np.load('output/eerad_NNLO_ranktrim_y7.npy')
assert np.allclose(g, g2), 'the two cumulants are on different tau grids'
m = (g > 0) & (g < 0.33)
pull = (C2[m] - C[m])/np.hypot(S[m], S2[m])
print(f'{int(m.sum())} values of tau below 0.33: largest |difference| {np.abs(pull).max():.2f} combined errors, '
      f'{int((np.abs(pull) > 1).sum())} beyond one')
