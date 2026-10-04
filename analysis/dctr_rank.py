"""Effective rank of the trained concatenation (DCTR) networks, against the score-expansion counts.

concat_baseline.py evaluates each trained concatenation ensemble on 5000 reference events at 48
parameter points and stores the singular values of that 5000 x 48 logit matrix, normalized to the
largest. Counting those above 3 percent gives the effective rank of the learned log ratio. The
score expansion predicts d+1 dominant terms and at most K2 = 1 + d + d(d+1)/2 at second order. The
counts follow it although nothing in the concatenation network imposes a rank: the low rank is a
property of the log ratio, which any network that learns it inherits.

    python dctr_rank.py            # threshold 0.03
    python dctr_rank.py 0.01
"""
import json, sys
import numpy as np

FILES = {'A': ('output/concat_baseline_A_prod.json', 1), 'B': ('output/concat_baseline_B_prod.json', 2),
         'C': ('output/concat_baseline_C_prod.json', 3), 'E': ('output/concat_baseline_E_silu.json', 8),
         'MIX': ('output/concat_baseline_MIX_silu.json', 17)}
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
print(f'{"stage":6s}{"d":>4s}{"d+1":>6s}{"K2":>6s}{"rank":>6s}   closure width')
for k, (f, d) in FILES.items():
    r = json.load(open(f))[k]
    sv = np.asarray(r['svd'])
    print(f'{k:6s}{d:4d}{d + 1:6d}{1 + d + d*(d + 1)//2:6d}{int((sv > thr).sum()):6d}   {r["master"]:.2f}')
