"""Singular-value spectrum of the learned f(Phi, theta) for stages the original run omitted.

Panel (b) of the head figure reads output/ab_analysis.json['svd_spectrum'], which was written
for the first three stages only. This adds entries without recomputing the rest, since the
other panels of that analysis depend on the older exports and rerunning the whole thing would
requote numbers the paper already carries.
"""
import json, os, sys
import numpy as np
from ab_analysis import Cond, svd_spectrum

out = 'output/ab_analysis.json'
d = json.load(open(out)) if os.path.exists(out) else {}
spec = d.setdefault('svd_spectrum', {})
for st in sys.argv[1:]:
    c = Cond(st)
    s = svd_spectrum(c)
    spec[st] = s
    e = np.cumsum(np.array(s)**2)/np.sum(np.array(s)**2)
    K2 = 1 + c.nt + c.nt*(c.nt+1)//2
    print(f'[{st}] d={c.nt}, K2={K2}, {len(s)} singular values')
    print(f'   normalized top 8: {np.round(s[:8], 4).tolist()}')
    for k in (c.nt, c.nt+1, 2*c.nt, min(K2, len(e))):
        print(f'   variance captured by rank {k:>2}: {100*e[k-1]:.2f}%')
json.dump(d, open(out, 'w'), indent=1)
print(f'wrote {out}, svd_spectrum now holds {sorted(spec)}')
