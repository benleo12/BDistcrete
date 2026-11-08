#!/usr/bin/env python3
"""Every .npz under ROOT whose arrays look like per-event observables (names of the CSV shapes, the
kaon fraction, held_* arrays, x_p): path, size, the matching keys and their shapes. Headers only.
    python scan_obs_cache.py ROOT OUT_TSV"""
import os, sys, zipfile, re
import numpy as np
ROOT, OUT = sys.argv[1], sys.argv[2]
PAT = re.compile(r'(1_minus_thrust|B_total|rho_heavy|mult_total|strange|thrust|xp|x_p|^held_|_held|nch$|mult)', re.I)
rows = []
for dp, dn, fn in os.walk(ROOT, followlinks=False):
    for f in sorted(fn):
        if not f.endswith('.npz'):
            continue
        p = os.path.join(dp, f)
        try:
            link = os.path.islink(p)
            with zipfile.ZipFile(p) as zf:
                keys = []
                for n in zf.namelist():
                    k = n[:-4]
                    if PAT.search(k):
                        with zf.open(n) as fh:
                            v = np.lib.format.read_magic(fh)
                            sh = (np.lib.format.read_array_header_1_0(fh) if v == (1, 0) else np.lib.format.read_array_header_2_0(fh))[0]
                        keys.append(f'{k}{list(sh)}')
            if keys:
                rows.append((os.path.relpath(p, ROOT), str(os.path.getsize(p)), 'link' if link else 'file',
                             str(len(keys)), ' '.join(keys[:12]) + (' ...' if len(keys) > 12 else '')))
        except Exception as e:
            rows.append((os.path.relpath(p, ROOT), str(os.path.getsize(p) if os.path.exists(p) else -1), 'ERR', '0', repr(e)[:200]))
with open(OUT, 'w') as fh:
    fh.write('path\tsize\tkind\tnkeys\tkeys\n')
    for r in rows:
        fh.write('\t'.join(r) + '\n')
print(len(rows), 'files with observable-like arrays')
