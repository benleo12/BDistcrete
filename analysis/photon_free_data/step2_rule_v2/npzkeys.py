#!/usr/bin/env python3
"""List the arrays of npz files (name, shape, dtype) from their npy headers, without loading them.
    python npzkeys.py FILE [FILE ...]"""
import sys, zipfile
import numpy as np
for path in sys.argv[1:]:
    try:
        with zipfile.ZipFile(path) as zf:
            out = []
            for n in zf.namelist():
                with zf.open(n) as fh:
                    v = np.lib.format.read_magic(fh)
                    sh, fo, dt = (np.lib.format.read_array_header_1_0(fh) if v == (1, 0) else np.lib.format.read_array_header_2_0(fh))
                out.append(f'{n[:-4]}{list(sh)}:{dt.str}')
        print(path, ' '.join(out))
    except Exception as e:
        print(path, 'ERROR', e)
