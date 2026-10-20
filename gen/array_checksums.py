"""md5 of every array in every particles_full_<rid>.npz of a directory, one line per run, so two
copies of a data set can be compared without moving them (zip headers differ, arrays must not).
    python array_checksums.py <dir> > checksums.txt"""
import sys, glob, hashlib, os, re
import numpy as np
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'particles_full_*.npz'))):
    z = np.load(f)
    h = ' '.join(f'{k}:{hashlib.md5(np.ascontiguousarray(z[k]).tobytes()).hexdigest()[:12]}' for k in sorted(z.files))
    print(re.search(r'_(\d+)\.npz$', f).group(1), h, flush=True)
