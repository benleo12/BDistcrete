"""Fraction of events with more than 100 kept particles (the input keeps the 100 most energetic),
over the training runs of the final models, from the full multiplicity in the shape files: the
mixture of the 720 runs (data_stagePURE17_v3) and the densified designs of Stages A to C
(gen/data_abc, training runs of gen/designs/stage<S>_design_aug.csv). Writes
output/truncation_fraction.json, which make_numbers.py reads. Run from analysis/.

    python trunc_frac.py [<mixture data> <Stage A-C data> <designs folder> [<output>]]

The defaults are data_stagePURE17_v3, ../gen/data_abc, ../gen/designs and
output/truncation_fraction.json. Every file must exist. The count is that of the network input, whose
particles include the beam photons (D4 leaves the inputs alone), so it needs mult_total as the original
shapes files store it, beam photons included. The shapes files of the photon-free (_nb) folders hold
the multiplicity without them and would count fewer truncated events than the network sees, so a
shapes file that resolves into an _nb folder is refused. The rerun counts it again with
symmetry/rerun/trunc_lmx.sbatch, on the original folders of the published run (read only; in RUN the
default folders are _nb links) and with BEAM_PHOTONS unset. Nothing is written through a link.
"""
import json, os, sys
import numpy as np, pandas as pd
import score_obs as SO
MIX, ABC, DES = (sys.argv[1:4] if len(sys.argv) >= 4 else ('data_stagePURE17_v3', '../gen/data_abc', '../gen/designs'))
OUT = sys.argv[4] if len(sys.argv) > 4 else 'output/truncation_fraction.json'
print(f'mixture {MIX}, Stages A-C {ABC}, designs {DES} -> {OUT}', flush=True)
if os.path.islink(OUT):
    raise RuntimeError(f'{OUT} is a link to {os.path.realpath(OUT)}: write to a file of its own')


def need(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f'missing input {p}')
    return p


sets = {'mixture (720 runs)': ['%s/shapes_run_%04d.csv' % (MIX, t['rid']) for t in json.load(open(need(f'{MIX}/meta.json')))['train']]}
for S in 'ABC':
    d = pd.read_csv(need(f'{DES}/stage{S}_design_aug.csv'))
    sets[f'Stage {S}'] = [f'{ABC}/shapes_run_{r:04d}.csv' for r in d[d.role == 'train'].run_id]
for files in sets.values():
    for f in files:
        if SO.photon_free(f):
            raise RuntimeError(f'{f} resolves to {os.path.realpath(f)}, in a photon-free (_nb) folder whose mult_total '
                               f'leaves out the beam photons; the network input keeps them: use the original shapes files')
out = {}
for name, files in sets.items():
    n = over = 0; mx = 0
    for f in files:
        m = pd.read_csv(need(f), usecols=['mult_total'])['mult_total'].values
        n += len(m); over += int((m > 100).sum()); mx = max(mx, int(m.max()))
    out[name] = dict(events=n, over_100=over, fraction=over/n, max=mx)
    print(name, n, over, '%.4f%%' % (100*over/n), 'max', mx, flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
