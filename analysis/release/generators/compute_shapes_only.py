#!/usr/bin/env python3
"""Fast event-shapes-only extraction from a HepMC3 file (no EFPs).
Reuses load_events + compute_hemisphere_shapes from compute_efps.py. The EFP graph
contractions are the slow part; the headline observables are event shapes (thrust,
broadenings, hemisphere masses, multiplicities), which are cheap. Writes
<output-dir>/shapes_run_<rid>.csv with the same columns as the existing pipeline.
The photons radiated by the beams are left out of every shape and of the multiplicities, as in
the paper's observables (compute_efps.py says how they are found); --beam-photons keep counts
them, as the shape files of an earlier draft did."""
import argparse, pandas as pd
from pathlib import Path
from compute_efps import load_events, compute_hemisphere_shapes
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--run-id',type=int,required=True); ap.add_argument('--max-events',type=int,default=None)
    ap.add_argument('--fmt',choices=['auto','hepmc2','hepmc3','herwig'],default='auto')
    ap.add_argument('--beam-photons',choices=['remove','keep'],default='remove',
                    help='remove (the paper) or keep the photons radiated by the beams')
    a=ap.parse_args()
    rows=[compute_hemisphere_shapes(ev) for ev in load_events(a.input,a.max_events,fmt=a.fmt,beam_photons=a.beam_photons)]
    out=Path(a.output_dir)/f'shapes_run_{a.run_id:04d}.csv'
    pd.DataFrame(rows).to_csv(out,index=False); print(f'wrote {out} ({len(rows)} events)')
if __name__=='__main__': main()
