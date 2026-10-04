#!/usr/bin/env bash
# Re-extract the anchor's second campaign from its raw HepMC with exact event shapes and
# parton-level thrust. Same events as data_stageC_1M, corrected observables, plus tau_parton.
set -u
PY=python
SD=.; SRC=${ANCHOR1M_HEPMC:?set to the directory of the 1M-campaign HepMC files}; DST=$SD/data_stageC_1M_v2
# compute_shapes_only wants an integer run id and writes shapes_run_<id>.csv, so the campaign's
# own index (run_00 ... run_31) is the id, and the particle file keeps the full parameter tag.
one () { local d=$1 tag=$(basename $1); local idx=${tag:4:2}; local rid=$((10#$idx))
  local shp=$(printf "$DST/shapes_run_%04d.csv" $rid)
  [ -f "$DST/particles_full_$tag.npz" ] && [ -f "$shp" ] && { echo "skip $tag"; return; }
  echo "$rid $tag" >> $DST/run_map.txt
  $PY $SD/release/generators/extract_particles_full_flavor.py --input $d/events.hepmc.gz --output $DST/particles_full_$tag.npz > $DST/extract_$tag.log 2>&1 || { echo "EXTRACT FAIL $tag"; return; }
  $PY $SD/release/generators/compute_shapes_only.py --input $d/events.hepmc.gz --output-dir $DST --run-id $rid > $DST/shapes_$tag.log 2>&1 || { echo "SHAPES FAIL $tag"; return; }
  echo "done $tag -> shapes_run_$(printf %04d $rid).csv"
}
export -f one; export PY SD DST
ls -d $SRC/run_* | xargs -P ${NPAR:-4} -I{} bash -c 'one {}'
echo "ANCHOR1M REEXTRACT DONE"
