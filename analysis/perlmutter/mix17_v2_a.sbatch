#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J mix17v2a
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/mix17v2a_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/mix17v2a_%j.err
# Part A of the corrected-shape rebuild: the MIX17aug mixture data rebuilt from the regenerated
# E and F runs (same seed, same designs, NEW directory) and verified against the data the head
# was trained on, with the E and Fauglong reference bundles replayed alongside.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
python3 -u rebuild_ref_lowmem.py E > logs/refv2_E.log 2>&1 &
pE=$!
python3 -u rebuild_ref_lowmem.py Fauglong > logs/refv2_Fauglong.log 2>&1 &
pF=$!
echo "=== mixture build $(date)"
export S_SRC=data_stageE H_SRC=data_stageF
export S_DESIGN=data_stageE/stageE_design.csv H_DESIGN=data_stageF/stageF_design_aug.csv
export DM_DST=$SCRATCH/stageDM17aug_v2/data DM_NTRAIN=192 DM_NHELD=10
mkdir -p "$DM_DST"
python3 -u make_mixture_data.py || { echo "MIXTURE BUILD FAILED"; exit 2; }
LINK=$SCRATCH/maxent_code/data_stageDM17aug_v2
[ -e "$LINK" ] && [ ! -L "$LINK" ] && { echo "REFUSING: $LINK is a real directory"; exit 3; }
rm -f "$LINK"; ln -s "$DM_DST" "$LINK"; ls -ld "$LINK"
echo "=== mixture verify $(date)"
python3 -u verify_mix_v2.py data_stageDM17aug data_stageDM17aug_v2 || { echo "MIXTURE VERIFY FAILED"; exit 4; }
rc=0; wait $pE || rc=1; wait $pF || rc=$((rc+2))
for t in E Fauglong; do echo "--- $t"; tail -9 logs/refv2_$t.log; done
echo "PART A DONE rc=$rc $(date)"
exit $rc
