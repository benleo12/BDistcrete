#!/bin/bash
# Assemble the mixture data set from the existing single-generator samples. Pure CPU work:
# every mixture run is a resampled blend of one Sherpa run and one Herwig run, drawn without
# replacement, so nothing is generated and no GPU is needed.
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -J mixdata
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/mixdata_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/mixdata_%j.err
set -e
cd $SCRATCH/maxent_code
mkdir -p logs output
module load python >/dev/null 2>&1 || module load pytorch/2.6.0
export S_SRC=${S_SRC:-data_stageE} H_SRC=${H_SRC:-data_stageF}
export S_DESIGN=${S_DESIGN:-data_stageE/stageE_design.csv}
export H_DESIGN=${H_DESIGN:-data_stageF/stageF_design.csv}
export DM_DST=${DM_DST:-$SCRATCH/stageDM17/data}
export DM_NTRAIN=${DM_NTRAIN:-192} DM_NHELD=${DM_NHELD:-10}
mkdir -p "$DM_DST"
echo "building into $DM_DST from $S_SRC and $H_SRC, $DM_NTRAIN train runs"
python3 -u make_mixture_data.py
rc=$?
# A symlink beside the other stages, so the training driver finds it the same way. Remove any
# existing entry FIRST: `ln -sfn` against an existing directory silently creates the link
# inside it rather than replacing it, which is how a stray directory here once shadowed the
# real data set. Refuse if a non-symlink directory holds a finished build.
# The symlink name must track DM_DST, or a second mixture build silently shadows the first.
LINK=$SCRATCH/maxent_code/${DM_LINK:-data_stageDM17}
if [ -d "$LINK" ] && [ ! -L "$LINK" ]; then
  if [ -f "$LINK/meta.json" ]; then
    echo "REFUSING to replace $LINK: it is a real directory holding a finished build"; exit 3
  fi
  echo "removing a partial non-symlink directory at $LINK"; rm -rf "$LINK"
fi
rm -f "$LINK"
ln -s "$DM_DST" "$LINK"
ls -ld "$LINK"
echo "MIXTURE DATA BUILD DONE rc=$rc $(date)"
du -sh "$DM_DST"
exit $rc
