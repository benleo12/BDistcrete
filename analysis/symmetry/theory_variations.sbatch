#!/bin/bash
#SBATCH -p debugq
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J thvars
#SBATCH -o logs/theory_variations_%j.out
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export TORCH_THREADS=4 RIVET_REF=$MAXENT_DIR/rivet_ref
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
python3 -u theory_variations.py MIX output/models/MIXGEOV3S9_cond.npz output/models/MIXGEOV3S9_ref_v2_slim.npz output/thrust_targets_nlo_point.npz || echo "THVARS FAILED"
