#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 05:00:00
#SBATCH -N 2
#SBATCH -J profvar
#SBATCH -o logs/profvar_%j.out
#SBATCH -e logs/profvar_%j.err
# Direct profile of one fit (or fit variant) on Perlmutter: the alpha_0 columns in parallel, a merge
# that also takes any seed part files, then the neighbour polish repeated until no node improves by
# more than TOL.
#   TAG             output tag: columns output/profile_${TAG}_central_a0<a0>.json
#   MODEL_TAG       the trained head, e.g. MIX17aug
#   A0_LIST         the alpha_0 columns to compute, space separated (may be empty: polish only)
#   STARTS_PATTERN  optional, e.g. output/profile_MIX17ext_central_a0{a0}.json, used where it exists
#   STARTS_FIXED    optional, one column file used as extra starts for every new column
#   MIX_FORM, LAST_BIN, USE_NCH, NSCAN, DCHI_MAX, NEIGH, MAXPASS, TOL pass through
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export RIVET_REF=rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32
export EXPORT_PATH=output/models/${MODEL_TAG:?}_cond.npz REF_PATH=output/models/${MODEL_TAG}_ref_v2_slim.npz
export DCHI_MAX=${DCHI_MAX:-15} NEIGH=${NEIGH:-8} NSCAN=${NSCAN:-96}
TAG=${TAG:?}; MAXPASS=${MAXPASS:-4}; TOL=${TOL:-0.01}; NP=$((SLURM_NNODES*32))
read -ra A0S <<< "${A0_LIST:-}"; NC=${#A0S[@]}
echo "=== $TAG ($MODEL_TAG, mix form ${MIX_FORM:-additive}, last bin ${LAST_BIN:-none}, nch ${USE_NCH:-1}): $NC new columns $(date)"
if [ $NC -gt 0 ]; then
  cat > logs/profvar_${TAG}_task.sh <<EOS
#!/bin/bash
A0S=(${A0_LIST}); a0=\${A0S[\$SLURM_PROCID]}
sf=""
if [ -n "${STARTS_PATTERN:-}" ]; then f=\$(echo "${STARTS_PATTERN:-}" | sed "s/{a0}/\$a0/")
  # only a complete column file: another job may be writing it
  [ -f "\$f" ] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "\$f" 2>/dev/null && sf="\$f"; fi
if [ -n "${STARTS_FIXED:-}" ]; then sf="\${sf:+\$sf,}${STARTS_FIXED:-}"; fi
STARTS_FROM="\$sf" ONLY_A0=\$a0 PROFILE_OUT=output/profile_${TAG}_central_a0\$a0.json \\
  python3 -u profile_column.py ${MODEL_TAG} > logs/profile_column_${TAG}_a0\$a0.log 2>&1
echo "column \$a0 exit \$?"
EOS
  chmod +x logs/profvar_${TAG}_task.sh
  srun -N $SLURM_NNODES -n $NC -c 8 --cpu-bind=cores logs/profvar_${TAG}_task.sh
  echo "=== columns done $(date)"
fi
python3 -u polish_parallel.py merge $TAG central > logs/polish_${TAG}_merge0.log 2>&1
cat logs/polish_${TAG}_merge0.log
for pass in $(seq 1 $MAXPASS); do
  rm -f output/polish_${TAG}_central_part*.json
  echo "=== polish pass $pass $(date)"
  srun -N $SLURM_NNODES -n $NP -c 8 --cpu-bind=cores bash -c \
    "FROM_MERGED=1 PART=\$SLURM_PROCID NPART=$NP python3 -u polish_parallel.py part $TAG central > logs/polish_${TAG}_pass${pass}_part\$SLURM_PROCID.log 2>&1"
  FROM_MERGED=1 python3 -u polish_parallel.py merge $TAG central > logs/polish_${TAG}_pass${pass}_merge.log 2>&1
  cat logs/polish_${TAG}_pass${pass}_merge.log
  worst=$(grep "nodes improved" logs/polish_${TAG}_pass${pass}_merge.log | sed 's/.*largest improvement //')
  python3 -c "import sys; sys.exit(0 if float('$worst') < $TOL else 1)" && { echo "CONVERGED after pass $pass"; break; }
done
echo "PROFILE DONE $TAG $(date)"
