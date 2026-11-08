#!/bin/bash
# The scoring side of the rerun, submitted in order with its dependencies (on the login node of
# Symmetry; it only submits). Every job is named lmx_*, runs in RUN and writes under RUN/output and
# RUN/logs. Prints the job ids.
#   export MAXENT_DIR=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
#   bash $MAXENT_DIR/symmetry/rerun/submit_scoring_lmx.sh
# Before anything is submitted: the partition and the other sbatch values (LMX_CPUS, LMX_MEM, LMX_GRES, NPAR, NPAR_CAT:
# plain numbers, gpu:h200:<n>) are checked, setup_lmx.py checks RUN and makes its links,
# lmx_fingerprint.py precheck checks that the particle caches and output/bundle_rule.json follow RUN's
# beam_photons.py, the first submission moves every published copy that sits under the name of a rerun output into
# output/published_copies (setup_lmx.py --park), and the start mark output/.lmx_rerun_start is written.
# A second submission stops while the mark exists. RESUME=1 submits again (cancel the jobs of the first submission
# before), keeping the mark: a scoring is then skipped only when its result carries the fingerprint of the current
# code and data record, of its spec line and of this mark (lmx_fingerprint.py); every other step runs again. Any
# rewrite of the _nb folders, refreshed bundles or designs after the mark needs the mark removed (start over).
# Every job runs on preq (CPU nodes). The DCTR scorings (std_cat, strict_cat, SvdRank) and the toy may run on
# h200preq instead, only once the authors approve it (LMX_GPU_APPROVED=1), with the GPU request on this command
# line (LMX_CPUS, LMX_MEM and NPAR_CAT are optional), for example
#   LMX_GPU_APPROVED=1 LMX_PARTITION=h200preq LMX_GRES=gpu:h200:1 LMX_CPUS=12 LMX_MEM=128G NPAR_CAT=1 bash .../submit_scoring_lmx.sh
set -euo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
[ "${MAXENT_DIR:-}" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN"; exit 2; }
cd "$RUN"
# the partition (F6): preq, or h200preq with the authors' approval; a GPU request only on h200preq
case "${LMX_PARTITION:-preq}" in
  preq) ;;
  h200preq) [ "${LMX_GPU_APPROVED:-}" = 1 ] || { echo "LMX_PARTITION=h200preq needs LMX_GPU_APPROVED=1 (the authors' approval of a GPU partition)"; exit 2; } ;;
  *) echo "LMX_PARTITION=${LMX_PARTITION}: only preq, or h200preq with LMX_GPU_APPROVED=1"; exit 2 ;;
esac
[ -z "${LMX_GRES:-}" ] || [ "${LMX_PARTITION:-}" = h200preq ] || { echo "LMX_GRES=$LMX_GRES needs LMX_PARTITION=h200preq"; exit 2; }
# the other values that go into the sbatch options must be plain (a space would add options, a second -p among them)
plain() {   # name, value, pattern
  [[ "$2" =~ $3 ]] || { echo "$1='$2' is not of the form $3: nothing done"; exit 2; }
}
[ -z "${LMX_CPUS:-}" ] || plain LMX_CPUS "$LMX_CPUS" '^[1-9][0-9]*$'
[ -z "${LMX_MEM:-}" ] || plain LMX_MEM "$LMX_MEM" '^[1-9][0-9]*[KMGT]?$'
[ -z "${LMX_GRES:-}" ] || plain LMX_GRES "$LMX_GRES" '^gpu:h200:[1-8]$'
plain NPAR "${NPAR:-2}" '^[1-9][0-9]*$'
plain NPAR_CAT "${NPAR_CAT:-${NPAR:-2}}" '^[1-9][0-9]*$'
SB=/cm/shared/apps/slurm/current/bin/sbatch; H=symmetry/rerun; NPAR=${NPAR:-2}; NPAR_CAT=${NPAR_CAT:-$NPAR}
# the jobs that may use a GPU: preq without one unless LMX_PARTITION and LMX_GRES say otherwise
GPU="-p ${LMX_PARTITION:-preq}${LMX_GRES:+ --gres=$LMX_GRES}${LMX_CPUS:+ -c $LMX_CPUS}${LMX_MEM:+ --mem=$LMX_MEM}"
opts_ok() {   # the options of one sbatch call: exactly one partition option, -p preq or -p h200preq (last guard, F6)
  local n=0 prev= a
  for a in "$@"; do
    case "$a" in
      --partition*|-p?*) echo "sbatch option $a: only -p preq or -p h200preq" >&2; return 2 ;;
      -p) n=$((n + 1)) ;;
      *) [ "$prev" != -p ] || [ "$a" = preq ] || [ "$a" = h200preq ] || { echo "partition $a: only preq or h200preq" >&2; return 2; } ;;
    esac
    prev=$a
  done
  [ "$n" -eq 1 ] || { echo "$n partition options in '$*': exactly one" >&2; return 2; }
}
opts_ok $GPU || exit 2
PY=/mnt/beegfs/bassi/reweighting/env/torch/bin/python
# what the submitting shell must not pass on to a job (--export=ALL), cleared before any sbatch (F4): every knob of
# the scripts of the scoring side (the scorers' LADDER_*, DM_*, STAGE?_DATA, STAGE?_CSV, PHOTON_FREE_DIR_RE,
# BEAM_PHOTONS_ALLOW_MIXED, and those of the other steps), the keys of a spec line, TAGS, TORCH_PY, and every
# SBATCH_*, SALLOC_*, SRUN_* (sbatch would apply them to every job; the options below decide)
for v in $(compgen -e | grep -E '^(LADDER_|DM_|SBATCH_|SALLOC_|SRUN_|VALID_|SM_|WIFI_|REF_|RECAL_|WIDTHS_|CLOSURE_|MIX_|XP_|KSCAN_|COVERAGE_|GENSPLIT_|TOY_|RELEASE_)|^STAGE._(DATA|CSV)$|^(STAGED_OUT|STAGE_TABLE_SUBSET|PHOTON_FREE_DIR_RE|BEAM_PHOTONS_ALLOW_MIXED|TORCH_PY|NDEN|NDEN_PER|NFIT_PER|DROP|OUTLIER_SIGMA|LMX_CHECK|T_DIR|POBS_DIR|BEAM_PHOTONS|TAGS|SPECS)$' || true); do
  unset "$v"
done
unset OUT STAGE NET SEED K ENS STEPS MODEL DATA TMODE TRAINED ABC E_CSV F_CSV STAGEW_CSV REF_RUNS NREF_TOTAL SUFFIX DUMP TRUNCATE LOGIT_CACHE
if [ -e output/.lmx_rerun_start ]; then
  if [ "${RESUME:-}" != 1 ]; then
    echo "output/.lmx_rerun_start exists: the rerun was submitted before. RESUME=1 submits it again, keeping only the"
    echo "scorings whose result carries the fingerprint of the current code, its spec line and this mark."
    $PY -B $H/lmx_fingerprint.py status; exit 2
  fi
  $PY -B $H/lmx_fingerprint.py status              # an unreadable mark stops here (exit 2, F7)
fi
mkdir -p logs                    # Slurm opens the log of every job (#SBATCH -o RUN/logs/...) before the job starts
$PY -B $H/setup_lmx.py           # links and checks; stops on a problem
$PY -B $H/lmx_fingerprint.py precheck                 # the particle caches and output/bundle_rule.json (F3)
[ -e output/.lmx_rerun_start ] || $PY -B $H/setup_lmx.py --park    # the first submission: published copies aside (F1)
$PY -B $H/lmx_fingerprint.py mark                     # writes the mark, or (RESUME=1) compares the code with it
arr() {   # spec file, name, lines per job, partition options, [dependency] -> job id of an array over the file
  local n; n=$(grep -v '^[[:space:]]*#' "$H/specs/$1.txt" | grep -c '[^[:space:]]')
  opts_ok $4 || return 2
  $SB --parsable $4 -J "lmx_$2" --array=0-$(( (n + $3 - 1)/$3 - 1 )) ${5:+--dependency=$5} \
      --export=ALL,MAXENT_DIR=$RUN,SPECS=$H/specs/$1.txt,NPAR=$3 $H/score_p2_array_lmx.sbatch
}
one() {   # script, name, [dependency], [extra export], [partition options]
  opts_ok ${5:--p preq} || return 2
  $SB --parsable ${5:--p preq} -J "lmx_$2" ${3:+--dependency=$3} --export=ALL,MAXENT_DIR=$RUN${4:+,$4} "$H/$1"
}
STD_FAC=$(arr std_fac stdfac $NPAR "-p preq"); STD_CAT=$(arr std_cat stdcat $NPAR_CAT "$GPU")
RECAL=$(one recal_lc_lmx.sbatch recal)
TRUNC=$(one trunc_lmx.sbatch trunc)                   # the truncation fraction and the parameter count, counted again
STRICT_FAC=$(arr strict_fac strictfac $NPAR "-p preq" afterany:$STD_FAC)
STRICT_CAT=$(arr strict_cat strictcat $NPAR_CAT "$GPU" afterany:$STD_CAT)
SVD=$(arr svd svd $NPAR_CAT "$GPU" afterany:$STD_CAT)
APPLY=$(one apply_T_lmx.sbatch applyT afterany:$STD_FAC:$RECAL)
APPLYREL=$(one apply_T_rel_lmx.sbatch applyTrel afterany:$STD_FAC:$RECAL)   # Stages D and E: only the release waits for it
WIDTHS=$(one strict_closure_lmx.sbatch widths afterok:$APPLY)
V4=$(one rescore_v4_lmx.sbatch rescorev4 afterok:$APPLY)
STAGEC=$(one stagec_lmx.sbatch stagec afterok:$APPLY)
W0=$(one wifi_cov_lmx.sbatch wifiS9 afterok:$APPLY TAG=MIXGEOV3S9)
W1=$(one wifi_cov_lmx.sbatch wifiS9s1 afterok:$APPLY TAG=MIXGEOV3S9s1)
W2=$(one wifi_cov_lmx.sbatch wifiS9s2 afterok:$APPLY TAG=MIXGEOV3S9s2)
W8=$(one wifi_cov_lmx.sbatch wifiM8 afterok:$APPLY TAG=MIXGEOV3M8S9)
CLOSMIX=$(one closure_mix_lmx.sbatch closmix afterok:$WIDTHS)
TOY=$(one toy_final_lmx.sbatch toy "" "" "$GPU")      # the toy model (E5), a job of its own
# the end of the scoring side: the K scan, the wide scan, and the check of every result (output/scoring_check.json)
COLLECT=$(one collect_lmx.sbatch collect afterany:$STD_FAC:$STD_CAT:$RECAL:$TRUNC:$STRICT_FAC:$STRICT_CAT:$SVD:$APPLY:$WIDTHS:$V4:$STAGEC:$W0:$W1:$W2:$W8:$CLOSMIX:$TOY)
echo "standard tests $STD_FAC (fac) $STD_CAT (cat); recalibration $RECAL; counts $TRUNC; strict $STRICT_FAC $STRICT_CAT; svd $SVD"
echo "apply T $APPLY; widths $WIDTHS; v4 $V4; Stage C $STAGEC; wifi $W0 $W1 $W2 $W8; closure $CLOSMIX; toy $TOY; collect $COLLECT"
echo "apply T of Stages D and E for the release: $APPLYREL (the release jobs wait for it, RERUN_SCORE.md section 2)"
echo "DCTR scorings, SvdRank and the toy: $GPU (NPAR_CAT=$NPAR_CAT); the rest: -p preq (NPAR=$NPAR)"
