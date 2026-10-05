#!/bin/bash
# Set up the two 1M campaigns on Perlmutter. Run AFTER sw_bootstrap.sh finishes.
#   $SCRATCH/anchor1M/meps : MEPS@NLO (alaric build, commit c74190c5a), 50 x 20k,
#                            7-point mu_R/mu_F variations, RS_Enhance as in the 20k run
#   $SCRATCH/anchor1M/ref  : Stage C reference (v3.0.4), 27 grid + 5 held-out, 40k each
# This script writes all cards + sbatch files and launches the MEPS integration
# (nohup, login node). Submit the arrays with:  bash ~/anchor1M_submit.sh
set -uo pipefail
ACCT=${NERSC_ACCOUNT:?set to your NERSC project}
BASE=$SCRATCH/anchor1M
mkdir -p $BASE/meps/integrate $BASE/ref $BASE/logs

# ---------------------------------------------------------------- MEPS card
meps_card () {  # $1 events  $2 seed  $3 outname
cat <<EOF
# 1M campaign card: identical to css_test/Run_nom.yaml except EVENTS, RANDOM_SEED,
# EVENT_OUTPUT, and SCALE_VARIATIONS extended from the correlated pair to the
# full 7-point mu_R/mu_F set (squared-factor convention).
TAGS:
  YCUT: 2.0

EVENTS: $1
RANDOM_SEED: $2
EVENT_OUTPUT: HepMC3_GenEvent[$3]

SHOWER_GENERATOR: CSS
NLOMC_GENERATOR: CSS
FRAGMENTATION: Ahadic
PDF_SET: None

ALPHAS(MZ): 0.1188
ORDER_ALPHAS: 2

BEAMS: [11, -11]
BEAM_ENERGIES: 45.6

ME_GENERATORS: [Comix, Amegic, Internal]

PROCESSES:
- 11 -11 -> 93 93 93{2}:
    CKKW: pow(10,-\$(YCUT)/2.00)*E_CMS
    Order: {QCD: 0, EW: 2}
    RS_Enhance_Factor: 10
    2->2:
      NLO_Mode: MC@NLO
      NLO_Order: {QCD: 1, EW: 0}
      Loop_Generator: Internal
      ME_Generator: Amegic
      RS_ME_Generator: Comix
SCALE_VARIATIONS: [[0.25,0.25],[4.0,4.0],[0.25,1.0],[4.0,1.0],[1.0,0.25],[1.0,4.0]]
EOF
}
meps_card 0 8000 events_int > $BASE/meps/integrate/Sherpa.yaml

# integration + makelibs runner (Amegic writes its libraries on first pass)
cat > $BASE/meps/run_integration.sh <<'EOS'
#!/bin/bash
set -u
module load gcc-native/13 2>/dev/null || module load gcc-native/12
cd $SCRATCH/anchor1M/meps/integrate
S=$HOME/sw/sherpa-alaric/bin/Sherpa
$S > int1.log 2>&1
if [ -x ./makelibs ]; then ./makelibs -j 16 > makelibs.log 2>&1; $S > int2.log 2>&1; fi
echo "INTEGRATION EXIT $? (check Results.zip)"; ls -la Results.zip 2>/dev/null
EOS
chmod +x $BASE/meps/run_integration.sh

# MEPS event-generation array: 50 tasks x 20k = 1M
for i in $(seq 1 50); do
  d=$BASE/meps/task_$(printf '%02d' $i); mkdir -p $d
  meps_card 20000 $((8000+i)) events_$(printf '%02d' $i) > $d/Sherpa.yaml
done
cat > $BASE/meps/meps_array.sbatch <<EOS
#!/bin/bash
#SBATCH -A $ACCT
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --array=1-50
#SBATCH -o logs/meps_%a.out
module load gcc-native/13 2>/dev/null || module load gcc-native/12
d=$BASE/meps/task_\$(printf '%02d' \$SLURM_ARRAY_TASK_ID)
cd \$d
ln -sf ../integrate/Process Process
cp -n ../integrate/Results.zip . 2>/dev/null
\$HOME/sw/sherpa-alaric/bin/Sherpa > run.log 2>&1
rc=\$?
gzip -f events_* 2>/dev/null
echo "TASK \$SLURM_ARRAY_TASK_ID EXIT \$rc"
EOS

# ---------------------------------------------------------------- reference
ref_card () {  # $1 alpha_s  $2 strange  $3 kt0  $4 events  $5 seed
cat <<EOF
# Stage C reference regeneration: identical to gen_stageC.sh card (Sherpa v3.0.4)
# except EVENTS and RANDOM_SEED.
EVENTS: $4
RANDOM_SEED: $5
EVENT_OUTPUT: HepMC3_GenEvent[events]

BEAMS: [11, -11]
BEAM_ENERGIES: 45.6

ALPHAS(MZ): $1
ORDER_ALPHAS: 1

SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
AHADIC:
  STRANGE_FRACTION: $2
  KT_0: $3

PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
EOF
}
idx=0
for a in 0.112 0.120 0.128; do for s in 0.30 0.46 0.65; do for k in 0.80 1.21 1.80; do
  d=$BASE/ref/run_$(printf '%02d' $idx)_a${a}_s${s}_k${k}; mkdir -p $d
  ref_card $a $s $k 40000 $((7000+idx)) > $d/Sherpa.yaml
  idx=$((idx+1))
done; done; done
# held-out interior points (same as gen_stageC.sh 6900-6904)
for p in "0.116 0.38 1.00" "0.124 0.55 1.50" "0.120 0.46 1.21" "0.114 0.60 0.95" "0.126 0.35 1.60"; do
  set -- $p
  d=$BASE/ref/run_$(printf '%02d' $idx)_a$1_s$2_k$3; mkdir -p $d
  ref_card $1 $2 $3 40000 $((7000+idx)) > $d/Sherpa.yaml
  idx=$((idx+1))
done
ls -d $BASE/ref/run_* > $BASE/ref/runlist.txt
n=$(wc -l < $BASE/ref/runlist.txt)
cat > $BASE/ref/ref_array.sbatch <<EOS
#!/bin/bash
#SBATCH -A $ACCT
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --array=1-$n
#SBATCH -o logs/ref_%a.out
module load gcc-native/13 2>/dev/null || module load gcc-native/12
d=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $BASE/ref/runlist.txt)
cd \$d
\$HOME/sw/sherpa-3.0.4/bin/Sherpa > run.log 2>&1
rc=\$?
for c in events.hepmc3 events events.hepmc; do [ -f \$c ] && mv \$c events.hepmc && break; done
gzip -f events.hepmc 2>/dev/null
echo "TASK \$SLURM_ARRAY_TASK_ID \$d EXIT \$rc"
EOS

# ---------------------------------------------------------------- submit helper
cat > ~/anchor1M_submit.sh <<EOS
#!/bin/bash
sbatch $BASE/ref/ref_array.sbatch
sbatch $BASE/meps/meps_array.sbatch
squeue -u \$USER
EOS
chmod +x ~/anchor1M_submit.sh
echo "CAMPAIGN LAID OUT under $BASE"
echo "launching MEPS integration now (nohup on login node)"
nohup $BASE/meps/run_integration.sh > $BASE/logs/integration.log 2>&1 &
echo "integration pid $!"
