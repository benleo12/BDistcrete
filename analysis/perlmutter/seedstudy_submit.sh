#!/bin/bash
# Submit the full seed study and mixture-head control (perlmutter/seedstudy.sbatch), four tasks
# per GPU node. Seeds 10, 20, 30, 40 for both networks at every Table 4 stage, seed 0 of the
# factorized network retrained under identical inputs, the exact mixture head on the
# concatenation and late-fusion networks (seeds 0, 10, 20), and two more late-fusion seeds.
set -u
cd $SCRATCH/BDistcrete/analysis
JOBS=(
 "mixe:MIX:0 mixe:MIX:10 mixe:MIX:20 mixl:MIX:0"
 "mixl:MIX:10 mixl:MIX:20 late:MIX:10 late:MIX:20"
 "fact:MIX:0 fact:MIX:10 fact:MIX:20 fact:MIX:30"
 "fact:MIX:40 concat:MIX:10 concat:MIX:20 concat:MIX:30"
 "concat:MIX:40 fact:E:0 fact:E:10 fact:E:20"
 "fact:E:30 fact:E:40 concat:E:10 concat:E:20"
 "concat:E:30 concat:E:40 late:C:10 late:C:20"
 "fact:A:0 fact:A:10 fact:A:20 fact:A:30"
 "fact:A:40 fact:B:0 fact:B:10 fact:B:20"
 "fact:B:30 fact:B:40 fact:C:0 fact:C:10"
 "fact:C:20 fact:C:30 fact:C:40 concat:A:10"
 "concat:A:20 concat:A:30 concat:A:40 concat:B:10"
 "concat:B:20 concat:B:30 concat:B:40 concat:C:10"
 "concat:C:20 concat:C:30 concat:C:40"
)
for t in "${JOBS[@]}"; do
  sbatch --parsable -J seedstudy --export=ALL,TASKS="$t" perlmutter/seedstudy.sbatch | sed "s/\$/  $t/"
done
