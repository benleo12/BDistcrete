#!/bin/bash
# The remaining alpha_0 columns of the MIX17 direct profile, four at a time, nearest first.
cd .
printf "%s\n" 0.41 0.45 0.39 0.47 0.37 0.49 0.35 0.51 0.33 0.53 0.31 0.55 0.29 0.57 0.27 0.59 0.25 0.61 0.63 0.65 | xargs -P 4 -n 1 ./run_one_column.sh
echo "ALL COLUMNS DONE $(date)"
