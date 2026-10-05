#!/bin/bash
# Build the block-seed binary in a copy, then smoke-test seed indices 7, 50 and 150 at LO.
set -u
cd $SCRATCH/eerad3_thrust
rm -rf eerad3-1.0-blk && cp -r eerad3-1.0 eerad3-1.0-blk && cd eerad3-1.0-blk
python3 ../perl2/patch_eerad3.py src/eerad3.f || { echo PATCH_FAILED; exit 1; }
echo "--- patched region ---"; grep -n -A3 "inum.gt.699" src/eerad3.f; grep -n "3000\*iblk\|ichar('E')" src/eerad3.f
rm -f obj/*.o eerad3 eerad3_combine eerad3_dist
make > ../perl2/build.log 2>&1; echo "make exit $?"; tail -3 ../perl2/build.log; ls -la eerad3
echo "--- smoke ---"
rm -rf ../smoke && mkdir -p ../smoke && cd ../smoke
printf '%s\n' '1d-8 ! y0' '4 ! iaver' '1d-5 ! cutvar' '1 ! imom' '1 ! iang' '0 ! nloop' '0 ! icol' 'M ! itag' '1 1 ! iwarm iprod' '2 2 ! itmax' '20000 2000 2000 ! nshot' > smoke.input
for n in 7 50 150; do
  mkdir -p r$n && (cd r$n && ../../eerad3-1.0-blk/eerad3 -i ../smoke.input -n $n > run.log 2>&1; echo "n=$n exit $?")
  grep -h "seed block\|Output filenames\|\[LO\]" r$n/run.log
  echo "   files: $(ls r$n | grep -v run.log | head -4 | tr '\n' ' ') ... ($(ls r$n | wc -l) files)"
done
echo "--- independence: 50 vs 150 share the table row, must differ ---"
grep -h "\[LO\]" r50/run.log r150/run.log
echo "--- stock binary for reference, n=7 ---"
mkdir -p s7 && (cd s7 && ../../eerad3-1.0/eerad3 -i ../smoke.input -n 7 > run.log 2>&1); grep -h "\[LO\]" s7/run.log
echo SMOKE_DONE
