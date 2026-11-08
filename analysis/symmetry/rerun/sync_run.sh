#!/bin/bash
# Copies the scripts and inputs of the rerun from the working tree (analysis/ of ~/BDistcrete) to RUN on Symmetry
# (I3): the union of symmetry/rerun/files_score.txt (the scoring side) and symmetry/files_prop.txt (the propagation
# side), paths relative to analysis/, globs allowed, one per line, # comments. On the laptop, with the VPN up:
#   bash ~/BDistcrete/analysis/symmetry/rerun/sync_run.sh            # copy, then compare the md5 on both sides
#   bash ~/BDistcrete/analysis/symmetry/rerun/sync_run.sh --check    # compare only
# It packs with COPYFILE_DISABLE=1 tar (no macOS metadata files), leaves out __pycache__, *.pyc and .DS_Store, and
# stops before copying anything when a pattern matches no file, when a listed file lies in RUN/output where a job of
# the rerun writes (lmx_fingerprint.py outputs), or when a file it would write is a link in RUN or lies in a folder
# that resolves outside RUN (nothing is written through a link). Afterwards it compares the md5 of every file on both
# sides and exits 1 listing each mismatch; after a copy whose md5 match it writes them to
# RUN/symmetry/rerun/files_synced.md5, which setup_lmx.py checks (every file there, with that md5). While the start
# mark output/.lmx_rerun_start exists it copies only with RESUME=1 (new scoring code changes the fingerprints, so a
# resubmission scores those results again), and never a design (*.csv of analysis/) that differs from RUN's: the
# fingerprints do not record the designs, so a changed one needs the mark removed (start over).
set -euo pipefail
A=$(cd "$(dirname "$0")/../.." && pwd)                 # analysis/ of the working tree
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
remote() { ssh -o BatchMode=yes -o ConnectTimeout=8 symmetry-direct "$1"; }
PY=${PY:-python3}
LISTS="symmetry/rerun/files_score.txt symmetry/files_prop.txt"
cd "$A"
for f in $LISTS; do [ -f "$f" ] || { echo "missing $A/$f (one of the two lists)"; exit 2; }; done
# the files, expanded here; stops on a pattern without a file and on a file a job of the rerun writes in RUN/output
FILES=$(PYTHONDONTWRITEBYTECODE=1 "$PY" -B - $LISTS <<'PY'
import fnmatch, importlib.util, sys
s = importlib.util.spec_from_file_location('lmx_fingerprint', 'symmetry/rerun/lmx_fingerprint.py')
FP = importlib.util.module_from_spec(s); s.loader.exec_module(FP)
try:
    files = sorted({f for lst in sys.argv[1:] for f in FP.expand(lst)})
except FP.Stop as e:
    sys.exit(f'{e}: nothing copied')
outs = [p for p, _ in FP.outputs()]
bad = [f for f in files if f.startswith('output/') and any(fnmatch.fnmatch(f, p) for p in outs)]
if bad:
    sys.exit('a job of the rerun writes these in RUN/output, so they are not copied (take them out of the lists): '
             + ' '.join(bad))
print('\n'.join(files))
PY
) || exit 2
N=$(printf '%s\n' "$FILES" | grep -c .)
# the md5 of every file here
HERE_MD5=$(printf '%s\n' "$FILES" | PYTHONDONTWRITEBYTECODE=1 "$PY" -B -c '
import hashlib, sys
for f in sys.stdin.read().split():
    print(hashlib.md5(open(f, "rb").read()).hexdigest() + "  " + f)')
remote true || { echo "Symmetry does not answer (is the VPN connected?): nothing copied or compared"; exit 2; }
if [ "${1:-}" != --check ]; then
  MARKED=$(remote "test -e $RUN/output/.lmx_rerun_start && echo yes || echo no")
  if [ "$MARKED" = yes ] && [ "${RESUME:-}" != 1 ]; then
    echo "$RUN/output/.lmx_rerun_start exists: the rerun was submitted. RESUME=1 copies anyway (then resubmit with"
    echo "RESUME=1: a scoring whose code changed is done again); --check only compares. Nothing copied."
    exit 2
  fi
  CSV=$(printf '%s\n' "$HERE_MD5" | awk '$2 !~ /\// && $2 ~ /\.csv$/')
  if [ "$MARKED" = yes ] && [ -n "$CSV" ]; then    # RESUME=1: the code may change under the mark, the designs may not
    THERE_CSV=$(printf '%s\n' "$CSV" | awk '{print $2}' | remote "cd $RUN && while IFS= read -r f; do
        if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo \"missing  \$f\"; fi; done")
    if [ "$CSV" != "$THERE_CSV" ]; then
      echo "a design (*.csv) differs from RUN's while the rerun runs; the scorings' fingerprints do not record the"
      echo "designs, so a changed one needs the mark removed (start over). Nothing copied. Differing (RUN's side):"
      diff <(printf '%s\n' "$CSV") <(printf '%s\n' "$THERE_CSV") | grep '^>' | sed 's/^> /  /' || true
      exit 2
    fi
  fi
  # nothing is written through a link: no file to copy may be a link in RUN or lie below a folder outside RUN
  LINKS=$(printf '%s\n' "$FILES" | remote "cd $RUN 2>/dev/null || exit 0; while IFS= read -r f; do
      if [ -L \"\$f\" ]; then echo \"\$f is a link to \$(readlink -f \"\$f\")\"; fi
      d=\$(dirname \"\$f\"); while [ ! -e \"\$d\" ]; do d=\$(dirname \"\$d\"); done
      case \"\$(readlink -f \"\$d\")\" in $RUN|$RUN/*) ;; *) echo \"\$f lies in \$d, which resolves to \$(readlink -f \"\$d\")\" ;; esac
    done")
  if [ -n "$LINKS" ]; then echo "$LINKS"; echo "nothing copied: remove or move these links in RUN first"; exit 2; fi
  printf '%s\n' "$FILES" | COPYFILE_DISABLE=1 tar -c -h -f - --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store -T - \
    | remote "mkdir -p $RUN && cd $RUN && tar -x -f -"
  echo "copied $N files to $RUN"
fi
# the md5 of every file in RUN
THERE_MD5=$(printf '%s\n' "$FILES" | remote "cd $RUN 2>/dev/null; while IFS= read -r f; do
    if [ -f \"\$f\" ] && [ \"\$(pwd)\" = $RUN ]; then md5sum \"\$f\"; else echo \"missing  \$f\"; fi; done")
BAD=$(diff <(printf '%s\n' "$HERE_MD5") <(printf '%s\n' "$THERE_MD5") | grep '^>' | sed 's/^> //' || true)
if [ -n "$BAD" ] || [ "$(printf '%s\n' "$THERE_MD5" | grep -c .)" != "$N" ]; then
  echo "md5 differs between the working tree and RUN (or the file is missing in RUN):"
  printf '%s\n' "$BAD" | awk '{print "  " $2 " (" $1 " in RUN)"}'
  exit 1
fi
if [ "${1:-}" != --check ]; then     # the record of this copy, which setup_lmx.py checks
  printf '%s\n' "$HERE_MD5" | remote "cd $RUN/symmetry/rerun && cat > .files_synced.md5.tmp\$\$ && mv .files_synced.md5.tmp\$\$ files_synced.md5"
fi
echo "md5 equal on both sides for all $N files"
