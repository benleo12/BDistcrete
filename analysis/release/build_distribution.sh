#!/bin/bash
# Build the archive that actually goes out, and refuse if anything internal reached it.
#
# The working bundle is not the distributable one. It carries TODO_BEFORE_PUBLISHING.md, which is
# an internal checklist that names open problems and would be read as release documentation, plus
# editor backups from whatever was last changed. Relying on someone remembering to delete those
# is how they ship. This enforces .distignore instead, and then runs the self-test against the
# staged copy so what goes out is what was tested.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE=${1:-/tmp/gentune_dist}
NAME=gentune-$(cat "$HERE/VERSION")

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$HERE/." "$STAGE/$NAME"

cd "$STAGE/$NAME"
while IFS= read -r pat; do
  case "$pat" in ''|\#*) continue;; esac
  find . -name "$pat" -exec rm -rf {} + 2>/dev/null || true
done < "$HERE/.distignore"

# Enforce, do not assume. A pattern that survived means the removal above did not match it.
bad=0
while IFS= read -r pat; do
  case "$pat" in ''|\#*) continue;; esac
  found=$(find . -name "$pat" 2>/dev/null | head -5)
  [ -z "$found" ] || { echo "REFUSING: $pat is still present:"; echo "$found"; bad=1; }
done < "$HERE/.distignore"
[ "$bad" -eq 0 ] || exit 2

# Anything that names an open problem should not be in a release either.
leaked=$(grep -rl "TODO_BEFORE_PUBLISHING" . 2>/dev/null | head -3 || true)
[ -z "$leaked" ] || { echo "REFUSING: these files still point at the internal checklist:"; echo "$leaked"; exit 2; }

# What goes out has to pass its own test in the form it goes out in.
python -m gentune.selftest models || { echo "REFUSING: the staged copy does not pass its self-test"; exit 3; }

# Running the test just created __pycache__ inside the staged copy, so clean again AFTER it
# rather than before. Cleaning only before is how a pycache directory shipped.
while IFS= read -r pat; do
  case "$pat" in ''|\#*) continue;; esac
  find . -name "$pat" -exec rm -rf {} + 2>/dev/null || true
done < "$HERE/.distignore"
post=$(find . \( -name '__pycache__' -o -name '*.bak' -o -name '*.bak_*' \) | head -5)
[ -z "$post" ] || { echo "REFUSING: still present after the second clean:"; echo "$post"; exit 2; }

cd "$STAGE"
tar czf "$NAME.tar.gz" "$NAME"
echo
echo "built $STAGE/$NAME.tar.gz  ($(du -h "$NAME.tar.gz" | cut -f1))"
echo "contents, top level:"
ls "$NAME" | sed 's/^/  /'
