#!/usr/bin/env bash
# Download data archives of docs/DATA.md, check their SHA256 and unpack them into analysis/.
#
#   tools/get_data.sh sec6                     # the archives one paper item needs (groups below)
#   tools/get_data.sh B1_exports_ABC B2_held_ABC
#   tools/get_data.sh --list                   # archives, sizes and groups
#   tools/get_data.sh --unlink                 # remove the reference links (before retraining)
#
# Options: --base URL (default $BDISTCRETE_DATA_URL or the GitHub release data-v1), --keep (keep
# the downloaded archives in analysis/data_downloads/), --force (unpack again an archive already
# unpacked). Each archive is checked against SHA256SUMS of the release before it is unpacked, and
# every file it holds against analysis/data_checksums/<name>.sha256 after.
set -euo pipefail

BASE=${BDISTCRETE_DATA_URL:-https://github.com/benleo12/BDistcrete/releases/download/data-v1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/analysis
DL=$A/data_downloads          # matches the data_* rule of .gitignore

# name, approximate size, what it is for (docs/DATA.md has the full table)
ARCHIVES="
A0_theory_logs.tar        0.04  raw EERAD3 logs, to rebuild the fixed-order cumulant (App. D)
B1_exports_ABC.tar        0.77  Stage A, B, C networks and references (Table 1, Secs. 5.2, 5.3, App. B)
B2_held_ABC.tar           1.03  Stage A, B, C held-out runs (Table 1, rates, Fig. 5, App. B)
B3_held_C_valid.tar       0.74  Stage C validation runs 6950-6961 (Fig. 6, tails)
B4_held_C_floor.tar       1.23  Stage C runs 6970-6971 (offsets at 800k, accuracy law)
B5_mix7.tar               0.81  seven-parameter mixture networks, reference, held runs (Sec. 5.6, Table 2)
B6_sec6_MIX17.tar         0.94  seventeen-parameter mixture network and reference (Sec. 6, App. E)
B7_C1M.tar                0.51  three-parameter C_1M network and reference (Sec. 6 family, App. A)
B8_E.tar                  0.88  Sherpa eight-parameter network and reference (Table 3)
B9_F.tar                  1.10  Herwig eight-parameter network and reference (Table 3)
B10_held_8param.tar       1.51  held runs of E, F and the mixture, with designs (Table 3, Sec. 5.7)
B11_bench.tar             0.67  mixture reference with particles, concatenation weights (Sec. 5.8)
B12_appA_v2.tar           1.12  exact-axis references and held runs of A, B, C (App. A)
B13_C1M_particles.tar.gz  0.63  C_1M reference with its particles (only for package_release.py)
C1_train_A.tar            1.02  Stage A training runs 6100-6114
C2_train_B.tar            1.72  Stage B training runs 6600-6627
C3_train_C.tar            1.66  Stage C training runs 6800-6826
C4_train_C1M.tar          0.83  Stage C runs 7800-7826 (the C_1M network)
C5_widebox.tar            1.67  wide coupling range runs 9000-9020, 9100-9105 (Sec. 5.5)
C6_train_DM2.tar          1.83  seven-parameter mixture training runs 9700-9795
C7a_train_E.tar           1.48  Sherpa eight-parameter training runs 7000-7023
C7b_train_E.tar           1.48  runs 7024-7047
C7c_train_E.tar           1.47  runs 7048-7071
C7d_train_E.tar           1.48  runs 7072-7095
C8a_train_F.tar           1.27  Herwig eight-parameter training runs 8000-8020
C8b_train_F.tar           1.24  runs 8021-8041
C8c_train_F.tar           1.29  runs 8042-8063
C8d_train_F.tar           1.54  runs 8100-8123
C8e_train_F.tar           1.48  runs 8124-8147
C9a_train_MIX17.tar       1.81  seventeen-parameter mixture training runs 9700-9795
C9b_train_MIX17.tar       1.81  runs 9796-9891
"

group() {
  case $1 in
    sec6)    echo B6_sec6_MIX17 ;;
    family)  echo B7_C1M ;;
    table1)  echo B1_exports_ABC B2_held_ABC ;;
    sec5)    echo B1_exports_ABC B2_held_ABC B3_held_C_valid B4_held_C_floor ;;
    table2)  echo B5_mix7 ;;
    table3)  echo B6_sec6_MIX17 B8_E B9_F B10_held_8param ;;
    cost)    echo B6_sec6_MIX17 B11_bench ;;
    appA)    echo B1_exports_ABC B7_C1M B12_appA_v2 ;;
    appD)    echo A0_theory_logs ;;
    core)    echo A0_theory_logs B1_exports_ABC B2_held_ABC B3_held_C_valid B4_held_C_floor B5_mix7 B6_sec6_MIX17 \
                  B7_C1M B8_E B9_F B10_held_8param ;;
    train-A) echo C1_train_A B2_held_ABC ;;
    train-B) echo C2_train_B B2_held_ABC ;;
    train-C) echo C3_train_C B2_held_ABC ;;
    train-C1M) echo C3_train_C C4_train_C1M B2_held_ABC ;;
    widebox) echo C5_widebox ;;
    train-DM2) echo C6_train_DM2 B5_mix7 ;;
    train-E) echo C7a_train_E C7b_train_E C7c_train_E C7d_train_E B10_held_8param ;;
    train-F) echo C8a_train_F C8b_train_F C8c_train_F C8d_train_F C8e_train_F B10_held_8param ;;
    train-MIX17) echo C9a_train_MIX17 C9b_train_MIX17 B10_held_8param ;;
    *) return 1 ;;
  esac
}
GROUPS_HELP="sec6 family table1 sec5 table2 table3 cost appA appD core train-A train-B train-C train-C1M"
GROUPS_HELP="$GROUPS_HELP widebox train-DM2 train-E train-F train-MIX17"

# Names the scripts read, pointed at the file that serves them. The slim references hold every
# observable the scripts read but no particle arrays. A link is made only where no real file
# exists, so an archive with the full file (B11, B13) takes precedence.
LINKS="
C_1M_ref_v2.npz C_1M_ref_v2_slim.npz
C_1M_ref.npz C_1M_ref_v2.npz
E_ref.npz E_ref_v2_slim.npz
Fauglong_ref.npz Fauglong_ref_v2_slim.npz
MIX17aug_ref.npz MIX17aug_ref_v2_slim.npz
DMDMXSc10_ref.npz DMDMXSc0_ref.npz
A_ref_v2.npz A_ref_v2_slim.npz
B_ref_v2.npz B_ref_v2_slim.npz
C_ref_v2.npz C_ref_v2_slim.npz
"

die() { echo "get_data.sh: $*" >&2; exit 1; }

sha256() {
  if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}

fetch() {   # fetch FILE: download BASE/FILE into DL, resuming a partial download
  if command -v curl > /dev/null; then
    curl -fL --retry 3 -C - -o "$DL/$1" "$BASE/$1"
  else
    wget -c -O "$DL/$1" "$BASE/$1"
  fi
}

file_of() {   # the archive file of a name, from the table above
  local f
  f=$(printf '%s\n' "$ARCHIVES" | awk -v n="$1" '$1 == n".tar" || $1 == n".tar.gz" {print $1}')
  [ -n "$f" ] || die "unknown archive or group '$1' (see --list)"
  echo "$f"
}

make_links() {
  local M=$A/output/models
  [ -d "$M" ] || return 0
  printf '%s\n' "$LINKS" | while read -r link target; do
    [ -n "$link" ] || continue
    if [ -e "$M/$target" ] && { [ ! -e "$M/$link" ] || [ -L "$M/$link" ]; }; then
      ln -sfn "$target" "$M/$link"
    fi
  done
}

unlink_all() {
  local M=$A/output/models
  printf '%s\n' "$LINKS" | while read -r link target; do
    [ -n "$link" ] || continue
    if [ -L "$M/$link" ]; then rm "$M/$link"; echo "removed output/models/$link"; fi
  done
}

get_one() {   # get_one NAME
  local name=$1 f want got out
  f=$(file_of "$name")
  if [ $FORCE = 0 ] && [ -f "$A/data_checksums/$name.sha256" ]; then
    echo "== $name: already unpacked (--force to unpack again)"; return 0
  fi
  echo "== $name"
  want=$(awk -v f="$f" '$2 == f {print $1}' "$DL/SHA256SUMS")
  [ -n "$want" ] || die "$f is not in SHA256SUMS of $BASE"
  if [ ! -f "$DL/$f" ] || [ "$(sha256 "$DL/$f" | cut -d' ' -f1)" != "$want" ]; then
    fetch "$f" || true
  fi
  got=$(sha256 "$DL/$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || die "$f: SHA256 $got does not match the release ($want). Delete $DL/$f and retry."
  # an archive that brings a real file replaces the link of the same name
  tar -tf "$DL/$f" | while read -r m; do if [ -L "$A/$m" ]; then rm "$A/$m"; fi; done
  tar -xf "$DL/$f" -C "$A"
  if ! out=$(cd "$A" && sha256 -c "data_checksums/$name.sha256" 2>&1); then
    printf '%s\n' "$out" | grep -v ': OK$' >&2
    die "$name: files differ from data_checksums/$name.sha256 after unpacking"
  fi
  echo "   $(wc -l < "$A/data_checksums/$name.sha256" | tr -d ' ') files checked"
  [ $KEEP = 1 ] || rm -f "$DL/$f"
}

KEEP=0; FORCE=0; NAMES=()
while [ $# -gt 0 ]; do
  case $1 in
    --base) BASE=${2%/}; shift ;;
    --keep) KEEP=1 ;;
    --force) FORCE=1 ;;
    --list)
      printf '%s\n' "$ARCHIVES" | awk 'NF {printf "  %-26s %5s GB  ", $1, $2; $1 = $2 = ""; sub(/^ +/, ""); print}'
      echo "groups: $GROUPS_HELP"; exit 0 ;;
    --unlink) unlink_all; exit 0 ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) NAMES+=("$1") ;;
  esac
  shift
done
[ ${#NAMES[@]} -gt 0 ] || { awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 1; }

SEL=()
for n in "${NAMES[@]}"; do
  if g=$(group "$n"); then
    for x in $g; do SEL+=("$x"); done
  else
    file_of "$n" > /dev/null; SEL+=("$n")
  fi
done

mkdir -p "$DL"
rm -f "$DL/SHA256SUMS"; fetch SHA256SUMS
for n in $(printf '%s\n' "${SEL[@]}" | awk '!seen[$0]++'); do get_one "$n"; done
make_links
echo "done. Data are in $A (see docs/DATA.md for what each archive unlocks)."
