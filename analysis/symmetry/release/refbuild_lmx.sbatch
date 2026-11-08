#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J lmx_refbuild
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# Builds the paper's deposit again from the rerun, in lmx_runs/deposit_lin: reference_MIX17.npz, build_reference.json
# (the record of the build) and SHA256SUMS. SHA256SUMS lists those two files and the published targets_thrust_nlo.npz
# with bare names: it is the check a user runs after joining the parts in the repository's deposit/. The build script is
# analysis/release/tools_build_reference.py, sha256 90e8df8d7ffcecbd8b1a25fcd221aab32cef9cfca556fa059862420889743107
# (BUILD_SHA below). The job copies it into the deposit folder and runs it there as build_reference.py, with
# BEAM_PHOTONS=remove.
# reference_MIX17.npz holds the event vectors a(Phi) of the mixture network's export (output/models/MIXGEOV3S9_cond.npz)
# bitwise, and the per-event observables without the photons radiated by the beams, from the bundle remade without
# them (output/models/MIXGEOV3S9_ref_v2_slim.npz, listed in output/bundle_rule.json). Its meta member gives the sha256
# of every array. Before anything is written, the job and the build check that the export carries its refitted
# temperature and that the packaged MIX17 head is the export bitwise; the order of the events and the logits; and the
# paper's central predictions of the linear mixture against output/three_bands_MIXGEOV3S9_lin.{json,npz}, which must
# hold the paper's bands (2000 points of the seed-11 Latin hypercube over the full box, the default tunes at the
# fraction 1/2, DELPHI as the data). The calculation did not change, so targets_thrust_nlo.npz stays as published: the
# build checks that it holds output/thrust_targets_nlo_point.npz bitwise. Then every array of the new file is compared
# with the copy of the same events remade in lmx_runs/release_ref, and REFBUILD_OK.json records every output with its
# sha256.
# It runs after the end of the scoring side (lmx_collect), the three bands (lmx_3b_MIXGEOV3S9) and the packaged
# networks (lmx_package), from the rerun's folder lmx_runs/maxent_code:
#   sbatch --export=ALL,MAXENT_DIR=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code symmetry/release/refbuild_lmx.sbatch
# It ends with REFBUILD DONE, or REFBUILD FAILED and no REFBUILD_OK.json.
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
D=/mnt/beegfs/bassi/reweighting/lmx_runs/deposit_lin    # the new deposit
REFRESHED=/mnt/beegfs/bassi/reweighting/lmx_runs/release_ref/reference_MIX17.npz   # the copy refreshed under rule v3, read only
PUBM=/mnt/beegfs/bassi/reweighting/maxent_code/output/models                       # the published bundles, read only
TARGETS_SHA=7e125ae4e03da92255f0f2ae92b91553a86cd016b334eaf11b01ea3873a835ca       # the published targets_thrust_nlo.npz
RULE_MD5=ea7f82ed77c067ee727114a72d038d93                                          # beam_photons.py, rule v3
BUILD_SHA=90e8df8d7ffcecbd8b1a25fcd221aab32cef9cfca556fa059862420889743107         # analysis/release/tools_build_reference.py
UPLOAD="reference_MIX17.npz targets_thrust_nlo.npz build_reference.json"           # the files SHA256SUMS lists, in order
echo "=== lmx_refbuild on $(hostname) $(date), job ${SLURM_JOB_ID:-none}"
need_nb data_stagePURE17_v3 || exit 2
# the record of the release phase of this rerun, and the start mark
"$PY" -B symmetry/rerun/lmx_fingerprint.py has-record applyTrel || { echo "apply_T_rel_lmx.sbatch has not ended well in this rerun"; exit 2; }
"$PY" -B symmetry/rerun/lmx_fingerprint.py status || exit 2
# the scoring side has ended well in this rerun: the record of lmx_collect, and output/scoring_check.json of this start
# mark, which lmx_collect writes last
"$PY" -B symmetry/rerun/lmx_fingerprint.py has-record collect || { echo "lmx_collect has not ended well in this rerun"; exit 2; }
"$PY" -B - <<'PY' || { echo "output/scoring_check.json is missing or not of this rerun: lmx_collect has not ended well"; exit 2; }
import json
c, m = json.load(open('output/scoring_check.json')), json.load(open('output/.lmx_rerun_start'))
assert c.get('mark_run_id') == m['run_id'] and c.get('files'), (c.get('mark_run_id'), m.get('run_id'))
print(f'the scoring side ended well in run {m["run_id"]}: output/scoring_check.json lists {len(c["files"])} checked files')
PY
need_file output/models/MIXGEOV3S9_cond.npz output/models/MIXGEOV3S9_ref.npz output/models/MIXGEOV3S9_ref_v2_slim.npz \
  $PUBM/MIXGEOV3S9_ref_v2_slim.npz output/temperature/MIXGEOV3S9.json output/thrust_targets_nlo_point.npz \
  output/three_bands_MIXGEOV3S9_lin.json output/three_bands_MIXGEOV3S9_lin.npz data_stagePURE17_v3/meta.json \
  release_lmx/gentune/__init__.py release_lmx/models/MIX17_head.npz release_lmx/models/MIX17_trunk.npz \
  release_lmx/models/MIX17_selftest.npz release_lmx/models/MANIFEST.json release/tools_build_reference.py \
  $REFRESHED $D/targets_thrust_nlo.npz || exit 2
# the build script is the release's tools_build_reference.py, whose sha256 the deposit's meta will carry
[ "$(sha256sum < release/tools_build_reference.py | cut -c1-64)" = "$BUILD_SHA" ] \
  || { echo "release/tools_build_reference.py is not the build script of the release (sha256 $BUILD_SHA): copy it again"; exit 2; }
# the outputs of this rerun that the build reads were written after its start mark (earlier copies were moved aside
# when the rerun started)
for f in output/three_bands_MIXGEOV3S9_lin.json output/three_bands_MIXGEOV3S9_lin.npz output/temperature/MIXGEOV3S9.json \
         release_lmx/models/MIX17_head.npz release_lmx/models/MIX17_trunk.npz release_lmx/models/MIX17_selftest.npz \
         release_lmx/models/MANIFEST.json; do
  [ "$f" -nt output/.lmx_rerun_start ] || { echo "$f is not newer than output/.lmx_rerun_start: not an output of this rerun"; exit 2; }
done
# beam-photon rule v3 in RUN, and every bundle read listed with it in output/bundle_rule.json
grep -q '^RULE_VERSION = 3$' beam_photons.py || { echo "RUN's beam_photons.py is not rule v3"; exit 2; }
[ "$(md5sum < beam_photons.py | cut -c1-32)" = "$RULE_MD5" ] || { echo "RUN's beam_photons.py is not the file of rule v3 (md5 $RULE_MD5)"; exit 2; }
need_rule output/models/MIXGEOV3S9_ref_v2_slim.npz output/models/MIXGEOV3S9_ref.npz $REFRESHED || exit 2
# the export carries its refitted temperature, and the packaged head and its manifest carry the export's temperature
"$PY" -B - <<'PY' || { echo "EXPORT TEMPERATURE CHECK FAILED"; exit 3; }
import json, numpy as np, score_obs as SO
e = 'output/models/MIXGEOV3S9_cond.npz'
T = SO.check_export_T(e, float(np.load(e)['temperature']))
h = float(np.load('release_lmx/models/MIX17_head.npz')['temperature'])
m = float(json.load(open('release_lmx/models/MANIFEST.json'))['MIX17']['temperature'])
assert abs(h - T) < 1e-12 and abs(m - T) < 1e-12, (T, h, m)
print(f'MIXGEOV3S9: the export carries T = {T}, its refit; the packaged MIX17 head and MANIFEST.json the same')
PY
# the bands are the paper's, not a test: the linear mixture, 2000 points of the seed-11 Latin hypercube over the full
# box, the default tunes of both generators at xi = 1/2 (the CENTRAL of three_bands.sbatch), DELPHI
"$PY" -B - <<'PY' || { echo "the three bands of MIXGEOV3S9 are not the paper's"; exit 2; }
import json
import numpy as np
CENTRAL = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18,
           0.102337, 0.654714, 3.528693, 1.849375, 0.914156, 0.374094, 0.33107, 0.78, 0.5]
b = json.load(open('output/three_bands_MIXGEOV3S9_lin.json'))
assert b.get('mix') == 'linear' and b.get('box') == 'full' and b.get('nodes') == 2000 and 'hypercube_size' not in b, \
    (b.get('mix'), b.get('box'), b.get('nodes'), b.get('hypercube_size'))
assert b['central']['theta'] == CENTRAL and b['export'].endswith('MIXGEOV3S9_cond.npz'), (b['central']['theta'], b['export'])
assert b.get('test') == 'delphi', f'compared with {b.get("test")}, not DELPHI'
# the points: three_bands.py's Latin hypercube with SEED=11 over the box of the export (gentune.bands.lhs_nodes)
norm = np.load('output/models/MIXGEOV3S9_cond.npz')['norm'].astype(np.float64)
rng = np.random.default_rng(11)
U = (np.argsort(rng.random((2000, len(norm))), axis=0) + rng.random((2000, len(norm))))/2000*2 - 1
x = np.load('output/three_bands_MIXGEOV3S9_lin.npz')['nodes']
assert x.shape == U.shape, x.shape
d = float(np.max(np.abs(x - (norm[:, 0] + norm[:, 1]*U))/norm[:, 1]))
assert d < 1e-12, f'the points are not the Latin hypercube of seed 11 (they differ by {d:.2e} of a half width)'
print(f'three bands: {b["mix"]}, {b["nodes"]} points of seed 11 over the {b["box"]} box, the default tunes at xi = 1/2, {b["test"]}')
PY
# the deposit folder: lmx_runs/deposit_lin itself, no link into release_ref or the published maxent_code
[ -d "$D" ] && [ "$(realpath "$D")" = "$D" ] || { echo "$D must be a folder of its own, made before the first submission"; exit 2; }
# no link inside it: a write to a path that is a link would land where the link points
[ -z "$(find "$D" -mindepth 1 -type l -print -quit)" ] || { echo "$D holds a link: $(find "$D" -mindepth 1 -type l | head -3 | tr '\n' ' ')"; exit 2; }
[ "$(sha256sum < $D/targets_thrust_nlo.npz | cut -c1-64)" = "$TARGETS_SHA" ] || { echo "$D/targets_thrust_nlo.npz is not the published file"; exit 2; }
# a failed earlier attempt leaves no output behind; the build script is copied again, the targets file stays
rm -f $D/build_reference.py $D/reference_MIX17.npz $D/reference_MIX17.partial.npz $D/build_reference.json \
  $D/compare_release_ref.json $D/SHA256SUMS $D/SHA256SUMS.tmp $D/REFBUILD_OK.json
cp release/tools_build_reference.py $D/build_reference.py || exit 2
cmp -s release/tools_build_reference.py $D/build_reference.py || { echo "the copy of the build script differs"; exit 2; }
( cd $D && env BEAM_PHOTONS=remove MAXENT_DIR=$L T_DIR=$L/output/temperature OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B -u build_reference.py --gentune $L/release_lmx --models $L/release_lmx/models \
    --targets-file $D/targets_thrust_nlo.npz --out $D ) || { echo "REFBUILD FAILED (build_reference.py)"; exit 3; }
# every array of the new file against the copy of the same events refreshed under rule v3 (its meta is still the
# published one, so only the arrays must be equal)
"$PY" -B - $D/reference_MIX17.npz $REFRESHED $D/compare_release_ref.json <<'PY' || { echo "REFBUILD FAILED (the refreshed copy differs)"; exit 3; }
import sys, json, numpy as np
new, old, out = sys.argv[1:4]
a, b = np.load(new), np.load(old)
keys = sorted((set(a.files) | set(b.files)) - {'meta'})
diff = [k for k in keys if k not in a.files or k not in b.files or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
        or a[k].tobytes() != b[k].tobytes()]
meta_same = a['meta'].tobytes() == b['meta'].tobytes()
json.dump(dict(new=new, refreshed=old, arrays_compared=keys, arrays_differ=diff, meta_equal=meta_same), open(out, 'w'), indent=1)
print(f'{len(keys) - len(diff)} of {len(keys)} arrays bitwise equal to {old}' + (f'; differ: {diff}' if diff else '')
      + f'; meta {"equal" if meta_same else "rebuilt"}')
sys.exit(1 if diff else 0)
PY
# SHA256SUMS of the three files to upload, with bare names: a user checks the download with it in a folder of its own.
# It is checked before it takes its name, so a failure leaves no SHA256SUMS behind.
( cd $D && sha256sum $UPLOAD > SHA256SUMS.tmp && sha256sum -c SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS ) \
  || { rm -f $D/SHA256SUMS.tmp; echo "REFBUILD FAILED (SHA256SUMS)"; exit 3; }
# the record: the mark, the job, every output with its sha256 and md5, the files to upload, the inputs with their sha256
"$PY" -B - $D $UPLOAD <<'PY' || { echo "REFBUILD FAILED (record)"; exit 3; }
import sys, os, json, hashlib, time
D, upload = sys.argv[1], sys.argv[2:]
def dig(p, a):
    h = hashlib.new(a)
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()
outs = upload + ['SHA256SUMS', 'build_reference.py', 'compare_release_ref.json']
ins = ['output/models/MIXGEOV3S9_cond.npz', 'output/models/MIXGEOV3S9_ref.npz', 'output/models/MIXGEOV3S9_ref_v2_slim.npz',
       'output/temperature/MIXGEOV3S9.json', 'output/thrust_targets_nlo_point.npz', 'output/three_bands_MIXGEOV3S9_lin.json',
       'output/three_bands_MIXGEOV3S9_lin.npz', 'output/bundle_rule.json', 'beam_photons.py', 'score_obs.py', 'bundle_rule.py',
       'release_lmx/models/MIX17_head.npz', 'release_lmx/models/MIX17_trunk.npz', 'release_lmx/models/MIX17_selftest.npz',
       'release_lmx/models/MANIFEST.json']
b = json.load(open(os.path.join(D, 'build_reference.json')))
assert b['outputs']['reference_MIX17.npz']['sha256'] == dig(os.path.join(D, 'reference_MIX17.npz'), 'sha256')
sums = ''.join(f'{dig(os.path.join(D, f), "sha256")}  {f}\n' for f in upload)
assert open(os.path.join(D, 'SHA256SUMS'), 'rb').read() == sums.encode(), 'SHA256SUMS is not the lines of the files to upload'
rec = dict(job=os.environ.get('SLURM_JOB_ID'), written=time.strftime('%c'),
           mark_run_id=json.load(open('output/.lmx_rerun_start'))['run_id'],
           outputs={f: dict(bytes=os.path.getsize(os.path.join(D, f)), sha256=dig(os.path.join(D, f), 'sha256'),
                            md5=dig(os.path.join(D, f), 'md5')) for f in outs},
           upload=upload + ['SHA256SUMS'],
           inputs={f: dig(f, 'sha256') for f in ins})
tmp = os.path.join(D, f'REFBUILD_OK.json.tmp{os.getpid()}')
json.dump(rec, open(tmp, 'w'), indent=1)
os.replace(tmp, os.path.join(D, 'REFBUILD_OK.json'))
print(f'=== {D}/REFBUILD_OK.json: run {rec["mark_run_id"]}, reference_MIX17.npz sha256 {rec["outputs"]["reference_MIX17.npz"]["sha256"]}')
PY
echo "REFBUILD DONE $(date)"
