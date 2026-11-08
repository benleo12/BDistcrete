#!/usr/bin/env python3
"""Prepare and check RUN for the scoring side of the rerun (light: links and checks, no computing).

    cd $RUN && MAXENT_DIR=$RUN /mnt/beegfs/bassi/reweighting/env/torch/bin/python -B symmetry/rerun/setup_lmx.py [--check]
    ... setup_lmx.py --park [--dry-run]

1. Checks the layout of D12: RUN is /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code, every file sync_run.sh
   copied is there with the md5 it recorded (symmetry/rerun/files_synced.md5, the files of files_score.txt and
   of the propagation's files_prop.txt), every data folder the scoring reads resolves to a photon-free (_nb)
   folder. Makes RUN/logs, where Slurm writes the log of every job.
2. Links the trained networks the scorings read into output/models (read-only links to the files of the
   published run in maxent_code/output/models): the MODEL of every line of specs/*.txt, the _cond.npz of
   every factorized one, and the exports the closure, learning-curve and learning-band steps read. A
   name that already exists is left alone if it resolves to the same file, and reported otherwise (an
   export apply_T.py --lmx rewrote with a refitted temperature is a file of its own).
3. Links the published results and bundles into output/published: the training record of every
   standard test (TRAINED), the published reference bundles of the slim and full kind (their
   particles, unchanged by D4, and their old observables, against which every refreshed bundle is
   checked), the published wifi bases (whose basis functions the rerun keeps), and the published
   truncation_fraction.json (and param_count.json, if there), which trunc_lmx.sbatch compares with its own count.
4. Checks the refreshed (photon-free) bundles the later steps need in output/models: each must be there and
   listed in output/bundle_rule.json (I1). Those are made by the observables task.
--check does 1 and 4 only and writes nothing (a missing RUN/logs is then reported).

--park (submit_scoring_lmx.sh, once, at the first submission, before the mark is written; it refuses while the
mark output/.lmx_rerun_start exists): every file of RUN/output that a job of the rerun writes, on either side
(lmx_fingerprint.outputs(): every OUT of specs/*.txt, the weight dumps of the wide scan, the refitted temperatures,
the outputs of every other step, the propagation side's outputs), is moved into output/published_copies/<the same
path below output/>, and a row is added to output/published_copies/LIST.tsv (path, size, md5, reason). No such file
can carry the fingerprint of the current mark, which is written afterwards. A link moves as a link (to the same
target). A copy already parked with the same content is not kept twice; a different one gets the suffix
.park<time>. output/published (the read-only links), the inputs (networks, refreshed bundles, the files of
output/bundle_rule.json) and the particle caches (output/pobs) are never touched, and nothing is moved if a file lies
in a folder that resolves outside RUN/output, or if output/published_copies (or a folder in it) resolves outside
RUN/output. --park --dry-run prints the list.
"""
import os, sys, glob, json, time, importlib.util

RUN = '/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code'
R = os.path.dirname(RUN)
PUBLISHED = '/mnt/beegfs/bassi/reweighting/maxent_code'
CHECK = '--check' in sys.argv
assert os.environ.get('MAXENT_DIR') == RUN, f'MAXENT_DIR must be {RUN}'
os.chdir(RUN)
assert os.path.realpath(RUN) == RUN, f'{RUN} resolves to {os.path.realpath(RUN)}'
_s = importlib.util.spec_from_file_location('lmx_fingerprint', os.path.join(RUN, 'symmetry', 'rerun', 'lmx_fingerprint.py'))
FP = importlib.util.module_from_spec(_s); _s.loader.exec_module(FP)


# ---- --park: the published copies under the names of the rerun's outputs go aside (F1)
def content(p):
    """(size, md5) of what a file or link holds; a dangling link: (0, '-')"""
    if not os.path.exists(p):
        return 0, '-'
    return os.path.getsize(p), FP.md5(p)


def why(p):
    """the reason a file is parked, for LIST.tsv"""
    if os.path.islink(p):
        t = os.path.realpath(p)
        return f'a link to {t}' + (' (the published run)' if t.startswith(PUBLISHED + '/') else '')
    fp = FP.stamp_of(p) if p.endswith('.json') else None
    if fp:
        return f'a result of an earlier rerun (lmx_fingerprint of run {fp.get("run_id")})'
    pub = os.path.join(PUBLISHED, p)
    if os.path.exists(pub):
        return 'a copy of the published file' if FP.md5(pub) == FP.md5(p) else 'not the published file of that name (an earlier rerun?)'
    return 'no published file of that name (an earlier rerun or a test?)'


def park(dry):
    if os.path.lexists(FP.MARK):
        print(f'--park runs only before the first submission: {FP.MARK} exists, so output/ holds this rerun\'s results '
              f'({FP.START_OVER})'); sys.exit(2)
    rule = json.load(open(FP.RULE)).get('bundles', {}) if os.path.exists(FP.RULE) else {}
    inputs = {os.path.realpath(b) for b in rule}
    never = ('output/published/', 'output/published_copies/', 'output/pobs/', 'output/logitcache/')
    found = {}
    for pat, writer in FP.outputs():
        for p in sorted(glob.glob(pat, recursive=True)):      # a dangling link is found too
            if os.path.isdir(p) and not os.path.islink(p):
                continue
            found.setdefault(os.path.normpath(p), writer)
    plan, out_real = [], os.path.realpath('output')

    def inside(path):       # path, or the nearest folder of it that exists, resolves into RUN/output
        while not os.path.lexists(path):
            path = os.path.dirname(path)
        r = os.path.realpath(path)
        return r == out_real or r.startswith(out_real + os.sep)
    for p, writer in sorted(found.items()):
        if p.startswith(never) or p.endswith(('_cond.npz', '_trunk.pt', '_concat.pt', '_ref.npz', '_ref_v2_slim.npz', '_basis.npz')) \
                or os.path.realpath(p) in inputs or p in (FP.RULE, FP.MARK, 'output/thrust_targets_nlo_point.npz', 'output/MANIFEST.tsv'):
            raise SystemExit(f'--park: {p} is an input or a protected file, but matches {writer}: fix lmx_fingerprint.outputs()')
        d = os.path.realpath(os.path.dirname(p))
        if d != out_real and not d.startswith(out_real + os.sep):
            raise SystemExit(f'--park: {p} lies in {os.path.dirname(p)}, which resolves to {d}, outside RUN/output: '
                             f'nothing moved (remove that link first)')
        size, m = content(p)
        dst = os.path.join('output/published_copies', os.path.relpath(p, 'output'))
        if not inside(os.path.dirname(dst)):         # never written through: a name there gets a new suffix
            raise SystemExit(f'--park: {dst} would lie in {os.path.realpath(os.path.dirname(dst))}, outside RUN/output: '
                             f'nothing moved (remove that link first)')
        reason = f'{writer}: {why(p)}'
        if os.path.lexists(dst):
            if os.path.islink(p) and os.path.islink(dst):
                same = os.readlink(dst) == os.path.abspath(os.path.join(os.path.dirname(p), os.readlink(p)))
            else:
                same = not os.path.islink(p) and not os.path.islink(dst) and content(dst)[1] == m
            if same:
                dst = None; reason += '; the same content is parked already, so this copy is removed'
            else:
                dst += time.strftime('.park%Y%m%d-%H%M%S')
                reason += f'; parked as {dst}'
        plan.append((p, dst, size, m, reason))
    lst = 'output/published_copies/LIST.tsv'
    if plan and (os.path.islink(lst) or not inside(os.path.dirname(lst))):
        raise SystemExit(f'--park: {lst} is a link or lies outside RUN/output: nothing moved')
    for p, dst, size, m, reason in plan:
        print(f'{"would park" if dry else "park"} {p}\t{size}\t{m}\t{reason}')
    print(f'--park: {len(plan)} files of output/ ' + ('would be' if dry else 'were') + ' moved into output/published_copies')
    if dry or not plan:
        return
    os.makedirs('output/published_copies', exist_ok=True)
    new = not os.path.exists(lst)
    with open(lst, 'a') as f:
        if new:
            f.write('path\tsize\tmd5\treason\n')
        for p, dst, size, m, reason in plan:
            if dst is None:
                os.remove(p)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.islink(p):         # a link moves as a link, to the same target
                    t = os.readlink(p)
                    os.symlink(t if os.path.isabs(t) else os.path.abspath(os.path.join(os.path.dirname(p), t)), dst)
                    os.remove(p)
                else:
                    os.rename(p, dst)
            f.write(f'{p}\t{size}\t{m}\t{reason}\n')
            f.flush()
    print(f'--park: listed in {lst}')


if '--park' in sys.argv:
    park('--dry-run' in sys.argv)
    sys.exit(0)

problems = []
# Slurm opens the log of every job (#SBATCH -o RUN/logs/...) before the job starts, so the folder must be there first
if not os.path.isdir('logs'):
    if CHECK:
        problems.append(f'missing {RUN}/logs (the Slurm logs; setup_lmx.py without --check makes it)')
    else:
        os.makedirs('logs')
        print(f'made {RUN}/logs')

# ---- 1. scripts, designs, data
try:
    FP.expand(FP.FILES)
except (FP.Stop, OSError) as e:
    problems.append(f'{e} (copy the working tree first: symmetry/rerun/sync_run.sh)')
SYNCED = 'symmetry/rerun/files_synced.md5'      # written by sync_run.sh after a copy whose md5 matched on both sides
if not os.path.exists(SYNCED):
    problems.append(f'missing {SYNCED} (copy the working tree with symmetry/rerun/sync_run.sh)')
else:
    rows = [l.split(None, 1) for l in open(SYNCED) if l.strip()]
    gone = [f.strip() for _, f in rows if not os.path.isfile(f.strip())]
    moved = [f.strip() for h, f in rows if os.path.isfile(f.strip()) and FP.md5(f.strip()) != h]
    if gone or moved:
        problems.append(f'{len(gone)} files that sync_run.sh copied are missing in RUN ({", ".join(gone[:5])}) and '
                        f'{len(moved)} have changed since ({", ".join(moved[:5])}): run symmetry/rerun/sync_run.sh again')
DATA = ['data_stagePURE17_v3', 'data_stagePURE17_v3_h400', 'data_stagePURE17_v3ev20', 'data_stagePURE17_orig',
        'data_stagePURE17_orig_h400', 'data_stagePURE17_orig48', 'data_stagePURE17_orig48_h400', 'data_abc_h400',
        f'{R}/gen/data_abc', f'{R}/gen/data_abc_blind', f'{R}/gen/data_widebox']
for d in DATA:
    real = os.path.realpath(d)
    if not os.path.isdir(d):
        problems.append(f'missing data folder {d}')
    elif not real.endswith('_nb'):
        problems.append(f'{d} resolves to {real}, not a photon-free (_nb) folder')
designs = f'{R}/gen/designs'
if not os.path.isdir(designs):
    if CHECK:
        problems.append(f'missing {designs} (the design files, a link to /mnt/beegfs/bassi/reweighting/gen/designs)')
    else:
        os.symlink('/mnt/beegfs/bassi/reweighting/gen/designs', designs)
        print(f'linked {designs} -> /mnt/beegfs/bassi/reweighting/gen/designs')

# ---- 2. the networks
spec = [{k: v for k, v in d.items() if k != 'file'} for d in FP.spec_lines()]
nets = set()
for s in spec:
    nets.add(s['MODEL'])
    if s['MODEL'].endswith('_trunk.pt'):
        nets.add(s['MODEL'].replace('_trunk.pt', '_cond.npz'))
MIXTAGS = ['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3M8S9', 'MIXGEOV3S27F', 'MIXGEOV3E20S9',
           'MIXGEOV3E160S9', 'MIXGEOV3E320S9', 'MIXGEOV4S9', 'MIXGEOV4L']
for t in MIXTAGS:
    nets |= {f'output/models/{t}_cond.npz', f'output/models/{t}_trunk.pt'}
nets.add('output/models/C_P2augfullfac0_cond.npz')


def link(name, target, optional=False):
    if not os.path.exists(target):
        if not optional:
            problems.append(f'missing published file {target}')
        return
    if os.path.lexists(name):
        if os.path.realpath(name) != os.path.realpath(target):
            print(f'kept {name} (resolves to {os.path.realpath(name)}, not {os.path.realpath(target)})')
        return
    if not CHECK:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        os.symlink(target, name)


for n in sorted(nets):
    link(n, f'{PUBLISHED}/{n}')
# ---- 3. published results and bundles
for s in spec:
    if s.get('TRAINED'):
        link(s['TRAINED'], f'{PUBLISHED}/output/{os.path.basename(s["TRAINED"])}')
for t in MIXTAGS:
    for k in ('ref.npz', 'ref_v2_slim.npz'):
        link(f'output/published/models/{t}_{k}', f'{PUBLISHED}/output/models/{t}_{k}')
link('output/published/models/C_P2augfullfac0_ref.npz', f'{PUBLISHED}/output/models/C_P2augfullfac0_ref.npz')
for t in ('MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3M8S9'):
    link(f'output/published/wifi_{t}_basis.npz', f'{PUBLISHED}/output/wifi_{t}_basis.npz')
# the published counts, which trunc_lmx.sbatch counts again and compares (F28, F29)
link('output/published/truncation_fraction.json', f'{PUBLISHED}/output/truncation_fraction.json')
link('output/published/param_count.json', f'{PUBLISHED}/output/param_count.json', optional=True)

# ---- 4. the refreshed bundles the later steps read: there, and listed in output/bundle_rule.json (I1)
need = [f'output/models/{t}_ref_v2_slim.npz' for t in MIXTAGS] + ['output/models/C_P2augfullfac0_ref.npz']
try:
    listed = {os.path.realpath(b) for b in json.load(open(FP.RULE)).get('bundles', {})}
except (OSError, ValueError, AttributeError):
    listed = None
miss = []
for f in need:
    if not os.path.exists(f):
        miss.append(f)
    elif os.path.realpath(f).startswith(PUBLISHED + '/'):
        problems.append(f'{f} is a link to the published (old) bundle {os.path.realpath(f)}')
    elif listed is not None and os.path.realpath(f) not in listed:
        problems.append(f'{f} is not listed in {FP.RULE}: not refreshed by the v3 cache step')
if miss:
    problems.append(f'missing refreshed bundles (the observables task\'s v3 cache step): {", ".join(miss)}')
if listed is None:
    problems.append(f'missing or unreadable {FP.RULE} (the observables task\'s v3 cache step)')
print(f'refreshed bundles: {len(need) - len(miss)} of {len(need)} present')
if problems:
    print('PROBLEMS:'); [print('  ' + p) for p in problems]
    sys.exit(1)
print('setup ' + ('checked' if CHECK else 'done') + f': {len(spec)} scorings, {len(nets)} network files')
