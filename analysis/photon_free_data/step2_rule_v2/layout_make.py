#!/usr/bin/env python3
"""Step 4 of B0-observables (D12): build the run layout under /mnt/beegfs/bassi/reweighting/lmx_runs.

  lmx_runs/maxent_code/data_<X>  -> maxent_code/data_<X>_nb   (Phase A's photon-free mirrors)
  lmx_runs/maxent_code/data_stagePURE17_v4 -> lmx_runs/data_nb/data_stagePURE17_v4_nb, a mirror made
                                    here (Phase A made none: 2048 of its training runs are gone):
                                    meta.json copied, every run that still exists linked, its shapes
                                    file to the _nb counterpart
  lmx_runs/gen/<g>                -> gen/<g>_nb, for every gen folder with an _nb mirror
  lmx_runs/gen/*.py, gen/designs/ copies of gen's small modules and design files
  lmx_runs/env                    -> /mnt/beegfs/bassi/reweighting/env (the batch scripts activate
                                    $R/env/torch with R the parent of MAXENT_DIR)
  lmx_runs/maxent_code/output/    the files of layout_plan.tsv: refreshed (written by
                                    refresh_bundles.py), copied, symlinked; MANIFEST.tsv
    python layout_make.py PLAN_TSV NPROC
Writes only under lmx_runs. Idempotent: an existing entry is checked, not rewritten.
"""
import os, sys, json, shutil, hashlib, glob
from multiprocessing import Pool

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN = f'{R}/maxent_code', f'{R}/gen'
ROOT = f'{R}/lmx_runs'
RUN = f'{ROOT}/maxent_code'
OUT = f'{RUN}/output'
STAGE = ['data_stagePURE17_v3', 'data_stagePURE17_v3_h400', 'data_stagePURE17_v3ev20', 'data_stagePURE17',
         'data_stagePURE17_orig', 'data_stagePURE17_orig_h400', 'data_stagePURE17_orig48',
         'data_stagePURE17_orig48_h400', 'data_abc_h400', 'data_abc_blind']
GENF = ['data_release', 'data_release_orig_held', 'data_v3', 'data_abc', 'data_abc_blind', 'data_abc_ext', 'data_ext',
        'data_trainext', 'data_widebox']


def inside(p):
    assert os.path.abspath(p).startswith(ROOT + '/'), p
    return p


def link(target, name):
    inside(name)
    if os.path.lexists(name):
        assert os.path.islink(name) and os.readlink(name) == target, (name, os.readlink(name) if os.path.islink(name) else 'not a link', target)
        return 'exists'
    os.makedirs(os.path.dirname(name), exist_ok=True)
    os.symlink(target, name)
    return 'made'


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_hash(args):
    """copy src to dst (temp name, then rename) and return the sha256 of both"""
    src, dst = args
    inside(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and not os.path.islink(dst) and os.path.getsize(dst) == os.path.getsize(src):
        hs, hd = sha256(src), sha256(dst)
        return dst, hs, hd, 'exists'
    tmp = f'{dst}.tmp{os.getpid()}'
    h = hashlib.sha256()
    with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
        while True:
            b = fi.read(1 << 24)
            if not b:
                break
            h.update(b); fo.write(b)
    shutil.copystat(src, tmp)
    os.replace(tmp, dst)
    return dst, h.hexdigest(), sha256(dst), 'made'


def hash_only(path):
    return path, sha256(path)


def v4_mirror():
    """lmx_runs/data_nb/data_stagePURE17_v4_nb: meta.json copied, each existing run linked"""
    src = f'{MC}/data_stagePURE17_v4'
    dst = inside(f'{ROOT}/data_nb/data_stagePURE17_v4_nb')
    os.makedirs(dst, exist_ok=True)
    rec = dict(npz=0, csv=0, gone=0, other=[])
    for n in sorted(os.listdir(src)):
        p = f'{src}/{n}'
        if n == 'meta.json':
            if not os.path.exists(f'{dst}/{n}'):
                shutil.copy2(p, f'{dst}/{n}')
            continue
        if not (n.startswith('particles_full_') or n.startswith('shapes_run_')):
            rec['other'].append(n); continue
        if not os.path.exists(p):                       # one of the 2048 deleted v4 runs
            rec['gone'] += 1; continue
        t = os.path.realpath(p)
        if n.startswith('particles_full_'):
            link(t, f'{dst}/{n}'); rec['npz'] += 1
        else:
            d = os.path.dirname(t)
            assert d.startswith(GEN + '/'), t
            rel = d[len(GEN) + 1:].split('/'); rel[0] += '_nb'
            nt = GEN + '/' + '/'.join(rel) + '/' + os.path.basename(t)
            assert os.path.isfile(nt), nt
            link(nt, f'{dst}/{n}'); rec['csv'] += 1
    return rec


if __name__ == '__main__':
    plan, nproc = sys.argv[1], int(sys.argv[2])
    log = {}
    os.makedirs(OUT + '/models', exist_ok=True)
    # data folders addressed relative to maxent_code
    log['data_links'] = {x: link(f'{MC}/{x}_nb', f'{RUN}/{x}') for x in STAGE}
    log['v4_mirror'] = v4_mirror()
    log['data_links']['data_stagePURE17_v4'] = link(f'{ROOT}/data_nb/data_stagePURE17_v4_nb', f'{RUN}/data_stagePURE17_v4')
    # gen
    log['gen_links'] = {g: link(f'{GEN}/{g}_nb', f'{ROOT}/gen/{g}') for g in GENF}
    log['gen_py'] = []
    for p in sorted(glob.glob(f'{GEN}/*.py')):
        d = f'{ROOT}/gen/{os.path.basename(p)}'
        if not os.path.exists(d):
            shutil.copy2(p, d)
        log['gen_py'].append(os.path.basename(p))
    os.makedirs(f'{ROOT}/gen/designs', exist_ok=True)
    log['gen_designs'] = []
    for p in sorted(glob.glob(f'{GEN}/designs/*')):
        if os.path.isfile(p):
            d = f'{ROOT}/gen/designs/{os.path.basename(p)}'
            if not os.path.exists(d):
                shutil.copy2(p, d)
            log['gen_designs'].append(os.path.basename(p))
    log['env_link'] = link(f'{R}/env', f'{ROOT}/env')
    # output/
    rows = [l.rstrip('\n').split('\t') for l in open(plan)]
    head = rows[0]
    rows = [dict(zip(head, r)) for r in rows[1:]]
    copies = [(r['source'], f"{OUT}/{r['path']}") for r in rows if r['kind'] == 'copied']
    res = {}
    with Pool(nproc) as pool:
        for dst, hs, hd, st in pool.imap_unordered(copy_hash, copies):
            assert hs == hd, (dst, hs, hd)
            res[dst] = (hd, st)
    for r in rows:
        if r['kind'] == 'symlinked':
            link(r['source'], f"{OUT}/{r['path']}")
    refreshed = [f"{OUT}/{r['path']}" for r in rows if r['kind'] == 'refreshed']
    missing = [p for p in refreshed if not os.path.isfile(p)]
    with Pool(nproc) as pool:
        rh = dict(pool.map(hash_only, [p for p in refreshed if os.path.isfile(p)]))
    man = []
    for r in rows:
        p = f"{OUT}/{r['path']}"
        if r['kind'] == 'copied':
            man.append((r['path'], 'copied', r['source'], os.path.getsize(p), res[p][0]))
        elif r['kind'] == 'refreshed':
            man.append((r['path'], 'refreshed', r['source'], os.path.getsize(p) if p in rh else -1, rh.get(p, 'MISSING')))
        else:
            man.append((r['path'], 'symlinked', os.readlink(p), os.path.getsize(os.path.realpath(p)), ''))
    tmp = f'{OUT}/MANIFEST.tsv.tmp'
    with open(tmp, 'w') as fh:
        fh.write('path\tkind\tsource\tsize\tsha256\n')
        for m in man:
            fh.write('\t'.join(str(x) for x in m) + '\n')
    os.replace(tmp, f'{OUT}/MANIFEST.tsv')
    log['output'] = dict(copied=len(copies), copied_new=sum(v[1] == 'made' for v in res.values()),
                         symlinked=sum(r['kind'] == 'symlinked' for r in rows), refreshed=len(refreshed),
                         refreshed_missing=missing,
                         bytes_copied=sum(m[3] for m in man if m[1] == 'copied'),
                         bytes_refreshed=sum(m[3] for m in man if m[1] == 'refreshed' and m[3] > 0))
    json.dump(log, open(f'{R}/lmx_phaseB/observables/results/layout_make.json', 'w'), indent=1)
    print(json.dumps(log['output'], indent=1))
    print('data links', log['data_links'], '\nv4 mirror', log['v4_mirror'], '\ngen links', log['gen_links'])
    sys.exit(1 if missing else 0)
