#!/usr/bin/env python3
"""B0's check_paths.py (B5: the results file; B6: stray temp files of an interrupted refresh are listed
and count as failures).

Check of the run layout (D12): every data path the paper's scripts use, as run from
RUN = /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code, must be photon-free.

The paths are the tokens of scan_data_tokens.py (every data_<...> and ../gen/<...> written in the
scripts of docs/REPRODUCE.md and in the scripts and batch files they run, and in REPRODUCE.md), plus
the folders Phase A mirrored. A token that names no folder in the original maxent_code layout either
(a variable name, a default no command of the paper uses) is listed as absent and not checked further.
For every other one:
  - it must exist under RUN, and the folder must resolve into a folder whose name ends in _nb or
    into /mnt/beegfs/bassi/reweighting/lmx_runs
  - every shapes_run_*.csv in it (k-subfolders included) must resolve into an _nb folder and be the
    _nb counterpart of the original's file (same name, same row count)
  - every particles_full_*.npz must exist; it resolves to the original particle file, which D4
    leaves unchanged (the network inputs), as in Phase A's mirrors; this is recorded, not failed
  - any other file (meta.json, design CSVs) must resolve into an _nb folder or lmx_runs
The output/ files are checked against MANIFEST.tsv: refreshed and copied files are regular files
inside lmx_runs with the recorded size (and sha256 with --hash); links resolve to existing files.
Temp files an interrupted refresh can leave (<cache>.npz.tmp<pid>.npz of refresh3_bundles.py,
MANIFEST.tsv.tmp<pid>, bundle_rule.json.tmp<pid>) in output/, output/models/ or lmx_runs/release_ref
are listed and count as failures.
    python check_paths.py TOKENS_TSV [--hash] [--rows]      exit 1 on any failure
"""
import os, re, sys, json, glob, hashlib

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN, ROOT = f'{R}/maxent_code', f'{R}/gen', f'{R}/lmx_runs'
RUN = f'{ROOT}/maxent_code'
NB = re.compile(r'_nb$')
PHASE_A = ['data_stagePURE17_v3', 'data_stagePURE17_v3_h400', 'data_stagePURE17_v3ev20', 'data_stagePURE17',
           'data_stagePURE17_orig', 'data_stagePURE17_orig_h400', 'data_stagePURE17_orig48',
           'data_stagePURE17_orig48_h400', 'data_abc_h400', 'data_abc_blind', 'data_stagePURE17_v4'] + \
          [f'../gen/{g}' for g in ('data_release', 'data_release_orig_held', 'data_v3', 'data_abc', 'data_abc_blind',
                                   'data_abc_ext', 'data_ext', 'data_trainext', 'data_widebox', 'designs')]
HASH = '--hash' in sys.argv
ROWS = '--rows' in sys.argv


def in_nb(p):
    return any(NB.search(c) for c in os.path.realpath(p).split(os.sep)[:-1]) or NB.search(os.path.realpath(p)) is not None


def in_root(p):
    return os.path.realpath(p).startswith(ROOT + '/')


def nrows(p):
    with open(p, 'rb') as fh:
        return sum(1 for _ in fh)


def check_folder(tok, fails, rec):
    run_p = os.path.normpath(os.path.join(RUN, tok))
    orig_p = os.path.normpath(os.path.join(MC, tok))
    r = dict(token=tok, run_path=run_p, realpath=os.path.realpath(run_p) if os.path.exists(run_p) else None)
    if not os.path.exists(orig_p):
        r['status'] = 'absent in the original layout too (not a folder any command uses)'
        rec.append(r); return
    if not os.path.exists(run_p):
        r['status'] = 'MISSING'; fails.append((tok, 'missing under RUN')); rec.append(r); return
    if not (in_nb(run_p) or in_root(run_p)):
        r['status'] = 'OUTSIDE'; fails.append((tok, f'resolves to {os.path.realpath(run_p)}')); rec.append(r); return
    n = dict(csv=0, npz=0, other=0, csv_bad=0, npz_bad=0, other_bad=0, npz_targets=set())
    dirs = [run_p] + [os.path.join(run_p, d) for d in sorted(os.listdir(run_p)) if os.path.isdir(os.path.join(run_p, d))]
    for d in dirs:
        od = os.path.join(orig_p, os.path.relpath(d, run_p))
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isdir(p):
                continue
            if f.startswith('shapes_run_') and f.endswith('.csv'):
                n['csv'] += 1
                ok = os.path.isfile(p) and in_nb(p)
                o = os.path.join(od, f)
                if ok and os.path.exists(o):
                    ok = os.path.basename(os.path.realpath(p)) == os.path.basename(os.path.realpath(o))
                    if ok and ROWS:
                        ok = nrows(p) == nrows(o)
                if not ok:
                    n['csv_bad'] += 1; fails.append((tok, f'{p} -> {os.path.realpath(p)}'))
            elif f.startswith('particles_full_') and f.endswith('.npz'):
                n['npz'] += 1
                if not os.path.isfile(p):
                    n['npz_bad'] += 1; fails.append((tok, f'{p}: no particle file'))
                else:
                    n['npz_targets'].add(os.path.dirname(os.path.realpath(p)))
            else:
                n['other'] += 1
                if not (in_nb(p) or in_root(p)):
                    n['other_bad'] += 1; fails.append((tok, f'{p} -> {os.path.realpath(p)}'))
    n['npz_targets'] = sorted(n['npz_targets'])
    r.update(status='ok' if not (n['csv_bad'] or n['npz_bad'] or n['other_bad']) else 'FAIL', **n)
    rec.append(r)


def check_manifest(fails):
    out = f'{RUN}/output'
    rows = [l.rstrip('\n').split('\t') for l in open(f'{out}/MANIFEST.tsv')]
    head, rows = rows[0], [dict(zip(rows[0], x)) for x in rows[1:]]
    st = dict(entries=len(rows), copied=0, refreshed=0, symlinked=0, bad=0)
    for m in rows:
        p = f"{out}/{m['path']}"
        st[m['kind']] += 1
        if m['kind'] in ('copied', 'refreshed'):
            ok = os.path.isfile(p) and not os.path.islink(p) and in_root(p) and os.path.getsize(p) == int(m['size'])
            if ok and HASH:
                h = hashlib.sha256()
                with open(p, 'rb') as fh:
                    for b in iter(lambda: fh.read(1 << 24), b''):
                        h.update(b)
                ok = h.hexdigest() == m['sha256']
        else:
            ok = os.path.islink(p) and os.path.exists(p)
        if not ok:
            st['bad'] += 1; fails.append(('output', p))
    return st


def stray_tmp(fails):
    o = f'{RUN}/output'
    found = sorted(glob.glob(f'{o}/*.npz.tmp[0-9]*.npz') + glob.glob(f'{o}/models/*.npz.tmp[0-9]*.npz') +
                   glob.glob(f'{ROOT}/release_ref/*.npz.tmp[0-9]*.npz') + glob.glob(f'{o}/MANIFEST.tsv.tmp[0-9]*') +
                   glob.glob(f'{o}/bundle_rule.json.tmp[0-9]*'))
    for p in found:
        fails.append(('stray temp file', p))
    return found


if __name__ == '__main__':
    trows = [l.rstrip('\n').split('\t') for l in list(open(sys.argv[1]))[1:]]
    toks = sorted({r[2] for r in trows if not r[2].startswith('ABS:')} | set(PHASE_A))
    fails, rec = [], []
    for t in toks:
        check_folder(t, fails, rec)
    # absolute paths written in the scripts: a default such as MAXENT_DIR=/mnt/.../maxent_code is
    # used whenever the variable is not exported, and then reads (and writes) the original folders
    absr = {}
    for scr, line, t in trows:
        if t.startswith('ABS:'):
            p = t[4:].rstrip('/.')
            ok = p.startswith(ROOT) or in_nb(p) or os.path.basename(p).startswith('particles_full_')   # particle files: the unchanged network inputs
            absr.setdefault(p, dict(ok=ok, used_in=[]))['used_in'].append(f'{scr}:{line}')
    abs_bad = {p: v['used_in'] for p, v in absr.items() if not v['ok']}
    for p, where in abs_bad.items():
        fails.append(('absolute path in a script', f'{p} ({len(where)} places: {", ".join(where[:6])}{" ..." if len(where) > 6 else ""})'))
    man = check_manifest(fails)
    stray = stray_tmp(fails)
    res = dict(run=RUN, tokens=len(toks), absolute_paths=absr, absolute_outside=abs_bad, stray_tmp=stray,
               checked=sum(r.get('status') in ('ok', 'FAIL') for r in rec),
               absent=[r['token'] for r in rec if r['status'].startswith('absent')], folders=rec, manifest=man,
               failures=fails[:200], n_failures=len(fails))
    json.dump(res, open(f'{R}/lmx_phaseB/observables_v3/results/check_paths.json', 'w'), indent=1, default=str)
    for r in rec:
        if r['status'] in ('ok', 'FAIL', 'MISSING', 'OUTSIDE'):
            print(f"{r['status']:7s} {r['token']:34s} -> {r['realpath']}  csv {r.get('csv', 0)} npz {r.get('npz', 0)} other {r.get('other', 0)}")
    print('absent in the original too:', res['absent'])
    print('manifest:', man)
    print(f'stray temp files of a refresh: {len(stray)} {stray[:5]}')
    rel_fail = [f for f in fails if f[0] not in ('absolute path in a script', 'output', 'stray temp file')]
    print(f'relative data paths from RUN: {len(rel_fail)} failures')
    print(f'absolute paths outside lmx_runs written in the scripts: {len(abs_bad)}')
    for p, where in abs_bad.items():
        print(f'   {p}: {len(where)} places, e.g. {where[:4]}')
    print(f'CHECK: {res["checked"]} folders checked, {len(fails)} failures')
    sys.exit(1 if fails else 0)
