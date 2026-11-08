#!/usr/bin/env python3
"""Build and check the <name>_nb mirror folders.

    python nb_links.py WORK npz      every mirror folder: copy the non-particle files (meta.json),
                                     link every particles_full_<rid>.npz to the original's target
                                     (a regular original: to the original file itself)
    python nb_links.py WORK csv      stage folders: every shapes_run_<rid>.csv that is a link in the
                                     original becomes a link to the same name in the _nb mirror of
                                     the folder it pointed to (the recomputed file); regular ones were
                                     written by nb_units.py
    python nb_links.py WORK verify   compare every mirror with its original (names, npz targets,
                                     meta.json, csv resolved into a _nb folder, row count = events)

Creates only folders whose names end in _nb (and files inside them). Never writes elsewhere.
"""
import os, sys, json, shutil, filecmp

WORK, MODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = '/mnt/beegfs/bassi/reweighting'
MC, GEN = f'{R}/maxent_code', f'{R}/gen'
GENF = ['data_release', 'data_release_orig_held', 'data_v3', 'data_abc', 'data_abc_blind',
        *[f'data_abc_ext/k{k}' for k in range(1, 5)], *[f'data_ext/k{k}' for k in range(1, 5)],
        *[f'data_trainext/k{k}' for k in range(1, 4)], 'data_widebox']
STAGE = ['data_stagePURE17_v3', 'data_stagePURE17_v3_h400', 'data_stagePURE17_v3ev20', 'data_stagePURE17',
         'data_stagePURE17_orig', 'data_stagePURE17_orig_h400', 'data_stagePURE17_orig48',
         'data_stagePURE17_orig48_h400', 'data_abc_h400', 'data_abc_blind']
FOLDERS = [f'{GEN}/{g}' for g in GENF] + [f'{MC}/{s}' for s in STAGE]


def nb_name(path):
    if path.startswith(GEN + '/'):
        rel = path[len(GEN) + 1:].split('/'); rel[0] += '_nb'
        return GEN + '/' + '/'.join(rel)
    if path.startswith(MC + '/'):
        rel = path[len(MC) + 1:]
        assert '/' not in rel, path
        return f'{MC}/{rel}_nb'
    raise ValueError(path)


def target_of(p):
    """the absolute target of a link (the original link's own target), or p itself for a file"""
    if os.path.islink(p):
        t = os.readlink(p)
        return t if os.path.isabs(t) else os.path.normpath(os.path.join(os.path.dirname(p), t))
    return os.path.abspath(p)


def is_part(n):
    return (n.startswith('particles_full_') and n.endswith('.npz')) or (n.startswith('shapes_run_') and n.endswith('.csv'))


def safe_new(path):
    assert os.path.basename(path.rstrip('/')).endswith('_nb') or os.path.basename(os.path.dirname(path)).endswith('_nb'), path


if MODE == 'npz':
    nl = nc = 0
    for src in FOLDERS:
        dst = nb_name(src)
        safe_new(dst)
        os.makedirs(dst, exist_ok=True)
        for n in sorted(os.listdir(src)):
            p = f'{src}/{n}'
            if os.path.isdir(p):
                continue                                    # k1..k4 are mirrored as folders of their own
            if n.startswith('particles_full_') and n.endswith('.npz'):
                q = f'{dst}/{n}'
                t = target_of(p)
                if os.path.lexists(q):
                    assert os.path.islink(q) and os.readlink(q) == t, q
                    continue
                os.symlink(t, q); nl += 1
            elif not is_part(n):
                q = f'{dst}/{n}'
                if not os.path.exists(q):
                    shutil.copy2(p, q); nc += 1
    print(f'npz links made {nl}, other files copied {nc}')

elif MODE == 'csv':
    nl = 0; regular = 0; missing = []
    for src in [f'{MC}/{s}' for s in STAGE]:
        dst = nb_name(src)
        for n in sorted(os.listdir(src)):
            if not (n.startswith('shapes_run_') and n.endswith('.csv')):
                continue
            p, q = f'{src}/{n}', f'{dst}/{n}'
            if os.path.islink(p):
                t = target_of(p)
                nt = nb_name(os.path.dirname(t)) + '/' + os.path.basename(t)
                if not os.path.exists(nt):
                    missing.append(nt); continue
                if os.path.lexists(q):
                    assert os.path.islink(q) and os.readlink(q) == nt, q
                    continue
                os.symlink(nt, q); nl += 1
            else:
                regular += 1
                if not os.path.isfile(q) or os.path.islink(q):
                    missing.append(q)
    print(f'csv links made {nl}, regular csv expected {regular}, missing {len(missing)} {missing[:5]}')
    sys.exit(1 if missing else 0)

elif MODE == 'verify':
    units = {}
    for f in os.listdir(f'{WORK}/units'):
        d = json.load(open(f'{WORK}/units/{f}'))
        units[os.path.realpath(d['csv_out'])] = d
    report = {}
    bad = []
    for src in FOLDERS:
        dst = nb_name(src)
        a = sorted(n for n in os.listdir(src) if not os.path.isdir(f'{src}/{n}'))
        b = sorted(n for n in os.listdir(dst) if not os.path.isdir(f'{dst}/{n}'))
        rec = dict(names_equal=a == b, n=len(a))
        if a != b:
            bad.append((dst, 'names', sorted(set(a) ^ set(b))[:5]))
        nn = nr = 0
        npz_of = {int(n[len('particles_full_'):-4]): n for n in a
                  if n.startswith('particles_full_') and n.endswith('.npz')}
        for n in a:
            p, q = f'{src}/{n}', f'{dst}/{n}'
            if n.endswith('.npz') and n.startswith('particles_full_'):
                if not (os.path.islink(q) and os.path.realpath(q) == os.path.realpath(p)):
                    bad.append((q, 'npz target'))
            elif n.startswith('shapes_run_') and n.endswith('.csv'):
                rq = os.path.realpath(q)
                if not os.path.isfile(rq) or '_nb/' not in rq:
                    bad.append((q, 'csv not in a _nb folder')); continue
                u = units.get(rq)
                if u is None:
                    bad.append((q, 'csv without unit record')); continue
                # the unit was computed from the same particles this folder links
                rid = int(n[len('shapes_run_'):-4])
                if rid not in npz_of or os.path.realpath(u['npz']) != os.path.realpath(f'{src}/{npz_of[rid]}'):
                    bad.append((q, 'unit npz differs', u['npz']))
                if os.path.getsize(rq) != u['out_bytes']:
                    bad.append((q, 'size differs from the unit record'))
                nn += 1; nr += u['n_events']
            else:
                if not filecmp.cmp(p, q, shallow=False):
                    bad.append((q, 'copy differs'))
        rec.update(csv=nn, events=nr)
        report[dst] = rec
    json.dump(dict(report=report, bad=bad), open(f'{WORK}/verify.json', 'w'), indent=1)
    for k, v in report.items():
        print(k, v)
    print('problems:', len(bad), bad[:10])
    sys.exit(1 if bad else 0)
