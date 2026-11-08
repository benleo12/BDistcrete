#!/usr/bin/env python3
"""Inventory of the event samples the paper's results read, and the manifest of the photon-free
recomputation (one unit per real particle file whose CSV must be recomputed).

Reads only: directory listings, symlink targets, meta.json, the design CSVs, and the npy header of
the mask array in each real particle file (event count). Writes only into OUT.

    python inventory.py OUT
"""
import os, sys, json, csv, zipfile, collections
import numpy as np

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN = f'{R}/maxent_code', f'{R}/gen'
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# folder -> what reads it (from docs/REPRODUCE.md, docs/DATA.md, CLAUDE.md, the batch scripts, LOG.md)
STAGE = collections.OrderedDict([
    ('data_stagePURE17_v3', 'mixture and Stages D/E: training, standard test (Table 1, 2), Fig. 3, '
                            'K scan Stage D (Fig. 6), \\SvdRank (MIX, E), trunc_frac, reference bundle '
                            'MIXGEOV3S9_ref_v2_slim (Figs. 8-12, \\Env, \\Tb), release_r8, learning curve '
                            '(V3S9, S27F, M8S9)'),
    ('data_stagePURE17_v3_h400', 'strict test of Stages D/E and the mixture (Table 1, 2), wifi_coverage '
                                 '(Fig. 5, \\Cov), release_r8_table, strict tests of every learning-curve row'),
    ('data_stagePURE17_v3ev20', 'learning curve MIXGEOV3E20S9 (\\LcEvTwenty)'),
    ('data_stagePURE17', 'meta.json read by make_numbers.py (held-out points of the 208-run set)'),
    ('data_stagePURE17_orig', 'learning curve, 208-run cells origfull (\\Lc Runs, \\Orig)'),
    ('data_stagePURE17_orig_h400', 'strict test of the origfull cells'),
    ('data_stagePURE17_orig48', 'origfull cells scored at the 48 held-out points (make_orig48.py)'),
    ('data_stagePURE17_orig48_h400', 'strict test of the same'),
    ('data_abc_h400', 'strict test of Stages A-C (Table 1, 2)'),
    ('data_abc_blind', 'blind test p2augblind of Stage C (cross-check of Fig. 4, no macro)'),
    ('data_stagePURE17_v4', 'learning curve MIXGEOV4S9, MIXGEOV4L (\\LcRunsFour, \\LcStepsFour)'),
])
GENF = collections.OrderedDict([
    ('data_release', 'the 224 released runs (via data_stagePURE17*, v3, v3_h400, orig*)'),
    ('data_release_orig_held', 'Perlmutter originals of held runs 8200, 8203 (via data_stagePURE17_orig)'),
    ('data_v3', 'the 544 v3 runs (via data_stagePURE17_v3, v3_h400, orig48)'),
    ('data_abc', 'Stages A-C: training, standard test, Fig. 4, xp_null, single_members, K scan, \\Calib, \\SvdRank'),
    ('data_abc_blind', 'twelve fresh Stage C runs 6950-6961 (Fig. 4, \\Map)'),
    ('data_abc_ext/k1', 'extensions of the 16 A-C held-out runs (source of data_abc_h400)'),
    ('data_abc_ext/k2', ''), ('data_abc_ext/k3', ''), ('data_abc_ext/k4', ''),
    ('data_ext/k1', 'extensions of the 48 mixture held-out runs (source of data_stagePURE17_v3_h400)'),
    ('data_ext/k2', ''), ('data_ext/k3', ''), ('data_ext/k4', ''),
    ('data_trainext/k1', 'extensions of the 720 training runs (source of the deleted v3e160/v3e320, '
                         'learning curve MIXGEOV3E160S9, MIXGEOV3E320S9)'),
    ('data_trainext/k2', ''), ('data_trainext/k3', ''),
    ('data_widebox', 'wide coupling scan (\\Wide), PFN cross-check of App. A.2'),
])


def nb_name(path):
    """the _nb mirror of a folder: gen/data_ext/k1 -> gen/data_ext_nb/k1, maxent_code/x -> x_nb"""
    if path.startswith(GEN + '/'):
        rel = path[len(GEN) + 1:].split('/')
        rel[0] = rel[0] + '_nb'
        return GEN + '/' + '/'.join(rel)
    if path.startswith(MC + '/'):
        rel = path[len(MC) + 1:].split('/')
        assert len(rel) == 1, path
        return MC + '/' + rel[0] + '_nb'
    raise ValueError(path)


def nevents(npz):
    with zipfile.ZipFile(npz) as zf:
        with zf.open('mask.npy') as fh:
            v = np.lib.format.read_magic(fh)
            shape = (np.lib.format.read_array_header_1_0(fh) if v == (1, 0)
                     else np.lib.format.read_array_header_2_0(fh))[0]
    return int(shape[0]), int(shape[1])


def scan(folder):
    """every particle and shapes file of a folder: kind, link target, exists"""
    rows = []
    for n in sorted(os.listdir(folder)):
        if not (n.startswith('particles_full_') and n.endswith('.npz')) and \
           not (n.startswith('shapes_run_') and n.endswith('.csv')):
            continue
        p = f'{folder}/{n}'
        link = os.path.islink(p)
        tgt = os.readlink(p) if link else None
        rows.append(dict(name=n, link=link, target=tgt, exists=os.path.exists(p),
                         real=os.path.realpath(p)))
    return rows


def rid_of(name):
    return int(name.split('_')[-1].split('.')[0])


inv = dict(stage={}, gen={}, missing_folders=[])
units = []                      # real particle files whose CSV is recomputed
real_seen = {}
# ---- gen folders
design_rids = {}
for S in 'ABC':
    for r in csv.DictReader(open(f'{GEN}/designs/stage{S}_design_aug.csv')):
        design_rids.setdefault('abc', set()).add(int(r['run_id']))
        if r['role'] == 'held':
            design_rids.setdefault('abc_held', set()).add(int(r['run_id']))
design_rids['widebox'] = {int(r['run_id']) for r in csv.DictReader(open(f'{GEN}/designs/stageW_design.csv'))}
m3 = json.load(open(f'{MC}/data_stagePURE17_v3/meta.json'))
design_rids['v3_train'] = {x['rid'] for x in m3['train']}
design_rids['v3_held'] = {x['rid'] for x in m3['held']}
SHERPA = {x['rid'] for x in m3['train'] + m3['held'] if x['n_herwig'] == 0}
HERWIG = {x['rid'] for x in m3['train'] + m3['held'] if x['n_herwig'] > 0}
expect = {'data_abc': design_rids['abc'], 'data_abc_blind': set(range(6950, 6962)),
          'data_widebox': design_rids['widebox']}
for k in range(1, 5):
    expect[f'data_abc_ext/k{k}'] = design_rids['abc_held']
    expect[f'data_ext/k{k}'] = design_rids['v3_held']
for k in range(1, 4):
    expect[f'data_trainext/k{k}'] = design_rids['v3_train']

for g, why in GENF.items():
    folder = f'{GEN}/{g}'
    if not os.path.isdir(folder):
        inv['missing_folders'].append(folder); continue
    rows = scan(folder)
    npz = {rid_of(r['name']): r for r in rows if r['name'].endswith('.npz')}
    cs = {rid_of(r['name']): r for r in rows if r['name'].endswith('.csv')}
    rec = dict(why=why, n_npz=len(npz), n_csv=len(cs),
               n_links=sum(r['link'] for r in rows), broken=[r['name'] for r in rows if not r['exists']],
               npz_without_csv=sorted(set(npz) - set(cs)), csv_without_npz=sorted(set(cs) - set(npz)))
    if g in expect:
        rec['expected'] = len(expect[g]); rec['missing_expected'] = sorted(expect[g] - set(npz))
        rec['extra'] = sorted(set(npz) - expect[g])
    nev = []
    for rid in sorted(npz):
        r = npz[rid]
        if r['link'] or rid not in cs or cs[rid]['link']:
            continue
        n, pmax = nevents(f'{folder}/{r["name"]}')
        nev.append(n)
        gen = 'H' if rid in HERWIG else 'S'
        out = nb_name(folder) + '/' + cs[rid]['name']
        units.append(dict(uid=f'{g.replace("/", "_")}__{rid}', npz=f'{folder}/{r["name"]}',
                          csv_in=f'{folder}/{cs[rid]["name"]}', csv_out=out, group=g, rid=rid,
                          gen=gen, n=n))
        real_seen[f'{folder}/{r["name"]}'] = units[-1]['uid']
    rec['events'] = dict(runs=len(nev), total=int(np.sum(nev)), min=int(min(nev)) if nev else 0,
                         max=int(max(nev)) if nev else 0)
    inv['gen'][g] = rec

# ---- stage folders in maxent_code
for s, why in STAGE.items():
    folder = f'{MC}/{s}'
    if not os.path.isdir(folder):
        inv['missing_folders'].append(folder); continue
    rows = scan(folder)
    npz = {rid_of(r['name']): r for r in rows if r['name'].endswith('.npz')}
    cs = {rid_of(r['name']): r for r in rows if r['name'].endswith('.csv')}
    meta = json.load(open(f'{folder}/meta.json')) if os.path.exists(f'{folder}/meta.json') else None
    rec = dict(why=why, n_npz=len(npz), n_csv=len(cs), other=sorted(n for n in os.listdir(folder)
               if not n.startswith(('particles_full_', 'shapes_run_'))),
               broken_npz=sorted(rid for rid, r in npz.items() if not r['exists']),
               broken_csv=sorted(rid for rid, r in cs.items() if not r['exists']))
    tg = collections.Counter()
    for r in rows:
        if r['link']:
            tg[('csv' if r['name'].endswith('csv') else 'npz', os.path.dirname(r['target'] if os.path.isabs(r['target'])
                else os.path.normpath(os.path.join(folder, r['target']))))] += 1
        else:
            tg[('csv' if r['name'].endswith('csv') else 'npz', '(regular file)')] += 1
    rec['targets'] = [dict(kind=k, dir=d, n=n) for (k, d), n in sorted(tg.items())]
    if meta:
        want = [x['rid'] for x in meta['train']] + [x['rid'] for x in meta['held']]
        rec['meta_runs'] = dict(train=len(meta['train']), held=len(meta['held']))
        rec['meta_missing_npz'] = sorted(r for r in want if r not in npz or not npz[r]['exists'])
        rec['meta_missing_csv'] = sorted(r for r in want if r not in cs or not cs[r]['exists'])
        rec['meta_events'] = dict(n_train_events=meta.get('n_train_events'), n_held_events=meta.get('n_held_events'))
    nev = []
    for rid in sorted(npz):
        r = npz[rid]
        if r['link'] or rid not in cs:
            continue
        assert not cs[rid]['link'], (folder, rid, 'regular npz with a linked csv')
        n, pmax = nevents(f'{folder}/{r["name"]}')
        nev.append(n)
        gen = 'H' if rid in HERWIG else 'S'
        units.append(dict(uid=f'{s}__{rid}', npz=f'{folder}/{r["name"]}', csv_in=f'{folder}/{cs[rid]["name"]}',
                          csv_out=nb_name(folder) + '/' + cs[rid]['name'], group=s, rid=rid, gen=gen, n=n))
    rec['regular_runs'] = dict(runs=len(nev), total=int(np.sum(nev)) if nev else 0)
    # every csv link must point into a folder that is itself mirrored
    mirrored = {f'{GEN}/{g}' for g in GENF} | {f'{MC}/{x}' for x in STAGE}
    rec['csv_link_dirs_not_mirrored'] = sorted({d['dir'] for d in rec['targets'] if d['kind'] == 'csv'
                                                and d['dir'] != '(regular file)' and d['dir'] not in mirrored})
    inv['stage'][s] = rec

# absent stage folders that the learning curve built (cleanup_learning_curve_data.sh, Oct 18)
for s in ('data_stagePURE17_v3e160', 'data_stagePURE17_v3e320'):
    inv['stage'][s] = dict(why='learning curve MIXGEOV3E160S9 / MIXGEOV3E320S9', present=os.path.isdir(f'{MC}/{s}'))
inv['gen_data_v4_present'] = os.path.isdir(f'{GEN}/data_v4')

json.dump(inv, open(f'{OUT}/inventory.json', 'w'), indent=1)
with open(f'{OUT}/units.tsv', 'w') as fh:
    fh.write('uid\tnpz\tcsv_in\tcsv_out\tgroup\trid\tgen\tn\n')
    for u in units:
        fh.write('\t'.join(str(u[k]) for k in ('uid', 'npz', 'csv_in', 'csv_out', 'group', 'rid', 'gen', 'n')) + '\n')
tot = sum(u['n'] for u in units)
print(f'{len(units)} units, {tot} events')
for g, rec in inv['gen'].items():
    print(f'gen/{g:22s} npz {rec["n_npz"]:5d} csv {rec["n_csv"]:5d} links {rec["n_links"]} broken {len(rec["broken"])} '
          f'missing-expected {len(rec.get("missing_expected", []))} extra {len(rec.get("extra", []))} '
          f'events {rec["events"]}')
for s, rec in inv['stage'].items():
    if 'n_npz' not in rec:
        print(f'{s:30s} present={rec["present"]}'); continue
    print(f'{s:30s} npz {rec["n_npz"]:5d} csv {rec["n_csv"]:5d} broken npz {len(rec["broken_npz"])} csv {len(rec["broken_csv"])} '
          f'meta {rec.get("meta_runs")} missing npz {len(rec.get("meta_missing_npz", []))} csv {len(rec.get("meta_missing_csv", []))} '
          f'regular {rec["regular_runs"]} other {rec["other"]}')
    for t in rec['targets']:
        print(f'      {t["kind"]} -> {t["dir"]}: {t["n"]}')
print('missing folders:', inv['missing_folders'], ' gen/data_v4 present:', inv['gen_data_v4_present'])
