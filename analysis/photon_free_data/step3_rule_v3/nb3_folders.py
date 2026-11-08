#!/usr/bin/env python3
"""NB_FOLDERS.txt against nb_links.py and the unit list, before anything is written. The listed folders
must be exactly nb_links.py's FOLDERS through its nb_name (read from its source with ast; the script is
not run); each must be a folder that is its own real path (no link, neither the folder nor a folder on
its way); and the _nb CSV of every unit of WORK/units.tsv must be a regular file (not a link) directly
inside one of them. nb3_units.py refuses any other destination.
    python nb3_folders.py WORK          WORK/results/nb_folders.json; exit 1 on any problem"""
import os, sys, ast, json

WORK = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ('R', 'MC', 'GEN', 'GENF', 'STAGE', 'FOLDERS')


def names(t):
    return [t.id] if isinstance(t, ast.Name) else [x for e in getattr(t, 'elts', [None]) for x in (names(e) if e else ['?'])]


src = f'{HERE}/nb_links.py'
tree = ast.parse(open(src).read())
keep = [n for n in tree.body if (isinstance(n, ast.Assign) and all(x in NAMES for t in n.targets for x in names(t))) or
        (isinstance(n, ast.FunctionDef) and n.name == 'nb_name')]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), src, 'exec'), ns)
seen = [ns['nb_name'](f) for f in ns['FOLDERS']]
listed = [l.strip() for l in open(f'{HERE}/NB_FOLDERS.txt') if l.strip() and not l.startswith('#')]
bad = []
if sorted(listed) != sorted(seen) or len(set(listed)) != len(listed):
    bad.append(dict(problem='NB_FOLDERS.txt is not what nb_links.py sees', only_listed=sorted(set(listed) - set(seen)),
                    only_nb_links=sorted(set(seen) - set(listed))))
for d in listed:
    if not os.path.isdir(d) or os.path.realpath(d) != d:
        bad.append(dict(problem='not a folder, or not its own real path (a link on the way)', folder=d,
                        real=os.path.realpath(d)))
real = {os.path.realpath(d): d for d in listed}
rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
units = [dict(zip(rows[0], r)) for r in rows[1:]]
per = {d: 0 for d in listed}
for u in units:
    p = u['csv_out']
    d = real.get(os.path.dirname(os.path.realpath(p)))
    if d is None or not os.path.isfile(p) or os.path.islink(p):
        bad.append(dict(problem='unit CSV outside the listed folders, missing or a link', uid=u['uid'], csv_out=p))
    else:
        per[d] += 1
rec = dict(listed=len(listed), nb_links=len(seen), units=len(units), folders_with_units=sum(v > 0 for v in per.values()),
           units_per_folder=per, problems=bad[:50], n_problems=len(bad))
os.makedirs(f'{WORK}/results', exist_ok=True)
json.dump(rec, open(f'{WORK}/results/nb_folders.json', 'w'), indent=1)
print(f"NB_FOLDERS: {len(listed)} listed, nb_links.py sees {len(seen)}; {len(units)} units in "
      f"{rec['folders_with_units']} of them; {len(bad)} problems {bad[:3]}")
sys.exit(1 if bad else 0)
