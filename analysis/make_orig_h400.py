"""data_stagePURE17_orig_h400: the 208-run release set (data_stagePURE17_orig) with its 16 held-out
runs replaced by their 400k-event versions from data_stagePURE17_v3_h400 (run plus four extensions),
for the strict test of the original-design mixture. Run from analysis/.

    python make_orig_h400.py
"""
import json, os
SRC, H400, DST = 'data_stagePURE17_orig', 'data_stagePURE17_v3_h400', 'data_stagePURE17_orig_h400'
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f == 'meta.json':
        continue
    p = f'{DST}/{f}'
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(os.path.realpath(f'{SRC}/{f}'), p)
a = json.load(open(f'{SRC}/meta.json')); b = json.load(open(f'{H400}/meta.json'))
hb = {h['rid']: h for h in b['held']}
for h in a['held']:
    rid = h['rid']
    assert hb[rid]['theta'] == h['theta'], rid
    for f in (f'particles_full_{rid:04d}.npz', f'shapes_run_{rid:04d}.csv'):
        os.remove(f'{DST}/{f}'); os.symlink(os.path.realpath(f'{H400}/{f}'), f'{DST}/{f}')
a['n_held_events'] = 400000
a['note'] = a.get('note', '') + ' | held runs at 400k events (run plus four extensions), from ' + H400
json.dump(a, open(f'{DST}/meta.json', 'w'))
print('held', [h['rid'] for h in a['held']], 'files', len(os.listdir(DST)))
