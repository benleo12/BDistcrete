#!/usr/bin/env python3
"""The learning-curve table of the training configuration: for every variant of the published
multiplicative mixture (MIXGEO: 208 runs x 80k events, K = 48, 4 members, 36k steps), the closure
width at the 16 held-out points of the paper, split by generator, the baryon count, the
temperature, and the configuration. Reads output/learning_curve/widths_v2_<TAG>.json.
    python learning_curve_table.py [--tex]
"""
import json, os, sys, statistics as st
D = 'output/learning_curve'
OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy', 'nbaryon', 'strange']
# tag: (runs, events per run, K, steps, note)
CFG = {
    'MIXGEOQ':     (52, 80000, 48, 36000, 'quarter runs'),
    'MIXGEOH':     (104, 80000, 48, 36000, 'half runs'),
    'MIXGEOE20':   (208, 20000, 48, 36000, 'quarter events'),
    'MIXGEOE40':   (208, 40000, 48, 36000, 'half events'),
    'MIXGEO':      (208, 80000, 48, 36000, 'published'),
    'MIXGEOR2':    (208, 80000, 48, 36000, 'published, second training'),
    'MIXGEOR3':    (208, 80000, 48, 36000, 'published, third training'),
    'MIXGEOE20S3': (208, 20000, 48, 108000, 'quarter events, 3x steps'),
    'MIXGEOK24':   (208, 80000, 24, 36000, 'K = 24'),
    'MIXGEOK96':   (208, 80000, 96, 36000, 'K = 96'),
    'MIXGEOQS3':   (52, 80000, 48, 108000, 'quarter runs, 3x steps'),
    'MIXGEOHS3':   (104, 80000, 48, 108000, 'half runs, 3x steps'),
    'MIXGEOE40S3': (208, 40000, 48, 108000, 'half events, 3x steps'),
    'MIXGEOS3':    (208, 80000, 48, 108000, '3x steps'),
    'MIXGEOS9':    (208, 80000, 48, 324000, '9x steps'),
    'MIXGEOS27':   (208, 80000, 48, 972000, '27x steps (Symmetry)'),
    'MIXGEOSYM':   (208, 80000, 48, 36000, 'published, Symmetry cross-check'),
    'MIXGEOV3E20S9': (720, 20000, 48, 324000, 'v3 runs, 20k events, 9x steps (Symmetry)'),
    'MIXGEOV3S9':  (720, 80000, 48, 324000, 'v3 runs, 9x steps (Symmetry)'),
    'MIXGEOV4S9':  (2768, 80000, 48, 324000, 'v4 runs, 9x steps (Symmetry)'),
    'MIXGEOV3S27F': (720, 80000, 48, 972000, 'v3 runs, 27x steps, full schedule (Symmetry)'),
    'MIXGEOV4L':   (2768, 80000, 48, 4320000, 'v4 runs, 4.32M steps, full schedule (Symmetry)'),
    'MIXGEOV3S9s1': (720, 80000, 48, 324000, 'v3 runs, 9x steps, second training (Symmetry)'),
    'MIXGEOV3S9s2': (720, 80000, 48, 324000, 'v3 runs, 9x steps, third training (Symmetry)'),
    'MIXGEOV3M8S9': (720, 80000, 48, 324000, 'v3 runs, 9x steps, 8 members (Symmetry)'),
    'MIXGEOV3E160S9': (720, 160000, 48, 324000, 'v3 runs, 160k events, 9x steps (Symmetry)'),
    'MIXGEOV3E320S9': (720, 320000, 48, 324000, 'v3 runs, 320k events, 9x steps (Symmetry)'),
}
# the 16 held-out points of the paper; the v3 data sets add 32 more (7208-7223, 8208-8223)
PAPER = {str(r) for r in list(range(7200, 7208)) + list(range(8200, 8208))}
rows = []
for tag, (nr, ne, K, steps, note) in CFG.items():
    f = f'{D}/widths_v2_{tag}.json'
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); pr_all = d['per_run']
    pr = {r: v for r, v in pr_all.items() if r in PAPER}        # scored on the paper's 16 points
    all48 = st.mean(st.mean(v[o] for o in OBS) for v in pr_all.values()) if len(pr_all) > len(pr) else None
    gen = {}
    for g, pre in (('S', '72'), ('H', '82')):
        rs = [r for r in pr if r.startswith(pre)]
        gen[g] = {o: st.mean(pr[r][o] for r in rs) for o in OBS}
    allw = st.mean(st.mean(pr[r][o] for o in OBS) for r in pr)
    rows.append((tag, nr, ne, nr*ne/1e6, K, steps, d['T'], allw,
                 st.mean(gen['S'].values()), st.mean(gen['H'].values()),
                 gen['S']['nbaryon'], gen['H']['nbaryon'], note + (f' [48 points: {all48:.3f}]' if all48 else '')))
hdr = f"{'tag':12s} {'runs':>4s} {'ev/run':>6s} {'Mev':>5s} {'K':>3s} {'steps':>6s} {'T':>5s} | {'all':>5s} {'Sherpa':>6s} {'Herwig':>6s} | {'bar S':>5s} {'bar H':>5s}  note"
print(hdr); print('-'*len(hdr))
for r in rows:
    print(f'{r[0]:12s} {r[1]:4d} {r[2]//1000:5d}k {r[3]:5.1f} {r[4]:3d} {r[5]//1000:5d}k {r[6]:5.3g} | '
          f'{r[7]:5.3f} {r[8]:6.3f} {r[9]:6.3f} | {r[10]:5.2f} {r[11]:5.2f}  {r[12]}')
