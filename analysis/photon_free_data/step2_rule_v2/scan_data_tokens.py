#!/usr/bin/env python3
"""Every data path written in the scripts of docs/REPRODUCE.md (and the scripts and batch files they
run), as tokens relative to the analysis folder (maxent_code on Symmetry): data_<...>, ../gen/<...>.
    python scan_data_tokens.py REPO OUT_TSV
REPO is the BDistcrete checkout. Writes script, line, token."""
import os, re, sys

REPO, OUT = sys.argv[1], sys.argv[2]
A = f'{REPO}/analysis'
doc = open(f'{REPO}/docs/REPRODUCE.md').read()
named = set(re.findall(r'[A-Za-z0-9_./-]+\.(?:py|sbatch|sh)', doc))
# the scripts and modules that the named ones run or import (read off the batch files and imports)
extra = ['stage_mixture.py', 'mixture_cfg.py', 'score_obs.py', 'beam_photons.py', 'wifi_fit.py', 'wifi_coverage.py',
         'three_bands.py', 'theory_variations.py', 'band_checks.py', 'kscan.py', 'xp_closure.py', 'ab_analysis.py',
         'package_release.py', 'make_stage_table.py', 'precision_gap.py', 'closure_vs_distance.py', 'fit_accuracy_law.py',
         'recalibrate_T.py', 'apply_T.py', 'check_linear_mix.py', 'conventions.py', 'crosscheck_data.py',
         'crosscheck_ef.py', 'crosscheck_torch.py', 'fit_alpha0.py', 'np_shift.py', 'lepcov/lepdata_cov.py',
         'symmetry/wifi_cov.sbatch', 'symmetry/score_p2.sbatch']
files = []
for n in sorted(named) + extra:
    n = n.replace('analysis/', '')
    for cand in (f'{A}/{n}', f'{A}/symmetry/{os.path.basename(n)}', f'{A}/phase2tools/{os.path.basename(n)}'):
        if os.path.isfile(cand) and cand not in files:
            files.append(cand); break
TOK = re.compile(r'(?<![A-Za-z0-9_/.])((?:\.\./gen/)?(?:data_[A-Za-z0-9_]+|designs)(?:/k\d)?)')
ABS = re.compile(r'(/mnt/beegfs/bassi/reweighting[A-Za-z0-9_./-]*)')
rows = []
for f in files:
    for i, line in enumerate(open(f, errors='replace'), 1):
        for t in TOK.findall(line):
            if t == 'designs' or t.startswith('data_') or t.startswith('../gen/'):
                rows.append((os.path.relpath(f, A), i, t))
        for t in ABS.findall(line):
            rows.append((os.path.relpath(f, A), i, 'ABS:' + t))
# the commands of REPRODUCE.md itself
for i, line in enumerate(doc.split('\n'), 1):
    for t in TOK.findall(line):
        if t.startswith('data_') or t.startswith('../gen/'):
            rows.append(('docs/REPRODUCE.md', i, t))
with open(OUT, 'w') as fh:
    fh.write('script\tline\ttoken\n')
    for r in rows:
        fh.write('\t'.join(str(x) for x in r) + '\n')
print(len(files), 'files scanned,', len(rows), 'tokens,', len({r[2] for r in rows}), 'distinct')
print(sorted({r[2] for r in rows}))
