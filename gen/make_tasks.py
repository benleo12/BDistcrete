"""Turn design CSVs into a task file for gen_batch.sbatch: one line per run,
'<script> <rid> <8 parameters> <events>', Sherpa and Herwig interleaved so every node gets both.
    python make_tasks.py <events> <out.txt> designs/stageE_design_v3.csv designs/stageF_design_v3.csv"""
import csv, sys, itertools, os
ROLES = os.environ.get('ROLES', 'train,held').split(',')
ev, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
script = {'E': 'gen_stageE.sh', 'F': 'gen_stageF.sh'}
lists = []
for f in files:
    g = 'E' if 'stageE' in f else 'F'
    rows = list(csv.reader(open(f)))[1:]
    lists.append([' '.join([script[g], r[0]] + r[2:] + [ev]) for r in rows if r[1] in ROLES])
lines = [x for pair in itertools.zip_longest(*lists) for x in pair if x]
open(out, 'w').write('# ' + ' '.join(sys.argv[1:]) + '\n' + '\n'.join(lines) + '\n')
print(f'{len(lines)} runs -> {out}')
