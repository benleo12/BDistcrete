#!/usr/bin/env python3
"""The summary of the PFN cross-check over seeds (App. A.2), output/crosscheck_seeds_summary.json, from the files of
run_crosscheck_seeds.sh (and of symmetry/pfn_lmx.sbatch in the rerun), output/crosscheck_seeds/<impl>_s<seed>.json.

No script of the repository wrote the published summary. This one gives it again from the published per-seed files,
key for key and bit for bit: the AUC of each training in seed order, and for each implementation its label, the number
of trainings, the mean and the standard deviation (numpy, ddof=1), then the mean of each PyTorch network less that of
the energyflow PFN. The task line is filled in from the per-seed files (events per class, epochs, batch, layer sizes).

    python -B crosscheck_summary.py [<per-seed folder> [<summary file>]]

Defaults: output/crosscheck_seeds and output/crosscheck_seeds_summary.json. It stops, writing nothing, unless every
implementation has the seeds 1 to 5, each file records the seed of its name, and all agree on the task. With
CROSSCHECK_DEVICE set, every PyTorch file must record that device (the rerun trains them on the CPU).

The fourth implementation, torchK1silu, is our conditional PFN at K = 1 as the paper builds it (SiLU, the sum over
particles divided by a scale fixed before training), the network App. A.2 compares with the package's ReLU; torchK1 and
torchplain keep the published ReLU. It is optional: left out when none of its files exist (the published run and the
rerun's job 592574 have none), else all five seeds are required, and the summary then carries its runs, its summary
and difference_K1silu_minus_ef. A torchK1silu file must record act 'silu' and emb_scale_mode 'auto'; a torchK1 or
torchplain file that records its activation must record 'relu', and a torchK1 file that records emb_scale_mode must
record '1.0' (crosscheck_torch.py records both since round B15).

The parameter counts of App. A.2: when every per-seed file of an implementation records its count (params; the
rerun's crosscheck_ef.py and crosscheck_torch.py write it, the published files do not), the summary also carries
params = {implementation: count} and, for the PyTorch networks, params_event_side = {implementation: count of the
event network alone}. The seeds of one implementation must agree on them. Without them the output is as before.
"""
import json, os, sys
import numpy as np

SEEDS = (1, 2, 3, 4, 5)
IMPL = {'ef': 'energyflow PFN (TensorFlow)',                       # the order and labels of the published summary
        'torchplain': 'PyTorch PFN, energyflow architecture',
        'torchK1': 'PyTorch conditional PFN collapsed to K=1',
        'torchK1silu': 'PyTorch conditional PFN collapsed to K=1, as in the paper (SiLU, pooled sum scaled)'}
OPTIONAL = ('torchK1silu',)                   # left out when none of its files exist
# the network a file must have trained: torchK1silu must record it, the others only where they record it
NET = {'torchplain': dict(act='relu'), 'torchK1': dict(act='relu', emb_scale_mode='1.0'),
       'torchK1silu': dict(act='silu', emb_scale_mode='auto')}
TASK = ('Sherpa wide scan, shower coupling 0.08 (run 9000) against 0.14 (run 9010), {per_class}k events per class, '
        '{epochs} epochs, batch {batch_size}, Adam 1e-3, same architecture ({phi})/({f})')

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join('output', 'crosscheck_seeds')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join('output', 'crosscheck_seeds_summary.json')
device = os.environ.get('CROSSCHECK_DEVICE')
bad, runs, task, counts = [], {}, set(), {'params': {}, 'params_event_side': {}}
for k in IMPL:
    if k in OPTIONAL and not any(os.path.exists(os.path.join(src, f'{k}_s{s}.json')) for s in SEEDS):
        continue
    runs[k] = []
    for s in SEEDS:
        f = os.path.join(src, f'{k}_s{s}.json')
        if not os.path.exists(f):
            bad.append(f'missing {f}'); continue
        d = json.load(open(f))
        if d.get('seed') != s:
            bad.append(f'{f} records seed {d.get("seed")!r}')
        if device and k != 'ef' and d.get('device') != device:
            bad.append(f'{f} records device {d.get("device")!r}, not {device}')
        for key, want in NET.get(k, {}).items():
            if (key in d or k == 'torchK1silu') and d.get(key) != want:
                bad.append(f'{f} records {key} {d.get(key)!r}, not {want!r}')
        task.add(json.dumps(dict(per_class=(d['n_train'] + d['n_test'])//2000, epochs=d['epochs'], batch_size=d['batch_size'],
                                 phi=','.join(map(str, d['Phi_sizes'])), f=','.join(map(str, d['F_sizes']))), sort_keys=True))
        runs[k].append(d['auc'])
        for c in counts:
            counts[c].setdefault(k, []).append(d.get(c))
if len(task) > 1:
    bad.append(f'the trainings differ in their task: {sorted(task)}')
COUNTS = {}                                   # App. A.2: only when every file of an implementation records the count
for c, per in counts.items():
    for k, v in per.items():
        if all(x is not None for x in v) and len(v) == len(SEEDS):
            if len(set(v)) > 1:
                bad.append(f'the seeds of {k} record different {c}: {v}')
            else:
                COUNTS.setdefault(c, {})[k] = int(v[0])
if bad:
    sys.exit('crosscheck_summary.py: nothing written:\n  ' + '\n  '.join(bad))
summary = {k: dict(label=IMPL[k], n=len(a), mean=float(np.mean(a)), sd=float(np.std(a, ddof=1))) for k, a in runs.items()}
res = dict(task=TASK.format(**json.loads(task.pop())), runs=runs, summary=summary,
           difference_plain_minus_ef=summary['torchplain']['mean'] - summary['ef']['mean'],
           difference_K1_minus_ef=summary['torchK1']['mean'] - summary['ef']['mean'])
if 'torchK1silu' in summary:
    res['difference_K1silu_minus_ef'] = summary['torchK1silu']['mean'] - summary['ef']['mean']
res.update(COUNTS)
json.dump(res, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
for k, v in summary.items():
    print(f'{k:11s} n={v["n"]} mean AUC {v["mean"]:.4f} sd {v["sd"]:.4f}  {np.round(runs[k], 4).tolist()}')
for c, v in COUNTS.items():
    print(f'{c}: ' + ', '.join(f'{k} {n}' for k, n in v.items()))
silu = f', K=1 SiLU - energyflow {res["difference_K1silu_minus_ef"]:+.5f}' if 'torchK1silu' in summary else ''
print(f'plain - energyflow {res["difference_plain_minus_ef"]:+.5f}, K=1 - energyflow {res["difference_K1_minus_ef"]:+.5f}{silu} -> {out}')
