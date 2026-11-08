#!/usr/bin/env python3
"""Assign the units of inventory/units.tsv to NTASK array tasks (largest first, to the least loaded
task) and write WORK/units.tsv with a 'task' column.   python plan.py WORK NTASK"""
import sys, heapq
WORK, NT = sys.argv[1], int(sys.argv[2])
rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/inventory/units.tsv')]
head, units = rows[0], rows[1:]
iN = head.index('n')
order = sorted(range(len(units)), key=lambda k: -int(units[k][iN]))
heap = [(0, t) for t in range(NT)]; task = {}
for k in order:
    load, t = heapq.heappop(heap); task[k] = t; heapq.heappush(heap, (load + int(units[k][iN]), t))
with open(f'{WORK}/units.tsv', 'w') as fh:
    fh.write('\t'.join(head + ['task']) + '\n')
    for k, u in enumerate(units):
        fh.write('\t'.join(u + [str(task[k])]) + '\n')
print(sorted(heap))
