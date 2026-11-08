#!/usr/bin/env python3
"""Dynamic version of nb_units.py (NB_TASKS: comma list of tasks to cover, default all; NB_MAXN: skip larger units): every worker takes the next unit nobody has claimed, so a node
that the busy queue grants keeps working until nothing is left (or its time limit is near).

    python nb_worker.py WORK NPROC [EXCLUDE_TASKS]      (EXCLUDE_TASKS e.g. "0": units of a task
                                                          that a static nb_units.py job is running)
A claim is a file WORK/claims/<uid> made with O_EXCL. It is stale, and is taken over, when it was
made by an earlier start of this same job (requeue after preemption) or is older than three times
the expected time of the unit plus 15 minutes. NB_TLIMIT (seconds) is the job's time limit: no unit
is started that would not end 5 minutes before it.
"""
import os, sys, time
WORK, NPROC = sys.argv[1], int(sys.argv[2])
EXCLUDE = set(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
sys.argv = [sys.argv[0], WORK, '0', '1', str(NPROC)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_units as NU                                      # noqa: E402  (process, safe)

TLIMIT = float(os.environ.get('NB_TLIMIT', '14400'))
T0 = time.time()
JOB = os.environ.get('SLURM_JOB_ID', 'local')
RST = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
CLAIMS = f'{WORK}/claims'


def est(u):
    return 4e-3 * int(u['n']) + 120


def claim(u):
    p = f'{CLAIMS}/{u["uid"]}'
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{JOB} {RST} {time.time():.0f} {os.uname()[1]}\n'.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                jid, rst, t, _host = open(p).read().split()
            except Exception:
                return False
            stale = (jid == JOB and int(rst) < RST) or (time.time() - float(t) > 3 * est(u) + 900)
            if not stale:
                return False
            try:
                os.rename(p, f'{p}.stale.{JOB}.{RST}.{os.getpid()}')      # atomic: one taker wins
            except FileNotFoundError:
                return False
    return False


def work(u):
    if os.path.exists(f'{WORK}/units/{u["uid"]}.json') and os.path.exists(u['csv_out']):
        return u['uid'], 'skip', 0.0
    if time.time() - T0 + est(u) > TLIMIT - 300:
        return u['uid'], 'notime', 0.0
    if not claim(u):
        return u['uid'], 'claimed', 0.0
    return NU.safe(u)


if __name__ == '__main__':
    import multiprocessing as mp
    os.makedirs(CLAIMS, exist_ok=True)
    rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
    units = [dict(zip(rows[0], r)) for r in rows[1:]]
    units = [u for u in units if u['task'] not in EXCLUDE]
    INCL = os.environ.get('NB_TASKS', 'all')
    if INCL != 'all':
        units = [u for u in units if u['task'] in set(INCL.split(','))]
    units = [u for u in units if int(u['n']) <= int(os.environ.get('NB_MAXN', '1000000000'))]
    units.sort(key=lambda u: -int(u['n']))
    print(f'worker job {JOB} restart {RST} on {os.uname()[1]}: {len(units)} candidate units, {NPROC} processes, '
          f'time limit {TLIMIT:.0f} s, {time.strftime("%c")}', flush=True)
    cnt = {}
    with mp.Pool(NPROC) as pool:
        for uid, st, dt in pool.imap_unordered(work, units):
            k = st.split()[0]
            cnt[k] = cnt.get(k, 0) + 1
            if k in ('done', 'FAIL'):
                print(f'{uid} {st} {dt:.1f} s (elapsed {time.time() - T0:.0f} s; {cnt})', flush=True)
    print(f'WORKER END {JOB}: {cnt}, {time.time() - T0:.0f} s, {time.strftime("%c")}', flush=True)
    sys.exit(1 if cnt.get('FAIL') else 0)
