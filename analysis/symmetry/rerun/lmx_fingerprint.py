#!/usr/bin/env python3
"""The fingerprints of the rerun: no result of another submission, of other code or data, or of the published
run is taken for a result of this rerun (E3, and F1-F7 of round B6).

A scoring (one line of specs/*.txt, run by score_p2_lmx.sbatch) has the fingerprint md5(code md5, spec line, run
id). The code md5 covers the scoring code (CODE, beam_photons.py among it) and output/bundle_rule.json, the record
of the refreshed bundles and of the beam-photon rule they were made with (the data part, F5). The spec line is its
KEY=VALUE pairs in the order of SPEC_KEYS. The run id is that of the start mark output/.lmx_rerun_start, which
submit_scoring_lmx.sh writes once and keeps on RESUME=1. The fingerprint goes into the result (key lmx_fingerprint;
in an r2_ladder.py file, {stage: result}, into its one stage), with the md5 of the files the scoring wrote beside it
(its refitted temperature, the weight dumps of the wide scan) and of the temperature it read. Every other step of
the scoring side removes its outputs when it starts and records them when it ends, in output/lmx_records/<step>.json,
with the run id and the md5 of all the scoring side's code (files_score.txt) and of output/bundle_rule.json.

    python -B symmetry/rerun/lmx_fingerprint.py precheck        # the particle caches, bundle_rule.json and its bundles (F3)
    python -B symmetry/rerun/lmx_fingerprint.py mark            # precheck, then write the mark (RESUME: compare)
    python -B symmetry/rerun/lmx_fingerprint.py status          # the mark and what changed since; writes nothing
    python -B symmetry/rerun/lmx_fingerprint.py fp              # '<fingerprint> <code md5>' of the line in the environment
    python -B symmetry/rerun/lmx_fingerprint.py keep  <result> <fp>               # exit 0 if the result is kept
    python -B symmetry/rerun/lmx_fingerprint.py stamp <result> <fp> <code md5>    # write it into the result
    python -B symmetry/rerun/lmx_fingerprint.py cachedir <folder>  # <folder>/<run id>_<md5(r2_ladder.py)[:12]> (F2)
    python -B symmetry/rerun/lmx_fingerprint.py record <step> <file>...           # the record of a step's outputs
    python -B symmetry/rerun/lmx_fingerprint.py has-record <step>                 # exit 0 if this rerun has it
    python -B symmetry/rerun/lmx_fingerprint.py check <file>...    # exit 1 listing each file without the fingerprint
    python -B symmetry/rerun/lmx_fingerprint.py check --all [--write]   # every scoring result; --write: scoring_check.json
    python -B symmetry/rerun/lmx_fingerprint.py outputs            # what the rerun's jobs write in output/ (setup_lmx.py --park)

Run from RUN (MAXENT_DIR); SPEC_KEYS comes from common_lmx.sh (or is read from it). A refusal exits 2 with a message.
kscan_collect.py and widebox_collect.py import check_files(); setup_lmx.py imports outputs().
"""
import glob, hashlib, json, os, re, sys, time

MARK = 'output/.lmx_rerun_start'
RULE = 'output/bundle_rule.json'            # I1, written by the observables task's v3 cache step
RECORDS = 'output/lmx_records'
CHECK_OUT = 'output/scoring_check.json'     # I4, written by collect_lmx at the end of the scoring side
CODE = ('r2_ladder.py', 'stage_mixture.py', 'mixture_cfg.py', 'score_obs.py', 'beam_photons.py',
        'symmetry/rerun/score_p2_lmx.sbatch', 'symmetry/rerun/common_lmx.sh')
FILES = 'symmetry/rerun/files_score.txt'    # every file of the working tree the scoring jobs read (I3)
SPECS = 'symmetry/rerun/specs'
START_OVER = f'remove {MARK} to start over'
RESTART_DATA = ('any rewrite of the _nb folders, of the refreshed bundles or of the designs after the mark needs the '
                f'mark removed ({START_OVER})')

# ---- what the steps of the scoring side write (besides the scorings of specs/*.txt), from their batch scripts
LC4 = ('MIXGEOV3E160S9', 'MIXGEOV3E320S9', 'MIXGEOV4S9', 'MIXGEOV4L')
WIDTH_TAGS = ('MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3M8S9', 'MIXGEOV3S27F', 'MIXGEOV3E20S9',
              'MIXGEOV3E160S9', 'MIXGEOV3E320S9')
WIFI_TAGS = ('MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3M8S9')
TOYS = ((48, 0), (48, 1), (48, 2), (1, 0), (2, 0), (3, 0), (4, 0), (8, 0), (16, 0))
STEPS = {   # step: (its batch script, the files it writes)
    'recal': ('recal_lc_lmx.sbatch', [f'output/recal_T_{t}.json' for t in LC4] + [f'output/temperature/{t}.json' for t in LC4]
              + ['output/recal_T_MIXGEOV3S9_check.json']),
    'widths': ('strict_closure_lmx.sbatch', [f'output/widths_v2_{t}{h}.json' for t in WIDTH_TAGS for h in ('', 'h400')]),
    'rescorev4': ('rescore_v4_lmx.sbatch', [f'output/widths_v2_{t}{h}r415.json' for t in ('MIXGEOV4S9', 'MIXGEOV4L')
                                            for h in ('', 'h400')]),
    'stagec': ('stagec_lmx.sbatch', ['output/valid_map_C_P2augfullfac0_report.json',
                                     'output/single_members_C_P2augfullfac0.json', 'output/xp_null.json']),
    **{f'wifi_{t}': ('wifi_cov_lmx.sbatch', [f'output/wifi_{t}_basis_rebuilt.npz', f'output/wifi_{t}_fit.npz',
                                            f'output/wifi_{t}_fit.json', f'output/wifi_{t}_diag.json',
                                            f'output/wifi_coverage_{t}.json']
                     + ([f'output/wifi_gensplit_{t}.json'] if t == 'MIXGEOV3S9' else [])) for t in WIFI_TAGS},
    'closmix': ('closure_mix_lmx.sbatch', ['output/fig_closure_mix.npz']),
    'toy': ('toy_final_lmx.sbatch', [f'output/toy_final_K{k}_s{s}.{e}' for k, s in TOYS for e in ('json', 'npz')]),
    'counts': ('trunc_lmx.sbatch', ['output/truncation_fraction.json', 'output/param_count.json']),
    'collect': ('collect_lmx.sbatch', ['output/rank_scan_final.json', 'output/widebox_wb18.json',
                                       'output/wb18_weights_single_0.083.npy', 'output/wb18_weights_union_0.083.npy']),
}
# the files of output/ the propagation side's jobs write (RERUN_PROP.md and its batch scripts: envelope_test,
# three_bands, band_checks, theory_variations, calc_bins, calc_vs_delphi, check_linear_mix, the benchmarks,
# make_numbers, figs_r8, plot_envelope, fig_cost, pfn_lmx): every name ends in _lin, but for these
PROP_OUTPUTS = ('output/*_lin.*', 'output/calc_bins_nlo_point.json', 'output/check_linear_mix.json',
                'output/numbers/**', 'output/figs/**', 'output/crosscheck_seeds/**', 'output/crosscheck_seeds_summary.json')
# the names make_numbers.py's pick() reads before the rerun's own result of the same training (the p2aug_* fallback,
# which the specs write for Stage B factorized s0-s2 and Stage C s2 of both networks): no job writes them, so a file
# of such a name in RUN/output is a stray copy (setup_lmx.py --park moves it aside)
PICK_FIRST = tuple(f'output/p2augfull{t}_{c}_s{k}.json' for t in ('', 'strict')
                   for c, ks in (('B_fac', (0, 1, 2)), ('C_fac', (2,)), ('C_cat', (2,))) for k in ks)


class Stop(Exception):
    """A clear refusal: the command line prints it and exits 2, without a traceback."""


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def md5_of(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def need_rule():
    if not os.path.exists(RULE):
        raise Stop(f'missing {RULE}: run the v3 cache step of the observables task first')


def code():
    """({file: md5} of the scoring code and of output/bundle_rule.json, the md5 of that dict)"""
    need_rule()
    files = {f: md5(f) for f in CODE}
    files[RULE] = md5(RULE)
    return files, md5_of(files)


def expand(listfile, root='.'):
    """The files a list names (one path or glob per line, relative to root, # comments), sorted, without bytecode"""
    out = []
    for line in open(listfile):
        p = line.split('#', 1)[0].strip()
        if not p:
            continue
        files = []
        for h in sorted(glob.glob(os.path.join(root, p), recursive=True)):
            files += [os.path.join(d, f) for d, _, fs in os.walk(h) for f in fs] if os.path.isdir(h) else [h]
        files = [os.path.relpath(f, root) for f in files
                 if '__pycache__' not in f.split(os.sep) and not f.endswith('.pyc') and os.path.basename(f) != '.DS_Store']
        if not files:
            raise Stop(f'{listfile}: nothing matches {p}')
        out += files
    return sorted(set(out))


def allcode():
    """({file: md5} of every file of files_score.txt and of bundle_rule.json, its md5): the code of the other steps"""
    need_rule()
    files = {f: md5(f) for f in expand(FILES)}
    files[RULE] = md5(RULE)
    return files, md5_of(files)


def spec_keys():
    if os.environ.get('SPEC_KEYS'):
        return os.environ['SPEC_KEYS'].split()
    m = re.search(r'^export SPEC_KEYS="([^"]+)"', open('symmetry/rerun/common_lmx.sh').read(), re.M)
    return m.group(1).split()


def spec(env=None):
    """The spec line of a scoring, from its environment (score_p2_lmx.sbatch) or from a parsed line"""
    env = os.environ if env is None else env
    return ' '.join(f'{k}={env[k]}' for k in spec_keys() if env.get(k))


def spec_lines():
    """Every scoring of specs/*.txt, as {KEY: value} (and 'file', its spec file)"""
    out = []
    for f in sorted(glob.glob(f'{SPECS}/*.txt')):
        for line in open(f):
            if line.strip() and not line.lstrip().startswith('#'):
                d = dict(kv.split('=', 1) for kv in line.split())
                d['file'] = f
                out.append(d)
    return out


def read_mark():
    """The start mark; a missing or unreadable one is a clear refusal (F7)"""
    if not os.path.exists(MARK):
        raise Stop(f'no {MARK}: submit with symmetry/rerun/submit_scoring_lmx.sh, which writes it')
    try:
        m = json.load(open(MARK))
        m['run_id'], m['code'], m['code_md5']
        return m
    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError) as e:
        raise Stop(f'{MARK} is not a readable mark ({e.__class__.__name__}): {START_OVER}')


def run_id():
    return read_mark()['run_id']


def fingerprint(c, s, rid):
    return hashlib.md5(f'{c}\n{s}\n{rid}'.encode()).hexdigest()


def entry(r):
    """the result itself, as make_numbers.load reads it: the top level, or the one stage of r2_ladder.py"""
    return r if 'master' in r else list(r.values())[0]


def stamp_of(path):
    """The lmx_fingerprint written into a result, or None"""
    try:
        e = entry(json.load(open(path)))
        fp = e.get('lmx_fingerprint') if isinstance(e, dict) else None
        return fp if isinstance(fp, dict) else None
    except (OSError, ValueError, UnicodeDecodeError, AttributeError, IndexError, TypeError):
        return None


def model_key(path):
    """as score_obs.model_key: the name of the network file a path resolves to, without its suffix"""
    b = os.path.basename(os.path.realpath(path))
    for s in ('_cond.npz', '_trunk.pt', '_concat.pt'):
        if b.endswith(s):
            return b[:-len(s)]
    raise Stop(f'{path}: not a model file')


def written_by_scoring(env):
    """({file: md5} the scoring in `env` wrote beside its result, {file: md5} of the temperature it read)"""
    tf = os.path.join(env.get('T_DIR') or 'output/temperature', f'{model_key(env["MODEL"])}.json')
    outputs, inputs = {}, {}
    if env.get('TMODE') == 'refit':
        if not os.path.exists(tf):
            raise Stop(f'{env["OUT"]}: the refit wrote no {tf}')
        outputs[tf] = md5(tf)
    elif env.get('TMODE') == 'table':
        inputs[tf] = md5(tf)
    if env.get('DUMP') == '1':
        dumps = sorted(glob.glob(f'output/models/{env["STAGE"]}{env.get("SUFFIX", "")}_w_*.npy'))
        if not dumps:
            raise Stop(f'{env["OUT"]}: DUMP=1, but the scoring wrote no weight dump')
        outputs.update({p: md5(p) for p in dumps})
    return outputs, inputs


def bundle_rule():
    """RUN's bundle_rule.py (I2), loaded from the file"""
    import importlib.util
    s = importlib.util.spec_from_file_location('bundle_rule', os.path.join(os.getcwd(), 'bundle_rule.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


# ---- F3: the particle caches and the refreshed bundles must follow the rule in RUN
def precheck():
    problems, rule = [], md5('beam_photons.py')
    caches = sorted(glob.glob('output/pobs/*.npz'))
    if caches:
        import numpy as np
        for f in caches:
            try:
                with np.load(f) as z:
                    made = str(z['beam_photons_md5']) if 'beam_photons_md5' in z.files else 'not recorded'
            except Exception as e:
                made = f'unreadable ({e.__class__.__name__})'
            if made != rule:
                problems.append(f'{f} records beam_photons.py md5 {made}, not {rule} (RUN/beam_photons.py): '
                                f'move output/pobs aside (mv output/pobs output/pobs_old); the caches are made again')
    if not os.path.exists(RULE):
        problems.append(f'missing {RULE}: run the v3 cache step first (the observables task)')
    else:
        try:
            have = json.load(open(RULE)).get('beam_photons_md5')
        except (ValueError, UnicodeDecodeError, AttributeError) as e:
            have = f'unreadable ({e.__class__.__name__})'
        if have != rule:
            problems.append(f'{RULE} records beam_photons.py md5 {have}, not {rule} (RUN/beam_photons.py): '
                            f'run the v3 cache step first')
        elif not os.path.exists('bundle_rule.py'):
            problems.append('missing RUN/bundle_rule.py (sync_run.sh copies it)')
        else:       # every bundle the table lists: there, under its real path, with the rule of RUN (I2, no hashes)
            problems += [f'{RULE}: {b}' for b in bundle_rule().verify(os.getcwd(), hash=False)]
    if problems:
        raise Stop('\n'.join(['the rerun cannot start, nothing is submitted:'] + ['  ' + p for p in problems]))
    print(f'precheck: {len(caches)} particle caches and {RULE} follow beam_photons.py {rule}')


def changed_since(m, files):
    return sorted(f for f in set(files) | set(m['code']) if m['code'].get(f) != files.get(f))


def mark():
    """Write the mark, or with one (RESUME=1) compare the code with it; refuse a rewrite of the bundles (F5)"""
    if os.path.exists(MARK):
        read_mark()
    precheck()
    files, c = code()
    if os.path.exists(MARK):
        m = read_mark()
        changed = changed_since(m, files)
        if RULE in changed:
            raise Stop(f'{RULE} changed since the mark of run {m["run_id"]}: the refreshed bundles were rewritten after '
                       f'the rerun started; {RESTART_DATA}')
        print(f'{MARK}: run {m["run_id"]} of {m["written"]}, code {m["code_md5"]}; code now {c}, ' +
              (f'changed since the mark: {", ".join(changed)} (every scoring is done again)' if changed else
               'unchanged (a scoring whose result carries its fingerprint is kept)'))
        print(f'RESUME=1 keeps the mark and every result with its fingerprint; {RESTART_DATA}')
    else:
        m = dict(run_id=f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}', written=time.strftime('%c'), code_md5=c, code=files)
        json.dump(m, open(MARK + '.tmp', 'w'), indent=1); os.replace(MARK + '.tmp', MARK)
        print(f'wrote {MARK}: run {m["run_id"]}, code {c}')


def status():
    m = read_mark()
    try:
        files, c = code()
        changed = changed_since(m, files)
        now = f'code now {c}' + (f', changed since the mark: {", ".join(changed)}' if changed else ', unchanged')
    except Stop as e:
        now = str(e)
    print(f'{MARK}: run {m["run_id"]} of {m.get("written", "?")}, code {m["code_md5"]}; {now}')


# ---- F1: the outputs of the rerun's jobs, and their check
def outputs():
    """[(pattern, writer)]: every file of output/ a job of the rerun writes, on either side"""
    L = []
    for d in spec_lines():
        L.append((d['OUT'], f'score_p2_lmx.sbatch ({os.path.basename(d["file"])})'))
        if d.get('DUMP') == '1':
            L.append((f'output/models/{d["STAGE"]}{d.get("SUFFIX", "")}_w_*.npy',
                      f'score_p2_lmx.sbatch (weight dump of {d["OUT"]})'))
    L.append(('output/temperature/*.json', 'score_p2_lmx.sbatch (TMODE=refit) and recal_lc_lmx.sbatch'))
    for step, (job, files) in STEPS.items():
        L += [(f, job) for f in files]
    L += [(f'{RECORDS}/*.json', 'every step of the scoring side (its record)'), (CHECK_OUT, 'collect_lmx.sbatch'),
          ('output/precision_gap_r8.json', 'table_lmx.sbatch (release phase)')]
    L += [(p, 'the propagation side (RERUN_PROP.md)') for p in PROP_OUTPUTS]
    L += [(p, 'no job: make_numbers.py would read it before the rerun\'s result (the p2aug_* fallback)') for p in PICK_FIRST]
    return L


def vouched(rid, c, a):
    """{file: (md5, who)}: the files the stamps of this rerun's scorings and the records of its steps vouch for"""
    v = {}
    for d in spec_lines():
        fp = stamp_of(d['OUT'])
        if fp and fp.get('run_id') == rid and fp.get('code_md5') == c:
            v.update({os.path.normpath(p): (m, d['OUT']) for p, m in (fp.get('outputs') or {}).items()})
    for f in sorted(glob.glob(f'{RECORDS}/*.json')):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        if r.get('run_id') == rid and r.get('code_md5') == a:
            v.update({os.path.normpath(p): (m, f) for p, m in r.get('files', {}).items()})
    return v


def check_files(paths):
    """[(file, reason)] for each file of `paths` whose fingerprint is missing, of another mark, or of other code than
    RUN's: a published copy, a result of another submission or code, a file changed after its step recorded it"""
    rid = run_id(); _, c = code()
    bad, v = [], None
    for p in (os.path.normpath(p) for p in paths):
        if not os.path.exists(p):
            bad.append((p, 'missing')); continue
        fp = stamp_of(p) if p.endswith('.json') else None
        if fp is not None:
            if fp.get('run_id') != rid:
                bad.append((p, f'its fingerprint is of run {fp.get("run_id")}, not of this mark ({rid})'))
            elif fp.get('code_md5') != c:
                bad.append((p, f'its fingerprint is of code {fp.get("code_md5")}, not of the code in RUN ({c})'))
            continue
        if v is None:
            v = vouched(rid, c, allcode()[1])
        if p not in v:
            bad.append((p, 'no fingerprint of this rerun (a published copy, or a step of another mark or code)'))
        elif md5(p) != v[p][0]:
            bad.append((p, f'changed since {v[p][1]} recorded it'))
    return bad


def check_all(write=False):
    """Every result of the scoring side: the scorings of specs/*.txt (each with the fingerprint of its own line),
    what they wrote beside them, and the record of every step. With write, output/scoring_check.json (I4), only
    when all of it passes"""
    rid = run_id(); _, c = code(); _, a = allcode()
    bad, ok, inputs = [], {}, {}
    for d in spec_lines():
        out = d['OUT']
        if not os.path.exists(out):
            bad.append((out, 'missing')); continue
        fp = stamp_of(out)
        if fp is None:
            bad.append((out, 'no fingerprint')); continue
        if fp.get('run_id') != rid or fp.get('code_md5') != c:
            bad.append((out, f'fingerprint of run {fp.get("run_id")} and code {fp.get("code_md5")}, not {rid} and {c}')); continue
        if fp.get('md5') != fingerprint(c, spec(d), rid):
            bad.append((out, 'its fingerprint is not that of its spec line')); continue
        ok[out] = md5(out)
        for p, h in (fp.get('outputs') or {}).items():
            if not os.path.exists(p) or md5(p) != h:
                bad.append((p, f'written by the scoring of {out}, changed or missing since'))
            else:
                ok[os.path.normpath(p)] = h
        inputs.update({os.path.normpath(p): (h, out) for p, h in (fp.get('inputs') or {}).items()})
    for step, (job, want) in STEPS.items():
        f = f'{RECORDS}/{step}.json'
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            bad.append((f, f'missing or unreadable: {job} did not end well in this rerun')); continue
        if r.get('run_id') != rid or r.get('code_md5') != a:
            bad.append((f, f'of run {r.get("run_id")} and code {r.get("code_md5")}, not {rid} and {a}')); continue
        for p in sorted(set(want) - set(r.get('files', {}))):
            bad.append((p, f'not recorded by {job}'))
        for p, h in r.get('files', {}).items():
            if not os.path.exists(p) or md5(p) != h:
                bad.append((p, f'recorded by {job}, changed or missing since'))
            else:
                ok[os.path.normpath(p)] = h
    for p, (h, out) in sorted(inputs.items()):     # the temperature a strict test read is the one its refit wrote
        if ok.get(p) != h:
            bad.append((p, f'read by the scoring of {out} as {h}, which is not the refit of this rerun now'))
    for p, why in bad:
        print(f'  {p}: {why}')
    if bad:
        print(f'SCORING CHECK FAILED: {len(bad)} problems' + (f'; {CHECK_OUT} not written' if write else ''))
        return 1
    if write:
        tmp = f'{CHECK_OUT}.tmp{os.getpid()}'
        json.dump({'mark_run_id': rid, 'code_md5': c, 'files': dict(sorted(ok.items()))}, open(tmp, 'w'), indent=1)
        os.replace(tmp, CHECK_OUT)
    print(f'scoring check: {len(ok)} files carry the fingerprint of run {rid}, code {c}' + (f' -> {CHECK_OUT}' if write else ''))
    return 0


def record(step, files):
    """The record of a step that ended well: its outputs with their md5, the run id and the code md5"""
    if step in STEPS:
        missing = sorted(set(STEPS[step][1]) - {os.path.normpath(f) for f in files})
        if missing:
            raise Stop(f'{step}: record every output of the step; {", ".join(missing)} not given')
    gone = [f for f in files if not os.path.exists(f)]
    if gone:
        raise Stop(f'{step}: the step did not write {", ".join(gone)}')
    rid = run_id(); _, a = allcode()
    r = dict(step=step, run_id=rid, code_md5=a, written=time.strftime('%c'),
             files={os.path.normpath(f): md5(f) for f in files})
    os.makedirs(RECORDS, exist_ok=True)
    f = f'{RECORDS}/{step}.json'; tmp = f'{f}.tmp{os.getpid()}'
    json.dump(r, open(tmp, 'w'), indent=1); os.replace(tmp, f)
    print(f'=== {f}: {len(files)} files of run {rid}, code {a}')


def has_record(step):
    try:
        return json.load(open(f'{RECORDS}/{step}.json')).get('run_id') == run_id()
    except (OSError, ValueError, AttributeError):
        return False


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ''
    if cmd == 'precheck':
        precheck()
    elif cmd == 'mark':
        mark()
    elif cmd == 'status':
        status()
    elif cmd == 'fp':
        _, c = code()
        print(fingerprint(c, spec(), run_id()), c)
    elif cmd == 'keep':
        out, fp = argv[2:4]
        if not os.path.exists(out):
            return 1
        try:
            have = entry(json.load(open(out))).get('lmx_fingerprint', {}).get('md5')
        except (ValueError, UnicodeDecodeError, AttributeError, IndexError, TypeError):
            print(f'=== {out} exists but is not a complete result: scored again'); return 1
        if have != fp:
            print(f'=== {out} exists with fingerprint {have}, not {fp}: scored again'); return 1
        print(f'=== {out} carries the fingerprint of this scoring ({fp}): kept')
    elif cmd == 'stamp':
        out, fp, c = argv[2:5]
        outs, ins = written_by_scoring(os.environ)
        r = json.load(open(out))
        entry(r)['lmx_fingerprint'] = dict(md5=fp, code_md5=c, spec=spec(), run_id=run_id(), outputs=outs, inputs=ins)
        json.dump(r, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
        print(f'=== {out}: fingerprint {fp}' + (f', with {", ".join(outs)}' if outs else ''))
    elif cmd == 'cachedir':
        print(f'{argv[2].rstrip("/")}/{run_id()}_{md5("r2_ladder.py")[:12]}')
    elif cmd == 'record':
        record(argv[2], argv[3:])
    elif cmd == 'has-record':
        if not has_record(argv[2]):
            print(f'{RECORDS}/{argv[2]}.json: no record of run {run_id()} (the step has not ended well in this rerun)')
            return 1
    elif cmd == 'check':
        if argv[2:3] == ['--all']:
            return check_all(write='--write' in argv[3:])
        bad = check_files(argv[2:])
        for p, why in bad:
            print(f'  {p}: {why}')
        print(f'{len(argv) - 2 - len(bad)} of {len(argv) - 2} files carry the fingerprint of this rerun')
        return 1 if bad else 0
    elif cmd == 'outputs':
        for p, w in outputs():
            print(f'{p}\t{w}')
    else:
        raise Stop(f'unknown command {cmd!r} (see the docstring)')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except Stop as e:
        print(e, file=sys.stderr)
        sys.exit(2)
