#!/usr/bin/env python3
"""Beam-photon rule v3 (beam_photons.py, RULE_VERSION 3) in every _nb CSV. Only the rows whose mask
differs from v2 (B0's rule, beam_photons_v2.py, md5 77d8092b...) are rewritten, from the ORIGINAL CSV as
B0 did (nb2_units.py):
  1_minus_thrust, B_total, rho_heavy = stored - (with - without), stored = the original CSV value,
      with/without = the rebuilt event with and without the v3 beam photons (nb_core: parts, scale_of,
      four, three); fallback: the stored value where |with - stored| > 1e-2 or the result is negative
      (or either is not finite);
  mult_total = stored - n_beam(v3);
  a row left without a beam photon gets its original text back; every other column keeps the
  original text. All other rows keep the text of the current _nb CSV, byte for byte.
A CSV with a changed row is written to a temp file in its own folder and renamed onto the _nb CSV;
a unit without one is only recorded. The destination must be a regular file directly inside one of the
folders of NB_FOLDERS.txt (the photon-free mirrors Phase A made): any other is refused. Every listed folder
must be its own real path (no link anywhere on the way), else nothing starts.

    python nb3_units.py WORK NPROC MODE [UIDS]
MODE 'run':  dynamic workers over WORK/units.tsv (claims in WORK/claims), replace the _nb CSVs.
MODE 'test': the units named in UIDS (comma list, all if empty); WORK/test/<uid>.{csv,json}, nothing
             replaced. NB_FULL=1 (test only): every row with a v3 beam photon recomputed from the
             original CSV, the current _nb CSV not used: the full recomputation the delta must equal.
Restart-safe as nb2_units.py: WORK/units/<uid>.json marks a unit done; the record goes first to
WORK/pending/<uid>.json with the md5 of the new text, so a unit interrupted after its rename is
recognised on restart and not computed from its own output.
Input check: the current _nb CSV has the header and rows of the original, mult_total = stored - n_beam(v2)
in every row whose mask does not change, and the original text in every row without a v2 beam photon;
a changed row may already hold its v3 text (a re-run), which is recorded.
"""
import os, sys, time, json, hashlib, traceback
import numpy as np

WORK, NPROC, MODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
UIDS = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nb_core as C                     # noqa: E402  Phase A's rebuild (md5 1ed8468b...)
import beam_photons as BP               # noqa: E402  v3
import beam_photons_v2 as BP2           # noqa: E402  B0's v2 (md5 77d8092b...)

assert BP.RULE_VERSION == 3 and not hasattr(BP2, 'RULE_VERSION'), 'beam_photons.py must be v3, _v2 must be v2'
NBF = {l.strip() for l in open(f'{HERE}/NB_FOLDERS.txt') if l.strip() and not l.startswith('#')}
_LINKED = sorted(d for d in NBF if os.path.realpath(d) != d)
if _LINKED:     # a link in a listed folder's path would make its target writable
    sys.exit(f'NB_FOLDERS.txt: {len(_LINKED)} folders are not their own real path (a link on the way), e.g. '
             f'{_LINKED[0]} -> {os.path.realpath(_LINKED[0])}: nothing done')
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
MISS_TOL = 1e-2
THRUST_EDGES = np.array([0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .14, .16, .18, .20,
                         .25, .30, .35, .40, .50])
FULL = os.environ.get('NB_FULL', '0') == '1'
TLIMIT = float(os.environ.get('NB_TLIMIT', '1e9'))
T0 = time.time()
JOB = os.environ.get('SLURM_JOB_ID', 'local')
RST = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
if FULL and MODE != 'test':
    sys.exit('NB_FULL=1 is for test mode only')


def hist(v, e):
    return [int((v < e[0]).sum())] + np.histogram(v, e)[0].tolist() + [int((v >= e[-1]).sum())]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def split_csv(text):
    lines = text.split('\n')
    trail = lines[-1] == ''
    return lines[0], (lines[1:-1] if trail else lines[1:]), trail


def parse(body, cols):
    out = np.empty((len(body), len(cols)))
    for i, l in enumerate(body):
        q = l.split(',')
        out[i] = [float(q[c]) for c in cols]
    return out


def photons(P, i, js, s):
    return [dict(j=int(j), E=float(P[i, j, 0]) * s, cos=float(P[i, j, 1]), phi=float(P[i, j, 2])) for j in js]


def why(i, I2, I3, F2, F3):
    """the v3 rule behind a changed event"""
    r = []
    if I3['n_axis_soft'][i] and (F3[i] & ~F2[i]).any():
        r.append('i_soft_on_axis')
    if I3['n_lone'][i]:
        r.append('ii_lone')
    if I3['n_conflict'][i] or I2['n_dropped'][i]:
        r.append('iii_tie' if I3['tie'][i] else 'iii_group')
    return '+'.join(r) or 'other'


def pair_hemis(P, M, F, base, i, s):
    """two-photon hemispheres of event i whose two particles are both base-flagged and disagree by
    more than C_GAP: [(a, b)] with a the harder"""
    out = []
    ct = P[i, :, 1]
    for sg in (1, -1):
        h = np.where(M[i] & (np.sign(ct) == sg))[0]
        if len(h) == 2 and base[i, h].all():
            e = P[i, h, 0].astype(np.float64) * s
            c = BP._implied_cos(ct[h].astype(np.float64), P[i, h, 2].astype(np.float64), e)
            if abs(c[0] - c[1]) > BP.C_GAP:
                a, b = (h[0], h[1]) if e[0] >= e[1] else (h[1], h[0])
                out.append((int(a), int(b)))
    return out


def diagnose(P, M, F3, I3):
    """the decisions the stored features cannot check, for the summary: (iii) pair hemispheres (two
    photons alone in a hemisphere that disagree), decided by (ii) (the harder on the axis, the hardest
    particle of a whole event, no missing momentum, the softer's transverse momentum relative to it below
    1e-6 of its energy: ratio), by a witness, by the hardest photon or not at all; every decision of (ii),
    with its missing momentum, ratio, largest companion and companions stored at cos theta = +-1, and the
    photons it would decide in truncated events, which it leaves out; tie events outside such pairs;
    flags left that disagree within C_GAP (the gray zone, kept); hard photons alone in their hemisphere
    on the axis, by phi (0/pi, +-pi/2, other) and event class"""
    ct = P[..., 1].astype(np.float64); ph = P[..., 2].astype(np.float64)
    neu = M & (P[..., 4] == 0) & (P[..., 5] == 0) & (np.abs(P[..., 3].astype(np.float64) + 3.0) < BP.LM_TOL)
    base = neu & (np.abs(np.abs(ph) - np.pi / 2) < BP.PHI_TOL)
    out = dict(pairs=[], ties_other=[], alone={})
    # pair hemispheres: only events where (ii) or (iii) acted, or a two-photon hemisphere is all flagged
    npos, nneg = (M & (ct > 0)).sum(1), (M & (ct < 0)).sum(1)
    two = (((npos == 2)[:, None] & (ct > 0)) | ((nneg == 2)[:, None] & (ct < 0))) & base
    cand = np.where((I3['n_lone'] > 0) | (I3['n_conflict'] > 0) | (two.sum(1) >= 2))[0]
    pair_rows = set()
    for i in cand:
        s, miss, pm = BP._balance(P[i:i + 1], M[i:i + 1])
        s, miss, pm = float(s[0]), float(miss[0]), pm[0]
        for a, b in pair_hemis(P, M, F3, base, i, s):
            Ea, Eb = float(P[i, a, 0]) * s, float(P[i, b, 0]) * s
            ratio = BP._hemi_pt(P[i], M[i], a, pm) / Ea
            ii = bool(abs(ct[i, a]) >= BP.COS_AXIS and miss < BP.P_MISS and Ea >= pm.max() and ratio < BP.PT_REL
                      and not M[i].all())
            dec = ('ii' if ii else 'tie' if I3['tie'][i] else 'witness' if I3['n_conflict'][i] else 'none')
            pair_rows.add(int(i))
            out['pairs'].append(dict(row=int(i), decided=dec, other_flags=int(base[i].sum()) - 2, hardest_kept=bool(F3[i, a]),
                                     softer_kept=bool(F3[i, b]), E_hard=Ea, E_soft=Eb, both_gt_1MeV=bool(Eb > 1e-3),
                                     ratio=float(ratio), on_axis=bool(abs(ct[i, a]) >= BP.COS_AXIS), miss=miss,
                                     hardest_particle=bool(Ea >= pm.max()), truncated=bool(M[i].all())))
    # every decision of (ii), and the photons it would decide in a truncated event (left out, so flagged
    # unless (iii) drops them): the missing momentum, the counted transverse momentum of the rest of the
    # hemisphere over E (ratio), its largest single part, and the parts stored at cos theta = +-1
    out['ii'], out['ii_truncated'] = [], []
    on = base & (np.abs(ct) >= BP.COS_AXIS)
    whole = M.sum(1) < M.shape[1]
    for i in np.where(on.any(1) & ((I3['n_lone'] > 0) | ~whole))[0]:
        s, miss, pm = BP._balance(P[i:i + 1], M[i:i + 1])
        s, miss, pm = float(s[0]), float(miss[0]), pm[0]
        for j in np.where(on[i])[0]:
            E = float(P[i, j, 0]) * s
            if miss >= BP.P_MISS or E < pm.max():
                continue
            parts, dead = BP._hemi_parts(P[i], M[i], j, pm)
            if parts.sum() >= BP.PT_REL * E:
                continue
            (out['ii'] if whole[i] else out['ii_truncated']).append(
                dict(row=int(i), E=E, miss=miss, ratio=float(parts.sum() / E), largest=float(parts.max() / E) if len(parts) else 0.0,
                     n_hemi=int(len(parts)), n_cos1=int(dead.sum()), kept=bool(F3[i, j])))
    # tie events of (iii) outside two-photon pair hemispheres (flags in different hemispheres, or in a
    # hemisphere with more particles): the hardest photon decided
    for i in np.where(I3['tie'])[0]:
        if int(i) in pair_rows:
            continue
        j = np.where(base[i])[0]
        out['ties_other'].append(dict(row=int(i), n_flags=int(len(j)), flags_pos=int((ct[i, j] > 0).sum()),
                                      hemi_sizes=[int(npos[i]), int(nneg[i])], kept=int(F3[i, j].sum()),
                                      E=[float(x) for x in P[i, j, 0].astype(np.float64) * BP.ECM]))
    # the gray zone of (iii): flags left that disagree beyond the tolerance but within C_GAP (kept)
    left = F3 & base
    e91 = P[..., 0].astype(np.float64) * BP.ECM
    c = BP._implied_cos(ct, ph, e91)
    spread = np.where(left, c, -np.inf).max(1) - np.where(left, c, np.inf).min(1)
    shift = 1e-12 / np.where(left, np.maximum(e91, 1e-300), np.inf).min(1)
    i = np.where((left.sum(1) >= 2) & (spread + shift > BP.C_TOL))[0]
    out['gray'] = []
    if len(i):
        s = BP._balance(P[i], M[i])[0]
        e = P[i, :, 0].astype(np.float64) * s[:, None]
        cs = BP._implied_cos(ct[i], ph[i], e)
        tol = np.where(e < BP.E_SOFT, BP.C_TOL_SOFT, BP.C_TOL)
        li = left[i]; hard = li & (tol == BP.C_TOL)
        sp_all = np.where(li, cs, -np.inf).max(1) - np.where(li, cs, np.inf).min(1)
        sp_hard = np.where(hard, cs, -np.inf).max(1) - np.where(hard, cs, np.inf).min(1)
        for k in np.where((sp_all > BP.C_TOL_SOFT) | (sp_hard > BP.C_TOL))[0]:
            j = np.where(li[k])[0]
            out['gray'].append(dict(row=int(i[k]), spread=float(sp_all[k]), E=[float(x) for x in e[k, j]]))
    # hard photons (E > 1 GeV) alone in their hemisphere at |cos| > 1 - 1e-6
    al = neu & (np.abs(ct) > 1 - 1e-6) & (P[..., 0].astype(np.float64) * BP.ECM > 1.0) & \
        (((ct > 0) & (npos == 1)[:, None]) | ((ct < 0) & (nneg == 1)[:, None]))
    for i, j in zip(*np.where(al)):
        pc = 'axis' if (P[i, j, 2] == 0 or abs(ph[i, j]) == BP.PI32) else 'plane' if base[i, j] else 'other'
        ec = 'base' if base[i].any() else ('rule_i' if I3['rule_i'][i] else 'none')
        miss = float(BP._balance(P[i:i + 1], M[i:i + 1])[1][0])
        k = f'{ec}|{pc}|flag{int(F3[i, j])}|miss{"<" if miss < BP.P_MISS else ">="}1e-4'
        out['alone'][k] = out['alone'].get(k, 0) + 1
    return out


def compute(u):
    """new text and statistics of one unit (nothing written)"""
    t0 = time.time()
    z = np.load(u['npz'])
    P, M = z['particles'], z['mask'].astype(bool)
    N, PMAX = M.shape
    text_in = open(u['csv_in']).read()
    head, body, trail = split_csv(text_in)
    if FULL:
        text_cur, body_cur = None, body
    else:
        text_cur = open(u['csv_out']).read()
        head_c, body_cur, trail_c = split_csv(text_cur)
        if head_c != head or trail_c != trail or len(body_cur) != len(body):
            raise RuntimeError(f"{u['uid']}: the _nb CSV has another header or row count")
    if len(body) != N:
        raise RuntimeError(f"{u['uid']}: rows {len(body)} / npz {N}")
    hl = head.split(',')
    col = [hl.index(o) for o in OBS]
    cm = hl.index('mult_total')
    t_load = time.time() - t0
    F3, I3 = BP.beam_photon_mask(P, M, detail=True)
    F2, I2 = BP2.beam_photon_mask(P, M, detail=True)
    nb3, nb2 = F3.sum(1), F2.sum(1)
    changed = (F3 != F2).any(1)
    S = parse(body, col + [cm]); mult = S[:, 3]; S = S[:, :3]
    # the current _nb CSV is B0's v2 output (or, in a changed row, already v3)
    state = dict(v2_rows=0, v3_rows=0, bad_rows=0)
    if not FULL:
        Sc = parse(body_cur, col + [cm]); mult_c = Sc[:, 3]; Sc = Sc[:, :3]
        ok_unch = (mult_c == mult - nb2)[~changed].all() and all(
            body_cur[i] == body[i] for i in np.where((nb2 == 0) & ~changed)[0])
        if not ok_unch:
            raise RuntimeError(f"{u['uid']}: the _nb CSV is not B0's v2 output (mult_total or a row without beam photons)")
        for i in np.where(changed)[0]:
            if mult_c[i] == mult[i] - nb2[i] and (nb2[i] > 0 or body_cur[i] == body[i]):
                state['v2_rows'] += 1
            elif mult_c[i] == mult[i] - nb3[i]:
                state['v3_rows'] += 1
            else:
                state['bad_rows'] += 1
        if state['bad_rows']:
            raise RuntimeError(f"{u['uid']}: {state['bad_rows']} changed rows hold neither the v2 nor the v3 multiplicity")
    else:
        Sc = None
    # rows to compute: the changed ones (delta), or every row with a v3 beam photon (full)
    rows = np.where(nb3 > 0)[0] if FULL else np.where(changed)[0]
    out_body = list(body) if FULL else list(body_cur)
    new = S.copy()
    stat_rows = dict(rewritten=0, reverted=0, fallback_miss=[0, 0, 0], fallback_neg=[0, 0, 0], nonfinite=[0, 0, 0])
    details = []
    for i in rows:
        if nb3[i] == 0:
            out_body[i] = body[i]
            stat_rows['reverted'] += 1
        else:
            f = P[i][M[i]]; b = F3[i][M[i]]
            zz, mm, UU = C.parts(f)
            s, _ = C.scale_of(zz, mm, UU)
            ev = C.four(zz, mm, UU, s)
            w = np.array(C.three(ev)); wo = np.array(C.three(ev[~b]))
            st = S[i]
            cand = st - (w - wo)
            miss = ~(np.abs(w - st) <= MISS_TOL)
            neg = ~miss & ~(cand >= 0)
            use = ~(miss | neg)
            q = body[i].split(',')
            for j, c in enumerate(col):
                if use[j]:
                    q[c] = repr(float(cand[j]))
                    new[i, j] = cand[j]
                stat_rows['fallback_miss'][j] += int(miss[j]); stat_rows['fallback_neg'][j] += int(neg[j])
                stat_rows['nonfinite'][j] += int(not np.isfinite(w[j]) or not np.isfinite(wo[j]))
            q[cm] = repr(float(mult[i] - nb3[i]))
            out_body[i] = ','.join(q)
            stat_rows['rewritten'] += 1
        if changed[i]:
            s_ev = float(BP._balance(P[i:i + 1], M[i:i + 1])[0][0])
            details.append(dict(row=int(i), why=why(i, I2, I3, F2, F3), nb2=int(nb2[i]), nb3=int(nb3[i]),
                                gained=photons(P, i, np.where(F3[i] & ~F2[i])[0], s_ev),
                                lost=photons(P, i, np.where(F2[i] & ~F3[i])[0], s_ev),
                                orig=[float(x) for x in S[i]] + [float(mult[i])],
                                v2=None if Sc is None else [float(x) for x in Sc[i]] + [float(mult_c[i])],
                                v3=[float(x) for x in new[i]] + [float(mult[i] - nb3[i])],
                                truncated=bool(M[i].sum() == PMAX)))
    text_new = '\n'.join([head] + out_body) + ('\n' if trail else '')
    t_main = time.time() - t0 - t_load
    # values per row: original (S), v2 (current CSV), v3 (the new text)
    V3 = Sc.copy() if Sc is not None else S.copy()
    if Sc is not None:
        V3[changed] = new[changed]
        for i in np.where(changed & (nb3 == 0))[0]:
            V3[i] = S[i]
    else:
        V3 = new
    mult3 = mult - nb3
    wh = {}
    for d in details:
        wh[d['why']] = wh.get(d['why'], 0) + 1
    stats = dict(uid=u['uid'], group=u['group'], rid=int(u['rid']), gen=u['gen'], npz=u['npz'], csv_in=u['csv_in'],
                 csv_out=u['csv_out'], n_events=int(N), pmax=int(PMAX), path='full' if FULL else 'delta',
                 rule_version=BP.RULE_VERSION, events_changed=int(changed.sum()), why=wh, input_state=state,
                 n_beam_events=int((nb3 > 0).sum()), n_beam_events_v2=int((nb2 > 0).sum()),
                 n_beam_sum=int(nb3.sum()), n_beam_sum_v2=int(nb2.sum()),
                 n_hist=np.bincount(np.minimum(nb3, 4), minlength=5).tolist(),
                 n_hist_v2=np.bincount(np.minimum(nb2, 4), minlength=5).tolist(),
                 mask=dict(rule_i_events=int(I3['rule_i'].sum()), axis_soft_events=int((I3['n_axis_soft'] > 0).sum()),
                           axis_soft_photons=int(I3['n_axis_soft'].sum()), lone_events=int((I3['n_lone'] > 0).sum()),
                           conflict_events=int((I3['n_conflict'] > 0).sum()), conflict_dropped=int(I3['n_conflict'].sum()),
                           tie_events=int(I3['tie'].sum()), v2_dropped_events=int((I2['n_dropped'] > 0).sum())),
                 rows=stat_rows, changed_events=details,
                 truncated=dict(events=int((M.sum(1) == PMAX).sum()), mult_gt_pmax=int((mult > PMAX).sum())),
                 mult=dict(sum_orig=float(mult.sum()), sum_v2=float((mult - nb2).sum()), sum_new=float(mult3.sum()),
                           min_new=float(mult3.min())),
                 obs={}, diag=diagnose(P, M, F3, I3), time=dict(load=t_load, main=t_main))
    for j, o in enumerate(OBS):
        v2c = Sc[:, j] if Sc is not None else None
        stats['obs'][o] = dict(sum_orig=float(S[:, j].sum()), sum_v2=None if v2c is None else float(v2c.sum()),
                               sum_new=float(V3[:, j].sum()), sumsq_new=float((V3[:, j] ** 2).sum()),
                               min_new=float(V3[:, j].min()), n_neg_new=int((V3[:, j] < 0).sum()),
                               n_differ_v2=None if v2c is None else int((V3[:, j] != v2c).sum()),
                               max_abs_new_minus_v2=None if v2c is None else float(np.abs(V3[:, j] - v2c).max()),
                               hist_new=hist(V3[:, j], THRUST_EDGES) if o == '1_minus_thrust' else None,
                               hist_v2=hist(v2c, THRUST_EDGES) if (o == '1_minus_thrust' and v2c is not None) else None)
    stats['time']['total'] = time.time() - t0
    stats['write'] = bool(changed.any()) if not FULL else True
    return text_new, stats


def dest(u):
    """the real path of the unit's _nb CSV: a regular file (not a link) directly inside a folder of
    NB_FOLDERS.txt, else RuntimeError"""
    dst = u['csv_out']
    real = os.path.realpath(dst)
    if os.path.dirname(real) not in NBF or not os.path.isfile(dst) or os.path.islink(dst):
        raise RuntimeError(f"{u['uid']}: {dst} is not a regular CSV directly inside a folder of NB_FOLDERS.txt")
    return real


def finish(u, text_new, stats):
    """write the record and, if a row changed, replace the _nb CSV atomically; mark done"""
    uid = u['uid']
    dst = dest(u)
    stats['new_md5'] = md5(text_new)
    stats['out_bytes'] = len(text_new.encode())
    os.makedirs(f'{WORK}/pending', exist_ok=True)
    pend = f'{WORK}/pending/{uid}.json'
    tmpj = f'{pend}.{os.getpid()}'
    json.dump(stats, open(tmpj, 'w'), indent=1)
    os.replace(tmpj, pend)
    if stats['write']:
        tmp = f'{os.path.dirname(dst)}/.{os.path.basename(dst)}.nb3tmp.{JOB}.{os.getpid()}'
        with open(tmp, 'w') as fh:
            fh.write(text_new)
        os.replace(tmp, dst)
    os.replace(pend, f'{WORK}/units/{uid}.json')


def recover(u):
    """a unit interrupted after its rename: the pending record names the md5 now on disk"""
    pend = f'{WORK}/pending/{u["uid"]}.json'
    if not os.path.exists(pend):
        return False
    s = json.load(open(pend))
    if md5(open(u['csv_out']).read()) == s['new_md5']:
        os.replace(pend, f'{WORK}/units/{u["uid"]}.json')
        return True
    os.remove(pend)
    return False


def est(u):
    return 1e-4 * int(u['n']) + 20


def claim(u):
    p = f'{WORK}/claims/{u["uid"]}'
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{JOB} {RST} {time.time():.0f} {os.uname()[1]}\n'.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                jid, rst, t, _h = open(p).read().split()
            except Exception:
                return False
            stale = (jid == JOB and int(rst) < RST) or (time.time() - float(t) > 3 * est(u) + 900)
            if not stale:
                return False
            try:
                os.rename(p, f'{p}.stale.{JOB}.{RST}.{os.getpid()}')
            except FileNotFoundError:
                return False
    return False


def work(u):
    uid = u['uid']
    try:
        if os.path.exists(f'{WORK}/units/{uid}.json'):
            return uid, 'skip', 0.0
        if time.time() - T0 + est(u) > TLIMIT - 300:
            return uid, 'notime', 0.0
        dest(u)
        if not claim(u):
            return uid, 'claimed', 0.0
        if recover(u):
            return uid, 'recovered', 0.0
        t0 = time.time()
        text_new, stats = compute(u)
        finish(u, text_new, stats)
        return uid, 'done' if stats['write'] else 'same', time.time() - t0
    except Exception:
        return uid, 'FAIL ' + traceback.format_exc().replace('\n', ' | '), 0.0


def test(u):
    t0 = time.time()
    text_new, stats = compute(u)
    os.makedirs(f'{WORK}/test', exist_ok=True)
    suf = '_full' if FULL else ''
    open(f'{WORK}/test/{u["uid"]}{suf}.csv', 'w').write(text_new)
    stats['new_md5'] = md5(text_new)
    json.dump(stats, open(f'{WORK}/test/{u["uid"]}{suf}.json', 'w'), indent=1)
    return u['uid'], 'test', time.time() - t0


if __name__ == '__main__':
    import multiprocessing as mp
    rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
    units = [dict(zip(rows[0], r)) for r in rows[1:]]
    if MODE == 'test':
        sel = [u for u in units if UIDS is None or u['uid'] in set(UIDS)]
        with mp.Pool(max(1, min(NPROC, len(sel)))) as pool:
            for uid, st, dt in pool.imap_unordered(test, sel):
                print(uid, st, f'{dt:.1f} s', flush=True)
        sys.exit(0)
    os.makedirs(f'{WORK}/claims', exist_ok=True); os.makedirs(f'{WORK}/units', exist_ok=True)
    # big units (4e5 events) spread through the list, so that few of them are in memory at once
    small = [u for u in units if int(u['n']) <= 80000]
    big = [u for u in units if int(u['n']) > 80000]
    small.sort(key=lambda u: -int(u['n']))
    order = []
    step = max(1, len(small) // max(1, len(big)))
    for k, u in enumerate(small):
        if k % step == 0 and big:
            order.append(big.pop(0))
        order.append(u)
    order += big
    if UIDS:
        order = [u for u in order if u['uid'] in set(UIDS)]
    print(f'worker job {JOB} restart {RST} on {os.uname()[1]}: {len(order)} candidate units, {NPROC} processes, '
          f'time limit {TLIMIT:.0f} s, {time.strftime("%c")}', flush=True)
    cnt = {}
    with mp.Pool(NPROC, maxtasksperchild=50) as pool:
        for uid, st, dt in pool.imap_unordered(work, order):
            k = st.split()[0]
            cnt[k] = cnt.get(k, 0) + 1
            if k in ('done', 'FAIL', 'recovered'):
                print(f'{uid} {st} {dt:.1f} s (elapsed {time.time() - T0:.0f} s; {cnt})', flush=True)
    print(f'WORKER END {JOB}: {cnt}, {time.time() - T0:.0f} s, {time.strftime("%c")}', flush=True)
    sys.exit(1 if cnt.get('FAIL') else 0)
