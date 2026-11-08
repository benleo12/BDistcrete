#!/usr/bin/env python3
"""Spot check of rewritten _nb CSVs against the original CSVs, text level, with the new mask:
same header, row count and trailing newline; a row without a beam photon byte-identical; in a row
with beam photons every column but the four byte-identical, mult_total = stored - n_beam, and each
of the three shapes either stored - (with - without) recomputed from scratch on a sample of events
(nb_core.event_delta) or, where the fallback holds, the stored text.

    python nb2_spot.py WORK NUNITS SEED [NEWDIR]
NEWDIR: read the new CSV from NEWDIR/<uid>.csv (test outputs) instead of the unit's csv_out.
Units are drawn from WORK/units/*.json (or NEWDIR/*.json)."""
import os, sys, json, glob, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_core as C
import beam_photons as BP

WORK, NU, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NEWDIR = sys.argv[4] if len(sys.argv) > 4 else None
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
recs = sorted(glob.glob(f'{NEWDIR or WORK + "/units"}/*.json'))
random.seed(SEED)
pick = random.sample(recs, min(NU, len(recs)))
bad = 0
out = []
for f in pick:
    u = json.load(open(f))
    a_text = open(u['csv_in']).read()
    b_path = f'{NEWDIR}/{u["uid"]}.csv' if NEWDIR else u['csv_out']
    b_text = open(b_path).read()
    a, b = a_text.split('\n'), b_text.split('\n')
    ok_shape = len(a) == len(b) and a[0] == b[0] and (a[-1] == '') == (b[-1] == '')
    head = a[0].split(','); col = [head.index(o) for o in OBS]; cm = head.index('mult_total')
    keep = [k for k in range(len(head)) if k not in col + [cm]]
    z = np.load(u['npz']); P, M = z['particles'], z['mask'].astype(bool)
    F = BP.beam_photon_mask(P, M); nb = F.sum(1)
    ra, rb = a[1:-1], b[1:-1]
    n_other = n_unflag_changed = n_mult_bad = 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        if nb[i] == 0:
            n_unflag_changed += x != y
            continue
        p, q = x.split(','), y.split(',')
        n_other += [p[k] for k in keep] != [q[k] for k in keep]
        n_mult_bad += q[cm] != repr(float(p[cm]) - int(nb[i]))
    # recompute events from scratch: an even sample of beam-photon events plus every event where
    # a shape kept its stored text
    idx = np.where(nb > 0)[0]
    samp = list(idx[:: max(1, len(idx) // 25)][:25])
    fb = [i for i in idx if any(ra[i].split(',')[c] == rb[i].split(',')[c] for c in col)]
    err = 0.0; n_fb_ok = n_fb_bad = 0
    for i in sorted(set(samp) | set(fb[:50])):
        m = M[i]
        w, wo, s = C.event_delta(P[i][m], F[i][m])
        st = np.array([float(ra[i].split(',')[c]) for c in col])
        tok = [rb[i].split(',')[c] for c in col]
        for j, c in enumerate(col):
            cand = st[j] - (w[j] - wo[j])
            fallback = not (abs(w[j] - st[j]) <= 1e-2) or not (cand >= 0)
            if fallback:
                if tok[j] == ra[i].split(',')[c]:
                    n_fb_ok += 1
                else:
                    n_fb_bad += 1
            else:
                err = max(err, abs(float(tok[j]) - cand))
    good = ok_shape and n_other == 0 and n_unflag_changed == 0 and n_mult_bad == 0 and err == 0.0 and n_fb_bad == 0
    bad += not good
    r = dict(uid=u['uid'], rows=len(rb), beam_events=int((nb > 0).sum()), header_rows_ok=ok_shape,
             rows_other_column_changed=n_other, rows_without_beam_photon_changed=n_unflag_changed,
             mult_wrong=n_mult_bad, recomputed_events=len(set(samp) | set(fb[:50])), recomputed_max_diff=err,
             fallback_tokens_ok=n_fb_ok, fallback_tokens_bad=n_fb_bad, ok=good)
    out.append(r)
    print(json.dumps(r), flush=True)
print(f'SPOT CHECK: {len(pick)} units, {bad} bad')
json.dump(out, open(f'{WORK}/results/spot_{SEED}{"_test" if NEWDIR else ""}.json', 'w'), indent=1)
