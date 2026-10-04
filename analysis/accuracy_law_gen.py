#!/usr/bin/env python3
"""Reconstructed producer of output/accuracy_law_generator.json (the closed-loop test of the
accuracy law, Sec. 5.3): the missing script was reverse-engineered from the file contents,
the paper text and the sibling conventions, and validated against the stored file (see
output/accuracy_law_recon.json and RECONSTRUCTION STATUS below).

WHAT IT DOES. The per-bin relative errors eps_b of the reweighted density are measured
directly from the 800k comparison at the central and one off-centre held-out point, with both
statistical contributions subtracted and no clipping (the estimate of eps_b^2 is unbiased and
may go negative bin by bin). They are inserted into the saturating generalization of the
accuracy law: the expected squared pull in a bin is 1 + eps_b^2 p_b^2 / v_b(N), where v_b(N)
is the same pooled variance the ruler itself uses at target statistics N, which reduces to
chi2/ndf = 1 + N/N* while the target error dominates and saturates once the fixed reference
error takes over. Predictions at N = 80k and N = 800k are compared with the measured closure
widths at the same statistics; the 80k prediction is fully out of sample because the 80k runs
(6902/6901) are independent of the 800k runs (6970/6971) that supplied eps_b.

MEASURED WIDTHS (validated exactly against the stored file):
 * the four kinematic observables 1-T, mult, B_tot, rho_H;
 * bin edges from make_bins(reference + 800k target), FIXED ONCE per point and observable and
   shared by the 80k and the 800k comparison;
 * full runs on the target side (80000 / 800000 events, no report-half split);
 * pooled width: sqrt( sum of squared pulls over all four observables / (n_bins_total - 1) ),
   with r2_ladder.pulls (pooled multinomial variance, 5-effective-event rule).
 -> reproduces the stored measured values exactly: 1.112/1.432 centre, 1.135/1.433 off-centre.

PREDICTED WIDTHS (reconstruction; see status below). Per kept bin of the 800k comparison, in
bin-probability units (qa = reweighted reference, qb = fresh 800k, na/nb = effective counts,
Na = reference effective total, qhat = pooled probability):
   eps_b^2 = [ (qa-qb)^2 - v_b ] / qb^2 ,   v_b = qhat^2 (1-qhat) (1/na + 1/(800000*qhat))
   E[pull^2](N) = 1 + eps_b^2 * qa^2 / v_b(N)
   v_b(N) = qhat_N^2 (1-qhat_N) (1/na + 1/(N*qb)) ,  qhat_N = (Na*qa + N*qb)/(Na + N)
   predicted width = sqrt( sum_b E[pull^2] / n_bins_total )
The naive target-dominated form drops the reference term 1/na from v_b(N).

RECONSTRUCTION STATUS (honest): the measured pipeline reproduces the stored file exactly.
The predicted pipeline reproduces the stored file to +-0.001 in the width: it gives
1.137/1.489 (centre) and 1.142/1.460 (off-centre) against stored 1.138/1.489 and 1.142/1.461,
i.e. two of four third decimals differ by one unit. At the precision the paper quotes (1.14,
1.49, 1.14, 1.46) all values agree. The stored 'naive would predict 2.4' is NOT reproduced:
this reconstruction gives 2.20/2.23 for the natural target-dominated form (and 2.71/2.77 if
the reference term is also dropped from the eps extraction), so the published 2.4 must come
from an intermediate convention that was not recovered; the discrepancy is recorded rather
than tuned away.

Writes --out (default output/accuracy_law_recon.json so the archived original is never
touched). After a retrain, rerun with --out output/accuracy_law_generator.json --force.
"""
import os, sys, json, argparse
os.environ.setdefault('LADDER_ACT', 'silu')
import numpy as np
import pandas as pd
from ab_analysis import Cond
from r2_ladder import make_bins, pulls, width_of, MIN_COUNT

CSV_OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy']
DATA = 'data_stageC'
POINTS = {
    'centre':     dict(theta=(0.120, 0.46, 1.21), runs={'k80': 6902, 'k800': 6970}),
    'off-centre': dict(theta=(0.124, 0.55, 1.50), runs={'k80': 6901, 'k800': 6971}),
}
NSTAT = {'k80': 80000.0, 'k800': 800000.0}


def load_run(rid):
    sh = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
    return {o: sh[o].values for o in CSV_OBS}


def hist_weighted(v, w, bins):
    """Bin probabilities, per-bin Kish effective counts, effective total (weighted side)."""
    h, _ = np.histogram(v, bins, weights=w); h2, _ = np.histogram(v, bins, weights=w**2)
    neff = np.where(h2 > 0, h**2/np.where(h2 > 0, h2, 1.0), 0.0)
    return h/h.sum(), neff, h.sum()**2/h2.sum()


def hist_unweighted(v, bins):
    n, _ = np.histogram(v, bins)
    return n/n.sum(), n.astype(float), float(n.sum())


def safe(x):
    return np.where(x > 0, x, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='output/accuracy_law_recon.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.force:
        sys.exit(f'{args.out} exists; pass --force to overwrite')

    cond = Cond('C')
    ref = np.load('output/models/C_ref.npz')
    robs = {o: ref[o].astype(float) for o in CSV_OBS}

    out = {}
    for tag, p in POINTS.items():
        f = cond.f_at(np.array(p['theta']))
        w = np.exp((f - f.max())/cond.T); w /= w.sum()
        runs = {kl: load_run(rid) for kl, rid in p['runs'].items()}

        # ---- bins from the 800k comparison, fixed once and shared by both statistics ----
        BINS = {o: make_bins(np.r_[robs[o], runs['k800'][o]], o) for o in CSV_OBS}

        # ---- measured pooled widths through those bins ----
        measured = {}
        for kl in ('k80', 'k800'):
            p2, ndf = 0.0, 0
            for o in CSV_OBS:
                pl, nb = pulls(robs[o], w, runs[kl][o], BINS[o])
                p2 += float(np.sum(pl**2)); ndf += nb
            measured[kl] = round(float(np.sqrt(p2/(ndf - 1))), 3)

        # ---- eps_b from the 800k comparison, prediction at both statistics ----
        per = {}
        for o in CSV_OBS:
            qa, na, Na = hist_weighted(robs[o], w, BINS[o])
            qb, nb, Nb = hist_unweighted(runs['k800'][o], BINS[o])
            qhat = (Na*qa + Nb*qb)/(Na + Nb)
            keep = (qhat > 0) & (qhat < 1) & (na >= MIN_COUNT) & (nb >= MIN_COUNT)
            v800 = qhat**2*(1-qhat)*(1/safe(na) + 1/safe(800000.0*qhat))
            eps2 = ((qa - qb)**2 - v800)/safe(qb)**2          # unclipped, unbiased
            per[o] = dict(qa=qa, qb=qb, na=na, Na=Na, eps2=eps2, keep=keep)
        predicted, naive = {}, {}
        for kl, N in NSTAT.items():
            s_sat, s_nai, nbt = 0.0, 0.0, 0
            for o in CSV_OBS:
                d = per[o]; k = d['keep']
                qhN = (d['Na']*d['qa'] + N*d['qb'])/(d['Na'] + N)
                vN = qhN**2*(1-qhN)*(1/safe(d['na']) + 1/safe(N*d['qb']))
                vN_naive = qhN**2*(1-qhN)*(1/safe(N*d['qb']))
                s_sat += float(np.sum(1 + d['eps2'][k]*d['qa'][k]**2/vN[k]))
                s_nai += float(np.sum(1 + d['eps2'][k]*d['qa'][k]**2/vN_naive[k]))
                nbt += int(k.sum())
            predicted[kl] = round(float(np.sqrt(s_sat/nbt)), 3)
            naive[kl] = round(float(np.sqrt(s_nai/nbt)), 3)

        out[tag] = dict(predicted=predicted, measured=measured)
        out.setdefault('naive_prediction_800k', {})[tag] = naive['k800']
        # per-bin accuracy summary: probability-weighted rms relative error over kept bins
        e2 = np.concatenate([per[o]['eps2'][per[o]['keep']] for o in CSV_OBS])
        qq = np.concatenate([per[o]['qb'][per[o]['keep']] for o in CSV_OBS])
        out.setdefault('epsilon_rms_pct', {})[tag] = round(
            100.0*float(np.sqrt(max(np.sum(e2*qq)/np.sum(qq), 0.0))), 2)
        print(f'{tag:>10}: measured {measured}  predicted {predicted}  naive(800k) {naive["k800"]}')

    out['convention'] = dict(
        observables=CSV_OBS,
        bins='make_bins(reference + 800k target), shared by both statistics',
        targets={t: POINTS[t]['runs'] for t in POINTS},
        measured='pooled sqrt(sum pulls^2 / (total bins - 1)), full runs, r2_ladder.pulls',
        eps_extraction='eps2 = ((qa-qb)^2 - qhat^2(1-qhat)(1/na + 1/(800k qhat)))/qb^2, unclipped',
        prediction='E[pull^2](N) = 1 + eps2 qa^2 / [qhatN^2(1-qhatN)(1/na + 1/(N qb))], '
                   'qhatN = (Na qa + N qb)/(Na+N); width = sqrt(mean over kept bins)',
        naive='same with the reference term 1/na dropped from v(N)',
        status='measured exact vs archived file; predicted within +-0.001; '
               'paper naive 2.4 not reproduced (this convention gives 2.20/2.23)')
    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
