#!/usr/bin/env python3
"""Summary of the rewritten _nb CSVs from the unit records: per sample (gen folder or merged/truncated
stage folder) and generator, the change of every rewritten column against the original CSVs and
against Phase A's _nb CSVs, the mask against Phase A's rule, the fallbacks, and the checks.
    python nb2_summary.py WORK     writes WORK/results/nb2_summary.{json,txt}"""
import os, sys, json, glob
import numpy as np

WORK = sys.argv[1]
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
recs = [json.load(open(f)) for f in sorted(glob.glob(f'{WORK}/units/*.json'))]
want = sum(1 for _ in open(f'{WORK}/units.tsv')) - 1
groups = {}
for r in recs:
    for key in (r['group'], f"{r['group']}|{r['gen']}", 'ALL', f"ALL|{r['gen']}"):
        g = groups.setdefault(key, dict(units=0, events=0, beam_events=0, beam_events_A=0, n_beam_sum=0, n_beam_sum_A=0,
                                        mask_events_differ=0, rule_i_events=0, rule_i_photons=0, consistency_events=0,
                                        consistency_dropped=0, other_differ=0, z_gained=0.0, z_lost=0.0,
                                        regression_mismatch=0, regression_checked=0, truncated=0, trunc_mult_gt_pmax=0,
                                        sig_events=0, sig_axis_off_beam=0, sig_E_off=[], mult_sum_orig=0.0, mult_sum_new=0.0,
                                        mult_min_new=1e9, ebeam_gt5=0, ebeam_gt20=0, paths={},
                                        obs={o: dict(sum_orig=0.0, sum_A=0.0, sum_new=0.0, fallback_miss=0, fallback_neg=0,
                                                     nonfinite=0, n_differ_A=0, n_neg_new=0, min_new=1e9, max_abs_new_minus_A=0.0,
                                                     sum_new_minus_A=0.0, hist_orig=None, hist_A=None, hist_new=None)
                                             for o in OBS}))
        g['units'] += 1; g['events'] += r['n_events']; g['beam_events'] += r['n_beam_events']
        g['beam_events_A'] += r['n_beam_events_A']; g['n_beam_sum'] += r['n_beam_sum']; g['n_beam_sum_A'] += r['n_beam_sum_A']
        m = r['mask']
        g['mask_events_differ'] += m['events_differ']; g['rule_i_events'] += m['rule_i_events']
        g['rule_i_photons'] += m['rule_i_photons']; g['consistency_events'] += m['consistency_events']
        g['consistency_dropped'] += m['consistency_dropped']; g['other_differ'] += m['other_differ']
        g['z_gained'] += m['z_gained']; g['z_lost'] += m['z_lost']
        g['regression_mismatch'] += r['regression_mismatch']; g['regression_checked'] += r['regression_checked']
        g['truncated'] += r['truncated']['events']; g['trunc_mult_gt_pmax'] += r['truncated']['mult_gt_pmax']
        s = r['signature_in_base_events']
        g['sig_events'] += s['events']; g['sig_axis_off_beam'] += s['axis_off_beam']; g['sig_E_off'] += s['E_off_GeV']
        g['mult_sum_orig'] += r['mult']['sum_orig']; g['mult_sum_new'] += r['mult']['sum_new']
        g['mult_min_new'] = min(g['mult_min_new'], r['mult']['min_new'])
        g['ebeam_gt5'] += r['n_ebeam_gt5']; g['ebeam_gt20'] += r['n_ebeam_gt20']
        p = r.get('path', 'full'); g['paths'][p] = g['paths'].get(p, 0) + 1
        for o in OBS:
            a, b = g['obs'][o], r['obs'][o]
            for k in ('sum_orig', 'sum_A', 'sum_new', 'fallback_miss', 'fallback_neg', 'nonfinite', 'n_differ_A',
                      'n_neg_new', 'sum_new_minus_A'):
                a[k] += b[k]
            a['min_new'] = min(a['min_new'], b['min_new'])
            a['max_abs_new_minus_A'] = max(a['max_abs_new_minus_A'], b['max_abs_new_minus_A'])
            for k in ('hist_orig', 'hist_A', 'hist_new'):
                a[k] = (np.array(b[k]) if a[k] is None else a[k] + np.array(b[k]))
out = {}
lines = [f'units {len(recs)} of {want}']
for key in sorted(groups, key=lambda k: (k.startswith('ALL'), k)):
    g = groups[key]
    rec = {k: v for k, v in g.items() if k not in ('obs', 'sig_E_off')}
    rec['sig_E_off_GeV'] = sorted(g['sig_E_off'])[-20:]
    rec['mult_mean_orig'] = g['mult_sum_orig'] / g['events']; rec['mult_mean_new'] = g['mult_sum_new'] / g['events']
    rec['obs'] = {}
    for o in OBS:
        a = g['obs'][o]
        ho, hA, hn = a['hist_orig'], a['hist_A'], a['hist_new']
        rec['obs'][o] = dict(mean_orig=a['sum_orig'] / g['events'], mean_A=a['sum_A'] / g['events'],
                             mean_new=a['sum_new'] / g['events'],
                             rel_new_vs_orig=a['sum_new'] / a['sum_orig'] - 1, rel_new_vs_A=a['sum_new'] / a['sum_A'] - 1,
                             fallback_miss=a['fallback_miss'], fallback_neg=a['fallback_neg'], nonfinite=a['nonfinite'],
                             events_differ_A=a['n_differ_A'], max_abs_new_minus_A=a['max_abs_new_minus_A'],
                             n_neg_new=a['n_neg_new'], min_new=a['min_new'],
                             bins_rel_new_vs_orig=[float(x) for x in np.where(ho > 0, hn / np.maximum(ho, 1) - 1, 0)],
                             bins_rel_new_vs_A=[float(x) for x in np.where(hA > 0, hn / np.maximum(hA, 1) - 1, 0)],
                             hist_orig=ho.tolist(), hist_A=hA.tolist(), hist_new=hn.tolist())
    out[key] = rec
    lines.append(f"\n== {key}: {g['units']} units, {g['events']:,} events, beam-photon events {g['beam_events']:,} "
                 f"(Phase A {g['beam_events_A']:,}); mask differs from Phase A in {g['mask_events_differ']} events "
                 f"(axis rule {g['rule_i_events']} events / {g['rule_i_photons']} photons, consistency {g['consistency_events']} "
                 f"events / {g['consistency_dropped']} photons, other {g['other_differ']}); regression {g['regression_mismatch']} "
                 f"mismatches in {g['regression_checked']:,}; paths {g['paths']}")
    lines.append(f"   mult_total mean {rec['mult_mean_orig']:.4f} -> {rec['mult_mean_new']:.4f} "
                 f"({rec['mult_mean_new'] / rec['mult_mean_orig'] - 1:+.4%}), min new {g['mult_min_new']}; "
                 f"beam photons {g['n_beam_sum']:,} (Phase A {g['n_beam_sum_A']:,}); truncated events {g['truncated']} "
                 f"(mult > 100: {g['trunc_mult_gt_pmax']}); (i) signature off the beam in events with base flags: "
                 f"{g['sig_axis_off_beam']} of {g['sig_events']}")
    for o in OBS:
        r = rec['obs'][o]
        lines.append(f"   {o:15s} mean {r['mean_orig']:.6f} (orig) {r['mean_A']:.6f} (Phase A) {r['mean_new']:.6f} (new): "
                     f"vs orig {r['rel_new_vs_orig']:+.4%}, vs Phase A {r['rel_new_vs_A']:+.5%}; events differing from Phase A "
                     f"{r['events_differ_A']} (max {r['max_abs_new_minus_A']:.3g}); fallback miss {r['fallback_miss']} neg "
                     f"{r['fallback_neg']} nonfinite {r['nonfinite']}; negative new {r['n_neg_new']}")
json.dump(out, open(f'{WORK}/results/nb2_summary.json', 'w'), indent=1)
open(f'{WORK}/results/nb2_summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
