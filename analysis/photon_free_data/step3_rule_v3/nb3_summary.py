#!/usr/bin/env python3
"""Summary of the v3 rewrite from the unit records (nb3_units.py): per sample (gen folder or merged/
truncated stage folder) and generator, the events whose mask changed from v2 and why, the rows
rewritten, the means of the rewritten columns (original, v2, v3) and the DELPHI thrust bins (v2, v3);
the pair hemispheres (two flagged photons alone in a hemisphere that disagree) that rule (ii) decides
(the harder on the axis, the hardest particle of a whole event, no missing momentum, the softer's
transverse momentum relative to it below 1e-6 of its energy: the harder loses its flag), by generator and
energy class of the softer (the B5 tie class: softer below 1 MeV); every decision of rule (ii), with the
largest missing momentum, summed and single transverse momentum over E and the companions stored at
cos theta = +-1, and the photons it would decide in truncated events, which it leaves out (they keep their
flag unless (iii) drops them); the tie events outside two-photon hemispheres (flags in
different hemispheres, or in a larger hemisphere), where the hardest photon decides; and the residual
ambiguity that no stored feature resolves:
  pair ties   two photons alone in a hemisphere that disagree, (ii) does not apply and nothing else
              decides (the hardest is kept); expected wrong = ties x the share of witness-decided pairs
              of the same generator and energy class that dropped the hardest photon;
  rule (i)    hard photons (> 1 GeV) alone in their hemisphere on the axis without missing momentum:
              their phi comes from rounding, 0/pi (the rule (i) pattern), +-pi/2 or other; in events
              with base flags the axis is off the beam, so the ratio 0/pi : other there, times the
              'other' photons of events without any flag, estimates the rule (i) false positives.
    python nb3_summary.py WORK     writes WORK/results/nb3_summary.{json,txt}"""
import os, sys, json, glob
import numpy as np

WORK = sys.argv[1]
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
recs = [json.load(open(f)) for f in sorted(glob.glob(f'{WORK}/units/*.json'))]
want = sum(1 for _ in open(f'{WORK}/units.tsv')) - 1
groups, pairs, alone, changed, gray, ties_other, dec_ii, trunc_ii = {}, {}, {}, [], {}, {}, {}, {}
for r in recs:
    for key in (r['group'], f"{r['group']}|{r['gen']}", 'ALL', f"ALL|{r['gen']}"):
        g = groups.setdefault(key, dict(units=0, units_rewritten=0, events=0, events_changed=0, why={}, rows_rewritten=0,
                                        rows_reverted=0, beam_events=0, beam_events_v2=0, n_beam_sum=0, n_beam_sum_v2=0,
                                        rule_i_events=0, axis_soft_events=0, lone_events=0, conflict_events=0, tie_events=0,
                                        v2_dropped_events=0, input_v3_rows=0, mult_sum_orig=0.0, mult_sum_v2=0.0, mult_sum_new=0.0,
                                        fallback_miss=[0, 0, 0], fallback_neg=[0, 0, 0],
                                        obs={o: dict(sum_orig=0.0, sum_v2=0.0, sum_new=0.0, n_differ_v2=0, n_neg_new=0,
                                                     max_abs_new_minus_v2=0.0) for o in OBS},
                                        hist_v2=None, hist_new=None))
        g['units'] += 1; g['units_rewritten'] += int(r['events_changed'] > 0); g['events'] += r['n_events']
        g['events_changed'] += r['events_changed']
        for k, v in r['why'].items():
            g['why'][k] = g['why'].get(k, 0) + v
        g['rows_rewritten'] += r['rows']['rewritten']; g['rows_reverted'] += r['rows']['reverted']
        g['beam_events'] += r['n_beam_events']; g['beam_events_v2'] += r['n_beam_events_v2']
        g['n_beam_sum'] += r['n_beam_sum']; g['n_beam_sum_v2'] += r['n_beam_sum_v2']
        for k in ('rule_i_events', 'axis_soft_events', 'lone_events', 'conflict_events', 'tie_events', 'v2_dropped_events'):
            g[k] += r['mask'][k]
        g['input_v3_rows'] += r['input_state']['v3_rows']
        g['mult_sum_orig'] += r['mult']['sum_orig']; g['mult_sum_v2'] += r['mult']['sum_v2']; g['mult_sum_new'] += r['mult']['sum_new']
        for j in range(3):
            g['fallback_miss'][j] += r['rows']['fallback_miss'][j]; g['fallback_neg'][j] += r['rows']['fallback_neg'][j]
        for o in OBS:
            a, b = g['obs'][o], r['obs'][o]
            a['sum_orig'] += b['sum_orig']; a['sum_v2'] += b['sum_v2']; a['sum_new'] += b['sum_new']
            a['n_differ_v2'] += b['n_differ_v2']; a['n_neg_new'] += b['n_neg_new']
            a['max_abs_new_minus_v2'] = max(a['max_abs_new_minus_v2'], b['max_abs_new_minus_v2'])
        t = r['obs']['1_minus_thrust']
        g['hist_v2'] = np.array(t['hist_v2']) if g['hist_v2'] is None else g['hist_v2'] + np.array(t['hist_v2'])
        g['hist_new'] = np.array(t['hist_new']) if g['hist_new'] is None else g['hist_new'] + np.array(t['hist_new'])
    for p in r['diag']['pairs']:
        k = (r['gen'], p['decided'], 'both>1MeV' if p['both_gt_1MeV'] else 'softer<1MeV')
        s = pairs.setdefault(k, dict(n=0, hardest_kept=0, hardest_dropped=0, both_kept=0, softer_kept=0, truncated=0,
                                     E_hard=[], ratio=[]))
        s['n'] += 1; s['hardest_kept'] += int(p['hardest_kept'] and not p['softer_kept'])
        s['hardest_dropped'] += int(not p['hardest_kept']); s['both_kept'] += int(p['hardest_kept'] and p['softer_kept'])
        s['softer_kept'] += int(p['softer_kept']); s['truncated'] += int(p.get('truncated', False))
        s['E_hard'].append(p['E_hard']); s['ratio'].append(p.get('ratio', float('nan')))
    for x in r['diag'].get('ties_other', []):
        ties_other.setdefault(r['gen'], []).append(dict(uid=r['uid'], **x))
    for x in r['diag'].get('ii', []):
        dec_ii.setdefault(r['gen'], []).append(dict(uid=r['uid'], **x))
    for x in r['diag'].get('ii_truncated', []):
        trunc_ii.setdefault(r['gen'], []).append(dict(uid=r['uid'], **x))
    for k, v in r['diag']['alone'].items():
        kk = f"{r['gen']}|{k}"
        alone[kk] = alone.get(kk, 0) + v
    gray.setdefault(r['gen'], []).extend(dict(uid=r['uid'], **x) for x in r['diag']['gray'])
    for d in r['changed_events']:
        changed.append(dict(uid=r['uid'], gen=r['gen'], **d))
out = dict(units=len(recs), units_wanted=want, groups={}, pairs={}, alone=alone, estimates={})
lines = [f'units {len(recs)} of {want}']
for key in sorted(groups, key=lambda k: (k.startswith('ALL'), k)):
    g = groups[key]
    rec = {k: v for k, v in g.items() if k not in ('obs', 'hist_v2', 'hist_new')}
    n = g['events']
    rec['mult_mean'] = [g['mult_sum_orig'] / n, g['mult_sum_v2'] / n, g['mult_sum_new'] / n]
    rec['obs'] = {}
    for o in OBS:
        a = g['obs'][o]
        rec['obs'][o] = dict(mean_orig=a['sum_orig'] / n, mean_v2=a['sum_v2'] / n, mean_new=a['sum_new'] / n,
                             rel_vs_orig=a['sum_new'] / a['sum_orig'] - 1, rel_vs_v2=a['sum_new'] / a['sum_v2'] - 1,
                             n_differ_v2=a['n_differ_v2'], n_neg_new=a['n_neg_new'], max_abs_new_minus_v2=a['max_abs_new_minus_v2'])
    hv, hn = g['hist_v2'], g['hist_new']
    rec['thrust_bins_v2'] = hv.tolist(); rec['thrust_bins_new'] = hn.tolist()
    rec['thrust_bins_rel_vs_v2'] = [float(b / a - 1) if a else 0.0 for a, b in zip(hv[1:-1], hn[1:-1])]
    out['groups'][key] = rec
    lines.append(f"== {key}: {g['units']} units ({g['units_rewritten']} rewritten), {n:,} events; mask changed in "
                 f"{g['events_changed']:,} events {g['why']}; rows rewritten {g['rows_rewritten']}, reverted to the original "
                 f"{g['rows_reverted']}; beam-photon events {g['beam_events']:,} (v2 {g['beam_events_v2']:,}), photons "
                 f"{g['n_beam_sum']:,} (v2 {g['n_beam_sum_v2']:,}); v3 decisions: rule (i) {g['rule_i_events']} events "
                 f"({g['axis_soft_events']} with a soft photon on the axis), lone {g['lone_events']}, consistency "
                 f"{g['conflict_events']} (ties {g['tie_events']}); v2 drops {g['v2_dropped_events']}")
    lines.append('   mult mean orig / v2 / v3: ' + ' / '.join(f'{x:.4f}' for x in rec['mult_mean']) +
                 f"  (v3 vs v2 {rec['mult_mean'][2] / rec['mult_mean'][1] - 1:+.5%})")
    for o in OBS:
        q = rec['obs'][o]
        lines.append(f"   {o}: mean orig {q['mean_orig']:.6f} v2 {q['mean_v2']:.6f} v3 {q['mean_new']:.6f}; v3 vs orig "
                     f"{q['rel_vs_orig']:+.4%}, vs v2 {q['rel_vs_v2']:+.5%}; values differing from v2 {q['n_differ_v2']}, "
                     f"negative {q['n_neg_new']}")
    lines.append('   DELPHI thrust bins, v3 vs v2 (%): ' + ', '.join(f'{100 * x:+.3f}' for x in rec['thrust_bins_rel_vs_v2']))
    lines.append(f"   fallbacks in rewritten rows (1-T, B, rho): miss {g['fallback_miss']}, negative {g['fallback_neg']}")
for k, s in sorted(pairs.items()):
    out['pairs']['|'.join(k)] = dict(n=s['n'], hardest_kept=s['hardest_kept'], hardest_dropped=s['hardest_dropped'],
                                     both_kept=s['both_kept'], softer_kept=s['softer_kept'], truncated=s['truncated'],
                                     E_hard_quartiles=np.quantile(s['E_hard'], [0, .25, .5, .75, 1]).tolist(),
                                     ratio_max=float(np.nanmax(s['ratio'])) if s['ratio'] else None)
# the decisions of (ii) in pair hemispheres, and the ties outside them
lines.append('== rule (ii) in pair hemispheres (the harder photon on the axis loses its flag)')
for (gen, dec, cls), s in sorted(pairs.items()):
    if dec == 'ii':
        out['estimates'][f'ii_pairs|{gen}|{cls}'] = out['pairs'][f'{gen}|{dec}|{cls}']
        lines.append(f"   {gen} {cls}: {s['n']} pairs, the softer kept in {s['softer_kept']}, truncated events {s['truncated']}; "
                     f"summed pT / E of the rest of the hemisphere at most {np.nanmax(s['ratio']):.3g}")
if not any(dec == 'ii' for (_, dec, _) in pairs):
    lines.append('   none')
lines.append('== every decision of rule (ii) (whole events), and the photons it would decide in truncated events (left out)')
for gen in sorted(set(dec_ii) | set(trunc_ii)):
    v, t = dec_ii.get(gen, []), trunc_ii.get(gen, [])
    mx = lambda k: max((x[k] for x in v), default=0.0)
    out['estimates'][f'ii|{gen}'] = dict(decisions=len(v), miss_max=mx('miss'), ratio_max=mx('ratio'), largest_max=mx('largest'),
                                         with_cos1_companion=sum(x['n_cos1'] > 0 for x in v), examples=v[:50],
                                         truncated_left_out=len(t), truncated_kept=sum(x['kept'] for x in t), truncated_examples=t[:50])
    lines.append(f"   {gen}: {len(v)} decisions; |sum p| at most {mx('miss'):.3g} GeV; summed pT / E of the rest of the hemisphere "
                 f"at most {mx('ratio'):.3g}, its largest part {mx('largest'):.3g}; {out['estimates'][f'ii|{gen}']['with_cos1_companion']} "
                 f"with a companion stored at cos theta = +-1; in truncated events {len(t)} left out "
                 f"({out['estimates'][f'ii|{gen}']['truncated_kept']} keep their flag)")
if not dec_ii and not trunc_ii:
    lines.append('   none')
lines.append('== ties outside two-photon hemispheres (the hardest photon decides)')
if not ties_other:
    lines.append('   none')
for gen in sorted(ties_other):
    v = ties_other[gen]
    out['estimates'][f'ties_other|{gen}'] = dict(events=len(v), different_hemispheres=sum(0 < x['flags_pos'] < x['n_flags'] for x in v),
                                                examples=v[:50])
    lines.append(f"   {gen}: {len(v)} events, of which {out['estimates'][f'ties_other|{gen}']['different_hemispheres']} with "
                 f"flags in both hemispheres")
# estimates
lines.append('== residual ambiguity')
for gen in sorted({k[0] for k in pairs}):
    for cls in ('both>1MeV', 'softer<1MeV'):
        w = pairs.get((gen, 'witness', cls)); t = pairs.get((gen, 'tie', cls))
        if not t:
            continue
        f = w['hardest_dropped'] / w['n'] if w and w['n'] else None
        est = dict(ties=t['n'], witness_decided=w['n'] if w else 0, witness_dropped_hardest=w['hardest_dropped'] if w else 0,
                   share_dropped=f, expected_wrong=None if f is None else t['n'] * f)
        out['estimates'][f'pair_ties|{gen}|{cls}'] = est
        fs = 'n/a' if f is None else f'{f:.3f}'
        ws = 'n/a' if f is None else f"{est['expected_wrong']:.0f}"
        lines.append(f"   pair ties {gen} {cls}: {t['n']} decided by the hardest photon; witnessed pairs {est['witness_decided']}, "
                     f"of which {est['witness_dropped_hardest']} dropped the hardest ({fs}); expected wrong {ws}")
for gen in sorted({k.split('|')[0] for k in alone}):
    get = lambda ec, pc: sum(v for k, v in alone.items() if k.startswith(f'{gen}|{ec}|{pc}|') and k.endswith('miss<1e-4'))
    ax, ot, pl = get('base', 'axis'), get('base', 'other'), get('base', 'plane')
    none_other, none_plane = get('none', 'other'), get('none', 'plane')
    ri = get('rule_i', 'axis')
    r = ax / ot if ot else None
    est = dict(base_axis=ax, base_other=ot, base_plane=pl, none_other=none_other, none_plane=none_plane, rule_i_axis=ri,
               ratio_axis_to_other=r, expected_rule_i_false_positives=None if r is None else none_other * r)
    out['estimates'][f'rule_i|{gen}'] = est
    rs = 'n/a' if r is None else f'{r:.2f}'
    fp = 'n/a' if r is None else f'{none_other * r:.0f}'
    lines.append(f"   rule (i) {gen}: alone photons without missing momentum in base events: phi 0/pi {ax}, other {ot}, "
                 f"+-pi/2 {pl} (ratio 0/pi:other {rs}); in events without any flag: other {none_other}, +-pi/2 "
                 f"{none_plane}; rule (i) events with such a photon {ri}; expected false positives {fp}")
for gen in sorted(gray):
    out['estimates'][f'gray|{gen}'] = dict(events=len(gray[gen]), examples=gray[gen][:50])
    lines.append(f"   gray zone {gen}: {len(gray[gen])} events keep flags that disagree beyond the tolerance but within 1e-3")
out['changed_events'] = changed[:5000]
os.makedirs(f'{WORK}/results', exist_ok=True)
json.dump(out, open(f'{WORK}/results/nb3_summary.json', 'w'), indent=1)
open(f'{WORK}/results/nb3_summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[:3]))
