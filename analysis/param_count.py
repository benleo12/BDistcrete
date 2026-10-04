#!/usr/bin/env python3
"""Archive the event-side parameter count quoted in App. B ("identical event-side parameter
count of 82457"), previously stored nowhere.

The count is measured on the actual production architecture: CondPFN of r2_ladder.py in the
kinematic 4-feature configuration (Stage A: cos theta, phi, log z, log sin^2 theta), EVENT
SIDE ONLY, i.e. the per-particle network Phi plus the downstream head A; the parameter
network B(theta), which has no energyflow counterpart, is excluded. Two head widths are
counted:
 * K=1  : the implementation-crosscheck configuration (crosscheck_torch.py trains
          CondPFN(C, 1, K=1); the energyflow side is PFN(input_dim=4, Phi=(100,100,256),
          F=(100,100,100), output_dim=1), whose parameter count is Phi + F + the single
          sigmoid output unit). This is how the published 82457 was counted, and it is the
          number the "identical parameter count" claim refers to: both implementations of
          the SAME K=1 classification task hold 82457 event-side parameters.
 * K=24 : the production head. The event side then ends in a 100->24 layer instead of
          100->1, adding 23*(100+1) = 2323 parameters for a total of 84780. The published
          82457 is therefore NOT the K=24 production event side; it is the matched K=1
          crosscheck configuration.
CPU-only. Writes output/param_count.json with the layer-by-layer breakdown.
"""
import os, json
os.environ.setdefault('LADDER_ACT', 'silu')
import torch
import torch.nn as nn
from r2_ladder import CondPFN, PHI_SIZES, F_SIZES

C_KIN = 4          # kinematic features: cos theta, phi, log z, log sin^2 theta
NTHETA = 1         # Stage A conditioning (irrelevant to the event side)


def breakdown(module, name):
    rows, tot = [], 0
    for i, layer in enumerate(module):
        if isinstance(layer, nn.Linear):
            n = layer.weight.numel() + layer.bias.numel()
            rows.append(dict(layer=f'{name}.Linear{len(rows)}',
                             shape=[layer.out_features, layer.in_features],
                             weights=layer.weight.numel(), biases=layer.bias.numel(),
                             params=n))
            tot += n
    return rows, tot


def count(K):
    torch.manual_seed(0)
    m = CondPFN(C_KIN, NTHETA, K=K)
    phi_rows, phi_tot = breakdown(m.phi, 'Phi')
    a_rows, a_tot = breakdown(m.A, 'F')
    b_rows, b_tot = breakdown(m.B, 'B')
    event_side = phi_tot + a_tot
    return dict(K=K, input_dim=C_KIN, phi_sizes=list(PHI_SIZES), f_sizes=list(F_SIZES),
                layers=phi_rows + a_rows, per_particle_Phi=phi_tot, event_head_F=a_tot,
                event_side_total=event_side, theta_side_B_excluded=b_tot,
                full_model_incl_B=event_side + b_tot)


def main():
    out = {}
    for K in (1, 24):
        r = count(K)
        out[f'K{K}'] = r
        print(f'--- CondPFN event side, {C_KIN} features, K={K} ---')
        for row in r['layers']:
            print(f'  {row["layer"]:>12} {str(row["shape"]):>12}: {row["params"]:>6}')
        print(f'  {"Phi total":>25}: {r["per_particle_Phi"]:>6}')
        print(f'  {"F total":>25}: {r["event_head_F"]:>6}')
        print(f'  {"EVENT SIDE":>25}: {r["event_side_total"]:>6}')
    out['published'] = 82457
    out['published_matches'] = 'K1'
    out['note'] = ('the published 82457 is the event side at K=1, the matched crosscheck '
                   'configuration (equal to the energyflow PFN with output_dim=1: Phi '
                   '36456 + F 45900 + output 101); the K=24 production event side holds '
                   f'{out["K24"]["event_side_total"]} (one 100->24 head layer instead of '
                   '100->1)')
    assert out['K1']['event_side_total'] == 82457, out['K1']['event_side_total']
    json.dump(out, open('output/param_count.json', 'w'), indent=1)
    print('-> output/param_count.json')


if __name__ == '__main__':
    main()
