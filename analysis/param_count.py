#!/usr/bin/env python3
"""Archive the event-side parameter count quoted in App. A.2 (the same 82457 adjustable coefficients
as the network of the energyflow package), previously stored nowhere.

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
 * K=24 : the head of an earlier draft. The event side then ends in a 100->24 layer instead
          of 100->1, adding 23*(100+1) = 2323 parameters for a total of 84780. The published
          82457 is therefore NOT the K=24 event side; it is the matched K=1 crosscheck
          configuration.
CPU-only. Writes output/param_count.json (PARAM_COUNT_OUT names another file) with the
layer-by-layer breakdown.

    python param_count.py [<trained trunk>_trunk.pt ...]

With trained trunks (the rerun gives the three Stage A trunks of output/models), the count is
also taken from the files: every tensor of each member's Phi and event head A, as trained (its
own inputs and K outputs), and the same layers with four inputs and one output. That must equal
the K=1 count above, so the networks the paper trained have the event side the text counts; the
script stops otherwise. Without arguments the output is as before.
"""
import os, json, sys
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


def from_trunk(path):
    """The event side of the members of a trained trunk (output/models/<tag>_trunk.pt), from its tensors"""
    sv = torch.load(path, map_location='cpu', weights_only=False)
    members = []
    for sd in sv['models']:
        ev = {k: tuple(v.shape) for k, v in sd.items() if k.startswith(('phi.', 'A.'))}
        lin = [k for k in ev if k.endswith('.weight') and len(ev[k]) == 2]          # the Linear layers, in order
        lin.sort(key=lambda k: (k.split('.')[0] != 'phi', int(k.split('.')[1])))
        first, last = lin[0], lin[-1]
        trained = sum(int(torch.Size(s).numel()) for s in ev.values())
        # the same layers with the four kinematic inputs and one output: the first weight has C_KIN
        # columns, the last weight one row and its bias one entry
        n_in, n_out = ev[first][1], ev[last][0]
        k1 = trained - ev[first][0]*(n_in - C_KIN) - (n_out - 1)*(ev[last][1] + 1)
        members.append(dict(layers=[[k, list(ev[k])] for k in lin], inputs=n_in, outputs=n_out,
                            event_side_trained=trained, event_side_K1_four_inputs=k1,
                            other_tensors=sorted(k for k in ev if k not in lin and not k.endswith('.bias'))))
    shapes = {json.dumps(m['layers']) for m in members}
    return dict(file=path, realpath=os.path.realpath(path), members=len(members), same_layers_in_every_member=len(shapes) == 1,
                K=sv.get('K'), phi_sizes=list(sv.get('phi_sizes', ())), f_sizes=list(sv.get('f_sizes', ())), **members[0])


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
                   '36456 + F 45900 + output 101); the event side with the K=24 head of an '
                   f'earlier draft holds {out["K24"]["event_side_total"]} (one 100->24 head layer '
                   'instead of 100->1)')
    assert out['K1']['event_side_total'] == 82457, out['K1']['event_side_total']
    if len(sys.argv) > 1:
        out['trained_trunks'] = []
        for p in sys.argv[1:]:
            t = from_trunk(p)
            out['trained_trunks'].append(t)
            print(f'--- {p}: {t["members"]} members, {t["inputs"]} inputs, {t["outputs"]} outputs, event side as trained '
                  f'{t["event_side_trained"]}, with four inputs and one output {t["event_side_K1_four_inputs"]}')
            assert t['same_layers_in_every_member'], f'{p}: the members have different layers'
            assert t['event_side_K1_four_inputs'] == out['K1']['event_side_total'], \
                f'{p}: {t["event_side_K1_four_inputs"]} event-side coefficients with one output, not {out["K1"]["event_side_total"]}'
    f = os.environ.get('PARAM_COUNT_OUT', 'output/param_count.json')
    json.dump(out, open(f, 'w'), indent=1)
    print(f'-> {f}')


if __name__ == '__main__':
    main()
