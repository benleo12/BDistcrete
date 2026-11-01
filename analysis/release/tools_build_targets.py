#!/usr/bin/env python3
"""Build targets_thrust_nlo.npz alone, from output/thrust_targets_nlo_point.npz, when the calculation changes and
the reference sample does not. The arrays and the metadata are those of tools_build_reference.py. Nothing is
recomputed: every array is copied, and the script proves it before it writes and again on what it reads back.

    python tools_build_targets.py ../output/thrust_targets_nlo_point.npz targets_thrust_nlo.npz
"""
import sys, json, hashlib, datetime, os
import numpy as np
SRC, OUT = sys.argv[1:3]
POWERS = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0),
          'tlog2t': (1, 2), 't2logt': (2, 1), 'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3),
          't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
FLOOR_REL = 0.002
def sha_array(v):
    a = np.asarray(v); h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes()); return h.hexdigest()
G = np.load(SRC, allow_pickle=False)
names = [str(k) for k in G['keys']]
assert set(names) == set(POWERS), 'moment names differ from targets_nlo_point.py'
assert len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'not a single-point calculation'
T = dict(moment_names=G['keys'], moment_powers=np.array([POWERS[k] for k in names], np.int64),
         central=G['central'][0, 0], sigma_pert=G['sigma_pert'][0, 0], window=G['window'],
         floor_rel=np.float64(FLOOR_REL), alpha_s=G['alphas'][0], alpha_0=G['alpha0'][0], pwin=G['pwin'][0, 0],
         scale_variations=G['scale'][0, 0], scale_labels=G['scale_labels'], scale_pwin=G['pwin_scale'][0, 0],
         np_variations=G['np_vars'][0, 0], np_labels=G['np_labels'],
         milan=G['milan'], sub_order_central=G['sub_order_central'], order=G['order'], calc_point=G['calc_point'])
labels = [str(v) for v in G['scale_labels']]
pairs = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
D = [np.sqrt(0.5)*(G['scale'][0, 0, labels.index(l)] - G['central'][0, 0]) for p in pairs for l in p]
D.append(G['scale'][0, 0, labels.index("(1.0, 1.0, 'modR', 1.0)")] - G['central'][0, 0]); D = np.stack(D, 1)
rec = float(np.max(np.abs(D @ D.T - G['sigma_pert'][0, 0]))/np.max(np.abs(G['sigma_pert'][0, 0])))
assert rec < 1e-12, f'sigma_pert is not the documented sum over the variations ({rec:.1e})'
doc = dict(name='targets_thrust_nlo', version='1.1', built=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
    build_script_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
    source={os.path.basename(SRC): hashlib.sha256(open(SRC, 'rb').read()).hexdigest()},
    calculation=('NNLL+NLO thrust with a dispersive power correction, at the published pair (alpha_s, alpha_0) of Gehrmann, Luisoni '
                 'and Monni, Eur. Phys. J. C 73 (2013) 2265, log-R, with their subtraction to O(alpha_s^3) (output/calc_point.json, targets_nlo_point.py)'),
    keys=dict(moment_names='the 14 moments, in order', moment_powers='(a, b) of each moment: f(tau) = tau^a (ln tau)^b, tau clipped at 1e-6',
        central='c_k = <f_k(tau)> over window[0] <= tau < window[1], normalized inside the window',
        sigma_pert=('Sigma_c of the calculation, sum over d d^T with d the scale variation minus central: the two log-R pairs '
                    '(muR x2 and x1/2, x_V x2 and x1/2) at weight 1/2 each, the mod-R scheme at central scales at weight 1'),
        window='[low, high) in tau = 1 - T, the only events the calculation constrains',
        floor_rel="the paper's diagonal floor: the maximum-entropy step uses Sigma = sigma_pert + diag((floor_rel*|central|)^2)",
        alpha_s='alpha_s(m_Z) of the point', alpha_0='alpha_0 of the dispersive model at the point',
        pwin='the calculated window fraction (not imposed by the paper)',
        scale_variations='the moments for each scale and scheme variation at the same point, rows scale_labels',
        scale_labels='(x_muR, x_V, matching scheme, x_NNLO) of each row', scale_pwin='the window fraction of each scale variation',
        np_variations='the moments for each variation of the dispersive model, rows np_labels',
        np_labels='Milan factor +20 percent, -20 percent, the renormalon subtraction at the other order',
        milan='the central Milan factor', sub_order_central='order of the central renormalon subtraction',
        order='perturbative order of the fixed-order part (2: NLO)', calc_point='the point and its source, JSON'),
    arrays={k: dict(dtype=str(np.asarray(v).dtype), shape=list(np.shape(v)), sha256=sha_array(v)) for k, v in T.items()})
T['meta'] = np.array(json.dumps(doc, indent=1))
tmp = OUT + '.tmp.npz'; np.savez(tmp, **T)
R = np.load(tmp, allow_pickle=False)
for k, v in doc['arrays'].items():
    assert sha_array(R[k]) == v['sha256'], k
assert all(np.array_equal(R[a], G[b][0, 0]) for a, b in (('central', 'central'), ('sigma_pert', 'sigma_pert'), ('scale_variations', 'scale'), ('np_variations', 'np_vars')))
os.replace(tmp, OUT)
print(f'wrote {OUT}: alpha_s {float(R["alpha_s"]):.4f}, alpha_0 {float(R["alpha_0"]):.3f}, pwin {float(R["pwin"]):.4f}, sub_order_central {int(R["sub_order_central"])}, np_labels {[str(x) for x in R["np_labels"]]}, {os.path.getsize(OUT)} bytes; Sigma rebuilt to {rec:.1e}')
