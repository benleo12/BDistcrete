"""R2b: wide-box reach study on the R1 full-event data (data_widebox), one grid, three
reference schemes, frozen recipe. alpha_s over [0.08,0.20]. Answers "how wide a box can one
reference cover" in closure units (no ESS):
  single-centre : reference = the centre run only (9010, alpha_s=0.14); sharp near centre,
                  fails at the far edges.
  pool-all      : reference = union of all 21 training runs; uniform but blurry.
  local-union   : reference = the reach-sized window of nearest runs, slid across the box.
Writes output/widebox_final.json + a per-target closure array for the figure."""
import json, numpy as np, torch, torch.nn as nn, pandas as pd
from r2_ladder import (CondPFN, build_feats, pulls, width_of, make_bins, emb_all,
                       fit_feature_norm, apply_feature_norm)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DATA = 'data_widebox'
ASV = [round(0.080 + 0.006*i, 4) for i in range(21)]                 # 0.080..0.200
TRAIN = {9000+i: ASV[i] for i in range(21)}
HELD = {9100: 0.083, 9101: 0.107, 9102: 0.131, 9103: 0.155, 9104: 0.179, 9105: 0.197}
CENTRE = 9010                                                        # alpha_s = 0.140
A0, ASC = 0.140, 0.060
def tn(a): return (a - A0) / ASC
OBS = ['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy']
ENS, K, STEPS, WIN = 4, 24, 9000, 7
# The three schemes must be compared at the SAME total reference statistics. Budgeting per
# RUN instead would give single-centre 1 run, local-union 7 and pool-all 21, i.e. a 21x range
# of reference size, and the pull denominator carries the reference error, so the comparison
# would be decided by sample size rather than by where the reference sits in the box.
NREF_TOTAL = 63000


def load():
    F = {}; M = {}; SH = {}
    for rid in list(TRAIN) + list(HELD):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        f, m = build_feats(d['particles'], d['mask'], flavor=False)
        F[rid] = f; M[rid] = m; SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
    # standardize with statistics pooled over the TRAINING runs, applied to every run,
    # matching the ladder recipe (per-run standardization would erase the signal)
    FMU, FSD = fit_feature_norm([F[r] for r in TRAIN], [M[r] for r in TRAIN])
    for rid in F:
        F[rid] = apply_feature_norm(F[rid], M[rid], FMU, FSD)
    return F, M, SH


def train_eval(ref_rids, F, M, SH, targets, tnf=tn, train_rids=None, dump_tag=None):
    # equal TOTAL reference budget for every scheme
    per = max(1, NREF_TOTAL // len(ref_rids))
    rng = np.random.default_rng(0)
    rf, rm, robs = [], [], {o: [] for o in OBS}
    for rid in ref_rids:
        n = len(F[rid]); idx = rng.choice(n, min(per, n), replace=False)
        rf.append(F[rid][idx]); rm.append(M[rid][idx])
        for o in OBS: robs[o].append(SH[rid][o].values[idx])
    ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
    robs = {o: np.concatenate(v) for o, v in robs.items()}; Nref = len(ref_f)
    # a local-union reference can only support targets inside its own window, so the training
    # targets must match the reference, otherwise far-box runs are presented at |tnf| >> 1
    train_rids = list(TRAIN) if train_rids is None else list(train_rids)
    models = []
    for sd in range(ENS):
        torch.manual_seed(sd); m = CondPFN(ref_f.shape[-1], 1, K=K).to(DEV)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
        g = torch.Generator().manual_seed(sd)
        bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
        for st in range(STEPS):
            j = train_rids[int(torch.randint(len(train_rids), (1,), generator=g))]
            ir = torch.randint(Nref, (1024,), generator=g); it = torch.randint(len(F[j]), (1024,), generator=g)
            fb = torch.cat([ref_f[ir], F[j][it].to(DEV)]); mb = torch.cat([ref_m[ir], M[j][it].to(DEV)])
            th = torch.full((2048, 1), tnf(TRAIN[j]), device=DEV)
            loss = bce(m.f_from(m.emb(fb, mb), th), lab); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval(); models.append(m)
    Es = [emb_all(m, ref_f, ref_m) for m in models]
    out = {}; ctrl = {}
    uni = np.full(Nref, 1.0/Nref)
    for rid in targets:
        a = HELD[rid]; th = torch.full((Nref, 1), tnf(a), device=DEV)
        with torch.no_grad():
            f = np.mean([m.f_from(E, th).cpu().numpy() for m, E in zip(models, Es)], 0)
        w = np.exp(f - f.max()); w /= w.sum(); gs = []; gu = []; nbtot = 0
        for o in OBS:
            ot = SH[rid][o].values
            bins = make_bins(np.r_[robs[o], ot], o)
            p, nb = pulls(robs[o], w, ot, bins)
            pu, _ = pulls(robs[o], uni, ot, bins)
            gs.append(p); gu.append(pu); nbtot += nb
        # same ruler as the ladder: sqrt(chi^2/ndf) about zero, whose null value is 1
        out[a] = width_of(np.concatenate(gs), nbtot)
        ctrl[a] = dict(N_eff=float(1.0/np.sum(w**2)), N_ref=int(Nref),
                       width_uniform=width_of(np.concatenate(gu), nbtot))
        if dump_tag is not None:
            np.save(f'output/wb_weights_{dump_tag}_{a:g}.npy', w)
    return out, ctrl


def main():
    print(f'device={DEV}')
    F, M, SH = load()
    res = {}
    print('-- single-centre --')
    res['single_centre'], res['single_centre_ctrl'] = train_eval([CENTRE], F, M, SH, list(HELD), dump_tag='single')
    print('-- pool-all --')
    res['pool_all'], res['pool_all_ctrl'] = train_eval(list(TRAIN), F, M, SH, list(HELD))
    print('-- local-union (per target, window of nearest runs) --')
    loc = {}
    for rid, a in HELD.items():
        near = sorted(TRAIN, key=lambda r: abs(TRAIN[r]-a))[:WIN]
        amin = min(TRAIN[r] for r in near); amax = max(TRAIN[r] for r in near)
        ac = (amin+amax)/2; asc = max((amax-amin)/2, 1e-3)
        lw, lc = train_eval(near, F, M, SH, [rid], tnf=lambda x, ac=ac, asc=asc: (x-ac)/asc,
                            train_rids=near, dump_tag='union')
        loc[a] = lw[a]; res.setdefault('local_union_ctrl', {})[a] = lc[a]
        print(f'   a={a}: {loc[a]:.2f}')
    res['local_union'] = loc
    res['summary'] = {k: round(float(np.mean(list(v.values()))), 2) for k, v in
                      [('single_centre', res['single_centre']), ('pool_all', res['pool_all']),
                       ('local_union', res['local_union'])]}
    json.dump(res, open('output/widebox_ctrl.json', 'w'), indent=1)
    print('WIDEBOX:', res['summary'])


if __name__ == '__main__':
    main()
