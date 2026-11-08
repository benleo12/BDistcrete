#!/usr/bin/env python3
"""Hadron-level convention factors of Sherpa and Herwig at their default tunes: factors.json.

Reads the per-event tables of the convention samples (lmx_phaseA/conventions, made by conv_process.py:
data/<gen>_<conv>/chunk_*.npz and chunk_*.json) and writes, for each generator, the ratios c/a, nu/a,
cnu/a and b/a with statistical errors, per DELPHI d11-x01-y02 bin, per ALEPH d54 bin and for the means.

Conventions. Every event shape has the beam photons removed (decision D4: status-1 photons with
pT <= 1e-9 E, the generators' initial-state-radiation photons; none exist with ISR off).
  a    the training samples: ISR on, hadrons with c*tau > 10 mm stable, neutrinos left out
  c    a with every hadron of lifetime below 1 ns decayed (c*tau < 299.79 mm: K0S, Lambda, Sigma+, Sigma-,
       Xi0, Xi-, Omega- and their antiparticles; nothing else changes status)
  nu   a with the neutrinos kept in every observable
  cnu  c and nu together
  b    a with ISR off
Distributions are fractions of all events (D3): a bin holds the fraction of all events in it, and a
ratio of two conventions is the ratio of these fractions (or densities; the bin width cancels).
Means are over all events. tau_win and the 14 window moments are means over the events inside the
window 0.05 <= tau < 1/3, as the calculation's targets (thrust_targets_nlo_point.npz) and
directlib.Model.observables define them. P_win is the fraction of all events in the window.

Estimators (statistical errors from per-event influence functions, as in analyze_conv.py):
  c/a    same events: the c sample against the same events with the decays undone (its 'col'
         columns, which rebuild convention a exactly)
  cnu/a  same events: the c sample with neutrinos ('_nu' columns) against its 'col' columns
  nu/a   same events: the a sample with and without neutrinos, pooled with the c sample's 'colnu'
         and 'col' columns (also convention a with and without neutrinos)
  b/a    independent samples: b pooled with the d sample's 'col' columns, against a pooled with the
         c sample's 'col' columns
  nbaryon: c/a = nu/a = cnu/a = 1 exactly, with no error: every hadron that c decays leaves exactly
         one baryon (Lambda, Sigma, Xi, Omega) or none (K0S), and neutrinos are not baryons. b/a comes
         from the per-chunk tallies (the npz has no per-event baryon count), with errors from the
         chunk-to-chunk scatter; the a sample against the c sample checks the exact c/a = 1.

  python3 -B make_factors.py --data DIR --rivet DIR --targets FILE --out DIR
"""
import argparse, datetime, glob, hashlib, json, os, re
import numpy as np

GENS = ('sherpa', 'herwig')
MEANS = ['one_minus_thrust', 'B_total', 'rho_heavy', 'P_win', 'tau_win', 'nch']     # nbaryon separately
COL = {'one_minus_thrust': 'tau', 'B_total': 'bt', 'rho_heavy': 'rho', 'nch': 'nch'}
MINCOUNT = 25                                          # a bin with fewer events in any sample is flagged
LONG = (310, 3122, 3222, 3112, 3322, 3312, 3334)
NEED = [p + s for p in ('tau', 'bt', 'rho') for s in ('', '_nu', '_col', '_colnu')] + ['nch', 'nch_col', 'ncol']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def yoda_table(path, ana, tab, in_T):
    """Bin edges in tau = 1-T (ascending), values and total relative errors of one YODA scatter."""
    txt = open(path).read()
    m = re.search(rf'/REF/{ana}/{tab}\n(.*?)\nEND YODA', txt, re.S)
    rows = []
    for ln in m.group(1).split('\n'):
        f = ln.split()
        if len(f) == 6 and not ln.startswith('#'):
            try:
                rows.append([float(x) for x in f])
            except ValueError:
                pass
    x, dxm, dxp, y, em, ep = np.array(rows).T
    lo, hi = (1 - (x + dxp), 1 - (x - dxm)) if in_T else (x - dxm, x + dxp)
    o = np.argsort(lo); lo, hi = np.round(lo[o], 9), np.round(hi[o], 9)
    assert np.allclose(lo[1:], hi[:-1])
    return np.r_[lo, hi[-1]], y[o], (0.5*(np.abs(em) + np.abs(ep))/y)[o]


def load(d):
    """All chunks of one sample: per-event columns (float64) and the per-chunk tallies."""
    files = sorted(glob.glob(f'{d}/chunk_*.npz'))
    cols = {}
    chunks = []
    for f in files:
        z = np.load(f)
        for k in NEED:
            cols.setdefault(k, []).append(z[k])
        J = json.load(open(f[:-4] + '.json'))
        n = J['n_events']; assert n == len(z['tau'])
        fs = {}
        for k, v in J['fs_pid'].items():                    # signed pid -> count; sum both signs
            a = abs(int(k)); fs[a] = fs.get(a, 0) + int(v)
        nbar = sum(v for a, v in fs.items() if 1000 <= a < 1000000)
        chunks.append(dict(seed=int(os.path.basename(f)[6:-4]), n=n, nbaryon=nbar,
                           nlong=sum(fs.get(a, 0) for a in LONG), fs=fs))
    S = {k: np.concatenate(v) for k, v in cols.items()}
    for k in list(S):
        if S[k].dtype.kind in 'fi':
            S[k] = S[k].astype(np.float64)
    return S, chunks


def variant(S, v):
    """The observables of one convention realized by one sample's columns.
    v: 'plain' (the sample's own convention), 'nu' (with neutrinos), 'col' (decays undone), 'colnu'."""
    suf = {'plain': '', 'nu': '_nu', 'col': '_col', 'colnu': '_colnu'}[v]
    nch = S['nch_col'] if v.startswith('col') else S['nch']       # neutrinos carry no charge
    return dict(tau=S['tau' + suf], bt=S['bt' + suf], rho=S['rho' + suf], nch=nch)


def pool(*Vs):
    return {k: np.concatenate([V[k] for V in Vs]) for k in Vs[0]}


def funs(t, keys):
    t = np.clip(t, 1e-6, None); L = np.log(t)
    F = {'logt': L, 'tau': t, 'tlogt': t*L, 'log2t': L**2, 'tau2': t**2, 'tlog2t': t*L**2, 't2logt': t**2*L,
         'log3t': L**3, 'tau3': t**3, 'tlog3t': t*L**3, 't3logt': t**3*L, 't2log2t': t**2*L**2, 'log4t': L**4,
         'tau4': t**4}
    return {k: F[k] for k in keys}


class Obs:
    """Every scalar of one convention on one set of events, as (numerator, denominator) per event:
    the estimate is mean(num)/mean(den), its influence (num - X den)/mean(den)."""
    def __init__(self, V, win, keys, u=None):
        self.V, self.keys, self.N = V, keys, len(V['tau'])
        self.u = np.ones(self.N) if u is None else u
        t = V['tau']; self.w = ((t >= win[0]) & (t < win[1])).astype(np.float64)

    def parts(self, name):
        if name in COL:
            return self.V[COL[name]], None
        if name == 'P_win':
            return self.w, None
        if name == 'tau_win':
            return self.V['tau']*self.w, self.w
        return funs(self.V['tau'], [name])[name]*self.w, self.w

    def get(self, name):
        a, b = self.parts(name); u = self.u
        mb = (u if b is None else u*b).mean()
        x = (u*a).mean()/mb
        psi = u*(a - x*(1.0 if b is None else b))/mb
        return float(x), psi


def var_of(psi):
    return float((psi**2).sum()/len(psi)**2)


def r_same(E1, E2, name):
    x1, p1 = E1.get(name); x2, p2 = E2.get(name); r = x1/x2
    return r, float(np.sqrt(var_of((p1 - r*p2)/x2)))


def r_ind(E1, E2, name):
    x1, p1 = E1.get(name); x2, p2 = E2.get(name); r = x1/x2
    return r, float(abs(r)*np.sqrt(var_of(p1)/x1**2 + var_of(p2)/x2**2))


def bin_index(t, edges):
    i = np.searchsorted(edges, t, side='right') - 1
    i[(t < edges[0]) | (t >= edges[-1])] = -1
    return i


def bins_same(t1, t2, edges):
    """Ratio of the fractions of all events per bin, two conventions of the same events:
    var(r) = (n1 + r^2 n2 - 2 r n12)/n2^2."""
    K = len(edges) - 1
    i1, i2 = bin_index(t1, edges), bin_index(t2, edges)
    n1 = np.bincount(i1[i1 >= 0], minlength=K).astype(float); n2 = np.bincount(i2[i2 >= 0], minlength=K).astype(float)
    both = (i1 == i2) & (i1 >= 0)
    n12 = np.bincount(i1[both], minlength=K).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = n1/n2
        e = np.sqrt(np.maximum(n1 + r**2*n2 - 2*r*n12, 0))/n2
    return r, e, n1, n2


def bins_ind(t1, t2, edges):
    """The same for two independent samples (binomial errors of each fraction)."""
    K = len(edges) - 1
    i1, i2 = bin_index(t1, edges), bin_index(t2, edges)
    n1 = np.bincount(i1[i1 >= 0], minlength=K).astype(float); n2 = np.bincount(i2[i2 >= 0], minlength=K).astype(float)
    p1, p2 = n1/len(t1), n2/len(t2)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = p1/p2
        e = r*np.sqrt((1 - p1)/n1 + (1 - p2)/n2)
    return r, e, n1, n2


def tally_mean(chunks):
    """Mean per event of a per-chunk total, with the ratio-estimator variance from the chunk scatter."""
    n = np.array([c['n'] for c in chunks], float); T = np.array([c['nbaryon'] for c in chunks], float)
    m = T.sum()/n.sum(); K = len(n)
    v = K/(K - 1)*((T - m*n)**2).sum()/n.sum()**2
    return float(m), float(np.sqrt(v))


def clean(x):
    """JSON-safe float (None for nan or inf)."""
    x = float(x)
    return x if np.isfinite(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/beegfs/bassi/reweighting/lmx_phaseA/conventions/data')
    ap.add_argument('--rivet', default='/mnt/beegfs/bassi/reweighting/maxent_code/rivet_ref')
    ap.add_argument('--targets', default='/mnt/beegfs/bassi/reweighting/maxent_code/output/thrust_targets_nlo_point.npz')
    ap.add_argument('--out', default='/mnt/beegfs/bassi/reweighting/lmx_phaseB/conventions/results')
    ap.add_argument('--data-label', default=None, help='where the data live, for the meta block (default: --data)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(a.targets, allow_pickle=True)
    KEYS = [str(k) for k in z['keys']]
    WIN = (float(z['window'][0]), float(z['window'][1]))
    calc_c = z['central'][0, 0]; calc_sp = np.sqrt(np.diag(z['sigma_pert'][0, 0]))
    NAMES = MEANS + KEYS

    BIN = {}
    dpath = f'{a.rivet}/DELPHI_1996_S3430090.yoda'; apath = f'{a.rivet}/ALEPH_2004_S5765862.yoda'
    BIN['delphi_y02'] = yoda_table(dpath, 'DELPHI_1996_S3430090', 'd11-x01-y02', False)
    BIN['aleph_d54'] = yoda_table(apath, 'ALEPH_2004_S5765862', 'd54-x01-y01', True)

    out = {}
    meta_samples, meta_counts, checks, values = {}, {}, {}, {}
    for g in GENS:
        S, CH = {}, {}
        for cv in 'abcd':
            S[cv], CH[cv] = load(f'{a.data}/{g}_{cv}')
        meta_samples[g] = {cv: dict(events=int(len(S[cv]['tau'])), chunks=len(CH[cv]),
                                    seeds=[min(c['seed'] for c in CH[cv]), max(c['seed'] for c in CH[cv])])
                           for cv in 'abcd'}
        # the conventions realized by the samples' columns
        Va, Vanu = variant(S['a'], 'plain'), variant(S['a'], 'nu')
        Vc, Vcnu = variant(S['c'], 'plain'), variant(S['c'], 'nu')
        Vca, Vcanu = variant(S['c'], 'col'), variant(S['c'], 'colnu')     # c events as a, as nu
        Vb, Vdb = variant(S['b'], 'plain'), variant(S['d'], 'col')         # b, and d events as b
        Vd = variant(S['d'], 'plain')
        A, Anu = pool(Va, Vca), pool(Vanu, Vcanu)                          # same events, a and nu
        B = pool(Vb, Vdb)
        E = {k: Obs(V, WIN, KEYS) for k, V in dict(c=Vc, cnu=Vcnu, a_c=Vca, A=A, Anu=Anu, B=B, a=Va, b=Vb,
                                                    d=Vd, b_d=Vdb).items()}
        est = {'c_over_a': lambda n: r_same(E['c'], E['a_c'], n),
               'nu_over_a': lambda n: r_same(E['Anu'], E['A'], n),
               'cnu_over_a': lambda n: r_same(E['cnu'], E['a_c'], n),
               'b_over_a': lambda n: r_ind(E['B'], E['A'], n)}
        G = {'means': {}}
        for n in NAMES:
            G['means'][n] = {}
            for rk, f in est.items():
                r, e = f(n)
                G['means'][n][rk] = r; G['means'][n][rk + '_err'] = e
        # nbaryon: exact for c, nu, cnu; ISR from the tallies
        mA, eA = tally_mean(CH['a'] + CH['c']); mB, eB = tally_mean(CH['b'] + CH['d'])
        rb = mB/mA
        G['means']['nbaryon'] = dict(c_over_a=1.0, c_over_a_err=0.0, nu_over_a=1.0, nu_over_a_err=0.0,
                                     cnu_over_a=1.0, cnu_over_a_err=0.0, b_over_a=rb,
                                     b_over_a_err=float(rb*np.hypot(eA/mA, eB/mB)))
        # order the means as asked: the seven named first, then the 14 moments
        order = ['one_minus_thrust', 'B_total', 'rho_heavy', 'P_win', 'tau_win', 'nch', 'nbaryon'] + KEYS
        G['means'] = {k: G['means'][k] for k in order}
        # bins
        meta_counts[g] = {}
        for bn, (ed, ydat, rel) in BIN.items():
            r_c, e_c, n_c, n_ac = bins_same(Vc['tau'], Vca['tau'], ed)
            r_n, e_n, n_nu, n_A = bins_same(Anu['tau'], A['tau'], ed)
            r_cn, e_cn, n_cnu, _ = bins_same(Vcnu['tau'], Vca['tau'], ed)
            r_b, e_b, n_B, n_A2 = bins_ind(B['tau'], A['tau'], ed)
            assert np.array_equal(n_A, n_A2)
            G[bn] = dict(lo=[float(x) for x in ed[:-1]], hi=[float(x) for x in ed[1:]],
                         c_over_a=[clean(x) for x in r_c], c_over_a_err=[clean(x) for x in e_c],
                         nu_over_a=[clean(x) for x in r_n], nu_over_a_err=[clean(x) for x in e_n],
                         cnu_over_a=[clean(x) for x in r_cn], cnu_over_a_err=[clean(x) for x in e_cn],
                         b_over_a=[clean(x) for x in r_b], b_over_a_err=[clean(x) for x in e_b])
            low = np.minimum.reduce([n_c, n_ac, n_cnu, n_A, n_B]) < MINCOUNT
            meta_counts[g][bn] = dict(events_a_pooled=n_A.astype(int).tolist(), events_b_pooled=n_B.astype(int).tolist(),
                                      events_c=n_c.astype(int).tolist(), events_a_from_c=n_ac.astype(int).tolist(),
                                      events_cnu=n_cnu.astype(int).tolist(), low_stats=low.tolist(),
                                      data_rel_err=[float(x) for x in rel])
        out[g] = G
        # checks: the same convention from independent samples; the decay factor without ISR; direct estimators
        ck = {}
        ck['a vs a rebuilt from c (means, ratio)'] = {n: list(r_ind(E['a'], E['a_c'], n)) for n in NAMES}
        ck['b vs b rebuilt from d (means, ratio)'] = {n: list(r_ind(E['b'], E['b_d'], n)) for n in NAMES}
        ck['d/b same events (decays without ISR, means)'] = {n: list(r_same(E['d'], E['b_d'], n)) for n in NAMES}
        ck['direct c/a: c sample against a sample (means)'] = {n: list(r_ind(E['c'], E['a'], n)) for n in NAMES}
        nu_a_only = {n: r_same(Obs(Vanu, WIN, KEYS), E['a'], n) for n in NAMES}
        nu_c_only = {n: r_same(Obs(Vcanu, WIN, KEYS), E['a_c'], n) for n in NAMES}
        ck['nu/a from the a sample alone, and from the c sample alone (means)'] = {
            n: [nu_a_only[n][0], nu_a_only[n][1], nu_c_only[n][0], nu_c_only[n][1]] for n in NAMES}
        mA_only, eA_only = tally_mean(CH['a']); mC_only, eC_only = tally_mean(CH['c'])
        mb_only, eb_only = tally_mean(CH['b']); md_only, ed_only = tally_mean(CH['d'])
        ck['nbaryon per event: a, c, b, d samples (tallies), and c/a, d/b from independent samples'] = dict(
            a=[mA_only, eA_only], c=[mC_only, eC_only], b=[mb_only, eb_only], d=[md_only, ed_only],
            c_over_a_independent=[mC_only/mA_only, mC_only/mA_only*np.hypot(eA_only/mA_only, eC_only/mC_only)],
            d_over_b_independent=[md_only/mb_only, md_only/mb_only*np.hypot(eb_only/mb_only, ed_only/md_only)])
        for bn, (ed, _, _) in BIN.items():
            def chi2_ind(t1, t2):
                r, e, n1, n2 = bins_ind(t1, t2, ed)
                ok = (n1 >= MINCOUNT) & (n2 >= MINCOUNT) & (e > 0)
                return [float((((r - 1)/e)[ok]**2).sum()), int(ok.sum())]
            ck[f'{bn}: chi2/ndf of a vs a rebuilt from c'] = chi2_ind(Va['tau'], Vca['tau'])
            ck[f'{bn}: chi2/ndf of b vs b rebuilt from d'] = chi2_ind(Vb['tau'], Vdb['tau'])
            r1, e1, _, _ = bins_same(Vc['tau'], Vca['tau'], ed); r2, e2, m1, m2 = bins_same(Vd['tau'], Vdb['tau'], ed)
            ok = (np.minimum(m1, m2) >= MINCOUNT) & (e1 > 0) & (e2 > 0)
            ck[f'{bn}: chi2/ndf of c/a (ISR on) against d/b (ISR off), independent'] = [
                float((((r1 - r2)**2/(e1**2 + e2**2))[ok]).sum()), int(ok.sum())]
            rdir, edir, nd1, nd2 = bins_ind(Vc['tau'], Va['tau'], ed)
            ok = (np.minimum(nd1, nd2) >= MINCOUNT) & (edir > 0)
            ck[f'{bn}: chi2/ndf of direct c/a (c sample vs a sample) against the same-event c/a'] = [
                float((((rdir - r1)/edir)[ok]**2).sum()), int(ok.sum())]
        checks[g] = ck
        values[g] = {k: {n: E[k].get(n)[0] for n in NAMES} for k in ('A', 'B', 'c', 'cnu', 'Anu')}
        values[g]['A']['nbaryon'] = mA; values[g]['B']['nbaryon'] = mB
        print(f'{g}: done', flush=True)

    # the calculation's own uncertainty, for the window-moment residuals
    calc = {k: dict(central=float(calc_c[i]), sigma_pert=float(calc_sp[i]),
                    sigma_with_floor=float(np.hypot(calc_sp[i], 0.002*abs(calc_c[i]))))
            for i, k in enumerate(KEYS)}

    data_md5 = {}
    for g in GENS:
        for cv in 'abcd':
            for f in sorted(glob.glob(f'{a.data}/{g}_{cv}/chunk_*')):
                data_md5[f'{g}_{cv}/{os.path.basename(f)}'] = md5(f)
    manifest = hashlib.md5(''.join(f'{v}  {k}\n' for k, v in sorted(data_md5.items())).encode()).hexdigest()

    meta = dict(
        title='Hadron-level convention factors of Sherpa and Herwig at their default tunes',
        made=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        script=dict(name='make_factors.py', md5=md5(os.path.abspath(__file__)),
                    command='python3 -B make_factors.py --data <DATA> --rivet <RIVET_REF> --targets '
                            '<thrust_targets_nlo_point.npz> --out <OUT>'),
        conventions=dict(
            a='our samples, as the training samples: initial-state radiation on, hadrons with c*tau > 10 mm stable '
              '(K0S, Lambda, Sigma+, Sigma-, Xi0, Xi-, Omega- and antiparticles stable), neutrinos excluded, '
              'beam photons removed (D4)',
            c='as a, but hadrons with lifetimes below 1 ns decayed (c*tau < 299.79 mm: K0S, Lambda, Sigma+, Sigma-, '
              'Xi0, Xi-, Omega- and antiparticles; no other species changes status); neutrinos excluded, ISR on, beam '
              'photons removed. This is DELPHI\'s stable-particle definition (proper lifetime above 1 ns stable); '
              'OPAL (3e-10 s) and L3 (3.3e-10 s) are equivalent in practice, since no hadron lies between 0.3 and 1 ns',
            nu='as a, but neutrinos included in every observable',
            cnu='c and nu together: hadrons with lifetimes below 1 ns decayed and neutrinos included',
            b='as a, but initial-state radiation off (no beam photons exist); 10 mm stable, neutrinos excluded',
            beam_photons='status-1 photons with pT <= 1e-9 E (Sherpa writes them with px = py = 0, Herwig with '
                         '|px| ~ 1e-19 GeV); the nearest other photon has pT/E >= 1e-5'),
        ratio='every entry is a plain ratio X/a (not X/a - 1) of the same observable in convention X and in '
              'convention a, each generator at its default tune; multiply a convention-a prediction by it',
        observables=dict(
            distributions='fractions of all events per bin (D3), no renormalization over a range of bins; a bin '
                          'ratio is the ratio of these fractions. Bins: delphi_y02 = DELPHI d11-x01-y02 (the '
                          'all-particle thrust, D5), aleph_d54 = ALEPH d54-x01-y01 (91.2 GeV, converted from T to '
                          'tau = 1-T); lo and hi are in tau',
            one_minus_thrust='mean of tau = 1-T over all events (exact thrust axis, compute_hemisphere_shapes of '
                             'gen/compute_efps.py)',
            B_total='mean total jet broadening over all events',
            rho_heavy='mean heavy hemisphere mass rho_H over all events',
            P_win='fraction of all events with 0.05 <= tau < 1/3',
            tau_win='mean of tau over the events with 0.05 <= tau < 1/3 (directlib tau_win; equal to the moment tau)',
            nch='mean charged multiplicity over all events, the pipeline rule: charge != 0 '
                '(extract_particles_full_flavor.charge) among the 100 most energetic non-neutrino status-1 '
                'particles; in c the decay products count; neutrinos carry no charge, so nu/a = 1 exactly',
            nbaryon='mean number of status-1 (anti)baryons (1000 <= |pid| < 1e6) per event, as the pipeline '
                    'nbaryon; c/a = nu/a = cnu/a = 1 exactly (each decayed hyperon leaves one baryon, K0S none, '
                    'neutrinos are not baryons); b/a from the per-chunk tallies',
            window_moments='the 14 keys of thrust_targets_nlo_point.npz: means of f(tau) over the events with '
                           '0.05 <= tau < 1/3 (conditional on the window, like the calculation\'s targets; '
                           'f = log tau, tau, tau log tau, ... with tau clipped at 1e-6 as in directlib.funs). The '
                           'ratio of a moment over all events, <f 1_win>, is this ratio times the P_win ratio'),
        estimators=dict(
            c_over_a='same events: the c sample against the same events with the decays of K0S, Lambda, Sigma, Xi '
                     'and Omega undone (conv_process.py col columns, convention a exactly)',
            cnu_over_a='same events: the c sample with neutrinos against its col columns (a without neutrinos)',
            nu_over_a='same events: the a sample with and without neutrinos, pooled with the c sample\'s colnu and '
                      'col columns (convention a rebuilt, with and without neutrinos)',
            b_over_a='independent samples: the b sample pooled with the d sample\'s col columns, against the a '
                     'sample pooled with the c sample\'s col columns',
            errors='statistical, from per-event influence functions (delta method): same-event ratios keep the '
                   'correlation; independent ratios add the two relative variances. nbaryon b/a: chunk-to-chunk '
                   'scatter of the per-chunk baryon totals (ratio estimator)',
            weights='unweighted, like the pipeline (Sherpa ISR-on weights change the means by at most 1.4e-4 '
                    'relative, lmx_phaseA/conventions/results/summary.txt)'),
        sample_sizes={g: dict(**{f'{cv}_events': meta_samples[g][cv]['events'] for cv in 'abcd'},
                              **{f'{cv}_chunks': meta_samples[g][cv]['chunks'] for cv in 'abcd'},
                              a_pooled_events=meta_samples[g]['a']['events'] + meta_samples[g]['c']['events'],
                              b_pooled_events=meta_samples[g]['b']['events'] + meta_samples[g]['d']['events'],
                              events_per_ratio=dict(
                                  c_over_a=meta_samples[g]['c']['events'], cnu_over_a=meta_samples[g]['c']['events'],
                                  nu_over_a=meta_samples[g]['a']['events'] + meta_samples[g]['c']['events'],
                                  b_over_a=[meta_samples[g]['b']['events'] + meta_samples[g]['d']['events'],
                                            meta_samples[g]['a']['events'] + meta_samples[g]['c']['events']]),
                              seed_range=[min(meta_samples[g][cv]['seeds'][0] for cv in 'abcd'),
                                          max(meta_samples[g][cv]['seeds'][1] for cv in 'abcd')])
                      for g in GENS},
        source=dict(
            samples='lmx_phaseA/conventions (Phase A): Sherpa 3.0.4 (card of gen/gen_stageE.sh) and Herwig 7.3.0p1 '
                    '(card of gen/gen_stageF.sh), e+e- -> hadrons at 91.2 GeV, each at the CENTRAL point of '
                    'analysis/symmetry/envelope_test.sbatch (= the default tunes); variants: Sherpa b PDF_LIBRARY: '
                    'None, c HADRON_DECAYS: {Max_Proper_Lifetime: 299.792458}; Herwig b e+/e- PDF NoPDF, '
                    'c DecayHandler:MaxLifeTime 299.792458*mm; d = b and c. Per-event tables of conv_process.py; '
                    'the HepMC records were deleted after reduction',
            data_dir=a.data_label or a.data, data_files=len(data_md5), data_manifest_md5=manifest,
            data_manifest_note='md5 of the sorted lines "<md5>  <gen>_<conv>/<file>" of all chunk files',
            rivet_ref={os.path.basename(dpath): md5(dpath), os.path.basename(apath): md5(apath)},
            targets=dict(file=os.path.basename(a.targets), md5=md5(a.targets),
                         note='only keys, window, central and sigma_pert are read')),
        window=list(WIN), moment_keys=KEYS,
        application=dict(
            note='decision D8: per-generator, per-bin factors, measured at the default tunes, applied to each '
                 'generator\'s prediction after imposing the calculation and before mixing with (1-xi, xi); the '
                 'uncorrected predictions are kept as well',
            DELPHI='c_over_a (hadrons with lifetimes below 1 ns decayed)', OPAL_mean_one_minus_thrust='cnu_over_a',
            L3_nch='no factor (L3 18.63 is without K0S and Lambda decay products, our convention)'),
        calc_uncertainty=dict(note='the calculation\'s own uncertainty of each window moment: sigma_pert from '
                                   'thrust_targets_nlo_point.npz, and with the 0.2% floor the tilt adds (directlib)',
                              moments=calc),
        bin_counts=meta_counts, mincount_flag=MINCOUNT, checks=checks, values_for_reference=values,
    )
    res = dict(meta=meta, sherpa=out['sherpa'], herwig=out['herwig'])
    with open(f'{a.out}/factors.json', 'w') as fh:
        json.dump(res, fh, indent=1, allow_nan=False)
    print(f'wrote {a.out}/factors.json', flush=True)


if __name__ == '__main__':
    main()
