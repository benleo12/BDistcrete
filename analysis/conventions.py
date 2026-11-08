#!/usr/bin/env python3
"""The hadron level of each measurement (D8): per-generator factors that take a prediction of our samples to the
hadron level a measurement is corrected to. Each generator's prediction is multiplied by its own factors after the
calculation is imposed and before the two generators are mixed, so the mixture stays linear in xi:

    corrected = (1 - xi) f_S * y_S + xi f_H * y_H,

with f_g a factor per bin of a distribution or per mean, measured at the default tunes (the conventions task).
Our samples are convention a: hadrons with c tau > 10 mm stable, neutrinos left out, ISR on with the photons radiated
by the beams removed. The comparisons of the paper use

    DELPHI thrust distribution (d11-x01-y02)   c/a     hadrons with lifetimes below 1 ns decayed
    OPAL <1-T>                                 cnu/a   those decays, and the neutrinos included
    L3 <n_ch>                                  none    L3 quotes it without the K0S and Lambda decay products

The factors file, factors.json, is a JSON object with "meta", "sherpa" and "herwig". "meta" defines the conventions
(a, c = lifetimes below 1 ns decayed, nu = neutrinos included, cnu = both, b = ISR off) and records the sample sizes
and the source. Each generator holds "delphi_y02" and "aleph_d54", each with "lo" and "hi" (bin edges) and the arrays
c_over_a, nu_over_a, cnu_over_a and b_over_a with a matching _err array each, and "means": {name: {"c_over_a": x,
"c_over_a_err": e, ...}} for one_minus_thrust, B_total, rho_heavy, P_win, tau_win, nch, nbaryon and the fourteen
window moments of thrust_targets_nlo_point.npz. Every entry is a plain ratio (not ratio - 1), and every distribution
a fraction of all events.

The file is looked for at CONV_FACTORS (when set), then /mnt/beegfs/bassi/reweighting/lmx_runs/conventions/
factors.json, then analysis/conventions/factors.json next to this script; load() fails with the places it looked
when none exists.

    python conventions.py [factors.json]       # checks a file and prints its factors
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLACES = ('/mnt/beegfs/bassi/reweighting/lmx_runs/conventions/factors.json', os.path.join(HERE, 'conventions', 'factors.json'))
GENS = ('sherpa', 'herwig')
GEN_OF = {'S': 'sherpa', 'H': 'herwig'}
KINDS = ('c_over_a', 'nu_over_a', 'cnu_over_a', 'b_over_a')
TABLES = ('delphi_y02', 'aleph_d54')
MOMENTS = ('logt', 'tau', 'tlogt', 'log2t', 'tau2', 'tlog2t', 't2logt', 'log3t', 'tau3', 'tlog3t', 't3logt', 't2log2t',
           'log4t', 'tau4')
MEANS = ('one_minus_thrust', 'B_total', 'rho_heavy', 'P_win', 'tau_win', 'nch', 'nbaryon') + MOMENTS
# the names of the predictions in three_bands.py and envelope_test.py, by their names in the factors file
MEAN_KEY = {'tau_win': 'tau_win', 'pwin': 'P_win', 'thrust': 'one_minus_thrust', 'B_total': 'B_total',
            'rho_heavy': 'rho_heavy', 'nch': 'nch', 'nbaryon': 'nbaryon'}
# D8: the factor of every comparison with a measurement (None: no factor)
DELPHI_KIND = 'c_over_a'
MEAN_KIND = {'thrust': 'cnu_over_a',     # OPAL <1-T>
             'nch': None}               # L3 <n_ch>
COMPARED = {'thrust': 'OPAL <1-T> (cnu/a)', 'nch': 'L3 <n_ch> (no factor)'}


def find(path=None):
    """The factors file: path, else CONV_FACTORS, else the first of PLACES that exists."""
    cands = [path] if path else ([os.environ['CONV_FACTORS']] if os.environ.get('CONV_FACTORS') else list(PLACES))
    for p in cands:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError('no factors file for the hadron-level conventions (D8): looked for ' + ', '.join(map(str, cands))
                            + '. It is written by the conventions task; set CONV_FACTORS to another file, or CONVENTIONS=none '
                            'to run without the corrected comparisons.')


def problems(d):
    """Everything in the parsed file d that differs from the agreed format, as a list of strings."""
    out = []
    for k in ('meta',) + GENS:
        if k not in d:
            out.append(f'no "{k}"')
    for g in GENS:
        G = d.get(g, {})
        for t in TABLES:
            T = G.get(t)
            if T is None:
                out.append(f'{g}: no "{t}"'); continue
            lo, hi = np.asarray(T.get('lo', []), float), np.asarray(T.get('hi', []), float)
            if len(lo) == 0 or len(lo) != len(hi) or np.any(hi <= lo):
                out.append(f'{g}/{t}: bad edges lo/hi')
            for k in KINDS:
                for kk in (k, k + '_err'):
                    v = np.asarray(T.get(kk, []), float)
                    if len(v) != len(lo):
                        out.append(f'{g}/{t}: {kk} has {len(v)} entries for {len(lo)} bins')
                    elif kk == k and not np.all(np.isfinite(v) & (v > 0)):
                        out.append(f'{g}/{t}: {kk} not finite and positive')
        M = G.get('means', {})
        for nm in MEANS:
            if nm not in M:
                out.append(f'{g}/means: no "{nm}"'); continue
            for k in KINDS:
                for kk in (k, k + '_err'):
                    if kk not in M[nm]:
                        out.append(f'{g}/means/{nm}: no "{kk}"')
                    elif kk == k and not (np.isfinite(M[nm][kk]) and M[nm][kk] > 0):
                        out.append(f'{g}/means/{nm}: {kk} = {M[nm][kk]} not finite and positive')
    return out


class Factors:
    """The factors of one file. hist() and mean() return them for the bins and means the scripts use."""

    def __init__(self, path=None, strict=True):
        self.path = find(path)
        self.d = json.load(open(self.path))
        bad = problems(self.d)
        if bad and strict:
            raise ValueError(f'{self.path} is not in the agreed format of the factors file: ' + '; '.join(bad[:12])
                             + (f'; and {len(bad) - 12} more' if len(bad) > 12 else ''))
        self.problems = bad
        self.meta = self.d.get('meta', {})

    def hist(self, gen, table, kind, lo, hi):
        """The factors of the bins (lo, hi) of table for generator gen (sherpa, herwig, S or H): every bin must be one of
        the table's bins (edges to 1e-6)."""
        T = self.d[GEN_OF.get(gen, gen)][table]
        tl, th, v = np.asarray(T['lo'], float), np.asarray(T['hi'], float), np.asarray(T[kind], float)
        out = np.empty(len(lo))
        for i, (l, h) in enumerate(zip(np.asarray(lo, float), np.asarray(hi, float))):
            j = np.where((np.abs(tl - l) < 1e-6) & (np.abs(th - h) < 1e-6))[0]
            if len(j) != 1:
                raise ValueError(f'{self.path}: no bin [{l:g}, {h:g}) in {gen}/{table}')
            out[i] = v[j[0]]
        return out

    def mean(self, gen, name, kind):
        """The factor of one mean, by its name in the factors file or in the scripts (thrust, pwin, ...)."""
        return float(self.d[GEN_OF.get(gen, gen)]['means'][MEAN_KEY.get(name, name)][kind])

    def vectors(self, obk, lo, hi):
        """D8 for the predictions of three_bands.py: {S, H: factor vector} over (the means obk, then the DELPHI bins
        lo, hi), with c/a in the bins, cnu/a on <1-T> (OPAL) and 1 elsewhere (L3 <n_ch> takes no factor, and the
        other means are not compared with a measurement), and a record of what was applied."""
        vec = {}
        for g in ('S', 'H'):
            m = [self.mean(g, k, MEAN_KIND[k]) if MEAN_KIND.get(k) else 1.0 for k in obk]
            vec[g] = np.r_[np.asarray(m, float), self.hist(g, 'delphi_y02', DELPHI_KIND, lo, hi)]
        rec = dict(file=self.path, meta=self.meta, delphi=f'delphi_y02 {DELPHI_KIND}',
                   means={k: (MEAN_KIND[k] or 'none') for k in obk if k in MEAN_KIND},
                   compared={k: v for k, v in COMPARED.items() if k in obk},
                   note='applied to each generator after imposing and before mixing; the other means take no factor')
        return vec, rec


def load(path=None, strict=True):
    return Factors(path, strict)


def apply(per_gen, a, vec):
    """sum_g a_g vec_g * y_g: per_gen = {S, H: y_g}, a = {S, H: weight}, vec = {S, H: factors} (None: no factors)."""
    return sum(a[g]*(per_gen[g] if vec is None else vec[g]*per_gen[g]) for g in per_gen if a.get(g, 0) != 0)


if __name__ == '__main__':
    F = load(sys.argv[1] if len(sys.argv) > 1 else None, strict=False)
    print(f'{F.path}: {len(F.problems)} problems with the format' + (':' if F.problems else ''))
    for p in F.problems:
        print('  ', p)
    print('meta:', json.dumps(F.meta)[:600])
    for g in GENS:
        T = F.d.get(g, {}).get('delphi_y02')
        if T:
            print(f'{g} delphi_y02 {DELPHI_KIND}:', ' '.join(f'{v:.4f}' for v in T[DELPHI_KIND]))
        for k, kind in MEAN_KIND.items():
            if kind:
                print(f'{g} {MEAN_KEY[k]} {kind}: {F.mean(g, k, kind):.5f}')
    sys.exit(1 if F.problems else 0)
