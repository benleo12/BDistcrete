#!/usr/bin/env python3
"""Archive the released-intercept fit of the toy accuracy law (paper Sec. 5.1: "releasing
the intercept instead returns the same N* with the intercept landing on the measured null").

Extends output/kscan_toy_law_ruler.json IN PLACE with a top-level key 'released_intercept'
holding {Nstar, intercept, R2} for both estimators (paper ruler and old kscan ruler). The
fit is the two-parameter least squares chi2/ndf = c + N/N* on the STORED clean-arm points
(law.N_clean vs law.chi2_clean_paper / law.chi2_clean_old); no retraining and no
re-evaluation happen here. The identical fit exists in toy_law_ruler_merge.py as free_fit();
the recomputation is cross-checked against the stored law.free_fit_* triples when present.
"""
import json
import numpy as np

FN = 'output/kscan_toy_law_ruler.json'


def free_fit(N, y):
    """Two-parameter LS fit chi2 = c + N/N* (toy_law_ruler_merge.free_fit verbatim)."""
    N = np.asarray(N, float); y = np.asarray(y, float)
    A = np.vstack([N, np.ones_like(N)]).T
    (sl, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ [sl, c]
    r2 = 1 - float(((y - pred)**2).sum())/float(((y - y.mean())**2).sum())
    return (1.0/sl if sl > 0 else float('nan')), float(c), r2


def main():
    d = json.load(open(FN))
    law = d['law']
    N = law['N_clean']
    out = {}
    for est, key in [('paper', 'chi2_clean_paper'), ('old', 'chi2_clean_old')]:
        Ns, c, r2 = free_fit(N, law[key])
        out[est] = dict(Nstar=Ns, intercept=c, R2=r2)
        stored = law.get(f'free_fit_{est}')
        if stored is not None:
            assert np.allclose([Ns, c, r2], stored), (est, stored, (Ns, c, r2))
        print(f'{est:>5}: N* = {Ns/1e3:.1f}k  intercept = {c:.4f}  R^2 = {r2:.4f}'
              + ('  (matches stored free_fit)' if stored is not None else ''))
    out['note'] = ('two-parameter LS fit c + N/N* on the stored clean-arm chi2/ndf points; '
                   'paper-ruler intercept 0.858 lands on the measured null 0.856 '
                   '(law.null_chi2_paper mean) and N*=89.4k equals the null-corrected 89.4k')
    d['released_intercept'] = out
    json.dump(d, open(FN, 'w'), indent=1)
    print(f'-> {FN} extended in place with key "released_intercept"')


if __name__ == '__main__':
    main()
