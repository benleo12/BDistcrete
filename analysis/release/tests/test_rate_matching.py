"""Normalization, correlated variations and full response of sequential rate matching."""
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gentune.maxent import Targets, load_targets, _Response


def main():
    rng = np.random.default_rng(51)
    tau = .5*rng.beta(1.2, 3.0, 4000)
    inside = (tau >= .05) & (tau < 1/3)
    f = np.column_stack([tau[inside], tau[inside]**2])
    c = np.array([.14, .026])
    calc = Targets(['tau', 'tau2'], c, np.diag([.008, .003])**2,
                   [.05, 1/3], variations={'shift': c+[.003, .001]},
                   rate=.45, rate_variations={'shift': .47}, meta={'alpha_s':.1137})
    lw0 = -8*tau+.3*rng.normal(size=len(tau))
    lw, _, info = calc.impose(lw0, inside, f)
    w = np.exp(lw)
    np.testing.assert_allclose([w.sum(), w[inside].sum()], [1, .45], rtol=0, atol=2e-15)
    shape = Targets(['tau', 'tau2'], c, calc.sigma, [.05, 1/3])
    ls, _, _ = shape.impose(lw0, inside, f)
    ws = np.exp(ls)
    for mask in (inside, ~inside):
        np.testing.assert_allclose(w[mask]/w[mask].sum(), ws[mask]/ws[mask].sum(), rtol=2e-14)
    lv, _, _ = calc.impose(lw0, inside, f, which='shift')
    np.testing.assert_allclose(np.exp(lv)[inside].sum(), .47, rtol=0, atol=2e-15)
    O = np.column_stack([tau, inside, tau>.2, rng.normal(size=len(tau))])
    direction = rng.normal(size=len(tau))
    R = _Response(w, (inside, f, info))
    influence, _ = R.remove(O-w@O, prefactor_response=True)
    analytic = (w[:, None]*influence).T@direction
    eps = 2e-5
    values = []
    for sign in (-1, 1):
        z, _, _ = calc.impose(lw0+sign*eps*direction, inside, f)
        values.append(np.exp(z)@O)
    numerical = (values[1]-values[0])/(2*eps)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-10)
    assert abs(analytic[1]) < 1e-14
    with tempfile.TemporaryDirectory() as folder:
        for suffix in ('npz', 'json'):
            path = Path(folder)/('targets.'+suffix)
            calc.save(path)
            loaded = load_targets(path)
            assert loaded.rate == calc.rate and loaded.rate_variations == calc.rate_variations
            assert loaded.meta == calc.meta
            zz, _, _ = loaded.impose(lw0, inside, f)
            np.testing.assert_allclose(zz, lw, rtol=0, atol=1e-13)
    print('PASS: rate, conditional shapes, paired variations, finite-difference response and target roundtrip')


if __name__ == '__main__':
    main()
