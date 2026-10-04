#!/usr/bin/env python3
"""Combine many independent EERAD3 NNLO production logs into a thrust coefficient.

EERAD3 with nloop=-2 prints, per rank, several histogram blocks:
  '(1-T)/sig dsig/d(1-T)'  -- the tau-WEIGHTED distribution; its integral is the NNLO
                              coefficient of <1-T> (a good, well-behaved normalisation check).
  '1/sig dsig/d(1-T)'      -- the UNWEIGHTED distribution A_3(tau) in
                              (1/sig_0) dsig/dtau = sum_n (alpha_s/2pi)^n A_n(tau).
                              At NNLO this has large per-bin +/- cancellations at small tau.

Two things make the naive combine fail, both handled here:
  1. Per-bin ERRORS are 'NaN' unless itmax2>=2, and even then the small-tau NNLO bins are
     wild. So ranks are combined by CROSS-RANK statistics (per-bin mean and std/sqrt(N) over
     independent seeds), NOT by within-rank inverse-variance. This is the correct estimator
     for embarrassingly-parallel VEGAS and needs no finite within-rank error.
  2. The distribution is heavy-tailed at low stats, so a symmetric trimmed mean is reported
     alongside the plain mean; bins where they disagree beyond the error are flagged as
     outlier-dominated rather than silently trusted.

usage: eerad_nnlo_combine.py TAG 'glob' [--fetch DIR] [--save] [--trim 0.05]
"""
import numpy as np, glob, sys, re, subprocess, os

# literature (alpha_s/2pi)^n coefficients of <1-T>, for the normalisation check
# Published first-moment coefficients, sigma_0 normalization, (alpha_s/2pi)^n:
#   LO/NLO: exact / GGGH 0903.4658 Table 3.  NNLO: the two published determinations
#   DISAGREE at 25%: GGGH (EERAD3, no close-to-two-jet systematic) 867.60 +- 21.19,
#   Weinzierl 0909.5056 Table 1 (dedicated two-jet slicing) 1100 +- 30.  Our own runs
#   use the same program as GGGH and should land near 867.6.
LIT = {'LO': 2.1035, 'NLO': 44.999, 'NNLO': 867.6, 'NNLO_weinzierl': 1100.0}
TOTAL_RE = re.compile(r'\bonly \[(LO|NLO|NNLO)\]\s+([-0-9.Ee+]+)\s+\+-\s+([-0-9.Ee+]+)')
NUM = r'[-+]?(?:\d+\.?\d*[Ee][-+]?\d+|\d+\.\d*|\d+|NaN|nan|Infinity|-?Inf)'
ROW_RE = re.compile(rf'^\s+({NUM})\s+({NUM})\s+({NUM})\s*$')


def f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except ValueError:
        return np.nan


def blocks(path):
    out, cur, hdr = [], None, None
    for ln in open(path, errors='replace'):
        if 'distribution' in ln:
            if cur is not None and cur:
                out.append((hdr, np.array(cur)))
            hdr, cur = ln.strip(), []
        elif cur is not None:
            m = ROW_RE.match(ln)
            if m:
                cur.append([f(m.group(1)), f(m.group(2)), f(m.group(3))])
            elif ln.startswith(' sum'):
                if cur:
                    out.append((hdr, np.array(cur)))
                cur, hdr = None, None
    if cur:
        out.append((hdr, np.array(cur)))
    return out


def get_block(path, kind):
    b = [(h, B) for h, B in blocks(path) if kind in h and len(B) > 1]
    return max(b, key=lambda x: len(x[1]))[1] if b else None


def cross_rank(arrays, trim=0.0):
    """arrays: list of (nbin,) value vectors on a common grid. Returns mean, err(=std/sqrt N),
    and a trimmed mean (symmetric trim fraction per bin)."""
    V = np.array(arrays)                      # (nrank, nbin)
    finite = np.isfinite(V)
    n = finite.sum(0)
    mean = np.where(n > 0, np.nansum(np.where(finite, V, 0.0), 0)/np.maximum(n, 1), np.nan)
    var = np.where(n > 1, np.nansum(np.where(finite, (V-mean)**2, 0.0), 0)/np.maximum(n-1, 1), np.nan)
    err = np.sqrt(var/np.maximum(n, 1))
    if trim > 0:
        tm = np.full(V.shape[1], np.nan)
        for j in range(V.shape[1]):
            col = np.sort(V[finite[:, j], j])
            if len(col) >= 5:
                k = int(len(col)*trim)
                tm[j] = col[k:len(col)-k].mean() if len(col)-2*k > 0 else col.mean()
            elif len(col):
                tm[j] = col.mean()
        return mean, err, tm, n
    return mean, err, mean.copy(), n


def fetch(subdir, tag):
    d = f'logs/eerad_{tag}'; os.makedirs(d, exist_ok=True)
    cmd = ("ssh -i ~/.ssh/nersc -o CertificateFile=~/.ssh/nersc-cert.pub -o IdentitiesOnly=yes "
           "-o StrictHostKeyChecking=no -o LogLevel=ERROR ${NERSC_USER}@perlmutter.nersc.gov "
           f"'cd $SCRATCH/eerad3_thrust/{subdir} && tar cz prod_*.log' 2>/dev/null | tar xz -C {d}")
    subprocess.run(cmd, shell=True, check=True)
    return f'{d}/prod_*.log'


def main():
    tag, pat = sys.argv[1], sys.argv[2]
    if '--fetch' in sys.argv:
        pat = fetch(sys.argv[sys.argv.index('--fetch')+1], tag)
    trim = float(sys.argv[sys.argv.index('--trim')+1]) if '--trim' in sys.argv else 0.05
    paths = sorted(glob.glob(pat))
    order = {'LO': 'LO', 'NLO': 'NLO'}.get(tag.split('_')[0], 'NNLO')

    uw, wt, totals, bad = [], [], [], 0
    grid = None
    for p in paths:
        U = get_block(p, '1/sig'); W = get_block(p, '(1-T)/sig')
        m = TOTAL_RE.search(open(p, errors='replace').read())
        if U is None:
            bad += 1; continue
        if grid is None:
            grid = U[:, 0]
        n0 = min(len(U), len(grid))
        uw.append(U[:n0, 1])
        if W is not None:
            wt.append(W[:len(grid), 1])
        if m:
            totals.append((f(m.group(2)), f(m.group(3))))
    if not uw:
        print(f'{tag}: no usable logs among {len(paths)}'); return
    L = min(len(v) for v in uw); grid = grid[:L]
    uw = [v[:L] for v in uw]
    mean, err, tmean, n = cross_rank(uw, trim=trim)
    w = grid[1]-grid[0]

    print(f'{tag}: {len(uw)} usable ranks ({bad} unusable of {len(paths)}), {L} bins, '
          f'width {w:.5f}, range [{grid[0]:.5f}, {grid[-1]:.5f}]')
    # check A: unweighted distribution integrates to the same total as the weighted block
    intA = float(np.nansum(mean*w))
    if wt:
        wm, we, _, _ = cross_rank([v[:L] for v in wt])
        wtau = grid; ww = wtau[1]-wtau[0]
        m1 = float(np.nansum(wm*ww))
        print(f'  check A  weighted-block integral (=<1-T> coeff): {m1:10.3f}  '
              f'literature {order} {LIT[order]:.1f}')
    if totals:
        T = np.array(totals); tot = T[:, 0].mean(); tote = T[:, 0].std()/np.sqrt(len(T))
        print(f'  check B  mean of printed [{order}] totals        : {tot:10.3f} +- {tote:.3f}')
    print(f'  unweighted integral sum_bins A dtau              : {intA:10.3f}  '
          f'(small-tau cancellations make this noisier than the weighted total)')
    # outlier diagnosis: bins where plain vs trimmed mean disagree beyond the error
    if trim > 0:
        bad_bins = np.isfinite(err) & (np.abs(mean-tmean) > np.maximum(err, 1e-30))
        print(f'  trim {trim:.0%}: {int(bad_bins.sum())}/{L} bins outlier-dominated '
              f'(plain vs trimmed mean differ > 1 sigma)')
    med_rel = np.nanmedian(np.abs(err/mean)[(grid > 0.05) & (grid < 0.30)])
    print(f'  per-bin rel error (cross-rank), median tau in [0.05,0.30]: {med_rel:.3f}')

    cum = np.concatenate([np.cumsum((mean*w)[::-1])[::-1][1:], [0.0]])
    ecum = np.sqrt(np.concatenate([np.cumsum(((err*w)**2)[::-1])[::-1][1:], [0.0]]))
    if '--save' in sys.argv:
        np.save(f'output/eerad_{tag}.npy', np.vstack([grid, mean, err, -cum, ecum]))
        print(f'  saved output/eerad_{tag}.npy')
    for t in (0.0375, 0.1025, 0.2025, 0.3025):
        i = np.argmin(abs(grid-t))
        print(f'   tau={grid[i]:.4f}  A={mean[i]:11.3f} +- {err[i]:9.3f}   Sigma-coeff={-cum[i]:11.4f}')


if __name__ == '__main__':
    main()
