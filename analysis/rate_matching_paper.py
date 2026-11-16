#!/usr/bin/env python3
"""Recompute the paper's rate-and-shape predictions from the released reference.

No training or event generation. The same 2000 parameter points are reused.
All outputs go to a new directory; shape-only outputs remain available.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'release'), str(HERE/'release/examples')]
from gentune import axes, bands, load_head, mixture
from gentune.maxent import load_targets, _Response
from reproduce_paper import Reference, OBS, factor_vectors

STATE = {}


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False)+'\n')


def predict(w):
    ref = STATE['ref']
    first = ref.predictions(w)
    nch = np.bincount(STATE['nch_bins'], weights=w, minlength=24)
    rho = np.bincount(STATE['rho_bins'][STATE['rho_ok']], weights=w[STATE['rho_ok']], minlength=18)
    rho /= np.diff(STATE['rho_edges'])
    return np.r_[first, nch, rho]


def evaluate(theta, which='central'):
    head, ref, calc = (STATE[k] for k in ('head', 'ref', 'calc'))
    w0 = mixture.weights(head, ref.A, theta)
    w, info = mixture.impose(w0, ref.inwin, ref.F, calc, which=which)
    if not all(i['converged'] for i in info.values()):
        raise RuntimeError('shape solve did not converge')
    return {g: predict(w[g]) for g in w}, w0, w, info


def worker(index):
    rows, _, _, info = evaluate(STATE['nodes'][index])
    return index, rows['sherpa'], rows['herwig'], max(i['residual'] for i in info.values())


def columns(j0, j1):
    ref = STATE['ref']
    X = np.zeros((ref.N, j1-j0))
    for k in range(j0, j1):
        if k == 0:
            X[:, k-j0] = ref.tau
        elif k == 1:
            X[:, k-j0] = ref.inwin
        elif k < 7:
            X[:, k-j0] = ref.X[:, k-2]
        elif k < 27:
            X[:, k-j0] = (ref.b == k-7)/ref.width[k-7]
        else:
            X[:, k-j0] = STATE['nch_bins'] == k-27
    return X


def central_response(theta, w, info, wifi):
    """Full learning covariance and shared-reference statistical covariance.

    Rate matching is differentiated through the conditional means. The
    derivative of P0^2 in the shape penalty is retained in both responses.
    """
    ref, head = STATE['ref'], STATE['head']
    gs = ('sherpa', 'herwig')
    R = {g: _Response(w[g], (ref.inwin, ref.F, info[g])) for g in gs}
    basis = {}
    for g, xi in (('sherpa', 0.), ('herwig', 1.)):
        t = theta.copy(); t[-1] = xi
        basis[g] = head.wifi_basis(ref.A, t, check_box=False)
    Y = {g: predict(w[g])[:51] for g in gs}
    gradients = {g: np.zeros((51, 17)) for g in gs}
    influence = np.zeros((ref.N, 51))
    own_stat = {g: np.zeros(51) for g in gs}
    for j0 in range(0, 51, 4):
        j1 = min(j0+4, 51); X = columns(j0, j1)
        combined = 0.
        for g in gs:
            residual = X-Y[g][None, j0:j1]
            if j0 == 0:
                residual[:, 0] = np.where(ref.inwin, ref.tau-Y[g][0], 0)/w[g][ref.inwin].sum()
            r, _ = R[g].remove(residual, prefactor_response=True)
            u = w[g][:, None]*r
            if j0 <= 1 < j1:
                u[:, 1-j0] = 0.             # the full-window fraction is an exact input
            gradients[g][j0:j1] = u.T@basis[g]
            own_stat[g][j0:j1] = np.sqrt(np.sum(u*u, axis=0))
            combined = combined+.5*u
        influence[:, j0:j1] = combined
    C = wifi['C']
    covariance = sum((1/3 if a==b else 1/6)*(gradients[a]@C@gradients[b].T) for a in gs for b in gs)
    stat_cov = influence.T@influence
    return gradients, covariance, stat_cov, own_stat


def corrected_covariance(gradients, C, factors):
    G = {g: gradients[g]*factors[g][:, None] for g in gradients}
    return sum((1/3 if a==b else 1/6)*(G[a]@C@G[b].T) for a in G for b in G)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'output/rate_shape')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='development: compute at most this many unfinished points')
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    head = load_head(HERE/'release/models/MIX17_head.npz')
    calc = load_targets(HERE/'release/targets_thrust_rate_shape.npz')
    assert calc.rate is not None
    ref = Reference(HERE/'release/reference_MIX17.npz', head, calc)
    original = np.load(HERE/'output/envelope_test_MIXGEOV3S9_full_lin.npz')
    nodes = original['TH'].copy(); labels = original['labels'].copy()
    assert nodes.shape == (2000, 17)
    nch_edges = original['nch_edges']; rho_edges = original['rho_edges']
    nch_bins = np.searchsorted(nch_edges, ref.X[:, 3], side='right')-1
    rho_bins = np.searchsorted(rho_edges, ref.X[:, 2], side='right')-1
    STATE.update(head=head, calc=calc, ref=ref, nodes=nodes, nch_bins=nch_bins,
                 rho_bins=rho_bins, rho_ok=(rho_bins>=0)&(rho_bins<18), rho_edges=rho_edges)
    print(f'{ref.N:,} events loaded; target {calc.rate:.12f}; {time.monotonic()-start:.1f}s', flush=True)
    checkpoint = out/'generator_draws.npz'
    fingerprint = hashlib.sha256()
    for path in (HERE/'release/models/MIX17_head.npz', HERE/'release/targets_thrust_rate_shape.npz',
                 HERE/'output/envelope_test_MIXGEOV3S9_full_lin.npz'):
        fingerprint.update(path.read_bytes())
    input_digest = fingerprint.hexdigest()
    if checkpoint.exists():
        z = np.load(checkpoint)
        if 'input_digest' not in z.files or str(z['input_digest']) != input_digest:
            raise ValueError('Checkpoint inputs changed; use a new output directory or remove the checkpoint.')
        assert np.array_equal(z['nodes'], nodes) and float(z['target']) == calc.rate
        S, H, done = z['sherpa'].copy(), z['herwig'].copy(), z['done'].copy()
    else:
        S, H, done = np.zeros((2000, 69)), np.zeros((2000, 69)), np.zeros(2000, bool)
    todo = np.flatnonzero(~done)
    if args.limit is not None:
        todo = todo[:args.limit]

    def save():
        np.savez_compressed(checkpoint, sherpa=S, herwig=H, done=done, nodes=nodes, target=calc.rate,
                            input_digest=np.array(input_digest))

    if len(todo):
        context = mp.get_context('fork')
        with context.Pool(args.workers) as pool:
            for index, ys, yh, residual in pool.imap_unordered(worker, todo, chunksize=1):
                S[index], H[index], done[index] = ys, yh, True
                if done.sum()%25 == 0 or done.all():
                    save(); print(f'{done.sum()}/2000 points, {time.monotonic()-start:.1f}s, residual {residual:.2g}', flush=True)
        save()
    if not done.all():
        print('Checkpoint saved; complete all 2000 points before publishing.', flush=True)
        return
    theta = axes.default('MIX17')
    parts, w0, w, info = evaluate(theta)
    before = {g: predict(w0[g]) for g in w0}
    factors27 = factor_vectors()[0]
    factors = {g: np.r_[factors27[g], np.ones(42)] for g in factors27}
    wifi = np.load(HERE/'release/models/MIX17_wifi.npz')
    print('Computing central learning and statistical responses', flush=True)
    gradients, CL, CS, stat_own = central_response(theta, w, info, wifi)
    # For the converted covariance, statistical terms must also combine the
    # generators before squaring. Recompute just those terms with their factors.
    Gconv = {g: gradients[g]*factors[g][:51, None] for g in gradients}
    CLconv = corrected_covariance(gradients, wifi['C'], {g:factors[g][:51] for g in factors})
    stat_conv = np.zeros(51)
    responses = {g:_Response(w[g], (ref.inwin, ref.F, info[g])) for g in w}
    for j0 in range(0, 51, 4):
        j1=min(j0+4, 51); X=columns(j0,j1); total=0.
        for g in w:
            r=X-parts[g][None,j0:j1]
            if j0==0:
                r[:,0]=np.where(ref.inwin,ref.tau-parts[g][0],0)/calc.rate
            r=responses[g].remove(r,prefactor_response=True)[0]
            u=w[g][:,None]*r
            if j0<=1<j1:u[:,1-j0]=0.
            total=total+.5*factors[g][None,j0:j1]*u
        stat_conv[j0:j1]=np.sqrt(np.sum(total*total,axis=0))
    varied = {g:{} for g in w}; deltas={g:{} for g in w}
    for label in calc.variations:
        rows, _, _, _ = evaluate(theta, label)
        for g in rows:
            varied[g][label]=rows[g]
            deltas[g][label]=rows[g]-parts[g]
    raw_mix=.5*(parts['sherpa']+parts['herwig'])
    conv_mix=.5*sum(parts[g]*factors[g] for g in parts)
    before_mix=.5*sum(before.values())
    before_conv=.5*sum(before[g]*factors[g] for g in before)
    xi=nodes[:,-1,None]
    draws=(1-xi)*S+xi*H
    draws_conv=(1-xi)*S*factors['sherpa']+xi*H*factors['herwig']
    dconv={g:{k:v*factors[g] for k,v in deltas[g].items()} for g in deltas}
    up=bands.theory_band_mix(deltas['sherpa'],deltas['herwig'],True)
    dn=bands.theory_band_mix(deltas['sherpa'],deltas['herwig'],False)
    upc=bands.theory_band_mix(dconv['sherpa'],dconv['herwig'],True)
    dnc=bands.theory_band_mix(dconv['sherpa'],dconv['herwig'],False)
    arrays=dict(labels=labels, theta=theta, nodes=nodes, pure_draws_sherpa=S,pure_draws_herwig=H,
                before=before_mix, after=raw_mix, before_conv=before_conv,after_conv=conv_mix,
                draws=draws,draws_conv=draws_conv, generator=draws.std(0,ddof=1),
                generator_conv=draws_conv.std(0,ddof=1), theory_up=up,theory_down=dn,
                theory_up_conv=upc,theory_down_conv=dnc,C_learn=CL,C_stat=CS,
                C_learn_conv=CLconv,stat_conv=stat_conv,
                lo=ref.lo,hi=ref.hi,nch_edges=nch_edges,rho_edges=rho_edges,mix=np.array('linear'))
    for g in parts:
        arrays[g+'_before']=before[g];arrays[g+'_after']=parts[g]
        arrays[g+'_J_omega']=gradients[g];arrays[g+'_stat']=stat_own[g]
        arrays[g+'_theory_up']=bands.theory_band(deltas[g],True)
        arrays[g+'_theory_down']=bands.theory_band(deltas[g],False)
        for k,v in deltas[g].items():arrays[g+'_delta_'+k]=v
    np.savez_compressed(out/'rate_shape.npz',**arrays)
    central={g:dict(predictions=dict(zip(OBS,parts[g][:7])), shape_fraction=info[g]['pwin_shape'],
                    n_eff=info[g]['n_eff'],max_weight_over_mean=float(ref.N*w[g].max()),
                    residual=info[g]['residual']) for g in parts}
    mixed_weights=.5*(w['sherpa']+w['herwig'])
    central['mixture']=dict(predictions=dict(zip(OBS,raw_mix[:7])),
                           n_eff=float(1/np.sum(mixed_weights**2)),
                           max_weight_over_mean=float(ref.N*mixed_weights.max()))
    report=dict(prescription='Sequential conditional-moment fit followed by exact QCD window-rate matching.',
                calculation=dict(window=calc.window,pwin=calc.rate,alpha_s=calc.meta.get('alpha_s'),alpha_0=calc.meta.get('alpha_0')),
                n_events=ref.N,nodes=2000,central=central,
                uncertainty=dict(generator='Same 2000 paired parameter draws; correction applied per generator before mixing.',
                                 theory='Same varied conditional moments and window fraction at each variation; uniform-xi variance average.',
                                 learning='Full sequential response including the shape penalty prefactor, shared WiFi covariance.',
                                 statistical='Shared-event influence functions, including the shape fit and exact rate correction.'),
                seconds=time.monotonic()-start)
    dump(out/'rate_shape.json',report)
    print('Wrote complete rate_shape.npz/json',flush=True)


if __name__=='__main__':
    main()
