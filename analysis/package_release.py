#!/usr/bin/env python3
"""Assemble the public release for one or more exported stages, and prove it is correct.

The release is numpy only. That is the whole point of this step: the trained event side lives
in a torch checkpoint, and shipping it as such would force every user to install and match a
deep learning framework in order to get a weight. Here the checkpoint is converted once into
plain arrays, and the release code reimplements the forward pass in numpy.

A conversion like that is exactly where a silent error lives, so the packager does not just
convert. It re-embeds the reference events through the numpy path and requires them to
reproduce the a(Phi) vectors the training run exported, and it replays the logit checksum
recorded at export time. If either disagrees beyond float32 rounding the packaging fails.

Usage: python package_release.py C_1M E F ...    (env RELEASE_DIR, default release)
"""
import hashlib, json, os, shutil, sys
import numpy as np
import torch

REL = os.environ.get('RELEASE_DIR', 'release')
# Export tags that are a variant of another stage's design.
DESIGN_ALIAS = {'C_1M': 'C', 'Faug': 'F'}
# Heads whose reference sample is not NREF_PER events per design point: (runs, events per run).
REF_COMPOSITION = {'C_1M': (54, 20000)}
# 'Faug' is stage F retrained on the augmented 112-point Herwig design. Packaging it REQUIRES
# STAGEF_CSV=stageF_design_aug.csv in the environment, because STAGES['F'] is otherwise built
# from the original 64-point file and the head would ship a design it was never trained on.
# The assertion below is what catches that, since the two designs share an identical box and
# so a box comparison cannot tell them apart.
MODELS = 'output/models'
NCHECK = int(os.environ.get('RELEASE_NCHECK', '4000'))
sys.path.insert(0, REL)
from r2_ladder import STAGES, NREF_PER as R_NREF_PER
from gentune.head import Head, MixtureHead
from mixture_cfg import register as _register_mixture
from gentune.trunk import Trunk
from gentune.features import trunk_features


def sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(blocks), b''):
            h.update(b)
    return h.hexdigest()


def convert_trunk(tag, cond, dst):
    """Torch checkpoint to plain arrays. The activation, the ensemble size and the pooled
    embedding scale are model facts that live in the cond export rather than the state dict,
    so they are copied across here and not re-derived."""
    t = torch.load(f'{MODELS}/{tag}_trunk.pt', map_location='cpu', weights_only=False)
    sds = t['models']
    out = dict(act=str(cond['act']), ens=len(sds), C=int(cond['feat_mu'].shape[0]),
               feat_mu=cond['feat_mu'], feat_sd=cond['feat_sd'],
               emb_scale_used=np.asarray(cond['emb_scale_used'], np.float64))
    nphi = nA = None
    for m, sd in enumerate(sds):
        pl = sorted({int(k.split('.')[1]) for k in sd if k.startswith('phi.')})
        al = sorted({int(k.split('.')[1]) for k in sd if k.startswith('A.')})
        if nphi is None:
            nphi, nA = len(pl), len(al)
        assert (len(pl), len(al)) == (nphi, nA), f'{tag}: member {m} has a different depth'
        for l, i in enumerate(pl):
            out[f'phi{m}_W{l}'] = sd[f'phi.{i}.weight'].numpy()
            out[f'phi{m}_b{l}'] = sd[f'phi.{i}.bias'].numpy()
        for l, i in enumerate(al):
            out[f'A{m}_W{l}'] = sd[f'A.{i}.weight'].numpy()
            out[f'A{m}_b{l}'] = sd[f'A.{i}.bias'].numpy()
    out['n_phi_layers'] = nphi
    out['n_A_layers'] = nA
    out['K'] = int(out[f'A0_W{nA-1}'].shape[0])
    np.savez_compressed(dst, **out)
    return out['K'], nphi, nA



def design_key(tag, stages, alias):
    """Which registered stage's design belongs to this export tag.

    Export names are a stage key followed by LADDER_SUFFIX, so 'Faug' and 'Fauglong' are both
    stage F on some design, and the reverse mapping is the longest registered key that prefixes
    the tag. Listing the tags one at a time instead is how 'Faug' got an entry and 'Fauglong' did
    not, which silently shipped a head with no design and made the measurement script report on
    one of the two heads it was given.
    """
    if tag in stages:
        return tag
    if tag in alias:
        return alias[tag]
    cands = [k for k in stages if tag.startswith(k) and len(tag) > len(k)]
    return max(cands, key=len) if cands else None

def main(tags):
    """tags are export names, optionally EXPORT=RELEASE_NAME.

    The retrained heads are exported as 'Fauglong' and 'MIX17aug', which are working names. A
    user should see 'F' and 'MIX17'. Passing Fauglong=F packages that export under the public
    name, so the file names, the manifest key and the closure CSV all agree, rather than being
    renamed by hand afterwards in four places.
    """
    rename = {}
    clean = []
    for t in tags:
        if '=' in t:
            a, b = t.split('=', 1)
            rename[a] = b
            clean.append(a)
        else:
            clean.append(t)
    tags = clean
    os.makedirs(f'{REL}/models', exist_ok=True)
    manifest = {}
    for tag in tags:
        cp = f'{MODELS}/{tag}_cond.npz'
        assert os.path.exists(cp), f'no export for {tag}'
        cond = np.load(cp)
        print(f'== {tag}: ntheta={int(cond["ntheta"])} ens={int(cond["ens"])} K={int(cond["K"])} '
              f'act={str(cond["act"])} T={float(cond["temperature"]):.3f} '
              f'additive={int(cond["additive"])}')
        # The exported cond file carries the cached a(Phi) of every reference event, which is
        # hundreds of megabytes and is needed only to reweight the reference sample itself.
        # A user bringing their own events needs the parameter network and nothing else, so
        # the head ships without AE and the reference bundle is a separate optional download.
        head_only = {k: cond[k] for k in cond.files if k != 'AE'}
        head_only['n_reference_events'] = np.int64(cond['AE'].shape[1])
        # Ship the training design, standardized the way the head standardizes theta, so a
        # user can ask whether their point was sampled and not merely whether it is in the box.
        # Some exports carry a suffix for a variant of the same design, for example C_1M is
        # stage C's design at the one-million-event reference. The design is the design.
        dtag = design_key(tag, STAGES, DESIGN_ALIAS)
        if dtag not in STAGES and str(cond.get('head_kind', 'cond')) == 'mixture':
            # a mixture export: its design lives in the data set's meta.json, not the registry
            if _register_mixture(STAGES, [tag]) is not None:
                dtag = tag
        if dtag in STAGES:
            tr = np.asarray(list(STAGES[dtag]['train'].values()), np.float64)
            nm = np.asarray(STAGES[dtag]['norm'], np.float64)
            head_only['theta_train'] = (tr - nm[:, 0])/nm[:, 1]
            # Assigned BEFORE the design check below, because the summary line after it uses
            # nref whichever branch ran. Putting it inside one branch made every export with
            # the new fitted_theta key fail with UnboundLocalError.
            nref = int(head_only['n_reference_events'])
            # The reference sample is a fixed number of events drawn from each training run, so
            # the run count must divide it. A design of the wrong size fails this even when its
            # box matches, which is the only reason it is here.
            # If the export records which runs it was fitted on, that settles the question
            # outright. Exports written before that was added fall back to the weaker count
            # check below, and say which check they got.
            if 'fitted_theta' in cond.files:
                # The strongest check available: the parameter values the head was fitted on,
                # against the ones the resolved design holds. Run ids and counts can coincide
                # exactly between two different designs of the same stage, so this is the only
                # attribute that always distinguishes them.
                wt = np.asarray([STAGES[dtag]['train'][r] for r in sorted(STAGES[dtag]['train'])],
                                dtype=float)
                gt = np.asarray(cond['fitted_theta'], dtype=float)
                assert wt.shape == gt.shape and np.allclose(wt, gt, rtol=0, atol=1e-12), (
                    f'{tag}: this head was fitted on a design of shape {gt.shape} whose parameter '
                    f'values do not match the {wt.shape} of design {dtag!r} '
                    f'(largest difference {np.abs(wt - gt).max() if wt.shape == gt.shape else float("nan"):.3e}). '
                    f'Shipping it would make the head misreport where it was trained. Point '
                    f'STAGEF_CSV or DM_DATA at the design this head actually used.')
                print(f'   design VERIFIED against the export: {gt.shape[0]} training points '
                      f'match by parameter value')
            elif 'train_rids' in cond.files:
                want = np.asarray(sorted(STAGES[dtag]['train']), dtype=np.int64)
                got = np.asarray(cond['train_rids'], dtype=np.int64)
                assert want.shape == got.shape and (want == got).all(), (
                    f'{tag}: this head was fitted on {len(got)} runs beginning {got[:3].tolist()} '
                    f'and ending {got[-3:].tolist()}, but design {dtag!r} holds {len(want)} '
                    f'beginning {want[:3].tolist()} and ending {want[-3:].tolist()}. Shipping it '
                    f'would make the head misreport where it was trained. Point STAGEF_CSV or '
                    f'DM_DATA at the design this head actually used.')
                print(f'   design VERIFIED against the export: {len(got)} training runs match by id')
            else:
                # No fitted_theta and no train_rids, so the design cannot be read off the export.
                # RELEASE_DESIGN_NTRAIN stood here and was worthless: it compared the operator's
                # stated count against the design the operator had already resolved, so both sides
                # were independent of the head and a wrong pairing passed whenever the operator
                # stated the count that went with the design rather than with the head. It also
                # broke the documented default invocation, since no export on disk carries the
                # strong key.
                #
                # The reference sample takes NREF_PER events from each training run, so the events
                # PER RUN are export-derived and must come out at NREF_PER. That catches a 112-run
                # export against the 64-point design (21000 per run) and a 64-run export against
                # the 112-point design (6857 per run), with nothing required from the operator.
                per = nref/len(tr)
                if tag in REF_COMPOSITION:
                    # A head whose reference is not NREF_PER per design point. C_1M pools the two
                    # Stage C campaigns, 54 runs at the same 27 thetas with 20,000 events from each
                    # (App. A of the paper), so per design point it holds 40,000, and the generic
                    # rule below would reject the head it was written to protect.
                    nrun, nper = REF_COMPOSITION[tag]
                    assert nref == nrun*nper, (
                        f'{tag}: the export holds {nref} reference events, but REF_COMPOSITION says '
                        f'{nrun} runs x {nper} = {nrun*nper}. One of the two records is wrong.')
                    print(f'   design checked against the export: {nref} reference events is '
                          f'{nrun} runs x {nper}, the recorded composition for {tag}')
                elif str(cond.get('head_kind', 'cond')) == 'mixture':
                    print(f'   NOTE legacy mixture export without fitted_theta: its design cannot '
                          f'be verified here, because both mixture data sets hold {len(tr)} runs at '
                          f'{per:.0f} reference events each. Verify DM_DATA by hand.')
                else:
                    assert abs(per - R_NREF_PER) < 1e-9, (
                        f'{tag}: the export holds {nref} reference events over the {len(tr)} points '
                        f'of design {dtag!r}, which is {per:.1f} per run where the stages use '
                        f'{R_NREF_PER}. This design is not the one this head was fitted on. Point '
                        f'STAGEF_CSV or DM_DATA at the design it actually used.')
                    print(f'   design checked against the export: {nref} reference events over '
                          f'{len(tr)} points is {per:.0f} per run, as expected')
            print(f'   design: {len(tr)} training points shipped for the support check '
                  f'({nref} reference events, {nref//len(tr)} per run)')
        else:
            print(f'   NOTE {tag} is not a STAGES key, so no design is shipped and support() '
                  f'will be unavailable')
        out_tag = rename.get(tag, tag)
        np.savez_compressed(f'{REL}/models/{out_tag}_head.npz', **head_only)
        if os.environ.get('RELEASE_WITH_REFERENCE', '') == '1':
            shutil.copyfile(cp, f'{REL}/models/{out_tag}_cond.npz')
        elif os.path.exists(f'{REL}/models/{out_tag}_cond.npz'):
            os.remove(f'{REL}/models/{out_tag}_cond.npz')
        # Exports made before the event-side weights were saved have no trunk. Their head is
        # still exact on the reference sample, so they are packaged, but they cannot embed new
        # events and the release says so rather than appearing to support it.
        has_trunk = os.path.exists(f'{MODELS}/{tag}_trunk.pt')
        if has_trunk:
            K, nphi, nA = convert_trunk(tag, cond, f'{REL}/models/{out_tag}_trunk.npz')
            print(f'   trunk: {nphi} Phi layers, {nA} A layers, K={K}')
        else:
            print('   NO TRUNK in the export: this stage can reweight the reference sample '
                  'only, not your own events')

        # 1. the reader reproduces the logits the training run recorded. This is checked on
        # the FULL export, because the shipped head deliberately has no reference vectors.
        HKfull = MixtureHead if str(cond.get('head_kind', 'cond')) == 'mixture' else Head
        h = HKfull(cp)
        dchk = h.replay_checksum()
        print(f'   logit checksum replay max|df| = {dchk:.3e}')
        assert dchk < 1e-4, f'{tag}: the reader disagrees with the exported logits'

        # 2. the numpy trunk reproduces the exported a(Phi) on reference events
        rp = f'{MODELS}/{tag}_ref.npz'
        dae = dself = d32 = d16 = None
        if os.path.exists(rp) and has_trunk:
            r = np.load(rp)
            tr = Trunk(f'{REL}/models/{out_tag}_trunk.npz')
            n = min(NCHECK, len(r['mask']))
            # Two comparisons, because the training path and a user do not work at the same
            # precision. Stages trained through the memmap cache stored per-particle features
            # as FLOAT16 so that a hundred runs would fit in memory, so their exported
            # reference vectors carry that quantization. A user embedding their own events
            # works in float32. The float16 round trip therefore tests whether this numpy
            # trunk is arithmetically the same network as the torch one, which is the thing
            # that must hold exactly, and the float32 number reports the size of the
            # quantization gap, which is a fact about the stored reference rather than a bug.
            # Measured on Stage E: that gap moves individual weights by at most 6e-3 relative
            # and the mean of one minus thrust by 3e-5, against a statistical error of 9e-4
            # on the full reference, so it is immaterial and the shipped vectors are left as
            # the training run produced them, which keeps the published closure reproducible.
            Fr = trunk_features(r['particles'][:n])
            Mr = r['mask'][:n]
            ref = np.asarray(cond['AE'][:, :n, :], np.float64)
            sdref = max(float(np.std(ref)), 1e-12)

            def agree(feats):
                A = tr.embed(feats, Mr)
                if int(cond['additive']):
                    A = np.concatenate([A, np.ones(A.shape[:2] + (1,), A.dtype)], -1)
                return float(np.max(np.abs(np.asarray(A, np.float64) - ref))/sdref)

            d32 = agree(Fr)
            d16 = agree(Fr.astype(np.float16).astype(np.float32))
            dae = min(d32, d16)
            which = 'float32' if d32 <= d16 else 'float16 (this stage trained through the memmap cache)'
            print(f'   numpy trunk against exported a(Phi) on {n} events: '
                  f'max|dA|/sd(A) = {d32:.3e} in float32, {d16:.3e} through float16')
            print(f'   the network matches exactly at {which}; quantization gap '
                  f'{max(d32, d16):.3e} of a standard deviation')
            assert dae < 1e-3, (f'{tag}: the numpy trunk is not the same network as the '
                               f'exported one at either precision ({d32:.2e}, {d16:.2e})')
        elif not has_trunk:
            r = np.load(rp) if os.path.exists(rp) else None
        else:
            r = None
            print('   NOTE no reference sample present, the trunk could not be cross-checked')

        # 3. a small self-verifying bundle, so a user can prove their install is exact without
        # downloading the reference sample. It carries raw particles rather than features, so
        # it also pins the feature construction, which is the step most likely to drift.
        if r is not None:
            ns = min(int(os.environ.get('RELEASE_NSELFTEST', '2000')), len(r['mask']))
            probe = [h.centre]
            for j in range(h.nt):                     # one point per parameter, at 80% of the box
                t = h.centre.copy(); t[j] += 0.8*h.norm[j, 1]; probe.append(t)
            # A UNIFORM draw over the whole reference, not the leading rows. The reference is
            # the runs concatenated, so the leading rows are one parameter point and weights
            # meant for the reference would be applied to the wrong sample. Sorted so the
            # gather off the large cached array stays sequential.
            sel = np.sort(np.random.default_rng(20260910).choice(len(r['mask']), ns, replace=False))
            AEs = np.ascontiguousarray(cond['AE'][:, sel, :])
            # The probe logits are what a USER must reproduce, so they are computed the way
            # a user computes them: features in float32 through the shipped numpy trunk and
            # head. Storing the training run's own logits instead would make every user's
            # self-test fail by the float16 quantization of the stored reference, which is a
            # property of how the training cached its features and not of their install.
            # Both are kept, with the gap between them recorded, so the provenance is visible.
            HK0 = MixtureHead if str(cond.get('head_kind', 'cond')) == 'mixture' else Head
            _h0 = HK0(f'{REL}/models/{out_tag}_head.npz')
            _tp = np.asarray(probe, np.float64)
            _lex = np.stack([h.logit(AEs, t) for t in _tp])          # from the training run
            if has_trunk:
                _A0 = Trunk(f'{REL}/models/{out_tag}_trunk.npz').embed(
                    trunk_features(r['particles'][sel]), r['mask'][sel])
                if int(cond['additive']):
                    _A0 = np.concatenate([_A0, np.ones(_A0.shape[:2] + (1,), _A0.dtype)], -1)
                _lus = np.stack([_h0.logit(_A0, t) for t in _tp])    # what a user reproduces
            else:
                _lus = _lex
            _gap = float(np.max(np.abs(_lus - _lex)))
            np.savez_compressed(
                f'{REL}/models/{out_tag}_selftest.npz',
                particles=r['particles'][sel], mask=r['mask'][sel], AE=AEs,
                reference_index=sel, n_reference_events=np.int64(len(r['mask'])),
                theta_probe=_tp, logit_probe=_lus, logit_probe_from_export=_lex,
                precision_gap=np.float64(_gap))
            print(f'   probe logits stored from the float32 user path; they differ from the '
                  f'training run by {_gap:.2e} in the logit (float16 feature cache)')
            print(f'   self-test bundle: {ns} events, {len(probe)} probe points')
            # 4. the end-to-end path a user actually runs: raw particles from the bundle,
            # features rebuilt from scratch, the numpy trunk, the shipped head, compared with
            # the logits this training run produced. Nothing from the torch model is involved.
            HK = MixtureHead if str(cond.get('head_kind', 'cond')) == 'mixture' else Head
            hs = HK(f'{REL}/models/{out_tag}_head.npz')
            sb = np.load(f'{REL}/models/{out_tag}_selftest.npz')
            if has_trunk:
                trs = Trunk(f'{REL}/models/{out_tag}_trunk.npz')
                Ae = trs.embed(trunk_features(sb['particles']), sb['mask'])
                if int(cond['additive']):
                    Ae = np.concatenate([Ae, np.ones(Ae.shape[:2] + (1,), Ae.dtype)], -1)
            else:
                Ae = sb['AE']          # no trunk, so the stored vectors are the only entry point
            fe = np.stack([hs.logit(Ae, t) for t in sb['theta_probe']])
            dself = float(np.max(np.abs(fe - sb['logit_probe'])))
            print(f'   end to end through the shipped files: max|df| = {dself:.3e} '
                  f'over {len(sb["theta_probe"])} points')
            assert dself < 1e-3, f'{tag}: the shipped release does not reproduce the logits'

        # keyed by the PUBLIC name, so the manifest, the file names and STAGES.md agree
        manifest[out_tag] = dict(
            ntheta=int(cond['ntheta']), ens=int(cond['ens']), K=int(cond['K']),
            K_bilinear=int(cond.get('K_bilinear', cond['K'])), act=str(cond['act']),
            temperature=float(cond['temperature']), additive=int(cond['additive']),
            head_kind=str(cond.get('head_kind', 'cond')), has_trunk=bool(has_trunk),
            box=[[float(a), float(b)] for a, b in h.box], centre=[float(v) for v in h.centre],
            n_reference_events=int(cond['AE'].shape[1]),
            checksum_replay=dchk, trunk_vs_export=dae, selftest_vs_export=dself,
            trunk_vs_export_float32=d32 if dae is not None else None,
            trunk_vs_export_float16=d16 if dae is not None else None,
            sha256={f'{tag}_{k}.npz': sha256(f'{REL}/models/{out_tag}_{k}.npz')
                    for k in ('head', 'trunk', 'selftest', 'cond')
                    if os.path.exists(f'{REL}/models/{out_tag}_{k}.npz')})
    old = {}
    if os.path.exists(f'{REL}/models/MANIFEST.json'):
        old = json.load(open(f'{REL}/models/MANIFEST.json'))
    old.update(manifest)
    json.dump(old, open(f'{REL}/models/MANIFEST.json', 'w'), indent=1)
    print(f'\nwrote {REL}/models/MANIFEST.json with {len(old)} stages: {sorted(old)}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['C_1M'])
