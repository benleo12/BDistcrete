#!/usr/bin/env python3
"""Isolated sequential rate+shape trial from the published caches.

Run rate_matching_targets.py first. This script writes only --out and never
replaces published results. It fixes the common-state full-window rate after
the existing soft shape fit; it is not a joint 15-constraint soft optimum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'release'))
from gentune.bands import theory_band, theory_band_mix
import figs_r8 as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


def json_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False) + '\n')


def partition(h, p, lo, hi, window):
    """Recover the mass on each side of W, including its straddling bin."""
    h, p = np.asarray(h), np.asarray(p)
    widths = hi-lo
    inside = (lo >= window[0]-1e-12) & (hi <= window[1]+1e-12)
    crosses = np.flatnonzero((lo < window[1]) & (hi > window[1]))
    assert len(crosses) == 1 and np.any(np.isclose(lo, window[0]))
    mass = h*widths
    iw = np.zeros_like(mass)
    iw[..., inside] = mass[..., inside]
    iw[..., crosses[0]] = p - mass[..., inside].sum(axis=-1)
    ow = mass-iw
    assert iw.min() > -2e-12 and ow.min() > -2e-12
    return iw, ow


def correct(h, p, target, lo, hi, window):
    p, target = np.asarray(p), np.asarray(target)
    assert np.all((p > 0) & (p < 1)) and np.all((target > 0) & (target < 1))
    iw, ow = partition(h, p, lo, hi, window)
    a, b = target/p, (1-target)/(1-p)
    new = (a[..., None]*iw+b[..., None]*ow)/(hi-lo)
    assert np.allclose(new @ (hi-lo), 1, rtol=0, atol=2e-10)
    return new


def corrected_jacobian(h, p, target, jac, jp, lo, hi, window):
    iw, ow = partition(h, p, lo, hi, window)
    inside = (lo >= window[0]-1e-12) & (hi <= window[1]+1e-12)
    cross = np.flatnonzero((lo < window[1]) & (hi > window[1]))[0]
    dm = jac*(hi-lo)[:, None]
    di = np.zeros_like(dm)
    di[inside] = dm[inside]
    di[cross] = jp-dm[inside].sum(axis=0)
    do = dm-di
    a, b = target/p, (1-target)/(1-p)
    return (a*di+b*do-iw[:, None]*a/p*jp+ow[:, None]*b/(1-p)*jp)/(hi-lo)[:, None]


def load_case(output, generator, window, target, V, lo, hi, C):
    gx, short = (0, 'S') if generator == 'sherpa' else (1, 'H')
    pure = V[f'xi{gx}']
    h = np.r_[pure['h_imposed_low'], pure['h_imposed']]
    hc = np.r_[pure['h_imposed_conv_low'], pure['h_imposed_conv']]
    factors = np.divide(hc, h, out=np.ones_like(h), where=h != 0)
    assert np.allclose(h*factors, hc, rtol=1e-12, atol=1e-15)
    p = pure['o_imposed']['pwin']
    new = correct(h, p, target, lo, hi, window)
    cases = {}
    paths = {}
    for box in ('full', 'tune'):
        path = output/f'envelope_test_MIXGEOV3S9_{box}_{generator}_lin.npz'
        paths[box] = str(path)
        with np.load(path) as z:
            labels = [str(l) for l in z['labels']]
            n = len(labels)
            columns = [n+labels.index(f'thr_{a:.3f}_{b:.3f}') for a, b in zip(lo, hi)]
            ip = n+labels.index('pwin')
            draws = z['Y_theta'][:, columns]
            ps = z['Y_theta'][:, ip]
            transformed = correct(draws, ps, target, lo, hi, window)
            cases[box] = dict(before=draws, after=transformed, pwin=ps, theta=z['TH'].copy())
            if box == 'tune':
                assert np.allclose(z['y_central'][columns], h, rtol=2e-8, atol=1e-12)
                J = z[f'J_omega_{short}'][columns]
                jp = z[f'J_omega_{short}'][ip]
                Jnew = corrected_jacobian(h, p, target, J, jp, lo, hi, window)
                assert np.allclose(z['C_learn'][np.ix_(columns, columns)], J@C@J.T,
                                   rtol=1e-9, atol=1e-15)
    variation_h, variation_p = {}, {}
    for key, varied in V['variations'].items():
        delta = np.r_[varied[f'dh_{generator}_low'], varied[f'dh_{generator}']]
        variation_h[key] = h+delta
        variation_p[key] = p+varied[f'do_{generator}']['pwin']
    return dict(h=h, h_converted=hc, new=new, new_converted=new*factors,
                factors=factors, p=p, a=target/p, b=(1-target)/(1-p),
                draws=cases, jac=J, jac_new=Jnew, p_jac=jp,
                variation_h=variation_h, variation_p=variation_p, source_files=paths)


def save_plot(fig, out, name):
    fig.savefig(out/(name+'.pdf'))
    fig.savefig(out/(name+'.png'), dpi=240)
    plt.close(fig)


def draw_plots(out, d, lo, hi, data, err, cb, summary):
    F.style()
    widths = hi-lo
    edges = np.r_[lo, hi[-1]]
    mix = d['mixture']
    # Full spectrum and fixed-data ratio: no normalization is refitted to data.
    fig, axes = plt.subplots(2, 1, figsize=(F.TW, 4.8), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    fig.subplots_adjust(left=.13, right=.97, bottom=.13, top=.88, hspace=.10)
    for ax in axes:
        F.trim(ax)
        ax.axvline(.05, color=F.THEORY, lw=.7, ls=(0, (3, 3)))
        ax.axvline(1/3, color=F.THEORY, lw=.7, ls=(0, (3, 3)))
    for key, label, color, style in (('h_converted', 'Shape only', F.BASE_DARK, '--'),
                                     ('new_converted', 'Shape + rate', F.METHOD, '-')):
        h = mix[key]
        axes[0].stairs(h, edges, baseline=None, color=color, ls=style, label=label, lw=1.25)
        axes[1].stairs(h/data, edges, baseline=None, color=color, ls=style, lw=1.25)
    axes[0].errorbar((lo+hi)/2, data, yerr=err, fmt='o', color=F.DATA, ms=2.5, lw=.6, label='DELPHI')
    axes[1].errorbar((lo+hi)/2, np.ones_like(data), yerr=err/data,
                     fmt='o', color=F.DATA, ms=2.5, lw=.6)
    axes[1].axhline(1, color=F.REF, lw=.65)
    axes[0].set_yscale('log'); axes[0].set_ylim(.002, 90)
    axes[0].set_ylabel(r'$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}\tau$')
    axes[1].set_ylabel('Prediction / DELPHI'); axes[1].set_xlabel(r'$\tau=1-T$')
    axes[1].set_xlim(0, .5); axes[1].set_ylim(.4, 2.6); axes[1].set_yticks([.5, 1, 1.5, 2, 2.5])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(.55, .985))
    save_plot(fig, out, 'thrust_comparison')

    internal = (lo >= .05-1e-12) & (hi <= .30+1e-12)
    ei = np.r_[lo[internal], hi[internal][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(F.TW, 3.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.31, top=.90, wspace=.13)
    for ax, phase, title in zip(axes, ('before', 'after'), ('(a) Shape only', '(b) Shape + rate')):
        key = 'h_converted' if phase == 'before' else 'new_converted'
        c = mix[key][internal]/data[internal]
        gen = mix['generator_'+phase][internal]/data[internal]
        up, dn = (mix['theory_'+side+'_'+phase][internal]/data[internal] for side in ('up', 'down'))
        ax.stairs(c+gen, ei, baseline=c-gen, fill=True, color=F.GEN, alpha=.23, lw=0)
        ax.stairs(c+up, ei, baseline=c-dn, fill=True, color=F.THEORY, alpha=.22, lw=0)
        ax.stairs(c, ei, baseline=None, color=F.METHOD, lw=1.3)
        for g, ls in (('sherpa', (0, (4, 2))), ('herwig', (0, (1, 1.2)))):
            ax.stairs(d[g][key][internal]/data[internal], ei, baseline=None, color=F.METHOD, ls=ls, lw=1)
        ax.errorbar((lo[internal]+hi[internal])/2, np.ones(internal.sum()),
                    yerr=err[internal]/data[internal], fmt='o', color=F.DATA, ms=2, lw=.55)
        ax.axhline(1, color=F.REF, lw=.65)
        ax.set_title(title, loc='left', pad=6); ax.set_xlim(.05, .30)
        ax.set_xticks([.05, .10, .15, .20, .25, .30]); ax.set_xlabel(r'$\tau=1-T$')
        ax.set_ylim(.68, 1.22); F.trim(ax)
    axes[0].set_ylabel('Prediction / DELPHI')
    fig.legend(handles=[Line2D([], [], color=F.METHOD, label='Mixture'),
                        Line2D([], [], color=F.METHOD, ls='--', label='Sherpa'),
                        Line2D([], [], color=F.METHOD, ls=':', label='Herwig'),
                        Patch(facecolor=F.GEN, alpha=.23, label='Generator'),
                        Patch(facecolor=F.THEORY, alpha=.22, label='Theory')],
               loc='lower center', bbox_to_anchor=(.55, .015), ncol=3)
    save_plot(fig, out, 'window_comparison')

    # Same fraction and final-state definition as the current paper's Fig. 9.
    fig, axes = plt.subplots(1, 2, figsize=(F.TW, 2.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.97, bottom=.30, top=.88, wspace=.13)
    for ax, phase, title in zip(axes, ('before', 'after'), ('(a) Shape only', '(b) Shape + rate')):
        ax.axvspan(summary['data_fraction']-summary['data_error'], summary['data_fraction']+summary['data_error'],
                   color=F.BASE, alpha=.15, lw=0)
        ax.axvline(summary['data_fraction'], color=F.REF, lw=.65, ls=(0, (3, 3)))
        for y, g in zip((2, 1, 0), ('sherpa', 'herwig', 'mixture')):
            s = summary['generators'][g][phase]
            c = s['converted_fraction']
            ax.plot([c-s['generator_sd'], c+s['generator_sd']], [y, y], color=F.GEN, lw=7, alpha=.27, solid_capstyle='butt')
            ax.plot([c-s['learning_sd'], c+s['learning_sd']], [y, y], color=F.LEARN, lw=2.2, solid_capstyle='butt')
            ax.errorbar(c, y, xerr=[[s['theory_down']], [s['theory_up']]], fmt='o', ms=3.2,
                        color=F.METHOD, ecolor=F.THEORY, lw=1.3, capsize=3, capthick=1)
        ax.set_title(title, loc='left', pad=7); ax.set_xlim(.31, .49)
        ax.set_xticks([.32, .36, .40, .44, .48]); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel(r'$P(0.05\leq\tau<0.30)$'); ax.set_yticks([2, 1, 0], ['Sherpa', 'Herwig', 'Mixture'])
        ax.set_ylim(-.5, 2.5); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); F.trim(ax)
    fig.legend(handles=[Line2D([], [], color=F.GEN, lw=4, alpha=.5, label='Generator'),
                        Line2D([], [], color=F.THEORY, marker='|', label='Theory'),
                        Line2D([], [], color=F.LEARN, lw=2, label='Learning'),
                        Patch(facecolor=F.BASE, alpha=.15, label='DELPHI')],
               loc='lower center', bbox_to_anchor=(.55, .015), ncol=4)
    save_plot(fig, out, 'fraction_comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    output = HERE/'output'
    theory_file = out/'theory_targets.json'
    theory = json.loads(theory_file.read_text())
    vf = output/'theory_variations_MIX_lin.json'
    V = json.loads(vf.read_text())
    cbfile = output/'calc_bins_nlo_point.json'; CB = json.loads(cbfile.read_text())['delphi']
    lo, hi = np.r_[V['lo_low'], V['lo']], np.r_[V['hi_low'], V['hi']]
    widths = hi-lo; data, err = np.r_[V['data_low'], V['data']], np.r_[V['err_low'], V['err']]
    window = np.array([.05, 1/3]); target = theory['central']['pwin']
    with np.load(output/'wifi_MIXGEOV3S9_fit.npz') as z:
        C = z['C'].copy()
    d = {g: load_case(output, g, window, target, V, lo, hi, C) for g in ('sherpa', 'herwig')}
    internal = (lo >= .05-1e-12) & (hi <= .30+1e-12)
    fraction_weights = widths*internal
    keylabel = lambda k: k.split(':', 1)[1]
    for g in d:
        case = d[g]
        case['delta_before'], case['delta_after'] = {}, {}
        case['fraction_delta_before'], case['fraction_delta_after'] = {}, {}
        for key in V['variations']:
            h = case['variation_h'][key]; pg = case['variation_p'][key]
            tp = theory['variations'][key]['pwin']
            case['delta_before'][keylabel(key)] = (h-case['h'])*case['factors']
            transformed = correct(h, pg, tp, lo, hi, window)*case['factors']
            case['delta_after'][keylabel(key)] = transformed-case['new_converted']
            for phase in ('before', 'after'):
                case['fraction_delta_'+phase][keylabel(key)] = float(case['delta_'+phase][keylabel(key)]@fraction_weights)
        case['generator_before'] = (case['draws']['full']['before']*case['factors']).std(axis=0, ddof=1)
        case['generator_after'] = (case['draws']['full']['after']*case['factors']).std(axis=0, ddof=1)
        for phase, J in (('before', case['jac']), ('after', case['jac_new'])):
            jc = J*case['factors'][:, None]
            case['learning_'+phase] = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', jc, C, jc), 0))
            case['theory_up_'+phase] = theory_band(case['delta_'+phase], True)
            case['theory_down_'+phase] = theory_band(case['delta_'+phase], False)
    d['mixture'] = {}
    mix = d['mixture']
    for key in ('h', 'h_converted', 'new', 'new_converted'):
        mix[key] = .5*(d['sherpa'][key]+d['herwig'][key])
    for phase in ('before', 'after'):
        for side in ('up', 'down'):
            mix['theory_'+side+'_'+phase] = theory_band_mix(d['sherpa']['delta_'+phase], d['herwig']['delta_'+phase], side=='up')
        draws = []
        for g in ('sherpa', 'herwig'):
            case = d[g]; xi = case['draws']['full']['theta'][:, -1]
            # Pure archives fix xi; the published mixture archive supplies its common draw.
            draws.append(case['draws']['full'][phase]*case['factors'])
        with np.load(output/'envelope_test_MIXGEOV3S9_full_lin.npz') as z:
            theta = z['TH']; xi = theta[:, -1]
            for g, own in (('sherpa', slice(0, 8)), ('herwig', slice(8, 16))):
                assert np.array_equal(theta[:, own], d[g]['draws']['full']['theta'][:, own])
            mixed = (1-xi[:, None])*draws[0]+xi[:, None]*draws[1]
            if phase == 'before':
                n = len(z['labels']); labs = [str(v) for v in z['labels']]
                cols = [n+labs.index(f'thr_{a:.3f}_{b:.3f}') for a, b in zip(lo, hi)]
                raw = ((1-xi[:, None])*d['sherpa']['draws']['full']['before']+
                       xi[:, None]*d['herwig']['draws']['full']['before'])
                assert np.allclose(raw, z['Y_theta'][:, cols], rtol=1e-12, atol=1e-14)
        mix['draws_'+phase] = mixed
        mix['generator_'+phase] = mixed.std(axis=0, ddof=1)
        Js = [d[g]['jac' if phase=='before' else 'jac_new']*d[g]['factors'][:, None] for g in ('sherpa', 'herwig')]
        cov = sum((1/3 if a==b else 1/6)*(Js[a]@C@Js[b].T) for a in range(2) for b in range(2))
        mix['cov_learning_'+phase] = cov
        mix['learning_'+phase] = np.sqrt(np.maximum(np.diag(cov), 0))
    # The displayed fraction integrates the converted twelve bins, not the full constraint window.
    cblo, cbhi = np.asarray(CB['lo']), np.asarray(CB['hi'])
    sel = (cblo >= .05-1e-12) & (cbhi <= .30+1e-12)
    dw = (cbhi-cblo)[sel]
    data_fraction = float(fraction_weights@data)
    data_error = float(np.hypot(np.linalg.norm(dw*np.asarray(CB['stat'])[sel]), dw@np.asarray(CB['sys'])[sel]))
    summary = dict(prescription='Sequential exact rate matching after the existing soft shape fit; not a joint soft optimum.',
                   constraint_window=window, displayed_window=[.05, .30], target_fraction=target,
                   displayed_final_state='Fixed per-generator conversion to DELPHI, applied after matching.',
                   data_fraction=data_fraction, data_error=data_error, generators={})
    numerical_checks = {}
    for g in ('sherpa', 'herwig', 'mixture'):
        case = d[g]; record = {}
        for phase in ('before', 'after'):
            key = 'h_converted' if phase=='before' else 'new_converted'
            point = float(case[key]@fraction_weights)
            draws = case['draws_'+phase] if g=='mixture' else case['draws']['full'][phase]*case['factors']
            if g=='mixture':
                lsd = float(np.sqrt(max(fraction_weights@case['cov_learning_'+phase]@fraction_weights, 0)))
                up = theory_band_mix(d['sherpa']['fraction_delta_'+phase], d['herwig']['fraction_delta_'+phase], True)
                dn = theory_band_mix(d['sherpa']['fraction_delta_'+phase], d['herwig']['fraction_delta_'+phase], False)
            else:
                J = case['jac' if phase=='before' else 'jac_new']*case['factors'][:, None]
                slope = fraction_weights@J
                lsd = float(np.sqrt(max(slope@C@slope, 0)))
                up = theory_band(case['fraction_delta_'+phase], True)
                dn = theory_band(case['fraction_delta_'+phase], False)
            record[phase] = dict(converted_fraction=point, generator_sd=float((draws@fraction_weights).std(ddof=1)),
                                 learning_sd=lsd, theory_up=float(up), theory_down=float(dn),
                                 relative_difference_from_data=point/data_fraction-1)
        if g!='mixture':
            record.update(full_window_fraction_before=case['p'], full_window_fraction_after=target,
                          window_weight_factor=case['a'], outside_weight_factor=case['b'])
            inside = (lo >= window[0]-1e-12) & (hi <= window[1]+1e-12)
            numerical_checks[g] = dict(total_probability_before=float(case['h']@widths),
                                       total_probability_after=float(case['new']@widths),
                                       maximum_conditional_bin_relative_change=float(np.max(np.abs(case['new'][inside]/target/(case['h'][inside]/case['p'])-1))))
            pg = case['draws']['full']['pwin']
            numerical_checks[g]['all_2000_draws_normalized'] = bool(np.allclose(case['draws']['full']['after']@widths, 1, atol=2e-10, rtol=0))
            numerical_checks[g]['minimum_original_draw_fraction'] = float(pg.min())
            numerical_checks[g]['maximum_original_draw_fraction'] = float(pg.max())
        summary['generators'][g] = record
    summary['checks'] = numerical_checks
    summary['mixture_first_bin_ratio_to_data'] = dict(before=float(mix['h_converted'][0]/data[0]), after=float(mix['new_converted'][0]/data[0]))
    summary['uncertainty_conventions'] = dict(generator='SD of the same2000 parameter draws, corrected separately before mixing.',
        theory='The same rate and shape variation are used together; Eq.(thband), mixture variance averaged over uniform xi.',
        learning='Differentiate the normalization correction; retain full bin and shared-generator covariance.',
        statistical='Finite-reference statistical errors are not included in the displayed component bands.')
    sources = [vf, cbfile, theory_file, output/'wifi_MIXGEOV3S9_fit.npz',
               output/'envelope_test_MIXGEOV3S9_full_lin.npz']
    sources += [Path(d[g]['source_files'][b]) for g in ('sherpa', 'herwig') for b in ('full', 'tune')]
    summary['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(out/'summary.json', summary)
    arrays = dict(lo=lo, hi=hi, data=data, data_error=err)
    for g, case in d.items():
        for key in ('h', 'h_converted', 'new', 'new_converted', 'generator_before', 'generator_after',
                    'learning_before', 'learning_after', 'theory_up_before', 'theory_down_before', 'theory_up_after', 'theory_down_after'):
            arrays[g+'_'+key] = case[key]
    np.savez_compressed(out/'predictions.npz', **arrays)
    draw_plots(out, d, lo, hi, data, err, CB, summary)
    print(json.dumps({g: summary['generators'][g] for g in summary['generators']}, indent=2, default=json_value))
    print('Saved', out)


if __name__ == '__main__':
    main()
