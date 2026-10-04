#!/usr/bin/env python3
"""Every number quoted in Sec. 6 of the paper, computed from the result files in output/ and written
to output/sec6_numbers.json under the names the text uses (for example AS, AS_EHI, AS_ELO for the
coupling and its experimental errors). The values are formatted as printed, and where the paper's
wording depends on a result (for example whether a variant moves the coupling), the chosen wording is
stored as well, so every statement of Sec. 6 can be traced to the files it comes from.

    python make_sec6_numbers.py
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, '.')

O = os.environ.get('SEC6_RESULTS', 'output/')
J = lambda f: json.load(open(O + f))
fit = J('profile_MIX17ext_central.json'); summ = J('profile_MIX17ext_central_summary.json')
var = J('profile_MIX17ext_central_rows_var.json'); ex = J('profile_MIX17ext_central_exchange.json')
chk = J('profile_MIX17ext_central_check.json'); flo = J('profile_MIX17ext_central_floorscan.json')
shr = J('profile_MIX17ext_central_rows_shrink0.90.json'); win = J('profile_MIX17win_central_rows.json')
geo = J('profile_MIX17geo_central_rows.json'); ps = J('pseudo_rows_profile_MIX17ext_central_summary.json')
c1m = J('profile_C_1Mext_central_rows.json'); rows = J('profile_MIX17ext_central_rows.json')
WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
        12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen'}
f4 = lambda x: f'{x:.4f}'; f3 = lambda x: f'{x:.3f}'; f1 = lambda x: f'{x:.1f}'
def sgn(x, nd):
    s = f'{x:+.{nd}f}'
    return s.replace('+', '+').replace('-', '-')
N = {}
# the values and errors of both fit parameters come from the continuous profiles (profile_rows.py)
pa, p0 = rows['variations']['central'], rows['alpha_0_profile']
N['AS'], N['AS_EHI'], N['AS_ELO'] = f4(pa['value']), f4(pa['err_hi']), f4(pa['err_lo'])
N['A0'], N['A0_EHI'], N['A0_ELO'] = f3(p0['value']), f3(p0['err_hi']), f3(p0['err_lo'])
N['RHO'] = f"{rows['rho']:+.2f}"
N['CHI2'] = f1(pa['chi2min'])
s = var['summary']
N['AS_PHI'] = f4(max(0.0, s['pert_env_alpha_s'][1])); N['AS_PLO'] = f4(max(0.0, -s['pert_env_alpha_s'][0]))
npm = max(abs(s['np_env_alpha_s'][0]), abs(s['np_env_alpha_s'][1]))
# the nonperturbative model: a term in the result only when it moves the coupling visibly
N['AS_NP'] = rf'\,\pm {npm:.4f}\,({{\rm np}})' if npm >= 0.00005 else ''
N['NP_LEAD'] = ('The nonperturbative model does not change the coupling.' if npm < 0.00005 else
                'The nonperturbative model contributes little to the error on the coupling.')
N['NP_DAS_SENT'] = ((f"In all three cases the coupling changes by less than $10^{{{int(np.ceil(np.log10(npm))) if npm > 0 else -6}}}$."
                     if npm < 0.00005 else f"In all three cases the coupling changes by at most ${npm:.4f}$."))
N['NP_DA0_SUB3'] = f"{s['np_d_alpha_0'][2]:+.2f}"
# what drives the asymmetric perturbative error
vv = {k: r['value'] - pa['value'] for k, r in var['variations'].items() if k.startswith('(')}
kmax = max(vv, key=vv.get)
c2v = [r['chi2min'] for k, r in var['variations'].items() if k.startswith('(')]
N['VAR_C2'] = (f" Under the variations the $\\chi^2$ ranges from ${min(c2v):.0f}$ to ${max(c2v):.0f}$, and the envelope keeps every "
               "variation regardless of how well it describes the data.")
N['PERT_LEAD'] = ('' if 'modR' not in kmax else
                  f", and the largest shifts come from the modified-$R$ matching scheme, which at central scales alone moves the coupling by ${vv["(1.0, 1.0, 'modR', 1.0)"]:+.4f}$")
nup = s['n_up']
N['NUP_CLAUSE'] = ('all eleven variations move the coupling up' if nup == 11 else
                   f'{WORD[nup]} of the eleven variations move the coupling up')
th = np.array(summ['theta']); N['FRAC'] = f"{th[-1]:.2f}"; N['KL'] = f"{summ['kl']:.4f}"
q = c1m['variations']['central']
N['C_AS'], N['C_EHI'], N['C_ELO'], N['C_CHI2'] = f4(q['value']), f4(q['err_hi']), f4(q['err_lo']), f1(q['chi2min'])
N['C_DCHI'] = f1(q['chi2min'] - pa['chi2min']); N['C_DAS'] = f4(abs(q['value'] - pa['value']))
N['C_DERR'] = f"{np.ceil(100*max(abs(q[k]/pa[k] - 1) for k in ('err_lo', 'err_hi'))):.0f}"

ned = len([e for e in summ['edges'] if e[0] != 'fraction'])
N['NEDGE'] = WORD[ned]
TT = lambda x: r'\texttt{' + x.replace('_', r'\_') + '}'
NAME = {'S:pt_max': 'PT_MAX', 'S:gamma_l': 'GAMMA_L', 'S:strange_fraction': 'STRANGE_FRACTION', 'S:alpha_l': 'ALPHA_L',
        'S:baryon_fraction': 'BARYON_FRACTION', 'S:alpha_g': 'ALPHA_G', 'S:beta_l': 'BETA_L', 'S:alphas': 'ALPHAS(MZ)',
        'H:alpha_fsr': 'AlphaIn', 'H:ptmin': 'pTmin', 'H:clmax': 'ClMaxLight', 'H:clpow': 'ClPowLight', 'H:psplit': 'PSplitLight',
        'H:pwtsquark': 'PwtSquark', 'H:pwtdiquark': 'PwtDIquark', 'H:clsmr': 'ClSmrLight'}
eS = [TT(NAME[e[0]]) for e in summ['edges'] if e[0].startswith('S:')]; eH = [TT(NAME[e[0]]) for e in summ['edges'] if e[0].startswith('H:')]
lst = lambda L: L[0] if len(L) == 1 else ', '.join(L[:-1]) + ' and ' + L[-1]
N['EDGE_LIST'] = 'namely ' + ' and '.join(x for x in ([f'{lst(eS)} for Sherpa'] if eS else []) + ([f'{lst(eH)} for Herwig'] if eH else []))
N['SHRINK_PCT'] = f"{100*shr['shrink']:.0f}"
d = shr['variations']['central']['value'] - pa['value']; N['SHRINK_DAS'] = f'{d:+.4f}'
N['SHRINK_CMP'] = ('well inside the experimental error' if abs(d) < 0.5*min(pa['err_hi'], pa['err_lo'])
                   else 'comparable to the experimental error')
N['NP_DA0'] = f"${s['np_d_alpha_0'][0]:+.2f}$ and ${s['np_d_alpha_0'][1]:+.2f}$"   # Milan factor up and down
N['NP_DAS'] = f"{max(abs(x) for x in s['np_env_alpha_s']):.4f}"
w = win['variations']['central']
N['WIN_AS'], N['WIN_EHI'], N['WIN_ELO'] = f4(w['value']), f4(w['err_hi']), f4(w['err_lo'])
dw = w['value'] - pa['value']; nsd = abs(dw)/(pa['err_lo'] if dw < 0 else pa['err_hi'])
N['WIN_CLAUSE'] = (', so the coupling is determined by the shape of the distribution inside the window.' if nsd < 0.5 else
                   f", {'lower' if dw < 0 else 'higher'} by ${abs(dw):.4f}$, about {('one' if nsd < 1.5 else WORD.get(int(round(nsd)), f'{nsd:.0f}'))} "
                   f"experimental standard deviation{'' if nsd < 1.5 else 's'} of \\Eq{{alphasfit}}. The bins above the window and the "
                   "multiplicity constrain the generator parameters, which still shape the distribution inside the window within "
                   "the theory uncertainty of the anchored moments, so they also affect the fitted coupling. The multiplicity "
                   "enters with the L3 error alone, although the reweighting reproduces mean multiplicities only to between a few "
                   "tenths of a percent and one percent (Secs.~\\ref{sec:anypoint} and~\\ref{sec:bigboxes}), and this fit also "
                   "covers the extreme case in which the multiplicity has no weight at all.")

g = geo['variations']['central']; N['GEO_AS'], N['GEO_CHI2'] = f4(g['value']), f1(g['chi2min'])
N['GEO_EHI'], N['GEO_ELO'] = f4(g['err_hi']), f4(g['err_lo'])
same_c = abs(g['value'] - pa['value']) < 0.25*min(pa['err_lo'], pa['err_hi'])
same_e = all(abs(g[k]/pa[k] - 1) < 0.25 for k in ('err_lo', 'err_hi'))
if same_c and same_e:
    N['GEO_SENTENCE'] = 'The result therefore does not depend on how the mixture interpolates between the two generators.'
elif same_c:
    N['GEO_SENTENCE'] = ('The central value therefore does not depend on how the mixture interpolates between the two '
                         'generators, while its errors change by up to '
                         f"{100*max(abs(g[k]/pa[k] - 1) for k in ('err_lo', 'err_hi')):.0f} percent.")
else:
    raise SystemExit(f'the product form moves the coupling to {g["value"]:.4f}: write this sentence by hand')
NUMW = {**WORD, 20: 'twenty', 30: 'thirty', 38: 'thirty-eight', 39: 'thirty-nine', 40: 'forty', 50: 'fifty'}
WORDH = lambda h: {1: 'one hundred', 2: 'two hundred', 3: 'three hundred', 4: 'four hundred', 5: 'five hundred', 6: 'six hundred', 7: 'seven hundred', 8: 'eight hundred', 9: 'nine hundred'}.get(h, f'{100*h}')
N['NPSEUDO'] = NUMW.get(ps['n_ok'], str(ps['n_ok']))
if ps['n_ok'] < ps['n']:
    if not os.environ.get('SEC6_ALLOW_DROPPED'):
        raise SystemExit(f"pseudo-data sets {ps.get('dropped')} have no usable minimum: rerun them with longer rows")
    N['NPSEUDO'] = f"{NUMW.get(ps['n_ok'], str(ps['n_ok']))} of {NUMW.get(ps['n'], str(ps['n']))}"
sd = np.sqrt(0.68*0.32/ps['n_ok'])
N['PS_CLAUSE'] = ('consistent with the 68 percent expected' if abs(ps['coverage'] - 0.68) <= sd else
                  'against 68 percent expected, so the quoted errors are slightly conservative' if ps['coverage'] > 0.68 else
                  'against 68 percent expected, so the quoted errors are slightly too small')
N['PS_BIAS'], N['PS_BERR'] = f"{ps['bias']:+.4f}", f"{ps['bias_err']:.4f}"
N['PS_COV'] = f"{100*ps['coverage']:.0f}"
N['PS_SCAT'], N['PS_MERR'] = f"{ps['scatter']:.4f}", f"{ps['median_error']:.4f}"
per = summ['per']; N['CH_A'], N['CH_D'] = f1(per['aleph']), f1(per['delphi'])
rd = summ['residuals']['delphi']
up = [z for lo, hi, z in zip(rd['lo'], rd['hi'], rd['resid']) if lo >= 0.06 - 1e-9 and hi <= 0.14 + 1e-9]
dn = [z for lo, hi, z in zip(rd['lo'], rd['hi'], rd['resid']) if lo >= 0.20 - 1e-9 and hi <= 0.30 + 1e-9]
N['D_UP'], N['D_DN'] = f'+{max(up):.1f}', f'{min(dn):.1f}'
# the parton-level check
N['P_RMS'], N['P_MAX'] = f"{chk['rms_pull']:.1f}", f"{chk['max_pull']:.1f}"
N['G_SHIFT'], N['D_SHIFT'] = f"{abs(chk['gen_shift']):.4f}", f"{abs(chk['dispersive_shift']):.4f}"
sf = chk['cutoff_scan_fit']; lo_ = min(sf, key=lambda r: r['ptmin'])
impl = [r['alpha_0_implied'] for r in sf if r['alpha_0_implied'] == r['alpha_0_implied']]
a0i = min(impl); N['A0_IMPL'] = f'{np.floor(10*a0i)/10:.1f}'
N['G_SHIFT_LO'], N['P_RMS_LO'] = f"{abs(lo_['gen_shift']):.4f}", f"{lo_['rms_pull']:.1f}"
hw = chk['cutoff_scan_herwig']
N['HW_LO'] = f"{abs(min(hw, key=lambda r: r['ptmin'])['gen_shift']):.4f}"; N['HW_HI'] = f"{abs(max(hw, key=lambda r: r['ptmin'])['gen_shift']):.4f}"
# holding alpha_0 at the implied value: the alpha_0 profile of the fit, where at each alpha_0 the
# coupling and the nuisance parameters are minimized (profile_rows.py columns)
cols = rows['alpha_0_profile']['cols']; ca0, cas, cc2 = (np.array(cols[k]) for k in ('alpha0', 'alphas', 'chi2'))
a0h = float(N['A0_IMPL'])
if not (ca0.min() <= a0h <= ca0.max()):
    raise SystemExit(f'the held alpha_0 {a0h} lies outside the computed alpha_0 columns {ca0.min()}-{ca0.max()}: extend them')
ah = float(np.interp(a0h, ca0, cas))
N['AS_HELD'] = f'{ah:.3f}'; N['AS_HELD_SHIFT'] = f'${ah - pa["value"]:.3f}$'
N['CHI2_HELD'] = f'{np.interp(a0h, ca0, cc2) - pa["chi2min"]:.0f}'
# the transport table and the text after it
R = ex['rows']
def row(k, label, nd_m, nd_b, nd_n):
    r = R[k]
    kn = '--' if r['knobs_after'] is None else f"${r['knobs_after']:.{nd_n}f}$"
    return (f"{label} & ${r['unanchored']:.{nd_m}f}$ & ${r['anchored']:.{nd_m}f}$ & ${r['theory']:.{nd_b}f}$ & "
            f"${r['np_model']:.{nd_b}f}$ & {kn} \\\\")
TAB = '\n'.join([row('tau_win', r'$\langle \tau\rangle_{\rm window}$', 5, 5, 4), row('thrust', r'$\langle 1-T\rangle$', 5, 5, 4),
                 row('B_total', r'$\langle B_{\mathrm{tot}}\rangle$', 5, 5, 4), row('rho_heavy', r'$\langle \rho_H\rangle$', 5, 5, 4),
                 r'\midrule', row('nch', r'$\langle n_{\rm ch}\rangle$', 3, 3, 3), row('nbaryon', r'$\langle n_{\rm baryon}\rangle$', 4, 4, 3)])
T = R['thrust']; tot = np.sqrt(T['knobs_after']**2 + T['theory']**2 + 0.00068**2); z = (T['anchored'] - 0.06671)/tot
N['TW'], N['TW_P'] = f"{R['tau_win']['anchored']:.5f}", f"{R['tau_win']['theory']:.5f}"
N['T'], N['T_N'], N['T_P'] = f"{T['anchored']:.4f}", f"{T['knobs_after']:.4f}", f"{T['theory']:.4f}"
N['T_SIG'] = (f'{WORD[int(round(10*abs(z)))]} tenths of a standard deviation' if abs(z) < 0.95 else f'{abs(z):.1f} standard deviations')
N['T_DIR'] = 'above' if z > 0 else 'below'
pk = J('profile_MIX17ext_central_peak.json')
N['PK_MOD'], N['PK_DAT'] = f"{100*pk['anchored']['below']:.1f}", f"{100*pk['data_below']:.1f}"
N['PK_SHIFT'] = f"{pk['window_shift']:.4f}"
N['NCH'], N['NCH_N'] = f"{R['nch']['anchored']:.2f}", f"{R['nch']['knobs_after']:.2f}"
# the calculation-only fits of Sec. 6.4 (calc_window_fits.py) and the ALEPH fit of App. D (thrust_chain.py)
from scipy.stats import norm
cw = J('calc_window_fits.json'); cs = J('chain_summary.json')
own, cond = cw['own'], cw['conditional']
N['CW_DEL'] = f"{cw['delphi_window'][0]['grid_min']:.1f}"; N['CW_DEL_VAR'] = f"{min(r['grid_min'] for r in cw['delphi_window'][1:]):.1f}"
N['CW_LO_C2'], N['CW_LO_AS'] = f"{own['lower']['chi2']:.1f}", f"{own['lower']['alpha_s']:.3f}"
N['CW_UP_C2'], N['CW_UP_AS'] = f"{own['upper']['chi2']:.1f}", f"{own['upper']['alpha_s']:.3f}"
N['CW_DCHI'] = f"{own['delta_chi2']:.1f}"; N['CW_SIG'] = f"{norm.isf(0.5*np.exp(-0.5*own['delta_chi2'])):.1f}"
N['CW_COND_DCHI'] = f"{cond['delta_chi2']:.1f}"
N['CW_COND_ATLEAST'] = 'at least ' if (cond['lower']['edge'] or cond['upper']['edge']) else ''
N['CW_OWN_ALL'], N['CW_OWN_ALEPH'] = f"{own['whole']['alpha_s']:.4f}", f"{own['aleph']['alpha_s']:.4f}"
N['CW_COND_ALL'] = f"{cond['whole']['alpha_s']:.4f}"
N['CW_OWN_ERR'] = f"{max(own['whole']['err_hi'], own['whole']['err_lo']):.4f}"
N['CW_COND_EHI'], N['CW_COND_ELO'] = f"{cond['whole']['err_hi']:.4f}", f"{cond['whole']['err_lo']:.4f}"
N['APPD_AS'] = f"{cs['alpha_s']:.4f}"
ao = cw['aleph_own_variations']; N['APPD_CHI2'] = f"{ao[0]['chi2']:.1f}"
vc = [r['chi2']/26 for r in ao[1:]]; N['APPD_VC2_LO'], N['APPD_VC2_HI'] = f"{min(vc):.2f}", f"{max(vc):.2f}"
N['SHAPE_SHIFT'] = f"{round(cond['whole']['alpha_s'] - own['whole']['alpha_s'], 3):.3f}"
ph, pl, eh, el = float(N['AS_PHI']), float(N['AS_PLO']), pa['err_hi'], pa['err_lo']
N['DOMINANCE'] = ('with an uncertainty dominated by the calculation' if ph > eh and pl > el else
                  'with the upward uncertainty dominated by the calculation' if ph > eh else
                  'with the downward uncertainty dominated by the calculation' if pl > el else
                  'with an uncertainty dominated by the data')
json.dump(N, open(os.path.join(O, 'sec6_numbers.json'), 'w'), indent=1)
for k, v in N.items():
    print(f'{k:16s} {v}')
