#!/usr/bin/env python3
"""Build the paper's tables and existing figures from rate_shape.npz."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import figs_r8 as F
from rate_matching_paper import serial, dump
from gentune import bands
from gentune.maxent import load_targets
from reproduce_paper import OBS, factor_vectors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=HERE/'output/rate_shape')
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--numbers',type=Path,required=True)
    args=p.parse_args();out=args.input.resolve();figs=args.figures.resolve();figs.mkdir(parents=True,exist_ok=True)
    z=np.load(out/'rate_shape.npz');old=np.load(HERE/'output/three_bands_MIXGEOV3S9_lin.npz')
    original=json.loads((HERE/'output/three_bands_MIXGEOV3S9_lin.json').read_text())
    V=json.loads((HERE/'output/theory_variations_MIX_lin.json').read_text())
    report=json.loads((out/'rate_shape.json').read_text())
    calc=load_targets(HERE/'release/targets_thrust_rate_shape.npz')
    factors27=factor_vectors()[0];factors={g:np.r_[factors27[g],np.ones(42)] for g in factors27}
    theta=z['theta'];nodes=z['nodes'];lo=z['lo'];hi=z['hi'];low=lo<.05-1e-12;fit=~low
    width=hi-lo;internal=(lo>=.05-1e-12)&(hi<=.30+1e-12)
    before=z['before'];after=z['after'];beforec=z['before_conv'];afterc=z['after_conv']
    gsd=z['generator'].copy();gsdc=z['generator_conv'].copy();gsd[1]=gsdc[1]=0.
    learn=np.sqrt(np.maximum(np.diag(z['C_learn']),0));learnc=np.sqrt(np.maximum(np.diag(z['C_learn_conv']),0))
    stat=np.sqrt(np.maximum(np.diag(z['C_stat']),0));statc=z['stat_conv'];up=z['theory_up'];dn=z['theory_down']
    upc=z['theory_up_conv'];dnc=z['theory_down_conv']
    mean_dict=lambda a:dict(zip(OBS,map(float,a[:7])))
    q=lambda *a:np.sqrt(sum(v*v for v in a))
    band_dict=lambda d,sd:{k:dict(mean=float(d[:,i].mean()),sd=float(sd[i]),p16=float(np.percentile(d[:,i],16)),p84=float(np.percentile(d[:,i],84))) for i,k in enumerate(OBS)}
    c=dict(theta=theta.tolist(),generator_only=mean_dict(before),imposed=mean_dict(after),
           learning_band=mean_dict(learn),statistical=mean_dict(stat),theory_up=mean_dict(up),theory_down=mean_dict(dn),
           theory_band=mean_dict(np.sqrt((up*up+dn*dn)/2)),combined_up=mean_dict(q(gsd[:51],learn,stat,up[:51])),
           combined_down=mean_dict(q(gsd[:51],learn,stat,dn[:51])))
    c['combined']=mean_dict(q(gsd[:51],learn,stat,np.sqrt((up[:51]**2+dn[:51]**2)/2)))
    for key in ('learning_band_generator_only','statistical_generator_only','combined_generator_only'):
        c[key]=original['central'][key]
    cc=dict(generator_only=mean_dict(beforec),imposed=mean_dict(afterc),learning_band=mean_dict(learnc),
            statistical=mean_dict(statc),theory_up=mean_dict(upc),theory_down=mean_dict(dnc),
            combined_up=mean_dict(q(gsdc[:51],learnc,statc,upc[:51])),combined_down=mean_dict(q(gsdc[:51],learnc,statc,dnc[:51])))
    cc['combined']=mean_dict(q(gsdc[:51],learnc,statc,np.sqrt((upc[:51]**2+dnc[:51]**2)/2)))
    for key in ('learning_band_generator_only','statistical_generator_only','combined_generator_only'):
        cc[key]=original['conv']['central'][key]
    c['generators']={}
    with np.load(HERE/'release/models/MIX17_wifi.npz') as wifi:C=wifi['C'].copy()
    for g in ('sherpa','herwig'):
        J=z[g+'_J_omega'];l=np.sqrt(np.maximum(np.einsum('ik,kl,il->i',J,C,J),0))
        c['generators'][g]=dict(generator_only=mean_dict(z[g+'_before']),imposed=mean_dict(z[g+'_after']),
                                learning_band=mean_dict(l),statistical=mean_dict(z[g+'_stat']),
                                theory_up=mean_dict(z[g+'_theory_up']),theory_down=mean_dict(z[g+'_theory_down']))
    j=dict(mix='linear',box='full',nodes=2000,test='delphi',central=c,
           calc=dict(window=calc.window,alpha_s=calc.meta['alpha_s'],alpha_0=calc.meta['alpha_0'],pwin=calc.rate),
           prescription='Conditional shape fit followed by exact rate matching, separately before mixing.',
           generator_band=dict(before=original['generator_band']['before'],after=band_dict(z['draws'],gsd)),
           conv=dict(central=cc,generator_band=dict(before=original['conv']['generator_band']['before'],after=band_dict(z['draws_conv'],gsdc))),
           conventions=original['conventions'])
    prefix=out/'three_bands_MIXGEOV3S9_lin';dump(prefix.with_suffix('.json'),j)
    arr={k:old[k].copy() for k in ('lo','hi','lo_low','hi_low','data','err','data_low','err_low','h_central_before','h_central_before_low',
                                  'h_central_before_conv','h_central_before_conv_low','gen_hist_before','gen_hist_before_low',
                                  'gen_hist_before_conv','gen_hist_before_conv_low','learn_hist_before','learn_hist_before_low',
                                  'learn_hist_before_conv','learn_hist_before_conv_low','comb_hist_before','comb_hist_before_low',
                                  'comb_hist_before_conv','comb_hist_before_conv_low',
                                  'stat_hist_before','stat_hist_before_low','stat_hist_before_conv','stat_hist_before_conv_low',
                                  'rows_before','H_before','rows_before_conv','H_before_conv')}
    arr.update(mix=np.array('linear'),obs=np.array(OBS),nodes=nodes,rows_after=z['draws'][:,:7],H_after=z['draws'][:,7:27][:,fit],
               H_after_conv=z['draws_conv'][:,7:27][:,fit],rows_after_conv=z['draws_conv'][:,:7],
               C_learn=z['C_learn'][:27,:27],C_learn_conv=z['C_learn_conv'][:27,:27])
    for suffix,h,g,l,s,u,d in [('',after,gsd,learn,stat,up,dn),('_conv',afterc,gsdc,learnc,statc,upc,dnc)]:
        vals=dict(h_central=h[7:27],gen_hist=g[7:27],learn_hist=l[7:27],stat_hist=s[7:27],
                  theory_hist_up=u[7:27],theory_hist_down=d[7:27],comb_hist_up=q(g[7:27],l[7:27],s[7:27],u[7:27]),
                  comb_hist_down=q(g[7:27],l[7:27],s[7:27],d[7:27]))
        for key,value in vals.items():arr[key+suffix]=value[fit];arr[key+suffix+'_low']=value[low]
    for g in ('sherpa','herwig'):
        for suffix,factor in [('',np.ones(20)),('_conv',factors[g][7:27])]:
            for key,values in [('h_central_'+g,z[g+'_after'][7:27]),('h_central_before_'+g,z[g+'_before'][7:27]),
                               ('theory_hist_up_'+g,z[g+'_theory_up'][7:27]),('theory_hist_down_'+g,z[g+'_theory_down'][7:27])]:
                values=values*factor;arr[key+suffix]=values[fit];arr[key+suffix+'_low']=values[low]
    np.savez_compressed(prefix.with_suffix('.npz'),**arr)
    # The existing full-range figure reads the central pure-generator curves.
    V['prescription']=j['prescription']
    V['neff_imposed']=report['central']['mixture']['n_eff']
    for g,xi in (('sherpa',0),('herwig',1)):
        row=z[g+'_after'];vc=row*factors[g];pure=V[f'xi{xi}']
        for key,value in [('h_imposed',row[7:27]),('h_imposed_conv',vc[7:27])]:
            pure[key]=value[fit].tolist();pure[key+'_low']=value[low].tolist()
        pure['o_imposed']=mean_dict(row);pure['o_imposed_conv']=mean_dict(vc)
        pure['neff_imposed']=report['central'][g]['n_eff']
    for key,value in [('h_imposed',after[7:27]),('h_imposed_conv',afterc[7:27])]:
        V[key]=value[fit].tolist();V[key+'_low']=value[low].tolist()
    V['o_imposed']=mean_dict(after);V['o_imposed_conv']=mean_dict(afterc)
    V['variations']={}
    for label in calc.variations:
        family='scale:' if label.startswith('(') else 'np:'
        record={}
        delta={g:z[g+'_delta_'+label] for g in ('sherpa','herwig')}
        dm=.5*sum(delta.values())
        dmc=.5*sum(delta[g]*factors[g] for g in delta)
        record.update(dh=dm[7:27][fit].tolist(),dh_low=dm[7:27][low].tolist(),do=mean_dict(dm),
                      dh_conv=dmc[7:27][fit].tolist(),dh_conv_low=dmc[7:27][low].tolist(),do_conv=mean_dict(dmc))
        for g in delta:
            record['dh_'+g]=delta[g][7:27][fit].tolist();record['dh_'+g+'_low']=delta[g][7:27][low].tolist()
            record['do_'+g]=mean_dict(delta[g])
        V['variations'][family+label]=record
    dump(out/'theory_variations_MIX_lin.json',V)
    # Existing Fig.7: retain its spectra and fixed-before ratio panels.
    full=np.load(HERE/'output/envelope_test_MIXGEOV3S9_full_lin.npz')
    nominal=np.load(HERE/'output/envelope_test_MIXGEOV3S9_tune_lin.npz')
    np.savez_compressed(out/'distribution_variations.npz',labels=full['labels'],mix=np.array('linear'),lo=full['lo'],hi=full['hi'],
                        Y_theta=np.c_[full['Y_theta'][:,:69],z['draws']],thrust_edges=full['thrust_edges'],nch_edges=full['nch_edges'])
    np.savez_compressed(out/'distribution_nominal.npz',labels=full['labels'],mix=np.array('linear'),centre=theta,
                        y_central=np.r_[nominal['y_central'][:69],after])
    os.environ.update(BANDS_NPZ=str(prefix.with_suffix('.npz')),BANDS_JSON=str(prefix.with_suffix('.json')),
                      VARS_JSON=str(out/'theory_variations_MIX_lin.json'),R8_OUT=str(figs))
    F.OUT=str(figs);F.style();F.bands()
    from polish_paper_figures import draw_budget
    draw_budget()
    # Existing main window figure now compares the full bin rates with data.
    data=np.r_[old['data_low'],old['data']];err=np.r_[old['err_low'],old['err']]
    edges=np.r_[lo[internal],hi[internal][-1]]
    fig,axs=plt.subplots(1,2,figsize=(F.TW,3.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.97,bottom=.31,top=.90,wspace=.13)
    for k,ax in enumerate(axs):
        h=beforec[7:27] if k==0 else afterc[7:27]
        if k==0:
            upper=np.r_[old['comb_hist_before_conv_low'],old['comb_hist_before_conv']]
            lower=upper;color=F.BASE_DARK;fill=F.BASE
        else:
            upper=q(gsdc[7:27],learnc[7:27],statc[7:27],upc[7:27]);lower=q(gsdc[7:27],learnc[7:27],statc[7:27],dnc[7:27]);color=F.METHOD;fill=F.METHOD
        ax.stairs((h+upper)[internal]/data[internal],edges,baseline=(h-lower)[internal]/data[internal],fill=True,color=fill,alpha=.22,lw=0)
        ax.stairs(h[internal]/data[internal],edges,baseline=None,color=color,lw=1.3)
        for g,xi,ls in [('sherpa',0,(0,(4,2))),('herwig',1,(0,(1,1.2)))]:
            raw=z[g+('_before' if k==0 else '_after')]*factors[g]
            ax.stairs(raw[7:27][internal]/data[internal],edges,baseline=None,color=color,ls=ls,lw=1)
        ax.errorbar((lo[internal]+hi[internal])/2,np.ones(internal.sum()),yerr=err[internal]/data[internal],fmt='o',ms=2.2,lw=.6,color=F.DATA)
        ax.axhline(1,color=F.REF,lw=.65);ax.set_xlim(.05,.30);ax.set_ylim(.67,1.22)
        ax.set_xticks([.05,.10,.15,.20,.25,.30]);ax.set_xlabel(r'$\tau=1-T$')
        ax.set_title('(a) Before' if k==0 else '(b) After',loc='left',pad=6);F.trim(ax)
    axs[0].set_ylabel('Prediction / DELPHI')
    fig.legend(handles=[Line2D([],[],color=F.METHOD,label='Mixture'),Line2D([],[],color=F.METHOD,ls='--',label='Sherpa'),
                        Line2D([],[],color=F.METHOD,ls=':',label='Herwig'),Line2D([],[],color=F.DATA,ls='',marker='o',label='DELPHI')],
               loc='lower center',bbox_to_anchor=(.55,.03),ncol=4)
    fig.savefig(figs/'r8_window_shape.pdf');fig.savefig(figs/'r8_window_shape.png',dpi=240);plt.close(fig)
    # Derived manuscript numbers are emitted separately from unchanged validation macros.
    macros={}
    put=lambda key,value:macros.__setitem__('Rate'+key,str(value))
    fmt=lambda value,digits=2:f'{float(value):.{digits}g}'
    def sci(value):
        s=fmt(value)
        if 'e' not in s:return s
        a,b=s.split('e');return r'\ensuremath{'+a+r'\times10^{'+str(int(b))+'}}'
    fw=width*internal
    fd=float(data@fw);fp=float(afterc[7:27]@fw)
    put('WindowFraction',f'{100*calc.rate:.2f}');put('DataFraction',f'{100*fd:.2f}');put('PredictionFraction',f'{100*fp:.2f}')
    put('FractionExcess',f'{100*(fp/fd-1):.2f}');put('PeakRatio',f'{afterc[7]/data[0]:.2f}')
    put('LastInternalDeficit',fmt(100*(1-afterc[7:27][internal][-1]/data[internal][-1])))
    relgen=100*gsdc[7:27][internal]/afterc[7:27][internal]
    put('BinGenMin',sci(relgen.min()));put('BinGenMax',sci(relgen.max()))
    for name,value in [('TheoryRateUp',100*up[1]/after[1]),('TheoryRateDown',100*dn[1]/after[1]),
                       ('StatMeansMax',100*np.max(stat[:7]/np.abs(after[:7])))]:put(name,sci(value))
    put('Thrust',f'{afterc[2]:.5f}');put('ThrustUp',f'{q(gsdc[2],learnc[2],statc[2],upc[2]):.5f}');put('ThrustDown',f'{q(gsdc[2],learnc[2],statc[2],dnc[2]):.5f}')
    put('Nch',f'{after[5]:.2f}');put('NchGen',fmt(gsd[5]));put('NchDataPull',fmt((18.63-after[5])/np.hypot(.11,q(gsd[5],learn[5],stat[5],up[5]))))
    for g,tag in [('sherpa','Sherpa'),('herwig','Herwig')]:
        put('MaxWeight'+tag,fmt(json.loads((out/'rate_shape.json').read_text())['central'][g]['max_weight_over_mean']))
    put('MaxWeightMid',fmt(json.loads((out/'rate_shape.json').read_text())['central']['mixture']['max_weight_over_mean']))
    args.numbers.write_text('% Generated by analysis/rate_matching_outputs.py; rate-and-shape prescription.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    dump(out/'manuscript_numbers.json',macros)
    print('Wrote compatible result files, figures and manuscript numbers')


if __name__=='__main__':main()
