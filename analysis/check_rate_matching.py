#!/usr/bin/env python3
"""Independent cache, event-weight and full-reference derivative checks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'release'),str(HERE/'release/examples')]
from gentune import load_head,axes
from gentune.maxent import load_targets
from reproduce_paper import Reference
from rate_matching_trial import correct


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    z=np.load(HERE/'output/rate_shape/rate_shape.npz');calc=load_targets(HERE/'release/targets_thrust_rate_shape.npz')
    checks={};lo,hi=z['lo'],z['hi']
    for g in ('sherpa','herwig'):
        with np.load(HERE/f'output/envelope_test_MIXGEOV3S9_full_{g}_lin.npz') as old:
            labels=list(map(str,old['labels']));n=len(labels)
            cols=[n+labels.index(f'thr_{a:.3f}_{b:.3f}') for a,b in zip(lo,hi)]
            ps=old['Y_theta'][:,n+labels.index('pwin')]
            expected=correct(old['Y_theta'][:,cols],ps,calc.rate,lo,hi,calc.window)
            actual=z['pure_draws_'+g][:,7:27]
            difference=float(np.max(np.abs(expected-actual)/np.maximum(np.abs(expected),1e-12)))
            checks[g+'_all_2000_thrust_relative_difference']=difference
            assert difference<3e-6,difference
    head=load_head(HERE/'release/models/MIX17_head.npz');ref=Reference(HERE/'release/reference_MIX17.npz',head,calc)
    theta=axes.default('MIX17');rng=np.random.default_rng(92)
    with np.load(HERE/'release/models/MIX17_wifi.npz') as wifi:
        published=head._w_published(wifi)
    direction=rng.normal(size=17);direction[0]=0;direction/=np.linalg.norm(direction)
    eps=2e-5;indices=[0,1,2,5,24]
    for g,xi in (('sherpa',0.),('herwig',1.)):
        th=theta.copy();th[-1]=xi;basis=head.wifi_basis(ref.A,th,False);base=basis@published;values=[]
        for sign in (-1,1):
            lw,_,info=calc.impose(base+sign*eps*(basis@direction),ref.inwin,ref.F)
            assert info['converged'];values.append(ref.predictions(np.exp(lw))[indices])
        numeric=(values[1]-values[0])/(2*eps)
        analytic=z[g+'_J_omega'][indices]@direction
        difference=float(np.max(np.abs(numeric-analytic)))
        checks[g+'_full_reference_directional_derivative_max_absolute_difference']=difference
        assert np.allclose(numeric,analytic,rtol=3e-5,atol=3e-9),(g,numeric,analytic)
    checks.update(passed=True,number_of_reference_events=ref.N,parameter_points=2000,
                  fixed_rate_non_theory_response='Analytically zero; explicitly checked by the finite differences.')
    args.output.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
