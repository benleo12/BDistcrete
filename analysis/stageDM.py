#!/usr/bin/env python3
"""Stage D (mixture): seven continuous parameters,
    theta = (alpha_s^S, STRANGE_FRACTION, KT_0, AlphaIn^H, PwtSquark^H, ClMaxLight^H, f),
    q = (1-f) q_Sherpa(theta_S) + f q_Herwig(theta_H).
Data: make_stageDM_data.py (96 train x 25k, 10 held x 80k). Reference 96 x 3400 = 326k,
Stage C's size. Second-order rank count K2 = 1 + 7 + 28 = 36; LADDER_K should exceed it."""
import os, json
os.environ.setdefault('LADDER_SUFFIX', 'DM'); os.environ.setdefault('LADDER_K', '40')
os.environ.setdefault('LADDER_DEAD', 'report'); os.environ.setdefault('LADDER_REVIVE', '1')
import r2_ladder as R
meta = json.load(open(f"{os.environ.get('DM_DATA', 'data_stageDM')}/meta.json"))
train = {m['rid']: tuple(m['theta']) for m in meta['train']}; held = {m['rid']: tuple(m['theta']) for m in meta['held']}
R.STAGES['DM'] = dict(
    data=os.environ.get('DM_DATA', 'data_stageDM'), flavor=True, ntheta=7, train=train, held=held,
    norm=[(0.120, 0.008), (0.475, 0.175), (1.30, 0.50), (0.1186, 0.008), (0.325, 0.175), (3.80, 1.00), (0.5, 0.5)],
    obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'], flav_obs=['nbaryon', 'strange'])
R.NREF_PER = 3400
print(f"STAGE D MIXTURE: {len(train)} train x 25k, {len(held)} held x 80k, ntheta=7, K={R.K}, ref {len(train)*R.NREF_PER}, device {R.DEV}", flush=True)
result = R.run_stage('DM')
json.dump(result, open(os.environ.get('STAGED_OUT', 'output/stageDM.json'), 'w'), indent=1)
print('STAGE D MIXTURE DONE')
