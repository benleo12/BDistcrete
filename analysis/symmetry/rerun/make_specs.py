#!/usr/bin/env python3
"""The scorings of the rerun, one line each, for score_p2_array_lmx.sbatch.

Every model of the paper's scoring side is scored again with the code that scored it, on the
photon-free observables (D4), with its temperature refitted on the standard test (D11). A line is
KEY=VALUE pairs (no spaces in a value), read by score_p2_lmx.sbatch:

  OUT       the result file, under output/, the name of the published result it replaces
  STAGE NET SEED K ENS   as for score_p2.sbatch (K the rank of the factorized head)
  MODEL     the network (output/models/..., a link to the trained file)
  DATA      the data folder: RUN/data_* or RUN/../gen/data_*, links to the photon-free (_nb) folders
  TMODE     refit (the standard test: fit the temperature again, by the training run's procedure,
            and write it to output/temperature/<model>.json) or table (read it from there)
  TRAINED   the published standard-test result of the training, whose record (selected step, stop
            trajectory) a standard test keeps (output/published/<name>, a link)
  ABC=1     the densified design of Stages A to C (LADDER_ABC_DESIGN=RUN/../gen/designs)
  E_CSV F_CSV  the designs of the single-generator stages
  STAGEW_CSV REF_RUNS NREF_TOTAL SUFFIX DUMP  the wide coupling scan (references and weight dumps)
  TRUNCATE=1   the truncation to leading singular components (SvdRank)
  LOGIT_CACHE  (DCTR networks) the folder of the log ratios kept between their scorings

Writes specs/std_fac.txt, std_cat.txt (TMODE=refit) and strict_fac.txt, strict_cat.txt, svd.txt
(TMODE=table, after the standard tests). Run from analysis/symmetry/rerun/:  python make_specs.py
"""
import os

R = '/mnt/beegfs/bassi/reweighting/lmx_runs'        # RUN's parent: gen/ and the run folder
GEN = f'{R}/gen'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'specs')
lines = {k: [] for k in ('std_fac', 'std_cat', 'strict_fac', 'strict_cat', 'svd')}


def model(stage_tag, net):
    return f'output/models/{stage_tag}_{"concat.pt" if net == "cat" else "trunk.pt"}'


def add(group, **kw):
    # a DCTR network's log ratio at the held-out points is kept on disk by its standard test and read
    # by its strict test and SvdRank (the same reference and points; LADDER_LOGIT_CACHE in r2_ladder.py)
    if kw.get('NET') == 'cat':
        kw['LOGIT_CACHE'] = 'output/logitcache'
    lines[group].append(' '.join(f'{k}={v}' for k, v in kw.items() if v not in (None, '')))


# ---- Tables 1 and 2: Stages A to C on the densified designs (DESIGN_TAG augfull, and the five
# trainings of DESIGN_TAG aug the table reads: B factorized seeds 0-2, C seed 2 of both networks)
for S in 'ABC':
    for net in ('fac', 'cat'):
        for k in range(3):
            aug = (S == 'B' and net == 'fac') or (S == 'C' and k == 2)
            dt, pre = ('aug', 'p2aug') if aug else ('augfull', 'p2augfull')
            m = model(f'{S}_P2{dt}{net}{k}', net)
            std = f'{pre}_{S}_{net}_s{k}.json'
            add(f'std_{net}', OUT=f'output/{std}', STAGE=S, NET=net, SEED=k, K=48, MODEL=m, DATA=f'{GEN}/data_abc',
                ABC=1, TMODE='refit', TRAINED=f'output/published/{std}')
            add(f'strict_{net}', OUT=f'output/{pre}strict_{S}_{net}_s{k}.json', STAGE=S, NET=net, SEED=k, K=48, MODEL=m,
                DATA='data_abc_h400', ABC=1, TMODE='table')
# ---- the Sherpa and Herwig stages (code names E and F) on the 720 runs (DESIGN_TAG v3full)
CSV = dict(E_CSV='stageE_design_v3all.csv', F_CSV='stageF_design_v3all.csv')
for S in 'EF':
    for net in ('fac', 'cat'):
        for k in range(3):
            m = model(f'{S}_P2v3full{net}{k}', net); std = f'p2v3full_{S}_{net}_s{k}.json'
            add(f'std_{net}', OUT=f'output/{std}', STAGE=S, NET=net, SEED=k, K=48, MODEL=m, DATA='data_stagePURE17_v3',
                TMODE='refit', TRAINED=f'output/published/{std}', **CSV)
            add(f'strict_{net}', OUT=f'output/p2v3fullstrict_{S}_{net}_s{k}.json', STAGE=S, NET=net, SEED=k, K=48, MODEL=m,
                DATA='data_stagePURE17_v3_h400', TMODE='table', **CSV)
# ---- the mixture: the DCTR network (DESIGN_TAG v3full) and the factorized MIXGEOV3S9{,s1,s2}, whose
# standard test is the training result stage_mixture_<tag>.json; its strict test reads the links
# MIXP2v3fullfac<k> of the same networks (the temperature table resolves both names to one model)
for k in range(3):
    m = model(f'MIXP2v3fullcat{k}', 'cat'); std = f'p2v3full_MIX_cat_s{k}.json'
    add('std_cat', OUT=f'output/{std}', STAGE='MIX', NET='cat', SEED=k, K=48, MODEL=m, DATA='data_stagePURE17_v3',
        TMODE='refit', TRAINED=f'output/published/{std}')
    add('strict_cat', OUT=f'output/p2v3fullstrict_MIX_cat_s{k}.json', STAGE='MIX', NET='cat', SEED=k, K=48, MODEL=m,
        DATA='data_stagePURE17_v3_h400', TMODE='table')
    tag = 'MIXGEOV3S9' + ('' if k == 0 else f's{k}'); std = f'stage_mixture_{tag}.json'
    add('std_fac', OUT=f'output/{std}', STAGE='MIX', NET='fac', SEED=k, K=48, MODEL=model(tag, 'fac'),
        DATA='data_stagePURE17_v3', TMODE='refit', TRAINED=f'output/published/{std}')
    add('strict_fac', OUT=f'output/p2v3fullstrict_MIX_fac_s{k}.json', STAGE='MIX', NET='fac', SEED=k, K=48,
        MODEL=model(f'MIXP2v3fullfac{k}', 'fac'), DATA='data_stagePURE17_v3_h400', TMODE='table')
# ---- the original designs, the 208-run row (DESIGN_TAG origfull): E with stageE_design.csv, F with
# stageF_design_aug.csv on data_stagePURE17_orig, strict on data_stagePURE17_v3_h400 (their 8 held-out
# points); the mixture strict on data_stagePURE17_orig_h400, and at the 48 points of the 720-run set
OCSV = dict(E_CSV='stageE_design.csv', F_CSV='stageF_design_aug.csv')
for S in ('E', 'F', 'MIX'):
    for net in ('fac', 'cat'):
        for k in range(3):
            m = model(f'{S}_P2origfull{net}{k}' if S != 'MIX' else f'MIXP2origfull{net}{k}', net)
            std = f'p2origfull_{S}_{net}_s{k}.json'
            csv = OCSV if S != 'MIX' else {}
            add(f'std_{net}', OUT=f'output/{std}', STAGE=S, NET=net, SEED=k, K=48, MODEL=m, DATA='data_stagePURE17_orig',
                TMODE='refit', TRAINED=f'output/published/{std}', **csv)
            add(f'strict_{net}', OUT=f'output/p2origfullstrict_{S}_{net}_s{k}.json', STAGE=S, NET=net, SEED=k, K=48, MODEL=m,
                DATA='data_stagePURE17_v3_h400' if S != 'MIX' else 'data_stagePURE17_orig_h400', TMODE='table', **csv)
            if S == 'MIX' and net == 'fac':
                add('strict_fac', OUT=f'output/p2origfull48_MIX_fac_s{k}.json', STAGE=S, NET=net, SEED=k, K=48, MODEL=m,
                    DATA='data_stagePURE17_orig48', TMODE='table')
                add('strict_fac', OUT=f'output/p2origfull48strict_MIX_fac_s{k}.json', STAGE=S, NET=net, SEED=k, K=48, MODEL=m,
                    DATA='data_stagePURE17_orig48_h400', TMODE='table')
# ---- the K scan of Fig. 6 (factorized, seed 0); K = 48 is the Table 1 cell above
for S in 'ABC':
    for K in (1, 2, 3, 4, 6, 8, 16):
        m = model(f'{S}_P2augK{K}fac0', 'fac'); std = f'p2augK{K}_{S}_fac_s0.json'
        add('std_fac', OUT=f'output/{std}', STAGE=S, NET='fac', SEED=0, K=K, MODEL=m, DATA=f'{GEN}/data_abc', ABC=1,
            TMODE='refit', TRAINED=f'output/published/{std}')
        add('strict_fac', OUT=f'output/p2augK{K}strict_{S}_fac_s0.json', STAGE=S, NET='fac', SEED=0, K=K, MODEL=m,
            DATA='data_abc_h400', ABC=1, TMODE='table')
for K in (1, 2, 4, 6, 8, 9, 10, 12, 16, 24, 32, 45):
    m = model(f'E_P2v3K{K}fac0', 'fac'); std = f'p2v3K{K}_E_fac_s0.json'
    add('std_fac', OUT=f'output/{std}', STAGE='E', NET='fac', SEED=0, K=K, MODEL=m, DATA='data_stagePURE17_v3',
        TMODE='refit', TRAINED=f'output/published/{std}', E_CSV='stageE_design_v3all.csv')
    add('strict_fac', OUT=f'output/p2v3K{K}strict_E_fac_s0.json', STAGE='E', NET='fac', SEED=0, K=K, MODEL=m,
        DATA='data_stagePURE17_v3_h400', TMODE='table', E_CSV='stageE_design_v3all.csv')
# ---- the wide coupling scan (DESIGN_TAG wb18): three references of 63000 events each
WB = [('single', 'stageW_design.csv', '9010'), ('pool', 'stageW_design.csv', '')] + \
     [(f'local{r}', f'stageW_local_{r}.csv', '') for r in range(9100, 9106)]
for name, csv, refruns in WB:
    std = f'p2wb18{name}_W_fac_s0.json'
    add('std_fac', OUT=f'output/{std}', STAGE='W', NET='fac', SEED=0, K=48, MODEL=model(f'W_P2wb18{name}fac0', 'fac'),
        DATA=f'{GEN}/data_widebox', STAGEW_CSV=f'{GEN}/designs/{csv}', REF_RUNS=refruns, NREF_TOTAL=63000,
        SUFFIX=f'_P2wb18{name}fac0', DUMP=1, TMODE='refit', TRAINED=f'output/published/{std}')
# ---- the learning curve: the mixtures whose training data are still there are scored as trained
# (their temperature refit); the others (E160, E320, V4S9, V4L) are refitted by recalibrate_T.py
for tag, data, ens in (('MIXGEOV3S27F', 'data_stagePURE17_v3', 4), ('MIXGEOV3M8S9', 'data_stagePURE17_v3', 8),
                       ('MIXGEOV3E20S9', 'data_stagePURE17_v3ev20', 4)):
    std = f'stage_mixture_{tag}.json'
    add('std_fac', OUT=f'output/{std}', STAGE='MIX', NET='fac', SEED=0, K=48, ENS=ens, MODEL=model(tag, 'fac'),
        DATA=data, TMODE='refit', TRAINED=f'output/published/{std}')
# ---- SvdRank: the first DCTR training of each model, truncated, on its standard test, at its refit
for S, m, data in (('A', 'A_P2augfullcat0', f'{GEN}/data_abc'), ('B', 'B_P2augfullcat0', f'{GEN}/data_abc'),
                   ('C', 'C_P2augfullcat0', f'{GEN}/data_abc'), ('E', 'E_P2v3fullcat0', 'data_stagePURE17_v3'),
                   ('MIX', 'MIXP2v3fullcat0', 'data_stagePURE17_v3')):
    add('svd', OUT=f'output/p2trunc_{S}_cat_s0.json', STAGE=S, NET='cat', SEED=0, K=48, MODEL=model(m, 'cat'), DATA=data,
        ABC=1 if S in 'ABC' else '', E_CSV='stageE_design_v3all.csv' if S == 'E' else '', TRUNCATE=1, TMODE='table')

os.makedirs(OUT, exist_ok=True)
for g, ls in lines.items():
    with open(os.path.join(OUT, f'{g}.txt'), 'w') as f:
        f.write(f'# {len(ls)} scorings, written by make_specs.py\n' + '\n'.join(ls) + '\n')
    print(f'{g}: {len(ls)} scorings')
