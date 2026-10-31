# Reproducing the paper

This file maps every figure, table and quoted number of draft r9 of the paper, which presents the
method as DCTR with a factorized classifier, to the command and the inputs that make it. Section,
figure, table and equation numbers are those of draft r9. The tag `v2.0.1` has this file for draft
r8. All commands run from `analysis/`, and the result files named below are in `analysis/output/`,
which is in git. **(data)** marks a step that needs the network exports or event samples of
`docs/DATA.md`, and **(GPU)** a training.

## Before you start

- Create the environment with `conda env create -f environment.yml`. The figures are typeset with
  LaTeX (`usetex`, with `amsmath` and `amssymb`), which the environment does not install.
- The scripts that read the LEP measurements (`thrust_chain.py`, `calc_bins.py`, and everything
  that imports `directlib.py`) look for them in a local Rivet installation. Set
  `export RIVET_REF=rivet_ref` to use the copies that ship in `analysis/rivet_ref/`.
- The batch scripts in `symmetry/` are those that ran on the Symmetry cluster at Perimeter. Set
  `export MAXENT_DIR=$PWD` in `analysis/` and submit from there with `sbatch --export=ALL`. Their
  logs go to `logs/`. Their `#SBATCH` lines name the partitions of Symmetry.
- The batch scripts set the training settings of the paper (four bootstrap members, a fifth of each
  run set aside, SiLU), and `K` and `STEPS` must be passed as in the commands below.
  `train_p2.sbatch` requires both, and `train_mixgeo.sbatch` defaults to 36000 steps.
  `r2_ladder.py` alone defaults to other settings.
- `kscan_submit.sh` and `widebox_submit.sh` change to `MAXENT_DIR`, by default the Symmetry copy
  `/mnt/beegfs/bassi/reweighting/maxent_code`, and submit the batch scripts by name from there, as
  they sit in the analysis folder on Symmetry. Elsewhere they need copies of `symmetry/*.sbatch` in
  `analysis/`.
- A rerun writes its result files and exports under the published names. Copy the shipped files
  first if you want to compare with them.

## Figures

`python figs_r8.py` draws Figs. 3 to 9 and 13 to 15 and writes `output/r8_<name>.pdf`.
`python plot_envelope.py output/envelope_test_MIXGEOV3S9 tune full` draws Figs. 10 to 12, and the
same figures for each box alone. The paper's `figs/r8_<name>.pdf` are copies of these files. With
names, `figs_r8.py` draws only those figures.

| figure | what it shows | drawn from (`output/`) | inputs made by | drawn by |
|---|---|---|---|---|
| 1 | overview of the method | | TikZ in `main.tex` | |
| 2 | the factorized network | | TikZ in `main.tex` | |
| 3 | the mixture at a held-out Sherpa and a held-out Herwig point | `fig_closure_mix.npz` | `sbatch --export=ALL symmetry/closure_mix.sbatch`, which runs `fig_closure_mix.py 7214 8211` on `MIXGEOV3S9` and `data_stagePURE17_v3` **(data)** | `python figs_r8.py closure_mix` |
| 4 | the toy model | `toy_final_K48_s0.npz`, `toy_final_K48_s0.json` | `sbatch --array=0-8 --export=ALL symmetry/toy_final.sbatch` (`toy_final.py`, which generates its own events) | `python figs_r8.py toy` |
| 5 | Stage C across its box | `valid_map_C_P2augfullfac0_report.json` | `VALID_MODEL=C_P2augfullfac0 VALID_DATA=../gen/data_abc VALID_BLIND_DATA=../gen/data_abc_blind VALID_HALF=report VALID_OUT=output/valid_map_C_P2augfullfac0_report.json python eval_valid.py` **(data)**. The dotted tick is the `mean` of `xp_null.json`, from `python xp_null.py ../gen/data_abc` **(data)** | `python figs_r8.py map`, which reads that mean for the dotted tick (`XP_NULL` overrides it) |
| 6 | the wide coupling scan | `widebox_wb18.json`, `wb/W_P2wb18*_w_9100.npy` | `bash widebox_submit.sh` **(data, GPU)**, which generates the runs into `gen/data_widebox` and trains the three references, then `python widebox_collect.py`. The paper's networks were trained with `STEPS=18000` and the tags `wb18single`, `wb18pool` and `wb18local<run>`, in place of the 324000 steps and the `wb` tags written in the script. The weight files are the ones the training writes into `output/models/` (`LADDER_DUMP_WEIGHTS=1`), copied into `output/wb/` | `python figs_r8.py widebox` |
| 7 | coverage of the learning band | `wifi_coverage_MIXGEOV3S9.json`, `..._MIXGEOV3S9s1.json`, `..._MIXGEOV3S9s2.json` | `TAG=<tag> sbatch --export=ALL symmetry/wifi_cov.sbatch` for each of the three tags **(data)** | `python figs_r8.py coverage` |
| 8 | the number of terms | `rank_scan_final.json` | `bash kscan_submit.sh` **(data, GPU)**, then `python kscan_collect.py`, which reads the result files of the scan and the K = 48 cells of Table 1 in `output/` | `python figs_r8.py kscan` |
| 9 | the cost at one parameter point | `bench_final_cpu_4.json`, `bench_final_cuda_12.json` | `sbatch symmetry/bench_final_cpu.sbatch` and `sbatch symmetry/bench_final_gpu.sbatch` (`bench_cost_final.py`) **(data)** | `python figs_r8.py cost` |
| 10 | thrust and charged multiplicity over 400 variations, against the linear covariance, without the calculation | `envelope_test_MIXGEOV3S9_tune.npz`, `envelope_test_MIXGEOV3S9_full.npz` | `symmetry/envelope_test.sbatch` with `BOX=tune` and `BOX=full`, as in "The comparison with 400 variations" below **(data)** | `python plot_envelope.py output/envelope_test_MIXGEOV3S9 tune full` |
| 11 | the correlations between the bins of Fig. 10 | as Fig. 10 | as Fig. 10 | as Fig. 10 |
| 12 | thrust over the same variations with the calculation imposed | as Fig. 10 | as Fig. 10 | as Fig. 10 |
| 13 | the thrust distribution and its three bands | `three_bands_MIXGEOV3S9.npz`, `three_bands_MIXGEOV3S9.json`, `theory_variations_MIX.json` | `TAG=MIXGEOV3S9 sbatch --export=ALL symmetry/three_bands.sbatch` and `sbatch --export=ALL symmetry/theory_variations.sbatch` **(data)** | `python figs_r8.py bands` |
| 14 | the bands of the seven means | `three_bands_MIXGEOV3S9.json` | as Fig. 13 | `python figs_r8.py obs` |
| 15 | the calculation against ALEPH and DELPHI | `calc_bins_nlo_point.json` | `THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python calc_bins.py` | `python figs_r8.py aleph` |

## Tables

| table | what it shows | from (`output/` unless stated) | made by |
|---|---|---|---|
| 1 | closure widths of every model, both networks, three sets of members | Stages A to C: `p2augfull_<S>_<net>_s<k>.json` and `p2augfullstrict_<S>_<net>_s<k>.json`, with the `p2aug` files where a `p2augfull` file is absent. Stages D and E: `p2v3full{,strict}_{E,F}_<net>_s<k>.json`. Mixture: `stage_mixture_MIXGEOV3S9{,s1,s2}.json` and `p2v3fullstrict_MIX_fac_s<k>.json` (factorized), `p2v3full{,strict}_MIX_cat_s<k>.json` (concatenation) | the trainings and strict tests of "Trained models" below **(data, GPU)**, then `make_numbers.py` (`\Acc{Aug,Dense}*`) |
| 2 | closure widths per observable, Stages D and E and the mixture | as Table 1 | `make_numbers.py` (`\ObsDense*`) |
| 3 | the seven means and their bands | `three_bands_MIXGEOV3S9.json`, `three_bands_MIXGEOV3S9.npz` | as Fig. 13, then `make_numbers.py` (`\Tb<observable>*`) |
| 4 | parameters of Stages A to C | the designs `gen/designs/stage{A,B,C}_design_aug.csv` (App. A.1) | typed in `main.tex` |
| 5 | parameters of Stages D and E | the designs `analysis/release/generators/stage{E,F}_design_v3all.csv` and the boxes in `analysis/release/STAGES.md` | typed in `main.tex` |
| 6 | loss and closure width of the mean of members and of the fitted basis weights | `wifi_MIXGEOV3S9{,s1,s2}_diag.json` | `wifi_diag.py`, run by `symmetry/wifi_cov.sbatch` **(data)**, then `make_numbers.py` (`\Wifi{Loss,Width}*`) |

## Quoted numbers

```bash
python phase2tools/make_numbers.py output
```

writes `phase2tools/numbers.tex`, 624 macros identical to the paper's `numbers.tex`, and
`phase2tools/numbers.json`, from the result files in `output/` and the held-out points of the
208-run set in `data_stagePURE17/meta.json`. Copy `numbers.tex` into the paper
source. `LC_DIR` can point the learning-curve and coverage files to another folder, and `ENV_DIR` the
files of the comparison with 400 variations. The macros
come in families:

| macros | where in the paper | read from (`output/`) | made by |
|---|---|---|---|
| `\Acc{Aug,Dense}*`, `\AccFinal*`, `\ObsDense*`, `\Mix{Sherpa,Herwig}Points*` | Tables 1 and 2, Secs. 3.4, 3.6 and 3.7 | the files of Table 1 | the trainings and strict tests below |
| `\LcRunsThreeStrictAll`, `\LcMembersStrictAll` | Sec. 3.1 | `widths_v2_<tag>{,h400}{,r415}.json` of the learning-curve trainings, `p2origfull*_MIX_fac_s<k>.json`, `p2origfull48*_MIX_fac_s<k>.json` | `symmetry/train_mixgeo.sbatch`, `strict_closure.sbatch`, `rescore_v4.sbatch`, `train_p2.sbatch` and `score_p2.sbatch` **(data, GPU)** |
| `\Calib*` | Sec. 3.2, two halves of one fresh run | `p2augfull_{A,B,C}_fac_s0.json` | the trainings of Stages A to C |
| `\Toy*` | Sec. 3.3 | `toy_final_K<K>_s<k>.json` | `symmetry/toy_final.sbatch` |
| `\Map*`, `\XpNullMean` | Sec. 3.5, Fig. 5 | `valid_map_C_P2augfullfac0_report.json`, `xp_null.json` | as Fig. 5 **(data)** |
| `\Wide*` | Sec. 3.5 | `p2wb18{single,pool,local9100,...,local9105}_W_fac_s0.json` | `widebox_submit.sh` with the settings of Fig. 6 **(data, GPU)** |
| `\Cov*`, `\CovEight*` | Secs. 3.6 and 6, Fig. 7 | `wifi_coverage_MIXGEOV3S9{,s1,s2}.json`, `wifi_coverage_MIXGEOV3M8S9.json` | `symmetry/wifi_cov.sbatch` **(data)** |
| `\SvdRank*` | Sec. 3.7 | `p2trunc_{A,B,C,E,MIX}_cat_s0.json` | `symmetry/score_p2.sbatch` with `LADDER_TRUNCATE=1` on the first concatenation training of each model **(data)** |
| `\Cost*` | Secs. 1, 3.7, 4.2, 4.3 and 6, Fig. 9 | `bench_final_cpu_4.json`, `bench_final_cuda_12.json` | as Fig. 9. Fig. 9 and `\Cost{Fac,Late,Cat}{Cpu,Gpu}` time the weights and one backward pass (the derivatives of one prediction), `\CostW*` the weights alone, and `\CostVar*` the weights alone at the 400 variations of Sec. 4.3. `\CostRestCpu` is the rest of a step of `bench_cost_final.py`, mostly the maximum-entropy weight and its derivatives, to one figure. The fit times `\CostFit*` of draft r8 (6000 steps, `COST_STEPS` in `make_numbers.py`) are written but not used |
| `\CalcAlphas`, `\CalcAlphaZero` | Sec. 5.2, Eq. (C.9) | `three_bands_MIXGEOV3S9.json` | as Fig. 13, from the fixed point of the calculation below |
| `\Tb*` | Secs. 1, 2.3, 5.2 to 5.5 and 6, Apps. C.2 and D.2, Table 3 | `three_bands_MIXGEOV3S9{,s1,s2}.{json,npz}`, `theory_variations_MIX.json`, `calc_vs_delphi_MIXGEOV3S9.json`, `band_checks_MIXGEOV3S9.json` | as Fig. 13, `TAG=<tag> sbatch --export=ALL symmetry/three_bands.sbatch` for `s1` and `s2`, `TAG=MIXGEOV3S9 sbatch --export=ALL symmetry/band_checks.sbatch` **(data)**, and `NPZ=output/three_bands_MIXGEOV3S9.npz python calc_vs_delphi.py MIXGEOV3S9` |
| `\Kappa*`, `\Single*`, `\TruncMaxPercent`, `\WifiNewton*`, `\Wifi{Loss,Width}*` | Sec. 3.6, App. A.2, Table 6 | the training results of Table 1, `single_members_C_P2augfullfac0.json`, `truncation_fraction.json`, `wifi_MIXGEOV3S9{,s1,s2}_{fit,diag}.json` | `python single_members.py C_P2augfullfac0 C ../gen/data_abc`, `python trunc_frac.py` and `symmetry/wifi_cov.sbatch` **(data)** |
| `\Env*` | Secs. 1, 4.3, 5.3 and 6 | `envelope_test_MIXGEOV3S9_<box>.{json,npz}` for the boxes `full`, `tune`, `tune_xifull`, `tune_sherpa` and `tune_herwig` | "The comparison with 400 variations" below **(data)** |

Some statements of App. A.2 rest on result files without a macro. The number of coefficients of
the event network comes from `python param_count.py` (`output/param_count.json`). The comparison
with the energyflow PFN
comes from `CROSSCHECK_DATA=../gen/data_widebox EF_PYTHON=<python of the bdistcrete-tf environment> bash run_crosscheck_seeds.sh`, which
trains on runs 9000 and 9010 of the wide coupling scan (`output/crosscheck_seeds/`,
`output/crosscheck_seeds_summary.json`) **(data)**. The explanation of Table 6 by the reference
events of the other generator rests on `python wifi_gensplit.py output/wifi_MIXGEOV3S9_basis.npz`
(`output/wifi_gensplit_MIXGEOV3S9.json`) **(data)**.

## Trained models

Every model of Table 1 is trained three times, the three sets of members, with seeds 0, 1 and 2
(`LADDER_SEED_OFFSET` 0, 1000 and 2000). Each training writes its result on the standard test, and
the strict test scores the saved model again on held-out runs with five times the statistics.

**Stages A to C.** The densified designs are `gen/designs/stage{A,B,C}_design_aug.csv` and the runs
are in `gen/data_abc`. The strict test reads `data_abc_h400`, made by `python make_abc_strict.py`
from `gen/data_abc` and the extensions in `gen/data_abc_ext/` **(data)**.

```bash
export MAXENT_DIR=$PWD K=48 STEPS=324000 DESIGN_TAG=augfull ABC_DESIGN=../gen/designs
export SPECS="A:fac:0 A:fac:1 A:fac:2 A:cat:0 A:cat:1 A:cat:2 B:fac:0 B:fac:1 B:fac:2 B:cat:0 B:cat:1 B:cat:2 C:fac:0 C:fac:1 C:fac:2 C:cat:0 C:cat:1 C:cat:2"
J=$(sbatch --parsable --array=0-17 --export=ALL symmetry/train_p2.sbatch)
sbatch --array=0-17 --dependency=aftercorr:$J --export=ALL symmetry/score_p2_array.sbatch
```

This writes `output/p2augfull{,strict}_<S>_<net>_s<k>.json` and the exports
`output/models/<S>_P2augfull<net><k>_*`. Five trainings of the published table (B factorized, seeds 0 to
2, and C factorized and C concatenation, seed 2) are read from the `p2aug` files of a run with
`DESIGN_TAG=aug`. That run had early stopping switched on but trained the full 324000 steps, and
`make_numbers.py` uses a `p2aug` file only when its training reached 324000 steps.

**Stages D and E, and the concatenation network of the mixture.** The designs are
`stageE_design_v3all.csv` and `stageF_design_v3all.csv` in `analysis/`, and the runs are those of
`data_stagePURE17_v3` **(data)**.

```bash
export MAXENT_DIR=$PWD K=48 STEPS=324000 DESIGN_TAG=v3full DM_DATA=data_stagePURE17_v3 \
       E_DATA=data_stagePURE17_v3 E_CSV=stageE_design_v3all.csv F_DATA=data_stagePURE17_v3 F_CSV=stageF_design_v3all.csv
export SPECS="E:fac:0 E:fac:1 E:fac:2 E:cat:0 E:cat:1 E:cat:2 F:fac:0 F:fac:1 F:fac:2 F:cat:0 F:cat:1 F:cat:2 MIX:cat:0 MIX:cat:1 MIX:cat:2"
J=$(sbatch --parsable --array=0-14 --export=ALL symmetry/train_p2.sbatch)
sbatch --array=0-14 --dependency=aftercorr:$J --export=ALL symmetry/score_p2_array.sbatch
```

The strict test reads `data_stagePURE17_v3_h400`. This writes `output/p2v3full{,strict}_*` and the
exports `output/models/{E,F}_P2v3full<net><k>_*` and `output/models/MIXP2v3fullcat<k>_*`.

**The factorized mixture.** `symmetry/train_mixgeo.sbatch` trains it, builds its reference bundle
(`rebuild_ref_lowmem.py`), scores it on the standard test (`ladder_widths_big.py`) and embeds the
basis of its learning band (`wifi_embed.py`). With `CHAIN=1` a training longer than one queue slot
continues in a second job.

```bash
export MAXENT_DIR=$PWD
for k in 0 1 2; do
  T=MIXGEOV3S9; [ $k = 0 ] || T=MIXGEOV3S9s$k
  A="ALL,TAG=$T,DATA=data_stagePURE17_v3,STEPS=324000,CHAIN=1,LADDER_SEED_OFFSET=$((1000*k))"
  J=$(sbatch --parsable --export=$A symmetry/train_mixgeo.sbatch)
  sbatch --dependency=afterany:$J --export=$A symmetry/train_mixgeo.sbatch
done
```

This writes `output/stage_mixture_<tag>.json` (the standard test of Table 1),
`output/widths_v2_<tag>.json`, `output/wifi_<tag>_basis.npz` and the exports
`output/models/<tag>_*`. The strict test of Table 1 is `score_p2_array.sbatch` with
`DESIGN_TAG=v3full` and `SPECS="MIX:fac:0 MIX:fac:1 MIX:fac:2"`, which reads
`output/models/MIXP2v3fullfac<k>_trunk.pt`. Make `MIXP2v3fullfac<k>_trunk.pt` and
`MIXP2v3fullfac<k>_cond.npz` links to the files of `MIXGEOV3S9`, `MIXGEOV3S9s1` and `MIXGEOV3S9s2`
first, as they are on Symmetry. `MMAP_BASE` sets where a training keeps the memory map of its
training set (see the header of `train_mixgeo.sbatch`).
`TAGS="MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2" sbatch --export=ALL symmetry/strict_closure.sbatch`
writes the same test as `output/widths_v2_<tag>h400.json`.

**The learning curve of Sec. 3.1.** The other mixtures are trained the same way: `MIXGEOV3E20S9`,
`MIXGEOV3E160S9` and `MIXGEOV3E320S9` (2 × 10^4 to 3.2 × 10^5 events per run), `MIXGEOV3S27F`
(972000 steps), `MIXGEOV4S9` and `MIXGEOV4L` (2768 runs, rescored against a reference of equal size
by `symmetry/rescore_v4.sbatch`) and `MIXGEOV3M8S9` (eight members, `ENS=8`). Their data sets are
built by `symmetry/build_dataset.sbatch` (`make_dataset_v3.py`). The row of 208 runs is
`train_p2.sbatch` with `DESIGN_TAG=origfull` on `data_stagePURE17_orig`. It is scored by
`score_p2.sbatch` at its own 16 held-out points, with the strict test on the data set of
`make_orig_h400.py`, and at the 48 held-out points of the 720-run set, on the data sets of
`make_orig48.py`.

**The data sets of the mixture.** `data_stagePURE17_v3` holds the 720 training runs and the 48
held-out points at 8 × 10^4 events, and `data_stagePURE17_v3_h400` the same with the held-out runs
at 4 × 10^5 events. Both are built from the generated runs by `make_dataset_v3.py`, which
`symmetry/build_dataset.sbatch` runs once for every entry of `SPECS`:

```bash
export MAXENT_DIR=$PWD SPECS="data_stagePURE17_v3:release,v3:0:0 data_stagePURE17_v3_h400:release,v3:0:4"
sbatch --export=ALL symmetry/build_dataset.sbatch
```

`make_dataset_v3.py` and `make_abc_strict.py` read and write under `MAXENT_DIR` and its parent, by
default the Symmetry copy.
The samples themselves come from `gen/` (`gen/README.md`).

## The calculation at its fixed point

```bash
export RIVET_REF=rivet_ref
THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python thrust_chain.py    # output/chain_summary_nlo.json, moments_joint_nlo.json
python targets_nlo_point.py                                # output/thrust_targets_nlo_point.npz
THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python calc_bins.py       # output/calc_bins_nlo_point.json, Fig. 15
```

The fixed point α_s(M_Z) = 0.1135, α_0 = 0.43 of Eq. (C.9) is the pair at which the χ² of the
calculation alone against ALEPH in the window is smallest. `thrust_chain.py` finds it and writes it
to `output/chain_summary_nlo.json`. `targets_nlo_point.py` evaluates the fourteen windowed moments,
the window fraction, every scale and scheme variation and the three variations of the dispersive
model at that point, and the covariance Σ_c of Eq. (5.6). These steps run from the repository
alone. The resummation tables of ARES (`ares_recovered/`) and the fixed-order coefficients of
EERAD3 (`output/eerad_*.npy`) ship with it. Rebuilding those needs ARES and EERAD3 themselves
(`ares_recovered/ares_edits.txt`, `eerad3/`).

## The comparison with 400 variations

`envelope_test.py` tests the linear covariance of Sec. 4.2 against 400 variations, each an exact
reweighting of the reference sample to one draw from the prior of Sec. 4.1, first without the
calculation (Sec. 4.3) and then with it imposed (Sec. 5.3). `BOX` sets the ranges: `full` (the
trained box), `tune` (each range a third of its width around the default tunes, and ξ in [1/3, 2/3]),
`tune_xifull` (the same with ξ over [0, 1]), and `tune_sherpa` and `tune_herwig` (one generator
alone, ξ fixed at 0 or 1). It reads the export, reference bundle and learning fit of `MIXGEOV3S9`
and the calculation of `output/thrust_targets_nlo_point.npz` **(data)**.

```bash
export MAXENT_DIR=$PWD TAG=MIXGEOV3S9
BOX=full NREF=1000 sbatch --export=ALL symmetry/envelope_test.sbatch
for b in tune tune_xifull tune_sherpa tune_herwig; do BOX=$b sbatch --export=ALL symmetry/envelope_test.sbatch; done
python plot_envelope.py output/envelope_test_MIXGEOV3S9 tune full    # Figs. 10 to 12
```

Each box takes about an hour and a half on eight CPU threads and needs 4 GB of memory. It writes
`output/envelope_test_MIXGEOV3S9_<box>.json`, with the bands, their ratios and bootstrap
uncertainties and the comparison of the correlations, and `.npz`, with the draws and the
covariances. On the full box, `NREF=1000` adds 1000 more draws, which measure the sampling noise
of 400 variations. `make_numbers.py` reads these files for the `\Env*` macros.

## The released networks

```bash
export MAXENT_DIR=$PWD
sbatch --export=ALL symmetry/package_r8.sbatch     # release_r8/: E, F and MIX17
sbatch --export=ALL symmetry/table_r8.sbatch       # release_r8_table/: STAGES.md, closure_*.csv, closure_summary.json
sbatch --export=ALL symmetry/accuracy_r8.sbatch    # release_r8_table/: accuracy_law.json, accuracy_law_points.csv
```

**(data)** The packager converts `E_P2v3fullfac0`, `F_P2v3fullfac0` and `MIXGEOV3S9` to the numpy
files of `analysis/release/models/` and checks them against the trained networks.
`table_r8.sbatch` also writes `output/precision_gap_r8.json`. The files are then copied into
`analysis/release/`. `release/models/MIX17_wifi.npz` is a copy of `output/wifi_MIXGEOV3S9_fit.npz`.

## Earlier draft with an α_s fit (not in the paper)

The earlier draft fitted α_s(M_Z) and α_0 to the LEP data with the generator parameters profiled, in
its own Sec. 6, on networks trained to an earlier standard and on the samples of the release
`data-v1`. Drafts r8 and r9 have none of this. Its scripts and result files stay in the repository,
and `python make_sec6_numbers.py` recomputes the numbers of that Sec. 6 from `output/`. The
walk-throughs of that draft, section by section with the archives each one needs, are in the git
history: `git show 1d66811:docs/REPRODUCE.md`. Two things of draft r8 go back to it. Its cost
estimate (`\CostFit*`) took its 6000 steps per fit from the forty pseudo-data fits of that draft,
which needed a median of 6315 evaluations of the χ² and its gradient per fit. Draft r9 quotes the
cost of the 400 variations instead. The seed study of that draft (`perlmutter/seedstudy.sbatch`) is
superseded by the three sets of members of every model in Table 1.
