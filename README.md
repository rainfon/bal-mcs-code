# Reproduction code

MATLAB scripts for the experiments. Run them from this folder. Each script sets its own seeds, sample sizes, and file names.

## Software

- MATLAB R2022a
- Optimization Toolbox (`linprog`)
- Parallel Computing Toolbox (`parpool`)
- A C compiler supported by `mex`, used once to build `har_chord_mex`

There is no Python or R dependency.

Compile the sampler on the machine that will run the scripts:

```matlab
cd('path/to/this/folder')
mex('-R2018a','har_chord_mex.c')
```

The compiled file is platform-specific (`har_chord_mex.mexa64` on Linux, `har_chord_mex.mexmaca64` on Apple silicon). Do not copy a compiled file from another operating system. `RQ3_Region_Diagnostics` and `RQ4_Xi_Sensitivity` continue with the MATLAB sampler if this file is absent. The other experiment scripts stop.

## How to run one script

Run one script at a time. Each script requests 16 parallel workers.

```matlab
cd('path/to/this/folder')
RQ1_Configuration_Selection
```

From a shell, with one BLAS thread per worker:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  matlab -batch "cd('path/to/this/folder'); RQ1_Configuration_Selection"
```

On a machine with fewer than 16 cores, open the script and lower `cfg.workers` before starting. A full run uses four chains, 100 burn-in transitions, and 200000 production transitions per chain.

## Scripts and outputs

| Script | What it writes |
|---|---|
| `RQ1_Configuration_Selection` | `RQ1_Configuration_Selection.tex` |
| `RQ2_Strategy_Comparison` | `RQ2_Strategy_Comparison.tex` |
| `RQ3_Region_Diagnostics` | `Figures/Figure3.pdf`, `RQ3_Region_Diagnostics.tex`, `RQ3_Region_Diagnostics.fig` |
| `RQ3_Query_Policy` | `RQ3_Query_Policy.tex` |
| `RQ3_Common_Test` | `Figures/Figure4.pdf`, `RQ3_Common_Test.tex`, `RQ3_Common_Test.fig` |
| `RQ4_Ns_Sensitivity` | `Figures/Figure5.pdf`, `RQ4_Ns_Sensitivity.tex`, `RQ4_Ns_Sensitivity.fig` |
| `RQ4_Xi_Sensitivity` | `Figures/Figure6.pdf`, `RQ4_Xi_Sensitivity.fig` |
| `RQ4_Trajectory_Count` | `Figures/Figure7.pdf`, `RQ4_Trajectory_Count.tex`, `RQ4_Trajectory_Count.fig` |
| `RQ4_EpsilonPhi` | `Figures/Figure8.pdf`, `RQ4_EpsilonPhi.tex`, `RQ4_EpsilonPhi.fig` |
| `RQ4_EtaSigma` | `Figures/Figure9.pdf`, `RQ4_EtaSigma.tex`, `RQ4_EtaSigma.fig` |
| `RQ4_Sampler_Configs` | `RQ4_Sampler_Configs.tex` |
| `RQ4_Diagnostics_Nine` | `RQ4_Diagnostics_Nine.tex` |
| `RQ4_Ns_Sensitivity_Appendix` | `RQ4_Ns_Sensitivity_Appendix.tex` |
| `Appendix_Pos_Tracking` | `Figures/FigureA1.pdf`, `Appendix_Pos_Tracking.tex`, `Appendix_Pos_Tracking.fig` |
| `Appendix_Feedback_Robustness` | `Figures/FigureA2.pdf`, `Appendix_Feedback_Robustness.tex`, `Appendix_Feedback_Robustness.fig` |

`RQ1_Configuration_Selection` builds the 27 configurations. `RQ2_Strategy_Comparison` runs AVG--TP--TC with Random, HCAI, HAIW, MS, SM, ES, LS, and Core-set. Batch selection, trajectory coverage, and greedy coverage are inside these scripts.

Figure PDFs go to `Figures/` inside this folder. Set the environment variable `BAL_FIGURE_DIR` to write them somewhere else. The workflow diagram (Figure 1) and the illustrative threshold figure (Figure 2) are not produced by these experiment scripts.

The `.tex` files are generated table fragments.

## Data and preprocessing

`Datasets/` contains the 13 CSV files read by the scripts: DBS, MPG, CPU, ESL, BCC, CEV, ERA, LEV, MMG, HS, NJN, NZ, and SI. Each file has criterion columns whose names start with `g` and one `Class` column. `readData`, inside each experiment script, reads those columns and rescales a criterion to \([0,1]\) when its observed range lies outside that interval. It does not reverse a criterion direction. `splitData` forms the reference set, the candidate pool, and the test set. There is no separate preprocessing program.

## Seeds and sampling

Seeds are assigned at the top of each script.

- Split, chain, and feedback seeds in RQ1, RQ2, RQ3, the \(\xi\), \(\varepsilon_{\Phi}\), and \(\eta_{\sigma}\) scripts, and the two appendix scripts: `20261009`, `271828`, and `314159`. `RQ3_Query_Policy` uses `cfg.seed = 20261009` for all three roles.
- `RQ4_Trajectory_Count` uses split seed `2026` and the same chain and feedback seeds.
- The hit-and-run screens and the retained-sample comparison use split seed `20261009` and chain seed `271828`.

Sampling uses four chains, 100 burn-in transitions, and 200000 production transitions. With \(N_s = 2000\), each chain retains 500 models at an interval of 400. The direction mixture is a Hessian-preconditioned coordinate direction with probability 0.80 and a Gaussian direction with probability 0.20.
