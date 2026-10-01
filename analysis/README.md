# Analysis

Every figure in the thesis comes from one notebook reading one file.

```
master_metrics.csv          1,424 tidy rows — the only input
spin_master_figures.ipynb   builds all 15 figures from it
figures/                    the figures it produced (PDF + PNG)
build_master_db.py          rebuilds master_metrics.csv from results/
make_figures.py             the shared palette and rcParams the notebook uses
runtime/                    recovered wall-clock timings
```

The notebook is committed with its outputs, so the figures render on GitHub
without running anything.

## master_metrics.csv

Long format: one row per (variant, model, domain, arm, parameter, metric). `arm`
is the condition — baseline, spin, iter, global, percat, soft — and `param_type`
with `param_value` carry n for iterSPIN and α for softSPIN. Any number in the
thesis is one filter away.

Rebuild it from the raw run outputs with `python build_master_db.py`, which walks
`results/` and writes the table.

## figures/

| | figure | what it settles |
|---|---|---|
| F01 | fairness–privacy plane | the trade-off is model-specific, not method-specific |
| F02 | iteration dose–response | three models improve with n, one collapses |
| F03 | soft-α dose–response | benefit and cost both decay, cost faster |
| F04 | cost–benefit Pareto | where the method stops paying for itself |
| F05 | HSIC vs behaviour | whether representational decoupling predicts behaviour |
| F06 | per-round selector dynamics | why Qwen2.5-3B breaks |
| F07 | clinical vs general transfer | same model, both domains |
| F08 | clinical fairness by benchmark | where the gain actually sits |
| F09 | σ-robustness | the HSIC readings across kernel bandwidths |
| F10 | budget-matched arms | iterSPIN against same-size control cuts |
| F11 | coupling specificity | how much of the zeroed set is fairness-only |
| F12 | MMLU vs perplexity | two capability measures, one story |
| F13 | compute cost | what iteration costs |
| F14 | calibration-sample ablation | sensitivity to the 128-prompt draw |
| F15 | clinical privacy correction | the empties-as-safe sensitivity |

## runtime/

| file | holds |
|---|---|
| `thesis_runtime_table.csv` | variant, model, domain, step, condition, wall clock |
| `runtime_summary.csv` | one row per notebook, with completion verdicts |
| `runtime_rounds.csv` | 261 typed rounds — iterSPIN rounds, layer groups, α passes, generation passes |
| `runtime_phases.csv` | phase timings from the Kaggle log clock |
| `NOTEBOOK_INDEX.csv` | per notebook: the metrics it measures and the files it writes |
| `RUNTIME_PROVENANCE.md` | how the timings were recovered and the checks they passed |

≈138 hours of Kaggle T4 wall clock across 53 runs, recovered from kernel logs and
executed-notebook metadata rather than estimated. Runtimes are per-run
measurements: several notebooks take a parameter (`ONLY_N`, the α set, `SNAP`),
so the same notebook takes different times at different settings.
