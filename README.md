# Beyond Hard Suppression

**Soft-scaling and iterative variants of SPIN for fairness–privacy decoupling in LLMs, with a clinical-domain evaluation**

Amina Menaa · MSc Data Science and AI, Middlesex University Dubai · supervised by
Dr. Ikram Ur Rehman.

Notebooks and results behind the poster. SPIN (Qian et al., 2025) zeroes a tiny
shared set of MLP weights to raise both fairness and privacy refusal rates without
retraining. This work asks whether scaling those weights beats zeroing them
(**softSPIN**), whether re-finding them over several rounds finds a better set
(**iterSPIN**), and whether any of it holds on clinical data.

**See `Poster.pdf` for the method, results and limitations.** This file covers
only how the repository is laid out and how to run it.

## Layout

```
notebooks/<variant>/<model>/step<N>.ipynb
results/<variant>/<model>/step<N>_<stage>/
```

Variants: `spin` (the reproduction), `soft`, `iter`, and the same three with a
`-healthcare` suffix for the clinical case study.
Models: `Qwen2-7B` (SPIN's own model), `Qwen2.5-3B`, `Llama-3.2-3B`, `Gemma-2-2B`.

| step | what it does | key outputs |
|---|---|---|
| 1 identification | scores importance on fairness, privacy and general-capability prompts, then takes the coupled set | `coupled_*.json`, `iter_summary_*.json` |
| 2 metrics | decoupling and capability measures per coupled set | `metrics_*.json` |
| 3 generation | SALAD-Bench answers; EquityMedQA and MedSafetyBench for the clinical chapters | `answers_*.csv` |
| 4 judge | MD-Judge safe-rate scoring with paired exact McNemar | `judge_*.json`, `labels_*.csv` |

Notebook suffixes: `_control` (the same-size top-k control cuts, global and
per-category), `_mmlu` (step 2 extended with MMLU), `_authorsample` (the
author-sample identification protocol — the one reported for the general-domain
7B reproduction).

Step-2 coverage is uneven by design: softSPIN and both clinical chapters add
normalised HSIC and linear CKA on top of raw HSIC, because rescaling weights
shrinks activations and raw HSIC falls whether or not dependence does.

## Reading the result filenames

Result files keep their provenance in the filename, `__` standing in for a path
separator:

```
Iter-SPIN__Qwen2-7B__results__metrics_batch_Qwen2-7B-Instruct.json
└ source area ┘ └ model ┘ └ subdir ┘ └───────── the file itself ─────────┘
```

`n1 … n10` is the iterSPIN round count, `a0.00 … a1.00` the softSPIN α,
`global` / `percat` the two control conditions.

Names containing `_superseded_` are earlier partial or split runs, kept for
completeness and not cited. `results/iter/Qwen2.5-0.5B/` is an early feasibility
screen, also not cited.

## Running these

Built for **Kaggle, GPU T4 × 2, internet enabled**. Each notebook installs its own
pins (`transformers==4.46.3`, `datasets==2.18.0`, `sentencepiece`) and clones
[DEAN](https://github.com/AI45Lab/DEAN), SPIN's released code, for the calibration
and evaluation data.

Steps 2–4 consume the coupled-index JSONs written by step 1, attached as Kaggle
dataset sources. Each notebook picks its model through a `MODEL_ID` line near the
top with exactly one line uncommented; the per-model extraction ratio `p = q` and
the tokenizer flag sit on that same line so they cannot drift apart. Longer
notebooks checkpoint per round, per layer group or per generation batch, so a
timeout resumes rather than restarts.

**≈138 hours** of T4 wall clock across the 53 runs. Several notebooks take a
parameter (`ONLY_N`, the α set, `SNAP`), so the same notebook legitimately takes
different times at different settings.

Checkpoint caches (`control_cache_*.pt`) are excluded as regenerable
intermediates. DEAN's data is not vendored; the notebooks clone it.
