# Data

Everything the notebooks read, so a run can be reproduced without hunting for
sources. Laid out the way DEAN lays out `src/data/`.

```
data/general/    calibration and evaluation sets for the general-domain chapters
data/clinical/   the clinical case study
```

The notebooks clone [DEAN](https://github.com/AI45Lab/DEAN) at runtime and read
its `src/data/`; the `general/` files here are the same ones, vendored so the
inputs are pinned alongside the results rather than tracking whatever DEAN's main
branch holds later.

## general/

| file | role | source |
|---|---|---|
| `alpaca_cleaned_no_safety_train.csv` | general-capability calibration (128 sampled, seed 0) | Alpaca-cleaned, safety rows removed — as distributed by DEAN |
| `beaver_train330k_fairness_safe_1k.csv` | fairness calibration (128 sampled) | BeaverTails 330k, safe fairness split |
| `beaver_train330k_privacy_safe_1k.csv` | privacy calibration (128 sampled) | BeaverTails 330k, safe privacy split |
| `salad_fairness.csv` | fairness evaluation, n = 2,165 | SALAD-Bench (Li et al., 2024) |
| `salad_privacy.csv` | privacy evaluation, n = 657 | SALAD-Bench (Li et al., 2024) |

Step 1 draws its calibration samples with `datasets.shuffle(seed=0).select(...)`,
so the 128 prompts per axis are fixed and reproducible from these files.

## clinical/

| file | role | source |
|---|---|---|
| `equitymedqa_ratings.xlsx` | raw rating sheet the fairness side is built from | EquityMedQA (Pfohl et al., 2024, *Nature Medicine*) |
| `cal_fair_health.csv` | fairness calibration, derived | built by step 1 from the sheet above |
| `cal_priv_health.csv` | privacy calibration, derived | built by step 1 from MedSafetyBench |
| `cal_general_health.csv` | general-capability calibration, derived | built by step 1 |
| `eval_fair_health.csv` | fairness evaluation, n = 1,416 | derived |
| `eval_priv_health.csv` | privacy evaluation, n = 657 | derived |
| `clinical_privacy_corrected.csv` | correction applied to the clinical privacy set | this work |

The privacy side draws on [MedSafetyBench](https://github.com/AI4LIFE-GROUP/med-safety-bench)
(Han et al., 2024; MIT). Its repository is not vendored here — the derived
`cal_priv_health.csv` and `eval_priv_health.csv` are what the notebooks consume.

Capability is measured against WikiText-2, MMLU and MedMCQA, all fetched at
runtime from Hugging Face rather than stored here.

## Licensing

These are third-party research datasets, redistributed here for reproducibility
of a student thesis and subject to their original terms, not to this
repository's licence. Each is attributed above; cite the original authors, not
this repository, if you use the data itself.
