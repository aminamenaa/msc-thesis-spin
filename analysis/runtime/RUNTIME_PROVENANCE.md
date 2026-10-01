# Runtime Provenance — SPIN notebooks

How the runtime figures in `thesis_runtime_table.csv` were obtained, why the
notebook files in the repo are not the source, and how the numbers were validated.

Generated 2026-09-30. Scripts: `fetch_kaggle_runtimes.sh`, `parse_kaggle_logs.py`.

## Summary

| | |
|---|---|
| Notebooks in repo | 53 |
| Kaggle slugs fetched | 52 (plus 1 notebook with no slug) |
| Runs with usable runtime | **53 of 53** |
| Aggregate wall clock recovered | **138.07 h** |
| Typed rounds extracted | 261 (129 with printed seconds) |

Totals by variant and domain:

| variant | domain | wall clock |
|---|---|---|
| iter | general | 59.95 h |
| iter | healthcare | 9.52 h |
| soft | general | 50.58 h |
| soft | healthcare | 10.12 h |
| spin | general | 4.25 h |
| spin | healthcare | 3.66 h |

## Why not the repo's .ipynb files

The notebooks were downloaded with `kaggle kernels pull`, which returns notebook
**source only** — with or without `-m`. The `-m` flag writes
`kernel-metadata.json`, holding configuration (title, slug, `enable_gpu`,
`enable_internet`, `dataset_sources`) and no timing. The Kaggle API exposes no
run-duration field at all: every `duration` in the CLI refers to API-token
expiry, and `lastRunTime` is a listing timestamp, not a length.

Consequently 40 of the 53 local notebooks carry no execution metadata whatsoever:
no `papermill` block, no cell `execution` timestamps, no `execution_count`, no
outputs. Their notebook-level metadata is exactly `['kernelspec', 'language_info']`
(plus `accelerator` on 8).

## Source 1 — Kaggle run logs (49 notebooks)

`kaggle kernels output` returns the run's artifacts, including the kernel log,
a JSON array of

```json
{"stream_name": "stdout"|"stderr", "time": <seconds since run start>, "data": "..."}
```

The final entry's `time` is the run's wall clock, and every printed line is
stamped on the same monotonic clock. This is better provenance than papermill
notebook metadata: it survives independently of the notebook file and gives
per-phase timing for every step type.

The log is returned in a separate response field from the output files, so
`--file-pattern` with a deliberately non-matching pattern fetches the log while
downloading none of the bulk — 3.6 MB for all 52 logs instead of roughly 2.7 GB
of checkpoints and cloned repositories.

## Source 2 — executed .ipynb files (4 notebooks)

Three notebooks had no usable log because their *latest* Kaggle version was never
run: two produced only `__results__.html` with no executed notebook written, and
one had a log containing pip output alone. A fourth had no Kaggle slug in the
fetch list at all, so no log was ever available for it. Executed copies supplied
all four directly, each carrying a genuine `papermill` block with all cells
`completed` and no exception:

| notebook | folder | wall clock | phase sum | run date |
|---|---|---|---|---|
| `iter-spin-step-1-7b-qwen` | Iter-Spin Work | 3:27:40 | 3:24:10 (10 rounds) | 2026-08-07 |
| `iter-spin-step-3-qwen-3b` | Iter-Qwen 3B | 11:01:36 | 11:00:05 (28 gen passes) | 2026-08-27 |
| `softspin-step-1-qwen-3b` | Soft-Qwen 3B | 0:10:16 | 0:08:47 (9 groups) | 2026-08-28 |
| `iter-spin-step-2-benchmark-included` | Iter-Spin Work | 1:45:58 | n/a (prints no timings) | 2026-09-29 |

The fourth notebook's papermill block breaks down as: 26.0 s pip, 196.1 s model +
data load, 670.5 s baseline (HSIC + perplexity + MMLU), 5418.3 s the eight-set
scoring loop, 41.0 s bootstrap CIs.

Archived at `_executed_notebooks/<folder>/<slug>.ipynb`. These are the only
carrier of those four runtimes — **do not strip their outputs.**

For the first three, the run dates differ from the unexecuted latest versions
whose logs were fetched, so those figures describe the earlier executed versions.

## Validation

Three independent checks, all passing.

**1. `coupling_rounds.csv` reproduced exactly.** Summing the per-round seconds
recovered from the logs reproduces all five entries of
`thesis_master_data/coupling_rounds.csv` **to the second**:

| model | domain | coupling_rounds | recovered phase sum | notebook total |
|---|---|---|---|---|
| gemma-2-2b-it | general | 1:19:32 | **1:19:32** | 1:21:13 |
| Llama-3.2-3B-Instruct | general | 1:25:25 | **1:25:25** | 1:27:22 |
| Qwen2.5-3B-Instruct | general | 1:57:28 | **1:57:28** | 1:59:28 |
| Qwen2-7B-Instruct | general | 3:24:10 | **3:24:10** | 3:27:40 |
| Qwen2-7B-Instruct | healthcare | 3:43:59 | **3:43:59** | 3:47:07 |

The notebook total exceeds the phase sum by 1.7–3.5 min in every case: pip
install plus model load, which `coupling_rounds.csv` excludes by construction.
This confirms `coupling_rounds.csv` was derived from these notebooks and that the
recovery is faithful.

**2. Printed timings vs the log's own clock.** Where a notebook prints its own
elapsed seconds via `time.time()`, those agree with Kaggle's independent
timestamp stream to within **0.55 s** at worst. The runtimes are not
self-reported.

**3. Stale papermill blocks in the repo's .ipynb files.** 13 repo notebooks
carried a `papermill` block whose `input_path` is `__notebook__.ipynb`, Kaggle's
name for the *executed* artifact. Those blocks are inherited fossils: identical
`start_time` values (to the microsecond) appear across three or four different
model folders, which cannot describe separate runs. The logs resolve every case:

- The `0:52:18` block replicated across all four Soft-SPIN step-2 notebooks
  belongs to **Soft-Qwen 3B alone**. Its log gives 3143.67 s to the
  executed-notebook write versus the block's 3138.5 s — a 5.17 s gap that is
  nbconvert overhead. The real times for the other three are Gemma 1:07:02,
  Llama 0:44:09, Qwen-7B 1:33:39, none of them 0:52:18.
- The Iter-SPIN step-2/3/4 blocks (`0:01:52` failed, `0:00:30` failed, `0:55:56`
  truncated) are from abandoned early attempts. Real runs: step-2 1:20:10–1:44:36,
  step-3 8:03:30–8:37:15, step-4 3:59:38–4:31:35.

**Do not cite any papermill figure from the repo's .ipynb files.** 11 of the 13
contradict the run logs, by up to 8.6 hours. The four notebooks under
`_executed_notebooks/` are the exception — their blocks are genuine, and are
corroborated by their own printed phase sums and, for the 7B step-1 case, by
`coupling_rounds.csv`.

## Coverage

All 53 notebooks have a runtime. No gaps remain.

`Iter-Spin Work/` holds two step-2 notebooks — the original (1:07:08) and the
MMLU-extended variant (1:45:58, `condition = main+mmlu` in the table). Report them
separately; they are different runs of different code, not a duplicate.

## Caveats for the write-up

- Runtimes are Kaggle wall clock, including package installation, model download
  and load, and roughly 3–6 s of nbconvert export. `compute_seconds` in the table
  excludes the export; `runtime_seconds` does not.
- All GPU runs report `accelerator: nvidiaTeslaT4`. The adjacent
  `isGpuEnabled: false` in Kaggle's metadata block is a known quirk of that field
  and does not mean the run was CPU-only.
- Wall clock is not compute cost: these runs share Kaggle's multi-tenant
  hardware, so identical work can vary by minutes between sessions.
- `iter-spin-step-3-qwen-3b` ran 11:01:36 in a single session, past the usual 9 h
  GPU allowance. It completed cleanly (`exception: None`, every cell `completed`).
- The variant/model/domain mapping in `thesis_runtime_table.csv` is derived from
  folder names, not from the notebooks' own contents. Verify before citing.

## A note on the two 7B step-2 notebooks

`Iter-Spin Work/` holds two step-2 notebooks. They are different code and
different runs, not a duplicate:

| notebook | runtime | MMLU in code | sets scored in its last run |
|---|---|---|---|
| `iter-spin-step-2` | 1:07:08 | no | 13 (iter + global + percat) |
| `iter-spin-step-2-benchmark-included` | 1:45:58 | yes | 8 (global + percat only) |

The 8-set run on 2026-09-29 had only the two control dataset sources attached, so
its discovery found no `coupled_iter_*` JSONs. Its raw `/kaggle/working` output is
therefore partial and **should not be used** — it adds nothing.

The integrated results are complete. `Iter-SPIN/Qwen2-7B/results/metrics_batch_Qwen2-7B-Instruct_withacc.json`
carries all 13 sets with MMLU (iter at n = 1, 2, 3, 5, 10; baseline MMLU 0.473,
iter 0.472 / 0.463 / 0.461 / 0.463 / 0.453) plus the full 52 bootstrap CI keys
(3 variants x 5 n x 4 sigmas). It agrees with the no-MMLU
`metrics_batch_Qwen2-7B-Instruct.json` on HSIC and perplexity for all 13 shared
sets with zero differences, so it is a strict superset of that run.

All four iter models therefore have MMLU parity: 13 sets each, `n_mmlu=1000`,
iter scored at every n. No re-run is needed.

## Files

| file | contents |
|---|---|
| `thesis_runtime_table.csv` | the citable table: variant, model, domain, step, condition, runtime |
| `runtime_summary.csv` | one row per notebook, with verdicts and fossil cross-check |
| `runtime_phases.csv` | phases from log-clock deltas |
| `runtime_rounds.csv` | 261 typed rounds (round, group, alpha, n-variant, gen pass) |
| `notebook_runtime_metadata.csv` | audit of what the repo's .ipynb files themselves carry |
| `_kaggle_logs/` | the 52 source logs (3.6 MB) |
| `_executed_notebooks/` | the 4 executed notebooks carrying their own timings |
| `_kaggle_fetch_report.tsv` | fetch status per slug — rows accumulate across runs |
