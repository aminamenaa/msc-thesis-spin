# Iter-SPIN testing (thesis Contribution 2)

Iterative-refinement variant of SPIN: re-identify coupled weights on the **progressively suppressed**
model for `N_MAX` rounds, hard-zeroing per round. Training-free. `n=1` reduces to original SPIN.

## What's here

- **`SPIN_iter_step1.ipynb`** — the only new notebook. Runs the iterative loop and dumps cumulative
  coupled-index snapshots `coupled_iter_n{1,2,3,5,10}.json`. GPU T4x2, ~15–20 min/round
  (~2.5–3 h for 10 rounds), checkpointed per round (resumable).

## Run order

1. **`SPIN_iter_step1.ipynb`** → produces `coupled_iter_n1.json … coupled_iter_n10.json`.
   - *Sanity:* `coupled_iter_n1.json` should match the faithful 7B `coupled_indices.json` (n=1 = SPIN).
2. **Reuse `../7B testing/SPIN_7B_step2_metrics.ipynb` unchanged**, once per snapshot
   (swap the input `coupled_indices.json` → `coupled_iter_n{k}.json`). This gives the
   **n-ablation curve**: HSIC (decoupling) and perplexity (capability) vs n. This is the core
   experiment — it tests the proposal's stated assumption that importance stays a valid coupling
   proxy on a partially-suppressed model.
3. **Reuse `../7B testing/SPIN_7B_step3_generate.ipynb` + `SPIN_7B_step4_judge.ipynb`** only for the
   **chosen best n** (generating/judging all 5 snapshots on full SALAD is 15–25 h and unnecessary).

## What to watch

- Does decoupling (HSIC↓) keep improving with n, or saturate?
- Does capability (perplexity) hold, or collapse at high n? No theoretical bound on cumulative
  capability loss — if it blows up, that is a **finding**, not a bug.
- New-weights-per-round should shrink as the set saturates (zeroed weights auto-drop, `|W|=0`).
