# ROME for Qwen2.5-7B: Two-Phase Evaluation Workflow

This document describes the full **two-phase evaluation pipeline** for [ROME](https://github.com/nicola-decao/KnowledgeEditor) (Rank-One Model Editing) on **Qwen/Qwen2.5-7B** using [EasyEdit](https://github.com/zjunlp/EasyEdit) on [Modal](https://modal.com) (serverless A100 GPUs). The design ensures **reproducibility**, **no dataset drift**, and **research integrity** (strict train/holdout split, objective hyperparameter selection).

---

## Overview

| Phase | Script | Purpose |
|-------|--------|---------|
| **1** | `run_rome_grid.py` | Hyperparameter grid search on 150 seeded tuning samples; writes `tuning_indices_used.json` and `rome_tuning_results.csv` to a **cloud Volume** (persists if laptop disconnects). |
| **2** | `run_rome_final_eval.py` | Definitive baseline evaluation on the **holdout set**; reads tuning indices from the Volume, writes `rome_final_baseline_metrics.csv` to the Volume. |
| **Pull** | `pull_rome_results.py` | Downloads all result files from the cloud Volume to your local directory. Run anytime after Phase 1 or 2 completes (or if you disconnected). |

**Cloud storage:** Results are written to a Modal Volume (`rome-results`) so runs complete and persist even if your laptop disconnects. Pull results locally with `modal run pull_rome_results.py`.

Shared logic (seeds, dataset loading, metrics, composite score) lives in **`rome_utils.py`** so both phases stay consistent.

---

## Prerequisites

- **Python 3.10+** (for local entrypoints and `modal run`).
- **Modal:** `pip install modal` then `modal token new`.
- Run all commands from the **RomeForQwen** repo root.

---

## Reproducibility

- **Seed:** All randomness uses seed **16** (set in `rome_utils.set_seeds(16)`).
- **EasyEdit:** Both Modal scripts use `add_local_dir(EasyEdit)` to bundle the **local** EasyEdit source from the workspace. The workspace `EasyEdit/` must include the Generalization broadcast fix in `easyeditor/evaluate/evaluate.py`.
- **Data:** CounterFact is loaded from Hugging Face dataset **`azhx/counterfact`** with a pinned dataset revision (`COUNTERFACT_REVISION`) in both scripts to prevent dataset drift.
- **Generalization metric:** EasyEdit’s `evaluate.py` is patched to broadcast a single target to multiple rephrase prompts. Without this fix, `zip(prompts, "Antarctica")` yields character pairs and Generalization ≈ 0. Both Modal scripts verify the fix at startup and fail fast if it is missing.

---

## Inputs (required)

### 1. `qwen_known_indices.json`

- **Location:** Repo root (`RomeForQwen/qwen_known_indices.json`).
- **Format:** Either:
  - JSON array of integers, or
  - JSON object containing a `case_ids` list (supported; matches your uploaded file).
- **Semantics:** Integers are treated as **CounterFact `case_id` values** (preferred). If the dataset does not contain a `case_id` field, they are treated as positional indices.
- **Constraint:** After deduplication and range checks, at least **150** indices must remain (for the Phase 1 sample).

Example:

```json
[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, ...]
```

### 2. CounterFact dataset source (automatic)

- **Dataset:** `azhx/counterfact` (Hugging Face Datasets).
- **Split:** `test` (21,919 rows).
- **Pinned revision:** `COUNTERFACT_REVISION = c01c413f856ee38f5c080c9fc5e87aff478e2ff9` in both scripts.
- **Requirement:** The Modal container must be able to reach Hugging Face to download dataset artifacts.

---

## Phase 1: Hyperparameter grid search

**Script:** `run_rome_grid.py`

### What it does

1. Calls `rome_utils.set_seeds(16)`.
2. Loads `qwen_known_indices.json` (supports list or dict-with-`case_ids`).
3. Downloads `azhx/counterfact` (split `test`) at pinned `COUNTERFACT_REVISION`.
4. Deduplicates and validates known IDs against the dataset; uses **seeded** `random.sample` to pick **150** tuning IDs; saves them locally as `tuning_indices_used.json`.
5. Filters the downloaded dataset via `rome_utils.load_and_filter_dataset(tuning_indices_used, dataset_records=...)` and builds editor requests with `rome_utils.record_to_request` (uses **all** `paraphrase_prompts` and **all** `neighborhood_prompts` per record for more robust Generalization/Locality).
5. Runs a grid over **layers** ∈ {15, 20, 24, 27}, **v_lr** ∈ {5e-1, 1e-1, 5e-2}, **v_num_grad_steps** ∈ {20, 30, 40} (36 configs). For each config, ROME uses `rewrite_module_tmp: "model.layers.{}.mlp.down_proj"` (Qwen SwiGLU).
6. Aggregates metrics with `rome_utils.extract_metrics()` (Efficacy, Generalization, Locality, and **n_efficacy**, **n_generalization**, **n_locality**) and computes the **composite score** with `rome_utils.calculate_composite_score()` (harmonic mean of the three metrics).

### Run

```bash
cd RomeForQwen
modal run --detach run_rome_grid.py
```

### Outputs (cloud Volume → pull locally)

| File | Description |
|------|-------------|
| **`tuning_indices_used.json`** | The 150 dataset indices used for tuning. **Required for Phase 2.** |
| **`rome_tuning_results.csv`** | One row per config: `layers`, `v_lr`, `v_num_grad_steps`, `Efficacy`, `Generalization`, `Locality`, `n_efficacy`, `n_generalization`, `n_locality`, **`composite_score`**. Use **composite_score** for an objective “best” config. |
| **`rome_grid_metadata.json`** | Seed, dataset revision, grid definition, and the selection rule used to pick the best config. |

---

## Phase 2: Definitive holdout evaluation

**Script:** `run_rome_final_eval.py`

### What it does

1. Calls `rome_utils.set_seeds(16)`.
2. Loads `qwen_known_indices.json` and **`tuning_indices_used.json`** from the cloud Volume (written by Phase 1).
3. **Validation:** Asserts `set(tuning_indices).issubset(set(known_indices))` — integrity check that tuning indices are a subset of known indices.
4. Computes **holdout set:** `holdout_indices = sorted(set(known_indices) - set(tuning_indices))`.
5. Downloads `azhx/counterfact` (split `test`) at pinned `COUNTERFACT_REVISION`.
6. Loads records with `rome_utils.load_and_filter_dataset(holdout_indices, dataset_records=...)` and runs ROME with the **best** hyperparameters (set at top of the remote function).
6. Aggregates with `rome_utils.extract_metrics()`; returns per-edit metrics and a summary including **n_efficacy**, **n_generalization**, **n_locality**.

### Set best hyperparameters

Before running, set the winning config from Phase 1. In `run_rome_final_eval.py`, at the **top of the remote function** `run_rome_final_eval`, edit:

```python
BEST_LAYER = 20        # from rome_tuning_results.csv
BEST_V_LR = 5e-1
BEST_V_STEPS = 30
```

(Choose the row with highest **composite_score** in `rome_tuning_results.csv`.)

### Run

```bash
cd RomeForQwen
modal run --detach run_rome_final_eval.py
```

**Requires:** `qwen_known_indices.json` (local), and **`tuning_indices_used.json`** in the cloud Volume (from Phase 1).

### Outputs (cloud Volume → pull locally)

Results are written to the same Modal Volume. Pull with `modal run pull_rome_results.py`.

| File | Description |
|------|-------------|
| **`rome_final_baseline_metrics.csv`** | Per-edit rows (`edit_index`, `original_index`, `Efficacy`, `Generalization`, `Locality`), then a **MEAN** row, then an **N** row whose cells are **n_efficacy**, **n_generalization**, **n_locality** (sample sizes for each metric). |
| **`rome_final_eval_metadata.json`** | Dataset revision, best hyperparameters used, split sizes (known/tuning/holdout), and a summary of mean metrics + N counts. |

The local entrypoint also **prints** Efficacy, Generalization, and Locality with their N counts.

---

## Pull results from cloud

**Script:** `pull_rome_results.py`

If your laptop disconnected during a run, or you want to download results to a different machine:

```bash
cd RomeForQwen
modal run pull_rome_results.py
```

This fetches all result files from the `rome-results` Volume and writes them to the current directory.

---

## Shared module: `rome_utils.py`

| Function | Description |
|----------|-------------|
| **`set_seeds(seed=16)`** | Sets `random`, `numpy`, and `torch` (including CUDA/cuDNN) for reproducibility. |
| **`download_counterfact_dataset(split=\"test\", revision=...)`** | Downloads `azhx/counterfact` at a pinned revision and returns a list of records. |
| **`load_and_filter_dataset(indices_to_keep, dataset_records=..., id_field=\"case_id\")`** | Filters records to match provided IDs (`case_id`) or positional indices if no ID field exists; returns in the order of `indices_to_keep`. |
| **`extract_metrics(metrics_list)`** | Parses EasyEdit per-edit output. Returns mean **Efficacy**, **Generalization**, **Locality** and sample sizes **n_efficacy**, **n_generalization**, **n_locality** (some records lack paraphrase/locality prompts). |
| **`calculate_composite_score(efficacy, generalization, locality)`** | Harmonic mean `3 / (1/e + 1/g + 1/l)`. Used for objective hyperparameter selection. Returns `None` if any metric is missing or non-positive. |
| **`record_to_request(record)`** | Converts a CounterFact-style record to EasyEdit editor request format (including locality dict). |
| **`load_indices_file(path)`** | Loads indices from either a list-of-ints JSON or a dict containing `case_ids`. |

---

## Workflow diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  RomeForQwen (repo root)                                         │
│  ├── qwen_known_indices.json   (required)                        │
│  ├── (no local CounterFact file required; downloaded via HF)     │
│  ├── rome_utils.py              (shared logic)                    │
│  ├── run_rome_grid.py           (Phase 1)                          │
│  ├── run_rome_final_eval.py     (Phase 2)                          │
│  ├── README.md                                                   │
│  └── WORKFLOW.md (this file)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │  Phase 1: modal run run_rome_grid.py      │
        ▼                                             │
┌───────────────────────────────────────┐            │
│  Modal (A100, seed 16, pinned EasyEdit)│            │
│  • known_indices → sample 150          │            │
│  • load_and_filter_dataset(tuning)     │            │
│  • 108 configs → extract_metrics +    │            │
│    composite_score                     │            │
└───────────────────────────────────────┘            │
        │                                             │
        │  tuning_indices_used.json                    │
        │  rome_tuning_results.csv (incl. composite)  │
        ▼                                             │
┌───────────────────────────────────────┐            │
│  Set BEST_LAYER, BEST_V_LR, BEST_V_   │            │
│  STEPS in run_rome_final_eval.py      │            │
└───────────────────────────────────────┘            │
        │                                             │
        │  Phase 2: modal run run_rome_final_eval.py  │
        ▼                                             │
┌───────────────────────────────────────┐            │
│  Modal (A100, seed 16, same image)     │            │
│  • Assert tuning ⊆ known              │            │
│  • holdout = known \ tuning            │            │
│  • load_and_filter_dataset(holdout)   │            │
│  • ROME(best) → extract_metrics        │            │
└───────────────────────────────────────┘            │
        │                                             │
        │  per_edit + summary (with N counts)         │
        ▼                                             │
┌───────────────────────────────────────┐            │
│  Local                                │            │
│  rome_final_baseline_metrics.csv      │            │
│  (per-edit + MEAN + N row)            │            │
└───────────────────────────────────────┘
```

---

## Summary

- **Phase 1:** Grid search on 150 tuning samples → `tuning_indices_used.json` + `rome_tuning_results.csv` (use **composite_score** to pick best).
- **Phase 2:** Holdout evaluation with best params → `rome_final_baseline_metrics.csv` (with **N** row for metric sample sizes).
- **Reproducibility:** Seed 16; EasyEdit pinned to commit `41937637c2171b9cf1f929c143231d45a79f7787`; CounterFact pinned to HF revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- **Integrity:** Phase 2 asserts tuning ⊆ known; holdout is strictly disjoint from tuning.
