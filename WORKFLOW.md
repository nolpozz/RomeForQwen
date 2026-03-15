# ROME for Qwen2.5-7B: Two-Phase Evaluation Workflow

This document describes the full **two-phase evaluation pipeline** for [ROME](https://github.com/nicola-decao/KnowledgeEditor) (Rank-One Model Editing) on **Qwen/Qwen2.5-7B** using [EasyEdit](https://github.com/zjunlp/EasyEdit) on [Modal](https://modal.com) (serverless A100 GPUs). The design ensures **reproducibility**, **no dataset drift**, and **research integrity** (strict train/holdout split, objective hyperparameter selection).

---

## Overview

| Phase | Script | Purpose |
|-------|--------|---------|
| **1** | `run_rome_grid.py` | Hyperparameter grid search on 150 seeded tuning samples; outputs `tuning_indices_used.json` and `rome_tuning_results.csv` (with **composite score** for objective selection). |
| **2** | `run_rome_final_eval.py` | Definitive baseline evaluation on the **holdout set** (known indices minus tuning indices); outputs `rome_final_baseline_metrics.csv` with per-edit metrics, means, and **N counts** for each metric. |

Shared logic (seeds, dataset loading, metrics, composite score) lives in **`rome_utils.py`** so both phases stay consistent.

---

## Prerequisites

- **Python 3.10+** (for local entrypoints and `modal run`).
- **Modal:** `pip install modal` then `modal token new`.
- Run all commands from the **RomeForQwen** repo root.

---

## Reproducibility

- **Seed:** All randomness uses seed **16** (set in `rome_utils.set_seeds(16)`).
- **EasyEdit:** Pinned to commit `41937637c2171b9cf1f929c143231d45a79f7787` in both Modal scripts to prevent repo drift. Update the `EASYEDIT_COMMIT` constant when intentionally upgrading.
- **Data:** The same CounterFact JSON file must be used in both phases (see below). No HuggingFace fallback—dataset path is required to avoid order/length drift.

---

## Inputs (required)

### 1. `qwen_known_indices.json`

- **Location:** Repo root (`RomeForQwen/qwen_known_indices.json`).
- **Format:** JSON array of integers: indices into the CounterFact dataset that the model “knows.”
- **Constraint:** After deduplication and range checks, at least **150** indices must remain (for the Phase 1 sample).

Example:

```json
[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, ...]
```

### 2. CounterFact dataset (required)

- **Location:** One of:
  - `RomeForQwen/data/counterfact/counterfact-train.json`
  - `RomeForQwen/counterfact-train.json`
- **Format:** JSON **list** of CounterFact-style records (fields: `prompt`, `target_new`, `ground_truth`, `subject`, `rephrase_prompt`, `locality_prompt`, `locality_ground_truth`).
- **Important:** The same file is used in Phase 1 and Phase 2 so that indices in `qwen_known_indices.json` and `tuning_indices_used.json` refer to the same dataset. There is no HuggingFace fallback.

---

## Phase 1: Hyperparameter grid search

**Script:** `run_rome_grid.py`

### What it does

1. Calls `rome_utils.set_seeds(16)`.
2. Loads `qwen_known_indices.json` and resolves the CounterFact data path (errors if not found).
3. Deduplicates known indices and restricts to valid range; uses **seeded** `random.sample` to pick **150** tuning indices; these are returned and saved locally as `tuning_indices_used.json`.
4. Loads the dataset via `rome_utils.load_and_filter_dataset(tuning_indices_used, data_path)` and builds editor requests with `rome_utils.record_to_request`.
5. Runs a grid over **layers** ∈ {15, 20, 24, 27}, **v_lr** ∈ {5e-1, 1e-1, 5e-2}, **v_num_grad_steps** ∈ {20, 30, 40} (108 configs). For each config, ROME uses `rewrite_module_tmp: "model.layers.{}.mlp.down_proj"` (Qwen SwiGLU).
6. Aggregates metrics with `rome_utils.extract_metrics()` (Efficacy, Generalization, Locality, and **n_efficacy**, **n_generalization**, **n_locality**) and computes the **composite score** with `rome_utils.calculate_composite_score()` (harmonic mean of the three metrics).

### Run

```bash
cd RomeForQwen
modal run run_rome_grid.py
```

### Outputs (repo root)

| File | Description |
|------|-------------|
| **`tuning_indices_used.json`** | The 150 dataset indices used for tuning. **Required for Phase 2.** |
| **`rome_tuning_results.csv`** | One row per config: `layers`, `v_lr`, `v_num_grad_steps`, `Efficacy`, `Generalization`, `Locality`, `n_efficacy`, `n_generalization`, `n_locality`, **`composite_score`**. Use **composite_score** for an objective “best” config (e.g. row with max composite_score). |

---

## Phase 2: Definitive holdout evaluation

**Script:** `run_rome_final_eval.py`

### What it does

1. Calls `rome_utils.set_seeds(16)`.
2. Loads `qwen_known_indices.json` and **`tuning_indices_used.json`** (from Phase 1).
3. **Validation:** Asserts `set(tuning_indices).issubset(set(known_indices))` — integrity check that tuning indices are a subset of known indices.
4. Computes **holdout set:** `holdout_indices = sorted(set(known_indices) - set(tuning_indices))`.
5. Loads records with `rome_utils.load_and_filter_dataset(holdout_indices, data_path)` and runs ROME with the **best** hyperparameters (set at top of the remote function).
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
modal run run_rome_final_eval.py
```

**Requires:** `qwen_known_indices.json`, **`tuning_indices_used.json`** (from Phase 1), and the same CounterFact data file.

### Outputs (repo root)

| File | Description |
|------|-------------|
| **`rome_final_baseline_metrics.csv`** | Per-edit rows (`edit_index`, `original_index`, `Efficacy`, `Generalization`, `Locality`), then a **MEAN** row, then an **N** row whose cells are **n_efficacy**, **n_generalization**, **n_locality** (sample sizes for each metric). |

The local entrypoint also **prints** Efficacy, Generalization, and Locality with their N counts.

---

## Shared module: `rome_utils.py`

| Function | Description |
|----------|-------------|
| **`set_seeds(seed=16)`** | Sets `random`, `numpy`, and `torch` (including CUDA/cuDNN) for reproducibility. |
| **`load_and_filter_dataset(indices_to_keep, data_path)`** | Loads EasyEdit `CounterFactDataset(data_path)`, filters to `indices_to_keep` (valid range only), returns list of records in order of `indices_to_keep`. Prevents dataset drift by using a single canonical path. |
| **`extract_metrics(metrics_list)`** | Parses EasyEdit per-edit output. Returns mean **Efficacy**, **Generalization**, **Locality** and sample sizes **n_efficacy**, **n_generalization**, **n_locality** (some records lack paraphrase/locality prompts). |
| **`calculate_composite_score(efficacy, generalization, locality)`** | Harmonic mean `3 / (1/e + 1/g + 1/l)`. Used for objective hyperparameter selection. Returns `None` if any metric is missing or non-positive. |
| **`record_to_request(record)`** | Converts a CounterFact-style record to EasyEdit editor request format (including locality dict). |

---

## Workflow diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  RomeForQwen (repo root)                                         │
│  ├── qwen_known_indices.json   (required)                        │
│  ├── data/counterfact/counterfact-train.json  (required)         │
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
- **Reproducibility:** Seed 16; EasyEdit pinned to commit `41937637c2171b9cf1f929c143231d45a79f7787`; single CounterFact data path for both phases.
- **Integrity:** Phase 2 asserts tuning ⊆ known; holdout is strictly disjoint from tuning.
