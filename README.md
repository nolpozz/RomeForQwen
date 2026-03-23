# RomeForQwen

A **reproducible two-phase evaluation pipeline** for [ROME](https://github.com/nicola-decao/KnowledgeEditor) (Rank-One Model Editing) on **Qwen/Qwen2.5-7B**, using the [EasyEdit](https://github.com/zjunlp/EasyEdit) library and [Modal](https://modal.com) for serverless A100 GPU runs. Designed for research rigor: fixed seeds, pinned dependencies, strict train/holdout split, and objective hyperparameter selection via a composite score.

---

## What’s in this repo

| Item | Description |
|------|-------------|
| **`rome_utils.py`** | Shared utilities: `set_seeds(16)`, `load_and_filter_dataset()`, `extract_metrics()` (with N counts), `calculate_composite_score()` (harmonic mean), `record_to_request()`. |
| **`run_rome_grid.py`** | **Phase 1** — Modal script for ROME hyperparameter grid search on 150 tuning samples. Writes results to the Modal Volume `rome-results` (safe for detach/disconnect). |
| **`run_rome_final_eval.py`** | **Phase 2** — Modal script for definitive ROME baseline evaluation on the holdout set. Validates tuning ⊆ known; writes results to the same Volume. |
| **`WORKFLOW.md`** | Full workflow: inputs, Phase 1 & 2 steps, outputs, `rome_utils` API, and diagram. |
| **`README.md`** | This file — project overview and quick start. |

---

## Quick start

1. **Prerequisites**
   - Python 3.10+
   - [Modal](https://modal.com): `pip install modal` then `modal token new`

2. **Inputs** (in repo root)
   - **`qwen_known_indices.json`** — either a JSON array of integers or a JSON object containing `case_ids`. Integers are treated as CounterFact `case_id` values.
   - **No local CounterFact file required** — dataset is downloaded from Hugging Face `azhx/counterfact` (split `test`) at a pinned revision.

3. **Phase 1 — Grid search**
   ```bash
   cd RomeForQwen
   modal run --detach run_rome_grid.py
   ```
   Results are written to the Modal Volume `rome-results` (so you can close your laptop). Pull them locally with:
   ```bash
   modal run pull_rome_results.py
   ```
   Pick the row with the highest **`composite_score`** in `rome_tuning_results.csv`.

4. **Phase 2 — Holdout evaluation**
   - In `run_rome_final_eval.py`, set `BEST_LAYER`, `BEST_V_LR`, and `BEST_V_STEPS` at the top of the remote function to the chosen config.
   ```bash
   modal run --detach run_rome_final_eval.py
   ```
   Pull results locally with `modal run pull_rome_results.py`.

---

## Reproducibility

- **Random seed:** 16 (in `rome_utils.set_seeds(16)`).
- **EasyEdit:** Installed as PyPI package `easyeditor==0.0.1` in both Modal scripts.
- **Data:** CounterFact is downloaded from `azhx/counterfact` at a pinned dataset revision (`COUNTERFACT_REVISION`) in both scripts to prevent drift.

---

## Metrics and composite score (ROME paper alignment)

| Metric | Description | Source |
|--------|-------------|--------|
| **Efficacy** | Success rate on the rewrite | EasyEdit `rewrite_acc` |
| **Generalization** | Paraphrase accuracy across `paraphrase_prompts` | `rephrase_acc` |
| **Locality** | Neighborhood specificity across `neighborhood_prompts` | locality `*_acc` |
| **Portability** | One-hop inference (edited fact in related contexts) | portability `*_acc` when dataset provides it |

**Composite score** = harmonic mean of (Efficacy, Generalization, Locality), used in Phase 1 to select the best hyperparameters objectively.

**Note:** The `azhx/counterfact` dataset includes Efficacy, Generalization, and Locality. Portability requires one-hop prompts (e.g. `alternative_prompts`, `one_hop`); it is captured when present and written to the final eval outputs.

## Result artifacts (written to Modal Volume)

- **`tuning_indices_used.json`** — the 150 tuning case IDs used (required for Phase 2).
- **`rome_tuning_results.csv`** — per-config metrics including `composite_score` (and `Portability` when available).
- **`rome_grid_metadata.json`** — seed, dataset revision, grid definition, selection rule.
- **`rome_final_baseline_metrics.csv`** — per-edit Efficacy, Generalization, Locality, Portability + MEAN row with `composite_score` + N row.
- **`rome_final_eval_metadata.json`** — dataset revision, best hyperparameters used, and full summary (all metrics).

**Pull results anytime** with `modal run pull_rome_results.py` (from `RomeForQwen`).

---

## License and references

- ROME: [KnowledgeEditor](https://github.com/nicola-decao/KnowledgeEditor).
- EasyEdit: [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit).
- Model: [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B).

For full details, inputs, outputs, and the two-phase workflow, see **[WORKFLOW.md](WORKFLOW.md)**.
