# Implementation Notes: ROME for Qwen2.5-7B

This document provides a comprehensive, verbose account of all implementation details, compatibility measures, and modifications applied to reproduce ROME (Rank-One Model Editing) baselines on **Qwen/Qwen2.5-7B** for research and paper submission. It is intended for reproducibility, reviewer transparency, and future maintenance.

---

## 1. Project Overview

### 1.1 Objective

We implement a two-phase evaluation pipeline for **ROME** on **Qwen2.5-7B**:

1. **Phase 1 (Grid search):** Hyperparameter tuning on 150 CounterFact samples (from a known-indices set) to select layer, learning rate, and gradient steps.
2. **Phase 2 (Final evaluation):** Definitive baseline evaluation on the holdout set (known indices minus tuning indices) using the best hyperparameters.

The pipeline runs on [Modal](https://modal.com) (serverless A100 GPUs) and uses the [EasyEdit](https://github.com/zjunlp/EasyEdit) implementation of ROME.

### 1.2 Components

| Component | Description |
|-----------|-------------|
| **ROME** | Rank-One Model Editing (Meng et al.); a knowledge-editing method that applies a low-rank update to specific MLP layers. |
| **EasyEdit** | Unified framework for model editing; we use its ROME implementation and `BaseEditor`. |
| **Model** | `Qwen/Qwen2.5-7B` (Hugging Face); decoder-only, 28 layers, SwiGLU MLP. |
| **Data** | CounterFact (`azhx/counterfact`), pinned revision. |

---

## 2. Compatibility Issues: Qwen2 and ROME

### 2.1 Symptoms

When ROME (via EasyEdit) runs a forward pass on Qwen2 models to extract layer activations, the following error can occur:

```
TypeError: unsupported operand type(s) for +: 'Tensor' and 'dict'
```

**Location:** `Qwen2DecoderLayer.forward`, at the residual-add line:

```python
hidden_states = residual + hidden_states
```

### 2.2 Root Cause: Nethook (Primary)

**Primary fix:** `EasyEdit/easyeditor/util/nethook.py`

ROME uses `nethook.Trace` to capture activations at specific layers (e.g. `model.layers.{L}.mlp.down_proj`). The Trace registers a PyTorch forward hook with `with_kwargs=True`. PyTorch then invokes the hook as `hook(module, args, kwargs, result)` — four arguments, with `result` being the actual output tensor. The original nethook treated the third argument (`kwargs`) as the output and returned it, replacing the real tensor with a dict. The fix correctly uses `result` as the output.

In some transformers versions, Qwen2 attention can also return structured objects; the `qwen2_rome_compat.py` patch unwraps those. The MLP does not need unwrapping.

### 2.3 Where the Issues Originated

- **Nethook bug:** In EasyEdit's `nethook.py`; fixed by correctly mapping PyTorch's 4-argument hook signature.
- **Attention unwrapping:** Optional safeguard in `qwen2_rome_compat.py` for attention return-type inconsistencies in some transformers versions.

### 2.4 Affected Versions

- `transformers` 4.40.2, 4.44.2, 4.57.1
- PyTorch 2.0+ (uses `with_kwargs=True` for hooks)

---

## 3. Qwen2–ROME Compatibility Patch

### 3.1 Design Principles

1. **Minimal intervention:** Only fix the decoder layer’s handling of the attention return; do not change ROME’s algorithm.
2. **Transparency:** Patch is isolated in a dedicated module, documented, and easy to disable.
3. **Robustness:** Handle dict-like and Cache-like returns by extracting the tensor when possible.

### 3.2 Implementation

**File:** `qwen2_rome_compat.py`

**Behavior:** The patch replaces `Qwen2DecoderLayer.forward` with a patched version that:

1. Calls `self.self_attn(...)` and receives `raw`.
2. Sets `attn_output = raw[0]`.
3. If `attn_output` is not a `torch.Tensor`, attempts to extract a tensor using `_ensure_tensor(attn_output)` (recursively searches dicts, tuples, ModelOutput-like objects for a suitable tensor).
4. Proceeds with `hidden_states = residual + attn_tensor`, then the standard path: layer norm, MLP, residual, outputs.
5. Returns the same structure as the original (hidden states, optional attention weights, optional cache).

**Invocation:** The patch is applied at the start of both Modal entrypoints, *before* importing EasyEdit or loading the model:

```python
import qwen2_rome_compat
qwen2_rome_compat.apply_qwen2_rome_compat_patch()
```

### 3.3 Scientific Validity

- The ROME algorithm (rank-one update, compute_u, compute_v, layer selection) is unchanged.
- The patch only corrects the model’s forward so that the residual add receives a tensor instead of a dict.
- The activations ROME uses are the same as they would be with a correct attention return type.
- This is a compatibility fix, not a methodological change.

---

## 4. Modifications to EasyEdit

To run ROME on Qwen2.5-7B, several edits were made to the local EasyEdit copy. These are limited to compatibility and ROME-only use.

### 4.1 ROME-Only Import Reduction

**Rationale:** EasyEdit supports many algorithms (MELO, DOLA, DeCO, etc.). Their imports pull in dependencies (e.g. MELO’s bundled PEFT) that conflict with our environment. We use only ROME, so we restrict imports.

**Files modified:**

- `easyeditor/__init__.py`: Exports only `ROMEHyperParams`, `apply_rome_to_model`, `execute_rome`, and `BaseEditor`.
- `easyeditor/editors/__init__.py`: Exports only `editor` (BaseEditor).
- `easyeditor/models/__init__.py`: Exports only ROME-related models.
- `easyeditor/util/alg_dict.py`: `ALG_DICT` reduced to `{'ROME': apply_rome_to_model}`.
- `easyeditor/editors/editor.py`: Removed `compute_sent_metric` usage and LORA unwrapping for `generate_edit`.
- `easyeditor/evaluate/__init__.py`: Imports only `evaluate` and `evaluate_utils`; removed `multimodal_evaluate`, `personality_evaluate`, `safety_evaluate`, `concept_evaluate`, `evaluate_uns`.
- `easyeditor/evaluate/evaluate.py`: Removed MELO/LORA import and LORA unwrapping in `compute_edit_quality`.

### 4.2 Model Loading

**File:** `easyeditor/editors/editor.py`

- For Qwen2: added `attn_implementation="eager"` to `AutoModelForCausalLM.from_pretrained(...)` to force eager attention and avoid SDPA/flash paths that can return incompatible structures.

### 4.3 Schema Coercion (CounterFact)

**Files:** `easyeditor/models/rome/rome_main.py`, `easyeditor/evaluate/evaluate.py`, `RomeForQwen/rome_utils.py`

- Added `_coerce_text(x)` to robustly coerce `prompt`, `target_new`, `subject`, and `ground_truth` to plain strings when they come as dicts or lists (schema differences in CounterFact).
- Used in `execute_rome`, `compute_rewrite_or_rephrase_quality`, and `record_to_request`.

### 4.4 Nethook (Critical Fix)

**File:** `easyeditor/util/nethook.py`

- **Bug:** With `register_forward_hook(..., with_kwargs=True)`, PyTorch calls the hook as `hook(module, args, kwargs, result)`. The original code treated the third argument (`kwargs`) as the output and returned it, replacing the real tensor output with a dict.
- **Fix:** Added a wrapper `hook_with_kwargs(m, args, kwargs, result)` that passes `result` as the output and `kwargs` only for input capture. The real module output is preserved.

### 4.5 Representation Extraction

**File:** `easyeditor/models/rome/repr_tools.py`

- Tokenizer output passed to the model is restricted to `input_ids` and `attention_mask`; `token_type_ids` and other optional keys are omitted to avoid unexpected behavior.
- Added `use_cache=False` to the model call to avoid cache-related return-value confusion.

---

## 5. RomeForQwen Utilities

### 5.1 `rome_utils.py`

- **`set_seeds(seed)`:** Sets seeds for `random`, `numpy`, and `torch` for reproducibility.
- **`download_counterfact_dataset(...)`:** Loads CounterFact from Hugging Face with a pinned revision.
- **`load_and_filter_dataset(...)`:** Filters records by case IDs or indices.
- **`extract_metrics(...)`:** Extracts Efficacy, Generalization, Locality and their counts from EasyEdit outputs.
- **`calculate_composite_score(...)`:** Harmonic mean of the three metrics for hyperparameter selection.
- **`record_to_request(...)`:** Converts CounterFact records to EasyEdit request format, with `_coerce_text` for robustness.
- **`load_indices_file(...)`:** Loads `qwen_known_indices.json` (list or dict with `case_ids`).

### 5.2 Dataset Handling

- **Source:** `azhx/counterfact`, split `test`, revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- **Known indices:** `qwen_known_indices.json`; treated as `case_id` values when the dataset has that field, otherwise as positional indices.

---

## 6. Environment and Reproducibility

### 6.1 Pinned Versions (Modal Image)

| Package | Version |
|---------|---------|
| Python | 3.10 |
| transformers | 4.40.2 |
| torch | (from Modal base) |
| datasets | ≥2.14.0 |
| accelerate | 0.21.0 |
| huggingface_hub | ≥0.34.0, &lt;1.0 |

Other dependencies (higher, einops, hydra-core, etc.) are installed as required by EasyEdit.

### 6.2 Local Mounts

- **EasyEdit:** Local `EasyEdit/` directory mounted at `/root/EasyEdit` (modified copy).
- **RomeForQwen:** Local `RomeForQwen/` mounted at `/root/project`.

### 6.3 Cloud Results Volume

- **Volume name:** `rome-results` (Modal Volume, created on first use).
- **Purpose:** Persist `tuning_indices_used.json`, `rome_tuning_results.csv`, and `rome_final_baseline_metrics.csv` so results survive laptop disconnects.
- **Pull locally:** `modal run pull_rome_results.py`.

### 6.4 Seeds

- All experiments use seed **16** via `rome_utils.set_seeds(16)`.

---

## 7. Evaluation Pipeline Summary

### 7.1 Phase 1: Grid Search

- **Script:** `run_rome_grid.py`
- **Actions:** Sample 150 tuning indices, run 36 configs (layers ∈ {15,20,24,27}, v_lr ∈ {5e-1,1e-1,5e-2}, v_num_grad_steps ∈ {20,30,40}), compute composite score, write `rome_tuning_results.csv` and `tuning_indices_used.json` to cloud Volume.

### 7.2 Phase 2: Final Evaluation

- **Script:** `run_rome_final_eval.py`
- **Actions:** Build holdout set, run ROME with best hyperparameters, compute per-edit metrics, write `rome_final_baseline_metrics.csv` to cloud Volume. Reads `tuning_indices_used.json` from Volume.

### 7.3 ROME Configuration

- **Rewrite module:** `model.layers.{}.mlp.down_proj` (Qwen2 SwiGLU MLP).
- **Layer module:** `model.layers.{}`.
- **Fact token:** `subject_last`.
- **Mom2:** Wikipedia, 100k samples, float32.

---

## 8. File Inventory

| File | Purpose |
|------|---------|
| `qwen2_rome_compat.py` | Qwen2–ROME compatibility patch. |
| `rome_utils.py` | Shared utilities (seeds, data, metrics). |
| `run_rome_grid.py` | Phase 1 grid search (Modal). |
| `run_rome_final_eval.py` | Phase 2 holdout evaluation (Modal). |
| `pull_rome_results.py` | Pull result files from cloud Volume to local. |
| `qwen_known_indices.json` | Indices used for tuning/holdout split. |
| `WORKFLOW.md` | User-facing workflow description. |
| `IMPLEMENTATION_NOTES.md` | This document. |

---

## 9. Citation and Attribution

- **ROME:** Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022).
- **EasyEdit:** Wang et al., "EasyEdit: An Easy-to-use Framework to Edit Large Language Models" (https://github.com/zjunlp/EasyEdit).
- **CounterFact:** Meng et al., "Mass-Editing Memory in a Transformer" (2022).
- **Model:** Qwen2.5-7B, Alibaba Cloud.

---

## 10. Reproducibility Checklist

To reproduce the experiments:

1. Ensure the EasyEdit directory contains the modifications described in Section 4.
2. Run `modal run run_rome_grid.py` from `RomeForQwen`. Results are saved to the cloud Volume.
3. Pull results with `modal run pull_rome_results.py`, inspect `rome_tuning_results.csv`, and set `BEST_LAYER`, `BEST_V_LR`, `BEST_V_STEPS` in `run_rome_final_eval.py` accordingly.
4. Run `modal run run_rome_final_eval.py`. Results are saved to the cloud Volume.
5. Pull final metrics with `modal run pull_rome_results.py`.
6. Use seed 16 and the pinned dataset revision for strict reproducibility.
