# Knowledge Update in LLMs: ROME Baseline Report

*For inclusion in COLM paper comparing knowledge-editing methods*

---

## Executive Summary

We reproduce **ROME** (Rank-One Model Editing; Meng et al., NeurIPS 2022) on **Qwen/Qwen2.5-7B** using the CounterFact benchmark. ROME applies a rank-one update to MLP layers to edit factual associations. On a held-out evaluation set of 468 samples, ROME achieves **100% Efficacy**, **72.8% Generalization**, and **76.6% Locality**, with a composite score of **0.815**.

---

## 1. Method: ROME

ROME (Rank-One Model Editing) treats an MLP module as a key-value store: the key encodes the subject, the value encodes the fact. It performs a rank-one modification of the MLP down-projection matrix to insert a new association. The method localizes factual knowledge to specific layers (middle MLPs) and the last token of the subject.

**Key hyperparameters (best config):**
- Edit layer: 15 (of 28)
- Learning rate (v_lr): 0.1
- Gradient steps: 40

---

## 2. Experimental Setup

### 2.1 Model and Data

| Component | Specification |
|-----------|---------------|
| **Model** | Qwen/Qwen2.5-7B (decoder-only, 28 layers, SwiGLU MLP) |
| **Dataset** | CounterFact (`azhx/counterfact`), split `test`, revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9` |
| **Known indices** | 618 case_ids (pre-filtered, min_prob_true ≥ 0.05) |
| **Tuning set** | 150 samples (seeded random sample) |
| **Holdout set** | 468 samples (known − tuning) |
| **Seed** | 16 |

### 2.2 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Efficacy** | Success rate on the direct rewrite prompt (target_new) |
| **Generalization** | Paraphrase accuracy across all `paraphrase_prompts` |
| **Locality** | Neighborhood specificity across all `neighborhood_prompts` (no unintended changes) |
| **Composite** | Harmonic mean of Efficacy, Generalization, Locality |

Evaluation uses teacher-forcing token-level accuracy (EasyEdit default). Portability is not evaluated; the dataset lacks one-hop prompts.

### 2.3 Hyperparameter Selection

A grid search over 36 configurations (layers ∈ {15, 20, 24, 27}, v_lr ∈ {0.5, 0.1, 0.05}, steps ∈ {20, 30, 40}) was run on the tuning set. The configuration with the **highest composite score** was selected for holdout evaluation.

---

## 3. Results

### 3.1 Grid Search (Tuning Set, N=150)

The best configuration was **layer 15, v_lr=0.1, steps=40** with composite score **0.812** on the tuning set.

Top 5 configurations by composite score:

| Layers | v_lr | Steps | Efficacy | Generalization | Locality | Composite |
|--------|------|-------|----------|----------------|----------|-----------|
| 15 | 0.1 | 40 | 0.993 | 0.720 | 0.769 | **0.812** |
| 15 | 0.1 | 30 | 0.993 | 0.717 | 0.770 | 0.811 |
| 15 | 0.1 | 20 | 0.993 | 0.703 | 0.783 | 0.809 |
| 15 | 0.05 | 40 | 1.000 | 0.660 | 0.823 | 0.804 |
| 15 | 0.05 | 30 | 1.000 | 0.653 | 0.825 | 0.802 |

*Note:* Deeper layers (24, 27) showed very low efficacy and composite scores, indicating that layer 15 is the most effective intervention point for Qwen2.5-7B on this task.

### 3.2 Holdout Evaluation (N=468)

| Metric | Score | N |
|--------|-------|---|
| **Efficacy** | **100.0%** | 468 |
| **Generalization** | **72.8%** | 468 |
| **Locality** | **76.6%** | 468 |
| **Composite** | **0.815** | — |

Efficacy is perfect on the holdout set. Generalization (paraphrase robustness) and Locality (neighborhood preservation) are strong and well-balanced, indicating that ROME reliably updates the target fact without unduly affecting unrelated knowledge.

---

## 4. Implementation Notes

### 4.1 Qwen2 Compatibility

ROME was designed for GPT-style models. Running it on Qwen2 required two compatibility fixes:

1. **Nethook fix (EasyEdit):** PyTorch’s forward hook with `with_kwargs=True` passes four arguments `(module, args, kwargs, result)`. The original nethook mistakenly used `kwargs` as the output; the fix correctly uses `result`.
2. **Qwen2 attention patch:** Qwen2’s attention can return dict-like objects. A patch in `qwen2_rome_compat.py` unwraps these before the residual add to avoid `TypeError: unsupported operand type(s) for +: 'Tensor' and 'dict'`.

These changes are **compatibility fixes only**; the ROME algorithm (rank-one update, compute_u, compute_v) is unchanged.

### 4.2 Framework and Environment

- **Framework:** EasyEdit (Wang et al.), with ROME-only imports to avoid dependency conflicts.
- **Hardware:** Modal serverless A100-40GB GPUs.
- **Pinned:** Dataset revision, seed 16, transformers 4.40.2.

### 4.3 Two-Phase Pipeline

To avoid overfitting the evaluation set:

- **Phase 1:** Grid search on 150 tuning samples → select best hyperparameters by composite score.
- **Phase 2:** Final evaluation on 468 holdout samples with the selected config.

Tuning indices are persisted in a Modal Volume so the split is reproducible.

---

## 5. Limitations

| Limitation | Notes |
|------------|-------|
| **Different model** | Original ROME used GPT-J/GPT-2 XL; we use Qwen2.5-7B. Results are not directly comparable. |
| **Portability not evaluated** | CounterFact (azhx) lacks one-hop prompts; Portability is omitted. |
| **Single seed** | No variance estimates; one run per configuration. |
| **Model version** | Hugging Face model ID not pinned; exact reproducibility may require pinning revision. |

---

## 6. Comparison to Original ROME Paper

### 6.1 Original Paper Results (CounterFact, Table 4)

| Model | Efficacy | Generalization (PS) | Specificity / Locality (NS) | Composite (S) |
|-------|----------|---------------------|-----------------------------|---------------|
| **GPT-2 XL** | 100.0% | **96.4%** | 75.4% | 89.2 |
| **GPT-J** | 99.9% | **99.1%** | 78.9% | 91.5 |

### 6.2 Our Reproduction (Qwen2.5-7B)

| Metric | Result |
|--------|--------|
| **Efficacy** | **100.0%** |
| **Generalization** | **72.8%** |
| **Locality** | **76.6%** |
| **Composite** | 0.815 |

### 6.3 Interpretation

| Question | Answer |
|----------|--------|
| Is ROME equally effective on Qwen? | **Yes** — 100% efficacy, on par with the paper. |
| Does it preserve locality equally well? | **Yes** — 76.6% vs. 75–79% in the paper. |
| Does it generalize equally well? | **No** — 72.8% vs. 96–99% in the paper; ~20–25 point gap. |
| Are its limitations the same? | **Partly** — locality is similar; generalization is the main difference. |

**Summary:** ROME on Qwen2.5-7B matches the paper on direct rewrites (Efficacy) and neighborhood preservation (Locality), but Generalization to paraphrases is noticeably weaker than reported for GPT-2 XL and GPT-J. ROME remains strong and well-localized on Qwen; the main drop is in paraphrase robustness.

### 6.4 Possible Causes for the Generalization Gap

- **Model architecture:** Qwen2.5-7B differs from GPT-2/GPT-J; layer 15 may not be as optimal for generalization as the layers used in the paper.
- **Evaluation methodology:** The paper uses probability-based metrics (e.g. P[o*] > P[oc]); EasyEdit uses token-level accuracy. The metrics are not identical.
- **Data split:** The paper used fixed test sets (7,500 for GPT-2 XL, 2,000 for GPT-J); we use 468 holdout samples from a different filtered set.
- **Dataset version:** We use `azhx/counterfact` on Hugging Face; schema differences from the original CounterFact may affect scores.

---

## 7. Comparison Placeholder

*[Insert second method (e.g., fine-tuning, MEND, IKE, etc.) results here for COLM paper comparison.]*

| Method | Efficacy | Generalization | Locality | Composite |
|--------|----------|----------------|----------|-----------|
| **ROME** | 100.0% | 72.8% | 76.6% | 0.815 |
| *Method 2* | — | — | — | — |

---

## 8. Reproducibility

- **Code:** RomeForQwen pipeline (`run_rome_grid.py`, `run_rome_final_eval.py`).
- **Data:** `azhx/counterfact` at revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- **Details:** See `IMPLEMENTATION_NOTES.md` and `REPRODUCTION.md` in this repository.

---

## References

- Meng et al., "Locating and Editing Factual Associations in GPT," NeurIPS 2022.
- Meng et al., "Mass-Editing Memory in a Transformer," 2022 (CounterFact).
- Wang et al., EasyEdit: https://github.com/zjunlp/EasyEdit
