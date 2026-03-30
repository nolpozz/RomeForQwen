# Knowledge Update in LLMs: ROME Baseline Report

*For inclusion in COLM paper comparing knowledge-editing methods*

---

## Executive Summary

We reproduce **ROME** (Rank-One Model Editing; Meng et al., NeurIPS 2022) on **Qwen/Qwen2.5-7B** using the CounterFact benchmark. ROME applies a rank-one update to MLP layers to edit factual associations. On a held-out evaluation set of 468 samples, ROME achieves **100% Efficacy (ES)**, **94.34% Generalization (PS)**, and **64.38% Specificity (NS)**, with a composite score **S = 0.8303** (latest pulled Phase 2 run; `rome_final_eval_metadata.json`).

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

### 2.2 Evaluation Metrics (Aligned with ROME Paper)

| Metric | Paper Variable | Definition | Implementation |
|--------|----------------|------------|----------------|
| **ES** (Efficacy Score) | Efficacy | P(o*) > P(oc) post-edit on rewrite prompt | Probability comparison (mean token NLL) |
| **PS** (Paraphrase Score) | Generalization | P(o*) > P(oc) on paraphrase prompts | Probability comparison |
| **NS** (Neighborhood Score) | Specificity | P(oc) > P(o*) on neighborhood prompts | Probability comparison |
| **GE** (Fluency) | Fluency | N-gram entropy of generations | Weighted bi-/tri-gram entropy |
| **S** (Composite) | Composite | Harmonic mean of ES, PS, NS | Same |

Metrics use `eval_metric="paper_rome"`: ES/PS count success when the model assigns higher probability to the counterfactual object o* than the original oc; NS counts success when P(oc) > P(o*) on neighborhood prompts (model favors the original object over the edit target). Per edit, NS is the mean over neighborhood prompts; Phase 2 uses `max_nb_prompts=10` (first 10 entries; CounterFact provides 10). Portability is not evaluated; the dataset lacks one-hop prompts.

### 2.3 Hyperparameter Selection

A grid search over 36 configurations (layers ∈ {15, 20, 24, 27}, v_lr ∈ {0.5, 0.1, 0.05}, steps ∈ {20, 30, 40}) was run on the tuning set. The configuration with the **highest composite score S** was selected for holdout evaluation.

---

## 3. Results

### 3.1 Grid Search (Tuning Set, N=150)

The best configuration was **layer 15, v_lr=0.1, steps=40** with composite score **S = 0.812** on the tuning set.

Top 5 configurations by composite score:

| Layers | v_lr | Steps | ES (Efficacy) | PS (Generalization) | NS (Specificity) | S (Composite) |
|--------|------|-------|---------------|---------------------|------------------|---------------|
| 15 | 0.1 | 40 | 0.993 | 0.720 | 0.769 | **0.812** |
| 15 | 0.1 | 30 | 0.993 | 0.717 | 0.770 | 0.811 |
| 15 | 0.1 | 20 | 0.993 | 0.703 | 0.783 | 0.809 |
| 15 | 0.05 | 40 | 1.000 | 0.660 | 0.823 | 0.804 |
| 15 | 0.05 | 30 | 1.000 | 0.653 | 0.825 | 0.802 |

*Note:* Deeper layers (24, 27) showed very low efficacy and composite scores, indicating that layer 15 is the most effective intervention point for Qwen2.5-7B on this task.

### 3.2 Holdout Evaluation (N=468)

Numbers below match the **pulled** Phase 2 artifact `rome_final_eval_metadata.json` (and aggregate row `MEAN` in `rome_final_baseline_metrics.csv`).

| Metric | Paper Variable | Score | N |
|--------|----------------|-------|---|
| **ES** | Efficacy | **100.0%** | 468 |
| **PS** | Generalization | **94.34%** | 468 |
| **NS** | Specificity | **64.38%** | 468 |
| **GE** | Fluency | 6.214 | 468 |
| **S** | Composite | **0.8303** | — |

**NS (Specificity).** Mean NS is **0.6438** averaged over 468 edits. Each edit’s NS is itself the mean of up to **10** binary neighborhood outcomes (one per `neighborhood_prompts` entry; `max_nb_prompts=10` in the pipeline). Per-edit values span the full 0–1 range in the CSV (e.g. many edits sit around 0.5–0.9, some near 0), so the headline NS is **well below** efficacy and reflects inconsistent preservation on related subjects rather than a single failure mode.

Efficacy is perfect on the holdout set. Generalization (paraphrase robustness) remains strong at **94.34%**. Specificity (**64.38%**) is lower than in the original paper, consistent with spillover of the edit to semantically related queries under this probability-based NS definition.

---

## 4. Implementation Notes

### 4.1 Qwen2 Compatibility

ROME was designed for GPT-style models. Running it on Qwen2 required two compatibility fixes:

1. **Nethook fix (EasyEdit):** PyTorch's forward hook with `with_kwargs=True` passes four arguments `(module, args, kwargs, result)`. The original nethook mistakenly used `kwargs` as the output; the fix correctly uses `result`.
2. **Qwen2 attention patch:** Qwen2's attention can return dict-like objects. A patch in `qwen2_rome_compat.py` unwraps these before the residual add to avoid `TypeError: unsupported operand type(s) for +: 'Tensor' and 'dict'`.

These changes are **compatibility fixes only**; the ROME algorithm (rank-one update, compute_u, compute_v) is unchanged.

### 4.2 Framework and Environment

- **Framework:** EasyEdit (Wang et al.), with ROME-only imports to avoid dependency conflicts.
- **Hardware:** Modal serverless A100-40GB GPUs.
- **Pinned:** Dataset revision, seed 16, transformers 4.40.2.

### 4.3 Two-Phase Pipeline

To avoid overfitting the evaluation set:

- **Phase 1:** Grid search on 150 tuning samples → select best hyperparameters by composite score S.
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

| Model | ES (Efficacy) | PS (Generalization) | NS (Specificity) | S (Composite) |
|-------|---------------|---------------------|------------------|---------------|
| **GPT-2 XL** | 100.0% | **96.4%** | 75.4% | 89.2 |
| **GPT-J** | 99.9% | **99.1%** | 78.9% | 91.5 |

### 6.2 Our Reproduction (Qwen2.5-7B)

| Metric | Paper Variable | Result |
|--------|----------------|--------|
| **ES** | Efficacy | **100.0%** |
| **PS** | Generalization | **94.34%** |
| **NS** | Specificity | **64.38%** |
| **S** | Composite | 0.8303 |

### 6.3 Interpretation

| Question | Answer |
|----------|--------|
| Is ROME equally effective on Qwen? | **Yes** — 100% ES, on par with the paper. |
| Does it preserve specificity equally well? | **No** — 64.38% NS vs. 75–79% in the paper; ~11–15 point gap. |
| Does it generalize equally well? | **Nearly** — 94.34% PS vs. 96–99% in the paper; small gap. |
| Are its limitations the same? | **Partly** — efficacy and generalization are strong; specificity is the main difference. |

**Summary:** ROME on Qwen2.5-7B matches the paper on direct rewrites (ES) and approaches it on paraphrase generalization (PS). Specificity (NS ≈ 64.38%) is noticeably lower, indicating that the rank-one update on Qwen2.5-7B may affect a broader set of queries than on GPT-2 XL / GPT-J.

### 6.4 Possible Causes for the Specificity Gap

- **Model architecture:** Qwen2.5-7B differs from GPT-2/GPT-J; the key space at layer 15 may have more overlap between edited and neighborhood subjects.
- **Evaluation methodology:** We use probability-based metrics (P(o*) > P(oc), P(oc) > P(o*)) aligned with the paper; remaining differences may stem from tokenization or sequence-probability formulation.
- **Data split:** The paper used fixed test sets (7,500 for GPT-2 XL, 2,000 for GPT-J); we use 468 holdout samples from a different filtered set.
- **Dataset version:** We use `azhx/counterfact` on Hugging Face; schema differences from the original CounterFact may affect scores.

---

## 7. Comparison Placeholder

*[Insert second method (e.g., fine-tuning, MEND, IKE, etc.) results here for COLM paper comparison.]*

| Method | ES (Efficacy) | PS (Generalization) | NS (Specificity) | S (Composite) |
|--------|---------------|---------------------|------------------|---------------|
| **ROME** | 100.0% | 94.34% | 64.38% | 0.8303 |
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
