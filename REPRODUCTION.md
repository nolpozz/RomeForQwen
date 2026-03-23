# ROME Reproduction: Explanations, Choices, and Shortcomings

This document consolidates all design decisions, rationale, and known limitations of this ROME reproduction for **Qwen/Qwen2.5-7B** on CounterFact. It is intended for paper submission transparency and reviewer reference.

---

## 1. Why Portability Is Not Included in the Dataset

### 1.1 Original ROME Paper vs. Later Work

The **original ROME paper** (Meng et al., NeurIPS 2022, "Locating and Editing Factual Associations in GPT") explicitly describes CounterFact as enabling "quantitative testing of **specificity and generalization**" when learning a counterfactual. The paper’s main metrics are:

- **Efficacy** — success on the direct rewrite prompt
- **Generalization** — robustness across paraphrases
- **Specificity (Locality)** — no unintended changes to other facts

**Portability** (one-hop inference: applying the edited fact in related contexts, e.g. "if Eiffel Tower is in Rome, what city do you visit to see it?") is **not** a core metric in the original ROME paper’s CounterFact evaluation. It appears in later work:

- **EasyEdit** and **KnowledgeEditor** support a fourth "portability" evaluation (e.g. one-hop, subject alias, inverse relation).
- Portability is often evaluated on **zsRE** or other datasets that include one-hop prompts, not on the original CounterFact release.

### 1.2 The `azhx/counterfact` Hugging Face Dataset

We use **`azhx/counterfact`** on Hugging Face, a community conversion of the original CounterFact. Its schema includes:

- `paraphrase_prompts` — used for **Generalization**
- `neighborhood_prompts` — used for **Locality**

It does **not** include one-hop or alternative-subject prompts needed for **Portability**. The original `counterfact.json` from [rome.baulab.info/data](https://rome.baulab.info/data/dsets/) was built for Efficacy, Generalization, and Locality. Portability prompts would require a separate structure (e.g. `alternative_prompts`, `one_hop` relations) that was either:

1. Never part of the original CounterFact release, or  
2. Present in a different format/version that was not carried over to the Hugging Face conversion

### 1.3 What We Do

Our pipeline **captures Portability when it exists** in the request (e.g. if `record["portability"]` is non-empty). With `azhx/counterfact`, it remains empty, so we report **Efficacy, Generalization, and Locality only** — the three metrics the original ROME paper used for CounterFact.

---

## 2. Design Choices and Rationale

### 2.1 Model Choice: Qwen2.5-7B

- **Choice:** Use Qwen/Qwen2.5-7B instead of GPT-J/GPT-2 XL from the original paper.
- **Rationale:** Reproduce ROME on a modern, widely used open model. Qwen2.5-7B has a compatible architecture (decoder-only, MLP structure) for ROME’s rank-one update.
- **Implication:** Results are not directly comparable to the original paper’s tables; we are evaluating ROME on a different model.

### 2.2 Dataset: `azhx/counterfact` (Pinned Revision)

- **Choice:** Hugging Face `azhx/counterfact`, split `test`, revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- **Rationale:** Standard, accessible source; pinned revision avoids dataset drift.
- **Implication:** Schema may differ slightly from the original JSON; we handle coercion in `rome_utils.record_to_request()`.

### 2.3 Two-Phase Pipeline (Tuning vs. Holdout)

- **Choice:** 150 tuning samples for hyperparameter search; holdout = known indices − tuning.
- **Rationale:** Avoid overfitting to the evaluation set; use an objective composite score for hyperparameter selection.
- **Implication:** Holdout size depends on `qwen_known_indices.json` (618 case_ids); ~468 holdout samples.

### 2.4 Composite Score for Hyperparameter Selection

- **Choice:** Harmonic mean of Efficacy, Generalization, and Locality.
- **Rationale:** Single objective that balances all three metrics; avoids ad hoc tuning.
- **Tie-breaker:** Max Efficacy, then Generalization, then Locality (documented in `rome_grid_metadata.json`).

### 2.5 Known Indices: `qwen_known_indices.json`

- **Choice:** Use a pre-filtered set of 618 case_ids (e.g. `min_prob_true >= 0.05`).
- **Rationale:** Focus on cases where the model has non-trivial probability on the correct answer before editing.
- **Shortcoming:** The referenced `counterfact_test_candidates.json` is not in this repo; the exact filtering procedure is not fully documented here.

### 2.6 Evaluation Metric: Token-Level Accuracy (Teacher Forcing)

- **Choice:** EasyEdit’s default: teacher-forcing evaluation, token-level match.
- **Rationale:** Matches common practice in knowledge-editing benchmarks; deterministic and reproducible.
- **Note:** The original ROME paper uses similar token-level evaluation for fill-in-the-blank prompts.

### 2.7 Single Seed (16)

- **Choice:** All runs use seed 16.
- **Rationale:** Reproducibility; simpler pipeline.
- **Shortcoming:** No variance estimates (e.g. mean ± std over seeds); single-run results.

### 2.8 Modal + Cloud Volume

- **Choice:** Run on Modal with a persistent Volume for results.
- **Rationale:** No local GPU required; results persist across disconnects.

---

## 3. Known Shortcomings and Limitations

### 3.1 Reproducibility

| Issue | Severity | Mitigation |
|-------|----------|------------|
| **Model version not pinned** | Medium | `Qwen/Qwen2.5-7B` is loaded without a Hugging Face `revision`. The model can change over time. For strict reproducibility, pin the model commit. |
| **Known indices origin** | Medium | `qwen_known_indices.json` references `counterfact_test_candidates.json`, which is not included. The filtering (e.g. `min_prob_true`) should be documented or the candidate file added. |
| **CUDA nondeterminism** | Low | Seeds are set; cuDNN is configured for determinism. Some CUDA ops can still vary; minor run-to-run differences are possible. |
| **Single seed** | Medium | No multi-seed runs; variance is not reported. |

### 3.2 Metrics and Evaluation

| Issue | Severity | Mitigation |
|-------|----------|------------|
| **Portability not computed** | Low | Dataset lacks one-hop prompts. Document as limitation; pipeline supports Portability when data is available. |
| **Pre-edit baseline not saved** | Low | EasyEdit computes pre-edit metrics; we only persist post-edit. For a "before vs. after" comparison, pre-edit metrics would need to be added. |
| **No baseline (unedited) comparison** | Low | We report edited-model metrics only. A baseline run on the unedited model would strengthen the narrative. |

### 3.3 Methodological

| Issue | Severity | Mitigation |
|-------|----------|------------|
| **Different model** | Known | We use Qwen2.5-7B, not GPT-J/GPT-2 XL. Results are not directly comparable to the original paper. |
| **Possible data contamination** | Unknown | Qwen2.5-7B may have seen CounterFact or similar data during training. This is a general concern for knowledge-editing benchmarks. |
| **Grid size in docs** | Minor | `IMPLEMENTATION_NOTES.md` and `WORKFLOW.md` mention "108 configs"; the actual grid is 36 (4×3×3). |

---

## 4. Modifications to Upstream Code

### 4.1 EasyEdit

- **Nethook fix:** Correct handling of PyTorch’s 4-argument forward hook (`result` vs. `kwargs`).
- **Qwen2 attention patch:** `qwen2_rome_compat.py` unwraps dict-like attention outputs.
- **ROME-only imports:** Reduced to avoid dependency conflicts.
- **Schema coercion:** `_coerce_text()` for CounterFact field variations.
- **Generalization broadcast fix:** Ensure single target is broadcast to multiple rephrase prompts.

### 4.2 RomeForQwen

- **`rome_utils.record_to_request()`:** Maps `paraphrase_prompts` and `neighborhood_prompts` to EasyEdit’s expected format; `portability` left empty for `azhx/counterfact`.
- **`extract_metrics()`:** Extracts Efficacy, Generalization, Locality, and Portability (when present).

---

## 5. Checklist for Paper Submission

When reporting these results, we recommend:

- [ ] **State the model** explicitly: Qwen/Qwen2.5-7B (with version/date if known).
- [ ] **State the dataset** and revision: `azhx/counterfact`, split `test`, revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- [ ] **Clarify metrics:** Efficacy, Generalization, Locality; Portability not evaluated (dataset limitation).
- [ ] **Document the split:** 150 tuning, ~468 holdout; tuning indices from `tuning_indices_used.json`.
- [ ] **Note single-seed limitation** if claiming strict statistical significance.
- [ ] **Cite** ROME (Meng et al.), EasyEdit, and CounterFact appropriately.
- [ ] **Link** to this reproduction repo and `IMPLEMENTATION_NOTES.md` for full details.

---

## 6. References

- Meng et al., "Locating and Editing Factual Associations in GPT," NeurIPS 2022.
- Meng et al., "Mass-Editing Memory in a Transformer," 2022 (CounterFact).
- Wang et al., EasyEdit: https://github.com/zjunlp/EasyEdit
- ROME project: https://rome.baulab.info/
