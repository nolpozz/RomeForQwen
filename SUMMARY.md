## Summary: ROME on Qwen2.5-7B (CounterFact)

This repository reproduces the **ROME** knowledge editing baseline (Meng et al., NeurIPS 2022 / arXiv:2202.05262) on **Qwen/Qwen2.5-7B** using a **two-phase protocol** (tuning + holdout) and **paper-identical CounterFact metrics** (the official `kmeng01/rome` evaluation code path).

### What’s implemented

- **Editing algorithm**: ROME (rank-one update to MLP down-projection at a selected layer).
  - **Qwen mapping**: the rewrite module is `model.layers.{}.mlp.down_proj` (SwiGLU MLP).
  - **Compatibility-only patches**: Qwen2-specific output unwrapping and a hook-output fix (see `IMPLEMENTATION_NOTES.md`). The ROME math (`compute_u`, `compute_v`, update application) is not changed.
- **Dataset**: CounterFact via Hugging Face `azhx/counterfact`, split `test`, **pinned revision**
  `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
- **Evaluation**: ES/PS/NS and composite score **S** computed with `eval_metric="paper_rome"`,
  matching the official ROME release scoring logic.
  - Implementation: `EasyEdit/easyeditor/evaluate/rome_paper_eval.py`
  - Switch: `eval_metric` is plumbed through the pipeline; for Modal scripts the **default is `paper_rome`**.
  - Parity check: `test_paper_rome_eval_parity.py` verifies our scoring matches a literal port of the official ROME loop.

### Two-phase protocol (validity)

ROME reports are sensitive to hyperparameters (layer, learning rate, steps). To avoid tuning on the evaluation set, we use:

- **Phase 1 (tuning)**: grid search on **150** edits drawn (seeded) from a fixed list of `case_id`s.
- **Phase 2 (holdout)**: final baseline on the remaining **468** edits (`known − tuning`).

Tuning indices are saved to a persistent Modal Volume (`rome-results`) so the split is reproducible across machines and reruns.

### Metrics (paper-aligned definitions)

CounterFact metrics in the official ROME release are **probability comparisons between the new object \(o^*\) and the original object \(o_c\)**, implemented via **mean token negative log-likelihood (NLL)** over the answer string.

- **ES (Efficacy Score)**: on the rewrite prompt(s), success if the model prefers the edited object \(o^*\) over \(o_c\).
- **PS (Paraphrase Score / Generalization)**: same predicate on paraphrase prompts.
- **NS (Neighborhood Score / Specificity)**: on neighborhood prompts, success if the model still prefers \(o_c\) over \(o^*\).
- **S (Composite)**: harmonic mean of ES, PS, NS.

In this repo, `paper_rome` scoring mirrors:

- `kmeng01/rome/experiments/py/eval_utils_counterfact.py` (`test_batch_prediction`)
- `kmeng01/rome/experiments/summarize.py` (strict inequality success predicates; harmonic mean)

For additional details and diffs vs legacy `prob_compare`, see `METRICS_PARITY.md`.

### Phase 1: grid search run (what it does)

Script: `run_rome_grid.py`

- **Grid (36 configs)**:
  - `layers ∈ {15, 20, 24, 27}`
  - `v_lr ∈ {0.5, 0.1, 0.05}`
  - `v_num_grad_steps ∈ {20, 30, 40}`
- **Selection rule**: maximize **S** (harmonic mean of ES/PS/NS).
- **Artifacts (in Modal Volume, pulled locally)**:
  - `rome_tuning_results.csv`
  - `rome_grid_metadata.json`
  - `tuning_indices_used.json`

### Phase 2: final baseline metrics (paper_rome)

Script: `run_rome_final_eval.py`

Pulled artifact: `rome_final_eval_metadata.json` (shows `eval_metric: "paper_rome"`).

- **Holdout size**: 468 edits
- **ES**: 1.0 (100.0%)
- **PS**: 0.9433760683760684 (94.34%)
- **NS**: 0.6438034188034187 (64.38%)
- **S**: 0.83026784216142 (0.8303)

Per-edit rows are in `rome_final_baseline_metrics.csv` (with MEAN and N rows appended).

### How to reproduce (Modal)

Prereq: `modal token new` and a working Modal account.

1) **Phase 1: run grid search**

```bash
cd RomeForQwen
modal run --detach run_rome_grid.py
```

2) **Pull artifacts**

```bash
cd RomeForQwen
modal run pull_rome_results.py
```

3) **Set best hyperparameters**

Edit `run_rome_final_eval.py` to match the best row in `rome_tuning_results.csv`:
`BEST_LAYER`, `BEST_V_LR`, `BEST_V_STEPS`.

4) **Phase 2: run final holdout**

```bash
cd RomeForQwen
modal run --detach run_rome_final_eval.py
```

5) **Pull final artifacts**

```bash
cd RomeForQwen
modal run pull_rome_results.py
```

Verify the pull:

- `rome_final_eval_metadata.json` contains `"eval_metric": "paper_rome"`.

### Why this is a valid ROME recreation (and what is comparable)

- **Algorithmic fidelity**: the core ROME update is unchanged; Qwen2 changes are limited to compatibility fixes required to execute the same method on a different architecture.
- **Metric fidelity**: ES/PS/NS/S are computed via **official ROME CounterFact evaluation** (`paper_rome`), not a project-specific reinterpretation.
- **What is comparable**:
  - The *definitions* of ES/PS/NS/S match the released ROME code.
  - The *numbers* are comparable as “ROME evaluated on CounterFact under official metrics” for this model/dataset revision/split.
- **What is not directly comparable to Table 4 in the paper**:
  - Different base model (Qwen2.5-7B vs GPT-2 XL / GPT-J).
  - Different CounterFact source/format (`azhx/counterfact` conversion) and a different (filtered) evaluation subset.
  - Hardware/precision and tokenizer differences can change NLLs and thus discrete outcomes near the decision boundary.

### Additional analysis outputs

- `PS_NS_UNDERPERFORMANCE.md`: lists which edits underperform on PS/NS (by case_id + `target_new`) and by how much, without exposing prompt text.

