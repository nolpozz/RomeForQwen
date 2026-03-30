# CounterFact metrics: official ROME vs this repo

This document ties **EasyEdit evaluation** in `EasyEdit/easyeditor/evaluate/` to the **released ROME code** ([github.com/kmeng01/rome](https://github.com/kmeng01/rome), Meng et al., NeurIPS 2022 / arXiv:2202.05262).

## Official references (source of truth)

| Component | File (kmeng01/rome) | Role |
|-----------|---------------------|------|
| Batched NLL scoring | `experiments/py/eval_utils_counterfact.py` — `test_batch_prediction` | For each prefix, scores `target_new` and `target_true` as **mean token NLL** (natural log) on suffixes tokenized as `tok(f" {answer}")`; full strings `f"{prefix} {suffix}"` with default tokenizer settings. |
| CounterFact driver | `experiments/py/eval_utils_counterfact.py` — `compute_rewrite_quality_counterfact` | Chains rewrite, paraphrase, neighborhood, and **empty** `attribute_prompts` through one batched call. |
| Discrete metrics + S | `experiments/summarize.py` | Rewrite/paraphrase: success iff `target_true > target_new` (NLL on true **greater** than on new → model prefers new). Neighborhood: success iff `target_true < target_new`. Composite uses `scipy.stats.hmean` over the three **case-level** means (same formula as our harmonic mean for positive rates). |

Predicates use **strict** inequality; ties count as failure.

## Switch in this repo

| `eval_metric` | Meaning |
|---------------|---------|
| `prob_compare` | Legacy behavior: sum of log-probs, `add_special_tokens=False`, `compute_sequence_log_probability` in `evaluate_utils.py`. Preserves historical CSVs. |
| `paper_rome` | Port of official `test_batch_prediction` + `summarize.py` discrete rules in `EasyEdit/easyeditor/evaluate/rome_paper_eval.py`, invoked from `compute_edit_quality` when `eval_metric == "paper_rome"`. |

Modal / scripts: **default is `paper_rome`**. Set environment variable **`ROME_EVAL_METRIC=prob_compare`** to force legacy scoring for historical comparisons.

## Diff summary (official vs former `prob_compare`)

| Metric | Official ROME | Former `prob_compare` | Match? |
|--------|---------------|------------------------|--------|
| ES / PS score | Mean NLL per answer string; compare means | Sum of log P(token); compare sums | **No** (differs when token lengths differ; also space/tokenizer handling) |
| NS predicate | `NLL(true) < NLL(new)` on neighbors | `sum log P(true) > sum log P(new)` | **No** (same tie-break direction but different scale/aggregation) |
| S | `hmean(ES, PS, NS)` on aggregated means | Same harmonic mean in `rome_utils.calculate_composite_score` | **Yes** (given same per-edit inputs) |
| Neighborhood gold | Same `target_new` / `target_true` from `requested_rewrite` for all prompts (official record) | Same strings in our pipeline (`ground_truth` + `target_new`) | **Yes** (for `azhx/counterfact` broadcast) |
| Aggregation | Per-case mean over paraphrases / neighbors; then mean over cases (`summarize.py`) | Per-edit mean then `extract_metrics` mean over edits | **Yes** (macro over edits) |

## Implementation files (this repo)

- `EasyEdit/easyeditor/evaluate/rome_paper_eval.py` — `test_batch_prediction_paper_rome`, `compute_counterfact_quality_paper_rome`
- `EasyEdit/easyeditor/evaluate/evaluate.py` — early branch in `compute_edit_quality` for `paper_rome`
- `RomeForQwen/run_rome_grid.py`, `run_rome_final_eval.py`, `test_one_edit.py` — read `ROME_EVAL_METRIC`
- `RomeForQwen/test_paper_rome_eval_parity.py` — reference loop vs `test_batch_prediction_paper_rome`

## Known non-reproducibility

- **CUDA / dtype / checkpoint revision**: NLL values depend on device, `float16` vs `float32`, and exact weights; discrete 0/1 labels should match for the same logits within floating tolerance.
- **Transformers tokenizer version**: BPE merges can shift token boundaries; use the same `transformers` pin as your run (e.g. Modal image).
- **`paper_rome` tokenizer defaults**: Matches official code (`tok(...)` without forcing `add_special_tokens=False`). Qwen vs GPT-2 will differ in specials; parity with the **original paper numbers** assumes GPT-2 XL / GPT-J as in the paper, not necessarily identical magnitudes on Qwen.

## Verification

Run:

```bash
python3 RomeForQwen/test_paper_rome_eval_parity.py
```

The script downloads `gpt2` into `RomeForQwen/.hf_parity_cache/` (sandbox-friendly) and asserts our batched NLL table matches an inlined copy of the official `test_batch_prediction` loop. Frozen strings live in `RomeForQwen/fixtures/paper_rome_eval_fixture.json`.

**GPT-2 padding:** Official code calls `tok(..., padding=True)` without setting `pad_token`. `rome_paper_eval` sets `pad_token = eos_token` when `pad_token` is missing so batching matches Hugging Face expectations (same pattern as many GPT-2 + padding setups).
