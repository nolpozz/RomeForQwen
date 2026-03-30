#!/usr/bin/env python3
"""
Parity checks: ``rome_paper_eval.test_batch_prediction_paper_rome`` vs the official
ROME reference loop (kmeng01/rome ``experiments/py/eval_utils_counterfact.py``).

Run (requires torch, transformers; downloads gpt2 on first run):
  PYTHONPATH=EasyEdit python3 RomeForQwen/test_paper_rome_eval_parity.py

Optional: pytest RomeForQwen/test_paper_rome_eval_parity.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import importlib.util
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_ROME_PE_PATH = REPO_ROOT / "EasyEdit" / "easyeditor" / "evaluate" / "rome_paper_eval.py"
_spec = importlib.util.spec_from_file_location("rome_paper_eval", _ROME_PE_PATH)
rome_paper_eval = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rome_paper_eval)

compute_counterfact_quality_paper_rome = rome_paper_eval.compute_counterfact_quality_paper_rome
paper_rome_discrete_neighborhood = rome_paper_eval.paper_rome_discrete_neighborhood
paper_rome_discrete_rewrite_or_paraphrase = rome_paper_eval.paper_rome_discrete_rewrite_or_paraphrase
test_batch_prediction_paper_rome = rome_paper_eval.test_batch_prediction_paper_rome


def _reference_kmeng01_test_batch_prediction(model, tok, prefixes, target_new, target_true, device):
    """Literal port of kmeng01/rome ``test_batch_prediction`` (torch device instead of hardcoded cuda)."""
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    )
    prompt_tok = {k: v.to(device) for k, v in prompt_tok.items()}
    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -F.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


def _test_batch_prediction_matches_reference() -> None:
    import os

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Keep HF cache inside the workspace (sandbox-friendly).
    hf_home = REPO_ROOT / ".hf_parity_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    cache = str(hf_home)

    tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache)
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    prefixes = [
        "The capital of France is",
        "Paris is in the country of",
    ]
    target_new = "Germany"
    target_true = "France"

    ours = test_batch_prediction_paper_rome(
        model, tok, prefixes, target_new, target_true, device
    )
    ref = _reference_kmeng01_test_batch_prediction(
        model, tok, prefixes, target_new, target_true, device
    )
    assert len(ours) == len(ref) == len(prefixes)
    for a, b in zip(ours, ref):
        assert a.keys() == b.keys()
        assert abs(a["target_new"] - b["target_new"]) < 1e-5
        assert abs(a["target_true"] - b["target_true"]) < 1e-5


def _test_discrete_predicates_match_summarize_py() -> None:
    rows = [
        {"target_new": 1.0, "target_true": 2.0},
        {"target_new": 2.0, "target_true": 1.0},
        {"target_new": 1.0, "target_true": 1.0},
    ]
    assert paper_rome_discrete_rewrite_or_paraphrase(rows) == [1.0, 0.0, 0.0]
    assert paper_rome_discrete_neighborhood(rows) == [0.0, 1.0, 0.0]


def _test_compute_counterfact_quality_record_shape() -> None:
    import os

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    hf_home = REPO_ROOT / ".hf_parity_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    cache = str(hf_home)

    tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache)
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    record = {
        "prompt": "The capital of France is",
        "target_new": "Berlin",
        "ground_truth": "Paris",
        "rephrase_prompt": ["France's capital city is"],
        "locality": {
            "neighborhood": {
                "prompt": ["London is in"],
                "ground_truth": ["Paris"],
            }
        },
    }
    out = compute_counterfact_quality_paper_rome(model, tok, record, device)
    assert "rewrite_acc" in out and len(out["rewrite_acc"]) == 1
    assert "rephrase_acc" in out and len(out["rephrase_acc"]) == 1
    assert "locality" in out and "neighborhood_acc" in out["locality"]
    assert len(out["locality"]["neighborhood_acc"]) == 1
    for x in out["rewrite_acc"] + out["rephrase_acc"] + out["locality"]["neighborhood_acc"]:
        assert x in (0.0, 1.0)


def test_batch_prediction_matches_reference():
    _test_batch_prediction_matches_reference()


def test_discrete_predicates_match_summarize_py():
    _test_discrete_predicates_match_summarize_py()


def test_compute_counterfact_quality_record_shape():
    _test_compute_counterfact_quality_record_shape()


if __name__ == "__main__":
    _test_discrete_predicates_match_summarize_py()
    print("ok: discrete predicates")
    _test_batch_prediction_matches_reference()
    print("ok: batch prediction vs kmeng01 reference")
    _test_compute_counterfact_quality_record_shape()
    print("ok: full record shape")
    print("All parity checks passed.")
