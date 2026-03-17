"""
Shared utilities for the ROME 2-phase evaluation pipeline (Qwen2.5-7B).
Ensures reproducibility, consistent dataset handling, and objective metric aggregation.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


def _coerce_text(x: Any) -> str:
    """
    Robustly coerce various CounterFact field variants into plain text.
    Handles strings, lists, dicts, and falls back to str(x).
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            if v:
                return _coerce_text(v)
        return ""
    if isinstance(x, dict):
        for key in ("str", "text", "value"):
            if key in x and x[key]:
                return _coerce_text(x[key])
        for v in x.values():
            if v:
                return _coerce_text(v)
        return ""
    return str(x)


def set_seeds(seed: int = 16) -> None:
    """Set explicit seeds for torch, numpy, and random to guarantee reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def download_counterfact_dataset(
    *,
    split: str = "test",
    revision: str = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9",
) -> list[dict]:
    """
    Download CounterFact from HuggingFace (`azhx/counterfact`) with a pinned revision.

    This is the canonical data source used by the Modal scripts to avoid local file drift.
    The pinned `revision` must be an immutable commit hash.
    """
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split=split, revision=revision)
    return [dict(r) for r in ds]


def load_and_filter_dataset(
    indices_to_keep: list[int],
    *,
    dataset_records: list[dict],
    id_field: str = "case_id",
) -> list[dict]:
    """
    Filter CounterFact records to match `indices_to_keep` exactly.

    - If `id_field` exists in records, treats `indices_to_keep` as *IDs* (e.g. case_id) and returns
      records in the order of `indices_to_keep` (dropping IDs not present).
    - Otherwise, treats `indices_to_keep` as *positional indices* into `dataset_records` (dropping out-of-range).

    This prevents dataset order/length drift because selection is done explicitly rather than by
    relying on implicit ordering across different dataset sources.
    """
    if not dataset_records:
        return []

    if isinstance(dataset_records[0], dict) and id_field in dataset_records[0]:
        by_id: dict[int, dict] = {}
        for r in dataset_records:
            try:
                rid = int(r.get(id_field))
            except Exception:
                continue
            by_id[rid] = r
        return [by_id[i] for i in indices_to_keep if i in by_id]

    n = len(dataset_records)
    valid = [i for i in indices_to_keep if 0 <= i < n]
    return [dataset_records[i] for i in valid]


def _get_all_acc_keys(dict_list: list[dict]) -> set:
    acc_keys = set()
    def recurse(d: dict) -> None:
        for k, v in d.items():
            if k.endswith("acc"):
                acc_keys.add(k)
            if isinstance(v, dict):
                recurse(v)
    for d in dict_list:
        recurse(d)
    return acc_keys


def extract_metrics(metrics_list: list[dict]) -> dict[str, Any]:
    """
    Parse EasyEdit per-edit output (list of dicts with 'post' key).
    Returns mean Efficacy, Generalization, Locality and sample size N for each metric,
    since some records lack paraphrase or locality prompts.
    """
    if not metrics_list:
        return {
            "Efficacy": None, "Generalization": None, "Locality": None,
            "n_efficacy": 0, "n_generalization": 0, "n_locality": 0,
        }
    efficacy_vals = []
    generalization_vals = []
    locality_vals = []
    for m in metrics_list:
        post = m.get("post", {})
        if "rewrite_acc" in post:
            v = post["rewrite_acc"]
            efficacy_vals.append(float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v))
        if "rephrase_acc" in post:
            v = post["rephrase_acc"]
            generalization_vals.append(float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v))
        if "locality" in post and post["locality"]:
            loc_flat = []
            for lkey in _get_all_acc_keys([post]):
                if lkey in post["locality"]:
                    v = post["locality"][lkey]
                    if isinstance(v, (list, tuple)):
                        loc_flat.extend(v)
                    else:
                        loc_flat.append(v)
            if loc_flat:
                locality_vals.append(float(np.mean(loc_flat)))
    return {
        "Efficacy": float(np.mean(efficacy_vals)) if efficacy_vals else None,
        "Generalization": float(np.mean(generalization_vals)) if generalization_vals else None,
        "Locality": float(np.mean(locality_vals)) if locality_vals else None,
        "n_efficacy": len(efficacy_vals),
        "n_generalization": len(generalization_vals),
        "n_locality": len(locality_vals),
    }


def calculate_composite_score(
    efficacy: float | None,
    generalization: float | None,
    locality: float | None,
) -> float | None:
    """
    Harmonic mean of Efficacy, Generalization, and Locality.
    Provides a single, mathematically objective score for hyperparameter selection.
    Returns None if any metric is missing or non-positive.
    """
    if efficacy is None or generalization is None or locality is None:
        return None
    if efficacy <= 0 or generalization <= 0 or locality <= 0:
        return None
    return 3.0 / (1.0 / efficacy + 1.0 / generalization + 1.0 / locality)


def record_to_request(record: dict) -> dict:
    """Convert a CounterFact-style record to EasyEdit editor request format (with locality dict).

    This function is robust to slight schema differences between CounterFact
    dumps. It prefers top-level `prompt` / `target_new` / `ground_truth`
    fields, but will fall back to common nested structures (e.g. the
    `requested_rewrite` dict used in some variants) when those are absent.
    """
    # Primary, flat schema
    prompt = record.get("prompt")
    target_new = record.get("target_new")
    ground_truth = record.get("ground_truth")

    # Fallback: nested requested_rewrite-style schema
    rw = record.get("requested_rewrite") or record.get("edit", {}).get("requested_rewrite")
    if rw and isinstance(rw, dict):
        if prompt is None:
            prompt = rw.get("prompt")
        if target_new is None:
            target_new = rw.get("target_new")
        if ground_truth is None:
            ground_truth = rw.get("ground_truth") or rw.get("target_true")

    if prompt is None or target_new is None:
        raise KeyError(
            f"record_to_request could not find prompt/target_new in record: keys={list(record.keys())}"
        )
    prompt = _coerce_text(prompt)
    target_new = _coerce_text(target_new)
    if ground_truth is None:
        # As a last resort, treat empty ground truth as end-of-text.
        ground_truth = "<|endoftext|>"
    else:
        ground_truth = _coerce_text(ground_truth)
    subject = record.get("subject")
    if not subject:
        if "," in str(prompt):
            subject = str(prompt).split(",")[0].strip()
        else:
            words = str(prompt).split()[:2]
            subject = " ".join(words) if words else "unknown"
    else:
        subject = _coerce_text(subject)
    rephrase = record.get("rephrase_prompt")
    loc_prompt = record.get("locality_prompt")
    loc_ground = record.get("locality_ground_truth")
    locality = {}
    if loc_prompt is not None and loc_ground is not None:
        locality["neighborhood"] = {"prompt": loc_prompt, "ground_truth": loc_ground}
    req = {
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "subject": subject,
        "portability": {},
        "locality": locality,
    }
    if rephrase is not None:
        req["rephrase_prompt"] = rephrase
    return req


def load_indices_file(path: str) -> list[int]:
    """
    Load an indices JSON file.

    Supported formats:
    - A JSON list of integers: [1, 2, 3, ...]
    - A JSON object containing 'case_ids': {"case_ids": [1, 2, ...], ...}
    """
    import json

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [int(i) for i in data]

    if isinstance(data, dict):
        if "case_ids" in data and isinstance(data["case_ids"], list):
            return [int(i) for i in data["case_ids"]]
        if "indices" in data and isinstance(data["indices"], list):
            return [int(i) for i in data["indices"]]

    raise ValueError(f"Unsupported indices file format at {path}")
