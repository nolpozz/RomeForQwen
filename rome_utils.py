"""
Shared utilities for the ROME 2-phase evaluation pipeline (Qwen2.5-7B).
Ensures reproducibility, consistent dataset handling, and objective metric aggregation.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


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


def load_and_filter_dataset(indices_to_keep: list[int], data_path: str) -> list[dict]:
    """
    Load the CounterFact dataset via EasyEdit and filter to the given indices.
    Preserves order of indices_to_keep. Only indices in valid range [0, len(data)) are kept.
    Prevents dataset order/length drift by using a single canonical data path.
    """
    from easyeditor import CounterFactDataset
    dataset = CounterFactDataset(data_path)
    data = dataset.data
    n = len(data)
    valid = [i for i in indices_to_keep if 0 <= i < n]
    return [data[i] for i in valid]


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
    """Convert a CounterFact-style record to EasyEdit editor request format (with locality dict)."""
    prompt = record["prompt"]
    target_new = record["target_new"]
    ground_truth = record["ground_truth"]
    subject = record.get("subject")
    if not subject:
        if "," in prompt:
            subject = prompt.split(",")[0].strip()
        else:
            words = prompt.split()[:2]
            subject = " ".join(words) if words else "unknown"
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
